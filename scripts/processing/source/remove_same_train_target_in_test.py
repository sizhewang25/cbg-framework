"""Remove test-set targets that already appear in the train set.

Given pre-split train/test files (CSV or parquet, canonical schema — see
``scripts/benchmark/v2/sources/README.md``), drop every test row whose
``target_id`` is also a train target. This is the leakage condition
``generic_presplit`` hard-errors on, so run this once before
``materialize-inputs``. Optionally (``--match-coords``) also drop test
targets sitting at the exact ``(target_lat, target_lon)`` of a train
target even when the ids differ.

Alongside the cleaned test set the script writes the diff:

- ``<output>.removed.<ext>`` — the removed test rows, verbatim.
- ``<output>.diff.json``     — counts + the removed target ids per reason.

CLI::

    python -m scripts.processing.source.remove_same_train_target_in_test \
        --train datasets/final/train.parquet \
        --test datasets/final/test.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_COL = "target_id"
_LAT_COL = "target_lat"
_LON_COL = "target_lon"


def _read_frame(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported extension {ext!r} for {path} (expected .csv or .parquet)")


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _resolve_col(df: pd.DataFrame, name: str, path: Path) -> str:
    """Find ``name`` in ``df`` case-insensitively, or exit."""
    lookup = {c.lower(): c for c in df.columns}
    col = lookup.get(name.lower())
    if col is None:
        raise SystemExit(f"Missing required column {name!r} in {path}")
    return col


def _target_ids(df: pd.DataFrame, col: str) -> pd.Series:
    """Target ids as stripped strings so CSV ints and parquet strings compare equal."""
    return df[col].astype(str).str.strip()


def find_overlapping_target_ids(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_col: str,
    test_col: str,
) -> set[str]:
    return set(_target_ids(train_df, train_col)) & set(_target_ids(test_df, test_col))


def find_coord_matched_target_ids(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_paths: tuple[str, str],
    test_paths: tuple[str, str],
    test_col: str,
) -> set[str]:
    """Test target ids whose exact (lat, lon) also belongs to some train target."""
    train_lat, train_lon = train_paths
    test_lat, test_lon = test_paths
    train_coords = set(zip(train_df[train_lat], train_df[train_lon]))
    test_coords = list(zip(test_df[test_lat], test_df[test_lon]))
    mask = pd.Series([c in train_coords for c in test_coords], index=test_df.index)
    return set(_target_ids(test_df, test_col)[mask])


def _default_output(test_path: Path) -> Path:
    return test_path.with_name(f"{test_path.stem}.disjoint{test_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True,
                        help="Train-set file (.csv or .parquet).")
    parser.add_argument("--test", type=Path, required=True,
                        help="Test-set file (.csv or .parquet) to clean.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Cleaned test set. Defaults to <test>.disjoint.<ext> "
                             "next to --test.")
    parser.add_argument("--target-col", type=str, default=_DEFAULT_TARGET_COL,
                        help="Target identifier column (case-insensitive). "
                             f"Default {_DEFAULT_TARGET_COL}.")
    parser.add_argument("--match-coords", action="store_true",
                        help="Additionally remove test targets whose exact "
                             f"({_LAT_COL}, {_LON_COL}) matches a train target, "
                             "even under a different id.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in (args.train, args.test):
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    train_df = _read_frame(args.train)
    test_df = _read_frame(args.test)

    train_col = _resolve_col(train_df, args.target_col, args.train)
    test_col = _resolve_col(test_df, args.target_col, args.test)

    removed_by_id = find_overlapping_target_ids(
        train_df, test_df, train_col=train_col, test_col=test_col
    )
    removed_by_coords: set[str] = set()
    if args.match_coords:
        train_lat = _resolve_col(train_df, _LAT_COL, args.train)
        train_lon = _resolve_col(train_df, _LON_COL, args.train)
        test_lat = _resolve_col(test_df, _LAT_COL, args.test)
        test_lon = _resolve_col(test_df, _LON_COL, args.test)
        removed_by_coords = find_coord_matched_target_ids(
            train_df, test_df,
            train_paths=(train_lat, train_lon),
            test_paths=(test_lat, test_lon),
            test_col=test_col,
        ) - removed_by_id

    removed_ids = removed_by_id | removed_by_coords
    remove_mask = _target_ids(test_df, test_col).isin(removed_ids)
    kept_df = test_df[~remove_mask].copy()
    removed_df = test_df[remove_mask].copy()

    if kept_df.empty:
        raise SystemExit(
            f"Every test target overlaps the train set ({len(removed_ids)} targets) — "
            "refusing to write an empty test set."
        )

    out_path = args.output or _default_output(args.test)
    removed_path = out_path.with_name(f"{out_path.stem}.removed{out_path.suffix}")
    diff_path = out_path.with_name(f"{out_path.stem}.diff.json")

    _write_frame(kept_df, out_path)
    _write_frame(removed_df, removed_path)

    test_targets_before = _target_ids(test_df, test_col).nunique()
    diff = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_path": str(args.train),
        "test_path": str(args.test),
        "match_coords": args.match_coords,
        "n_train_targets": _target_ids(train_df, train_col).nunique(),
        "n_test_targets_before": test_targets_before,
        "n_test_targets_after": _target_ids(kept_df, test_col).nunique(),
        "n_test_rows_before": len(test_df),
        "n_test_rows_after": len(kept_df),
        "n_removed_rows": len(removed_df),
        "removed_target_ids_by_id": sorted(removed_by_id),
        "removed_target_ids_by_coords": sorted(removed_by_coords),
    }
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(json.dumps(diff, indent=2) + "\n")

    logger.info("train targets : %d", diff["n_train_targets"])
    logger.info("test targets  : %d -> %d (removed %d: %d by id, %d by coords)",
                test_targets_before, diff["n_test_targets_after"],
                len(removed_ids), len(removed_by_id), len(removed_by_coords))
    logger.info("test rows     : %d -> %d (removed %d)",
                len(test_df), len(kept_df), len(removed_df))
    logger.info("  wrote %s", out_path)
    logger.info("  wrote %s", removed_path)
    logger.info("  wrote %s", diff_path)


if __name__ == "__main__":
    main()
