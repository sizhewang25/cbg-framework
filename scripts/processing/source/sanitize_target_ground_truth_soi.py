"""Sanitize  targets using iterative speed-of-Internet (SOI) pruning.

For each observed (VP, target) pair, compute:

  implied_km = (min RTT / 2) * 200

where 200 km/ms is 2/3c. A pair is an SOI violation when the geodesic
VP-target distance exceeds implied_km.

Target sanitization follows a two-stage policy based on per-target violation
fraction:
     1) remove full targets with violation fraction above threshold (default 5%),
     2) for remaining targets that still have violations, remove only violating
         (VP, target) pairs,
     3) recompute once for final audit.

For diagnostics only, we still compute `not_violate_till` by sorting target
pairs from smallest RTT to largest RTT and counting consecutive non-violating
pairs until the first violation.

Inputs are -style CSVs with VP/target coordinates and RTT columns. Column
names are normalized case-insensitively, so both upper-case and lower-case
schemas are supported.

Outputs:
  1) sanitized CSV (rows for kept targets)
  2) pair-level audit CSV (distance/implied/violation)
  3) removed-targets CSV
  4) summary JSON

CLI:
  python -m scripts.processing.source.sanitize_target_ground_truth_soi \
      --input datasets/test05-mainland.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path("datasets/test05-mainland.csv")
_SOI_SPEED_KM_MS = 200.0


def _default_output(input_path: Path) -> Path:
    return input_path.with_suffix("").with_suffix(".soi-target-sanitized.csv")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _resolve_rtt_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        col = explicit.strip().lower()
        if col not in df.columns:
            raise ValueError(f"RTT column not found: {explicit}")
        return col
    for candidate in ("rtt_ms_min", "rtt_ms"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Missing RTT column; expected one of: rtt_ms_min, rtt_ms")


def _pair_min_rtt_and_distance(df: pd.DataFrame, rtt_col: str, eps_km: float) -> pd.DataFrame:
    required = ["vp_id", "vp_lat", "vp_lon", "target_id", "target_lat", "target_lon", rtt_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    grp = (
        df.groupby(["vp_id", "target_id"], dropna=False)
        .agg(
            vp_lat=("vp_lat", "median"),
            vp_lon=("vp_lon", "median"),
            target_lat=("target_lat", "median"),
            target_lon=("target_lon", "median"),
            rtt_min_ms=(rtt_col, "min"),
            n_obs=(rtt_col, "size"),
        )
        .reset_index()
    )

    valid = grp[["vp_lat", "vp_lon", "target_lat", "target_lon", "rtt_min_ms"]].notna().all(axis=1)
    grp = grp[valid].copy()
    grp = grp[grp["rtt_min_ms"] > 0].copy()

    grp["distance_km"] = grp.apply(
        lambda r: haversine_distance(
            float(r["vp_lat"]),
            float(r["vp_lon"]),
            float(r["target_lat"]),
            float(r["target_lon"]),
        ),
        axis=1,
    )
    grp["implied_km"] = grp["rtt_min_ms"] * 0.5 * _SOI_SPEED_KM_MS
    grp["violation"] = grp["implied_km"] + eps_km < grp["distance_km"]
    grp["violation_ratio"] = np.where(
        grp["implied_km"] > 0,
        grp["distance_km"] / grp["implied_km"],
        np.nan,
    )
    return grp


def _target_violation_table(pair_stats: pd.DataFrame) -> pd.DataFrame:
    if pair_stats.empty:
        return pd.DataFrame(
            columns=[
                "target_id",
                "n_pairs",
                "n_violations",
                "violation_fraction",
                "all_pairs_violate",
                "median_target_lat",
                "median_target_lon",
                "max_violation_ratio",
                "not_violate_till",
            ]
        )

    # Count consecutive non-violating pairs from smallest RTT upwards.
    ordered = pair_stats.sort_values(["target_id", "rtt_min_ms", "vp_id"], kind="mergesort").copy()
    ordered["row_idx"] = ordered.groupby("target_id", dropna=False).cumcount()
    first_violation = (
        ordered[ordered["violation"]]
        .groupby("target_id", dropna=False)["row_idx"]
        .min()
        .rename("first_violation_idx")
        .reset_index()
    )

    agg = (
        pair_stats.groupby("target_id", dropna=False)
        .agg(
            n_pairs=("violation", "size"),
            n_violations=("violation", "sum"),
            median_target_lat=("target_lat", "median"),
            median_target_lon=("target_lon", "median"),
            max_violation_ratio=("violation_ratio", "max"),
        )
        .reset_index()
    )
    agg = agg.merge(first_violation, on="target_id", how="left")
    agg["not_violate_till"] = np.where(
        agg["first_violation_idx"].notna(),
        agg["first_violation_idx"],
        agg["n_pairs"],
    ).astype(int)
    agg = agg.drop(columns=["first_violation_idx"])
    agg["violation_fraction"] = agg["n_violations"] / agg["n_pairs"].clip(lower=1)
    agg["all_pairs_violate"] = agg["n_pairs"].gt(0) & agg["n_violations"].eq(agg["n_pairs"])
    return agg.sort_values(
        ["n_violations", "violation_fraction", "max_violation_ratio", "not_violate_till", "target_id"],
        ascending=[False, False, False, True, True],
    )


def sanitize_targets(
    df_raw: pd.DataFrame,
    rtt_col: str,
    eps_km: float,
    target_violation_fraction_for_removal: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    work = df_raw.copy()
    work["target_id"] = work["target_id"].astype(str)

    initial_pair_stats = _pair_min_rtt_and_distance(work, rtt_col=rtt_col, eps_km=eps_km)
    initial_pair_count = len(initial_pair_stats)
    initial_violation_count = int(initial_pair_stats["violation"].sum()) if initial_pair_count else 0

    removed_steps: list[dict[str, float | int | str]] = []
    pair_stats = _pair_min_rtt_and_distance(work, rtt_col=rtt_col, eps_km=eps_km)
    n_viol_before = int(pair_stats["violation"].sum()) if len(pair_stats) else 0

    target_stats = _target_violation_table(pair_stats)
    high_violation_targets = target_stats[
        target_stats["violation_fraction"] > target_violation_fraction_for_removal
    ].copy()

    for _, top in high_violation_targets.iterrows():
        target_id = str(top["target_id"])
        removed_steps.append(
            {
                "iteration": 1,
                "action": "remove_target",
                "target_id": target_id,
                "n_pairs": int(top["n_pairs"]),
                "n_violations": int(top["n_violations"]),
                "violation_fraction": float(top["violation_fraction"]),
                "not_violate_till": int(top["not_violate_till"]),
                "removed_violation_pairs": 0,
                "max_violation_ratio": float(top["max_violation_ratio"])
                if pd.notna(top["max_violation_ratio"]) else float("nan"),
                "pair_violations_before_removal": n_viol_before,
            }
        )

    if len(high_violation_targets):
        remove_targets_set = set(high_violation_targets["target_id"].astype(str))
        work = work[~work["target_id"].astype(str).isin(remove_targets_set)].copy()

    post_target_pair_stats = _pair_min_rtt_and_distance(work, rtt_col=rtt_col, eps_km=eps_km)
    remaining_violating_pairs = post_target_pair_stats[post_target_pair_stats["violation"]][["vp_id", "target_id"]].drop_duplicates()

    if len(remaining_violating_pairs):
        remaining_targets_stats = _target_violation_table(post_target_pair_stats)
        remaining_targets_stats = remaining_targets_stats[
            (remaining_targets_stats["n_violations"] > 0)
            & (remaining_targets_stats["violation_fraction"] <= target_violation_fraction_for_removal)
        ]
        n_viol_before_pair_drop = int(post_target_pair_stats["violation"].sum())
        for _, top in remaining_targets_stats.iterrows():
            target_id = str(top["target_id"])
            removed_pairs_for_target = int(
                (remaining_violating_pairs["target_id"].astype(str) == target_id).sum()
            )
            removed_steps.append(
                {
                    "iteration": 2,
                    "action": "remove_violating_pairs",
                    "target_id": target_id,
                    "n_pairs": int(top["n_pairs"]),
                    "n_violations": int(top["n_violations"]),
                    "violation_fraction": float(top["violation_fraction"]),
                    "not_violate_till": int(top["not_violate_till"]),
                    "removed_violation_pairs": removed_pairs_for_target,
                    "max_violation_ratio": float(top["max_violation_ratio"])
                    if pd.notna(top["max_violation_ratio"]) else float("nan"),
                    "pair_violations_before_removal": n_viol_before_pair_drop,
                }
            )

        drop_keys = set(
            zip(
                remaining_violating_pairs["vp_id"].astype(str),
                remaining_violating_pairs["target_id"].astype(str),
            )
        )
        mask_drop = work[["vp_id", "target_id"]].astype(str).apply(tuple, axis=1).isin(drop_keys)
        work = work[~mask_drop].copy()

    final_pair_stats = _pair_min_rtt_and_distance(work, rtt_col=rtt_col, eps_km=eps_km)
    removed_targets = pd.DataFrame(removed_steps)
    iterations = int(removed_targets["iteration"].max()) if not removed_targets.empty else 0
    metadata: dict[str, float | int] = {
        "initial_pair_count": initial_pair_count,
        "initial_pair_violations": initial_violation_count,
        "final_pair_count": len(final_pair_stats),
        "final_pair_violations": int(final_pair_stats["violation"].sum()) if len(final_pair_stats) else 0,
        "iterations": iterations,
        "target_violation_fraction_for_removal": float(target_violation_fraction_for_removal),
    }
    return work, final_pair_stats, removed_targets, metadata


def _write_summary(
    path: Path,
    *,
    input_path: Path,
    rtt_col: str,
    eps_km: float,
    n_rows_in: int,
    n_rows_out: int,
    n_targets_in: int,
    n_targets_out: int,
    n_pairs_initial: int,
    n_pair_violations_initial: int,
    n_pairs_final: int,
    n_pair_violations_final: int,
    n_targets_removed: int,
    n_pair_only_actions: int,
    n_pairs_removed_from_pair_actions: int,
    iterations: int,
) -> None:
    payload = {
        "input": str(input_path),
        "rtt_column": rtt_col,
        "speed_of_internet_km_per_ms": _SOI_SPEED_KM_MS,
        "epsilon_km": eps_km,
        "rows_input": n_rows_in,
        "rows_output": n_rows_out,
        "rows_removed": n_rows_in - n_rows_out,
        "targets_input": n_targets_in,
        "targets_output": n_targets_out,
        "targets_removed": n_targets_removed,
        "pair_only_actions": n_pair_only_actions,
        "pairs_removed_from_pair_actions": n_pairs_removed_from_pair_actions,
        "pair_count_initial": n_pairs_initial,
        "pair_violations_initial": n_pair_violations_initial,
        "pair_violation_fraction_initial": (
            (n_pair_violations_initial / n_pairs_initial) if n_pairs_initial else 0.0
        ),
        "pair_count_final": n_pairs_final,
        "pair_violations_final": n_pair_violations_final,
        "pair_violation_fraction_final": (
            (n_pair_violations_final / n_pairs_final) if n_pairs_final else 0.0
        ),
        "iterations": iterations,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Input  CSV")
    parser.add_argument("--output", type=Path, default=None, help="Sanitized output CSV")
    parser.add_argument(
        "--rtt-col",
        type=str,
        default=None,
        help="Optional RTT column name. Defaults to rtt_ms_min if present, else rtt_ms.",
    )
    parser.add_argument(
        "--epsilon-km",
        type=float,
        default=1e-9,
        help="Tolerance in km for violation check: implied_km + epsilon < distance_km",
    )
    parser.add_argument(
        "--target-violation-fraction-for-removal",
        type=float,
        default=0.05,
        help="Remove full target when violation_fraction is strictly above this threshold.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    raw_in = pd.read_csv(args.input)
    raw = _normalize_columns(raw_in)

    for col in ("vp_lat", "vp_lon", "target_lat", "target_lon"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if "vp_id" in raw.columns:
        raw["vp_id"] = raw["vp_id"].astype(str)
    if "target_id" in raw.columns:
        raw["target_id"] = raw["target_id"].astype(str)

    rtt_col = _resolve_rtt_column(raw, args.rtt_col)
    raw[rtt_col] = pd.to_numeric(raw[rtt_col], errors="coerce")

    kept, pair_stats, removed_targets, sanitize_meta = sanitize_targets(
        raw,
        rtt_col=rtt_col,
        eps_km=args.epsilon_km,
        target_violation_fraction_for_removal=args.target_violation_fraction_for_removal,
    )

    out_csv = args.output or _default_output(args.input)
    stem = out_csv.with_suffix("")
    pairs_csv = Path(str(stem) + ".pairs.csv")
    removed_csv = Path(str(stem) + ".removed_targets.csv")
    summary_json = Path(str(stem) + ".summary.json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_csv, index=False)
    pair_stats.to_csv(pairs_csv, index=False)
    removed_targets.to_csv(removed_csv, index=False)

    n_rows_in = len(raw)
    n_rows_out = len(kept)
    n_targets_in = int(raw["target_id"].nunique()) if "target_id" in raw.columns else 0
    n_targets_out = int(kept["target_id"].nunique()) if "target_id" in kept.columns else 0
    n_pairs_initial = int(sanitize_meta["initial_pair_count"])
    n_pair_violations_initial = int(sanitize_meta["initial_pair_violations"])
    n_pairs_final = int(sanitize_meta["final_pair_count"])
    n_pair_violations_final = int(sanitize_meta["final_pair_violations"])
    iterations = int(sanitize_meta["iterations"])
    if not removed_targets.empty and "action" in removed_targets.columns:
        n_targets_removed = int((removed_targets["action"] == "remove_target").sum())
        n_pair_only_actions = int((removed_targets["action"] == "remove_violating_pairs").sum())
    else:
        n_targets_removed = 0
        n_pair_only_actions = 0
    n_pairs_removed_from_pair_actions = (
        int(removed_targets.get("removed_violation_pairs", pd.Series(dtype=int)).fillna(0).sum())
        if not removed_targets.empty else 0
    )

    _write_summary(
        summary_json,
        input_path=args.input,
        rtt_col=rtt_col,
        eps_km=args.epsilon_km,
        n_rows_in=n_rows_in,
        n_rows_out=n_rows_out,
        n_targets_in=n_targets_in,
        n_targets_out=n_targets_out,
        n_pairs_initial=n_pairs_initial,
        n_pair_violations_initial=n_pair_violations_initial,
        n_pairs_final=n_pairs_final,
        n_pair_violations_final=n_pair_violations_final,
        n_targets_removed=n_targets_removed,
        n_pair_only_actions=n_pair_only_actions,
        n_pairs_removed_from_pair_actions=n_pairs_removed_from_pair_actions,
        iterations=iterations,
    )

    logger.info("SOI target GT sanitization")
    logger.info("  input rows/targets : %d / %d", n_rows_in, n_targets_in)
    logger.info("  kept rows/targets  : %d / %d", n_rows_out, n_targets_out)
    logger.info("  removed targets    : %d", n_targets_removed)
    logger.info("  iterations         : %d", iterations)
    logger.info("  pair violations (initial): %d / %d (%.2f%%)",
                n_pair_violations_initial, n_pairs_initial,
                (100.0 * n_pair_violations_initial / n_pairs_initial) if n_pairs_initial else 0.0)
    logger.info("  pair violations (final)  : %d / %d (%.2f%%)",
                n_pair_violations_final, n_pairs_final,
                (100.0 * n_pair_violations_final / n_pairs_final) if n_pairs_final else 0.0)
    logger.info("  rtt column         : %s", rtt_col)
    logger.info("  wrote %s", out_csv)
    logger.info("  wrote %s", pairs_csv)
    logger.info("  wrote %s", removed_csv)
    logger.info("  wrote %s", summary_json)


if __name__ == "__main__":
    main()