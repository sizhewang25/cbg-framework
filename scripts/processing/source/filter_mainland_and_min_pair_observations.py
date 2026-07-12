"""Filter raw  rows to mainland US and sufficiently observed VP-target pairs.

This script wires the mainland-US filter from
``scripts.benchmark.v2.sources.atnt_ant.filter_non_mainland_us_targets_and_vps``
and then keeps only rows whose target has at least N observing VPs
(default: 3).

CLI::

    python -m scripts.processing.source.filter_mainland_and_min_pair_observations \
        --input datasets/raw/20260417-20260504.csv \
        --min-observations 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.benchmark.v2.sources.atnt_ant import (
    filter_non_mainland_us_targets_and_vps,
)

logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path("datasets/raw/20260417-20260504.csv")
_DEFAULT_OUT_DIR = Path("datasets/final")
_DEFAULT_MIN_OBS = 3

_DEFAULT_VP_COL = "VP_ID"
_DEFAULT_TARGET_COL = "TARGET_ID"


def filter_pairs_by_min_observations(
    df: pd.DataFrame,
    *,
    vp_col: str,
    target_col: str,
    min_observations: int,
) -> pd.DataFrame:
    """Keep rows whose ``(vp_col, target_col)`` pair has >= ``min_observations`` rows."""
    if min_observations < 1:
        raise ValueError("min_observations must be >= 1")
    for col in (vp_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pair_counts = df.groupby([vp_col, target_col], dropna=False)[vp_col].transform("size")
    return df[pair_counts >= min_observations].copy()


def filter_pairs_by_min_target_vp_observations(
    df: pd.DataFrame,
    *,
    vp_col: str,
    target_col: str,
    min_observations: int,
) -> pd.DataFrame:
    """Keep rows where each target is observed by >= ``min_observations`` distinct VPs."""
    if min_observations < 1:
        raise ValueError("min_observations must be >= 1")
    for col in (vp_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    per_target_vp_n = df.groupby(target_col, dropna=False)[vp_col].transform("nunique")
    return df[per_target_vp_n >= min_observations].copy()


def _default_output(input_path: Path) -> Path:
    out_name = input_path.with_suffix("").with_suffix(f".mainland.csv").name
    return _DEFAULT_OUT_DIR / out_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT,
                        help="Raw  CSV to process.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV. Defaults to <input>.mainland.csv.")
    parser.add_argument("--min-observations", type=int, default=_DEFAULT_MIN_OBS,
                        help="Minimum observation threshold under --observation-mode. Default 3.")
    parser.add_argument("--vp-col", type=str, default=_DEFAULT_VP_COL,
                        help="VP identifier column. Default VP_ID.")
    parser.add_argument("--target-col", type=str, default=_DEFAULT_TARGET_COL,
                        help="Target identifier column. Default TARGET_ID.")
    parser.add_argument(
        "--observation-mode",
        choices=("target-vps", "pair-rows"),
        default="target-vps",
        help=(
            "How to count observations: "
            "target-vps=distinct VPs per target (default), "
            "pair-rows=row count per (VP,target) pair."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    raw_df = pd.read_csv(args.input)

    mainland_df = filter_non_mainland_us_targets_and_vps(raw_df)
    if args.observation_mode == "target-vps":
        kept_df = filter_pairs_by_min_target_vp_observations(
            mainland_df,
            vp_col=args.vp_col,
            target_col=args.target_col,
            min_observations=args.min_observations,
        )
    else:
        kept_df = filter_pairs_by_min_observations(
            mainland_df,
            vp_col=args.vp_col,
            target_col=args.target_col,
            min_observations=args.min_observations,
        )

    out_path = args.output or _default_output(args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(out_path, index=False)

    n_raw = len(raw_df)
    n_mainland = len(mainland_df)
    n_kept = len(kept_df)

    raw_pairs = raw_df[[args.vp_col, args.target_col]].drop_duplicates().shape[0] if all(
        c in raw_df.columns for c in (args.vp_col, args.target_col)
    ) else 0
    mainland_pairs = mainland_df[[args.vp_col, args.target_col]].drop_duplicates().shape[0]
    kept_pairs = kept_df[[args.vp_col, args.target_col]].drop_duplicates().shape[0]

    logger.info("stage 1: mainland filter")
    logger.info("  rows  : %d / %d (%.2f%%)",
                n_mainland, n_raw, 100 * n_mainland / n_raw if n_raw else 0.0)
    if args.observation_mode == "target-vps":
        logger.info("stage 2: keep targets with >= %d distinct %s (%s mode)",
                    args.min_observations, args.vp_col, args.observation_mode)
    else:
        logger.info("stage 2: min observations per (%s, %s) >= %d (%s mode)",
                    args.vp_col, args.target_col, args.min_observations, args.observation_mode)
    logger.info("  rows  : %d / %d (%.2f%% of mainland)",
                n_kept, n_mainland, 100 * n_kept / n_mainland if n_mainland else 0.0)
    logger.info("  pairs : %d / %d / %d (kept/mainland/raw)",
                kept_pairs, mainland_pairs, raw_pairs)
    if args.vp_col in kept_df.columns:
        logger.info("  VPs   : %d", kept_df[args.vp_col].nunique())
    if args.target_col in kept_df.columns:
        logger.info("  TGs   : %d", kept_df[args.target_col].nunique())
    logger.info("  wrote %s", out_path)


if __name__ == "__main__":
    main()
