"""
Computes a row filter threshold from a target kept-traffic fraction over
deduped ``(VP_ID, TARGET_NORM_CITY)`` pairs:

1) Deduplicate to one weight per VP-TG city pair.
2) Sort pair weights descending and cumulative-sum.
3) Pick the smallest threshold that keeps at least the requested traffic
    fraction (default ``0.95``) at the pair level.
4) Keep rows with ``WEIGHT >= threshold``.

``WEIGHT`` is the normalized owner-pair traffic share produced by

CLI::

    python -m scripts.processing.ant.filter_weighted_pairs \\
        --kept-traffic-fraction 0.95
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_KEPT_TRAFFIC_FRACTION = 0.95

_WEIGHT = "WEIGHT"
_TRAFFIC = "TRAFFIC_TB"
_VP = "VP_ID"
_CITY = "TARGET_NORM_CITY"


def filter_by_weight(df: pd.DataFrame, min_weight: float) -> pd.DataFrame:
    """Return the rows whose ``WEIGHT`` is >= ``min_weight``."""
    if _WEIGHT not in df.columns:
        raise ValueError(f"Missing {_WEIGHT} column")
    return df[df[_WEIGHT].fillna(0.0) >= min_weight].reset_index(drop=True)


def derive_threshold_for_kept_pair_traffic(
    df: pd.DataFrame,
    kept_traffic_fraction: float,
) -> tuple[float, float, int, int, int]:
    """Return threshold and pair-level retention stats.

    The threshold is computed over deduped ``(VP_ID, TARGET_NORM_CITY)``
    weights by descending cumulative sum.
    """
    if _WEIGHT not in df.columns:
        raise ValueError(f"Missing {_WEIGHT} column")
    if _VP not in df.columns or _CITY not in df.columns:
        raise ValueError(f"Missing {_VP} or {_CITY} column")
    if not (0 < kept_traffic_fraction <= 1):
        raise ValueError(
            f"kept_traffic_fraction must be in (0, 1], got {kept_traffic_fraction}"
        )

    pair = df[[_VP, _CITY, _WEIGHT]].copy()
    pair[_WEIGHT] = pd.to_numeric(pair[_WEIGHT], errors="coerce").fillna(0.0)

    # We expect one weight per pair; use max as a robust fallback.
    per_pair = (
        pair.groupby([_VP, _CITY], as_index=False)
        .agg(weight=(_WEIGHT, "max"), distinct_values=(_WEIGHT, "nunique"))
    )
    inconsistent_pairs = int((per_pair["distinct_values"] > 1).sum())

    weights = per_pair["weight"].to_numpy(dtype=float)
    total = float(weights.sum())
    if total <= 0:
        # Degenerate case: everything is weightless.
        return 0.0, 1.0, len(per_pair), len(per_pair), inconsistent_pairs

    weights_sorted = np.sort(weights)[::-1]
    target = kept_traffic_fraction * total
    cum = np.cumsum(weights_sorted)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(weights_sorted) - 1)
    threshold = float(weights_sorted[idx])

    kept_pairs = int((weights >= threshold).sum())
    achieved = float(weights[weights >= threshold].sum() / total)
    return threshold, achieved, kept_pairs, len(per_pair), inconsistent_pairs


def _default_output(input_path: Path, kept_traffic_fraction: float) -> Path:
    tag = f"keep{kept_traffic_fraction:g}"
    return input_path.with_suffix("").with_suffix(f".{tag}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--kept-traffic-fraction",
        type=float,
        default=_DEFAULT_KEPT_TRAFFIC_FRACTION,
        help=(
            "Keep at least this fraction of deduped VP-TG city pair traffic by "
            "reverse-solving a WEIGHT threshold. Default 0.95."
        ),
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV. Defaults to <input>.keep<fraction>.csv.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    min_weight, pair_cov, kept_pairs, total_pairs, inconsistent = (
        derive_threshold_for_kept_pair_traffic(df, args.kept_traffic_fraction)
    )
    kept = filter_by_weight(df, min_weight)

    out_path = args.output or _default_output(args.input, args.kept_traffic_fraction)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    n, k = len(df), len(kept)
    tot_tb = pd.to_numeric(df.get(_TRAFFIC), errors="coerce").sum()
    kept_tb = pd.to_numeric(kept.get(_TRAFFIC), errors="coerce").sum()
    logger.info(
        "target kept pair traffic >= %.3f -> derived WEIGHT threshold %.12g",
        args.kept_traffic_fraction,
        min_weight,
    )
    logger.info(
        "  pair-level coverage: %d / %d pairs (%.2f%% traffic)",
        kept_pairs,
        total_pairs,
        100 * pair_cov,
    )
    if inconsistent:
        logger.info(
            "  note: %d VP-TG city pairs had non-identical row weights; "
            "pair threshold used per-pair max(weight)",
            inconsistent,
        )
    logger.info("  rows    : %d / %d (%.1f%%)", k, n, 100 * k / n if n else 0.0)
    if tot_tb and tot_tb > 0:
        logger.info("  traffic : %.1f / %.1f TB (%.2f%%)",
                    kept_tb, tot_tb, 100 * kept_tb / tot_tb)
    if _CITY in df.columns:
        logger.info("  cities  : %d / %d", kept[_CITY].nunique(), df[_CITY].nunique())
    if _VP in df.columns:
        logger.info("  VPs     : %d / %d", kept[_VP].nunique(), df[_VP].nunique())
    logger.info("  wrote %s", out_path)


if __name__ == "__main__":
    main()
