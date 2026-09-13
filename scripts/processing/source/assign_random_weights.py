"""Attach SYNTHETIC random flow weights to a canonical mesh CSV.

Test-fixture generator, not a data-processing step. It exists so the
traffic-weighted path (filter_weighted_flows, TrafficWeightedCSVSource's
`eval_kept_traffic_fraction`) can be exercised end to end before real per-flow
traffic volume is available.

Draws one weight per row, i.i.d. uniform on [low, high], rounded to
`--decimals`. Deterministic in `--seed`.

WHAT A UNIFORM DRAW DOES AND DOES NOT EXERCISE
----------------------------------------------
Real traffic is heavy-tailed and concentrates on *targets*: a handful of
targets carry most of the bytes, so a 95% cut deletes most targets outright and
the surviving set is the "traffic-heavy VPs and TGs" the paper describes.

A uniform draw is flat and independent per row, so it does almost the opposite.
Keeping the top `f` of uniform traffic keeps a `sqrt(1 - f)`-thresholded
fraction `1 - sqrt(1 - f)` of rows -- about 78% of flows at f=0.95 -- and those
survivors are scattered evenly across every target. A target is eliminated only
if *all* of its flows fall below the threshold, which for a target observed by
k VPs has probability ~0.224^k: vanishing beyond a handful of VPs.

So this fixture validates the plumbing (threshold derivation, the eval-side
mask, fold independence, parquet round-trip) and NOT node elimination. For a
fixture that exercises node loss, draw a per-target mass from a heavy-tailed
distribution and spread it across that target's flows.

CLI::

    .venv/bin/python -m scripts.processing.source.assign_random_weights \
        --input datasets/final/<stem>.csv --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_WEIGHT = "weight"
_OUTPUT_SUFFIX = ".randweight.csv"


def _default_output(input_path: Path) -> Path:
    # with_name, not with_suffix: the latter eats a stage tag on a multi-dot
    # stem like `x.mainland.sanitized.csv`.
    return input_path.with_name(input_path.stem + _OUTPUT_SUFFIX)


def assign_random_weights(
    df: pd.DataFrame,
    *,
    seed: int,
    low: float = 0.0,
    high: float = 1.0,
    decimals: int = 3,
) -> pd.DataFrame:
    if high <= low:
        raise ValueError(f"--high must exceed --low, got low={low} high={high}")
    if low < 0:
        raise ValueError(f"weights must be >= 0, got low={low}")
    if decimals < 0:
        raise ValueError(f"--decimals must be >= 0, got {decimals}")
    out = df.copy()
    rng = np.random.default_rng(seed)
    out[_WEIGHT] = np.round(rng.uniform(low, high, size=len(out)), decimals)
    return out


def describe_cut(weights: np.ndarray, fraction: float) -> dict:
    """What `filter_weighted_flows` would do at this fraction — same kernel."""
    total = float(weights.sum())
    srt = np.sort(weights)[::-1]
    cum = np.cumsum(srt)
    idx = min(int(np.searchsorted(cum, fraction * total, side="left")), len(srt) - 1)
    threshold = float(srt[idx])
    kept = weights >= threshold
    return {
        "threshold": threshold,
        "kept_flows": int(kept.sum()),
        "kept_share_of_flows": float(kept.sum() / len(weights)),
        "achieved_traffic": float(weights[kept].sum() / total),
        "flows_at_threshold": int((weights == threshold).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="Canonical mesh CSV to copy and annotate.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <stem>.randweight.csv beside the input.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed; the draw is deterministic in it. Default 42.")
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--kept-traffic-fraction", type=float, default=0.95,
                        help="Reported only: what this cut would retain. Default 0.95.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing weight column (refused by default).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    if _WEIGHT in df.columns and not args.force:
        # Refuse by default: silently replacing measured traffic volume with a
        # random draw would be unrecoverable and invisible downstream.
        raise SystemExit(
            f"{args.input} already has a {_WEIGHT!r} column. Refusing to replace "
            f"real traffic with a synthetic draw; pass --force if that is intended."
        )

    out = assign_random_weights(
        df, seed=args.seed, low=args.low, high=args.high, decimals=args.decimals
    )
    out_path = args.output or _default_output(args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    w = out[_WEIGHT].to_numpy(dtype=float)
    cut = describe_cut(w, args.kept_traffic_fraction)
    logger.info("SYNTHETIC weights: uniform[%g, %g] rounded to %d dp, seed=%d",
                args.low, args.high, args.decimals, args.seed)
    logger.info("  rows        : %d", len(out))
    logger.info("  weight      : total %.6g | mean %.6g | zeros %d | distinct %d",
                w.sum(), w.mean(), int((w == 0).sum()), len(np.unique(w)))
    logger.info("  at %.2f cut  : threshold %.6g -> %d flows kept (%.1f%% of flows, "
                "%.2f%% of traffic; %d tie at the threshold)",
                args.kept_traffic_fraction, cut["threshold"], cut["kept_flows"],
                100 * cut["kept_share_of_flows"], 100 * cut["achieved_traffic"],
                cut["flows_at_threshold"])
    logger.info("  wrote %s", out_path)


if __name__ == "__main__":
    main()
