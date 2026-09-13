"""Prune low-traffic (vp_id, target_id) flows from a weighted mesh CSV.

Dataset-characterisation tool. The benchmark does NOT consume its output: with
`eval_kept_traffic_fraction` fixed in `GenericCSVSource`, the benchmark derives
the same threshold itself from the weight-bearing mesh. What this script adds is
the filtered CSV (for dataset figures) and the node-loss statistics the
benchmark does not report -- plus an independent cross-check that both
derivations agree.

Algorithm -- keyless, at flow level:
  1. `total` = sum of `weight` over every row. Mesh weights are typically shares
     of a larger universe (the mesh itself samples top targets per location) and
     need not sum to 1; renormalizing by `total` is what makes the fraction
     meaningful.
  2. Sort flow weights descending, cumulative-sum.
  3. Take the smallest prefix reaching `kept_traffic_fraction * total`; that
     prefix's last weight is the threshold w*.
  4. Keep rows with `weight >= w*`.

Pruning *edges* is the point. A VP or a target may lose every one of its flows
and vanish from the output entirely -- node presence becomes an outcome of
traffic rather than something preserved by construction. Node loss is reported
as a first-class statistic in the summary JSON.

The `weight` column is REQUIRED; its absence is a hard error, never a default.
`GenericCSVSource` fills a missing weight column with 1.0 everywhere, and under
that fill this filter degenerates into a 100%-retention no-op that would still
emit a file named `.traffic-weighted.csv`.

CLI::

    .venv/bin/python -m scripts.processing.source.filter_weighted_flows \
        --input datasets/final/<stem>.csv \
        --kept-traffic-fraction 0.95
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_KEPT_TRAFFIC_FRACTION = 0.95
_DEFAULT_VP_COL = "vp_id"
_DEFAULT_TARGET_COL = "target_id"
_DEFAULT_WEIGHT_COL = "weight"

_OUTPUT_SUFFIX = ".traffic-weighted.csv"
_SUMMARY_SUFFIX = ".summary.json"


def _default_output(input_path: Path) -> Path:
    # NOT `with_suffix("").with_suffix(...)`: on a multi-dot stem like
    # `x.mainland.sanitized.csv` that idiom eats `.sanitized`.
    return input_path.with_name(input_path.stem + _OUTPUT_SUFFIX)


def _default_summary(output_path: Path) -> Path:
    # Same trap, worse consequence: the with_suffix idiom would resolve to
    # `x.mainland.sanitized.summary.json` and clobber the SOI summary that
    # preprocess_cbg_raw_data.smk writes to the same OUTPUTS_DIR.
    return output_path.with_name(output_path.stem + _SUMMARY_SUFFIX)


def resolve_column(df: pd.DataFrame, requested: str) -> str:
    """Exact match, else a unique case-insensitive match. Lookup only -- the
    output header is never rewritten."""
    if requested in df.columns:
        return requested
    matches = [c for c in df.columns if c.lower() == requested.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"column {requested!r} is ambiguous case-insensitively: {matches}"
        )
    raise ValueError(
        f"missing required column {requested!r}; columns present: {list(df.columns)}"
    )


def build_flow_table(
    df: pd.DataFrame, *, vp_col: str, target_col: str, weight_col: str
) -> pd.DataFrame:
    """Validate the weight column and assert the mesh is edge-unique.

    Every ambiguous case is a hard error rather than a policy flag -- at flow
    level a silently-coerced weight becomes a deleted edge, and deleted edges
    cascade into deleted nodes, which is indistinguishable from the real signal
    this dataset exists to carry.
    """
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    n_bad = int(weights.isna().sum())
    if n_bad:
        bad = df.loc[weights.isna(), [vp_col, target_col]].head(3).to_dict("records")
        raise ValueError(
            f"{n_bad} rows have NaN/non-numeric {weight_col!r} (e.g. {bad}). "
            f"At flow level an unknown weight silently deletes an edge and can "
            f"delete a node with it -- fix the export rather than defaulting."
        )
    if (weights < 0).any():
        raise ValueError(f"{weight_col!r} must be >= 0 (found negative values)")

    dup = df.groupby([vp_col, target_col])[weight_col].nunique()
    conflicting = dup[dup > 1]
    if len(conflicting):
        raise ValueError(
            f"{len(conflicting)} ({vp_col}, {target_col}) flows carry conflicting "
            f"weights (e.g. {list(conflicting.index[:3])}). A value-based mask "
            f"would keep the heavy duplicate row and drop the light one, "
            f"splitting one physical flow across the filter boundary."
        )

    out = df.copy()
    out[weight_col] = weights
    return out


def derive_flow_weight_threshold(
    weights: np.ndarray, kept_traffic_fraction: float
) -> tuple[float, dict]:
    """Keyless descending-cumsum threshold over deduped flow weights."""
    if not 0 < kept_traffic_fraction <= 1:
        raise ValueError(
            f"kept_traffic_fraction must be in (0, 1], got {kept_traffic_fraction}"
        )
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(
            f"no traffic signal: total weight over {len(weights)} flows is "
            f"{total} -- every flow is weightless"
        )
    if len(weights) > 1 and float(weights.min()) == float(weights.max()):
        raise ValueError(
            f"no traffic signal: all {len(weights)} flows carry the identical "
            f"weight {float(weights[0])!r}, so any threshold retains 100% of "
            f"traffic at every fraction"
        )

    srt = np.sort(weights)[::-1]
    cum = np.cumsum(srt)
    idx = int(np.searchsorted(cum, kept_traffic_fraction * total, side="left"))
    # Load-bearing clamp: cum sums the sorted array while total sums the
    # original, so at frac=1.0 float error can push searchsorted past the end.
    idx = min(idx, len(srt) - 1)
    threshold = float(srt[idx])

    kept = weights >= threshold
    stats = {
        "kept_traffic_fraction_target": float(kept_traffic_fraction),
        "kept_traffic_fraction_achieved": float(weights[kept].sum() / total),
        "eval_pair_weight_min": threshold,
        "weight_total": total,
        "weight_kept": float(weights[kept].sum()),
        "prefix_flows": int(idx + 1),
        "flows_at_threshold": int((weights == threshold).sum()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_median": float(np.median(weights)),
        "flows_zero_weight": int((weights == 0).sum()),
    }
    return threshold, stats


def node_loss_stats(
    before: pd.DataFrame, after: pd.DataFrame, *, vp_col: str, target_col: str
) -> dict:
    """Node loss is the headline effect of flow pruning, so it is first-class."""
    vps_in = set(before[vp_col].astype(str))
    vps_out = set(after[vp_col].astype(str))
    tgs_in = set(before[target_col].astype(str))
    tgs_out = set(after[target_col].astype(str))

    vp_per_tg = after.groupby(target_col)[vp_col].nunique()
    buckets = {"1": 0, "2": 0, "3+": 0}
    for n in vp_per_tg:
        buckets["1" if n == 1 else "2" if n == 2 else "3+"] += 1

    return {
        "flows_in": int(len(before)),
        "flows_out": int(len(after)),
        "flows_retention_pct": float(100 * len(after) / len(before)) if len(before) else 0.0,
        "vps_in": len(vps_in), "vps_out": len(vps_out),
        "vps_eliminated": len(vps_in - vps_out),
        "vps_eliminated_ids": sorted(vps_in - vps_out),
        "targets_in": len(tgs_in), "targets_out": len(tgs_out),
        "targets_eliminated": len(tgs_in - tgs_out),
        "targets_surviving_ids": sorted(tgs_out),
        # preprocess_cbg_raw_data.smk enforces min_vp_obs=3 because
        # multilateration needs >=3 constraints. Flow pruning silently violates
        # that for SURVIVING targets -- reported, deliberately not enforced.
        "targets_out_by_vp_count": buckets,
        "targets_out_with_ge_3_vps": int(buckets["3+"]),
    }


def filter_flows(
    df: pd.DataFrame,
    *,
    kept_traffic_fraction: float,
    vp_col: str = _DEFAULT_VP_COL,
    target_col: str = _DEFAULT_TARGET_COL,
    weight_col: str = _DEFAULT_WEIGHT_COL,
) -> tuple[pd.DataFrame, dict]:
    """Return (kept rows, summary). Row order and columns are preserved."""
    vp_col = resolve_column(df, vp_col)
    target_col = resolve_column(df, target_col)
    weight_col = resolve_column(df, weight_col)

    work = build_flow_table(df, vp_col=vp_col, target_col=target_col, weight_col=weight_col)
    flows = work.drop_duplicates([vp_col, target_col])
    threshold, stats = derive_flow_weight_threshold(
        flows[weight_col].to_numpy(dtype=float), kept_traffic_fraction
    )

    kept = work[work[weight_col] >= threshold].reset_index(drop=True)
    summary = {
        "vp_column": vp_col, "target_column": target_col, "weight_column": weight_col,
        **stats,
        **node_loss_stats(work, kept, vp_col=vp_col, target_col=target_col),
    }
    overshoot = summary["kept_traffic_fraction_achieved"] - kept_traffic_fraction
    if overshoot > 0.01:
        logger.warning(
            "kept traffic overshoots target by %.2f pp: %d flows tie at the "
            "threshold weight %.12g and all are kept (>= semantics)",
            100 * overshoot, summary["flows_at_threshold"], threshold,
        )
    return kept, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="Weight-bearing mesh CSV (canonical schema).")
    parser.add_argument("--kept-traffic-fraction", type=float,
                        default=_DEFAULT_KEPT_TRAFFIC_FRACTION,
                        help="Retain at least this fraction of total flow traffic. Default 0.95.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Filtered CSV. Defaults to <stem>.traffic-weighted.csv.")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Summary JSON. Defaults to <output-stem>.summary.json.")
    parser.add_argument("--vp-col", default=_DEFAULT_VP_COL)
    parser.add_argument("--target-col", default=_DEFAULT_TARGET_COL)
    parser.add_argument("--weight-col", default=_DEFAULT_WEIGHT_COL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    kept, summary = filter_flows(
        df,
        kept_traffic_fraction=args.kept_traffic_fraction,
        vp_col=args.vp_col, target_col=args.target_col, weight_col=args.weight_col,
    )

    out_path = args.output or _default_output(args.input)
    summary_path = args.summary or _default_summary(out_path)
    summary = {"input": str(args.input), "output": str(out_path), **summary}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(
        "kept traffic >= %.3f -> eval_pair_weight_min=%.12g (%.2f%% achieved)",
        summary["kept_traffic_fraction_target"], summary["eval_pair_weight_min"],
        100 * summary["kept_traffic_fraction_achieved"],
    )
    logger.info("  flows   : %d / %d (%.2f%%)",
                summary["flows_out"], summary["flows_in"], summary["flows_retention_pct"])
    logger.info("  VPs     : %d / %d (%d eliminated)",
                summary["vps_out"], summary["vps_in"], summary["vps_eliminated"])
    logger.info("  targets : %d / %d (%d eliminated)",
                summary["targets_out"], summary["targets_in"], summary["targets_eliminated"])
    logger.info("  surviving targets by VP count: %s", summary["targets_out_by_vp_count"])
    logger.info("  wrote %s", out_path)
    logger.info("  wrote %s", summary_path)


if __name__ == "__main__":
    main()
