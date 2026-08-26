"""Derive a traffic-gated (vp_id, target_norm_city) whitelist from UP-city pairs.

This script reads a UP-city traffic table (for example
``asn_*_unique_up_city_pairs.mainland_us.csv``), sorts pairs by ``TOTAL_TB``
descending, then keeps the smallest prefix that reaches at least a requested
cumulative traffic fraction.

The output is a whitelist keyed by canonical columns:
- ``vp_id``
- ``target_norm_city``

CLI::

    .venv/bin/python -m scripts.processing.source.derive_up_city_whitelist \
        --kept-traffic-fraction 0.95 \
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path(
    "datasets/up_asn_traffic/"
)
_DEFAULT_KEPT_TRAFFIC_FRACTION = 0.95

_UP = "UP"
_TOTAL_TB = "TOTAL_TB"
_OUT_VP = "vp_id"
_OUT_CITY = "target_norm_city"
_CITY_CANDIDATES = (
    "LOC_CITY",
    "CAP_SERVER_NORM_CITY",
    "TARGET_NORM_CITY",
    "UP_CITY",
)


def _default_output(input_path: Path, kept_traffic_fraction: float) -> Path:
    tag = f"keep{kept_traffic_fraction:g}"
    return input_path.with_suffix("").with_suffix(f".{tag}.up-city-whitelist.csv")


def _default_summary(output_path: Path) -> Path:
    return output_path.with_suffix("").with_suffix(".summary.json")


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _resolve_city_column(df: pd.DataFrame) -> str:
    for col in _CITY_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Missing city column. Expected one of: "
        f"{', '.join(_CITY_CANDIDATES)}"
    )


def derive_whitelist(
    df: pd.DataFrame,
    kept_traffic_fraction: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return whitelist dataframe and summary statistics."""
    if not (0 < kept_traffic_fraction <= 1):
        raise ValueError(
            f"kept_traffic_fraction must be in (0, 1], got {kept_traffic_fraction}"
        )
    _require_columns(df, [_UP, _TOTAL_TB])
    city_col = _resolve_city_column(df)

    work = pd.DataFrame(
        {
            _OUT_VP: df[_UP].astype(str).str.strip().str.upper(),
            _OUT_CITY: df[city_col].map(normalize_city_name),
            _TOTAL_TB: pd.to_numeric(df[_TOTAL_TB], errors="coerce").fillna(0.0),
        }
    )

    work = work[(work[_OUT_VP] != "") & (work[_OUT_CITY] != "")]

    # Collapse to one traffic value per (vp_id, city) pair.
    per_pair = (
        work.groupby([_OUT_VP, _OUT_CITY], as_index=False)
        .agg(pair_total_tb=(_TOTAL_TB, "sum"), n_rows=(_TOTAL_TB, "size"))
        .sort_values("pair_total_tb", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    total_tb = float(per_pair["pair_total_tb"].sum())
    if total_tb <= 0:
        empty_out = per_pair[[_OUT_VP, _OUT_CITY, "pair_total_tb"]].copy()
        summary = {
            "kept_traffic_fraction_target": float(kept_traffic_fraction),
            "kept_traffic_fraction_achieved": 1.0,
            "total_pairs": int(len(per_pair)),
            "kept_pairs": int(len(per_pair)),
            "total_tb": 0.0,
            "kept_tb": 0.0,
            "threshold_tb": 0.0,
        }
        return empty_out, summary

    target_tb = kept_traffic_fraction * total_tb
    sorted_tb = per_pair["pair_total_tb"].to_numpy(dtype=float)
    csum = np.cumsum(sorted_tb)
    idx = int(np.searchsorted(csum, target_tb, side="left"))
    idx = min(idx, len(sorted_tb) - 1)
    threshold_tb = float(sorted_tb[idx])

    kept = per_pair[per_pair["pair_total_tb"] >= threshold_tb].copy()
    kept_tb = float(kept["pair_total_tb"].sum())
    achieved = kept_tb / total_tb if total_tb > 0 else 1.0

    summary = {
        "city_column": city_col,
        "kept_traffic_fraction_target": float(kept_traffic_fraction),
        "kept_traffic_fraction_achieved": float(achieved),
        "total_pairs": int(len(per_pair)),
        "kept_pairs": int(len(kept)),
        "total_tb": float(total_tb),
        "kept_tb": float(kept_tb),
        "threshold_tb": float(threshold_tb),
    }

    return kept[[_OUT_VP, _OUT_CITY, "pair_total_tb"]], summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT,
                        help="UP-city traffic CSV.")
    parser.add_argument(
        "--kept-traffic-fraction",
        type=float,
        default=_DEFAULT_KEPT_TRAFFIC_FRACTION,
        help="Keep at least this fraction of TOTAL_TB by pair prefix. Default 0.95.",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Whitelist output CSV.")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Optional summary JSON path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    whitelist, summary = derive_whitelist(df, args.kept_traffic_fraction)

    out_path = args.output or _default_output(args.input, args.kept_traffic_fraction)
    summary_path = args.summary or _default_summary(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    whitelist.to_csv(out_path, index=False)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(
        "target traffic >= %.3f -> kept %d / %d pairs (%.2f%% TB)",
        summary["kept_traffic_fraction_target"],
        summary["kept_pairs"],
        summary["total_pairs"],
        100.0 * summary["kept_traffic_fraction_achieved"],
    )
    logger.info("  threshold TOTAL_TB: %.12g", summary["threshold_tb"])
    logger.info("  wrote whitelist: %s", out_path)
    logger.info("  wrote summary  : %s", summary_path)


if __name__ == "__main__":
    main()
