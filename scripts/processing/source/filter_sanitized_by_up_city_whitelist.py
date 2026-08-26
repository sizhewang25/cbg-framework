"""Filter sanitized CBG rows by a (vp_id, target_norm_city) whitelist.

Expected sanitized input columns are canonical lowercase:
- ``vp_id``
- ``target_norm_city``

Whitelist CSV must contain:
- ``vp_id``
- ``target_norm_city``

CLI::

    .venv/bin/python -m scripts.processing.source.filter_sanitized_by_up_city_whitelist \
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

_DEFAULT_WHITELIST = Path(
    "datasets/up_asn_traffic/"
)

_VP = "vp_id"
_CITY = "target_norm_city"


def _default_output(input_path: Path) -> Path:
    return input_path.with_suffix("").with_suffix(".traffic-weighted.csv")


def _default_summary(output_path: Path) -> Path:
    return output_path.with_suffix("").with_suffix(".summary.json")


def _require_columns(df: pd.DataFrame, cols: list[str], dataset_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {dataset_name}: {missing}")


def _build_key(df: pd.DataFrame, *, vp_col: str, city_col: str) -> pd.Series:
    vp = df[vp_col].astype(str).str.strip().str.upper()
    city = df[city_col].map(normalize_city_name)
    return vp + "\t" + city


def filter_by_whitelist(sanitized: pd.DataFrame, whitelist: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    _require_columns(sanitized, [_VP, _CITY], "sanitized")
    _require_columns(whitelist, [_VP, _CITY], "whitelist")

    wl = whitelist.copy()
    wl_key = _build_key(wl, vp_col=_VP, city_col=_CITY)
    wl_keys = set(wl_key[wl_key.str.strip() != "\t"].tolist())

    san = sanitized.copy()
    san_key = _build_key(san, vp_col=_VP, city_col=_CITY)
    keep_mask = san_key.isin(wl_keys)
    kept = san[keep_mask].copy().reset_index(drop=True)

    n_rows_in = int(len(sanitized))
    n_rows_out = int(len(kept))
    vp_in = int(sanitized[_VP].astype(str).nunique())
    vp_out = int(kept[_VP].astype(str).nunique()) if n_rows_out else 0
    city_in = int(sanitized[_CITY].astype(str).nunique())
    city_out = int(kept[_CITY].astype(str).nunique()) if n_rows_out else 0

    stats: dict[str, int | float] = {
        "rows_in": n_rows_in,
        "rows_out": n_rows_out,
        "rows_retention_pct": (100.0 * n_rows_out / n_rows_in) if n_rows_in else 0.0,
        "vp_in": vp_in,
        "vp_out": vp_out,
        "city_in": city_in,
        "city_out": city_out,
        "whitelist_pairs": int(len(wl_keys)),
        "kept_unique_pairs": int(
            kept[[_VP, _CITY]].astype(str).drop_duplicates().shape[0]
        ) if n_rows_out else 0,
    }
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT,
                        help="Sanitized CBG CSV to filter.")
    parser.add_argument("--whitelist", type=Path, default=_DEFAULT_WHITELIST,
                        help="Whitelist CSV with vp_id and target_norm_city.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Filtered output CSV.")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Optional summary JSON output path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.whitelist.exists():
        raise SystemExit(f"Whitelist not found: {args.whitelist}")

    sanitized = pd.read_csv(args.input)
    whitelist = pd.read_csv(args.whitelist)

    kept, stats = filter_by_whitelist(sanitized, whitelist)

    out_path = args.output or _default_output(args.input)
    summary_path = args.summary or _default_summary(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    logger.info("rows    : %d / %d (%.2f%%)", stats["rows_out"], stats["rows_in"], stats["rows_retention_pct"])
    logger.info("VPs     : %d / %d", stats["vp_out"], stats["vp_in"])
    logger.info("cities  : %d / %d", stats["city_out"], stats["city_in"])
    logger.info("pairs   : %d (whitelist=%d)", stats["kept_unique_pairs"], stats["whitelist_pairs"])
    logger.info("wrote filtered CSV: %s", out_path)
    logger.info("wrote summary    : %s", summary_path)


if __name__ == "__main__":
    main()
