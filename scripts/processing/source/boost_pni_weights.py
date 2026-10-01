"""Scale up the flow weights of targets near a PNI site in a weighted mesh CSV.

SYNTHETIC fixture, the mirror image of the `pnizero` file: instead of zeroing
targets far from every PNI site, it multiplies the weight of every row whose
target lies within `--radius-km` of any site in `--pni-csv` by `--factor`.
Rows of far targets are left untouched, so the result models traffic that
concentrates at the interconnects. A row with weight 0 stays 0.

The 100 km default follows `configs/as01-pnizero-test.yaml`: as01's
target-to-nearest-PNI distances jump from 33 km to 182 km with nothing between,
so any radius in that gap selects the same 299 of 399 targets.

CLI::

    .venv/bin/python -m scripts.processing.source.boost_pni_weights \
        --input datasets/final/as01-20260728-20260802.mainland.sanitized.randweight.csv \
        --pni-csv datasets/pni/as01-us-pni.approx.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_WEIGHT = "weight"
_OUTPUT_SUFFIX = ".pniboost.csv"
_EARTH_RADIUS_KM = 6371.0


def _default_output(input_path: Path) -> Path:
    # with_name, not with_suffix: the latter eats a stage tag on a multi-dot stem.
    return input_path.with_name(input_path.stem + _OUTPUT_SUFFIX)


def nearest_pni_km(lat, lon, pni_lat, pni_lon) -> np.ndarray:
    """Distance from each (lat, lon) to its nearest PNI site, by haversine."""
    lat1, lon1 = (np.radians(np.asarray(v, dtype=float))[:, None] for v in (lat, lon))
    lat2, lon2 = (np.radians(np.asarray(v, dtype=float))[None, :] for v in (pni_lat, pni_lon))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return d.min(axis=1)


def boost_pni_weights(
    df: pd.DataFrame,
    pni: pd.DataFrame,
    *,
    radius_km: float = 100.0,
    factor: float = 100.0,
    decimals: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the boosted mesh and the per-target nearest-PNI distance (km)."""
    if factor <= 0:
        raise ValueError(f"--factor must be > 0, got {factor}")
    targets = df.groupby("target_id")[["target_lat", "target_lon"]].first()
    d_km = pd.Series(
        nearest_pni_km(targets["target_lat"], targets["target_lon"], pni["pni_lat"], pni["pni_lon"]),
        index=targets.index,
        name="nearest_pni_km",
    )
    near = df["target_id"].map(d_km <= radius_km).to_numpy(dtype=bool)
    out = df.copy()
    out.loc[near, _WEIGHT] = np.round(out.loc[near, _WEIGHT] * factor, decimals)
    return out, d_km


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="Weighted mesh CSV (must carry a weight column).")
    parser.add_argument("--pni-csv", type=Path, required=True,
                        help="PNI site list with pni_lat/pni_lon columns.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <stem>.pniboost.csv beside the input.")
    parser.add_argument("--radius-km", type=float, default=100.0,
                        help="Targets within this distance of a PNI site are boosted. Default 100.")
    parser.add_argument("--factor", type=float, default=100.0,
                        help="Multiplier applied to near-PNI weights. Default 100.")
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for p in (args.input, args.pni_csv):
        if not p.exists():
            raise SystemExit(f"Input not found: {p}")

    df = pd.read_csv(args.input)
    if _WEIGHT not in df.columns:
        raise SystemExit(f"{args.input} has no {_WEIGHT!r} column; run assign_random_weights first.")
    pni = pd.read_csv(args.pni_csv)

    out, d_km = boost_pni_weights(
        df, pni, radius_km=args.radius_km, factor=args.factor, decimals=args.decimals
    )
    out_path = args.output or _default_output(args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    near_t = d_km <= args.radius_km
    near_rows = out["target_id"].map(near_t).to_numpy(dtype=bool)
    w = out[_WEIGHT].to_numpy(dtype=float)
    inside = d_km[near_t].max() if near_t.any() else float("nan")
    outside = d_km[~near_t].min() if (~near_t).any() else float("nan")
    logger.info("SYNTHETIC PNI boost: x%g within %g km of %d sites in %s",
                args.factor, args.radius_km, len(pni), args.pni_csv)
    logger.info("  targets     : %d boosted / %d total (nearest-PNI gap %.1f km -> %.1f km)",
                int(near_t.sum()), len(d_km), inside, outside)
    logger.info("  rows        : %d boosted / %d total", int(near_rows.sum()), len(out))
    logger.info("  weight      : total %.6g | boosted share %.2f%%",
                w.sum(), 100 * w[near_rows].sum() / w.sum())
    logger.info("  wrote %s", out_path)


if __name__ == "__main__":
    main()
