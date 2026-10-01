"""Scale up the weights of flows near a PNI site in a weighted mesh CSV.

SYNTHETIC fixture. Multiplies the weight of selected rows by `--factor`; every
other row is left untouched, so the result models traffic that concentrates at
the interconnects. A row with weight 0 stays 0. Two selection rules:

  --rule target  every flow of a target within `--radius-km` of any site in
                 `--pni-csv` (the mirror image of the `pnizero` file, which
                 zeroes the targets far from every site).
  --rule flow    only flows where the target AND the VP are both within
                 `--radius-km` of the target's nearest site -- the same site
                 for both ends, so a VP near some other PNI does not count.
                 Approximates traffic that enters at the PNI serving the
                 target and is measured from a VP behind that same PNI.

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
_OUTPUT_SUFFIX = {"target": ".pniboost.csv", "flow": ".pniboost-flow.csv"}
_EARTH_RADIUS_KM = 6371.0


def _default_output(input_path: Path, rule: str) -> Path:
    # with_name, not with_suffix: the latter eats a stage tag on a multi-dot stem.
    return input_path.with_name(input_path.stem + _OUTPUT_SUFFIX[rule])


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Broadcasting haversine distance in km."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_pni(lat, lon, pni_lat, pni_lon) -> tuple[np.ndarray, np.ndarray]:
    """(index of, distance in km to) each point's nearest PNI site."""
    d = _haversine_km(
        np.asarray(lat, dtype=float)[:, None], np.asarray(lon, dtype=float)[:, None],
        np.asarray(pni_lat, dtype=float)[None, :], np.asarray(pni_lon, dtype=float)[None, :],
    )
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(d)), idx]


def boost_pni_weights(
    df: pd.DataFrame,
    pni: pd.DataFrame,
    *,
    rule: str = "target",
    radius_km: float = 100.0,
    factor: float = 100.0,
    decimals: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Return the boosted mesh, its boolean row mask, and each row's TG-side PNI id."""
    if factor <= 0:
        raise ValueError(f"--factor must be > 0, got {factor}")
    if rule not in ("target", "flow"):
        raise ValueError(f"--rule must be 'target' or 'flow', got {rule!r}")
    pni_lat = pni["pni_lat"].to_numpy(dtype=float)
    pni_lon = pni["pni_lon"].to_numpy(dtype=float)
    k, d_tg = nearest_pni(df["target_lat"], df["target_lon"], pni_lat, pni_lon)
    boost = d_tg <= radius_km
    if rule == "flow":
        d_vp = _haversine_km(df["vp_lat"], df["vp_lon"], pni_lat[k], pni_lon[k])
        boost &= d_vp <= radius_km
    out = df.copy()
    out.loc[boost, _WEIGHT] = np.round(out.loc[boost, _WEIGHT] * factor, decimals)
    tg_pni = pd.Series(pni["pni_id"].to_numpy()[k], index=df.index, name="tg_pni")
    return out, boost, tg_pni


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="Weighted mesh CSV (must carry a weight column).")
    parser.add_argument("--pni-csv", type=Path, required=True,
                        help="PNI site list with pni_lat/pni_lon columns.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <stem>.pniboost.csv (target) or <stem>.pniboost-flow.csv (flow) beside the input.")
    parser.add_argument("--rule", choices=("target", "flow"), default="target",
                        help="target: all flows of near-PNI targets. flow: only flows whose "
                             "VP and TG are both near the TG's nearest PNI. Default target.")
    parser.add_argument("--radius-km", type=float, default=100.0,
                        help="Distance threshold for the rule. Default 100.")
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

    out, boost, tg_pni = boost_pni_weights(
        df, pni, rule=args.rule, radius_km=args.radius_km, factor=args.factor,
        decimals=args.decimals,
    )
    out_path = args.output or _default_output(args.input, args.rule)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    w = out[_WEIGHT].to_numpy(dtype=float)
    logger.info("SYNTHETIC PNI boost: rule=%s, x%g within %g km of %d sites in %s",
                args.rule, args.factor, args.radius_km, len(pni), args.pni_csv)
    logger.info("  rows        : %d boosted / %d total", int(boost.sum()), len(out))
    logger.info("  targets     : %d with a boosted flow / %d total",
                out.loc[boost, "target_id"].nunique(), out["target_id"].nunique())
    logger.info("  vps         : %d with a boosted flow / %d total",
                out.loc[boost, "vp_id"].nunique(), out["vp_id"].nunique())
    logger.info("  pni sites   : %d of %d carry a boosted flow",
                tg_pni[boost].nunique(), len(pni))
    logger.info("  weight      : total %.6g | boosted share %.2f%%",
                w.sum(), 100 * w[boost].sum() / w.sum())
    logger.info("  wrote %s", out_path)


if __name__ == "__main__":
    main()
