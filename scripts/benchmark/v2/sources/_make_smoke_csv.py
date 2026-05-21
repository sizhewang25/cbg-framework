"""Generate a small synthetic CSV that matches GenericCSVSource's schema.

Run:
    python -m scripts.benchmark.v2.sources._make_smoke_csv [/tmp/smoke.csv]

Produces a CSV with ~750 rows across 25 VPs × 30 targets spread over the
continental US. RTT for each (vp, target) pair is a noisy linear function
of great-circle distance, with the slope chosen so that bounded_spline and
normal_dist LTDs see enough signal to fit cleanly.

This is enough to exercise every combo in scripts/benchmark/v2/config/smoke.yaml
without depending on the real Vultr / RIPE Atlas datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.libs.cbg.rtt_model import haversine_distance

# CONUS bounding box (rough).
LAT_RANGE = (24.5, 48.5)
LON_RANGE = (-124.0, -67.0)

N_VPS = 25
N_TARGETS = 30
SEED = 42

# RTT model: rtt_ms = slope_ms_per_km * distance_km + intercept + N(0, sigma).
# slope ≈ 0.013 ms/km matches the empirical envelope of real public Internet
# pings (theoretical lower bound at 2/3c is 0.01 ms/km).
SLOPE_MS_PER_KM = 0.013
INTERCEPT_MS = 5.0
NOISE_SIGMA_MS = 3.0


def _sample_points(n: int, rng: np.random.Generator) -> np.ndarray:
    lats = rng.uniform(*LAT_RANGE, size=n)
    lons = rng.uniform(*LON_RANGE, size=n)
    return np.column_stack([lats, lons])


def build_csv(out_path: Path) -> int:
    rng = np.random.default_rng(SEED)

    vp_coords = _sample_points(N_VPS, rng)
    target_coords = _sample_points(N_TARGETS, rng)
    asns = rng.integers(low=1000, high=64000, size=N_VPS)

    rows = []
    for vi, (vp_lat, vp_lon) in enumerate(vp_coords):
        vp_id = f"vp{vi:03d}"
        vp_asn = int(asns[vi])
        for ti, (tg_lat, tg_lon) in enumerate(target_coords):
            target_id = f"tg{ti:03d}"
            d_km = haversine_distance(vp_lat, vp_lon, tg_lat, tg_lon)
            noise = rng.normal(0.0, NOISE_SIGMA_MS)
            rtt = max(0.5, SLOPE_MS_PER_KM * d_km + INTERCEPT_MS + noise)
            rows.append(
                {
                    "vp_id": vp_id,
                    "vp_lat": float(vp_lat),
                    "vp_lon": float(vp_lon),
                    "target_id": target_id,
                    "target_lat": float(tg_lat),
                    "target_lon": float(tg_lon),
                    "rtt_ms": float(rtt),
                    "vp_asn": vp_asn,
                    "vp_country": "US",
                }
            )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/geoscale_smoke.csv")
    n = build_csv(out_path)
    print(f"wrote {out_path} ({n} rows, {N_VPS} VPs × {N_TARGETS} targets)")


if __name__ == "__main__":
    main()
