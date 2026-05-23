"""Calibrate the one-way speed limit S from the anchor-mesh.

S is used downstream by the agreement verifier in
`scripts/vp_selection/agreement.py`. We report it as p99 over fitted
per-anchor LP slopes — max is dominated by a small number of broken-data
anchors and doesn't survive as a stable calibration.

Pipeline:
  1. Load anchor-mesh pings via `RipeAtlasSource(ping_table=anchors_meshed_pings)`,
     PROBES_TO_ANCHORS setup. Source IPs become VpIds (each anchor as a "VP").
  2. Run SOI sanitization (default on) to drop anchors with structural
     GT-vs-RTT inconsistencies.
  3. For each anchor: call `RTTDistanceModel.fit()` directly with its default
     `baseline_slope = THEORETICAL_SLOPE = 0.01 ms/km` (= 2/3·c speed cap,
     the same constraint production CBG runs use). This is the LP under
     `LowEnvelopeLTD`; we bypass the wrapper to keep the calibration loop
     simple, but the LP itself is identical to production.
  4. Per-anchor implied one-way speed: v_i = 2 / slope_i (km/ms). The
     factor of 2 accounts for RTT being round-trip.
  5. **Detect pegged anchors**: any with `slope_i ≤ THEORETICAL_SLOPE + ε` —
     their data implies marginal propagation faster than 2/3·c, so the LP
     pinned them at the floor. These are degenerate fits (GT slippage,
     clock skew, sparse-data overfit) — exclude them from the calibration.
  6. **Headline S = p99** over the remaining fitted anchors. Robust to a
     single outlier. Max, p95, p50, etc. retained as JSON diagnostics.

Outputs `outputs/speed_calibration.json` and `outputs/speed_calibration.png`.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import default
from scripts.benchmark.v2.sources.base import DataSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.libs.cbg.rtt_model import (
    RTTDistanceModel,
    SPEED_OF_LIGHT_KM_MS,
    THEORETICAL_SLOPE,
    haversine_distance,
)

logger = logging.getLogger(__name__)

# Cho et al. 2024 calibrated one-way speed (km/ms). 0.51 c.
CHO_2024_REFERENCE_KM_PER_MS = 153.0
SPEED_OF_LIGHT_KM_PER_MS = SPEED_OF_LIGHT_KM_MS
SOI_SPEED_KM_PER_MS = (2.0 / 3.0) * SPEED_OF_LIGHT_KM_PER_MS  # 200 km/ms; matches THEORETICAL_SLOPE


def _slope_to_one_way_speed(slope_ms_per_km: float) -> float:
    """One-way speed in km/ms from RTT-vs-distance slope (ms/km). RTT/2 = d/v
    so v = 2d/RTT; at large d the LP slope approaches RTT/d, so v → 2/slope."""
    return 2.0 / slope_ms_per_km


def calibrate_from_source(source: DataSource) -> dict[str, Any]:
    """Run the calibration against an already-configured DataSource.

    Returns a dict with summary stats + per-anchor records. Caller is
    responsible for serialization and plotting.
    """
    samples = list(source.iter_fit_samples())
    logger.info("loaded %d anchor-mesh FitSamples", len(samples))
    if not samples:
        raise RuntimeError("no FitSamples produced — check ClickHouse / sanitize settings")

    by_vp: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rtts": [], "distances": [], "vp_coord": None}
    )
    for s in samples:
        d = haversine_distance(
            s.vp_coord.lat, s.vp_coord.lon,
            s.probe_coord.lat, s.probe_coord.lon,
        )
        bucket = by_vp[str(s.vp_id)]
        bucket["rtts"].append(float(s.latency))
        bucket["distances"].append(d)
        bucket["vp_coord"] = s.vp_coord

    pegged_tolerance = 1e-9
    per_anchor: list[dict[str, Any]] = []
    for vp_id, data in by_vp.items():
        vp_coord = data["vp_coord"]
        model = RTTDistanceModel(
            anchor_ip=vp_id,
            anchor_lat=vp_coord.lat,
            anchor_lon=vp_coord.lon,
        )
        try:
            model.fit(
                distances=np.array(data["distances"], dtype=float),
                rtts=np.array(data["rtts"], dtype=float),
            )
        except Exception as exc:
            logger.debug("fit failed for %s: %s", vp_id, exc)
            continue
        if not model.fitted or model.slope is None or model.slope <= 0:
            continue
        v_i = _slope_to_one_way_speed(model.slope)
        pegged = model.slope <= THEORETICAL_SLOPE + pegged_tolerance
        per_anchor.append({
            "vp_id": vp_id,
            "anchor_lat": model.anchor_lat,
            "anchor_lon": model.anchor_lon,
            "slope_ms_per_km": model.slope,
            "intercept_ms": model.intercept,
            "implied_one_way_speed_km_per_ms": v_i,
            "n_measurements": model.n_measurements,
            "pegged_at_baseline": pegged,
        })
    per_anchor.sort(key=lambda r: -r["implied_one_way_speed_km_per_ms"])

    if not per_anchor:
        raise RuntimeError("no anchors successfully fitted — check input data")

    fitted = [r for r in per_anchor if not r["pegged_at_baseline"]]
    pegged_anchors = [r for r in per_anchor if r["pegged_at_baseline"]]
    if not fitted:
        raise RuntimeError(
            "every anchor pegged at the LP baseline — data is too noisy"
            " for calibration. Inspect anchor GT."
        )

    speeds_fitted = np.array([r["implied_one_way_speed_km_per_ms"] for r in fitted])
    distribution = {
        "min": float(speeds_fitted.min()),
        "p5": float(np.percentile(speeds_fitted, 5)),
        "p25": float(np.percentile(speeds_fitted, 25)),
        "p50": float(np.percentile(speeds_fitted, 50)),
        "p75": float(np.percentile(speeds_fitted, 75)),
        "p95": float(np.percentile(speeds_fitted, 95)),
        "p99": float(np.percentile(speeds_fitted, 99)),
        "max": float(speeds_fitted.max()),
    }
    S = distribution["p99"]

    summary = {
        "S_one_way_km_per_ms": S,
        "S_as_fraction_of_c": S / SPEED_OF_LIGHT_KM_PER_MS,
        "S_statistic": "p99",
        "cho_reference_km_per_ms": CHO_2024_REFERENCE_KM_PER_MS,
        "cho_reference_fraction_c": CHO_2024_REFERENCE_KM_PER_MS / SPEED_OF_LIGHT_KM_PER_MS,
        "delta_vs_cho_pct": 100.0 * (S - CHO_2024_REFERENCE_KM_PER_MS) / CHO_2024_REFERENCE_KM_PER_MS,
        "lp_baseline_slope_ms_per_km": THEORETICAL_SLOPE,
        "lp_baseline_speed_cap_km_per_ms": SOI_SPEED_KM_PER_MS,
        "n_anchors_fitted": len(fitted),
        "n_anchors_pegged_at_baseline": len(pegged_anchors),
        "n_anchors_total": len(per_anchor),
        "n_samples_total": len(samples),
        "speed_distribution_km_per_ms_excluding_pegged": distribution,
        "fastest_anchors_excluding_pegged": fitted[:5],
        "pegged_anchors": pegged_anchors,
    }
    return {"summary": summary, "per_anchor": per_anchor, "samples": samples}


def write_results(result: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON + PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    per_anchor = result["per_anchor"]
    samples = result["samples"]

    json_path = output_dir / "speed_calibration.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "per_anchor": per_anchor}, f, indent=2)
    logger.info("wrote %s", json_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    by_vp: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in samples:
        d = haversine_distance(s.vp_coord.lat, s.vp_coord.lon,
                               s.probe_coord.lat, s.probe_coord.lon)
        by_vp[str(s.vp_id)].append((d, float(s.latency)))

    d_max = max((d for vp_pts in by_vp.values() for d, _ in vp_pts), default=20000.0)
    d_grid = np.linspace(0.0, d_max, 200)

    fig, ax = plt.subplots(figsize=(10, 7))
    fitted_records = [r for r in per_anchor if not r.get("pegged_at_baseline", False)]
    pegged_records = [r for r in per_anchor if r.get("pegged_at_baseline", False)]

    # Per-anchor fitted envelopes (faint)
    for r in fitted_records:
        slope = r["slope_ms_per_km"]
        intercept = r["intercept_ms"] or 0.0
        ax.plot(d_grid, slope * d_grid + intercept,
                color="gray", alpha=0.15, lw=0.5, zorder=1)

    # Per-anchor pegged envelopes (orange, dotted) — all on the SOI slope, varying intercepts
    for r in pegged_records:
        slope = r["slope_ms_per_km"]
        intercept = r["intercept_ms"] or 0.0
        ax.plot(d_grid, slope * d_grid + intercept,
                color="orange", alpha=0.5, lw=0.8, linestyle=":", zorder=2)

    # Reference lines
    cho_slope = 2.0 / CHO_2024_REFERENCE_KM_PER_MS
    ax.plot(d_grid, cho_slope * d_grid,
            color="blue", lw=1.5, linestyle="--", zorder=3,
            label=f"Cho 2024: 153 km/ms (0.51 c)")

    soi_slope = 2.0 / SOI_SPEED_KM_PER_MS  # = THEORETICAL_SLOPE
    ax.plot(d_grid, soi_slope * d_grid,
            color="green", lw=1.5, linestyle="--", zorder=3,
            label=f"SOI / 2/3·c: {SOI_SPEED_KM_PER_MS:.1f} km/ms (LP floor)")

    s_p99 = summary["S_one_way_km_per_ms"]
    p99_slope = 2.0 / s_p99
    ax.plot(d_grid, p99_slope * d_grid,
            color="red", lw=2, zorder=4,
            label=f"S (p99): v={s_p99:.1f} km/ms")

    if pegged_records:
        ax.plot([], [], color="orange", linestyle=":",
                label=f"pegged at LP floor ({len(pegged_records)} anchors, excluded)")

    # Subsample scatter for visual context
    rng = np.random.default_rng(0)
    flat_d, flat_r = [], []
    for vp_pts in by_vp.values():
        flat_d.extend(d for d, _ in vp_pts)
        flat_r.extend(r for _, r in vp_pts)
    if len(flat_d) > 5000:
        idx = rng.choice(len(flat_d), size=5000, replace=False)
        flat_d = [flat_d[i] for i in idx]
        flat_r = [flat_r[i] for i in idx]
    ax.scatter(flat_d, flat_r, s=2, color="black", alpha=0.15, zorder=0,
               label=f"anchor-mesh samples (subset of {len(samples):,})")

    ax.set_xlabel("Great-circle distance (km)")
    ax.set_ylabel("min RTT (ms)")
    ax.set_title("Per-anchor LP lower-envelope fits — anchor mesh, post-SOI filter")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / "speed_calibration.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", png_path)


def calibrate(output_dir: Path, sanitize: bool = True) -> dict[str, Any]:
    """Production entry: configure a RipeAtlasSource on the anchor mesh + run."""
    # NB: when ping_table = anchors_meshed_pings, both sides of each row are
    # anchors. `PROBES_TO_ANCHORS` here means `vp_id = src` (outgoing pings
    # per anchor); `ANCHORS_TO_PROBES` would mean `vp_id = dst` (incoming
    # pings per anchor). For per-anchor LP envelopes either is fine — the
    # mesh stores both directions — so we use the source's default.
    source = RipeAtlasSource(
        slice="all_anchors",
        setup=DataSource.PROBES_TO_ANCHORS,
        ping_table=default.ANCHORS_MESHED_PING_TABLE,
        sanitize=sanitize,
    )
    result = calibrate_from_source(source)
    result["summary"]["sanitize"] = sanitize
    result["summary"]["soi_removed_count"] = len(source._removed_ips)
    write_results(result, output_dir)
    return result["summary"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path,
                   default=Path("scripts/vp_selection/outputs"))
    p.add_argument("--no-sanitize", action="store_true",
                   help="Skip SOI filter (debugging only — will inflate S).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    summary = calibrate(output_dir=args.output_dir, sanitize=not args.no_sanitize)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
