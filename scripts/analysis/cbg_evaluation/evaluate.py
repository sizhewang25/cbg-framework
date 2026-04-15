"""Core evaluation loop: data loading, model fitting, pipeline execution.

Fits models once and shares them across all pipeline combinations.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.helpers import haversine  # noqa: E402
from scripts.analysis.cbg_evaluation.combinations import (  # noqa: E402
    COMBINATIONS,
    PipelineSpec,
)


@dataclass
class ProbeResult:
    """Per-probe geolocation result from one pipeline combination."""

    probe_ip: str
    true_lat: float
    true_lon: float
    estimated_lat: Optional[float]
    estimated_lon: Optional[float]
    error_km: Optional[float]
    n_circles: int
    min_rtt_ms: float
    did_intersect: bool


def load_and_prepare() -> Dict[str, Any]:
    """Load data and fit all models once.

    Returns dict with keys:
        df_asn, lp_models, octant_models, octant_delta,
        anchor_coords, probe_targets
    """
    from scripts.analysis.million_scale.evaluate_million_scale import (
        load_data,
        fit_lp_models,
    )
    from scripts.analysis.octant.octant_evaluation import fit_octant_models

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    _, df_asn = load_data()

    print("\n" + "=" * 60)
    print("FITTING LP MODELS")
    print("=" * 60)
    lp_models = fit_lp_models(df_asn)

    print("\n" + "=" * 60)
    print("FITTING OCTANT MODELS")
    print("=" * 60)
    octant_models, octant_delta = fit_octant_models(df_asn, target_coverage=0.80)

    # Build anchor_coords
    anchors = df_asn[["dst_ip", "anchor_latitude", "anchor_longitude"]].drop_duplicates()
    anchor_coords: Dict[str, Tuple[float, float]] = {}
    for _, row in anchors.iterrows():
        anchor_coords[row["dst_ip"]] = (row["anchor_latitude"], row["anchor_longitude"])

    # Build per-probe measurement dicts
    probe_targets: Dict[str, Dict[str, Any]] = {}
    for probe_ip, group in df_asn.groupby("src_ip"):
        measurements = dict(zip(group["dst_ip"], group["min_rtt"]))
        probe_targets[probe_ip] = {
            "measurements": measurements,
            "true_lat": float(group["probe_latitude"].iloc[0]),
            "true_lon": float(group["probe_longitude"].iloc[0]),
        }

    return {
        "df_asn": df_asn,
        "lp_models": lp_models,
        "octant_models": octant_models,
        "octant_delta": octant_delta,
        "anchor_coords": anchor_coords,
        "probe_targets": probe_targets,
    }


def build_pipeline(
    spec: PipelineSpec,
    lp_models: Dict,
    octant_models: Dict,
    octant_delta: float,
):
    """Instantiate a CBGPipeline from spec, injecting pre-fitted models."""
    from scripts.framework import CBGPipeline

    pipe = CBGPipeline.from_config(
        distance=spec.distance,
        filtering=spec.filtering,
        multilateration=spec.multilateration,
        centroid=spec.centroid,
    )

    if spec.needs_lp_fit:
        pipe.distance.fit(models=lp_models)
    elif spec.needs_octant_fit:
        pipe.distance.fit(models=octant_models, delta=octant_delta)

    return pipe


def evaluate_combination(
    spec: PipelineSpec,
    pipe,
    anchor_coords: Dict[str, Tuple[float, float]],
    probe_targets: Dict[str, Dict[str, Any]],
) -> List[ProbeResult]:
    """Run one pipeline across all probes."""
    results = []
    for probe_ip, target in probe_targets.items():
        location, circles_used = pipe.geolocate(
            target["measurements"], anchor_coords
        )

        true = (target["true_lat"], target["true_lon"])
        if location is not None:
            error_km = float(haversine(location, true))
            est_lat, est_lon = float(location[0]), float(location[1])
        else:
            error_km = None
            est_lat = est_lon = None

        min_rtt = float(min(target["measurements"].values()))

        results.append(
            ProbeResult(
                probe_ip=probe_ip,
                true_lat=target["true_lat"],
                true_lon=target["true_lon"],
                estimated_lat=est_lat,
                estimated_lon=est_lon,
                error_km=error_km,
                n_circles=len(circles_used),
                min_rtt_ms=min_rtt,
                did_intersect=len(circles_used) > 0 and location is not None,
            )
        )
    return results


def evaluate_all(
    combinations: List[PipelineSpec],
    lp_models: Dict,
    octant_models: Dict,
    octant_delta: float,
    anchor_coords: Dict[str, Tuple[float, float]],
    probe_targets: Dict[str, Dict[str, Any]],
) -> Dict[str, List[ProbeResult]]:
    """Run all combinations, return {combo_id: [ProbeResult]}."""
    all_results: Dict[str, List[ProbeResult]] = {}

    for spec in combinations:
        print(f"\n  Running {spec.combo_id}: {spec.label} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        pipe = build_pipeline(spec, lp_models, octant_models, octant_delta)
        results = evaluate_combination(spec, pipe, anchor_coords, probe_targets)
        elapsed = time.perf_counter() - t0

        success = [r for r in results if r.error_km is not None]
        errors = np.array([r.error_km for r in success])
        median = float(np.median(errors)) if len(errors) > 0 else float("nan")
        print(
            f"{len(success)}/{len(results)} probes, "
            f"median={median:.1f} km, {elapsed:.2f}s"
        )
        all_results[spec.combo_id] = results

    return all_results


def get_errors(results: List[ProbeResult]) -> np.ndarray:
    """Extract error array from results (only successful probes)."""
    return np.array([r.error_km for r in results if r.error_km is not None])


def print_statistics(
    all_results: Dict[str, List[ProbeResult]],
    specs: List[PipelineSpec],
) -> None:
    """Print comparison statistics table."""
    cols = [(s.combo_id, s.label, get_errors(all_results[s.combo_id])) for s in specs]
    cols = [(cid, label, e) for cid, label, e in cols if len(e) > 0]

    col_w = 18
    header = f"{'Metric':<22}" + "".join(f" {cid:>{col_w}}" for cid, _, _ in cols)
    sep = "=" * (22 + (col_w + 1) * len(cols))

    print(f"\n{sep}")
    print("CBG COMBINATION EVALUATION — STATISTICS")
    print(sep)
    print(header)
    print("-" * len(sep))

    metrics = [
        ("N (probes)", lambda e: f"{len(e)}"),
        ("Median (km)", lambda e: f"{np.median(e):.1f}"),
        ("Mean (km)", lambda e: f"{np.mean(e):.1f}"),
        ("P25 (km)", lambda e: f"{np.percentile(e, 25):.1f}"),
        ("P75 (km)", lambda e: f"{np.percentile(e, 75):.1f}"),
        ("P90 (km)", lambda e: f"{np.percentile(e, 90):.1f}"),
    ]
    for label, fn in metrics:
        row = f"{label:<22}" + "".join(f" {fn(e):>{col_w}}" for _, _, e in cols)
        print(row)

    print()
    print(f"{'Accuracy Thresholds':<22}" + "".join(f" {cid:>{col_w}}" for cid, _, _ in cols))
    print("-" * len(sep))
    for thresh in [50, 100, 250, 500, 1000]:
        row = f"  Within {thresh:4d} km      "
        for _, _, e in cols:
            pct = np.mean(e <= thresh) * 100
            row += f" {pct:>{col_w - 1}.1f}%"
        print(row)
    print(sep)
