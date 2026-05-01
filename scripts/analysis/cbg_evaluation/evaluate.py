"""Core evaluation loop: data loading, model fitting, pipeline execution.

Fits models once and shares them across all pipeline combinations.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.helpers import haversine  # noqa: E402
from scripts.analysis.cbg_evaluation.combinations import (  # noqa: E402
    COMBINATIONS,
    PipelineSpec,
)
from scripts.analysis.cbg_evaluation.benchmarking import (  # noqa: E402
    BenchmarkContext,
    BenchmarkRecorder,
    instrument_pipeline,
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
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


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

    setup_start = time.perf_counter()
    setup_benchmark_ms: Dict[str, float] = {}

    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    load_start = time.perf_counter()
    _, df_asn = load_data()
    setup_benchmark_ms["load_data_ms"] = (
        time.perf_counter() - load_start
    ) * 1000.0

    logger.info("=" * 60)
    logger.info("FITTING LP MODELS")
    logger.info("=" * 60)
    fitting_start = time.perf_counter()
    lp_models = fit_lp_models(df_asn)

    logger.info("=" * 60)
    logger.info("FITTING OCTANT MODELS")
    logger.info("=" * 60)
    octant_models, octant_delta = fit_octant_models(df_asn, target_coverage=0.80)
    setup_benchmark_ms["fitting_model_ms"] = (
        time.perf_counter() - fitting_start
    ) * 1000.0

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

    setup_benchmark_ms["total_setup_ms"] = (
        time.perf_counter() - setup_start
    ) * 1000.0

    return {
        "df_asn": df_asn,
        "lp_models": lp_models,
        "octant_models": octant_models,
        "octant_delta": octant_delta,
        "anchor_coords": anchor_coords,
        "probe_targets": probe_targets,
        "setup_benchmark_ms": setup_benchmark_ms,
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
        multilateration_kwargs=spec.multilateration_kwargs,
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
    benchmark_recorder: Optional[BenchmarkRecorder] = None,
) -> List[ProbeResult]:
    """Run one pipeline across all probes."""
    results = []
    benchmark_context = BenchmarkContext(spec.combo_id)
    instrumented = (
        instrument_pipeline(pipe, benchmark_recorder, benchmark_context)
        if benchmark_recorder is not None
        else nullcontext()
    )
    with instrumented:
        for probe_ip, target in probe_targets.items():
            benchmark_context.probe_ip = probe_ip
            total_meta: Dict[str, Any] = {}
            if benchmark_recorder is not None:
                with benchmark_recorder.measure(
                    spec.combo_id,
                    probe_ip,
                    "total_geolocate",
                    metadata=lambda: total_meta,
                    track_tracemalloc=False,
                ):
                    geo_result = pipe.geolocate_with_metadata(
                        target["measurements"], anchor_coords
                    )
                    total_meta.update(
                        success=geo_result.location is not None,
                        fallback_used=geo_result.fallback_used,
                        fallback_reason=geo_result.fallback_reason,
                    )
                benchmark_recorder.record_pipeline_overhead(spec.combo_id, probe_ip)
            else:
                geo_result = pipe.geolocate_with_metadata(
                    target["measurements"], anchor_coords
                )

            location = geo_result.location
            circles_used = geo_result.circles_used

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
                    did_intersect=geo_result.multilateration_success,
                    fallback_used=geo_result.fallback_used,
                    fallback_reason=geo_result.fallback_reason,
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
    benchmark_recorder: Optional[BenchmarkRecorder] = None,
) -> Dict[str, List[ProbeResult]]:
    """Run all combinations, return {combo_id: [ProbeResult]}."""
    all_results: Dict[str, List[ProbeResult]] = {}

    for spec in combinations:
        logger.info("Running %s: %s ...", spec.combo_id, spec.label)
        t0 = time.perf_counter()
        pipe = build_pipeline(spec, lp_models, octant_models, octant_delta)
        results = evaluate_combination(
            spec,
            pipe,
            anchor_coords,
            probe_targets,
            benchmark_recorder=benchmark_recorder,
        )
        elapsed = time.perf_counter() - t0

        success = [r for r in results if r.error_km is not None]
        errors = np.array([r.error_km for r in success])
        median = float(np.median(errors)) if len(errors) > 0 else float("nan")
        logger.info(
            "  %s: %d/%d probes, median=%.1f km, %.2fs",
            spec.combo_id, len(success), len(results), median, elapsed,
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

    logger.info(sep)
    logger.info("CBG COMBINATION EVALUATION — STATISTICS")
    logger.info(sep)
    logger.info(header)
    logger.info("-" * len(sep))

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
        logger.info(row)

    logger.info("")
    logger.info(f"{'Accuracy Thresholds':<22}" + "".join(f" {cid:>{col_w}}" for cid, _, _ in cols))
    logger.info("-" * len(sep))
    for thresh in [50, 100, 250, 500, 1000]:
        row = f"  Within {thresh:4d} km      "
        for _, _, e in cols:
            pct = np.mean(e <= thresh) * 100
            row += f" {pct:>{col_w - 1}.1f}%"
        logger.info(row)
    logger.info(sep)
