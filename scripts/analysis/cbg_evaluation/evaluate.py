"""Core evaluation loop: per-setting data loading, model caching, and execution."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.helpers import haversine  # noqa: E402
from scripts.analysis.cbg_evaluation.combinations import PipelineSpec  # noqa: E402
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


@dataclass
class PreparedEvaluationData:
    """Loaded CBG inputs for one setting evaluation."""

    df_asn: Any
    anchor_coords: Dict[str, Tuple[float, float]]
    probe_targets: Dict[str, Dict[str, Any]]
    data_fingerprint: str


@dataclass
class DistanceModelState:
    """Fitted or cached distance-model state needed by one setting."""

    lp_models: Dict
    octant_models: Dict
    octant_delta: Optional[float]


@dataclass
class SettingEvaluation:
    """End-to-end artifacts from evaluating one pipeline setting."""

    spec: PipelineSpec
    results: List[ProbeResult]
    anchor_coords: Dict[str, Tuple[float, float]]
    probe_targets: Dict[str, Dict[str, Any]]
    lp_models: Dict
    octant_models: Dict
    octant_delta: Optional[float]
    data_fingerprint: str
    benchmark_ms: Dict[str, float]


@dataclass
class EvaluationRun:
    """Results and per-setting artifacts from an evaluation run."""

    all_results: Dict[str, List[ProbeResult]]
    artifacts_by_combo: Dict[str, SettingEvaluation]


class DistanceModelCache:
    """Cache RTT-distance models by data fingerprint and model parameters."""

    def __init__(
        self,
        fit_lp_fn: Optional[Callable[[Any], Dict]] = None,
        fit_octant_fn: Optional[Callable[..., Tuple[Dict, float]]] = None,
    ):
        self._fit_lp_fn = fit_lp_fn
        self._fit_octant_fn = fit_octant_fn
        self._cache: Dict[Tuple[str, str, Tuple[Tuple[str, Any], ...]], Any] = {}

    def get_for_spec(
        self,
        spec: PipelineSpec,
        df_asn: Any,
        data_fingerprint: str,
        benchmark_recorder: Optional[BenchmarkRecorder] = None,
        benchmark_ms: Optional[Dict[str, float]] = None,
    ) -> DistanceModelState:
        """Return the fitted model state required by this setting."""
        if spec.needs_lp_fit:
            models = self._get_lp_models(
                spec,
                df_asn,
                data_fingerprint,
                benchmark_recorder,
                benchmark_ms,
            )
            return DistanceModelState(models, {}, None)

        if spec.needs_octant_fit:
            models, delta = self._get_octant_models(
                spec,
                df_asn,
                data_fingerprint,
                benchmark_recorder,
                benchmark_ms,
            )
            return DistanceModelState({}, models, delta)

        meta = {
            "model_family": "none",
            "cache_key": None,
            "cache_hit": None,
        }
        with _measure_setting_phase(
            spec,
            "model_cache_lookup",
            benchmark_ms,
            benchmark_recorder,
            metadata=lambda: meta,
            track_tracemalloc=False,
        ):
            pass
        return DistanceModelState({}, {}, None)

    def _get_lp_models(
        self,
        spec: PipelineSpec,
        df_asn: Any,
        data_fingerprint: str,
        benchmark_recorder: Optional[BenchmarkRecorder],
        benchmark_ms: Optional[Dict[str, float]],
    ) -> Dict:
        params: Tuple[Tuple[str, Any], ...] = ()
        key = ("low_envelope", data_fingerprint, params)
        cache_key = _format_cache_key(key)
        meta = {
            "model_family": "low_envelope",
            "cache_key": cache_key,
            "cache_hit": key in self._cache,
        }
        with _measure_setting_phase(
            spec,
            "model_cache_lookup",
            benchmark_ms,
            benchmark_recorder,
            metadata=lambda: meta,
            track_tracemalloc=False,
        ):
            cached = self._cache.get(key)
        if cached is not None:
            _ensure_metric(benchmark_ms, "fit_lp_model_ms")
            return cached

        fit_meta = {
            "model_family": "low_envelope",
            "cache_key": cache_key,
            "cache_hit": False,
        }
        with _measure_setting_phase(
            spec,
            "fit_lp_model",
            benchmark_ms,
            benchmark_recorder,
            metadata=lambda: fit_meta,
        ):
            fitted = self._fit_lp(df_asn)
        self._cache[key] = fitted
        return fitted

    def _get_octant_models(
        self,
        spec: PipelineSpec,
        df_asn: Any,
        data_fingerprint: str,
        benchmark_recorder: Optional[BenchmarkRecorder],
        benchmark_ms: Optional[Dict[str, float]],
    ) -> Tuple[Dict, float]:
        params = (("target_coverage", 0.80),)
        key = ("bounded_spline", data_fingerprint, params)
        cache_key = _format_cache_key(key)
        meta = {
            "model_family": "bounded_spline",
            "cache_key": cache_key,
            "cache_hit": key in self._cache,
        }
        with _measure_setting_phase(
            spec,
            "model_cache_lookup",
            benchmark_ms,
            benchmark_recorder,
            metadata=lambda: meta,
            track_tracemalloc=False,
        ):
            cached = self._cache.get(key)
        if cached is not None:
            _ensure_metric(benchmark_ms, "fit_octant_model_ms")
            return cached

        fit_meta = {
            "model_family": "bounded_spline",
            "cache_key": cache_key,
            "cache_hit": False,
        }
        with _measure_setting_phase(
            spec,
            "fit_octant_model",
            benchmark_ms,
            benchmark_recorder,
            metadata=lambda: fit_meta,
        ):
            fitted = self._fit_octant(df_asn, target_coverage=0.80)
        self._cache[key] = fitted
        return fitted

    def _fit_lp(self, df_asn: Any) -> Dict:
        if self._fit_lp_fn is not None:
            return self._fit_lp_fn(df_asn)
        from scripts.analysis.million_scale.evaluate_million_scale import fit_lp_models

        return fit_lp_models(df_asn)

    def _fit_octant(self, df_asn: Any, target_coverage: float) -> Tuple[Dict, float]:
        if self._fit_octant_fn is not None:
            return self._fit_octant_fn(df_asn, target_coverage=target_coverage)
        from scripts.analysis.octant.octant_evaluation import fit_octant_models

        return fit_octant_models(df_asn, target_coverage=target_coverage)


def load_and_prepare() -> Dict[str, Any]:
    """Load data and fit all models once.

    This is retained for compatibility. The benchmark runner uses
    `run_setting()` so each pipeline setting owns its end-to-end path.

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
    lp_fit_start = time.perf_counter()
    lp_models = fit_lp_models(df_asn)
    setup_benchmark_ms["fit_lp_model_ms"] = (
        time.perf_counter() - lp_fit_start
    ) * 1000.0

    logger.info("=" * 60)
    logger.info("FITTING OCTANT MODELS")
    logger.info("=" * 60)
    octant_fit_start = time.perf_counter()
    octant_models, octant_delta = fit_octant_models(df_asn, target_coverage=0.80)
    setup_benchmark_ms["fit_octant_model_ms"] = (
        time.perf_counter() - octant_fit_start
    ) * 1000.0
    setup_benchmark_ms["fitting_model_ms"] = (
        setup_benchmark_ms["fit_lp_model_ms"]
        + setup_benchmark_ms["fit_octant_model_ms"]
    )

    anchor_coords, probe_targets = prepare_evaluation_inputs(df_asn)
    data_fingerprint = fingerprint_dataframe(df_asn)

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
        "data_fingerprint": data_fingerprint,
        "setup_benchmark_ms": setup_benchmark_ms,
    }


def prepare_evaluation_inputs(
    df_asn: Any,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Dict[str, Any]]]:
    """Build anchor and probe dictionaries from the loaded ASN dataframe."""
    anchors = df_asn[["dst_ip", "anchor_latitude", "anchor_longitude"]].drop_duplicates()
    anchor_coords: Dict[str, Tuple[float, float]] = {}
    for _, row in anchors.iterrows():
        anchor_coords[row["dst_ip"]] = (row["anchor_latitude"], row["anchor_longitude"])

    probe_targets: Dict[str, Dict[str, Any]] = {}
    for probe_ip, group in df_asn.groupby("src_ip"):
        measurements = dict(zip(group["dst_ip"], group["min_rtt"]))
        probe_targets[probe_ip] = {
            "measurements": measurements,
            "true_lat": float(group["probe_latitude"].iloc[0]),
            "true_lon": float(group["probe_longitude"].iloc[0]),
        }

    return anchor_coords, probe_targets


def fingerprint_dataframe(df_asn: Any) -> str:
    """Return a stable content fingerprint for model-cache keys."""
    from pandas.util import hash_pandas_object

    digest = hashlib.sha256()
    digest.update(str(tuple(df_asn.shape)).encode("utf-8"))
    digest.update(json.dumps(list(map(str, df_asn.columns))).encode("utf-8"))
    try:
        digest.update(hash_pandas_object(df_asn, index=True).values.tobytes())
    except TypeError:
        digest.update(df_asn.to_json(orient="split", date_format="iso").encode("utf-8"))
    return digest.hexdigest()


def load_setting_data(
    spec: PipelineSpec,
    benchmark_recorder: Optional[BenchmarkRecorder] = None,
    benchmark_ms: Optional[Dict[str, float]] = None,
) -> PreparedEvaluationData:
    """Load and prepare data for one pipeline setting."""
    from scripts.analysis.million_scale.evaluate_million_scale import load_data

    with _measure_setting_phase(
        spec,
        "load_data",
        benchmark_ms,
        benchmark_recorder,
    ):
        _, df_asn = load_data()

    with _measure_setting_phase(
        spec,
        "prepare_data",
        benchmark_ms,
        benchmark_recorder,
    ):
        anchor_coords, probe_targets = prepare_evaluation_inputs(df_asn)

    with _measure_setting_phase(
        spec,
        "data_fingerprint",
        benchmark_ms,
        benchmark_recorder,
        track_tracemalloc=False,
    ):
        data_fingerprint = fingerprint_dataframe(df_asn)

    return PreparedEvaluationData(
        df_asn=df_asn,
        anchor_coords=anchor_coords,
        probe_targets=probe_targets,
        data_fingerprint=data_fingerprint,
    )


@contextmanager
def _measure_setting_phase(
    spec: PipelineSpec,
    phase: str,
    benchmark_ms: Optional[Dict[str, float]],
    benchmark_recorder: Optional[BenchmarkRecorder],
    metadata: Optional[Callable[[], Dict[str, Any]]] = None,
    track_tracemalloc: bool = True,
) -> Iterator[None]:
    """Measure a setting-level phase and mirror elapsed ms into benchmark_ms."""
    manager = (
        benchmark_recorder.measure(
            spec.combo_id,
            "",
            phase,
            metadata=metadata,
            track_tracemalloc=track_tracemalloc,
        )
        if benchmark_recorder is not None
        else nullcontext()
    )
    start = time.perf_counter()
    with manager:
        yield
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if benchmark_ms is not None:
        benchmark_ms[f"{phase}_ms"] = float(elapsed_ms)


def _ensure_metric(benchmark_ms: Optional[Dict[str, float]], key: str) -> None:
    if benchmark_ms is not None and key not in benchmark_ms:
        benchmark_ms[key] = 0.0


def _format_cache_key(key: Tuple[str, str, Tuple[Tuple[str, Any], ...]]) -> str:
    family, data_fingerprint, params = key
    params_json = json.dumps(dict(params), sort_keys=True)
    return f"{family}:{data_fingerprint[:16]}:{params_json}"


def build_pipeline(
    spec: PipelineSpec,
    lp_models: Dict,
    octant_models: Dict,
    octant_delta: Optional[float],
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


def run_setting(
    spec: PipelineSpec,
    model_cache: Optional[DistanceModelCache] = None,
    benchmark_recorder: Optional[BenchmarkRecorder] = None,
) -> SettingEvaluation:
    """Run one pipeline setting end-to-end from data loading to probe results."""
    cache = model_cache or DistanceModelCache()
    benchmark_ms: Dict[str, float] = {
        "fit_lp_model_ms": 0.0,
        "fit_octant_model_ms": 0.0,
    }

    with _measure_setting_phase(
        spec,
        "setting_total",
        benchmark_ms,
        benchmark_recorder,
        track_tracemalloc=False,
    ):
        prepared = load_setting_data(
            spec,
            benchmark_recorder=benchmark_recorder,
            benchmark_ms=benchmark_ms,
        )
        model_state = cache.get_for_spec(
            spec,
            prepared.df_asn,
            prepared.data_fingerprint,
            benchmark_recorder=benchmark_recorder,
            benchmark_ms=benchmark_ms,
        )
        benchmark_ms["fitting_model_ms"] = (
            benchmark_ms.get("fit_lp_model_ms", 0.0)
            + benchmark_ms.get("fit_octant_model_ms", 0.0)
        )

        with _measure_setting_phase(
            spec,
            "pipeline_build",
            benchmark_ms,
            benchmark_recorder,
            track_tracemalloc=False,
        ):
            pipe = build_pipeline(
                spec,
                model_state.lp_models,
                model_state.octant_models,
                model_state.octant_delta,
            )

        results = evaluate_combination(
            spec,
            pipe,
            prepared.anchor_coords,
            prepared.probe_targets,
            benchmark_recorder=benchmark_recorder,
        )

    return SettingEvaluation(
        spec=spec,
        results=results,
        anchor_coords=prepared.anchor_coords,
        probe_targets=prepared.probe_targets,
        lp_models=model_state.lp_models,
        octant_models=model_state.octant_models,
        octant_delta=model_state.octant_delta,
        data_fingerprint=prepared.data_fingerprint,
        benchmark_ms=benchmark_ms,
    )


def evaluate_all(
    combinations: List[PipelineSpec],
    model_cache: Optional[DistanceModelCache] = None,
    benchmark_recorder: Optional[BenchmarkRecorder] = None,
) -> EvaluationRun:
    """Run all combinations through per-setting end-to-end evaluation."""
    cache = model_cache or DistanceModelCache()
    all_results: Dict[str, List[ProbeResult]] = {}
    artifacts_by_combo: Dict[str, SettingEvaluation] = {}

    for spec in combinations:
        logger.info("Running %s: %s ...", spec.combo_id, spec.label)
        artifact = run_setting(
            spec,
            model_cache=cache,
            benchmark_recorder=benchmark_recorder,
        )
        results = artifact.results
        elapsed = artifact.benchmark_ms.get("setting_total_ms", 0.0) / 1000.0

        success = [r for r in results if r.error_km is not None]
        errors = np.array([r.error_km for r in success])
        median = float(np.median(errors)) if len(errors) > 0 else float("nan")
        logger.info(
            "  %s: %d/%d probes, median=%.1f km, %.2fs",
            spec.combo_id, len(success), len(results), median, elapsed,
        )
        all_results[spec.combo_id] = results
        artifacts_by_combo[spec.combo_id] = artifact

    return EvaluationRun(all_results, artifacts_by_combo)


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
