"""Phase-level time and memory benchmarking for CBG evaluation."""

from __future__ import annotations

import csv
import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import psutil


COMPONENT_PHASES = (
    "distance_estimation",
    "filtering",
    "multilateration",
    "centroid",
)


@dataclass
class BenchmarkContext:
    """Mutable context shared by per-combo component wrappers."""

    combo_id: str
    probe_ip: str = ""


@dataclass
class BenchmarkRecord:
    """One measured phase for one combo/probe."""

    combo_id: str
    probe_ip: str
    phase: str
    elapsed_ms: float
    tracemalloc_before_bytes: Optional[int] = None
    tracemalloc_current_bytes: Optional[int] = None
    tracemalloc_peak_bytes: Optional[int] = None
    tracemalloc_peak_delta_bytes: Optional[int] = None
    rss_before_bytes: Optional[int] = None
    rss_after_bytes: Optional[int] = None
    rss_delta_bytes: Optional[int] = None
    rss_high_water_before_bytes: Optional[int] = None
    rss_high_water_after_bytes: Optional[int] = None
    rss_high_water_delta_bytes: Optional[int] = None
    success: Optional[bool] = None
    fallback_used: Optional[bool] = None
    fallback_reason: Optional[str] = None
    n_input_constraints: Optional[int] = None
    n_filtered_constraints: Optional[int] = None
    region_type: Optional[str] = None
    model_family: Optional[str] = None
    cache_key: Optional[str] = None
    cache_hit: Optional[bool] = None
    dataset_id: Optional[str] = None
    input_csv: Optional[str] = None
    preselected: Optional[bool] = None
    n_rows: Optional[int] = None
    n_probes: Optional[int] = None
    n_anchors: Optional[int] = None
    selected_asns: Optional[str] = None

    def as_row(self) -> Dict[str, Any]:
        """Return a CSV-friendly row with stable column names."""
        return asdict(self)


class BenchmarkRecorder:
    """Collects per-phase benchmark rows and aggregate summaries."""

    def __init__(self):
        self.records: List[BenchmarkRecord] = []
        self._process = psutil.Process()
        self._max_rss_seen_bytes = self._rss_bytes()
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    @contextmanager
    def measure(
        self,
        combo_id: str,
        probe_ip: str,
        phase: str,
        metadata: Optional[Callable[[], Dict[str, Any]]] = None,
        track_tracemalloc: bool = True,
    ) -> Iterator[None]:
        """Measure one phase in milliseconds with Python and RSS memory stats."""
        rss_before = self._rss_bytes()
        rss_high_water_before = self._max_rss_seen_bytes
        if track_tracemalloc:
            tracemalloc_before, _ = tracemalloc.get_traced_memory()
            tracemalloc_before = int(tracemalloc_before)
            tracemalloc.reset_peak()
        else:
            tracemalloc_before = None
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if track_tracemalloc:
                current, peak = tracemalloc.get_traced_memory()
                current = int(current)
                peak = int(peak)
                peak_delta = max(0, peak - int(tracemalloc_before or 0))
            else:
                current = None
                peak = None
                peak_delta = None
            rss_after = self._rss_bytes()
            rss_high_water_after = max(rss_high_water_before, rss_after)
            rss_high_water_delta = max(
                0,
                rss_high_water_after - rss_high_water_before,
            )
            self._max_rss_seen_bytes = max(self._max_rss_seen_bytes, rss_after)
            extra = metadata() if metadata is not None else {}
            self.records.append(
                BenchmarkRecord(
                    combo_id=combo_id,
                    probe_ip=probe_ip,
                    phase=phase,
                    elapsed_ms=float(elapsed_ms),
                    tracemalloc_before_bytes=tracemalloc_before,
                    tracemalloc_current_bytes=current,
                    tracemalloc_peak_bytes=peak,
                    tracemalloc_peak_delta_bytes=peak_delta,
                    rss_before_bytes=int(rss_before),
                    rss_after_bytes=int(rss_after),
                    rss_delta_bytes=int(rss_after - rss_before),
                    rss_high_water_before_bytes=int(rss_high_water_before),
                    rss_high_water_after_bytes=int(rss_high_water_after),
                    rss_high_water_delta_bytes=int(rss_high_water_delta),
                    **extra,
                )
            )

    def record_pipeline_overhead(self, combo_id: str, probe_ip: str) -> None:
        """Record non-component time as total minus measured component phases."""
        probe_records = [
            r for r in self.records
            if r.combo_id == combo_id and r.probe_ip == probe_ip
        ]
        total = next(
            (r for r in reversed(probe_records) if r.phase == "total_geolocate"),
            None,
        )
        if total is None:
            return
        component_ms = sum(
            r.elapsed_ms for r in probe_records if r.phase in COMPONENT_PHASES
        )
        self.records.append(
            BenchmarkRecord(
                combo_id=combo_id,
                probe_ip=probe_ip,
                phase="pipeline_overhead",
                elapsed_ms=max(0.0, float(total.elapsed_ms - component_ms)),
                success=total.success,
                fallback_used=total.fallback_used,
                fallback_reason=total.fallback_reason,
            )
        )

    def write_raw_csv(self, output_path: Path) -> None:
        """Write raw per-phase rows."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(f".{output_path.name}.tmp")
        fieldnames = list(BenchmarkRecord.__dataclass_fields__.keys())
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.as_row())
        tmp_path.replace(output_path)

    def write_summary_json(self, output_path: Path) -> None:
        """Write aggregate phase summary grouped by combo and phase."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(f".{output_path.name}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(summarize_records(self.records), f, indent=2)
        tmp_path.replace(output_path)

    def _rss_bytes(self) -> int:
        return int(self._process.memory_info().rss)


def summarize_records(records: List[BenchmarkRecord]) -> Dict[str, Any]:
    """Aggregate benchmark records by combo and phase."""
    grouped: Dict[str, Dict[str, List[BenchmarkRecord]]] = {}
    for record in records:
        grouped.setdefault(record.combo_id, {}).setdefault(record.phase, []).append(record)

    combinations: Dict[str, Any] = {}
    for combo_id, phases in grouped.items():
        combinations[combo_id] = {"phases": {}}
        for phase, phase_records in phases.items():
            elapsed = np.array([r.elapsed_ms for r in phase_records], dtype=float)
            combinations[combo_id]["phases"][phase] = {
                "count": int(len(phase_records)),
                "total_ms": round(float(np.sum(elapsed)), 3),
                "mean_ms": round(float(np.mean(elapsed)), 3),
                "median_ms": round(float(np.median(elapsed)), 3),
                "p90_ms": round(float(np.percentile(elapsed, 90)), 3),
                "p95_ms": round(float(np.percentile(elapsed, 95)), 3),
                "max_ms": round(float(np.max(elapsed)), 3),
                **_memory_summary(phase_records),
            }

    return {
        "time_unit": "ms",
        "memory_unit": "bytes",
        "combinations": combinations,
    }


def _memory_summary(records: List[BenchmarkRecord]) -> Dict[str, Optional[float]]:
    peak = _values(records, "tracemalloc_peak_bytes")
    peak_delta = _values(records, "tracemalloc_peak_delta_bytes")
    rss_delta = _values(records, "rss_delta_bytes")
    rss_after = _values(records, "rss_after_bytes")
    rss_high_water_delta = _values(records, "rss_high_water_delta_bytes")
    return {
        "mean_tracemalloc_peak_mb": _mean_mb(peak),
        "max_tracemalloc_peak_mb": _max_mb(peak),
        "mean_tracemalloc_phase_peak_delta_mb": _mean_mb(peak_delta),
        "max_tracemalloc_phase_peak_delta_mb": _max_mb(peak_delta),
        "mean_rss_delta_mb": _mean_mb(rss_delta),
        "max_rss_after_mb": _max_mb(rss_after),
        "mean_rss_high_water_delta_mb": _mean_mb(rss_high_water_delta),
        "max_rss_high_water_delta_mb": _max_mb(rss_high_water_delta),
    }


def _values(records: List[BenchmarkRecord], field: str) -> List[float]:
    values = [getattr(r, field) for r in records]
    return [float(v) for v in values if v is not None]


def _mean_mb(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(np.mean(values)) / 1_000_000.0, 3)


def _max_mb(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(np.max(values)) / 1_000_000.0, 3)


@contextmanager
def instrument_pipeline(pipe, recorder: BenchmarkRecorder, context: BenchmarkContext):
    """Temporarily wrap pipeline components for phase-level benchmarking."""
    originals = {
        "distance_estimate": pipe.distance.estimate,
        "filtering_filter": pipe.filtering.filter,
        "multilateration_multilaterate": pipe.multilateration.multilaterate,
        "centroid_select": pipe.centroid.select,
    }

    def distance_estimate(measurements, anchor_coords):
        meta: Dict[str, Any] = {}
        with recorder.measure(
            context.combo_id,
            context.probe_ip,
            "distance_estimation",
            metadata=lambda: meta,
        ):
            circles = originals["distance_estimate"](measurements, anchor_coords)
            meta.update(
                success=bool(circles),
                n_input_constraints=len(measurements),
                n_filtered_constraints=len(circles),
            )
            return circles

    def filtering_filter(circles):
        meta: Dict[str, Any] = {}
        with recorder.measure(
            context.combo_id,
            context.probe_ip,
            "filtering",
            metadata=lambda: meta,
        ):
            filtered = originals["filtering_filter"](circles)
            meta.update(
                success=bool(filtered),
                n_input_constraints=len(circles),
                n_filtered_constraints=len(filtered),
            )
            return filtered

    def multilateration_multilaterate(circles):
        meta: Dict[str, Any] = {}
        with recorder.measure(
            context.combo_id,
            context.probe_ip,
            "multilateration",
            metadata=lambda: meta,
        ):
            result = originals["multilateration_multilaterate"](circles)
            meta.update(
                success=result.success,
                n_input_constraints=len(circles),
                n_filtered_constraints=len(result.circles_used or circles),
                region_type=_region_type(result),
            )
            return result

    def centroid_select(multilat_result):
        meta: Dict[str, Any] = {}
        with recorder.measure(
            context.combo_id,
            context.probe_ip,
            "centroid",
            metadata=lambda: meta,
        ):
            location = originals["centroid_select"](multilat_result)
            meta.update(
                success=location is not None,
                region_type=_region_type(multilat_result),
            )
            return location

    pipe.distance.estimate = distance_estimate
    pipe.filtering.filter = filtering_filter
    pipe.multilateration.multilaterate = multilateration_multilaterate
    pipe.centroid.select = centroid_select
    try:
        yield
    finally:
        pipe.distance.estimate = originals["distance_estimate"]
        pipe.filtering.filter = originals["filtering_filter"]
        pipe.multilateration.multilaterate = originals["multilateration_multilaterate"]
        pipe.centroid.select = originals["centroid_select"]


def _region_type(multilat_result) -> Optional[str]:
    if getattr(multilat_result, "region", None) is not None:
        return getattr(multilat_result.region, "geom_type", type(multilat_result.region).__name__)
    if getattr(multilat_result, "vertices", None) is not None:
        return "vertices"
    return None
