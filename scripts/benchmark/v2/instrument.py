"""Per-stage timing + peak-memory profiler for CBGModel.geolocate.

Implements the `StageInstrument` contract from `scripts.framework.v2.model`:
the runner creates one `TimingMemoryInstrument` per target, passes it as
`model.geolocate(obs, instrument=instr)`, and reads `instr.records` after the
call returns.

`tracemalloc` is started/stopped per stage so the recorded peak is bounded to
that stage's allocations (not the cumulative process heap). Note: for fast
stages (~10–100 µs) `tracemalloc.start/stop` adds measurable overhead — the
absolute per-call number is noisy. The runner records it anyway because the
spec asks for it; the trustworthy reading is the aggregated p50/p95/max across
the full target sweep (computed at `summarize` time).

For per-run process-level memory, use `peak_rss_bytes()` — psutil RSS snapshot.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import psutil


@dataclass(frozen=True)
class StageRecord:
    """One stage-execution measurement."""
    stage: str         # "ltd" | "mtl" | "ctr" | "fit"
    duration_ns: int
    peak_bytes: int    # peak Python allocation during the stage, per tracemalloc


class TimingMemoryInstrument:
    """Per-stage timing + tracemalloc-peak collector.

    Reusable across stages within one geolocate call (records appended in
    order). Construct a fresh instance per target so peaks aren't conflated
    across calls.
    """

    def __init__(self) -> None:
        self.records: list[StageRecord] = []

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        # Nested tracemalloc.start calls are supported (refcounted internally),
        # so this is safe even if the framework one day instruments a nested
        # stage. We snapshot the peak before this stage's stop() to attribute
        # allocations to the current stage only.
        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            dt_ns = time.perf_counter_ns() - t0
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.records.append(StageRecord(stage=stage, duration_ns=dt_ns, peak_bytes=peak))

    def record_named(self, stage: str, duration_ns: int, peak_bytes: int) -> None:
        """Append a record from out-of-band measurement (e.g. the fit stage,
        which is timed outside CBGModel.geolocate)."""
        self.records.append(StageRecord(stage=stage, duration_ns=duration_ns, peak_bytes=peak_bytes))

    def get(self, stage: str) -> Optional[StageRecord]:
        for r in self.records:
            if r.stage == stage:
                return r
        return None


@contextmanager
def measure_block(label: str = "block") -> Iterator["dict"]:
    """Standalone helper for one-shot measurements (used to wrap LTD.fit).

    Yields a dict that gets populated on context exit with `duration_ns` and
    `peak_bytes`. The label is just for debugging; it's not stored in the dict.
    """
    out: dict = {}
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter_ns()
    try:
        yield out
    finally:
        out["duration_ns"] = time.perf_counter_ns() - t0
        _current, peak = tracemalloc.get_traced_memory()
        out["peak_bytes"] = peak
        tracemalloc.stop()


def peak_rss_bytes() -> int:
    """Current process RSS in bytes. Useful as a coarse run-level peak.

    Matches the byte units used by `tracemalloc` peaks so the run.json /
    summary.parquet schema can stay uniform across stage-level and
    run-level memory measurements.
    """
    return int(psutil.Process().memory_info().rss)
