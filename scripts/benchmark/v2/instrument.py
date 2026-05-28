"""Per-stage timing + dual-channel memory profiler for CBGModel.geolocate.

Two complementary memory signals per stage:

  * `alloc_peak_bytes` — Python-allocator peak via `tracemalloc`. Catches
    NumPy/SciPy allocations (they route through `PyMem_RawMalloc`), but is
    BLIND to C-library allocations like Shapely/GEOS polygon intersections,
    which go through raw libc malloc. Use this for Python-side attribution.

  * `rss_peak_bytes` — max RSS increment over the stage window, sampled by a
    background thread polling `psutil.Process().memory_info().rss`. Catches
    everything (libc, mmap, JIT, ...) but at the cost of sampling resolution:
    stages shorter than the sample interval may report 0. Use this for the
    honest "how much memory did this stage actually need" number.

Run-level peak: `peak_rss_bytes()` returns `resource.getrusage().ru_maxrss`,
the kernel-tracked monotonic high-water mark for the calling process — a
TRUE peak, not the wobbly current-RSS snapshot psutil returns.

Implements the `StageInstrument` contract from `scripts.framework.v2.model`:
the runner creates one `TimingMemoryInstrument` per target, passes it as
`model.geolocate(obs, instrument=instr)`, and reads `instr.records` after.
"""

from __future__ import annotations

import resource
import sys
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import psutil


# Default polling interval for the background RSS sampler. 5 ms ≈ 200 Hz —
# fine enough for stages in the millisecond range, cheap enough to ignore
# under the "I don't care runtime" benchmarking regime.
DEFAULT_RSS_SAMPLE_INTERVAL_S = 0.005


@dataclass(frozen=True)
class StageRecord:
    """One stage-execution measurement.

    Both memory fields are in bytes. `alloc_peak_bytes` is tracemalloc's
    Python-allocator peak; `rss_peak_bytes` is the RSS-delta high-water mark
    observed by the background sampler over the stage's context window
    (clipped at 0).
    """
    stage: str         # "ltd" | "mtl" | "ctr" | "fit"
    duration_ns: int
    alloc_peak_bytes: int
    rss_peak_bytes: int


class RssSampler:
    """Background-thread RSS sampler.

    `start()` records a baseline RSS and spawns a daemon thread that polls
    `psutil.Process().memory_info().rss` every `interval_s` seconds, keeping
    the max observed. `stop()` joins the thread and returns
    `max(observed) - baseline` clipped to 0.

    Reusable: every `start()` resets baseline + max. Not thread-safe across
    concurrent callers, but one instance per stage (the use case) is fine.
    """

    def __init__(self, interval_s: float = DEFAULT_RSS_SAMPLE_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._proc = psutil.Process()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline = 0
        self._max = 0

    def start(self) -> None:
        self._baseline = self._proc.memory_info().rss
        self._max = self._baseline
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # `wait(interval)` returns early if stop() fires — that's the
        # responsive-shutdown trick (avoids sleeping past stop() by interval_s).
        while not self._stop_evt.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._max:
                self._max = rss
            self._stop_evt.wait(self._interval_s)

    def stop(self) -> int:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # One final read in case the largest peak happened between the last
        # sample and now (e.g. stage ended right after a fresh allocation).
        final = self._proc.memory_info().rss
        if final > self._max:
            self._max = final
        return max(0, self._max - self._baseline)


class TimingMemoryInstrument:
    """Per-stage timing + dual-channel memory collector.

    Reusable across stages within one geolocate call (records appended in
    order). Construct a fresh instance per target so peaks aren't conflated
    across calls. The optional `rss_sample_interval_s` lets tests pin a
    smaller interval; production callers should use the default.
    """

    def __init__(
        self,
        rss_sample_interval_s: float = DEFAULT_RSS_SAMPLE_INTERVAL_S,
    ) -> None:
        self.records: list[StageRecord] = []
        self._rss_sample_interval_s = rss_sample_interval_s

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        # tracemalloc supports nested start() (refcounted), so this is safe
        # even if framework wraps a nested stage one day.
        tracemalloc.start()
        tracemalloc.reset_peak()
        sampler = RssSampler(interval_s=self._rss_sample_interval_s)
        sampler.start()
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            dt_ns = time.perf_counter_ns() - t0
            rss_peak = sampler.stop()
            _current, alloc_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.records.append(StageRecord(
                stage=stage,
                duration_ns=dt_ns,
                alloc_peak_bytes=alloc_peak,
                rss_peak_bytes=rss_peak,
            ))

    def record_named(
        self,
        stage: str,
        duration_ns: int,
        alloc_peak_bytes: int,
        rss_peak_bytes: int,
    ) -> None:
        """Append a record from out-of-band measurement (e.g. the fit stage,
        which is timed via `measure_block` outside CBGModel.geolocate)."""
        self.records.append(StageRecord(
            stage=stage,
            duration_ns=duration_ns,
            alloc_peak_bytes=alloc_peak_bytes,
            rss_peak_bytes=rss_peak_bytes,
        ))

    def get(self, stage: str) -> Optional[StageRecord]:
        for r in self.records:
            if r.stage == stage:
                return r
        return None


@contextmanager
def measure_block(
    label: str = "block",
    *,
    rss_sample_interval_s: float = DEFAULT_RSS_SAMPLE_INTERVAL_S,
) -> Iterator["dict"]:
    """Standalone helper for one-shot measurements (used to wrap LTD.fit).

    Yields a dict that gets populated on context exit with `duration_ns`,
    `alloc_peak_bytes`, and `rss_peak_bytes`. The label is just for
    debugging; it's not stored in the dict.
    """
    out: dict = {}
    tracemalloc.start()
    tracemalloc.reset_peak()
    sampler = RssSampler(interval_s=rss_sample_interval_s)
    sampler.start()
    t0 = time.perf_counter_ns()
    try:
        yield out
    finally:
        out["duration_ns"] = time.perf_counter_ns() - t0
        out["rss_peak_bytes"] = sampler.stop()
        _current, alloc_peak = tracemalloc.get_traced_memory()
        out["alloc_peak_bytes"] = alloc_peak
        tracemalloc.stop()


def peak_rss_bytes() -> int:
    """Kernel-tracked lifetime max RSS for the calling process, in bytes.

    Uses `resource.getrusage(RUSAGE_SELF).ru_maxrss`. This is monotonic
    over the process lifetime and never lies — pages the kernel reclaims
    don't lower the recorded peak. The right call for "what was the
    largest the process ever got" questions.

    Unit conversion: Linux returns `ru_maxrss` in KIBIBYTES; macOS returns
    it in BYTES. We branch on `sys.platform` to normalize to bytes.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(maxrss)
    return int(maxrss) * 1024
