"""Per-stage timing + dual-channel memory profiler for CBGModel.geolocate.

Two complementary memory signals per stage. NEITHER DOMINATES THE OTHER —
they have disjoint blind spots, so both are recorded and they are never
combined (summing double-counts malloc-backed NumPy; taking the max discards
the pymalloc side):

  * `alloc_peak_bytes` — Python-allocator peak via `tracemalloc`. Sees Python
    objects and NumPy/SciPy buffers, but is BLIND to C-library allocations
    like Shapely/GEOS polygon intersections, which go through raw libc malloc.

  * `heap_peak_bytes` — sampled peak of libc heap-in-use, from
    `mallinfo2().uordblks + .hblkhd`. Sees GEOS/C and large NumPy buffers,
    but is BLIND to pymalloc: CPython satisfies small objects out of 1 MB
    obmalloc arenas taken via raw mmap, which `uordblks` never sees.

Measured on glibc/py3.12 (see notes/ and the test suite):

    workload                          alloc        heap
    pure-Python 1 MB list           1028 KB       6.9 KB
    numpy 32 MB np.ones             32.14 MB     32.16 MB
    Shapely/GEOS intersection        15.6 KB     1015 KB   (~64x)

WHY NOT RSS. This module previously sampled `psutil.Process().memory_info().rss`
and described it as the honest "how much did this stage need" number. It is not.
glibc's mmap threshold is DYNAMIC: once it observes frees of mmap'd blocks it
raises the threshold, after which allocations come from the reused heap and
never raise RSS. So a per-stage RSS delta actually measures "did this stage
raise the process high-water mark" — which after the first couple of targets is
structurally *no*. Measured: five identical 16 MB stages reported RSS deltas of
16.00, 15.91, 0.00, 0.00, 0.00 MB, while the heap channel reported 16.00 MB
every time. This is NOT a sampling-rate problem and cannot be fixed by a finer
interval. `RssSampler` is retained only as the non-glibc fallback.

Run-level peak: `peak_rss_bytes()` returns `resource.getrusage().ru_maxrss`,
the kernel-tracked monotonic high-water mark for the calling process — a TRUE
peak. That number is sound and is the one to quote for whole-run memory.

Implements the `StageInstrument` contract from `scripts.framework.v2.model`:
the runner creates one `TimingMemoryInstrument` per target, passes it as
`model.geolocate(obs, instrument=instr)`, and reads `instr.records` after.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import resource
import sys
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import psutil


# Polling interval for the background memory sampler.
#
# 1 ms, chosen from measurement. For allocations HELD for the stage duration
# the interval is irrelevant — 50 us / 500 us / 5 ms all return ~1000-1050 KB
# on the GEOS workload. But TRANSIENT allocations are a different story, and
# they occur in practice: the Monte-Carlo CTR briefly holds an extra 4 MiB
# batch during rejection sampling. Catch rate for a 2 ms spike inside a 250 ms
# stage, measured over 20 trials:
#
#     5.0 ms interval -> 14/20      <- bimodal, unusable
#     1.0 ms interval -> 20/20
#     0.2 ms interval -> 20/20
#
# At 5 ms this produced a visibly bimodal ctr_heap_peak_bytes on real runs
# (43 targets reading 21 MB, 34 reading 25 MB, for the same workload). 1 ms
# closes it with margin.
#
# Affordable because mallinfo2 is ~517 ns per read vs psutil RSS's ~16,667 ns
# (32x cheaper): at 1 ms that is a ~0.05% duty cycle. glibc arena-lock
# contention is the reason not to go lower still.
#
# The dominant instrument cost is the THREAD, not the sample: Thread.start() +
# join() is ~45 us, against stage p50s of 17-530 ms (~0.03%). A shared
# process-lifetime sampler thread with per-stage window resets would save that,
# but needs a lock and is not worth it at this ratio — deliberately rejected.
DEFAULT_SAMPLE_INTERVAL_S = 0.001

# Backwards-compatible alias; the old name is referenced by existing tests.
DEFAULT_RSS_SAMPLE_INTERVAL_S = DEFAULT_SAMPLE_INTERVAL_S


class _Mallinfo2(ctypes.Structure):
    """glibc `struct mallinfo2` — all ten fields are `size_t` (unlike the
    legacy 32-bit `struct mallinfo`, which silently truncates above 2 GB)."""
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


def _resolve_heap_probe() -> Optional[Callable[[], int]]:
    """Return a zero-arg callable giving libc heap bytes in use, or None.

    Detection is by SYMBOL PRESENCE, not `sys.platform`: musl exposes
    `mallinfo` but not `mallinfo2`, and macOS has neither. We deliberately do
    NOT fall back to the legacy 32-bit `mallinfo()` — a silently truncated
    reading above 2 GB is worse than no reading at all.

    The probe sums `uordblks` (main-arena bytes in use) and `hblkhd` (bytes in
    mmap'd blocks). Both are required and they do not double-count: glibc does
    not include `hblkhd` in `uordblks`, and a large NumPy buffer served by mmap
    shows up ONLY in `hblkhd` (measured: `uordblks` alone reported 0.00 MB for
    a 16 MB array on the iteration glibc chose to mmap).
    """
    try:
        lib = ctypes.CDLL(None)
        if not hasattr(lib, "mallinfo2"):
            return None
        fn = lib.mallinfo2
        fn.restype = _Mallinfo2
        fn.argtypes = []
        # Probe once — a wrong restype would blow up here rather than mid-run.
        probe = fn()
        _ = probe.uordblks + probe.hblkhd
    except (OSError, AttributeError, ValueError):
        return None

    def read_heap_bytes() -> int:
        m = fn()
        return int(m.uordblks) + int(m.hblkhd)

    return read_heap_bytes


#: Resolved once at import. None on non-glibc platforms.
_HEAP_PROBE: Optional[Callable[[], int]] = _resolve_heap_probe()


def heap_probe_available() -> bool:
    """True when the libc heap channel is usable on this platform."""
    return _HEAP_PROBE is not None


@dataclass(frozen=True)
class StageRecord:
    """One stage-execution measurement. Memory fields are in bytes.

    `alloc_peak_bytes` is tracemalloc's stage-local peak delta.
    `heap_peak_bytes` is the libc heap-in-use high-water mark observed by the
    background sampler over the stage's context window (clipped at 0), or None
    when the platform has no `mallinfo2` — NEVER 0 in that case, because a
    silent 0 is indistinguishable from a real measurement.
    `rss_peak_bytes` is the legacy psutil-RSS delta, populated only on the
    non-glibc fallback path (NULL otherwise); see the module docstring for
    why it is degenerate.
    """
    stage: str         # "ltd" | "mtl" | "ctr" | "fit"
    duration_ns: int
    alloc_peak_bytes: int
    rss_peak_bytes: Optional[int]
    heap_peak_bytes: Optional[int] = None


class _BaseSampler:
    """Background-thread high-water sampler over a caller-supplied reader.

    `start()` records a baseline and spawns a daemon thread polling `_read()`
    every `interval_s`, keeping the max. `stop()` joins and returns
    `max(observed) - baseline`, clipped to 0.

    Threading invariant: `self._max` is written ONLY by the sampler thread and
    read ONLY after `join()`, so no lock is needed. Do not add a concurrent
    reader without one.

    Reusable: every `start()` resets baseline + max. Not thread-safe across
    concurrent callers; one instance per stage (the use case) is fine.
    """

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline = 0
        self._max = 0

    def _read(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def start(self) -> None:
        if self._thread is not None:
            # Previously this silently leaked the running thread.
            raise RuntimeError("sampler already started; call stop() first")
        self._baseline = self._read()
        self._max = self._baseline
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # `wait(interval)` returns early if stop() fires — that's the
        # responsive-shutdown trick (avoids sleeping past stop() by interval_s).
        while not self._stop_evt.is_set():
            value = self._read()
            if value > self._max:
                self._max = value
            self._stop_evt.wait(self._interval_s)

    def stop(self) -> int:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # One final read in case the largest peak happened between the last
        # sample and now (e.g. stage ended right after a fresh allocation).
        final = self._read()
        if final > self._max:
            self._max = final
        return max(0, self._max - self._baseline)


class HeapSampler(_BaseSampler):
    """Samples libc heap bytes in use via `mallinfo2`.

    Immune to the heap-reuse artifact that makes RSS degenerate, because it
    tracks LOGICAL bytes in use rather than physical resident pages.

    Note `mallinfo2` takes each arena's mutex, so the sampler does contend
    with the measured thread's `malloc`. At ~517 ns per read on a 1 ms
    interval that is a ~0.05% duty cycle.
    """

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S) -> None:
        if _HEAP_PROBE is None:
            raise RuntimeError("mallinfo2 unavailable; use make_sampler()")
        super().__init__(interval_s=interval_s)
        self._probe = _HEAP_PROBE

    def _read(self) -> int:
        return self._probe()


class RssSampler(_BaseSampler):
    """Samples `psutil.Process().memory_info().rss`. LEGACY / FALLBACK ONLY.

    Retained for non-glibc platforms. Per the module docstring, per-stage RSS
    deltas collapse to zero after warmup because of glibc heap reuse, so this
    channel cannot rank stages. Prefer `HeapSampler`.
    """

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S) -> None:
        super().__init__(interval_s=interval_s)
        self._proc = psutil.Process()

    def _read(self) -> int:
        return self._proc.memory_info().rss


def make_sampler(
    interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
) -> tuple[_BaseSampler, str]:
    """Return `(sampler, channel_name)` — the heap channel where available.

    This is the single fallback seam: `"heap"` on glibc, `"rss"` elsewhere.
    Callers record the channel name so a run is self-describing about which
    sampler produced its numbers (a macOS-collected run is otherwise silently
    incomparable with a Linux one).
    """
    if _HEAP_PROBE is not None:
        return HeapSampler(interval_s=interval_s), "heap"
    return RssSampler(interval_s=interval_s), "rss"


@contextmanager
def _measure(stage_interval_s: float) -> Iterator[dict]:
    """Shared measurement body for `TimingMemoryInstrument` and `measure_block`.

    Yields a dict populated on exit with `duration_ns`, `alloc_peak_bytes`,
    `heap_peak_bytes`, `rss_peak_bytes`, `memory_channel`.

    On tracemalloc: `start()` is NOT refcounted in CPython — it does not stack,
    and a single `stop()` tears tracing down completely, after which an outer
    session's `get_traced_memory()` returns (0, 0). So we only start/stop a
    session we ourselves own.

    That guard forces the delta fix to land with it. The old code read the
    ABSOLUTE peak, which was delta-equivalent only because a fresh `start()`
    begins with `current ~= 0`. Under an outer session `reset_peak()` sets
    `peak := current`, which includes the caller's live set — a 1 MB stage
    under a 4 MB live set would report 5 MB. Subtracting `entry_current` makes
    the number stage-local in both cases.
    """
    out: dict = {}
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    entry_current, _entry_peak = tracemalloc.get_traced_memory()

    sampler, channel = make_sampler(interval_s=stage_interval_s)
    sampler.start()
    t0 = time.perf_counter_ns()
    try:
        yield out
    finally:
        out["duration_ns"] = time.perf_counter_ns() - t0
        sampled = sampler.stop()
        _current, peak = tracemalloc.get_traced_memory()
        if owns_tracing:
            tracemalloc.stop()
        # Defensive: reset_peak() makes peak >= entry_current by construction.
        out["alloc_peak_bytes"] = max(0, peak - entry_current)
        out["memory_channel"] = channel
        # Exactly one sampler runs per stage, so the inactive channel is
        # NULL — never 0. A 0 would be indistinguishable from "this stage
        # used no memory", which is the class of bug this module exists to
        # remove. Historical runs keep their recorded RSS values; new runs
        # on glibc simply declare that channel unmeasured.
        if channel == "heap":
            out["heap_peak_bytes"] = sampled
            out["rss_peak_bytes"] = None
        else:
            out["heap_peak_bytes"] = None
            out["rss_peak_bytes"] = sampled


class TimingMemoryInstrument:
    """Per-stage timing + dual-channel memory collector.

    Reusable across stages within one geolocate call (records appended in
    order). Construct a fresh instance per target so peaks aren't conflated
    across calls. The optional `sample_interval_s` lets tests pin a smaller
    interval; production callers should use the default.
    """

    def __init__(
        self,
        sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        *,
        rss_sample_interval_s: Optional[float] = None,
    ) -> None:
        self.records: list[StageRecord] = []
        # Legacy keyword retained so existing callers/tests keep working.
        if rss_sample_interval_s is not None:
            sample_interval_s = rss_sample_interval_s
        self._sample_interval_s = sample_interval_s
        self.memory_channel: str = "heap" if heap_probe_available() else "rss"

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        with _measure(self._sample_interval_s) as m:
            yield
        self.memory_channel = m["memory_channel"]
        self.records.append(StageRecord(
            stage=stage,
            duration_ns=m["duration_ns"],
            alloc_peak_bytes=m["alloc_peak_bytes"],
            rss_peak_bytes=m["rss_peak_bytes"],
            heap_peak_bytes=m["heap_peak_bytes"],
        ))

    def record_named(
        self,
        stage: str,
        duration_ns: int,
        alloc_peak_bytes: int,
        rss_peak_bytes: int,
        heap_peak_bytes: Optional[int] = None,
    ) -> None:
        """Append a record from out-of-band measurement (e.g. the fit stage,
        which is timed via `measure_block` outside CBGModel.geolocate)."""
        self.records.append(StageRecord(
            stage=stage,
            duration_ns=duration_ns,
            alloc_peak_bytes=alloc_peak_bytes,
            rss_peak_bytes=rss_peak_bytes,
            heap_peak_bytes=heap_peak_bytes,
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
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    rss_sample_interval_s: Optional[float] = None,
) -> Iterator["dict"]:
    """Standalone helper for one-shot measurements (used to wrap LTD.fit).

    Yields a dict populated on context exit with `duration_ns`,
    `alloc_peak_bytes`, `heap_peak_bytes`, `rss_peak_bytes` and
    `memory_channel`. The label is just for debugging; it's not stored.
    """
    if rss_sample_interval_s is not None:
        sample_interval_s = rss_sample_interval_s
    out: dict = {}
    with _measure(sample_interval_s) as m:
        yield out
    out.update(m)


def peak_rss_bytes() -> int:
    """Kernel-tracked lifetime max RSS for the calling process, in bytes.

    Uses `resource.getrusage(RUSAGE_SELF).ru_maxrss`. This is monotonic
    over the process lifetime and never lies — pages the kernel reclaims
    don't lower the recorded peak. The right call for "what was the
    largest the process ever got" questions, and the number to quote for
    whole-run memory (it matched cgroup v2 `memory.peak` within 4% in a
    controlled comparison).

    Unit conversion: Linux returns `ru_maxrss` in KIBIBYTES; macOS returns
    it in BYTES. We branch on `sys.platform` to normalize to bytes.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(maxrss)
    return int(maxrss) * 1024
