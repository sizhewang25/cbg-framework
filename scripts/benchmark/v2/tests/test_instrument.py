"""TimingMemoryInstrument records per-stage timing plus two memory channels:
tracemalloc (`alloc_peak_bytes`) and sampled libc heap-in-use
(`heap_peak_bytes`). peak_rss_bytes() returns the kernel-tracked lifetime
maxrss. `rss_peak_bytes` is the legacy fallback channel, written only when
mallinfo2 is unavailable.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
import unittest

from scripts.benchmark.v2 import instrument as instrument_mod
from scripts.benchmark.v2.instrument import (
    HeapSampler,
    RssSampler,
    TimingMemoryInstrument,
    heap_probe_available,
    make_sampler,
    measure_block,
    peak_rss_bytes,
)

requires_heap = unittest.skipUnless(
    heap_probe_available(), "mallinfo2 unavailable on this platform"
)


def _sampled_peak(rec):
    """The active channel's value, whichever sampler this platform used."""
    return rec.heap_peak_bytes if rec.heap_peak_bytes is not None else rec.rss_peak_bytes


class TestTimingMemoryInstrument(unittest.TestCase):
    def test_records_one_entry_per_stage_in_order(self) -> None:
        instr = TimingMemoryInstrument()
        with instr("ltd"):
            time.sleep(0.001)
        with instr("mtl"):
            time.sleep(0.001)
        self.assertEqual([r.stage for r in instr.records], ["ltd", "mtl"])
        self.assertTrue(all(r.duration_ns > 0 for r in instr.records))

    def test_alloc_peak_bytes_reflects_allocation_inside_stage(self) -> None:
        instr = TimingMemoryInstrument()
        with instr("ltd"):
            # Allocate ~1 MB inside the stage; tracemalloc should see it.
            big = bytearray(1024 * 1024)
            big[0] = 1  # ensure it's actually used
        rec = instr.get("ltd")
        self.assertIsNotNone(rec)
        self.assertGreater(rec.alloc_peak_bytes, 500_000)

    def test_sampled_peak_reflects_large_alloc_inside_stage(self) -> None:
        instr = TimingMemoryInstrument(sample_interval_s=0.001)
        with instr("mtl"):
            big = bytearray(50 * 1024 * 1024)  # 50 MB
            for i in range(0, len(big), 4096):  # touch pages to force commit
                big[i] = 1
            time.sleep(0.02)  # 20 ms — gives the sampler ~20 chances
        rec = instr.get("mtl")
        self.assertIsNotNone(rec)
        self.assertGreater(_sampled_peak(rec), 10_000_000)

    def test_sampled_peak_small_for_trivial_stage(self) -> None:
        instr = TimingMemoryInstrument(sample_interval_s=0.001)
        with instr("ltd"):
            pass
        rec = instr.get("ltd")
        self.assertIsNotNone(rec)
        # No allocations → near-zero delta. Allow a small slack for OS noise.
        self.assertLess(_sampled_peak(rec), 1_000_000)

    @requires_heap
    def test_heap_peak_is_stable_across_repetitions(self) -> None:
        """The regression this whole channel exists for.

        Five identical 16 MB stages. The legacy RSS sampler reports
        16.00, 15.91, 0.00, 0.00, 0.00 MB — glibc's dynamic mmap threshold
        rises after it sees frees of mmap'd blocks, after which the heap is
        reused and RSS never grows. The heap channel tracks LOGICAL bytes in
        use, so it must report the full size every single time.
        """
        seen = []
        for _ in range(5):
            instr = TimingMemoryInstrument(sample_interval_s=0.001)
            with instr("mtl"):
                buf = bytearray(16 * 1024 * 1024)
                buf[0] = 1
                time.sleep(0.005)
            seen.append(instr.get("mtl").heap_peak_bytes)
            del buf
            gc.collect()
        for i, value in enumerate(seen):
            self.assertGreater(
                value, 15_000_000,
                f"iteration {i} reported {value} B — the channel decayed, "
                f"which is exactly the RSS bug being fixed. All: {seen}",
            )

    @requires_heap
    def test_heap_peak_catches_c_allocation_tracemalloc_misses(self) -> None:
        """Shapely/GEOS allocates through raw libc malloc, which tracemalloc
        cannot see. This is the entire justification for a second channel."""
        try:
            import numpy as np
            from shapely.geometry import Point
        except ImportError:  # pragma: no cover
            self.skipTest("shapely/numpy unavailable")

        def geos_work():
            disks = [
                Point(np.random.uniform(-10, 10), np.random.uniform(-10, 10))
                .buffer(6.0, quad_segs=64)
                for _ in range(160)
            ]
            region = disks[0]
            for d in disks[1:60]:
                if region.intersects(d):
                    region = region.intersection(d)
            return region

        geos_work()  # warm up GEOS + its arenas
        instr = TimingMemoryInstrument(sample_interval_s=0.001)
        with instr("mtl"):
            geos_work()
        rec = instr.get("mtl")
        self.assertGreater(rec.heap_peak_bytes, 200_000)
        self.assertLess(rec.alloc_peak_bytes, 100_000)

    @requires_heap
    def test_heap_peak_agrees_with_tracemalloc_on_numpy(self) -> None:
        """Cross-validation: on a NumPy-dominated stage the two independent
        channels must agree, or the mallinfo2 reader is wrong."""
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("numpy unavailable")
        instr = TimingMemoryInstrument(sample_interval_s=0.001)
        with instr("ctr"):
            arr = np.ones(4_000_000, dtype=np.float64)  # 32 MB
            arr[0] = 1.0
            time.sleep(0.005)
        rec = instr.get("ctr")
        ratio = rec.heap_peak_bytes / rec.alloc_peak_bytes
        self.assertGreater(ratio, 0.9, f"channels disagree: {rec}")
        self.assertLess(ratio, 1.1, f"channels disagree: {rec}")

    def test_falls_back_to_rss_when_probe_missing(self) -> None:
        """On a platform without mallinfo2 the heap column must be NULL, never
        0 — a silent 0 is indistinguishable from a real measurement."""
        original = instrument_mod._HEAP_PROBE
        instrument_mod._HEAP_PROBE = None
        try:
            sampler, channel = make_sampler(interval_s=0.001)
            self.assertIsInstance(sampler, RssSampler)
            self.assertEqual(channel, "rss")
            instr = TimingMemoryInstrument(sample_interval_s=0.001)
            with instr("ltd"):
                pass
            rec = instr.get("ltd")
            self.assertIsNone(rec.heap_peak_bytes)
            self.assertEqual(instr.memory_channel, "rss")
        finally:
            instrument_mod._HEAP_PROBE = original

    def test_outer_tracemalloc_session_survives_a_stage(self) -> None:
        """CPython's tracemalloc.start() is NOT refcounted: it does not stack,
        and one stop() tears tracing down entirely. The instrument must only
        stop a session it started."""
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            outer = bytearray(1024 * 1024)
            outer[0] = 1
            instr = TimingMemoryInstrument(sample_interval_s=0.001)
            with instr("ltd"):
                inner = bytearray(256 * 1024)
                inner[0] = 1
            self.assertTrue(
                tracemalloc.is_tracing(),
                "the stage killed the caller's tracemalloc session",
            )
            _current, peak = tracemalloc.get_traced_memory()
            self.assertGreater(peak, 900_000, "outer peak was destroyed")
        finally:
            tracemalloc.stop()

    def test_stage_alloc_peak_is_stage_local_under_an_outer_session(self) -> None:
        """With an outer session running, reset_peak() sets peak := current,
        which includes the caller's live set. Subtracting the entry reading is
        what keeps the number stage-local."""
        tracemalloc.start()
        try:
            live = bytearray(4 * 1024 * 1024)  # 4 MB held across the stage
            live[0] = 1
            instr = TimingMemoryInstrument(sample_interval_s=0.001)
            with instr("ltd"):
                inner = bytearray(1024 * 1024)  # 1 MB allocated inside
                inner[0] = 1
            rec = instr.get("ltd")
            self.assertGreater(rec.alloc_peak_bytes, 900_000)
            self.assertLess(
                rec.alloc_peak_bytes, 2_000_000,
                "stage peak absorbed the caller's 4 MB live set",
            )
        finally:
            tracemalloc.stop()

    def test_get_returns_none_for_missing_stage(self) -> None:
        instr = TimingMemoryInstrument()
        with instr("ltd"):
            pass
        self.assertIsNone(instr.get("mtl"))

    def test_record_named_appends(self) -> None:
        instr = TimingMemoryInstrument()
        instr.record_named(
            "fit", duration_ns=12345, alloc_peak_bytes=678, rss_peak_bytes=910
        )
        rec = instr.get("fit")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.duration_ns, 12345)
        self.assertEqual(rec.alloc_peak_bytes, 678)
        self.assertEqual(rec.rss_peak_bytes, 910)


class TestMeasureBlock(unittest.TestCase):
    def test_block_populates_duration_alloc_and_rss(self) -> None:
        with measure_block("fit", rss_sample_interval_s=0.001) as out:
            big = bytearray(512 * 1024)
            big[0] = 1
        self.assertIn("duration_ns", out)
        self.assertIn("alloc_peak_bytes", out)
        self.assertIn("rss_peak_bytes", out)
        self.assertGreater(out["duration_ns"], 0)
        self.assertGreater(out["alloc_peak_bytes"], 200_000)
        # Exactly one sampler runs; the inactive channel is NULL, never 0.
        active = out["heap_peak_bytes"] if heap_probe_available() else out["rss_peak_bytes"]
        inactive = out["rss_peak_bytes"] if heap_probe_available() else out["heap_peak_bytes"]
        self.assertGreaterEqual(active, 0)
        self.assertIsNone(inactive)
        self.assertEqual(out["memory_channel"], "heap" if heap_probe_available() else "rss")


class TestPeakRSS(unittest.TestCase):
    def test_returns_positive_bytes(self) -> None:
        # 1 MB is a trivial floor — any live Python process clears this.
        self.assertGreater(peak_rss_bytes(), 1_000_000)

    def test_peak_rss_is_monotonic_under_allocation(self) -> None:
        """`getrusage(RUSAGE_SELF).ru_maxrss` is a kernel high-water mark,
        so a fresh sample after a large fresh allocation must be >= the
        previous sample.

        This is primarily a UNIT regression test: if `ru_maxrss` (KiB on
        Linux) were returned unscaled, the reading would be ~1024x too small
        and the growth assertion below could not hold.

        The allocation is sized against the CURRENT headroom rather than being
        a fixed 100 MB. `ru_maxrss` is a lifetime mark, so a fixed size stops
        moving it once any earlier test in the same process has peaked higher
        — which made this test order-dependent.
        """
        import psutil

        before = peak_rss_bytes()
        headroom = before - psutil.Process().memory_info().rss
        size = max(0, headroom) + 100 * 1024 * 1024
        # Touch every page to force commit so the kernel counts it in RSS.
        big = bytearray(size)
        for i in range(0, len(big), 4096):
            big[i] = 1
        after = peak_rss_bytes()
        self.assertGreaterEqual(after, before)
        # The mark must have grown by something close to the fresh 100 MB
        # beyond the headroom (generous slack for OS bookkeeping).
        self.assertGreater(after, before + 50_000_000)
        del big


class TestRssSampler(unittest.TestCase):
    def test_stop_returns_zero_when_nothing_allocated(self) -> None:
        sampler = RssSampler(interval_s=0.001)
        sampler.start()
        time.sleep(0.01)
        delta = sampler.stop()
        # No allocations during the window → near-zero delta.
        self.assertLess(delta, 1_000_000)

    def test_stop_captures_allocation_during_window(self) -> None:
        sampler = RssSampler(interval_s=0.001)
        sampler.start()
        big = bytearray(40 * 1024 * 1024)  # 40 MB
        for i in range(0, len(big), 4096):
            big[i] = 1
        time.sleep(0.02)
        delta = sampler.stop()
        self.assertGreater(delta, 10_000_000)

    def test_start_twice_raises(self) -> None:
        """Previously a second start() silently leaked the running thread."""
        sampler = RssSampler(interval_s=0.001)
        sampler.start()
        try:
            with self.assertRaises(RuntimeError):
                sampler.start()
        finally:
            sampler.stop()


@requires_heap
class TestHeapSampler(unittest.TestCase):
    def test_captures_allocation_during_window(self) -> None:
        sampler = HeapSampler(interval_s=0.001)
        sampler.start()
        big = bytearray(40 * 1024 * 1024)
        big[0] = 1
        time.sleep(0.01)
        delta = sampler.stop()
        self.assertGreater(delta, 10_000_000)

    def test_uordblks_alone_would_miss_mmap_blocks(self) -> None:
        """glibc serves large blocks via mmap, which lands in `hblkhd`, not
        `uordblks`. Summing both is why the probe sees them; this pins that."""
        import ctypes

        lib = ctypes.CDLL(None)
        fn = lib.mallinfo2
        fn.restype = instrument_mod._Mallinfo2
        fn.argtypes = []
        before = fn()
        big = bytearray(64 * 1024 * 1024)  # comfortably over any mmap threshold
        big[0] = 1
        after = fn()
        combined = (after.uordblks + after.hblkhd) - (before.uordblks + before.hblkhd)
        self.assertGreater(combined, 60_000_000)
        del big


if __name__ == "__main__":
    unittest.main()
