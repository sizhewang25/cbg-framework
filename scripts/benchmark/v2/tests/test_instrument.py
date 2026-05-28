"""TimingMemoryInstrument records per-stage timing, tracemalloc peak, and
RSS-sampled peak. peak_rss_bytes() returns the kernel-tracked lifetime maxrss.
"""

from __future__ import annotations

import time
import unittest

from scripts.benchmark.v2.instrument import (
    RssSampler,
    TimingMemoryInstrument,
    measure_block,
    peak_rss_bytes,
)


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

    def test_rss_peak_bytes_reflects_large_alloc_inside_stage(self) -> None:
        # Default sampler interval is 5 ms — use a tighter one so the test
        # doesn't have to sleep long, then hold the allocation across a
        # handful of intervals so the sampler reliably catches the peak.
        instr = TimingMemoryInstrument(rss_sample_interval_s=0.001)
        with instr("mtl"):
            big = bytearray(50 * 1024 * 1024)  # 50 MB
            for i in range(0, len(big), 4096):  # touch pages to force commit
                big[i] = 1
            time.sleep(0.02)  # 20 ms — gives the sampler ~20 chances
        rec = instr.get("mtl")
        self.assertIsNotNone(rec)
        self.assertGreater(rec.rss_peak_bytes, 10_000_000)

    def test_rss_peak_bytes_small_for_trivial_stage(self) -> None:
        instr = TimingMemoryInstrument(rss_sample_interval_s=0.001)
        with instr("ltd"):
            pass
        rec = instr.get("ltd")
        self.assertIsNotNone(rec)
        # No allocations → near-zero delta. Allow a small slack for OS noise.
        self.assertLess(rec.rss_peak_bytes, 1_000_000)

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
        self.assertGreaterEqual(out["rss_peak_bytes"], 0)


class TestPeakRSS(unittest.TestCase):
    def test_returns_positive_bytes(self) -> None:
        # 1 MB is a trivial floor — any live Python process clears this.
        self.assertGreater(peak_rss_bytes(), 1_000_000)

    def test_peak_rss_is_monotonic_under_allocation(self) -> None:
        """`getrusage(RUSAGE_SELF).ru_maxrss` is a kernel high-water mark,
        so a fresh sample after a large fresh allocation must be >= the
        previous sample. Allocate enough to push RSS above any prior peak."""
        before = peak_rss_bytes()
        # 100 MB — well above any baseline noise; touch every page to force
        # commit so the kernel actually counts it in RSS.
        big = bytearray(100 * 1024 * 1024)
        for i in range(0, len(big), 4096):
            big[i] = 1
        after = peak_rss_bytes()
        self.assertGreaterEqual(after, before)
        # And the maxrss must have grown by something close to the allocation
        # (allow generous slack for OS bookkeeping). If this assertion fails,
        # ru_maxrss is probably reporting in the wrong units.
        self.assertGreater(after, before + 50_000_000)


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


if __name__ == "__main__":
    unittest.main()
