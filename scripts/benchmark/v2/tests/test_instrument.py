"""TimingMemoryInstrument records per-stage timing + tracemalloc peaks."""

from __future__ import annotations

import time
import unittest

from scripts.benchmark.v2.instrument import (
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

    def test_peak_bytes_reflects_allocation_inside_stage(self) -> None:
        instr = TimingMemoryInstrument()
        with instr("ltd"):
            # Allocate ~1 MB inside the stage; tracemalloc should see it.
            big = bytearray(1024 * 1024)
            big[0] = 1  # ensure it's actually used
        rec = instr.get("ltd")
        self.assertIsNotNone(rec)
        self.assertGreater(rec.peak_bytes, 500_000)

    def test_get_returns_none_for_missing_stage(self) -> None:
        instr = TimingMemoryInstrument()
        with instr("ltd"):
            pass
        self.assertIsNone(instr.get("mtl"))

    def test_record_named_appends(self) -> None:
        instr = TimingMemoryInstrument()
        instr.record_named("fit", duration_ns=12345, peak_bytes=678)
        rec = instr.get("fit")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.duration_ns, 12345)
        self.assertEqual(rec.peak_bytes, 678)


class TestMeasureBlock(unittest.TestCase):
    def test_block_populates_duration_and_peak(self) -> None:
        with measure_block("fit") as out:
            big = bytearray(512 * 1024)
            big[0] = 1
        self.assertIn("duration_ns", out)
        self.assertIn("peak_bytes", out)
        self.assertGreater(out["duration_ns"], 0)
        self.assertGreater(out["peak_bytes"], 200_000)


class TestPeakRSS(unittest.TestCase):
    def test_returns_positive_bytes(self) -> None:
        # 1 MB is a trivial floor — any live Python process clears this.
        self.assertGreater(peak_rss_bytes(), 1_000_000)


if __name__ == "__main__":
    unittest.main()
