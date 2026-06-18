"""Closest-airport postprocessing over a targets.parquet.

`annotate_targets` appends the airport columns (truth/pred nearest IATA + km,
the airport-to-airport gap, and the exact match flag) without touching any
runner state. `summarize_airport` reduces them to a per-combo airport summary:
the exact match rate, a threshold (40 km) match rate, and pred-distance stats,
over SUCCESS/FALLBACK rows. `process_parquet` round-trips a file in place and
is idempotent.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark.v2.airports import AirportIndex
from scripts.benchmark.v2.airport_eval import (
    AIRPORT_COLUMNS,
    annotate_targets,
    process_parquet,
    summarize_airport,
)

_JFK = (40.6413, -73.7781)
_LGA = (40.7769, -73.8740)   # ~17 km from JFK — same NYC metro, different field
_LAX = (33.9416, -118.4085)
_LHR = (51.4700, -0.4543)


def _index() -> AirportIndex:
    return AirportIndex(
        pd.DataFrame(
            {
                "iata_code": ["JFK", "LGA", "LAX", "LHR"],
                "latitude_deg": [_JFK[0], _LGA[0], _LAX[0], _LHR[0]],
                "longitude_deg": [_JFK[1], _LGA[1], _LAX[1], _LHR[1]],
                "municipality": ["New York", "New York", "Los Angeles", "London"],
            }
        )
    )


def _targets() -> pd.DataFrame:
    # Truth always sits at JFK. A: hit (pred at JFK). B: transcontinental miss
    # (pred at LAX). C: failed prediction (null). D: same-metro near-miss (pred
    # at LGA) — exact match fails but the 40 km threshold should pass.
    return pd.DataFrame(
        {
            "target_id": ["A", "B", "C", "D"],
            "target_lat": [_JFK[0]] * 4,
            "target_lon": [_JFK[1]] * 4,
            "pred_lat": [40.66, _LAX[0], None, _LGA[0]],
            "pred_lon": [-73.79, _LAX[1], None, _LGA[1]],
            "status": ["SUCCESS", "SUCCESS", "ERROR", "SUCCESS"],
            "error_km": [3.0, 3900.0, None, 17.0],
        }
    )


class TestAnnotateTargets(unittest.TestCase):
    def test_adds_all_airport_columns(self) -> None:
        out = annotate_targets(_targets(), _index())
        for col in AIRPORT_COLUMNS:
            self.assertIn(col, out.columns)

    def test_truth_airport_always_populated(self) -> None:
        out = annotate_targets(_targets(), _index())
        self.assertEqual(list(out["truth_airport_iata"]), ["JFK"] * 4)
        self.assertTrue((out["truth_airport_km"] < 5).all())

    def test_exact_match_flag_true_on_hit_false_otherwise(self) -> None:
        out = annotate_targets(_targets(), _index()).set_index("target_id")
        self.assertEqual(out.loc["A", "pred_airport_iata"], "JFK")
        self.assertTrue(bool(out.loc["A", "airport_match"]))
        self.assertEqual(out.loc["B", "pred_airport_iata"], "LAX")
        self.assertFalse(bool(out.loc["B", "airport_match"]))
        # Same-metro neighbour: exact match is False (LGA != JFK)...
        self.assertEqual(out.loc["D", "pred_airport_iata"], "LGA")
        self.assertFalse(bool(out.loc["D", "airport_match"]))

    def test_pred_truth_airport_gap_distance(self) -> None:
        out = annotate_targets(_targets(), _index()).set_index("target_id")
        # A resolves to the same airport as truth → ~0 km gap.
        self.assertAlmostEqual(out.loc["A", "pred_truth_airport_km"], 0.0, places=3)
        # D is the JFK↔LGA gap: same metro, well under 40 km but clearly > 0.
        self.assertGreater(out.loc["D", "pred_truth_airport_km"], 5.0)
        self.assertLess(out.loc["D", "pred_truth_airport_km"], 40.0)
        # B is transcontinental.
        self.assertGreater(out.loc["B", "pred_truth_airport_km"], 3000.0)

    def test_null_prediction_yields_null_pred_columns(self) -> None:
        out = annotate_targets(_targets(), _index()).set_index("target_id")
        self.assertTrue(pd.isna(out.loc["C", "pred_airport_iata"]))
        self.assertTrue(pd.isna(out.loc["C", "pred_airport_km"]))
        self.assertTrue(pd.isna(out.loc["C", "pred_truth_airport_km"]))
        self.assertTrue(pd.isna(out.loc["C", "airport_match"]))

    def test_idempotent(self) -> None:
        idx = _index()
        once = annotate_targets(_targets(), idx)
        twice = annotate_targets(once, idx)
        pd.testing.assert_frame_equal(once, twice)


class TestSummarizeAirport(unittest.TestCase):
    def test_exact_match_rate_over_success_fallback_only(self) -> None:
        summ = summarize_airport(annotate_targets(_targets(), _index()))
        # Scored rows A, B, D (C is ERROR). Only A matches exactly → 1/3.
        self.assertEqual(summ["n"], 3)
        self.assertAlmostEqual(summ["airport_match_rate"], 1 / 3)

    def test_threshold_match_rate_is_more_forgiving(self) -> None:
        summ = summarize_airport(annotate_targets(_targets(), _index()), thresholds=(40.0,))
        # Within 40 km: A (0) and D (~17) pass, B (transcontinental) fails → 2/3.
        self.assertAlmostEqual(summ["airport_match_rate_within_40km"], 2 / 3)
        # Threshold rate must dominate the exact rate.
        self.assertGreater(summ["airport_match_rate_within_40km"], summ["airport_match_rate"])

    def test_custom_threshold(self) -> None:
        summ = summarize_airport(annotate_targets(_targets(), _index()), thresholds=(100.0,))
        self.assertIn("airport_match_rate_within_100km", summ)
        self.assertAlmostEqual(summ["airport_match_rate_within_100km"], 2 / 3)

    def test_pred_distance_stats_present(self) -> None:
        summ = summarize_airport(annotate_targets(_targets(), _index()))
        for col in ("pred_airport_km", "pred_truth_airport_km"):
            for stat in ("p5", "p50", "p95", "mean", "std"):
                self.assertIn(f"{col}_{stat}", summ)

    def test_empty_frame_is_safe(self) -> None:
        empty = annotate_targets(_targets(), _index()).iloc[0:0]
        summ = summarize_airport(empty)
        self.assertEqual(summ["n"], 0)
        self.assertTrue(math.isnan(summ["airport_match_rate"]))
        self.assertTrue(math.isnan(summ["airport_match_rate_within_40km"]))


class TestProcessParquet(unittest.TestCase):
    def test_roundtrip_adds_columns_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "targets.parquet"
            _targets().to_parquet(path)
            process_parquet(path, _index())
            back = pd.read_parquet(path)
            for col in AIRPORT_COLUMNS:
                self.assertIn(col, back.columns)

    def test_process_is_idempotent_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "targets.parquet"
            _targets().to_parquet(path)
            idx = _index()
            process_parquet(path, idx)
            first = pd.read_parquet(path)
            process_parquet(path, idx)
            second = pd.read_parquet(path)
            pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
