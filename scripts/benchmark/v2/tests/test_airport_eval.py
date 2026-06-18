"""Closest-airport postprocessing over a targets.parquet.

`annotate_targets` appends the five airport columns (truth/pred nearest IATA +
km, and the match flag) without touching any runner state. `summarize_airport`
reduces them to a per-combo airport summary (match rate + pred-distance stats)
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
_LAX = (33.9416, -118.4085)
_LHR = (51.4700, -0.4543)


def _index() -> AirportIndex:
    return AirportIndex(
        pd.DataFrame(
            {
                "iata_code": ["JFK", "LAX", "LHR"],
                "latitude_deg": [_JFK[0], _LAX[0], _LHR[0]],
                "longitude_deg": [_JFK[1], _LAX[1], _LHR[1]],
                "municipality": ["New York", "Los Angeles", "London"],
            }
        )
    )


def _targets() -> pd.DataFrame:
    # A: hit (pred near truth, both near JFK). B: transcontinental miss
    # (truth near JFK, pred near LAX). C: failed prediction (null pred).
    return pd.DataFrame(
        {
            "target_id": ["A", "B", "C"],
            "target_lat": [40.7580, 40.7580, 40.7580],
            "target_lon": [-73.9855, -73.9855, -73.9855],
            "pred_lat": [40.70, 34.05, None],
            "pred_lon": [-73.99, -118.25, None],
            "status": ["SUCCESS", "SUCCESS", "ERROR"],
            "error_km": [10.0, 3900.0, None],
        }
    )


class TestAnnotateTargets(unittest.TestCase):
    def test_adds_all_airport_columns(self) -> None:
        out = annotate_targets(_targets(), _index())
        for col in AIRPORT_COLUMNS:
            self.assertIn(col, out.columns)

    def test_truth_airport_always_populated(self) -> None:
        out = annotate_targets(_targets(), _index())
        # All three targets sit near JFK regardless of prediction success.
        self.assertEqual(list(out["truth_airport_iata"]), ["JFK", "JFK", "JFK"])
        self.assertTrue((out["truth_airport_km"] < 50).all())

    def test_match_flag_true_on_hit_false_on_miss(self) -> None:
        out = annotate_targets(_targets(), _index()).set_index("target_id")
        self.assertEqual(out.loc["A", "pred_airport_iata"], "JFK")
        self.assertTrue(bool(out.loc["A", "airport_match"]))
        self.assertEqual(out.loc["B", "pred_airport_iata"], "LAX")
        self.assertFalse(bool(out.loc["B", "airport_match"]))

    def test_null_prediction_yields_null_pred_columns_and_match(self) -> None:
        out = annotate_targets(_targets(), _index()).set_index("target_id")
        self.assertTrue(pd.isna(out.loc["C", "pred_airport_iata"]))
        self.assertTrue(pd.isna(out.loc["C", "pred_airport_km"]))
        self.assertTrue(pd.isna(out.loc["C", "airport_match"]))

    def test_idempotent(self) -> None:
        idx = _index()
        once = annotate_targets(_targets(), idx)
        twice = annotate_targets(once, idx)
        pd.testing.assert_frame_equal(once, twice)


class TestSummarizeAirport(unittest.TestCase):
    def test_match_rate_over_success_fallback_only(self) -> None:
        out = annotate_targets(_targets(), _index())
        summ = summarize_airport(out)
        # A matches, B does not; C (ERROR) excluded → 1/2.
        self.assertAlmostEqual(summ["airport_match_rate"], 0.5)
        self.assertEqual(summ["n"], 2)

    def test_pred_airport_km_stats_present(self) -> None:
        out = annotate_targets(_targets(), _index())
        summ = summarize_airport(out)
        for stat in ("p5", "p50", "p95", "mean", "std"):
            self.assertIn(f"pred_airport_km_{stat}", summ)

    def test_empty_frame_is_safe(self) -> None:
        empty = annotate_targets(_targets(), _index()).iloc[0:0]
        summ = summarize_airport(empty)
        self.assertEqual(summ["n"], 0)
        self.assertTrue(math.isnan(summ["airport_match_rate"]))


class TestProcessParquet(unittest.TestCase):
    def test_roundtrip_adds_columns_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "targets.parquet"
            _targets().to_parquet(path)
            process_parquet(path, _index())
            back = pd.read_parquet(path)
            for col in AIRPORT_COLUMNS:
                self.assertIn(col, back.columns)
            self.assertEqual(
                list(back.set_index("target_id")["truth_airport_iata"]),
                ["JFK", "JFK", "JFK"],
            )

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
