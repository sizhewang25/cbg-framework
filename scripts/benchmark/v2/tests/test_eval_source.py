"""eval-source metric math over a tiny hand-computed canonical CSV.

Fixture geometry (all on the equator so geodesic distances are exact
fractions of one degree of longitude, 111.1949 km at EARTH_RADIUS_KM=6371):

  T1 at (0, 0):
    V1 at (0, 1)  gc ~ 111.195 km,  rtt 3 ms  -> radius 300 km, inflation ~2.698
    V2 at (0, 5)  gc ~ 555.975 km,  rtt 6 ms  -> radius 600 km, inflation ~1.079
    (V1, T1) also has a duplicate 4 ms observation that dedupe must drop.
  T2 at (0, 10):
    V2 only       gc ~ 555.975 km,  rtt 20 ms -> single-VP degenerate case.

Checks: pair dedupe keeps min RTT; min_inflation may come from a *farther*
VP than closest_vp_km (V2 for T1); the inverse-RTT weighted mean matches the
hand computation; a single-VP target's weighted distance equals its gc.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark.v2.eval_source import (
    build_pairs,
    eval_source,
    load_canonical_csv,
    per_target_metrics,
)
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM, THEORETICAL_SLOPE

_DEG_KM = EARTH_RADIUS_KM * math.pi / 180  # ~111.1949 km per equatorial degree


def _obs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vp_id": ["V1", "V1", "V2", "V2"],
            "vp_lat": [0.0, 0.0, 0.0, 0.0],
            "vp_lon": [1.0, 1.0, 5.0, 5.0],
            "target_id": ["T1", "T1", "T1", "T2"],
            "target_lat": [0.0, 0.0, 0.0, 0.0],
            "target_lon": [0.0, 0.0, 0.0, 10.0],
            "rtt_ms": [3.0, 4.0, 6.0, 20.0],
        }
    )


def _write_csv(df: pd.DataFrame, dir_: Path) -> Path:
    path = dir_ / "tiny.csv"
    df.to_csv(path, index=False)
    return path


class TestBuildPairs(unittest.TestCase):
    def test_dedupes_pairs_to_min_rtt(self) -> None:
        pairs = build_pairs(_obs_frame())
        self.assertEqual(len(pairs), 3)
        v1t1 = pairs[(pairs["vp_id"] == "V1") & (pairs["target_id"] == "T1")]
        self.assertAlmostEqual(float(v1t1["rtt_ms"].iloc[0]), 3.0)

    def test_pair_geometry(self) -> None:
        pairs = build_pairs(_obs_frame()).set_index(["vp_id", "target_id"])
        row = pairs.loc[("V1", "T1")]
        self.assertAlmostEqual(row["gc_km"], _DEG_KM, places=3)
        self.assertAlmostEqual(row["radius_km"], 3.0 / THEORETICAL_SLOPE, places=6)
        self.assertAlmostEqual(
            row["inflation"], 3.0 / (THEORETICAL_SLOPE * _DEG_KM), places=6
        )


class TestPerTargetMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.per_target = per_target_metrics(build_pairs(_obs_frame())).set_index(
            "target_id"
        )

    def test_availability_and_geography(self) -> None:
        t1 = self.per_target.loc["T1"]
        self.assertEqual(int(t1["n_avail_vps"]), 2)
        self.assertAlmostEqual(t1["closest_vp_km"], _DEG_KM, places=3)

    def test_rtt_axis(self) -> None:
        t1 = self.per_target.loc["T1"]
        self.assertAlmostEqual(t1["min_rtt_ms"], 3.0)
        self.assertAlmostEqual(t1["best_radius_km"], 300.0, places=6)
        # min_inflation comes from V2 (farther but relatively less inflated),
        # not from the geographically closest VP.
        self.assertAlmostEqual(
            t1["min_inflation"], 6.0 / (THEORETICAL_SLOPE * 5 * _DEG_KM), places=6
        )

    def test_rtt_weighted_dist_matches_hand_computation(self) -> None:
        t1 = self.per_target.loc["T1"]
        expected = (_DEG_KM / 3 + 5 * _DEG_KM / 6) / (1 / 3 + 1 / 6)
        self.assertAlmostEqual(t1["rtt_weighted_dist_km"], expected, places=6)

    def test_single_vp_target_degenerates_to_its_gc(self) -> None:
        t2 = self.per_target.loc["T2"]
        self.assertEqual(int(t2["n_avail_vps"]), 1)
        self.assertAlmostEqual(t2["rtt_weighted_dist_km"], t2["closest_vp_km"], places=9)
        self.assertAlmostEqual(t2["closest_vp_km"], 5 * _DEG_KM, places=3)


class TestLoadCanonicalCsv(unittest.TestCase):
    def test_case_insensitive_columns_and_row_hygiene(self) -> None:
        df = _obs_frame().rename(columns=str.upper)
        df.loc[len(df)] = ["V9", 0.0, 2.0, "T1", 0.0, 0.0, -1.0]  # non-positive RTT
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_canonical_csv(_write_csv(df, Path(tmp)))
        self.assertEqual(len(loaded), 4)
        self.assertNotIn("V9", set(loaded["vp_id"]))

    def test_missing_column_raises(self) -> None:
        df = _obs_frame().drop(columns=["rtt_ms"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_canonical_csv(_write_csv(df, Path(tmp)))


class TestEvalSource(unittest.TestCase):
    def test_writes_outputs_and_summary_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(_obs_frame(), Path(tmp))
            stats = eval_source(csv_path, Path(tmp), thresholds=(500.0,))
            self.assertTrue(Path(stats["per_target_csv"]).exists())
            self.assertTrue(Path(stats["stats_json"]).exists())
        self.assertEqual(stats["n_targets"], 2)
        self.assertEqual(stats["n_pairs"], 3)
        self.assertEqual(stats["n_vps"], 2)
        # T1's weighted dist (~259 km) clears 500 km; T2's (~556 km) does not.
        self.assertAlmostEqual(
            stats["resolvability"]["rtt_weighted_dist_km"]["within_500km"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
