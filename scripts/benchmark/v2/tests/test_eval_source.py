"""eval-source metric math over a tiny hand-computed canonical CSV.

Fixture geometry (all on the equator so geodesic distances are exact
fractions of one degree of longitude, 111.1949 km at EARTH_RADIUS_KM=6371):

  T1 at (0, 0):
    V1 at (0, 1)  gc ~ 111.195 km,  rtt 3 ms  -> radius 300 km, inflation ~2.698
    V2 at (0, 5)  gc ~ 555.975 km,  rtt 6 ms  -> radius 600 km, inflation ~1.079
    (V1, T1) also has a duplicate 4 ms observation that dedupe must drop.
  T2 at (0, 12):
    V2 only       gc ~ 778.364 km,  rtt 20 ms -> single-VP degenerate case.
    (12 deg, not 10: at (0, 10) V2 would sit exactly on the Voronoi boundary
    between the two singleton cluster centroids, making the same-cluster
    assignment a floating-point coin flip.)

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
    cluster_targets,
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
            "target_lon": [0.0, 0.0, 0.0, 12.0],
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

    def test_shortest_ping_vp_tracks_min_rtt_not_min_distance(self) -> None:
        # The farther VP pings faster (both RTTs physically plausible), so
        # shortest_ping_vp_km must diverge from closest_vp_km.
        df = pd.DataFrame(
            {
                "vp_id": ["NEAR", "FAR"],
                "vp_lat": [0.0, 0.0],
                "vp_lon": [1.0, 5.0],
                "target_id": ["T", "T"],
                "target_lat": [0.0, 0.0],
                "target_lon": [0.0, 0.0],
                "rtt_ms": [8.0, 6.0],
            }
        )
        t = per_target_metrics(build_pairs(df)).set_index("target_id").loc["T"]
        self.assertAlmostEqual(t["closest_vp_km"], _DEG_KM, places=3)
        self.assertAlmostEqual(t["shortest_ping_vp_km"], 5 * _DEG_KM, places=3)

    def test_shortest_ping_equals_closest_when_fastest_is_nearest(self) -> None:
        t1 = self.per_target.loc["T1"]
        self.assertAlmostEqual(t1["shortest_ping_vp_km"], t1["closest_vp_km"], places=9)

    def test_single_vp_target_degenerates_to_its_gc(self) -> None:
        t2 = self.per_target.loc["T2"]
        self.assertEqual(int(t2["n_avail_vps"]), 1)
        self.assertAlmostEqual(t2["rtt_weighted_dist_km"], t2["closest_vp_km"], places=9)
        self.assertAlmostEqual(t2["closest_vp_km"], 7 * _DEG_KM, places=3)


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

    def test_target_clustering_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(_obs_frame(), Path(tmp))
            stats = eval_source(
                csv_path, Path(tmp), thresholds=(500.0,), cluster_radius_km=200.0
            )
            per_target = pd.read_csv(stats["per_target_csv"])
        clustering = stats["target_clustering"]
        # T1 and T2 are ~1334 km apart -> two singleton clusters at R=200.
        self.assertEqual(clustering["n_clusters"], 2)
        self.assertEqual(clustering["radius_km"], 200.0)
        self.assertAlmostEqual(clustering["targets_per_cluster"], 1.0)
        # Only T1's closest VP (~111 km) sits within R; T2's is ~778 km out.
        self.assertAlmostEqual(clustering["closest_vp_within_radius_share"], 0.5)
        # T1's min-RTT VP is also its closest (V1); T2's is out of range.
        self.assertAlmostEqual(
            clustering["shortest_ping_vp_within_radius_share"], 0.5
        )
        # Voronoi assignment: V1 (0,1) is nearest T1's centroid -> same
        # cluster; V2 (0,5) is nearer T1's centroid (5 deg) than T2's (7 deg)
        # -> T2's VP lands in the wrong cell.
        self.assertAlmostEqual(clustering["closest_vp_in_same_cluster_share"], 0.5)
        self.assertAlmostEqual(
            clustering["shortest_ping_vp_in_same_cluster_share"], 0.5
        )
        self.assertEqual(per_target["cluster_id"].nunique(), 2)

    def test_same_cluster_is_looser_than_within_radius(self) -> None:
        # TA's only VP is 3 deg (~334 km) away: beyond R=200 (strict check
        # fails) but still Voronoi-assigned to TA's cell, 3 deg vs 7 deg to
        # TB's (loose check passes). TB's VP at 1 deg passes both.
        df = pd.DataFrame(
            {
                "vp_id": ["VA", "VB"],
                "vp_lat": [0.0, 0.0],
                "vp_lon": [3.0, 9.0],
                "target_id": ["TA", "TB"],
                "target_lat": [0.0, 0.0],
                "target_lon": [0.0, 10.0],
                "rtt_ms": [5.0, 2.0],
            }
        )
        per_target = per_target_metrics(build_pairs(df))
        per_target, block = cluster_targets(per_target, radius_km=200.0)
        ta = per_target.set_index("target_id").loc["TA"]
        self.assertGreater(ta["closest_vp_km"], 200.0)
        self.assertTrue(bool(ta["closest_vp_in_same_cluster"]))
        self.assertAlmostEqual(block["closest_vp_within_radius_share"], 0.5)
        self.assertAlmostEqual(block["closest_vp_in_same_cluster_share"], 1.0)
        self.assertAlmostEqual(block["shortest_ping_vp_in_same_cluster_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
