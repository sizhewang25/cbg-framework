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

The precheck additions get their own equatorial fixtures: RTT rank
normalization, Spearman coherence + its guards, SOI violation shares, the
anycast disk-disjointness test, the per-cluster frame (cell gap + top-N
neighbors), the discriminative-set proximity ladder
(NO_PROXIMITY/HAS_NOT_USED_PROXIMITY/HAS_USED_PROXIMITY with Voronoi luck
as a diagnostic only), and the distance-mesh artifacts.
"""

from __future__ import annotations

import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark.v2.eval_source import (
    anycast_metrics,
    apply_eval_target_filters,
    build_pairs,
    cluster_targets,
    eval_source,
    load_canonical_csv,
    per_target_metrics,
    proximity_metrics,
    _derive_eval_pair_weight_min,
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
            stats = eval_source(csv_path, Path(tmp))
            self.assertTrue(Path(stats["per_target_csv"]).exists())
            self.assertTrue(Path(stats["stats_json"]).exists())
        self.assertEqual(stats["n_targets"], 2)
        self.assertEqual(stats["n_pairs"], 3)
        self.assertEqual(stats["n_vps"], 2)
        self.assertEqual(stats["metrics"]["rtt_weighted_dist_km"]["n"], 2)
        # Removed axes stay removed: no threshold table, no LTD-flavored
        # constraint radius (that analysis is LTD-specific, not a precheck).
        self.assertNotIn("resolvability", stats)
        self.assertNotIn("best_radius_km", stats["metrics"])

    def test_target_clustering_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(_obs_frame(), Path(tmp))
            stats = eval_source(csv_path, Path(tmp), cluster_radius_km=200.0)
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
        per_target, _clusters, block = cluster_targets(per_target, radius_km=200.0)
        ta = per_target.set_index("target_id").loc["TA"]
        self.assertGreater(ta["closest_vp_km"], 200.0)
        self.assertTrue(bool(ta["closest_vp_in_same_cluster"]))
        self.assertAlmostEqual(block["closest_vp_within_radius_share"], 0.5)
        self.assertAlmostEqual(block["closest_vp_in_same_cluster_share"], 1.0)
        self.assertAlmostEqual(block["shortest_ping_vp_in_same_cluster_share"], 1.0)


def _equator_frame(rows: list[tuple[str, float, str, float, float]]) -> pd.DataFrame:
    """Canonical frame from (vp_id, vp_lon, target_id, target_lon, rtt_ms)
    tuples, everything on the equator."""
    return pd.DataFrame(
        {
            "vp_id": [r[0] for r in rows],
            "vp_lat": 0.0,
            "vp_lon": [r[1] for r in rows],
            "target_id": [r[2] for r in rows],
            "target_lat": 0.0,
            "target_lon": [r[3] for r in rows],
            "rtt_ms": [r[4] for r in rows],
        }
    )


class TestRttRankNorm(unittest.TestCase):
    def test_rank_spans_zero_to_one_and_single_vp_is_zero(self) -> None:
        pairs = build_pairs(_obs_frame()).set_index(["vp_id", "target_id"])
        self.assertAlmostEqual(pairs.loc[("V1", "T1"), "rtt_rank_norm"], 0.0)
        self.assertAlmostEqual(pairs.loc[("V2", "T1"), "rtt_rank_norm"], 1.0)
        self.assertAlmostEqual(pairs.loc[("V2", "T2"), "rtt_rank_norm"], 0.0)

    def test_ties_share_the_lower_rank(self) -> None:
        df = _equator_frame([("A", 1.0, "T", 0.0, 5.0), ("B", 2.0, "T", 0.0, 5.0)])
        pairs = build_pairs(df)
        self.assertTrue((pairs["rtt_rank_norm"] == 0.0).all())


class TestDisagreementAndRank(unittest.TestCase):
    def test_agreement_when_closest_is_fastest(self) -> None:
        t1 = per_target_metrics(build_pairs(_obs_frame())).set_index("target_id").loc["T1"]
        self.assertTrue(bool(t1["closest_is_shortest_ping"]))
        self.assertAlmostEqual(t1["closest_to_shortest_ping_km"], 0.0, places=9)
        self.assertAlmostEqual(t1["closest_vp_rtt_rank"], 0.0)

    def test_disagreement_reports_vp_separation_and_buried_rank(self) -> None:
        # The far VP pings faster: the closest VP is the target's slowest
        # (rank 1.0) and the two VPs sit 4 degrees apart.
        df = _equator_frame(
            [("NEAR", 1.0, "T", 0.0, 8.0), ("FAR", 5.0, "T", 0.0, 6.0)]
        )
        t = per_target_metrics(build_pairs(df)).set_index("target_id").loc["T"]
        self.assertFalse(bool(t["closest_is_shortest_ping"]))
        self.assertAlmostEqual(t["closest_to_shortest_ping_km"], 4 * _DEG_KM, places=3)
        self.assertAlmostEqual(t["closest_vp_rtt_rank"], 1.0)


class TestSpearman(unittest.TestCase):
    def test_monotone_rtt_distance_gives_plus_one(self) -> None:
        df = _equator_frame(
            [(f"V{i}", float(i), "T", 0.0, 2.0 * i) for i in range(1, 5)]
        )
        t = per_target_metrics(build_pairs(df), spearman_min_pairs=4)
        self.assertAlmostEqual(float(t["rtt_dist_spearman"].iloc[0]), 1.0)

    def test_anti_monotone_gives_minus_one(self) -> None:
        df = _equator_frame(
            [(f"V{i}", float(i), "T", 0.0, 2.0 * (5 - i)) for i in range(1, 5)]
        )
        t = per_target_metrics(build_pairs(df), spearman_min_pairs=4)
        self.assertAlmostEqual(float(t["rtt_dist_spearman"].iloc[0]), -1.0)

    def test_guards_return_nan(self) -> None:
        few = _equator_frame(
            [(f"V{i}", float(i), "T", 0.0, 2.0 * i) for i in range(1, 5)]
        )
        t = per_target_metrics(build_pairs(few), spearman_min_pairs=8)
        self.assertTrue(math.isnan(float(t["rtt_dist_spearman"].iloc[0])))
        constant = _equator_frame(
            [(f"V{i}", float(i), "T", 0.0, 5.0) for i in range(1, 5)]
        )
        t = per_target_metrics(build_pairs(constant), spearman_min_pairs=4)
        self.assertTrue(math.isnan(float(t["rtt_dist_spearman"].iloc[0])))


class TestSoiViolations(unittest.TestCase):
    def test_violation_share_counts_sub_lightspeed_pairs(self) -> None:
        # V_BAD: 3 ms over 5 deg (~556 km) needs ~5.56 ms at 2/3c — impossible.
        df = _equator_frame(
            [("V_OK", 1.0, "T", 0.0, 3.0), ("V_BAD", 5.0, "T", 0.0, 3.0)]
        )
        t = per_target_metrics(build_pairs(df)).set_index("target_id").loc["T"]
        self.assertAlmostEqual(t["soi_violation_share"], 0.5)

    def test_colocated_pairs_leave_the_denominator(self) -> None:
        df = _equator_frame(
            [("V_COLO", 0.0, "T", 0.0, 1.0), ("V_OK", 1.0, "T", 0.0, 3.0)]
        )
        t = per_target_metrics(build_pairs(df)).set_index("target_id").loc["T"]
        self.assertAlmostEqual(t["soi_violation_share"], 0.0)


class TestAnycastMetrics(unittest.TestCase):
    def test_disjoint_low_rtt_disks_flag_anycast(self) -> None:
        # Two 200 km disks (2 ms each) ~2213 km apart cannot both hold a
        # unicast target.
        df = _equator_frame(
            [("A1", 0.1, "T", 0.0, 2.0), ("A2", 20.0, "T", 0.0, 2.0)]
        )
        row = anycast_metrics(build_pairs(df)).set_index("target_id").loc["T"]
        self.assertLess(row["vp_pair_disk_overlap_km"], 0.0)
        self.assertAlmostEqual(
            row["vp_pair_disk_overlap_km"], 400.0 - 19.9 * _DEG_KM, places=2
        )
        self.assertEqual(int(row["n_disjoint_sites"]), 2)
        self.assertTrue(bool(row["anycast_suspect"]))

    def test_overlapping_disks_are_consistent(self) -> None:
        df = _equator_frame(
            [("B1", 1.0, "T", 0.0, 3.0), ("B2", 2.0, "T", 0.0, 4.0)]
        )
        row = anycast_metrics(build_pairs(df)).set_index("target_id").loc["T"]
        self.assertGreater(row["vp_pair_disk_overlap_km"], 0.0)
        self.assertEqual(int(row["n_disjoint_sites"]), 1)
        self.assertFalse(bool(row["anycast_suspect"]))

    def test_delta_ceiling_prunes_slow_vps(self) -> None:
        # C2's 50 ms is outside min_rtt + 10 ms: the low set is a single VP,
        # so the pair test is undefined and nothing is flagged.
        df = _equator_frame(
            [("C1", 1.0, "T", 0.0, 2.0), ("C2", 20.0, "T", 0.0, 50.0)]
        )
        row = anycast_metrics(build_pairs(df)).set_index("target_id").loc["T"]
        self.assertTrue(math.isnan(row["vp_pair_disk_overlap_km"]))
        self.assertEqual(int(row["n_disjoint_sites"]), 1)
        self.assertFalse(bool(row["anycast_suspect"]))


class TestClusterFrame(unittest.TestCase):
    def test_cell_gap_and_mutual_neighbors_for_two_singletons(self) -> None:
        per_target = per_target_metrics(build_pairs(_obs_frame()))
        _, clusters, _ = cluster_targets(per_target, radius_km=200.0)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(clusters["is_singleton"].all())
        for gap in clusters["cell_gap_km"]:
            self.assertAlmostEqual(gap, 12 * _DEG_KM, places=1)
        # The two singletons are each other's first neighbor.
        self.assertEqual(
            sorted(clusters["neighbor1_cluster_id"]), sorted(clusters["cluster_id"])
        )
        self.assertTrue(
            (clusters["neighbor1_km"] == clusters["cell_gap_km"]).all()
        )

    def test_single_cluster_answer_space_has_infinite_gap(self) -> None:
        df = _equator_frame(
            [("V1", 1.0, "TA", 0.0, 3.0), ("V1", 1.0, "TB", 0.5, 3.0)]
        )
        per_target = per_target_metrics(build_pairs(df))
        per_target, clusters, _ = cluster_targets(per_target, radius_km=200.0)
        self.assertEqual(len(clusters), 1)
        self.assertTrue(np.isinf(clusters["cell_gap_km"].iloc[0]))
        self.assertTrue(np.isinf(per_target["target_distinguishable_vp_dist_km"]).all())
        self.assertNotIn("neighbor1_cluster_id", clusters.columns)


class TestProximityLadder(unittest.TestCase):
    """One target per label plus a single-VP target, singletons at R=200.

    Centroid layout (equator degrees): T_USE at 0, T_HAS at 12, T_NO at 40,
    T_LONE at 80. Gaps: 12 / 12 / 28 / 40 deg, so the discriminative bounds
    (gap/2) are ~667 / ~667 / ~1557 / ~2224 km.
    """

    def _per_target(self) -> pd.DataFrame:
        df = _equator_frame(
            [
                # T_USE: both VPs inside the 667 km bound; fastest is in D.
                ("U1", 1.0, "T_USE", 0.0, 3.0),
                ("U2", 5.0, "T_USE", 0.0, 6.0),
                # T_HAS: H_C (1 deg) is in D but slow; H_R (13 deg, ~1446 km)
                # is outside D yet fastest -> HAS_NOT_USED. H_R still sits
                # nearer T_HAS's centroid (13 deg) than T_NO's (15 deg):
                # directional luck.
                ("H_C", 11.0, "T_HAS", 12.0, 10.0),
                ("H_R", 25.0, "T_HAS", 12.0, 5.0),
                # T_NO: both VPs beyond the 1557 km bound. N1 (the fastest)
                # sits at 62 deg — nearer T_LONE's centroid (18 deg) than
                # T_NO's own (22 deg), so the baseline is NOT lucky here.
                # (60 deg would be an exact Voronoi tie — a float coin flip.)
                ("N1", 62.0, "T_NO", 40.0, 30.0),
                ("N2", 70.0, "T_NO", 40.0, 40.0),
                # T_LONE: a single VP in D — no sparse floor, labels are
                # purely D-based.
                ("S1", 80.1, "T_LONE", 80.0, 1.0),
            ]
        )
        pairs = build_pairs(df)
        per_target = per_target_metrics(pairs)
        per_target, _, _ = cluster_targets(per_target, radius_km=200.0)
        return proximity_metrics(pairs, per_target).set_index("target_id")

    def test_labels(self) -> None:
        out = self._per_target()
        self.assertEqual(
            out.loc["T_USE", "proximity_label"], "HAS_USED_PROXIMITY"
        )
        self.assertEqual(
            out.loc["T_HAS", "proximity_label"], "HAS_NOT_USED_PROXIMITY"
        )
        self.assertEqual(out.loc["T_NO", "proximity_label"], "NO_PROXIMITY")
        self.assertEqual(
            out.loc["T_LONE", "proximity_label"], "HAS_USED_PROXIMITY"
        )

    def test_discriminative_set_metrics(self) -> None:
        out = self._per_target()
        self.assertEqual(int(out.loc["T_USE", "n_discriminative_vps"]), 2)
        self.assertAlmostEqual(out.loc["T_USE", "best_discriminative_rtt_rank"], 0.0)
        self.assertEqual(int(out.loc["T_HAS", "n_discriminative_vps"]), 1)
        # T_HAS's only discriminative VP is its slowest.
        self.assertAlmostEqual(out.loc["T_HAS", "best_discriminative_rtt_rank"], 1.0)
        self.assertFalse(bool(out.loc["T_HAS", "shortest_ping_vp_is_discriminative"]))
        self.assertEqual(int(out.loc["T_NO", "n_discriminative_vps"]), 0)
        self.assertTrue(
            math.isnan(out.loc["T_NO", "best_discriminative_rtt_rank"])
        )

    def test_single_vp_target_labeled_by_d_only(self) -> None:
        out = self._per_target()
        # One VP is enough: no sparse floor gates the ladder, so a lone
        # discriminative VP that is also the fastest labels HAS_USED.
        self.assertTrue(bool(out.loc["T_LONE", "has_vp_proximity"]))
        self.assertTrue(
            bool(out.loc["T_LONE", "shortest_ping_vp_is_discriminative"])
        )
        self.assertEqual(
            out.loc["T_LONE", "proximity_label"], "HAS_USED_PROXIMITY"
        )

    def test_centroid_distances_match_target_distances_for_singletons(self) -> None:
        out = self._per_target()
        # Singleton clusters: centroid == target, so the centroid-based
        # distances equal the target-based ones.
        self.assertAlmostEqual(
            out.loc["T_USE", "closest_vp_to_centroid_km"],
            out.loc["T_USE", "closest_vp_km"], places=6,
        )
        self.assertAlmostEqual(
            out.loc["T_HAS", "shortest_ping_vp_to_centroid_km"],
            13 * _DEG_KM, places=3,
        )

    def test_voronoi_luck_is_diagnostic_not_label(self) -> None:
        out = self._per_target()
        # H_R lands in T_HAS's own Voronoi cell (13 deg < 15 deg to T_NO) —
        # baseline lucky — yet the label stays HAS_NOT_USED_PROXIMITY.
        self.assertTrue(bool(out.loc["T_HAS", "shortest_ping_vp_in_same_cluster"]))
        self.assertEqual(
            out.loc["T_HAS", "proximity_label"], "HAS_NOT_USED_PROXIMITY"
        )


class TestEvalSourcePrecheckOutputs(unittest.TestCase):
    def _ladder_csv(self, dir_: Path) -> Path:
        df = _equator_frame(
            [
                ("U1", 1.0, "T_USE", 0.0, 3.0),
                ("U2", 5.0, "T_USE", 0.0, 6.0),
                ("H_C", 11.0, "T_HAS", 12.0, 10.0),
                ("H_R", 25.0, "T_HAS", 12.0, 5.0),
                ("N1", 62.0, "T_NO", 40.0, 30.0),
                ("N2", 70.0, "T_NO", 40.0, 40.0),
                ("S1", 80.1, "T_LONE", 80.0, 1.0),
            ]
        )
        path = dir_ / "ladder.csv"
        df.to_csv(path, index=False)
        return path

    def test_proximity_summary_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._ladder_csv(Path(tmp))
            stats = eval_source(csv_path, Path(tmp), cluster_radius_km=200.0)
            per_target = pd.read_csv(stats["per_target_csv"])
            clusters = pd.read_csv(stats["clusters_csv"])
            vp_mesh = pd.read_csv(stats["vp_mesh_csv"], index_col=0)
            cl_mesh = pd.read_csv(stats["cluster_mesh_csv"], index_col=0)
        prox = stats["proximity"]
        # T_USE + T_LONE use proximity, T_HAS has it unused, T_NO lacks it.
        self.assertAlmostEqual(prox["has_used_proximity_share"], 0.5)
        self.assertAlmostEqual(prox["has_not_used_proximity_share"], 0.25)
        self.assertAlmostEqual(prox["no_proximity_share"], 0.25)
        # Opportunity = NO_PROXIMITY + HAS_NOT_USED_PROXIMITY: everywhere
        # the baseline lacks a correctness guarantee (T_NO and T_HAS).
        self.assertAlmostEqual(prox["cbg_opportunity_share"], 0.5)
        # Of the two opportunity targets, T_HAS's fastest VP Voronoi-assigns
        # to its own cell (lucky) while T_NO's assigns to T_LONE's cell.
        self.assertAlmostEqual(prox["opportunity_baseline_lucky_share"], 0.5)
        self.assertIn("proximity_label", per_target.columns)
        self.assertEqual(len(clusters), 4)
        # Mesh: symmetric with a zero diagonal, one row/col per unique VP.
        self.assertEqual(vp_mesh.shape, (7, 7))
        self.assertTrue((np.diag(vp_mesh.to_numpy()) == 0).all())
        self.assertAlmostEqual(
            float(vp_mesh.loc["U1", "U2"]), 4 * _DEG_KM, places=1
        )
        # Answer-space mesh is over cluster centroids, not raw targets —
        # 4 singleton clusters at 0/12/40/80 deg, so the closest centroid
        # pair sits 12 deg apart.
        self.assertEqual(cl_mesh.shape, (4, 4))
        m = cl_mesh.to_numpy()
        self.assertTrue((np.diag(m) == 0).all())
        self.assertAlmostEqual(
            float(m[m > 0].min()), 12 * _DEG_KM, places=1
        )

    def test_meshes_are_uncapped(self) -> None:
        """No size cap: the mesh artifacts are always written, however many
        unique endpoints there are — a pathological size is left to raise
        MemoryError rather than silently shrink or skip the artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._ladder_csv(Path(tmp))
            stats = eval_source(csv_path, Path(tmp), cluster_radius_km=200.0)
            self.assertTrue(Path(stats["vp_mesh_csv"]).exists())
            self.assertTrue(Path(stats["cluster_mesh_csv"]).exists())

    def test_rtt_quality_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._ladder_csv(Path(tmp))
            stats = eval_source(csv_path, Path(tmp), cluster_radius_km=200.0)
        quality = stats["rtt_quality"]
        self.assertIn("pair_soi_violation_share", quality)
        self.assertIn("anycast_suspect_share", quality)
        # T_HAS is the only closest/fastest disagreement among 4 targets.
        self.assertAlmostEqual(quality["closest_is_shortest_ping_share"], 0.75)


# Same fixture as test_sources.py's TestGenericCSVSource_EvalWeightFilter (see
# that class's docstring): t1 mixed (100, 3) -> survives losing 2.2.2.2, t2
# all-light (5) -> dropped, t3 all-heavy (50, 60) -> survives intact, t4
# mixed the other way (8, 12) -> survives losing 1.1.1.1. Reused here (not
# imported) to lock apply_eval_target_filters's math to the exact same
# per-target-survivor outcome the real GenericCSVSource produces.
_EVAL_MASK_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms,weight
    1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0,100
    2.2.2.2,47.0,-122.0,t1,40.0,-100.0,11.0,3
    1.1.1.1,33.0,-84.0,t2,41.0,-101.0,12.0,5
    1.1.1.1,33.0,-84.0,t3,42.0,-102.0,13.0,50
    2.2.2.2,47.0,-122.0,t3,42.0,-102.0,14.0,60
    1.1.1.1,33.0,-84.0,t4,43.0,-103.0,15.0,8
    2.2.2.2,47.0,-122.0,t4,43.0,-103.0,16.0,12
""").strip() + "\n"

_EVAL_MASK_SURVIVORS = {"t1", "t3", "t4"}


class TestApplyEvalTargetFilters(unittest.TestCase):
    def _load(self, tmp: Path, text: str = _EVAL_MASK_CSV) -> pd.DataFrame:
        path = tmp / "mask.csv"
        path.write_text(text)
        return load_canonical_csv(path)

    def test_min_obs_drops_sparse_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp))
            out, _ = apply_eval_target_filters(df, min_obs=2)
        self.assertEqual(set(out["target_id"].unique()), {"t1", "t3", "t4"})

    def test_eval_pair_weight_min_matches_source_survivors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp))
            out, _ = apply_eval_target_filters(df, eval_pair_weight_min=10.0)
        self.assertEqual(set(out["target_id"].unique()), _EVAL_MASK_SURVIVORS)
        # t1 loses its light (2.2.2.2, weight 3) obs; t4 loses 1.1.1.1;
        # t3 keeps both.
        self.assertEqual(
            set(out.loc[out["target_id"] == "t1", "vp_id"]), {"1.1.1.1"}
        )
        self.assertEqual(
            set(out.loc[out["target_id"] == "t4", "vp_id"]), {"2.2.2.2"}
        )
        self.assertEqual((out["target_id"] == "t3").sum(), 2)

    def test_min_obs_runs_before_the_mask(self) -> None:
        """min_obs=2 first drops the single-obs target (t2, which would
        anyway fail the weight mask), then the mask narrows what's left —
        same ordering as GenericCSVSource._ensure_loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp))
            out, _ = apply_eval_target_filters(df, min_obs=2, eval_pair_weight_min=10.0)
        self.assertEqual(set(out["target_id"].unique()), {"t1", "t3", "t4"})

    def test_threshold_dropping_every_target_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp))
            with self.assertRaises(ValueError):
                apply_eval_target_filters(df, eval_pair_weight_min=1000.0)

    def test_mutually_exclusive_args_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp))
            with self.assertRaises(ValueError):
                apply_eval_target_filters(
                    df, eval_pair_weight_min=1.0, eval_kept_traffic_fraction=0.5
                )

    def test_negative_threshold_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp))
            with self.assertRaises(ValueError):
                apply_eval_target_filters(df, eval_pair_weight_min=-1.0)

    def test_invalid_fraction_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp))
            with self.assertRaises(ValueError):
                apply_eval_target_filters(df, eval_kept_traffic_fraction=0.0)

    def test_absent_weight_column_defaults_make_low_thresholds_noop(self) -> None:
        csv = textwrap.dedent("""
            vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms
            1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0
            1.1.1.1,33.0,-84.0,t2,41.0,-101.0,11.0
        """).strip() + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            df = self._load(Path(tmp), csv)
            out, _ = apply_eval_target_filters(df, eval_pair_weight_min=1.0)
        self.assertEqual(set(out["target_id"].unique()), {"t1", "t2"})


# Same fixture as test_sources.py's TestGenericCSVSource_EvalKeptTrafficFraction.
_EVAL_KEPT_FRAC_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,target_city,rtt_ms,weight
    1.1.1.1,33.0,-84.0,t1,40.0,-100.0,atlanta,10.0,10
    2.2.2.2,47.0,-122.0,t1,40.0,-100.0,atlanta,11.0,1
    1.1.1.1,33.0,-84.0,t2,41.0,-101.0,boston,12.0,9
    2.2.2.2,47.0,-122.0,t2,41.0,-101.0,boston,13.0,1
""").strip() + "\n"


class TestEvalKeptTrafficFraction(unittest.TestCase):
    def test_fraction_derives_threshold_matching_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frac.csv"
            path.write_text(_EVAL_KEPT_FRAC_CSV)
            df = load_canonical_csv(path)
            # Per (vp_id,target_city) weights = [10, 1, 9, 1]; 95% -> threshold 1.
            threshold = _derive_eval_pair_weight_min(df, 0.95)
        self.assertEqual(threshold, 1.0)

    def test_zero_total_weight_derives_zero_threshold(self) -> None:
        csv = textwrap.dedent("""
            vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,target_city,rtt_ms,weight
            1.1.1.1,33.0,-84.0,t1,40.0,-100.0,atlanta,10.0,0
            2.2.2.2,47.0,-122.0,t2,41.0,-101.0,boston,11.0,0
        """).strip() + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero.csv"
            path.write_text(csv)
            df = load_canonical_csv(path)
            threshold = _derive_eval_pair_weight_min(df, 0.95)
        self.assertEqual(threshold, 0.0)

    def test_requires_target_city(self) -> None:
        csv = textwrap.dedent("""
            vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms,weight
            1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0,10
        """).strip() + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_city.csv"
            path.write_text(csv)
            df = load_canonical_csv(path)
            with self.assertRaises(ValueError):
                _derive_eval_pair_weight_min(df, 0.95)


class TestEvalSourceFilterIntegration(unittest.TestCase):
    """End-to-end: eval_source() itself narrows to the eval-filtered subset
    and records what it applied."""

    def test_eval_pair_weight_min_shrinks_scored_set_and_records_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mask.csv"
            csv_path.write_text(_EVAL_MASK_CSV)
            stats = eval_source(
                csv_path, Path(tmp), eval_pair_weight_min=10.0,
            )
        self.assertEqual(stats["n_targets"], 3)
        self.assertEqual(
            stats["eval_filters"],
            {
                "min_obs": None,
                "eval_pair_weight_min": 10.0,
                "eval_kept_traffic_fraction": None,
            },
        )

    def test_no_filters_leaves_stats_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mask.csv"
            csv_path.write_text(_EVAL_MASK_CSV)
            stats = eval_source(csv_path, Path(tmp))
        self.assertEqual(stats["n_targets"], 4)
        self.assertNotIn("eval_filters", stats)


if __name__ == "__main__":
    unittest.main()
