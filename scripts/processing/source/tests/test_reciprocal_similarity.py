"""Invariants of the three-stage bipartite similarity report.

The report's job is to make a *contrast* legible, so the things worth pinning are the ones
that would silently flatter the result:

  * the cell-level matrices must be built on the **union** of both footprints -- on the
    intersection, the raw stage would report near-perfect similarity by construction;
  * the outer-product flag must actually fire for a complete mesh, since it is the caveat
    that stops a cosine of 1.0 being read as evidence; and
  * the similarity measures must bottom out and top out correctly, so a saturated metric is
    recognizable as saturated.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules.grid import get_grid
from scripts.processing.source.reciprocal_similarity import (
    cell_biadjacency,
    divergence,
    graph_similarity,
    similarity_report,
)

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DAL = (32.7767, -96.7970)


def _mesh(rows: list[tuple[str, tuple, str, tuple]]) -> pd.DataFrame:
    return pd.DataFrame({
        "vp_id": [r[0] for r in rows],
        "vp_lat": [r[1][0] for r in rows],
        "vp_lon": [r[1][1] for r in rows],
        "target_id": [r[2] for r in rows],
        "target_lat": [r[3][0] for r in rows],
        "target_lon": [r[3][1] for r in rows],
        "rtt_ms": [1.0 + i for i in range(len(rows))],
    })


def _full_mesh(vps: dict, tgs: dict) -> pd.DataFrame:
    return _mesh([(v, vc, t, tc) for v, vc in vps.items() for t, tc in tgs.items()])


class TestGraphSimilarity(unittest.TestCase):
    def test_identical_matrices_are_maximally_similar(self):
        a = np.array([[2.0, 0.0], [1.0, 3.0]])
        s = graph_similarity(a, a.copy())
        self.assertEqual(s["jaccard_support"], 1.0)
        self.assertEqual(s["hamming_norm"], 0.0)
        self.assertEqual(s["cosine"], 1.0)
        self.assertEqual(s["frobenius_rel"], 0.0)

    def test_disjoint_supports_are_maximally_dissimilar(self):
        a = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([[0.0, 0.0], [0.0, 1.0]])
        s = graph_similarity(a, b)
        self.assertEqual(s["jaccard_support"], 0.0)
        self.assertEqual(s["cosine"], 0.0)
        self.assertEqual(s["hamming_norm"], 0.5)

    def test_jaccard_ignores_weights_while_frobenius_does_not(self):
        """Why both are reported: the support can agree exactly while the sizes do not."""
        a = np.array([[1.0, 1.0], [1.0, 1.0]])
        b = a * 10
        s = graph_similarity(a, b)
        self.assertEqual(s["jaccard_support"], 1.0)
        self.assertEqual(s["hamming_norm"], 0.0)
        self.assertEqual(s["cosine"], 1.0)
        self.assertGreater(s["frobenius_rel"], 0.5)


class TestUnionIndex(unittest.TestCase):
    def test_matrices_span_both_footprints_not_their_overlap(self):
        """On the intersection, two disjoint datasets would look identical (both empty)."""
        g = get_grid("h3")
        frames = {
            "public": _mesh([("v1", CHI, "t1", NYC)]),
            "private": _mesh([("q1", SJC, "g1", DAL)]),
        }
        mats, vc, tc = cell_biadjacency(
            frames, grid=g, resolution=g.DEFAULT_RESOLUTION
        )
        self.assertEqual(len(vc), 2)
        self.assertEqual(len(tc), 2)
        self.assertEqual(mats["public"].shape, (2, 2))
        # Each dataset occupies one cell-pair, and they are different ones.
        self.assertEqual(graph_similarity(mats["public"], mats["private"])["jaccard_support"], 0.0)


class TestDivergence(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence(self):
        a = np.array([1.0, 2.0, 3.0])
        d = divergence(a, a.copy(), unit="km")
        self.assertEqual(d["w1"], 0.0)
        self.assertEqual(d["ks"], 0.0)
        self.assertEqual(d["unit"], "km")

    def test_a_shift_shows_up_in_w1_in_native_units(self):
        a = np.zeros(50)
        d = divergence(a, a + 7.0, unit="km")
        self.assertAlmostEqual(d["w1"], 7.0, places=3)

    def test_non_finite_values_are_dropped_not_propagated(self):
        a = np.array([1.0, 2.0, np.nan, np.inf])
        d = divergence(a, np.array([1.0, 2.0]), unit="km")
        self.assertEqual(d["w1"], 0.0)


class TestReport(unittest.TestCase):
    def setUp(self):
        self.g = get_grid("h3")
        # Public: 1 VP, 1 target. Private: same places, but 3 co-located targets -- the
        # real shape of the operator mesh against RIPE anchors.
        self.pub = _full_mesh({"v1": CHI}, {"t1": NYC})
        self.priv = _full_mesh(
            {"q1": CHI},
            {"g1": NYC, "g2": (NYC[0] + 0.01, NYC[1]), "g3": (NYC[0] + 0.02, NYC[1])},
        )

    def _report(self):
        return similarity_report(
            self.pub, self.priv, grid=self.g, resolution=self.g.DEFAULT_RESOLUTION
        )

    def test_all_three_stages_are_measured(self):
        r = self._report()
        self.assertEqual(set(r["stages"]), {"raw", "reciprocal", "balanced"})

    def test_balancing_equalizes_node_counts_by_the_final_stage(self):
        n = self._report()["stages"]["balanced"]["nodes"]
        self.assertEqual(n["public"]["vps"], n["private"]["vps"])
        self.assertEqual(n["public"]["targets"], n["private"]["targets"])

    def test_size_agreement_improves_from_reciprocal_to_balanced(self):
        """What the balancing stage is for: the weighted measures, not the support ones."""
        s = self._report()["stages"]
        self.assertGreater(
            s["reciprocal"]["cell_graph"]["frobenius_rel"],
            s["balanced"]["cell_graph"]["frobenius_rel"],
        )

    def test_a_complete_mesh_is_flagged_as_a_forced_outer_product(self):
        """The caveat that keeps a cosine of 1.0 from being over-read."""
        flags = self._report()["stages"]["balanced"]["cell_graph"][
            "weights_are_forced_outer_product"
        ]
        self.assertTrue(flags["public"])
        self.assertTrue(flags["private"])

    def test_an_incomplete_mesh_is_not_flagged(self):
        pub = _full_mesh({"v1": CHI, "v2": SJC}, {"t1": NYC, "t2": DAL})
        pub = pub.drop(index=0).reset_index(drop=True)   # punch one edge out
        priv = _full_mesh({"q1": CHI, "q2": SJC}, {"g1": NYC, "g2": DAL})
        r = similarity_report(pub, priv, grid=self.g, resolution=self.g.DEFAULT_RESOLUTION)
        flags = r["stages"]["balanced"]["cell_graph"]["weights_are_forced_outer_product"]
        self.assertFalse(flags["public"])
        self.assertTrue(flags["private"])

    def test_residual_geometry_covers_the_uncontrolled_axes(self):
        res = self._report()["stages"]["balanced"]["residual_geometry"]
        self.assertEqual(
            set(res),
            {"nearest_measured_vp_km", "nearest_measured_target_km", "edge_length_km",
             "max_angular_gap_deg", "circular_variance"},
        )

    def test_the_ladder_reports_every_rung(self):
        lad = self._report()["stages"]["balanced"]["occupied_cell_ladder"]
        for name in ("public", "private"):
            for side in ("vp", "target"):
                self.assertEqual(set(lad[name][side]), {"5", "4", "3", "2"})


if __name__ == "__main__":
    unittest.main()
