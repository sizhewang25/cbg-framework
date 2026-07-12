from __future__ import annotations

import unittest

import numpy as np

from scripts.benchmark.v2.cluster_topn import (
    build_truth_neighbor_index,
    compute_topk_match_from_truth_neighbors,
    resolve_effective_top_n,
)


class TestResolveEffectiveTopN(unittest.TestCase):
    def test_caps_by_centroid_count(self) -> None:
        self.assertEqual(resolve_effective_top_n(10, 3), 3)

    def test_rejects_non_positive_top_n(self) -> None:
        with self.assertRaises(ValueError):
            resolve_effective_top_n(0, 3)


class TestBuildTruthNeighborIndex(unittest.TestCase):
    def test_shape_and_self_first(self) -> None:
        lat = np.array([0.0, 0.0, 1.0], dtype=float)
        lon = np.array([0.0, 1.0, 0.0], dtype=float)
        nn = build_truth_neighbor_index(lat, lon, top_n=3)
        self.assertEqual(nn.shape, (3, 3))
        self.assertTrue((nn[:, 0] == np.array([0, 1, 2])).all())


class TestComputeTopKMatch(unittest.TestCase):
    def setUp(self) -> None:
        # 4 centroids arranged so nearest-neighbor ordering is deterministic.
        lat = np.array([0.0, 0.0, 1.0, 4.0], dtype=float)
        lon = np.array([0.0, 1.0, 0.0, 4.0], dtype=float)
        self.nn = build_truth_neighbor_index(lat, lon, top_n=4)

    def test_top1_equivalence_to_exact_centroid_match(self) -> None:
        pred = np.array([0, 1, 2, 3, 1], dtype=int)
        truth = np.array([0, 1, 3, 3, 0], dtype=int)
        out = compute_topk_match_from_truth_neighbors(pred, truth, self.nn, top_n=1)
        expected = pred == truth
        self.assertTrue((out["match_top1"] == expected).all())

    def test_monotonicity(self) -> None:
        pred = np.array([1, 2, 3, 0, 2], dtype=int)
        truth = np.array([0, 0, 0, 3, 1], dtype=int)
        out = compute_topk_match_from_truth_neighbors(pred, truth, self.nn, top_n=4)
        m1, m2, m3, m4 = out["match_top1"], out["match_top2"], out["match_top3"], out["match_top4"]
        self.assertTrue((m1 <= m2).all())
        self.assertTrue((m2 <= m3).all())
        self.assertTrue((m3 <= m4).all())

    def test_pred_inside_truth_topk_counts_correct(self) -> None:
        # truth centroid 0 has neighbors [0,1,2,...], so pred=1 should be true at K>=2
        pred = np.array([1], dtype=int)
        truth = np.array([0], dtype=int)
        out = compute_topk_match_from_truth_neighbors(pred, truth, self.nn, top_n=3)
        self.assertFalse(bool(out["match_top1"][0]))
        self.assertTrue(bool(out["match_top2"][0]))
        self.assertTrue(bool(out["match_top3"][0]))

    def test_pred_outside_truth_topk_counts_incorrect(self) -> None:
        # truth centroid 0 and pred=3 should be false for small K.
        pred = np.array([3], dtype=int)
        truth = np.array([0], dtype=int)
        out = compute_topk_match_from_truth_neighbors(pred, truth, self.nn, top_n=2)
        self.assertFalse(bool(out["match_top1"][0]))
        self.assertFalse(bool(out["match_top2"][0]))

    def test_invalid_ids_are_false(self) -> None:
        pred = np.array([-1, 100, 0], dtype=int)
        truth = np.array([0, 1, -1], dtype=int)
        out = compute_topk_match_from_truth_neighbors(pred, truth, self.nn, top_n=3)
        for k in (1, 2, 3):
            self.assertFalse(bool(out[f"match_top{k}"][0]))
            self.assertFalse(bool(out[f"match_top{k}"][1]))
            self.assertFalse(bool(out[f"match_top{k}"][2]))


if __name__ == "__main__":
    unittest.main()
