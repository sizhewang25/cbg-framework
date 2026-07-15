"""cluster_ground_truth: duplicate-coordinate handling.

Clustering is done over unique (lat, lon) rows only (memory optimization —
see cluster_ground_truth.py docstring), then broadcast back to every input
point. These tests pin the two things that broadcast must get right:
member counts/singleton flags must reflect true input multiplicity (not
unique-location count), and the spherical centroid must be weighted by that
same multiplicity.
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.benchmark.v2.sources.cluster_ground_truth import cluster_ground_truth


class TestDuplicateCoordinates(unittest.TestCase):
    def test_member_count_reflects_multiplicity_not_unique_locations(self) -> None:
        # 5 targets at the exact same point -> one cluster, 5 members, not a
        # singleton even though there's only 1 unique location.
        lats = np.array([10.0] * 5)
        lons = np.array([20.0] * 5)
        res = cluster_ground_truth(lats, lons, radius_km=50.0)
        self.assertEqual(res.n_clusters, 1)
        self.assertEqual(int(res.member_counts[0]), 5)
        self.assertFalse(bool(res.member_counts[0] == 1))
        np.testing.assert_allclose(res.dist_km, 0.0, atol=1e-9)

    def test_centroid_is_weighted_by_full_multiplicity(self) -> None:
        # A at (0, 0) x3, B at (0, 1) x1 -> both well within one 50km-ish
        # region (they're ~111km apart though, so use a large radius to force
        # one cluster and isolate the centroid-weighting question).
        lats = np.array([0.0, 0.0, 0.0, 0.0])
        lons = np.array([0.0, 0.0, 0.0, 1.0])
        res = cluster_ground_truth(lats, lons, radius_km=500.0)
        self.assertEqual(res.n_clusters, 1)
        self.assertEqual(int(res.member_counts[0]), 4)
        # Weighted mean longitude over (0,0,0,1) is 0.25, not the unique-only
        # mean of (0,1) = 0.5.
        self.assertAlmostEqual(float(res.centroid_lon[0]), 0.25, places=3)

    def test_duplicates_do_not_change_cluster_membership_vs_no_duplicates(self) -> None:
        # Two far-apart singleton locations; adding duplicate copies of one
        # of them must not merge/split anything differently.
        base_lats = np.array([0.0, 10.0])
        base_lons = np.array([0.0, 10.0])
        res_base = cluster_ground_truth(base_lats, base_lons, radius_km=50.0)
        self.assertEqual(res_base.n_clusters, 2)

        dup_lats = np.array([0.0, 0.0, 0.0, 10.0])
        dup_lons = np.array([0.0, 0.0, 0.0, 10.0])
        res_dup = cluster_ground_truth(dup_lats, dup_lons, radius_km=50.0)
        self.assertEqual(res_dup.n_clusters, 2)
        counts = sorted(int(c) for c in res_dup.member_counts)
        self.assertEqual(counts, [1, 3])

    def test_diameter_uses_unique_rows_only(self) -> None:
        # Duplicates at distance 0 must not change the reported diameter.
        lats = np.array([0.0, 0.0, 0.0, 1.0])
        lons = np.array([0.0, 0.0, 0.0, 0.0])
        res = cluster_ground_truth(lats, lons, radius_km=500.0)
        self.assertEqual(res.n_clusters, 1)
        expected_deg_km = 1.0 * np.pi / 180 * 6371.0088
        self.assertAlmostEqual(float(res.diameter_km[0]), expected_deg_km, places=1)


if __name__ == "__main__":
    unittest.main()
