from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.benchmark.v2.modules.adjacency_metrics import (
    adjacency_concentration_from_representative,
    voronoi_adjacency_distances,
)


class TestVoronoiAdjacencyDistances(unittest.TestCase):
    def test_two_points_are_mutual_neighbors(self) -> None:
        lat = np.array([0.0, 0.0])
        lon = np.array([0.0, 1.0])

        out = voronoi_adjacency_distances(lat, lon)

        self.assertEqual(len(out), 2)
        self.assertEqual(len(out.loc[0, "adjacent_distances_km"]), 1)
        self.assertEqual(len(out.loc[1, "adjacent_distances_km"]), 1)
        self.assertAlmostEqual(out.loc[0, "mean_dist_km"], out.loc[1, "mean_dist_km"], places=6)
        self.assertAlmostEqual(out.loc[0, "median_dist_km"], out.loc[1, "median_dist_km"], places=6)

    def test_empty_input(self) -> None:
        out = voronoi_adjacency_distances(np.array([]), np.array([]))
        self.assertEqual(list(out.columns), ["adjacent_distances_km", "mean_dist_km", "median_dist_km"])
        self.assertEqual(len(out), 0)


class TestAdjacencyConcentrationFromRepresentative(unittest.TestCase):
    def test_weighted_concentration_from_representative_distances(self) -> None:
        # Cluster 0 has many targets in a denser zone (smaller representative
        # distance), so concentration should be positive.
        cluster_ids = np.array([0, 1, 2])
        n_members = pd.Series([50, 10, 5], index=pd.Index([0, 1, 2], name="cluster_id"))
        representative = pd.Series([100.0, 200.0, 300.0])

        out = adjacency_concentration_from_representative(cluster_ids, n_members, representative)

        self.assertAlmostEqual(out["a_cluster_mean_km"], 200.0, places=3)
        self.assertAlmostEqual(out["a_cluster_std_km"], np.std([100.0, 200.0, 300.0]), places=3)
        self.assertAlmostEqual(out["a_wgt_cluster_mean_km"], 130.769, places=3)
        self.assertAlmostEqual(out["r_concentration"], 0.653846, places=6)
        self.assertAlmostEqual(out["tg_concentration_index"], 0.346154, places=6)
        self.assertEqual(out["valid_n_clusters"], 3)
        self.assertEqual(out["total_targets_used"], 65)

    def test_no_valid_clusters(self) -> None:
        cluster_ids = np.array([0, 1])
        n_members = pd.Series([0, 0], index=pd.Index([0, 1], name="cluster_id"))
        representative = pd.Series([np.nan, np.nan])

        out = adjacency_concentration_from_representative(cluster_ids, n_members, representative)

        self.assertIsNone(out["a_cluster_mean_km"])
        self.assertIsNone(out["a_cluster_std_km"])
        self.assertIsNone(out["a_wgt_cluster_mean_km"])
        self.assertIsNone(out["r_concentration"])
        self.assertIsNone(out["tg_concentration_index"])
        self.assertEqual(out["valid_n_clusters"], 0)
        self.assertEqual(out["total_targets_used"], 0)


if __name__ == "__main__":
    unittest.main()
