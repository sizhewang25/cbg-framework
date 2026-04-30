"""Tests for the Monte Carlo sampled-medoid centroid."""

from __future__ import annotations

import unittest

from shapely.geometry import Point

from scripts.framework.centroid.monte_carlo_median import MonteCarloMedianCentroid
from scripts.framework.centroid.tests.helpers import (
    failed_vertices,
    rectangle_region,
    successful_region,
    successful_vertices,
    unequal_area_multipolygon,
)


class TestMonteCarloMedianCentroid(unittest.TestCase):
    def test_vertex_list_selects_sampled_medoid(self):
        centroid = MonteCarloMedianCentroid()

        point = centroid.select(
            successful_vertices([(0.0, 0.0), (0.0, 1.0), (0.0, 10.0)])
        )

        self.assertEqual(point, (0.0, 1.0))

    def test_single_vertex_returns_that_vertex(self):
        centroid = MonteCarloMedianCentroid()

        point = centroid.select(successful_vertices([(3.0, -7.0)]))

        self.assertEqual(point, (3.0, -7.0))

    def test_region_sampling_returns_feasible_point(self):
        region = rectangle_region()
        centroid = MonteCarloMedianCentroid(n_samples=8, seed=7)

        lat, lon = centroid.select(successful_region(region))

        self.assertTrue(region.contains(Point(lon, lat)))

    def test_multipolygon_region_sampling_returns_feasible_point(self):
        region = unequal_area_multipolygon()
        centroid = MonteCarloMedianCentroid(n_samples=32, seed=7)

        lat, lon = centroid.select(successful_region(region))

        self.assertTrue(region.contains(Point(lon, lat)))

    def test_zero_region_samples_falls_back_to_representative_point(self):
        region = rectangle_region()
        centroid = MonteCarloMedianCentroid(n_samples=0, seed=7)

        lat, lon = centroid.select(successful_region(region))

        self.assertTrue(region.contains(Point(lon, lat)))

    def test_failed_or_empty_results_return_none(self):
        centroid = MonteCarloMedianCentroid()

        self.assertIsNone(centroid.select(failed_vertices([(1.0, 2.0)])))
        self.assertIsNone(centroid.select(successful_vertices([])))


if __name__ == "__main__":
    unittest.main()
