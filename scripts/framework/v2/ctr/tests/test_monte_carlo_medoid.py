"""Tests for the v2 Monte Carlo sampled-medoid centroid."""

from __future__ import annotations

import unittest

from shapely.geometry import Point

from scripts.framework.v2.ctr.monte_carlo_medoid import MonteCarloMedoidCTR
from scripts.framework.v2.ctr.tests.helpers import (
    assert_coord_almost_equal,
    failed_vertices,
    rectangle_region,
    successful_region,
    successful_vertices,
    unequal_area_multipolygon,
)
from scripts.framework.v2.types import Error


class TestMonteCarloMedoidCTR(unittest.TestCase):
    def test_vertex_list_selects_sampled_medoid(self):
        ctr = MonteCarloMedoidCTR()

        result = ctr.select_centroid(
            successful_vertices([(0.0, 0.0), (0.0, 1.0), (0.0, 10.0)])
        )

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (0.0, 1.0))

    def test_single_vertex_returns_that_vertex(self):
        ctr = MonteCarloMedoidCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (3.0, -7.0))

    def test_region_sampling_returns_feasible_point(self):
        region = rectangle_region()
        ctr = MonteCarloMedoidCTR(n_samples=8, seed=7)

        result = ctr.select_centroid(successful_region(region))

        self.assertTrue(result.success)
        self.assertTrue(region.contains(Point(result.tg_coord.lon, result.tg_coord.lat)))

    def test_multipolygon_region_sampling_returns_feasible_point(self):
        region = unequal_area_multipolygon()
        ctr = MonteCarloMedoidCTR(n_samples=32, seed=7)

        result = ctr.select_centroid(successful_region(region))

        self.assertTrue(result.success)
        self.assertTrue(region.contains(Point(result.tg_coord.lon, result.tg_coord.lat)))

    def test_zero_region_samples_falls_back_to_representative_point(self):
        region = rectangle_region()
        ctr = MonteCarloMedoidCTR(n_samples=0, seed=7)

        result = ctr.select_centroid(successful_region(region))

        self.assertTrue(result.success)
        self.assertTrue(region.contains(Point(result.tg_coord.lon, result.tg_coord.lat)))

    def test_failed_or_empty_results_carry_error(self):
        ctr = MonteCarloMedoidCTR()

        for mtl in (failed_vertices([(1.0, 2.0)]), successful_vertices([])):
            result = ctr.select_centroid(mtl)
            self.assertFalse(result.success)
            self.assertIsNone(result.tg_coord)
            self.assertEqual(result.error, Error.EMPTY_REGION)

    def test_method_field_is_stamped(self):
        ctr = MonteCarloMedoidCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertEqual(result.method, "MonteCarloMedoidCTR")


if __name__ == "__main__":
    unittest.main()
