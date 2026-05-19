"""Tests for the v2 Shapely geometric centroid."""

from __future__ import annotations

import unittest

from shapely.geometry import Point

from scripts.framework.v2.ctr.geometric_centroid import GeometricCentroidCTR
from scripts.framework.v2.ctr.tests.helpers import (
    assert_coord_almost_equal,
    failed_vertices,
    rectangle_region,
    successful_region,
    successful_vertices,
    unequal_area_multipolygon,
)
from scripts.framework.v2.types import Error


class TestGeometricCentroidCTR(unittest.TestCase):
    def test_polygon_region_returns_area_centroid(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(successful_region(rectangle_region()))

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        assert_coord_almost_equal(self, result.tg_coord, (1.0, 2.0))

    def test_multipolygon_region_returns_area_weighted_centroid(self):
        region = unequal_area_multipolygon()
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(successful_region(region))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (1.0, 25.0 / 3.0))
        self.assertFalse(region.contains(Point(result.tg_coord.lon, result.tg_coord.lat)))

    def test_unordered_vertex_list_returns_ordered_polygon_area_centroid(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(
            successful_vertices([
                (2.0, 4.0),
                (0.0, 0.0),
                (2.0, 0.0),
                (0.0, 4.0),
            ])
        )

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (1.0, 2.0))

    def test_vertex_list_deduplicates_near_identical_crossings(self):
        ctr = GeometricCentroidCTR(dedupe_tolerance_deg=1e-6)

        result = ctr.select_centroid(
            successful_vertices([
                (0.0, 0.0),
                (0.0, 2.0),
                (2.0, 0.0),
                (0.0, 2.0 + 1e-8),
            ])
        )

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (2.0 / 3.0, 2.0 / 3.0))

    def test_two_vertex_list_returns_geodetic_midpoint(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(successful_vertices([(0.0, 0.0), (0.0, 2.0)]))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (0.0, 1.0))

    def test_single_vertex_list_returns_that_vertex(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (3.0, -7.0))

    def test_collinear_vertex_list_returns_degenerate_region(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(
            successful_vertices([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.tg_coord)
        self.assertEqual(result.error, Error.DEGENERATE_REGION)

    def test_dateline_vertex_list_uses_wrapped_longitude_ordering(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(
            successful_vertices([
                (-1.0, 179.0),
                (1.0, -179.0),
                (1.0, 179.0),
                (-1.0, -179.0),
            ])
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.tg_coord)
        self.assertAlmostEqual(result.tg_coord.lat, 0.0, places=7)
        self.assertAlmostEqual(abs(result.tg_coord.lon), 180.0, places=7)

    def test_failed_result_carries_error(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(failed_vertices([(1.0, 2.0)]))

        self.assertFalse(result.success)
        self.assertIsNone(result.tg_coord)
        self.assertEqual(result.error, Error.EMPTY_REGION)

    def test_method_field_is_stamped(self):
        ctr = GeometricCentroidCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertEqual(result.method, "GeometricCentroidCTR")


if __name__ == "__main__":
    unittest.main()
