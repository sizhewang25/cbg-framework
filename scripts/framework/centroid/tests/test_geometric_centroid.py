"""Tests for the Shapely geometric centroid."""

from __future__ import annotations

import unittest

from shapely.geometry import Point

from scripts.framework.centroid.geometric_centroid import GeometricCentroid
from scripts.framework.centroid.tests.helpers import (
    assert_point_almost_equal,
    failed_vertices,
    rectangle_region,
    successful_region,
    successful_vertices,
    unequal_area_multipolygon,
)


class TestGeometricCentroid(unittest.TestCase):
    def test_polygon_region_returns_area_centroid(self):
        centroid = GeometricCentroid()

        point = centroid.select(successful_region(rectangle_region()))

        self.assertEqual(point, (1.0, 2.0))

    def test_multipolygon_region_returns_area_weighted_centroid(self):
        region = unequal_area_multipolygon()
        centroid = GeometricCentroid()

        point = centroid.select(successful_region(region))

        assert_point_almost_equal(self, point, (1.0, 25.0 / 3.0))
        self.assertFalse(region.contains(Point(point[1], point[0])))

    def test_vertex_lists_are_not_converted_to_polygons(self):
        centroid = GeometricCentroid()

        point = centroid.select(
            successful_vertices([(0.0, 0.0), (0.0, 2.0), (2.0, 0.0)])
        )

        self.assertIsNone(point)

    def test_failed_result_returns_none(self):
        centroid = GeometricCentroid()

        self.assertIsNone(centroid.select(failed_vertices([(1.0, 2.0)])))


if __name__ == "__main__":
    unittest.main()
