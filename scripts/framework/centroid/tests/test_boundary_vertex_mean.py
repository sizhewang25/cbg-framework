"""Tests for the boundary-vertex mean centroid."""

from __future__ import annotations

import unittest

from scripts.framework.centroid.boundary_vertex_mean import BoundaryVertexMeanCentroid
from scripts.framework.centroid.tests.helpers import (
    assert_point_almost_equal,
    failed_vertices,
    polygon_with_asymmetric_hole,
    rectangle_region,
    successful_region,
    successful_vertices,
    two_rectangle_multipolygon,
    unequal_area_multipolygon,
)
from scripts.framework.types import MultilatResult


class TestBoundaryVertexMeanCentroid(unittest.TestCase):
    def test_vertex_list_uses_coordinate_mean_for_three_or_more_vertices(self):
        centroid = BoundaryVertexMeanCentroid()

        point = centroid.select(
            successful_vertices([(0.0, 0.0), (2.0, 4.0), (4.0, 8.0)])
        )

        self.assertEqual(point, (2.0, 4.0))

    def test_two_vertices_use_geodetic_midpoint(self):
        centroid = BoundaryVertexMeanCentroid()

        point = centroid.select(successful_vertices([(0.0, 0.0), (0.0, 2.0)]))

        assert_point_almost_equal(self, point, (0.0, 1.0))

    def test_single_vertex_returns_that_vertex(self):
        centroid = BoundaryVertexMeanCentroid()

        point = centroid.select(successful_vertices([(3.0, -7.0)]))

        self.assertEqual(point, (3.0, -7.0))

    def test_polygon_region_uses_exterior_vertices_without_closing_duplicate(self):
        centroid = BoundaryVertexMeanCentroid()

        point = centroid.select(successful_region(rectangle_region()))

        self.assertEqual(point, (1.0, 2.0))

    def test_polygon_region_includes_interior_ring_vertices(self):
        centroid = BoundaryVertexMeanCentroid()

        point = centroid.select(successful_region(polygon_with_asymmetric_hole()))

        self.assertEqual(point, (1.75, 2.25))

    def test_multipolygon_region_combines_all_polygon_boundaries(self):
        centroid = BoundaryVertexMeanCentroid()

        point = centroid.select(successful_region(two_rectangle_multipolygon()))

        self.assertEqual(point, (1.0, 6.0))

    def test_unequal_area_multipolygon_is_boundary_vertex_weighted(self):
        centroid = BoundaryVertexMeanCentroid()

        point = centroid.select(successful_region(unequal_area_multipolygon()))

        self.assertEqual(point, (1.0, 6.5))

    def test_failed_or_empty_results_return_none(self):
        centroid = BoundaryVertexMeanCentroid()

        self.assertIsNone(centroid.select(failed_vertices([(1.0, 2.0)])))
        self.assertIsNone(centroid.select(MultilatResult(vertices=[], success=True)))
        self.assertIsNone(centroid.select(MultilatResult(success=True)))


if __name__ == "__main__":
    unittest.main()
