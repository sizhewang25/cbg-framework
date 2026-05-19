"""Tests for the v2 boundary-vertex mean centroid."""

from __future__ import annotations

import unittest

from scripts.framework.v2.ctr.boundary_vertex_mean import BoundaryVertexMeanCTR
from scripts.framework.v2.ctr.tests.helpers import (
    assert_coord_almost_equal,
    failed_vertices,
    polygon_with_asymmetric_hole,
    rectangle_region,
    successful_region,
    successful_vertices,
    two_rectangle_multipolygon,
    unequal_area_multipolygon,
)
from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.types import Error


class TestBoundaryVertexMeanCTR(unittest.TestCase):
    def test_vertex_list_uses_coordinate_mean_for_three_or_more_vertices(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(
            successful_vertices([(0.0, 0.0), (2.0, 4.0), (4.0, 8.0)])
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        assert_coord_almost_equal(self, result.tg_coord, (2.0, 4.0))

    def test_two_vertices_use_arithmetic_mean(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(successful_vertices([(0.0, 0.0), (0.0, 2.0)]))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (0.0, 1.0))

    def test_single_vertex_returns_that_vertex(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (3.0, -7.0))

    def test_polygon_region_uses_exterior_vertices_without_closing_duplicate(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(successful_region(rectangle_region()))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (1.0, 2.0))

    def test_polygon_region_includes_interior_ring_vertices(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(successful_region(polygon_with_asymmetric_hole()))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (1.75, 2.25))

    def test_multipolygon_region_combines_all_polygon_boundaries(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(successful_region(two_rectangle_multipolygon()))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (1.0, 6.0))

    def test_unequal_area_multipolygon_is_boundary_vertex_weighted(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(successful_region(unequal_area_multipolygon()))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (1.0, 6.5))

    def test_failed_or_empty_results_carry_error(self):
        ctr = BoundaryVertexMeanCTR()

        for mtl in (
            failed_vertices([(1.0, 2.0)]),
            MTLResult(success=True, intersection=[]),
            MTLResult(success=True, intersection=None),
        ):
            result = ctr.select_centroid(mtl)
            self.assertFalse(result.success)
            self.assertIsNone(result.tg_coord)
            self.assertEqual(result.error, Error.EMPTY_REGION)

    def test_method_field_is_stamped(self):
        ctr = BoundaryVertexMeanCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertEqual(result.method, "BoundaryVertexMeanCTR")


if __name__ == "__main__":
    unittest.main()
