"""Tests for SphericalCircleMTL.

Ports scripts/framework/multilateration/tests/test_spherical_circle.py to the
v2 interface: LTDResult input, MTLResult output with a single `intersection`
field (`list[Coord]` for spherical methods).
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.mtl.base import CircleMTLMethod
from scripts.framework.v2.mtl.spherical_circle import SphericalCircleMTL
from scripts.framework.v2.mtl.tests.helpers import ltd_result
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Coord, Error

CAP_RADIUS_KM = 138.9936583057
TWO_CAP_LAT_DEG = 1.1456584648


class TestSphericalCircleMTL(unittest.TestCase):
    def test_empty_input_fails_with_insufficient_data(self):
        result = SphericalCircleMTL().multilaterate([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)
        self.assertIsNone(result.intersection)
        self.assertEqual(result.method, "SphericalCircleMTL")

    def test_single_circle_returns_four_cardinal_vertices(self):
        result = SphericalCircleMTL().multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(len(result.intersection), 4)
        expected = [(0.0, 0.997), (0.997, 0.0), (0.0, -0.997), (-0.997, 0.0)]
        for actual, want in zip(result.intersection, expected):
            self.assertIsInstance(actual, Coord)
            self.assertAlmostEqual(actual.lat, want[0], places=3)
            self.assertAlmostEqual(actual.lon, want[1], places=3)

    def test_two_overlapping_circles_return_two_crossing_vertices(self):
        result = SphericalCircleMTL().multilaterate([
            ltd_result("west", lat=0.0, lon=0.0, upper_km=CAP_RADIUS_KM),
            ltd_result("east", lat=0.0, lon=1.0, upper_km=CAP_RADIUS_KM),
        ])

        self.assertTrue(result.success)
        self.assertEqual(len(result.intersection), 2)
        verts = sorted([(c.lat, c.lon) for c in result.intersection])
        self.assertAlmostEqual(verts[0][0], -TWO_CAP_LAT_DEG, places=5)
        self.assertAlmostEqual(verts[0][1], 0.5, places=5)
        self.assertAlmostEqual(verts[1][0], TWO_CAP_LAT_DEG, places=5)
        self.assertAlmostEqual(verts[1][1], 0.5, places=5)

    def test_three_circles_filter_to_northern_crossing_vertex(self):
        result = SphericalCircleMTL().multilaterate([
            ltd_result("west", lat=0.0, lon=0.0, upper_km=CAP_RADIUS_KM),
            ltd_result("east", lat=0.0, lon=1.0, upper_km=CAP_RADIUS_KM),
            ltd_result("north", lat=TWO_CAP_LAT_DEG, lon=0.5, upper_km=100.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(len(result.intersection), 1)
        self.assertAlmostEqual(result.intersection[0].lat, TWO_CAP_LAT_DEG, places=5)
        self.assertAlmostEqual(result.intersection[0].lon, 0.5, places=5)

    def test_non_intersecting_circles_fail_with_no_intersection(self):
        result = SphericalCircleMTL().multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=50.0),
            ltd_result("b", lat=0.0, lon=10.0, upper_km=50.0),
        ])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.NO_INTERSECTION)
        self.assertIsNone(result.intersection)

    def test_filter_disabled_keeps_redundant_outer_and_yields_no_crossings(self):
        # Small disk fully inside a much larger disk: boundaries never cross,
        # so without preprocessing the pairwise loop produces no points.
        nested = [
            ltd_result("inner", lat=0.0, lon=0.0, upper_km=111.0),
            ltd_result("outer", lat=0.0, lon=0.5, upper_km=500.0),
        ]
        result = SphericalCircleMTL(enable_circle_filter=False).multilaterate(nested)

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.NO_INTERSECTION)

    def test_filter_enabled_drops_redundant_outer_circle(self):
        # Same nested pair as above. With the filter on, the redundant outer
        # disk is removed and the surviving inner disk returns its 4 cardinals.
        nested = [
            ltd_result("inner", lat=0.0, lon=0.0, upper_km=111.0),
            ltd_result("outer", lat=0.0, lon=0.5, upper_km=500.0),
            ltd_result("outer", lat=0.0, lon=-0.5, upper_km=1000.0),
        ]
        result = SphericalCircleMTL(enable_circle_filter=True).multilaterate(nested)

        self.assertTrue(result.success)
        self.assertEqual(len(result.intersection), 4)
        expected = [(0.0, 0.997), (0.997, 0.0), (0.0, -0.997), (-0.997, 0.0)]
        for actual, want in zip(result.intersection, expected):
            self.assertAlmostEqual(actual.lat, want[0], places=3)
            self.assertAlmostEqual(actual.lon, want[1], places=3)

    def test_registered_in_mtl_registry(self):
        self.assertIn("spherical_circle", MTL_REGISTRY)
        self.assertIs(MTL_REGISTRY["spherical_circle"], SphericalCircleMTL)

    def test_is_circle_family(self):
        self.assertTrue(issubclass(SphericalCircleMTL, CircleMTLMethod))


if __name__ == "__main__":
    unittest.main()
