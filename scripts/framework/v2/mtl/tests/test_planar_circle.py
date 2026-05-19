"""Tests for PlanarCircleMTL.

Ports scripts/framework/multilateration/tests/test_planar_circle.py.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.mtl.base import CircleMTLMethod
from scripts.framework.v2.mtl.planar_circle import PlanarCircleMTL
from scripts.framework.v2.mtl.tests.helpers import ltd_result
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Error


class TestPlanarCircleMTL(unittest.TestCase):
    def test_empty_input_fails_with_insufficient_data(self):
        result = PlanarCircleMTL(n_pts=4).multilaterate([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)
        self.assertIsNone(result.intersection)
        self.assertEqual(result.method, "PlanarCircleMTL")

    def test_single_equator_circle_becomes_unit_diamond(self):
        result = PlanarCircleMTL(n_pts=4).multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        self.assertAlmostEqual(result.intersection.area, 2.0, places=6)
        self.assertEqual(result.intersection.bounds, (-1.0, -1.0, 1.0, 1.0))

    def test_two_overlapping_circles_intersect_to_half_diamond(self):
        result = PlanarCircleMTL(n_pts=4).multilaterate([
            ltd_result("left", lat=0.0, lon=0.0, upper_km=111.0),
            ltd_result("right", lat=0.0, lon=1.0, upper_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        self.assertAlmostEqual(result.intersection.area, 0.5, places=6)
        self.assertEqual(result.intersection.bounds, (0.0, -0.5, 1.0, 0.5))

    def test_three_overlapping_circles_tighten_to_centered_diamond(self):
        result = PlanarCircleMTL(n_pts=4).multilaterate([
            ltd_result("left", lat=0.0, lon=0.0, upper_km=111.0),
            ltd_result("right", lat=0.0, lon=1.0, upper_km=111.0),
            ltd_result("center", lat=0.0, lon=0.5, upper_km=44.4),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        self.assertAlmostEqual(result.intersection.area, 0.32, places=6)
        for actual, want in zip(result.intersection.bounds, (0.1, -0.4, 0.9, 0.4)):
            self.assertAlmostEqual(actual, want, places=6)

    def test_disjoint_circles_fail_with_empty_region(self):
        result = PlanarCircleMTL(n_pts=4).multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=111.0),
            ltd_result("b", lat=0.0, lon=10.0, upper_km=111.0),
        ])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.EMPTY_REGION)
        self.assertIsNone(result.intersection)

    def test_registered_in_mtl_registry(self):
        self.assertIn("planar_circle", MTL_REGISTRY)
        self.assertIs(MTL_REGISTRY["planar_circle"], PlanarCircleMTL)

    def test_is_circle_family(self):
        self.assertTrue(issubclass(PlanarCircleMTL, CircleMTLMethod))


if __name__ == "__main__":
    unittest.main()
