"""Tests for PlanarAnnulusMTL.

Ports scripts/framework/multilateration/tests/test_planar_annulus.py.
"""

from __future__ import annotations

import unittest

from shapely.geometry import Point

from scripts.framework.v2.mtl.base import AnnulusMTLMethod
from scripts.framework.v2.mtl.planar_annulus import PlanarAnnulusMTL
from scripts.framework.v2.mtl.tests.helpers import ltd_result
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Error


class TestPlanarAnnulusMTL(unittest.TestCase):
    def test_empty_input_fails_with_insufficient_data(self):
        result = PlanarAnnulusMTL(n_pts=4).multilaterate([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)
        self.assertIsNone(result.intersection)
        self.assertEqual(result.method, "PlanarAnnulusMTL")

    def test_single_annulus_subtracts_inner_diamond(self):
        result = PlanarAnnulusMTL(n_pts=4).multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=222.0, lower_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        self.assertEqual(result.intersection.bounds, (-2.0, -2.0, 2.0, 2.0))
        self.assertAlmostEqual(result.intersection.area, 6.0, places=6)
        self.assertFalse(result.intersection.contains(Point(0.0, 0.0)))
        self.assertTrue(result.intersection.contains(Point(1.5, 0.0)))

    def test_disk_constraint_degenerates_to_planar_circle_shape(self):
        result = PlanarAnnulusMTL(n_pts=4).multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=111.0, lower_km=0.0),
        ])

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.intersection.area, 2.0, places=6)
        self.assertEqual(result.intersection.bounds, (-1.0, -1.0, 1.0, 1.0))

    def test_two_same_center_annuli_keep_max_inner_and_min_outer(self):
        result = PlanarAnnulusMTL(n_pts=4).multilaterate([
            ltd_result("wide", lat=0.0, lon=0.0, upper_km=444.0, lower_km=111.0),
            ltd_result("narrow", lat=0.0, lon=0.0, upper_km=333.0, lower_km=222.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.bounds, (-3.0, -3.0, 3.0, 3.0))
        self.assertAlmostEqual(result.intersection.area, 10.0, places=6)
        self.assertFalse(result.intersection.contains(Point(0.0, 0.0)))
        self.assertTrue(result.intersection.contains(Point(2.5, 0.0)))

    def test_inner_radius_covering_outer_disk_fails(self):
        # Distance() rejects lower_km > upper_km, so use lower_km == upper_km
        # (a degenerate annulus that the wrapped function should treat as empty).
        result = PlanarAnnulusMTL(n_pts=4).multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=222.0, lower_km=222.0),
        ])

        self.assertFalse(result.success)

    def test_registered_in_mtl_registry(self):
        self.assertIn("planar_annulus", MTL_REGISTRY)
        self.assertIs(MTL_REGISTRY["planar_annulus"], PlanarAnnulusMTL)

    def test_is_annulus_family(self):
        self.assertTrue(issubclass(PlanarAnnulusMTL, AnnulusMTLMethod))


if __name__ == "__main__":
    unittest.main()
