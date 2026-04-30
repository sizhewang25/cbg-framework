"""Tests for planar-circle multilateration."""

from __future__ import annotations

import unittest

from scripts.framework.multilateration.planar_circle import PlanarCircleMultilateration
from scripts.framework.multilateration.tests.helpers import circle


class TestPlanarCircleMultilateration(unittest.TestCase):
    def test_empty_input_fails(self):
        result = PlanarCircleMultilateration(n_pts=4).multilaterate([])

        self.assertFalse(result.success)
        self.assertIsNone(result.region)

    def test_single_equator_circle_becomes_unit_diamond(self):
        result = PlanarCircleMultilateration(n_pts=4).multilaterate([
            circle("a", radius_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.region.geom_type, "Polygon")
        self.assertEqual([c.vp_ip for c in result.circles_used], ["a"])
        self.assertAlmostEqual(result.region.area, 2.0, places=6)
        self.assertEqual(result.region.bounds, (-1.0, -1.0, 1.0, 1.0))

    def test_two_overlapping_circles_intersect_to_half_diamond(self):
        result = PlanarCircleMultilateration(n_pts=4).multilaterate([
            circle("left", lon=0.0, radius_km=111.0),
            circle("right", lon=1.0, radius_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.region.geom_type, "Polygon")
        self.assertEqual([c.vp_ip for c in result.circles_used], ["left", "right"])
        # Two unit diamonds centered one degree apart overlap between lon 0 and 1.
        # The lens has vertical length 2*min(x, 1-x), so area = 0.5 deg^2.
        self.assertAlmostEqual(result.region.area, 0.5, places=6)
        self.assertEqual(result.region.bounds, (0.0, -0.5, 1.0, 0.5))

    def test_three_overlapping_circles_tighten_the_two_circle_lens(self):
        result = PlanarCircleMultilateration(n_pts=4).multilaterate([
            circle("left", lon=0.0, radius_km=111.0),
            circle("right", lon=1.0, radius_km=111.0),
            circle("center", lon=0.5, radius_km=44.4),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.region.geom_type, "Polygon")
        self.assertEqual([c.vp_ip for c in result.circles_used], ["left", "right", "center"])
        # The first two diamonds form a lens. The centered third diamond has
        # radius 44.4/111 = 0.4 degrees and lies inside that lens, so the final
        # intersection is exactly that diamond: area = 2 * 0.4^2 = 0.32.
        self.assertAlmostEqual(result.region.area, 0.32, places=6)
        for actual, want in zip(result.region.bounds, (0.1, -0.4, 0.9, 0.4)):
            self.assertAlmostEqual(actual, want, places=6)

    def test_disjoint_circles_fail(self):
        result = PlanarCircleMultilateration(n_pts=4).multilaterate([
            circle("a", lon=0.0, radius_km=111.0),
            circle("b", lon=10.0, radius_km=111.0),
        ])

        self.assertFalse(result.success)
        self.assertEqual([c.vp_ip for c in result.circles_used], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
