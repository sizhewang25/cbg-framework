"""Tests for unweighted planar-annulus multilateration."""

from __future__ import annotations

import unittest

from shapely.geometry import Point

from scripts.framework.multilateration.planar_annulus import (
    PlanarAnnulusMultilateration,
)
from scripts.framework.multilateration.tests.helpers import circle


class TestPlanarAnnulusMultilateration(unittest.TestCase):
    def test_empty_input_fails(self):
        result = PlanarAnnulusMultilateration(n_pts=4).multilaterate([])

        self.assertFalse(result.success)
        self.assertIsNone(result.region)

    def test_single_annulus_subtracts_inner_diamond(self):
        result = PlanarAnnulusMultilateration(n_pts=4).multilaterate([
            circle("a", inner_radius_km=111.0, radius_km=222.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.region.geom_type, "Polygon")
        self.assertEqual(result.region.bounds, (-2.0, -2.0, 2.0, 2.0))
        # Outer diamond area is 8 deg^2, inner diamond area is 2 deg^2.
        self.assertAlmostEqual(result.region.area, 6.0, places=6)
        self.assertFalse(result.region.contains(Point(0.0, 0.0)))
        self.assertTrue(result.region.contains(Point(1.5, 0.0)))

    def test_disk_constraint_degenerates_to_planar_circle_shape(self):
        result = PlanarAnnulusMultilateration(n_pts=4).multilaterate([
            circle("a", inner_radius_km=0.0, radius_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.region.area, 2.0, places=6)
        self.assertEqual(result.region.bounds, (-1.0, -1.0, 1.0, 1.0))

    def test_two_same_center_annuli_keep_max_inner_and_min_outer_bounds(self):
        result = PlanarAnnulusMultilateration(n_pts=4).multilaterate([
            circle("wide", inner_radius_km=111.0, radius_km=444.0),
            circle("narrow", inner_radius_km=222.0, radius_km=333.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual([c.vp_ip for c in result.circles_used], ["wide", "narrow"])
        self.assertEqual(result.region.bounds, (-3.0, -3.0, 3.0, 3.0))
        # Same-center annulus intersection keeps inner=max(1,2)=2 degrees
        # and outer=min(4,3)=3 degrees. Diamond area: 2*3^2 - 2*2^2 = 10.
        self.assertAlmostEqual(result.region.area, 10.0, places=6)
        self.assertFalse(result.region.contains(Point(0.0, 0.0)))
        self.assertTrue(result.region.contains(Point(2.5, 0.0)))

    def test_three_same_center_annuli_tighten_the_inner_exclusion(self):
        result = PlanarAnnulusMultilateration(n_pts=4).multilaterate([
            circle("wide", inner_radius_km=111.0, radius_km=444.0),
            circle("middle", inner_radius_km=222.0, radius_km=333.0),
            circle("tight-inner", inner_radius_km=277.5, radius_km=333.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(
            [c.vp_ip for c in result.circles_used],
            ["wide", "middle", "tight-inner"],
        )
        self.assertEqual(result.region.bounds, (-3.0, -3.0, 3.0, 3.0))
        # The third annulus raises the inner exclusion to 2.5 degrees while
        # the outer bound remains 3 degrees: 2*3^2 - 2*2.5^2 = 5.5.
        self.assertAlmostEqual(result.region.area, 5.5, places=6)
        self.assertFalse(result.region.contains(Point(2.25, 0.0)))
        self.assertTrue(result.region.contains(Point(2.75, 0.0)))

    def test_inner_radius_covering_outer_disk_fails(self):
        result = PlanarAnnulusMultilateration(n_pts=4).multilaterate([
            circle("a", inner_radius_km=222.0, radius_km=111.0),
        ])

        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
