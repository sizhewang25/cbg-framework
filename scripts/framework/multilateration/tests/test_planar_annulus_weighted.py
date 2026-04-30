"""Tests for weighted planar-annulus multilateration."""

from __future__ import annotations

import unittest

from scripts.framework.multilateration.planar_annulus_weighted import (
    PlanarAnnulusWeightedMultilateration,
)
from scripts.framework.multilateration.tests.helpers import circle


class TestPlanarAnnulusWeightedMultilateration(unittest.TestCase):
    def test_empty_input_fails(self):
        result = PlanarAnnulusWeightedMultilateration().multilaterate([])

        self.assertFalse(result.success)
        self.assertIsNone(result.region)

    def test_zero_total_weight_fails(self):
        result = PlanarAnnulusWeightedMultilateration(
            weight_threshold=0.5,
            grid_resolution_deg=1.0,
        ).multilaterate([
            circle("a", radius_km=111.0, weight=0.0),
        ])

        self.assertFalse(result.success)

    def test_zero_threshold_returns_manual_grid_union(self):
        result = PlanarAnnulusWeightedMultilateration(
            weight_threshold=0.0,
            grid_resolution_deg=1.0,
        ).multilaterate([
            circle("a", radius_km=111.0, weight=1.0),
        ])

        self.assertTrue(result.success)
        # At the equator, 111 km maps to exactly 1 degree in this grid helper.
        # np.arange(-1, 1, 1) gives grid coordinates [-1, 0]. Each qualifying
        # grid cell is a 1x1 box around its point, so the union spans
        # [-1.5, 0.5] in both lon and lat and has area 4 deg^2.
        self.assertEqual(result.region.bounds, (-1.5, -1.5, 0.5, 0.5))
        self.assertAlmostEqual(result.region.area, 4.0, places=6)
        self.assertEqual([c.vp_ip for c in result.circles_used], ["a"])


if __name__ == "__main__":
    unittest.main()
