"""Tests for PlanarAnnulusWeightedMTL.

Ports scripts/framework/multilateration/tests/test_planar_annulus_weighted.py.

The wrapper computes per-VP weights from `latency` (`exp(-rtt/tau)`). Until
LTDResult.latency lands, the weighted variant returns INSUFFICIENT_DATA on any
plain LTDResult — that path is covered explicitly. The happy-path test uses a
duck-typed namespace from helpers.ltd_result_with_latency.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.mtl.base import AnnulusMTLMethod
from scripts.framework.v2.mtl.planar_annulus_weighted import PlanarAnnulusWeightedMTL
from scripts.framework.v2.mtl.tests.helpers import (
    ltd_result,
    ltd_result_with_latency,
)
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Error


class TestPlanarAnnulusWeightedMTL(unittest.TestCase):
    def test_empty_input_fails_with_insufficient_data(self):
        result = PlanarAnnulusWeightedMTL().multilaterate([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)
        self.assertIsNone(result.intersection)
        self.assertEqual(result.method, "PlanarAnnulusWeightedMTL")

    def test_missing_latency_fails_with_insufficient_data(self):
        """Plain LTDResult has no latency field; the wrapper must bail out."""
        result = PlanarAnnulusWeightedMTL().multilaterate([
            ltd_result("a", lat=0.0, lon=0.0, upper_km=111.0),
        ])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_zero_threshold_returns_manual_grid_union(self):
        """Mirrors v1 test_zero_threshold_returns_manual_grid_union.

        latency=0 → weight=exp(0)=1.0, matching the v1 fixture's weight=1.0.
        """
        result = PlanarAnnulusWeightedMTL(
            weight_threshold=0.0,
            grid_resolution_deg=1.0,
        ).multilaterate([
            ltd_result_with_latency(
                "a", lat=0.0, lon=0.0, upper_km=111.0, lower_km=0.0, latency=0.0
            ),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.bounds, (-1.5, -1.5, 0.5, 0.5))
        self.assertAlmostEqual(result.intersection.area, 4.0, places=6)

    def test_registered_in_mtl_registry(self):
        self.assertIn("planar_annulus_weighted", MTL_REGISTRY)
        self.assertIs(
            MTL_REGISTRY["planar_annulus_weighted"], PlanarAnnulusWeightedMTL
        )

    def test_is_annulus_family(self):
        self.assertTrue(issubclass(PlanarAnnulusWeightedMTL, AnnulusMTLMethod))


if __name__ == "__main__":
    unittest.main()
