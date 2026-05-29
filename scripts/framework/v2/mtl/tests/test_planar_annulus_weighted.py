"""Tests for PlanarAnnulusWeightedMTL.

Ports scripts/framework/multilateration/tests/test_planar_annulus_weighted.py.

The wrapper computes per-VP weights from `latency` (`exp(-rtt/tau)`). Until
LTDResult.latency lands, the weighted variant returns INSUFFICIENT_DATA on any
plain LTDResult — that path is covered explicitly. The happy-path test uses a
duck-typed namespace from helpers.ltd_result_with_latency.
"""

from __future__ import annotations

import unittest

from shapely.geometry import Point

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

    def test_zero_threshold_returns_full_disk_face(self):
        """Single annulus → one face (the whole disk) carrying all the weight.

        latency=0 → weight=exp(0)=1.0. Σw=1.0; threshold 0.0 → target 0.
        The top-and-only face clears 0 immediately, so the full disk is
        returned. With the legacy grid algorithm this used to assert a 4-cell
        grid union; the face-decomposition algorithm returns the disk polygon.
        """
        result = PlanarAnnulusWeightedMTL(weight_threshold=0.0).multilaterate([
            ltd_result_with_latency(
                "a", lat=0.0, lon=0.0, upper_km=111.0, lower_km=0.0, latency=0.0
            ),
        ])

        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        # 64-vertex polygon inscribed in the unit circle at the equator.
        self.assertEqual(result.intersection.bounds, (-1.0, -1.0, 1.0, 1.0))
        # Inscribed 64-gon area: (n/2)·sin(2π/n) ≈ 3.1365 for n=64.
        self.assertAlmostEqual(result.intersection.area, 3.1365, places=3)

    def test_registered_in_mtl_registry(self):
        self.assertIn("planar_annulus_weighted", MTL_REGISTRY)
        self.assertIs(
            MTL_REGISTRY["planar_annulus_weighted"], PlanarAnnulusWeightedMTL
        )

    def test_is_annulus_family(self):
        self.assertTrue(issubclass(PlanarAnnulusWeightedMTL, AnnulusMTLMethod))

    def test_highest_weight_only_collapses_disconnected_union(self):
        """Two far-apart VP pairs would union to a MultiPolygon under the
        legacy path. Pair (A,B) gets latency 0 (weight≈1.0) and pair (C,D)
        gets latency 50 ms (weight≈0.368), so the heavy pair's face is
        unambiguously face #1. With highest_weight_only=True the wrapper
        returns just that face — Polygon, contains the heavy-pair centroid,
        excludes the light-pair centroid."""
        results = [
            ltd_result_with_latency("A", 0.0,  0.0, upper_km=222.0, lower_km=0.0, latency=0.0),
            ltd_result_with_latency("B", 0.0, -1.0, upper_km=222.0, lower_km=0.0, latency=0.0),
            ltd_result_with_latency("C", 0.0,  5.0, upper_km=222.0, lower_km=0.0, latency=50.0),
            ltd_result_with_latency("D", 0.0,  6.0, upper_km=222.0, lower_km=0.0, latency=50.0),
        ]
        mtl = PlanarAnnulusWeightedMTL(
            highest_weight_only=True,
            enable_circle_filter=False,
        )
        result = mtl.multilaterate(results)
        self.assertTrue(result.success)
        self.assertEqual(result.intersection.geom_type, "Polygon")
        self.assertTrue(result.intersection.contains(Point(-0.5, 0.0)))   # heavy pair
        self.assertFalse(result.intersection.contains(Point(5.5, 0.0)))   # light pair excluded


if __name__ == "__main__":
    unittest.main()
