"""Tests for v2 CTR registry wiring."""

from __future__ import annotations

import unittest

import scripts.framework.v2  # noqa: F401  # Triggers v2 ctr registration imports.
from scripts.framework.v2.ctr.boundary_vertex_mean import BoundaryVertexMeanCTR
from scripts.framework.v2.ctr.geometric_centroid import GeometricCentroidCTR
from scripts.framework.v2.ctr.geometric_median import GeometricMedianCTR
from scripts.framework.v2.ctr.monte_carlo_median import MonteCarloMedianCTR
from scripts.framework.v2.registry import CTR_REGISTRY


class TestCTRRegistry(unittest.TestCase):
    def test_current_ctr_names_are_registered(self):
        expected = {
            "boundary_vertex_mean",
            "geometric_centroid",
            "monte_carlo_median",
            "geometric_median",
        }

        self.assertTrue(expected.issubset(CTR_REGISTRY.keys()))
        self.assertNotIn("arithmetic_mean", CTR_REGISTRY)

    def test_registered_classes_match_concrete_ports(self):
        self.assertIs(CTR_REGISTRY["boundary_vertex_mean"], BoundaryVertexMeanCTR)
        self.assertIs(CTR_REGISTRY["geometric_centroid"], GeometricCentroidCTR)
        self.assertIs(CTR_REGISTRY["geometric_median"], GeometricMedianCTR)
        self.assertIs(CTR_REGISTRY["monte_carlo_median"], MonteCarloMedianCTR)


if __name__ == "__main__":
    unittest.main()
