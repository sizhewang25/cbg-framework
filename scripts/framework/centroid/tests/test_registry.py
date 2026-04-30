"""Tests for centroid registry wiring."""

from __future__ import annotations

import unittest

import scripts.framework  # noqa: F401  # Triggers framework registration imports.
from scripts.framework.registry import CENTROID_REGISTRY


class TestCentroidRegistry(unittest.TestCase):
    def test_current_centroid_names_are_registered(self):
        expected = {
            "boundary_vertex_mean",
            "geometric_centroid",
            "monte_carlo_median",
            "geometric_median",
        }

        self.assertTrue(expected.issubset(CENTROID_REGISTRY.keys()))
        self.assertNotIn("arithmetic_mean", CENTROID_REGISTRY)


if __name__ == "__main__":
    unittest.main()
