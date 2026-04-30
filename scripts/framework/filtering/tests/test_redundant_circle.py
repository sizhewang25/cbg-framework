"""Tests for redundant-circle filtering."""

from __future__ import annotations

import unittest

from scripts.framework.filtering.redundant_circle import RedundantCircleFilter
from scripts.framework.filtering.tests.helpers import circle


class TestRedundantCircleFilter(unittest.TestCase):
    def test_single_constraint_is_returned_unchanged(self):
        circles = [circle("only", radius_km=100.0)]

        filtered = RedundantCircleFilter().filter(circles)

        self.assertEqual(filtered, circles)
        self.assertIsNot(filtered, circles)

    def test_removes_larger_same_center_circle(self):
        circles = [
            circle("wide", rtt_ms=30.0, radius_km=300.0),
            circle("tight", rtt_ms=10.0, radius_km=100.0),
        ]

        filtered = RedundantCircleFilter().filter(circles)

        self.assertEqual([c.vp_ip for c in filtered], ["tight"])

    def test_removes_redundant_chain_to_tightest_circle(self):
        circles = [
            circle("wide", rtt_ms=30.0, radius_km=300.0),
            circle("medium", rtt_ms=20.0, radius_km=200.0),
            circle("tight", rtt_ms=10.0, radius_km=100.0),
        ]

        filtered = RedundantCircleFilter().filter(circles)

        self.assertEqual([c.vp_ip for c in filtered], ["tight"])

    def test_keeps_non_containing_circles_in_input_order(self):
        circles = [
            circle("tight", rtt_ms=10.0, radius_km=100.0),
            # One degree of longitude at the equator is about 111 km, so two
            # 100 km circles centered 1 degree apart cannot contain each other.
            circle("offset", lon=1.0, rtt_ms=10.0, radius_km=100.0),
            circle("far", lon=10.0, rtt_ms=5.0, radius_km=50.0),
        ]

        filtered = RedundantCircleFilter().filter(circles)

        self.assertEqual([c.vp_ip for c in filtered], ["tight", "offset", "far"])

    def test_uses_outer_radius_and_preserves_kept_constraint_metadata(self):
        circles = [
            circle(
                "wide-annulus",
                rtt_ms=30.0,
                radius_km=300.0,
                inner_radius_km=250.0,
                weight=0.25,
            ),
            circle("tight", rtt_ms=10.0, radius_km=100.0, weight=0.75),
        ]

        filtered = RedundantCircleFilter().filter(circles)

        self.assertEqual([c.vp_ip for c in filtered], ["tight"])
        self.assertEqual(filtered[0].weight, 0.75)
        self.assertEqual(filtered[0].inner_radius_km, 0.0)


if __name__ == "__main__":
    unittest.main()
