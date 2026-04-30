"""Tests for the fixed speed-of-Internet distance model."""

from __future__ import annotations

import unittest

from scripts.framework.distance.speed_of_internet import SpeedOfInternetDistance
from scripts.framework.distance.tests.helpers import ANCHOR_COORDS


class TestSpeedOfInternetDistance(unittest.TestCase):
    def test_estimate_uses_two_thirds_c_default(self):
        model = SpeedOfInternetDistance()

        circles = model.estimate(
            {"anchor-a": 10.0, "anchor-b": 2.5},
            ANCHOR_COORDS,
        )

        self.assertEqual([c.vp_ip for c in circles], ["anchor-a", "anchor-b"])
        self.assertAlmostEqual(circles[0].radius_km, 1000.0)
        self.assertAlmostEqual(circles[1].radius_km, 250.0)
        self.assertEqual(circles[0].inner_radius_km, 0.0)
        self.assertEqual(circles[0].weight, 1.0)
        self.assertEqual(circles[0].to_legacy_tuple()[0:3], (40.0, -74.0, 10.0))

    def test_estimate_skips_missing_anchors_and_rtts_above_cutoff(self):
        model = SpeedOfInternetDistance(max_rtt_ms=10.0)

        circles = model.estimate(
            {
                "anchor-a": 10.0,
                "anchor-b": 10.001,
                "unknown-anchor": 1.0,
            },
            ANCHOR_COORDS,
        )

        self.assertEqual(len(circles), 1)
        self.assertEqual(circles[0].vp_ip, "anchor-a")
        self.assertAlmostEqual(circles[0].radius_km, 1000.0)

    def test_custom_speed_threshold_is_used(self):
        model = SpeedOfInternetDistance(speed_threshold=1.0)

        circles = model.estimate({"anchor-a": 10.0}, ANCHOR_COORDS)

        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0].radius_km, 1500.0)


if __name__ == "__main__":
    unittest.main()
