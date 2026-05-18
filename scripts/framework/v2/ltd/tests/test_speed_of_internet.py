"""Tests for SpeedOfInternetLTD — universal theoretical RTT-to-distance model.

The wrapper is stateless: a single `speed_ratio` hyperparameter, no fit, no
cutoffs, no calibration. `latency` is the RTT in milliseconds throughout.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.ltd.speed_of_internet import SpeedOfInternetLTD
from scripts.framework.v2.ltd.tests.helpers import ANCHOR_COORDS
from scripts.framework.v2.types import Latency, VpId


class TestSpeedOfInternetLTD(unittest.TestCase):
    def test_predict_uses_two_thirds_c_default(self):
        """speed_ratio defaults to 2/3 — radius_km == 100 * rtt_ms."""
        ltd = SpeedOfInternetLTD()
        obs = [
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.0)),
            (VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(2.5)),
        ]

        results = ltd.predict_all(obs)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        self.assertAlmostEqual(results[0].tg_distance.upper_km, 1000.0)
        self.assertEqual(results[0].tg_distance.lower_km, 0.0)
        self.assertEqual(results[0].vp_id, VpId("anchor-a"))
        self.assertEqual(results[0].vp_coord, ANCHOR_COORDS[VpId("anchor-a")])
        self.assertEqual(results[0].latency, Latency(10.0))
        self.assertEqual(results[0].method, "SpeedOfInternetLTD")
        self.assertAlmostEqual(results[1].tg_distance.upper_km, 250.0)

    def test_predict_uses_custom_speed_ratio(self):
        """speed_ratio=1.0 → radius_km == 150 * rtt_ms (signal at exactly c)."""
        ltd = SpeedOfInternetLTD(speed_ratio=1.0)

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.0)
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.tg_distance.upper_km, 1500.0)

    def test_predict_at_zero_rtt_returns_zero_distance(self):
        """RTT=0 (probe colocated with VP) is a legal success with Distance(upper_km=0)."""
        ltd = SpeedOfInternetLTD()

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(0.0)
        )

        self.assertTrue(result.success)
        self.assertEqual(result.tg_distance.upper_km, 0.0)
        self.assertEqual(result.tg_distance.lower_km, 0.0)
        self.assertFalse(result.tg_distance.is_annular)

    def test_predict_all_preserves_input_order_on_duplicates(self):
        """predict_all is dumb iteration; duplicates appear in input order."""
        ltd = SpeedOfInternetLTD()
        obs = [
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.0)),
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(5.0)),
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)),
        ]

        results = ltd.predict_all(obs)

        self.assertEqual(len(results), 3)
        for r, expected in zip(results, [1000.0, 500.0, 2000.0]):
            self.assertAlmostEqual(r.tg_distance.upper_km, expected)

    def test_fit_is_noop_success(self):
        """No state to fit; fit() always succeeds and leaves speed_ratio alone."""
        ltd = SpeedOfInternetLTD(speed_ratio=0.5)
        before = ltd.speed_ratio

        result = ltd.fit([])

        self.assertTrue(result.success)
        self.assertEqual(result.method, "SpeedOfInternetLTD")
        self.assertEqual(ltd.speed_ratio, before)

    def test_registered_in_ltd_registry(self):
        from scripts.framework.v2.registry import LTD_REGISTRY

        self.assertIn("speed_of_internet", LTD_REGISTRY)
        self.assertIs(LTD_REGISTRY["speed_of_internet"], SpeedOfInternetLTD)


if __name__ == "__main__":
    unittest.main()
