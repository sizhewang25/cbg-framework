"""Tests for SpeedOfInternetLTD — theoretical speed-of-Internet RTT-to-distance model.

Mirrors scripts/framework/distance/tests/test_speed_of_internet.py, ported to the
v2 interface (one-VP-at-a-time `_predict`, `Distance` dataclass, `Error` enum).
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.ltd.speed_of_internet import SpeedOfInternetLTD
from scripts.framework.v2.ltd.tests.helpers import ANCHOR_COORDS
from scripts.framework.v2.types import Error, Latency, VpId


class TestSpeedOfInternetLTD(unittest.TestCase):
    def test_predict_uses_two_thirds_c_default(self):
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
        self.assertAlmostEqual(results[1].tg_distance.upper_km, 250.0)

    def test_predict_applies_rtt_cutoff(self):
        ltd = SpeedOfInternetLTD(max_rtt_ms=10.0)

        ok = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.0)
        )
        too_high = ltd.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(10.001)
        )

        self.assertTrue(ok.success)
        self.assertAlmostEqual(ok.tg_distance.upper_km, 1000.0)
        self.assertFalse(too_high.success)
        self.assertEqual(too_high.error, Error.RTT_OUT_OF_RANGE)

    def test_predict_uses_custom_speed_threshold(self):
        ltd = SpeedOfInternetLTD(speed_threshold=1.0)

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

    def test_predict_failure_echoes_vp_id_coord_and_stamps_method(self):
        """On failure, vp_id and vp_coord are echoed; method is stamped on both paths."""
        ltd = SpeedOfInternetLTD(max_rtt_ms=10.0)

        success = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(5.0)
        )
        failure = ltd.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(20.0)
        )

        self.assertEqual(success.method, "SpeedOfInternetLTD")
        self.assertEqual(failure.method, "SpeedOfInternetLTD")
        self.assertEqual(failure.vp_id, VpId("anchor-b"))
        self.assertEqual(failure.vp_coord, ANCHOR_COORDS[VpId("anchor-b")])
        self.assertIsNone(failure.tg_distance)

    def test_fit_returns_success_for_stateless_model(self):
        """SpeedOfInternetLTD is stateless; fit always succeeds."""
        ltd = SpeedOfInternetLTD()

        result = ltd.fit([])

        self.assertTrue(result.success)
        self.assertEqual(result.method, "SpeedOfInternetLTD")

    def test_predict_all_preserves_input_order_on_duplicates(self):
        """predict_all is dumb iteration; duplicates appear in input order."""
        ltd = SpeedOfInternetLTD()
        obs = [
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.0)),
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(5.0)),
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)),
        ]

        results = ltd.predict_all(obs)

        self.assertEqual(
            [r.tg_distance.upper_km for r in results], [1000.0, 500.0, 2000.0]
        )

    def test_registered_in_ltd_registry(self):
        from scripts.framework.v2.registry import LTD_REGISTRY

        self.assertIn("speed_of_internet", LTD_REGISTRY)
        self.assertIs(LTD_REGISTRY["speed_of_internet"], SpeedOfInternetLTD)


if __name__ == "__main__":
    unittest.main()
