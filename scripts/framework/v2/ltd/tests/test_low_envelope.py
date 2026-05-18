"""Tests for LowEnvelopeLTD — per-anchor LP RTT-to-distance model.

Mirrors scripts/framework/distance/tests/test_low_envelope.py, ported to the v2
interface (per-VP `_predict`, typed `Error` codes replacing v1's silent skips).
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.ltd.low_envelope import LowEnvelopeLTD
from scripts.framework.v2.ltd.tests.helpers import (
    ANCHOR_COORDS,
    make_fitted_low_envelope_model,
    make_unfitted_low_envelope_model,
)
from scripts.framework.v2.types import Coord, Error, Latency, VpId


class TestLowEnvelopeLTD(unittest.TestCase):
    def test_predict_uses_only_fitted_positive_predictions(self):
        """Per-VP partitioning: fitted-and-positive → success; others → typed failure."""
        models = {
            VpId("anchor-a"): make_fitted_low_envelope_model("anchor-a"),
            VpId("anchor-b"): make_unfitted_low_envelope_model("anchor-b"),
            VpId("anchor-c"): make_fitted_low_envelope_model("anchor-c"),
        }
        ltd = LowEnvelopeLTD(models=models)

        ok = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(25.0)
        )
        unfitted = ltd.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(25.0)
        )
        # rtt=4 below intercept=5 → predict_distance returns non-positive → NUMERICAL_FAILURE
        bad = ltd.predict(
            VpId("anchor-c"), ANCHOR_COORDS[VpId("anchor-c")], Latency(4.0)
        )

        self.assertTrue(ok.success)
        self.assertAlmostEqual(ok.tg_distance.upper_km, 1000.0)
        self.assertEqual(ok.tg_distance.lower_km, 0.0)
        self.assertFalse(unfitted.success)
        self.assertEqual(unfitted.error, Error.VP_NOT_FITTED)
        self.assertFalse(bad.success)
        self.assertEqual(bad.error, Error.NUMERICAL_FAILURE)

    def test_predict_returns_vp_not_fitted_for_unknown_vp(self):
        ltd = LowEnvelopeLTD(
            models={VpId("anchor-a"): make_fitted_low_envelope_model("anchor-a")}
        )

        result = ltd.predict(VpId("unknown"), Coord(0.0, 0.0), Latency(25.0))

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_applies_rtt_cutoff(self):
        ltd = LowEnvelopeLTD(
            models={VpId("anchor-a"): make_fitted_low_envelope_model("anchor-a")},
            max_rtt_ms=10.0,
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.1)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.RTT_OUT_OF_RANGE)

    def test_predict_failure_echoes_vp_id_coord_and_stamps_method(self):
        ltd = LowEnvelopeLTD(
            models={VpId("anchor-a"): make_fitted_low_envelope_model("anchor-a")}
        )

        success = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(25.0)
        )
        failure = ltd.predict(VpId("unknown"), Coord(0.0, 0.0), Latency(25.0))

        self.assertEqual(success.method, "LowEnvelopeLTD")
        self.assertEqual(failure.method, "LowEnvelopeLTD")
        self.assertEqual(failure.vp_id, VpId("unknown"))
        self.assertEqual(failure.vp_coord, Coord(0.0, 0.0))
        self.assertIsNone(failure.tg_distance)

    def test_fit_returns_success_when_models_present(self):
        ltd = LowEnvelopeLTD(
            models={VpId("anchor-a"): make_fitted_low_envelope_model("anchor-a")}
        )

        result = ltd.fit([])

        self.assertTrue(result.success)
        self.assertEqual(result.method, "LowEnvelopeLTD")

    def test_fit_returns_insufficient_data_when_no_models(self):
        ltd = LowEnvelopeLTD()

        result = ltd.fit([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_registered_in_ltd_registry(self):
        from scripts.framework.v2.registry import LTD_REGISTRY

        self.assertIn("low_envelope", LTD_REGISTRY)
        self.assertIs(LTD_REGISTRY["low_envelope"], LowEnvelopeLTD)


if __name__ == "__main__":
    unittest.main()
