"""Tests for BoundedSplineLTD — per-anchor Octant spline + delta-band RTT-to-distance model.

Mirrors scripts/framework/distance/tests/test_bounded_spline.py, ported to the
v2 interface. v1's combined "skip unfitted and prediction-error" test splits into
two v2 cases because each maps to a distinct `Error` code (VP_NOT_FITTED vs
NUMERICAL_FAILURE).
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.ltd.bounded_spline import BoundedSplineLTD
from scripts.framework.v2.ltd.tests.helpers import (
    ANCHOR_COORDS,
    make_fitted_degenerate_octant_model,
    make_fitted_octant_model,
    make_unfitted_octant_model,
)
from scripts.framework.v2.types import Coord, Error, Latency, VpId


class TestBoundedSplineLTD(unittest.TestCase):
    def test_predict_creates_annular_constraints(self):
        """At RTT=20 the hand-derived hull bounds are [1900, 2100] km."""
        ltd = BoundedSplineLTD(
            models={VpId("anchor-a"): make_fitted_octant_model("anchor-a")}
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.tg_distance.lower_km, 1900.0)
        self.assertAlmostEqual(result.tg_distance.upper_km, 2100.0)
        self.assertTrue(result.tg_distance.is_annular)

    def test_predict_returns_degenerate_region_on_zero_bounds(self):
        ltd = BoundedSplineLTD(
            models={VpId("anchor-a"): make_fitted_degenerate_octant_model("anchor-a")}
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.DEGENERATE_REGION)

    def test_predict_returns_vp_not_fitted_for_unfitted_model(self):
        ltd = BoundedSplineLTD(
            models={VpId("anchor-a"): make_unfitted_octant_model("anchor-a")}
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_returns_numerical_failure_when_prediction_raises(self):
        """A fitted Octant model with fit_spline=False raises on bounds prediction."""
        ltd = BoundedSplineLTD(
            models={VpId("anchor-b"): make_fitted_octant_model("anchor-b", fit_spline=False)},
            delta=1.2,
        )

        result = ltd.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.NUMERICAL_FAILURE)

    def test_predict_returns_vp_not_fitted_for_unknown_vp(self):
        ltd = BoundedSplineLTD(
            models={VpId("anchor-a"): make_fitted_octant_model("anchor-a")}
        )

        result = ltd.predict(VpId("unknown"), Coord(0.0, 0.0), Latency(20.0))

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_applies_rtt_cutoff(self):
        ltd = BoundedSplineLTD(
            models={VpId("anchor-a"): make_fitted_octant_model("anchor-a")},
            max_rtt_ms=10.0,
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.1)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.RTT_OUT_OF_RANGE)

    def test_predict_failure_echoes_vp_id_coord_and_stamps_method(self):
        ltd = BoundedSplineLTD(
            models={VpId("anchor-a"): make_fitted_octant_model("anchor-a")}
        )

        success = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )
        failure = ltd.predict(VpId("unknown"), Coord(0.0, 0.0), Latency(20.0))

        self.assertEqual(success.method, "BoundedSplineLTD")
        self.assertEqual(failure.method, "BoundedSplineLTD")
        self.assertEqual(failure.vp_id, VpId("unknown"))
        self.assertEqual(failure.vp_coord, Coord(0.0, 0.0))
        self.assertIsNone(failure.tg_distance)

    def test_fit_returns_success_when_models_present(self):
        ltd = BoundedSplineLTD(
            models={VpId("anchor-a"): make_fitted_octant_model("anchor-a")}
        )

        result = ltd.fit([])

        self.assertTrue(result.success)
        self.assertEqual(result.method, "BoundedSplineLTD")

    def test_fit_returns_insufficient_data_when_no_models(self):
        ltd = BoundedSplineLTD()

        result = ltd.fit([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_registered_in_ltd_registry(self):
        from scripts.framework.v2.registry import LTD_REGISTRY

        self.assertIn("bounded_spline", LTD_REGISTRY)
        self.assertIs(LTD_REGISTRY["bounded_spline"], BoundedSplineLTD)


if __name__ == "__main__":
    unittest.main()
