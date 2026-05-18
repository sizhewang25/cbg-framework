"""Tests for BoundedSplineLTD — per-anchor Octant spline + delta-band model.

Mirrors scripts/framework/distance/tests/test_bounded_spline.py, ported to the
v2 interface. v1's combined "skip unfitted and prediction-error" test splits
into two v2 cases because each maps to a distinct `Error` code (VP_NOT_FITTED
vs NUMERICAL_FAILURE).

Prediction tests inject pre-fitted submodels via `ltd._submodels`; one
integration test exercises the real fit-from-FitSamples path.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.ltd.bounded_spline import BoundedSplineLTD
from scripts.framework.v2.ltd.tests.helpers import (
    ANCHOR_COORDS,
    make_bounded_spline_fit_samples,
    make_fitted_degenerate_octant_model,
    make_fitted_octant_model,
    make_unfitted_octant_model,
)
from scripts.framework.v2.types import Coord, Error, Latency, VpId


class TestBoundedSplineLTD(unittest.TestCase):
    def _ltd_with_submodels(self, *, deltas=None, **submodels) -> BoundedSplineLTD:
        """Inject submodels (and optional per-VP deltas) into the wrapper.

        `deltas` is a dict keyed by the same string VP IDs as `submodels`;
        unset VPs fall through to bare hull bounds.
        """
        ltd = BoundedSplineLTD()
        ltd._submodels = {VpId(k): v for k, v in submodels.items()}
        ltd._deltas = {VpId(k): v for k, v in (deltas or {}).items()}
        return ltd

    def test_predict_creates_annular_constraints(self):
        """At RTT=20 the hand-derived hull bounds are [900, 1100] km."""
        ltd = self._ltd_with_submodels(
            **{"anchor-a": make_fitted_octant_model("anchor-a")}
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.tg_distance.lower_km, 900.0)
        self.assertAlmostEqual(result.tg_distance.upper_km, 1100.0)
        self.assertTrue(result.tg_distance.is_annular)

    def test_predict_returns_degenerate_region_on_zero_bounds(self):
        ltd = self._ltd_with_submodels(
            **{"anchor-a": make_fitted_degenerate_octant_model("anchor-a")}
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.DEGENERATE_REGION)

    def test_predict_returns_vp_not_fitted_for_unfitted_model(self):
        ltd = self._ltd_with_submodels(
            **{"anchor-a": make_unfitted_octant_model("anchor-a")}
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_returns_numerical_failure_when_prediction_raises(self):
        """A fitted Octant model with fit_spline=False raises on bounds prediction.

        A per-VP delta forces predict_distance_bounds into the spline branch;
        with no spline, the underlying call raises SplineFitError, which the
        wrapper maps to NUMERICAL_FAILURE.
        """
        ltd = self._ltd_with_submodels(
            deltas={"anchor-b": 1.2},
            **{"anchor-b": make_fitted_octant_model("anchor-b", fit_spline=False)},
        )

        result = ltd.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.NUMERICAL_FAILURE)

    def test_predict_returns_vp_not_fitted_for_unknown_vp(self):
        ltd = self._ltd_with_submodels(
            **{"anchor-a": make_fitted_octant_model("anchor-a")}
        )

        result = ltd.predict(VpId("unknown"), Coord(0.0, 0.0), Latency(20.0))

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_failure_echoes_vp_id_coord_and_stamps_method(self):
        ltd = self._ltd_with_submodels(
            **{"anchor-a": make_fitted_octant_model("anchor-a")}
        )

        success = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )
        failure = ltd.predict(VpId("unknown"), Coord(0.0, 0.0), Latency(20.0))

        self.assertEqual(success.method, "BoundedSplineLTD")
        self.assertEqual(success.latency, Latency(20.0))
        self.assertEqual(failure.method, "BoundedSplineLTD")
        self.assertEqual(failure.vp_id, VpId("unknown"))
        self.assertEqual(failure.vp_coord, Coord(0.0, 0.0))
        self.assertEqual(failure.latency, Latency(20.0))
        self.assertIsNone(failure.tg_distance)

    def test_fit_returns_insufficient_data_when_no_samples(self):
        ltd = BoundedSplineLTD()

        result = ltd.fit([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_fit_from_samples_then_predict_recovers_hull_bounds(self):
        """Integration: fit(samples) → predict at RTT=20 recovers ~[900, 1100]."""
        ltd = BoundedSplineLTD(
            sample_coverage=0.8,
            cutoff_min_points=1,
            spline_n_knots=4,
            bin_size_ms=1000,
        )
        samples = make_bounded_spline_fit_samples("anchor-a")

        fit_result = ltd.fit(samples)
        pred = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertTrue(fit_result.success, msg=str(fit_result.args))
        self.assertIn(VpId("anchor-a"), fit_result.args["vps_fitted"])
        # Per-VP δ search runs against this VP's own data and spline.
        self.assertIn(VpId("anchor-a"), fit_result.args["deltas"])
        self.assertTrue(pred.success)
        # The parallel ±100 km band at RTT=20 yields hull bounds near [900, 1100].
        # Allow some slop for the spline/delta interaction.
        self.assertLess(pred.tg_distance.lower_km, 1000.0)
        self.assertGreater(pred.tg_distance.upper_km, 1000.0)

    def test_registered_in_ltd_registry(self):
        from scripts.framework.v2.registry import LTD_REGISTRY

        self.assertIn("bounded_spline", LTD_REGISTRY)
        self.assertIs(LTD_REGISTRY["bounded_spline"], BoundedSplineLTD)


if __name__ == "__main__":
    unittest.main()
