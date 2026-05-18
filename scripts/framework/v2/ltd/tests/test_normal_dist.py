"""Tests for NormalDistLTD — pooled-normal RTT-to-distance model.

Mirrors scripts/framework/distance/tests/test_spotter.py, ported to the v2
interface and renamed from v1's `SpotterDistance` to `NormalDistLTD` (the class
name now emphasizes the pooled-normal technique rather than the Spotter paper).
Weight assertions and the unknown-anchor case do not migrate: weights moved out
of LTD; v2's caller pre-joins the VP coord so no lookup happens here.
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.framework.v2.ltd.normal_dist import NormalDistLTD
from scripts.framework.v2.ltd.tests.helpers import (
    ANCHOR_COORDS,
    make_fitted_degenerate_spotter_model,
    make_fitted_spotter_model,
    make_unfitted_spotter_model,
)
from scripts.framework.v2.types import Error, Latency, VpId


class TestNormalDistLTD(unittest.TestCase):
    def test_predict_produces_annular_constraints_with_pooled_band(self):
        """All VPs share the same (mu, sigma, k) — the pooled-normal claim."""
        ltd = NormalDistLTD(model=make_fitted_spotter_model())

        obs = [
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)),
            (VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(20.0)),
        ]
        results = ltd.predict_all(obs)

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r.success)
            # mu(20)=2000, k*sigma=100 -> [1900, 2100]
            self.assertAlmostEqual(r.tg_distance.lower_km, 1900.0)
            self.assertAlmostEqual(r.tg_distance.upper_km, 2100.0)
            self.assertTrue(r.tg_distance.is_annular)

    def test_predict_clips_inner_radius_at_zero(self):
        """When k*sigma > mu, lower_km clamps to 0 (degenerates to disk)."""
        # mu(rtt)=20*rtt, sigma=50, k=2 → band=±100. At rtt=1: mu=20, inner_raw=-80→0; outer=120.
        model = make_fitted_spotter_model(
            p_mu=np.array([20.0, 0.0]),
            p_sigma=np.array([50.0]),
            k=2.0,
            rtt_min=0.0,
            rtt_max=20.0,
        )
        ltd = NormalDistLTD(model=model)

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(1.0)
        )

        self.assertTrue(result.success)
        self.assertEqual(result.tg_distance.lower_km, 0.0)
        self.assertAlmostEqual(result.tg_distance.upper_km, 120.0)
        self.assertFalse(result.tg_distance.is_annular)

    def test_predict_returns_degenerate_region_on_zero_width_band(self):
        ltd = NormalDistLTD(model=make_fitted_degenerate_spotter_model())

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.DEGENERATE_REGION)

    def test_predict_returns_vp_not_fitted_for_unfitted_model(self):
        ltd = NormalDistLTD(model=make_unfitted_spotter_model())

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_returns_vp_not_fitted_when_no_model(self):
        """Constructor with no model — every _predict reports VP_NOT_FITTED."""
        ltd = NormalDistLTD()

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_returns_rtt_out_of_range_outside_model_range(self):
        model = make_fitted_spotter_model(rtt_min=10.0, rtt_max=60.0)
        ltd = NormalDistLTD(model=model)

        below = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(5.0)
        )
        above = ltd.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(80.0)
        )
        ok = ltd.predict(
            VpId("anchor-c"), ANCHOR_COORDS[VpId("anchor-c")], Latency(30.0)
        )

        self.assertFalse(below.success)
        self.assertEqual(below.error, Error.RTT_OUT_OF_RANGE)
        self.assertFalse(above.success)
        self.assertEqual(above.error, Error.RTT_OUT_OF_RANGE)
        self.assertTrue(ok.success)

    def test_predict_applies_max_rtt_cutoff_before_prediction(self):
        ltd = NormalDistLTD(model=make_fitted_spotter_model(), max_rtt_ms=10.0)

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(10.1)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.RTT_OUT_OF_RANGE)

    def test_predict_failure_echoes_vp_id_coord_and_stamps_method(self):
        unfitted = NormalDistLTD(model=make_unfitted_spotter_model())
        fitted = NormalDistLTD(model=make_fitted_spotter_model())

        success = fitted.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )
        failure = unfitted.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(20.0)
        )

        self.assertEqual(success.method, "NormalDistLTD")
        self.assertEqual(failure.method, "NormalDistLTD")
        self.assertEqual(failure.vp_id, VpId("anchor-b"))
        self.assertEqual(failure.vp_coord, ANCHOR_COORDS[VpId("anchor-b")])
        self.assertIsNone(failure.tg_distance)

    def test_fit_returns_success_when_model_present(self):
        ltd = NormalDistLTD(model=make_fitted_spotter_model())

        result = ltd.fit([])

        self.assertTrue(result.success)
        self.assertEqual(result.method, "NormalDistLTD")

    def test_fit_returns_insufficient_data_when_no_model(self):
        ltd = NormalDistLTD()

        result = ltd.fit([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_registered_in_ltd_registry(self):
        from scripts.framework.v2.registry import LTD_REGISTRY

        self.assertIn("normal_dist", LTD_REGISTRY)
        self.assertIs(LTD_REGISTRY["normal_dist"], NormalDistLTD)


if __name__ == "__main__":
    unittest.main()
