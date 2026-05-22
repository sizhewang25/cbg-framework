"""Tests for NormalDistLTD — pooled-normal RTT-to-distance model.

Mirrors scripts/framework/distance/tests/test_spotter.py, ported to the v2
interface and renamed from v1's `SpotterDistance` to `NormalDistLTD`. The class
name now emphasizes the pooled-normal technique rather than the Spotter paper.
Weight assertions and the unknown-anchor case do not migrate: weights moved out
of LTD; v2's caller pre-joins the VP coord so no lookup happens here.

Prediction tests inject a pre-fitted SpotterRTTModel via `ltd._model`; one
integration test exercises the real fit-from-FitSamples path.
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.framework.v2.ltd.normal_dist import NormalDistLTD
from scripts.framework.v2.ltd.tests.helpers import (
    ANCHOR_COORDS,
    make_fitted_degenerate_spotter_model,
    make_fitted_spotter_model,
    make_normal_dist_fit_samples,
    make_unfitted_spotter_model,
)
from scripts.framework.v2.types import Error, Latency, VpId
from scripts.libs.spotter.spotter_model import THEORETICAL_SLOPE


class TestNormalDistLTD(unittest.TestCase):
    def _ltd_with_model(self, model) -> NormalDistLTD:
        ltd = NormalDistLTD()
        ltd._model = model
        return ltd

    def test_predict_produces_annular_constraints_with_pooled_band(self):
        """All VPs share the same (mu, sigma) — the pooled-normal claim."""
        ltd = self._ltd_with_model(make_fitted_spotter_model())

        obs = [
            (VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)),
            (VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(20.0)),
        ]
        results = ltd.predict_all(obs)

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r.success)
            # mu(20)=1000, sigma=50 -> [950, 1050]
            self.assertAlmostEqual(r.tg_distance.lower_km, 950.0)
            self.assertAlmostEqual(r.tg_distance.upper_km, 1050.0)
            self.assertTrue(r.tg_distance.is_annular)

    def test_predict_clips_inner_radius_at_zero(self):
        """When sigma > mu, lower_km clamps to 0 (degenerates to disk)."""
        # mu(rtt)=20*rtt, sigma=50 -> +/-sigma band of width 100. At rtt=1: mu=20,
        # inner_raw=-30 -> 0; raw outer=70, baseline cap = 1/THEORETICAL_SLOPE = 100,
        # so no clip -> outer=70.
        ltd = self._ltd_with_model(
            make_fitted_spotter_model(
                p_mu=np.array([20.0, 0.0]),
                p_sigma=np.array([50.0]),
                rtt_min=0.0,
                rtt_max=20.0,
            )
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(1.0)
        )

        self.assertTrue(result.success)
        self.assertEqual(result.tg_distance.lower_km, 0.0)
        self.assertAlmostEqual(result.tg_distance.upper_km, 70.0)
        self.assertFalse(result.tg_distance.is_annular)

    def test_predict_returns_degenerate_region_on_zero_width_band(self):
        ltd = self._ltd_with_model(make_fitted_degenerate_spotter_model())

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.DEGENERATE_REGION)

    def test_predict_returns_vp_not_fitted_for_unfitted_model(self):
        ltd = self._ltd_with_model(make_unfitted_spotter_model())

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_returns_vp_not_fitted_when_no_model(self):
        """Default-constructed ltd has no model — every _predict reports VP_NOT_FITTED."""
        ltd = NormalDistLTD()

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.VP_NOT_FITTED)

    def test_predict_returns_rtt_out_of_range_above_rtt_max_when_no_cutoff(self):
        """With cutoff_rtt unset (=0), the legacy rtt > rtt_max gate maps
        to RTT_OUT_OF_RANGE. Below rtt_min the model now linearly scales
        the bounds toward the origin — no rejection."""
        ltd = self._ltd_with_model(
            make_fitted_spotter_model(rtt_min=10.0, rtt_max=60.0)
        )

        above = ltd.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(80.0)
        )
        ok = ltd.predict(
            VpId("anchor-c"), ANCHOR_COORDS[VpId("anchor-c")], Latency(30.0)
        )

        self.assertFalse(above.success)
        self.assertEqual(above.error, Error.RTT_OUT_OF_RANGE)
        self.assertTrue(ok.success)

    def test_predict_succeeds_below_rtt_min_with_origin_line(self):
        """Below rtt_min the inner collapses to 0; outer scales linearly
        from outer(rtt_min) down to 0 at rtt=0. At rtt=0 the wrapper sees
        outer == inner == 0 and reports DEGENERATE_REGION."""
        ltd = self._ltd_with_model(
            make_fitted_spotter_model(rtt_min=10.0, rtt_max=60.0)
        )

        below = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(5.0)
        )

        self.assertTrue(below.success)
        # outer(10) = min(50*10 + 50, 1000) = 550. Scaled: (550/10) * 5 = 275.
        self.assertEqual(below.tg_distance.lower_km, 0.0)
        self.assertAlmostEqual(below.tg_distance.upper_km, 275.0)

    def test_predict_extends_outer_at_finite_sentinel_slope_above_cutoff(self):
        """Above cutoff_rtt: inner held flat at inner(cutoff); outer extends
        toward a finite sentinel on the 2/3·c bound — Octant paper's smooth-
        transition construction. No RTT_OUT_OF_RANGE rejection."""
        # mu(rtt)=50*rtt, sigma=50 -> ±σ band. cutoff_rtt=30 ->
        # inner(30)=1450, outer(30)=1550. With the default sentinel z at
        # (10_000 ms, 1_000_000 km), the extension slope is
        # (1_000_000 - 1550) / (10_000 - 30) ≈ 100.1454 km/ms, strictly
        # steeper than the asymptotic 100 km/ms.
        cutoff_rtt = 30.0
        sentinel_rtt = 10000.0
        outer_at_cutoff = 1550.0
        sentinel_dist = sentinel_rtt / THEORETICAL_SLOPE
        slope = (sentinel_dist - outer_at_cutoff) / (sentinel_rtt - cutoff_rtt)
        expected_outer = outer_at_cutoff + slope * (80.0 - cutoff_rtt)
        ltd = self._ltd_with_model(
            make_fitted_spotter_model(
                rtt_min=0.0,
                rtt_max=100.0,
                cutoff_rtt=cutoff_rtt,
                sentinel_rtt=sentinel_rtt,
            )
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(80.0)
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.tg_distance.lower_km, 1450.0)
        self.assertAlmostEqual(result.tg_distance.upper_km, expected_outer)
        # baseline cap at rtt=80 is 8000 km, so no 2/3·c clip
        self.assertLess(result.tg_distance.upper_km, 80.0 / THEORETICAL_SLOPE)

    def test_predict_returns_numerical_failure_when_bounds_raises(self):
        """rtt past sentinel raises ValueError in SpotterRTTModel; wrapper
        maps to NUMERICAL_FAILURE rather than letting it propagate."""
        ltd = self._ltd_with_model(
            make_fitted_spotter_model(
                rtt_min=10.0,
                rtt_max=10000.0,
                cutoff_rtt=50.0,
                sentinel_rtt=200.0,
            )
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(500.0)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.NUMERICAL_FAILURE)

    def test_predict_outer_above_cutoff_approaches_baseline_with_large_sentinel(self):
        """As sentinel_rtt → ∞, the cutoff extension reverts to the
        previous asymptotic 1/THEORETICAL_SLOPE slope (parallel to 2/3·c)."""
        ltd = self._ltd_with_model(
            make_fitted_spotter_model(
                rtt_min=0.0,
                rtt_max=100.0,
                cutoff_rtt=30.0,
                sentinel_rtt=1e9,
            )
        )

        result = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(80.0)
        )

        self.assertTrue(result.success)
        # outer(30) = 1550; large-sentinel slope ≈ 1/THEORETICAL_SLOPE
        # → 1550 + 50/THEORETICAL_SLOPE = 6550.
        self.assertAlmostEqual(result.tg_distance.upper_km, 6550.0, places=2)

    def test_predict_failure_echoes_vp_id_coord_and_stamps_method(self):
        unfitted = self._ltd_with_model(make_unfitted_spotter_model())
        fitted = self._ltd_with_model(make_fitted_spotter_model())

        success = fitted.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )
        failure = unfitted.predict(
            VpId("anchor-b"), ANCHOR_COORDS[VpId("anchor-b")], Latency(20.0)
        )

        self.assertEqual(success.method, "NormalDistLTD")
        self.assertEqual(success.latency, Latency(20.0))
        self.assertEqual(failure.method, "NormalDistLTD")
        self.assertEqual(failure.vp_id, VpId("anchor-b"))
        self.assertEqual(failure.vp_coord, ANCHOR_COORDS[VpId("anchor-b")])
        self.assertEqual(failure.latency, Latency(20.0))
        self.assertIsNone(failure.tg_distance)

    def test_fit_returns_insufficient_data_when_no_samples(self):
        ltd = NormalDistLTD()

        result = ltd.fit([])

        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_fit_from_samples_then_predict_produces_annular_bounds(self):
        """Integration: real fit(samples) → predict at known RTT yields a usable band.

        Uses small n_bins / min_per_bin so the test runs on a compact sample set.
        cutoff_min_points is also small so the sparse fixture clears the new
        Octant-style cutoff scan.
        """
        ltd = NormalDistLTD(
            n_bins=5, min_per_bin=2, deg_mu=1, deg_sigma=0,
            cutoff_min_points=1,
        )
        samples = make_normal_dist_fit_samples(
            "anchor-a", n_per_rtt=4, spread_km=100.0
        )

        fit_result = ltd.fit(samples)
        pred = ltd.predict(
            VpId("anchor-a"), ANCHOR_COORDS[VpId("anchor-a")], Latency(20.0)
        )

        self.assertTrue(fit_result.success, msg=str(fit_result.args))
        self.assertEqual(fit_result.method, "NormalDistLTD")
        self.assertIn("cutoff_rtt", fit_result.args)
        self.assertTrue(pred.success)
        # mu(20) ≈ 1000 from the parallel band centered at 50*rtt; the +/-sigma
        # band straddles the mean.
        self.assertLess(pred.tg_distance.lower_km, 1000.0)
        self.assertGreater(pred.tg_distance.upper_km, 1000.0)

    def test_registered_in_ltd_registry(self):
        from scripts.framework.v2.registry import LTD_REGISTRY

        self.assertIn("normal_dist", LTD_REGISTRY)
        self.assertIs(LTD_REGISTRY["normal_dist"], NormalDistLTD)


if __name__ == "__main__":
    unittest.main()
