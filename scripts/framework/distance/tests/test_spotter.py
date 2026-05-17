"""Tests for the Spotter pooled-normal distance-model wrapper."""

from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.framework.distance.spotter import SpotterDistance
from scripts.framework.distance.tests.helpers import (
    ANCHOR_COORDS,
    make_fitted_degenerate_spotter_model,
    make_fitted_spotter_model,
    make_unfitted_spotter_model,
)


class TestSpotterDistance(unittest.TestCase):
    def test_fit_accepts_prefitted_model(self):
        model = make_fitted_spotter_model()
        distance = SpotterDistance()

        distance.fit(model=model)

        self.assertIs(distance.model, model)

    def test_estimate_produces_annular_constraints_with_pooled_band(self):
        """All anchors share the same (p_mu, p_sigma, k) -- by Spotter's claim."""
        model = make_fitted_spotter_model()  # band: mu(d)=100d, sigma=50, k=2 -> +/-100
        distance = SpotterDistance(weight_tau_ms=20.0)
        distance.fit(model=model)

        rtt = 20.0
        circles = distance.estimate(
            {"anchor-a": rtt, "anchor-b": rtt}, ANCHOR_COORDS
        )

        self.assertEqual(len(circles), 2)
        for circle in circles:
            self.assertIn(circle.vp_ip, {"anchor-a", "anchor-b"})
            self.assertEqual(
                (circle.vp_lat, circle.vp_lon), ANCHOR_COORDS[circle.vp_ip]
            )
            self.assertEqual(circle.rtt_ms, rtt)
            # mu(20) = 2000, k*sigma = 100 -> [1900, 2100]
            self.assertAlmostEqual(circle.inner_radius_km, 1900.0)
            self.assertAlmostEqual(circle.radius_km, 2100.0)
            self.assertAlmostEqual(circle.weight, math.exp(-rtt / 20.0))

    def test_estimate_clips_inner_radius_at_zero(self):
        """When k*sigma > mu, inner_radius_km should clamp to 0 (full disk)."""
        # mu(rtt) = 20*rtt, sigma = 50, k = 2 -> band = +/- 100
        # At rtt = 1.0: mu = 20, inner_raw = -80 -> clipped to 0; outer = 120
        model = make_fitted_spotter_model(
            p_mu=np.array([20.0, 0.0]),
            p_sigma=np.array([50.0]),
            k=2.0,
            rtt_min=0.0,
            rtt_max=20.0,
        )
        distance = SpotterDistance()
        distance.fit(model=model)

        circles = distance.estimate({"anchor-a": 1.0}, ANCHOR_COORDS)

        self.assertEqual(len(circles), 1)
        self.assertEqual(circles[0].inner_radius_km, 0.0)
        self.assertAlmostEqual(circles[0].radius_km, 120.0)

    def test_estimate_skips_degenerate_zero_width_band(self):
        """A zero-width band (sigma = 0, so inner == outer) yields no constraint."""
        distance = SpotterDistance()
        distance.fit(model=make_fitted_degenerate_spotter_model())

        circles = distance.estimate({"anchor-a": 20.0}, ANCHOR_COORDS)

        self.assertEqual(circles, [])

    def test_estimate_skips_unfitted_model(self):
        distance = SpotterDistance()
        distance.fit(model=make_unfitted_spotter_model())

        circles = distance.estimate({"anchor-a": 20.0}, ANCHOR_COORDS)

        self.assertEqual(circles, [])

    def test_estimate_skips_rtt_outside_model_range(self):
        """predict_distance_bounds returns None outside [rtt_min, rtt_max] -- skip."""
        model = make_fitted_spotter_model(rtt_min=10.0, rtt_max=60.0)
        distance = SpotterDistance()
        distance.fit(model=model)

        circles = distance.estimate(
            {"anchor-a": 5.0, "anchor-b": 80.0, "anchor-c": 30.0},
            ANCHOR_COORDS,
        )

        # Only anchor-c (rtt=30) is in range
        self.assertEqual(len(circles), 1)
        self.assertEqual(circles[0].vp_ip, "anchor-c")

    def test_estimate_skips_unknown_anchors(self):
        distance = SpotterDistance()
        distance.fit(model=make_fitted_spotter_model())

        circles = distance.estimate({"unknown-anchor": 20.0}, ANCHOR_COORDS)

        self.assertEqual(circles, [])

    def test_estimate_applies_max_rtt_cutoff_before_prediction(self):
        distance = SpotterDistance(max_rtt_ms=10.0)
        distance.fit(model=make_fitted_spotter_model())

        circles = distance.estimate({"anchor-a": 10.1}, ANCHOR_COORDS)

        self.assertEqual(circles, [])

    def test_registered_under_name_spotter(self):
        """The distance variant is wired into the registry as 'spotter'."""
        from scripts.framework.registry import DISTANCE_REGISTRY

        self.assertIn("spotter", DISTANCE_REGISTRY)
        self.assertIs(DISTANCE_REGISTRY["spotter"], SpotterDistance)


if __name__ == "__main__":
    unittest.main()
