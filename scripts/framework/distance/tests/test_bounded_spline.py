"""Tests for the Octant bounded-spline distance-model wrapper."""

from __future__ import annotations

import math
import unittest

from scripts.framework.distance.bounded_spline import BoundedSplineDistance
from scripts.framework.distance.tests.helpers import (
    ANCHOR_COORDS,
    make_fitted_degenerate_octant_model,
    make_fitted_octant_model,
    make_unfitted_octant_model,
)


class TestBoundedSplineDistance(unittest.TestCase):
    def test_fit_accepts_prefitted_models_and_delta(self):
        models = {"anchor-a": make_fitted_octant_model("anchor-a")}
        distance = BoundedSplineDistance()

        distance.fit(models=models, delta=7.5)

        self.assertIs(distance.models, models)
        self.assertEqual(distance.delta, 7.5)

    def test_estimate_creates_annular_constraints_with_weights(self):
        model = make_fitted_octant_model("anchor-a")
        distance = BoundedSplineDistance(weight_tau_ms=20.0)
        distance.fit(models={"anchor-a": model}, delta=1.2)

        rtt = 20.0
        circles = distance.estimate({"anchor-a": rtt}, ANCHOR_COORDS)

        self.assertEqual(len(circles), 1)
        circle = circles[0]
        self.assertEqual(circle.vp_ip, "anchor-a")
        self.assertEqual((circle.vp_lat, circle.vp_lon), ANCHOR_COORDS["anchor-a"])
        self.assertEqual(circle.rtt_ms, rtt)
        self.assertAlmostEqual(circle.inner_radius_km, 1900.0)
        self.assertAlmostEqual(circle.radius_km, 2100.0)
        self.assertAlmostEqual(circle.weight, math.exp(-rtt / 20.0))

    def test_estimate_skips_degenerate_zero_bounds(self):
        model = make_fitted_degenerate_octant_model("anchor-a")
        distance = BoundedSplineDistance()
        distance.fit(models={"anchor-a": model})

        circles = distance.estimate({"anchor-a": 20.0}, ANCHOR_COORDS)

        self.assertEqual(circles, [])

    def test_estimate_skips_unfitted_and_prediction_error_models(self):
        distance = BoundedSplineDistance()
        distance.fit(models={
            "anchor-a": make_unfitted_octant_model("anchor-a"),
            # Real fitted model with no spline raises when delta-band prediction is requested.
            "anchor-b": make_fitted_octant_model("anchor-b", fit_spline=False),
        }, delta=1.2)

        circles = distance.estimate(
            {
                "anchor-a": 20.0,
                "anchor-b": 20.0,
                "unknown-anchor": 20.0,
            },
            ANCHOR_COORDS,
        )

        self.assertEqual(circles, [])

    def test_estimate_applies_cutoff_before_prediction(self):
        distance = BoundedSplineDistance(max_rtt_ms=10.0)
        distance.fit(models={"anchor-a": make_fitted_octant_model("anchor-a")})

        circles = distance.estimate({"anchor-a": 10.1}, ANCHOR_COORDS)

        self.assertEqual(circles, [])


if __name__ == "__main__":
    unittest.main()
