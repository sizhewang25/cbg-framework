"""Tests for the LP low-envelope distance-model wrapper."""

from __future__ import annotations

import unittest

from scripts.framework.distance.low_envelope import LowEnvelopeDistance
from scripts.framework.distance.tests.helpers import (
    ANCHOR_COORDS,
    make_fitted_low_envelope_model,
    make_unfitted_low_envelope_model,
)


class TestLowEnvelopeDistance(unittest.TestCase):
    def test_fit_accepts_prefitted_models(self):
        models = {"anchor-a": make_fitted_low_envelope_model("anchor-a")}
        distance = LowEnvelopeDistance()

        distance.fit(models=models)

        self.assertIs(distance.models, models)

    def test_estimate_uses_only_fitted_positive_model_predictions(self):
        distance = LowEnvelopeDistance()
        distance.fit(models={
            "anchor-a": make_fitted_low_envelope_model("anchor-a"),
            "anchor-b": make_unfitted_low_envelope_model("anchor-b"),
            "anchor-c": make_fitted_low_envelope_model("anchor-c"),
        })

        circles = distance.estimate(
            {
                "anchor-a": 25.0,
                "anchor-b": 25.0,
                "anchor-c": 4.0,
                "unknown-anchor": 25.0,
            },
            ANCHOR_COORDS,
        )

        self.assertEqual(len(circles), 1)
        circle = circles[0]
        self.assertEqual(circle.vp_ip, "anchor-a")
        self.assertEqual((circle.vp_lat, circle.vp_lon), ANCHOR_COORDS["anchor-a"])
        self.assertEqual(circle.rtt_ms, 25.0)
        self.assertAlmostEqual(circle.radius_km, 1000.0)
        self.assertEqual(circle.inner_radius_km, 0.0)

    def test_estimate_applies_cutoff_before_prediction(self):
        distance = LowEnvelopeDistance(max_rtt_ms=10.0)
        distance.fit(models={"anchor-a": make_fitted_low_envelope_model("anchor-a")})

        circles = distance.estimate({"anchor-a": 10.1}, ANCHOR_COORDS)

        self.assertEqual(circles, [])


if __name__ == "__main__":
    unittest.main()
