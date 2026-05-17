"""Unit tests for the Spotter pooled RTT-distance model.

Covers the three pieces of the pipeline laid out in
notes/2026-05-17-spotter-normality-check.md:

  1. fit_mu_sigma(): bin-mean / bin-std polynomial fits for mu(d), sigma(d).
  2. calibrate_k():  empirical k = quantile(|z|, target_coverage).
  3. SpotterRTTModel: fit + predict_distance_bounds + range gating.

The model is *pooled*: one (p_mu, p_sigma, k) for all anchors, by design.
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.libs.spotter.spotter_model import (
    SpotterRTTModel,
    calibrate_k,
    fit_mu_sigma,
)


class TestFitMuSigma(unittest.TestCase):
    """Bin-and-polyfit step (lifted from spotter_normality_check.fit_mu_sigma)."""

    def test_returns_polynomial_coeff_shapes(self):
        rng = np.random.default_rng(42)
        rtts = rng.uniform(0, 100, size=5000)
        dists = 100.0 * rtts + rng.normal(0, 200, size=5000)

        p_mu, p_sigma, centers, mus, sigmas = fit_mu_sigma(
            rtts, dists, n_bins=40, min_per_bin=30, deg_mu=3, deg_sigma=2
        )

        self.assertEqual(p_mu.shape, (4,))  # deg 3 -> 4 coeffs
        self.assertEqual(p_sigma.shape, (3,))  # deg 2 -> 3 coeffs
        self.assertEqual(centers.shape, mus.shape)
        self.assertEqual(centers.shape, sigmas.shape)

    def test_mu_polynomial_recovers_linear_relationship(self):
        """If dist = 100 * rtt + noise, polyval(p_mu, rtt) ~ 100 * rtt."""
        rng = np.random.default_rng(0)
        rtts = rng.uniform(0, 100, size=10000)
        dists = 100.0 * rtts + rng.normal(0, 100, size=10000)

        p_mu, _, _, _, _ = fit_mu_sigma(rtts, dists, n_bins=40)

        # mu(50) should be approximately 5000 km
        self.assertAlmostEqual(float(np.polyval(p_mu, 50.0)), 5000.0, delta=100.0)

    def test_sigma_polynomial_recovers_noise_level(self):
        rng = np.random.default_rng(1)
        rtts = rng.uniform(0, 100, size=10000)
        noise_std = 250.0
        dists = 100.0 * rtts + rng.normal(0, noise_std, size=10000)

        _, p_sigma, _, _, sigmas = fit_mu_sigma(rtts, dists, n_bins=40)

        # Mean of binned sigmas should be near the true noise std
        self.assertAlmostEqual(float(np.mean(sigmas)), noise_std, delta=40.0)
        # polyval at any RTT in the fit range should also be in that range
        self.assertAlmostEqual(float(np.polyval(p_sigma, 50.0)), noise_std, delta=50.0)

    def test_drops_underfilled_bins(self):
        """Bins with < min_per_bin points are dropped before polyfit."""
        # Most data clustered near rtt=10, sparse elsewhere
        rng = np.random.default_rng(2)
        dense = rng.normal(10, 1, size=2000)
        sparse = rng.uniform(50, 80, size=20)  # ~1 point per bin if 40 bins on [0, 80]
        rtts = np.concatenate([dense, sparse])
        dists = 100.0 * rtts + rng.normal(0, 100, size=len(rtts))

        _, _, centers, _, _ = fit_mu_sigma(
            rtts, dists, n_bins=40, min_per_bin=30
        )

        # Sparse region bin centers should be excluded
        self.assertTrue(np.all(centers < 40.0))


class TestCalibrateK(unittest.TestCase):
    """Empirical k = quantile(|z|, target_coverage)."""

    def test_k_recovers_normal_quantile_at_95(self):
        """For N(0,1) residuals, k(0.95) should be near 1.96."""
        rng = np.random.default_rng(7)
        # Linear mu, constant sigma: residuals are exactly N(0, 1) after standardization
        rtts = rng.uniform(0, 100, size=20000)
        true_mu = 100.0 * rtts
        true_sigma = 200.0
        dists = true_mu + rng.normal(0, true_sigma, size=20000)

        p_mu = np.array([100.0, 0.0])  # degree 1: 100*x + 0
        p_sigma = np.array([true_sigma])  # degree 0: constant
        k = calibrate_k(rtts, dists, p_mu, p_sigma, target_coverage=0.95)

        self.assertAlmostEqual(k, 1.96, delta=0.05)

    def test_k_recovers_normal_quantile_at_50(self):
        rng = np.random.default_rng(8)
        rtts = rng.uniform(0, 100, size=20000)
        dists = 100.0 * rtts + rng.normal(0, 200, size=20000)
        p_mu = np.array([100.0, 0.0])
        p_sigma = np.array([200.0])

        k = calibrate_k(rtts, dists, p_mu, p_sigma, target_coverage=0.50)

        # |z| has half-normal-like distribution; 50th percentile ~ 0.674
        self.assertAlmostEqual(k, 0.674, delta=0.05)

    def test_k_is_pooled_and_symmetric(self):
        """k is a single scalar -- one number for both inner and outer bound."""
        rng = np.random.default_rng(9)
        rtts = rng.uniform(0, 100, size=5000)
        dists = 100.0 * rtts + rng.normal(0, 200, size=5000)
        p_mu = np.array([100.0, 0.0])
        p_sigma = np.array([200.0])

        k = calibrate_k(rtts, dists, p_mu, p_sigma, target_coverage=0.95)

        self.assertIsInstance(k, float)
        self.assertGreater(k, 0.0)


class TestSpotterRTTModel(unittest.TestCase):
    """The class wrapping fit_mu_sigma + calibrate_k."""

    def test_unfitted_model_has_fitted_false(self):
        model = SpotterRTTModel()
        self.assertFalse(model.fitted)
        self.assertIsNone(model.predict_distance_bounds(20.0))

    def test_fit_sets_state(self):
        rng = np.random.default_rng(42)
        rtts = rng.uniform(0, 100, size=5000)
        dists = 100.0 * rtts + rng.normal(0, 200, size=5000)

        model = SpotterRTTModel()
        ok = model.fit(rtts, dists, target_coverage=0.95)

        self.assertTrue(ok)
        self.assertTrue(model.fitted)
        self.assertIsNotNone(model.p_mu)
        self.assertIsNotNone(model.p_sigma)
        self.assertGreater(model.k, 0.0)
        # Range gating bounds match the data
        self.assertAlmostEqual(model.rtt_min, float(rtts.min()), places=3)
        self.assertAlmostEqual(model.rtt_max, float(rtts.max()), places=3)

    def test_predict_distance_bounds_in_range(self):
        """Manually constructed model: mu = 100*d, sigma = 50, k = 2 -> (-100, +100) band."""
        model = SpotterRTTModel(
            p_mu=np.array([100.0, 0.0]),
            p_sigma=np.array([50.0]),
            k=2.0,
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )

        inner, outer = model.predict_distance_bounds(30.0)
        self.assertAlmostEqual(inner, 2900.0, places=3)
        self.assertAlmostEqual(outer, 3100.0, places=3)
        self.assertLess(inner, outer)

    def test_predict_distance_bounds_clips_inner_at_zero(self):
        """When mu - k*sigma < 0, inner clamps to 0 (full disk)."""
        model = SpotterRTTModel(
            p_mu=np.array([20.0, 0.0]),    # mu = 20 * rtt
            p_sigma=np.array([50.0]),       # sigma = 50
            k=2.0,                          # band = +/- 100
            rtt_min=0.0,
            rtt_max=20.0,
            fitted=True,
        )

        inner, outer = model.predict_distance_bounds(1.0)  # mu = 20, sigma = 50, k*sigma = 100
        self.assertEqual(inner, 0.0)
        self.assertAlmostEqual(outer, 120.0, places=3)

    def test_predict_distance_bounds_returns_none_below_range(self):
        model = SpotterRTTModel(
            p_mu=np.array([100.0, 0.0]),
            p_sigma=np.array([50.0]),
            k=2.0,
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )
        self.assertIsNone(model.predict_distance_bounds(5.0))

    def test_predict_distance_bounds_returns_none_above_range(self):
        model = SpotterRTTModel(
            p_mu=np.array([100.0, 0.0]),
            p_sigma=np.array([50.0]),
            k=2.0,
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )
        self.assertIsNone(model.predict_distance_bounds(80.0))

    def test_predict_distance_returns_mu_only(self):
        """predict_distance gives the pooled mean (no band)."""
        model = SpotterRTTModel(
            p_mu=np.array([100.0, 0.0]),
            p_sigma=np.array([50.0]),
            k=2.0,
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )
        self.assertAlmostEqual(model.predict_distance(30.0), 3000.0, places=3)

    def test_fit_then_predict_is_self_consistent(self):
        """End-to-end: fit on synthetic data, predict at midpoint, sanity-check bounds."""
        rng = np.random.default_rng(123)
        rtts = rng.uniform(0, 100, size=10000)
        dists = 100.0 * rtts + rng.normal(0, 200, size=10000)

        model = SpotterRTTModel()
        model.fit(rtts, dists, target_coverage=0.95)

        bounds = model.predict_distance_bounds(50.0)
        self.assertIsNotNone(bounds)
        inner, outer = bounds
        self.assertLess(inner, outer)
        # mu(50) should be ~5000, k near 1.96, sigma ~ 200 -> band ~ +/- 392
        center = 0.5 * (inner + outer)
        self.assertAlmostEqual(center, 5000.0, delta=200.0)
        self.assertAlmostEqual(outer - inner, 2 * 1.96 * 200.0, delta=200.0)


if __name__ == "__main__":
    unittest.main()
