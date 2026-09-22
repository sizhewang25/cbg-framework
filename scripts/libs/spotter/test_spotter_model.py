"""Unit tests for the Spotter pooled RTT-distance model.

Covers the two pieces of the pipeline laid out in
notes/2026-05-17-spotter-normality-check.md:

  1. fit_mu_sigma(): bin-mean / bin-std polynomial fits for mu(d), sigma(d).
  2. SpotterRTTModel: fit + predict_distance_bounds + range gating,
     plus the 2/3*c baseline clip and Octant-style cutoff_rtt.

The model is *pooled*: one (p_mu, p_log_sigma) for all anchors, by design.
The annulus is mu(d) +/- sigma(d) -- the paper's published band (Figure 3a).

Synthetic data uses slope 50 km/ms (mu = 50 * rtt) so every probe sits
below the 2/3*c speed-of-internet line and survives the new baseline
filter in fit().
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.libs.spotter.spotter_model import (
    DEFAULT_MONOTONE_MARGIN,
    SpotterRTTModel,
    THEORETICAL_SLOPE,
    calibrate_k,
    compute_cutoff_rtt,
    fit_mu_sigma,
)


class TestFitMuSigma(unittest.TestCase):
    """Monotone mu on raw pairs + log-sigma from its residuals."""

    def test_returns_polynomial_coeff_shapes(self):
        rng = np.random.default_rng(42)
        rtts = rng.uniform(0, 100, size=5000)
        dists = 50.0 * rtts + rng.normal(0, 200, size=5000)

        p_mu, p_log_sigma, centers, mus, sigmas = fit_mu_sigma(
            rtts, dists, n_bins=40, min_per_bin=30, deg_mu=3, deg_sigma=2
        )

        self.assertEqual(p_mu.shape, (4,))  # cubic -> 4 coeffs
        self.assertEqual(p_log_sigma.shape, (3,))  # deg 2 -> 3 coeffs
        self.assertEqual(centers.shape, mus.shape)
        self.assertEqual(centers.shape, sigmas.shape)

    def test_mu_polynomial_recovers_linear_relationship(self):
        """If dist = 50 * rtt + noise, polyval(p_mu, rtt) ~ 50 * rtt."""
        rng = np.random.default_rng(0)
        rtts = rng.uniform(0, 100, size=10000)
        dists = 50.0 * rtts + rng.normal(0, 100, size=10000)

        p_mu, _, _, _, _ = fit_mu_sigma(rtts, dists, n_bins=40)

        # mu(50) should be approximately 2500 km
        self.assertAlmostEqual(float(np.polyval(p_mu, 50.0)), 2500.0, delta=100.0)

    def test_sigma_polynomial_recovers_noise_level(self):
        rng = np.random.default_rng(1)
        rtts = rng.uniform(0, 100, size=10000)
        noise_std = 250.0
        dists = 50.0 * rtts + rng.normal(0, noise_std, size=10000)

        _, p_log_sigma, _, _, sigmas = fit_mu_sigma(rtts, dists, n_bins=40)

        # `sigmas` stays a LINEAR descriptive per-bin std, so it is still
        # directly comparable to the true noise level.
        self.assertAlmostEqual(float(np.mean(sigmas)), noise_std, delta=40.0)
        # p_log_sigma is log-space: exponentiate before comparing.
        self.assertAlmostEqual(
            float(np.exp(np.polyval(p_log_sigma, 50.0))), noise_std, delta=50.0
        )

    def test_drops_underfilled_bins(self):
        """Bins with < min_per_bin points are dropped before polyfit."""
        rng = np.random.default_rng(2)
        dense = rng.normal(10, 1, size=2000)
        sparse = rng.uniform(50, 80, size=20)  # ~1 point per bin if 40 bins on [0, 80]
        rtts = np.concatenate([dense, sparse])
        dists = 50.0 * rtts + rng.normal(0, 100, size=len(rtts))

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
        rtts = rng.uniform(0, 100, size=20000)
        true_mu = 50.0 * rtts
        true_sigma = 200.0
        dists = true_mu + rng.normal(0, true_sigma, size=20000)

        p_mu = np.array([50.0, 0.0])
        p_log_sigma = np.array([np.log(true_sigma)])
        k = calibrate_k(rtts, dists, p_mu, p_log_sigma, target_coverage=0.95)

        self.assertAlmostEqual(k, 1.96, delta=0.05)

    def test_k_recovers_normal_quantile_at_50(self):
        rng = np.random.default_rng(8)
        rtts = rng.uniform(0, 100, size=20000)
        dists = 50.0 * rtts + rng.normal(0, 200, size=20000)
        p_mu = np.array([50.0, 0.0])
        p_log_sigma = np.array([np.log(200.0)])

        k = calibrate_k(rtts, dists, p_mu, p_log_sigma, target_coverage=0.50)

        # |z| has half-normal-like distribution; 50th percentile ~ 0.674
        self.assertAlmostEqual(k, 0.674, delta=0.05)

    def test_k_is_pooled_and_symmetric(self):
        """k is a single scalar -- one number for both inner and outer bound."""
        rng = np.random.default_rng(9)
        rtts = rng.uniform(0, 100, size=5000)
        dists = 50.0 * rtts + rng.normal(0, 200, size=5000)
        p_mu = np.array([50.0, 0.0])
        p_log_sigma = np.array([np.log(200.0)])

        k = calibrate_k(rtts, dists, p_mu, p_log_sigma, target_coverage=0.95)

        self.assertIsInstance(k, float)
        self.assertGreater(k, 0.0)


class TestComputeCutoffRtt(unittest.TestCase):
    """The Octant-style sparse-tail cutoff scan."""

    def test_cutoff_at_right_edge_of_last_dense_bin(self):
        rng = np.random.default_rng(11)
        # Dense block in [0, 50): 600 points -> ~60 per 5 ms bin (well above 30).
        # Anchor min at exactly 0.0 so the 5 ms bins align on [0, 5), [5, 10), ...
        dense = np.concatenate([[0.0], rng.uniform(0.0, 50.0, size=599)])
        # Sparse tail in [70, 80): ~10 points across two 5 ms bins.
        sparse = rng.uniform(70.0, 80.0, size=10)
        rtts = np.concatenate([dense, sparse])

        cutoff = compute_cutoff_rtt(rtts, bin_size_ms=5.0, cutoff_min_points=30)

        self.assertAlmostEqual(cutoff, 50.0, places=6)

    def test_cutoff_clamped_to_max_when_all_bins_dense(self):
        rng = np.random.default_rng(12)
        rtts = rng.uniform(0.0, 40.0, size=600)
        cutoff = compute_cutoff_rtt(rtts, bin_size_ms=5.0, cutoff_min_points=30)
        # Last dense bin's right edge would be 40.0 (or 40.0+5*x). Clamped to max.
        self.assertLessEqual(cutoff, float(rtts.max()))
        self.assertGreaterEqual(cutoff, 35.0)

    def test_cutoff_empty_input_returns_zero(self):
        self.assertEqual(compute_cutoff_rtt(np.array([])), 0.0)


class TestSpotterRTTModel(unittest.TestCase):
    """The class wrapping fit_mu_sigma."""

    def test_unfitted_model_has_fitted_false(self):
        model = SpotterRTTModel()
        self.assertFalse(model.fitted)
        self.assertIsNone(model.predict_distance_bounds(20.0))

    def test_fit_sets_state(self):
        rng = np.random.default_rng(42)
        rtts = rng.uniform(0, 100, size=5000)
        dists = 50.0 * rtts + rng.normal(0, 200, size=5000)

        model = SpotterRTTModel()
        ok = model.fit(rtts, dists)

        self.assertTrue(ok)
        self.assertTrue(model.fitted)
        self.assertIsNotNone(model.p_mu)
        self.assertIsNotNone(model.p_log_sigma)
        # Range bounds reflect the data (after the baseline filter).
        self.assertGreaterEqual(model.rtt_min, float(rtts.min()))
        self.assertLessEqual(model.rtt_max, float(rtts.max()))
        # cutoff_rtt is set during fit, within the data range.
        self.assertGreater(model.cutoff_rtt, model.rtt_min)
        self.assertLessEqual(model.cutoff_rtt, model.rtt_max)

    def test_fit_without_target_coverage_keeps_k_at_one(self):
        """Default fit leaves k = 1.0 -- the paper's mu +/- sigma band."""
        rng = np.random.default_rng(43)
        rtts = rng.uniform(0, 100, size=5000)
        dists = 50.0 * rtts + rng.normal(0, 200, size=5000)

        model = SpotterRTTModel()
        ok = model.fit(rtts, dists)

        self.assertTrue(ok)
        self.assertEqual(model.k, 1.0)
        self.assertNotIn("target_coverage", model.metadata)
        self.assertEqual(model.metadata["k"], 1.0)

    def test_fit_with_target_coverage_calibrates_k(self):
        """Passing target_coverage triggers empirical k calibration."""
        rng = np.random.default_rng(44)
        rtts = rng.uniform(0, 100, size=20000)
        dists = 50.0 * rtts + rng.normal(0, 200, size=20000)

        model = SpotterRTTModel()
        ok = model.fit(rtts, dists, target_coverage=0.95)

        self.assertTrue(ok)
        # |z| ~ half-normal; 95th percentile ~ 1.96 for well-fit residuals.
        self.assertGreater(model.k, 1.5)
        self.assertLess(model.k, 2.5)
        self.assertAlmostEqual(model.metadata["target_coverage"], 0.95)
        self.assertEqual(model.metadata["k"], model.k)

    def test_fit_drops_sub_baseline_rows(self):
        """Rows where rtt < THEORETICAL_SLOPE * dist are physically impossible
        and must be dropped before binning."""
        rng = np.random.default_rng(13)
        rtts = rng.uniform(10.0, 100.0, size=2000)
        # All-valid base set: dist = 50 * rtt -> rtt = THEORETICAL_SLOPE * dist * 2.
        dists = 50.0 * rtts
        # Inject 50 impossible rows where dist is twice what the rtt could support.
        bad_rtts = np.full(50, 10.0)
        bad_dists = np.full(50, 2000.0)  # 10 < 0.01 * 2000 = 20
        rtts = np.concatenate([rtts, bad_rtts])
        dists = np.concatenate([dists, bad_dists])

        model = SpotterRTTModel()
        ok = model.fit(rtts, dists)

        self.assertTrue(ok)
        self.assertEqual(model.metadata["n_pairs"], 2000)

    def test_predict_distance_bounds_in_range(self):
        """Manually constructed model: mu = 50*d, sigma = 50 -> +/-sigma band of width 100.

        cutoff_rtt is unset (=0) so the legacy rtt_max gate applies. At
        rtt=30, mu=1500, raw outer=1550, baseline=3000 -> no clip.
        """
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),
            p_log_sigma=np.array([np.log(50.0)]),
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )

        inner, outer = model.predict_distance_bounds(30.0)
        self.assertAlmostEqual(inner, 1450.0, places=3)
        self.assertAlmostEqual(outer, 1550.0, places=3)
        self.assertLess(inner, outer)

    def test_predict_distance_bounds_uses_k_when_set(self):
        """With k = 2.0 the band widens to mu +/- 2*sigma.

        mu = 50*d, sigma = 50, k = 2 -> half-band = 100. At rtt=30: mu=1500,
        raw outer=1600, baseline=3000 -> no clip; raw inner=1400 (no clip).
        """
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),
            p_log_sigma=np.array([np.log(50.0)]),
            k=2.0,
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )

        inner, outer = model.predict_distance_bounds(30.0)
        self.assertAlmostEqual(inner, 1400.0, places=3)
        self.assertAlmostEqual(outer, 1600.0, places=3)

    def test_predict_distance_bounds_clips_inner_at_zero(self):
        """When mu - sigma < 0, inner clamps to 0; baseline cap may still apply."""
        model = SpotterRTTModel(
            p_mu=np.array([20.0, 0.0]),    # mu = 20 * rtt
            p_log_sigma=np.array([np.log(50.0)]),  # sigma = 50 km
            rtt_min=0.0,
            rtt_max=20.0,
            fitted=True,
        )

        # At rtt = 1.0: mu = 20, sigma = 50.
        # Inner raw = -30 -> clipped to 0.
        # Outer raw = 70, baseline = 1 / 0.01 = 100 -> no clip.
        inner, outer = model.predict_distance_bounds(1.0)
        self.assertEqual(inner, 0.0)
        self.assertAlmostEqual(outer, 70.0, places=3)

    def test_predict_distance_bounds_clips_outer_by_baseline(self):
        """Outer = min(mu + sigma, rtt / THEORETICAL_SLOPE)."""
        model = SpotterRTTModel(
            p_mu=np.array([100.0, 0.0]),   # mu = 100 * rtt -- right on the baseline
            p_log_sigma=np.array([np.log(50.0)]),
            rtt_min=0.0,
            rtt_max=100.0,
            fitted=True,
        )

        # At rtt = 20: mu = 2000, raw outer = 2050, baseline cap = 20 / 0.01 = 2000.
        # Inner stays at mu - sigma = 1950 (no inner clip).
        inner, outer = model.predict_distance_bounds(20.0)
        self.assertAlmostEqual(outer, 2000.0, places=3)
        self.assertAlmostEqual(inner, 1950.0, places=3)

    def test_predict_baseline_clip_degenerates_when_mu_exceeds_baseline(self):
        """When mu > rtt / THEORETICAL_SLOPE the outer clamp pushes outer
        below the inner band -- the wrapper reads that as a degenerate ring."""
        model = SpotterRTTModel(
            p_mu=np.array([100.0, 200.0]),  # mu = 100*rtt + 200
            p_log_sigma=np.array([np.log(50.0)]),
            rtt_min=0.0,
            rtt_max=100.0,
            fitted=True,
        )

        # At rtt = 1: mu = 300, raw outer = 350, baseline = 100 -> outer clipped to 100.
        # inner = max(0, 300 - 50) = 250 > outer.
        inner, outer = model.predict_distance_bounds(1.0)
        self.assertAlmostEqual(outer, 100.0, places=3)
        self.assertAlmostEqual(inner, 250.0, places=3)
        self.assertLess(outer, inner)

    def test_predict_uses_line_through_origin_below_rtt_min(self):
        """For rtt < rtt_min: inner = 0 and outer scales linearly from 0
        at rtt=0 to outer(rtt_min) at rtt=rtt_min. Mirrors the Octant
        hull convention below the leftmost vertex."""
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),    # mu = 50 * rtt
            p_log_sigma=np.array([np.log(50.0)]),  # sigma = 50 km
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )

        # At rtt_min=10: mu=500, sigma=50, raw outer = 550, baseline cap = 10/0.01 = 1000,
        # so outer_at_min = 550.
        # At rtt=5 (half of rtt_min): inner=0, outer = (550/10) * 5 = 275.
        inner, outer = model.predict_distance_bounds(5.0)
        self.assertEqual(inner, 0.0)
        self.assertAlmostEqual(outer, 275.0, places=3)
        # At rtt=0: both collapse to 0.
        inner0, outer0 = model.predict_distance_bounds(0.0)
        self.assertEqual(inner0, 0.0)
        self.assertEqual(outer0, 0.0)

    def test_predict_distance_bounds_returns_none_above_rtt_max_when_no_cutoff(self):
        """Legacy gate: with cutoff_rtt unset (0), rtt > rtt_max returns None."""
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),
            p_log_sigma=np.array([np.log(50.0)]),
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )
        self.assertIsNone(model.predict_distance_bounds(80.0))

    def test_predict_extends_outer_at_finite_sentinel_slope_above_cutoff(self):
        """With cutoff_rtt set: inner is held flat at inner(cutoff); outer
        extends from outer(cutoff) toward the fictitious sentinel
        z = (sentinel_rtt, sentinel_rtt / THEORETICAL_SLOPE) on the 2/3*c
        bound — Octant paper's smooth-transition construction."""
        cutoff_rtt = 50.0
        sentinel_rtt = 10000.0
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),    # mu = 50 * rtt
            p_log_sigma=np.array([np.log(50.0)]),  # sigma = 50 km (constant)
            rtt_min=10.0,
            rtt_max=100.0,
            cutoff_rtt=cutoff_rtt,
            sentinel_rtt=sentinel_rtt,
            fitted=True,
        )

        below = model.predict_distance_bounds(30.0)
        at_cutoff = model.predict_distance_bounds(50.0)
        above = model.predict_distance_bounds(80.0)

        # Below cutoff: mu=1500, sigma=50, raw outer=1550, baseline=3000 -> no clip.
        self.assertAlmostEqual(below[0], 1450.0, places=3)
        self.assertAlmostEqual(below[1], 1550.0, places=3)
        # At cutoff: mu=2500, sigma=50, raw outer=2550.
        self.assertAlmostEqual(at_cutoff[0], 2450.0, places=3)
        self.assertAlmostEqual(at_cutoff[1], 2550.0, places=3)
        # Above cutoff: inner stays at 2450 (flat); outer extends with finite-
        # sentinel slope = (10000/0.01 - 2550) / (10000 - 50) ≈ 100.2462 km/ms.
        outer_at_cutoff = 2550.0
        sentinel_dist = sentinel_rtt / THEORETICAL_SLOPE
        slope = (sentinel_dist - outer_at_cutoff) / (sentinel_rtt - cutoff_rtt)
        self.assertAlmostEqual(above[0], 2450.0, places=3)
        self.assertAlmostEqual(above[1], outer_at_cutoff + slope * 30.0, places=3)

    def test_large_sentinel_recovers_baseline_slope_above_cutoff(self):
        """As sentinel_rtt → ∞, the cutoff extension collapses to the
        asymptotic 1/THEORETICAL_SLOPE slope (parallel to 2/3*c)."""
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),
            p_log_sigma=np.array([np.log(50.0)]),
            rtt_min=10.0,
            rtt_max=100.0,
            cutoff_rtt=50.0,
            sentinel_rtt=1e9,
            fitted=True,
        )
        above = model.predict_distance_bounds(80.0)
        # outer(50)=2550; large-sentinel slope ≈ 1/THEORETICAL_SLOPE so
        # outer(80) ≈ 2550 + 30/THEORETICAL_SLOPE = 5550.
        self.assertAlmostEqual(above[0], 2450.0, places=3)
        self.assertAlmostEqual(above[1], 5550.0, places=2)

    def test_predict_raises_when_rtt_exceeds_sentinel(self):
        # rtt past the sentinel would push the outer above the 2/3*c bound.
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),
            p_log_sigma=np.array([np.log(50.0)]),
            rtt_min=10.0,
            rtt_max=10000.0,
            cutoff_rtt=50.0,
            sentinel_rtt=200.0,
            fitted=True,
        )
        with self.assertRaises(ValueError):
            model.predict_distance_bounds(500.0)

    def test_predict_distance_returns_mu_only(self):
        """predict_distance gives the pooled mean (no band)."""
        model = SpotterRTTModel(
            p_mu=np.array([50.0, 0.0]),
            p_log_sigma=np.array([np.log(50.0)]),
            rtt_min=10.0,
            rtt_max=60.0,
            fitted=True,
        )
        self.assertAlmostEqual(model.predict_distance(30.0), 1500.0, places=3)

    def test_fit_then_predict_is_self_consistent(self):
        """End-to-end: fit on synthetic data, predict at midpoint, sanity-check bounds."""
        rng = np.random.default_rng(123)
        rtts = rng.uniform(0, 100, size=10000)
        dists = 50.0 * rtts + rng.normal(0, 200, size=10000)

        model = SpotterRTTModel()
        model.fit(rtts, dists)

        bounds = model.predict_distance_bounds(50.0)
        self.assertIsNotNone(bounds)
        inner, outer = bounds
        self.assertLess(inner, outer)
        # mu(50) should be ~2500, sigma ~ 200 -> +/-sigma band of width ~ 400.
        # Baseline at rtt=50 is 5000, so outer won't be clipped.
        center = 0.5 * (inner + outer)
        self.assertAlmostEqual(center, 2500.0, delta=200.0)
        self.assertAlmostEqual(outer - inner, 2 * 200.0, delta=200.0)

    def test_sigma_is_positive_everywhere_by_construction(self):
        """Regression guard for the pathology the log parameterisation removes.

        The old `deg_sigma=2` fit was unconstrained in linear space and on real
        data dipped below zero at the low-RTT end -- *inside* the range it was
        fitted on. Measured by
        `scripts.analysis.v3.modules.figure_spotter_normality`: roots at 5.51 ms
        (as01), 2.22 (as02), 2.40 (as03), affecting 2,088 / 444 / 565 rows. The
        band then inverted and `predict_distance_bounds` returned a degenerate
        `(0, 0)`, which `normal_dist` maps to DEGENERATE_REGION -- so the VP
        silently stopped participating in the MTL at those RTTs. See
        notes/2026-09-18-spotter-normality-operator-mesh.md.

        Fitting `log sigma` makes that unreachable: `exp` has no zero. This test
        pins that, using the same left-edge shape that used to break.
        """
        rng = np.random.default_rng(17)
        # Steep slope + wide spread at the left edge is what produced a negative
        # sigma root under the old fit.
        rtts = rng.uniform(1.0, 80.0, size=8000)
        dists = np.abs(100.0 * rtts - 400.0 + rng.normal(0, 300, size=8000))

        model = SpotterRTTModel()
        self.assertTrue(model.fit(rtts, dists, min_per_bin=5, cutoff_min_points=5))

        grid = np.linspace(0.0, 300.0, 2000)
        sigmas = np.array([model.sigma_at(float(r)) for r in grid])
        self.assertTrue(np.all(sigmas > 0.0), "sigma must be positive everywhere")
        self.assertTrue(np.all(np.isfinite(sigmas)))

        # And the band no longer inverts at the left edge.
        inner, outer = model.predict_distance_bounds(2.0)
        self.assertLess(inner, outer)

    def test_mu_is_monotone_and_non_negative(self):
        """More delay cannot mean less distance, and distance cannot be < 0.

        The unconstrained `np.polyfit` guaranteed neither. On as01 it produced
        mu(0) = -228.5 km and, extrapolating, mu(200 ms) = 891.7 km -- i.e. the
        curve turned over and dived. notes/2026-05-17-spotter-normality-check.md
        records the same failure reaching -60,000 km on a narrow-support slice.

        The guarantee is scoped to [0, monotone_margin * max(rtt)], which is
        where the NNLS constraint is imposed. Callers never evaluate beyond it
        because `predict_*` clamps to `cutoff_rtt <= rtt_max`.
        """
        rng = np.random.default_rng(23)
        # Saturating shape: the case most likely to make a cubic turn over.
        rtts = rng.uniform(1.0, 80.0, size=6000)
        dists = 3000.0 * (1.0 - np.exp(-rtts / 25.0)) + rng.normal(0, 250, size=6000)
        dists = np.abs(dists)

        p_mu, _, _, _, _ = fit_mu_sigma(rtts, dists, min_per_bin=5)

        hi = float(rtts.max()) * DEFAULT_MONOTONE_MARGIN
        grid = np.linspace(0.0, hi, 4000)
        self.assertGreaterEqual(float(np.polyval(np.polyder(p_mu), grid).min()), -1e-6)
        self.assertGreaterEqual(float(np.polyval(p_mu, grid).min()), -1e-6)
        self.assertGreaterEqual(float(np.polyval(p_mu, 0.0)), -1e-6)

    def test_deg_mu_other_than_three_is_rejected(self):
        """The Bernstein construction is cubic-specific; fail rather than
        silently ignore the caller's degree."""
        rng = np.random.default_rng(31)
        rtts = rng.uniform(1.0, 80.0, size=500)
        dists = 50.0 * rtts
        with self.assertRaises(ValueError):
            fit_mu_sigma(rtts, dists, deg_mu=1, min_per_bin=5)


if __name__ == "__main__":
    unittest.main()
