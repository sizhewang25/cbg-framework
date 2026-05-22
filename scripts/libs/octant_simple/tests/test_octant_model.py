"""Unit tests for the simplified Octant model (high-cutoff variant only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.libs.octant_simple.octant_model import (
    DeltaSearchError,
    OctantFitError,
    OctantRTTModel,
    SplineFitError,
    THEORETICAL_SLOPE,
    compute_convex_hull_bounds,
    find_delta_for_coverage,
    fit_rtt_distance_spline,
    hull_inner_distance,
    hull_outer_distance,
)


# Hand-built parallel ±100 km band at 50 km/ms slope: for every RTT x in
# {10..50}, two probes at distance 50·x − 100 and 50·x + 100. So at RTT=20,
# hull is [900, 1100]. The 50 km/ms slope keeps every probe above the
# 2/3·c speed-of-internet line so OctantRTTModel.fit's baseline filter
# leaves the data intact.
def _band_samples():
    rtt_values = np.array([10, 20, 30, 40, 50], dtype=float)
    rtts = np.repeat(rtt_values, 2)
    distances = np.ravel(
        [[50.0 * rtt - 100.0, 50.0 * rtt + 100.0] for rtt in rtt_values]
    )
    return rtts, distances


class TestComputeConvexHullBounds(unittest.TestCase):
    def test_chains_enclose_the_parallel_band(self):
        """Hull at rtt=20 must bracket the data points (900, 1100)."""
        rtts, distances = _band_samples()
        hull = compute_convex_hull_bounds(
            rtts, distances, cutoff_min_points=1, bin_size_ms=1000
        )
        self.assertTrue(hull["success"], msg=hull["message"])
        upper_at_20 = hull_outer_distance(
            20.0, hull["hull_upper_rtts"], hull["hull_upper_distances"],
            cutoff_rtt=hull["cutoff_rtt"],
        )
        lower_at_20 = hull_inner_distance(
            20.0, hull["hull_lower_rtts"], hull["hull_lower_distances"],
            cutoff_rtt=hull["cutoff_rtt"],
        )
        self.assertAlmostEqual(upper_at_20, 1100.0)
        self.assertAlmostEqual(lower_at_20, 900.0)

    def test_returns_failure_when_too_few_points(self):
        hull = compute_convex_hull_bounds(np.array([10.0]), np.array([100.0]))
        self.assertFalse(hull["success"])
        self.assertEqual(hull["cutoff_rtt"], 0.0)

    def test_cutoff_rtt_clamped_to_max_rtt(self):
        # 10 points in a single bin (bin_size 1000) → that bin is dense,
        # bin scan would yield 10 + 1000 = 1010, but `cutoff_rtt` is clamped
        # to `max_rtt = 50` (the trusted region can never extend past the
        # last data point).
        rtts, distances = _band_samples()
        hull = compute_convex_hull_bounds(
            rtts, distances, cutoff_min_points=1, bin_size_ms=1000
        )
        self.assertAlmostEqual(hull["cutoff_rtt"], 50.0)

    def test_sparse_bin_caps_cutoff(self):
        # Dense bin starts at min_rtt=12 with bin_size=5 → right edge = 17.
        # The single outlier at rtt=50 is in a sparse bin, so cutoff stays at 17.
        rtts = np.concatenate([np.full(5, 12.0), np.array([50.0])])
        dists = np.concatenate([np.full(5, 1000.0), np.array([5000.0])])
        hull = compute_convex_hull_bounds(
            rtts, dists, cutoff_min_points=5, bin_size_ms=5
        )
        self.assertAlmostEqual(hull["cutoff_rtt"], 17.0)


class TestHullOuterDistance(unittest.TestCase):
    def test_interpolates_between_vertices(self):
        # Two-vertex hull: (10, 100) → (50, 500). At rtt=30 → 300.
        d = hull_outer_distance(
            30.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=1000.0
        )
        self.assertAlmostEqual(d, 300.0)

    def test_extends_above_cutoff_with_finite_sentinel_slope(self):
        # rtt=60, cutoff=50, hull tops out at 500. With the default
        # sentinel at 10_000 ms on the 2/3·c line, the slope to z is
        # (10_000/baseline − 500) / (10_000 − 50), strictly greater than the
        # asymptotic 1/baseline.
        cutoff_rtt, cutoff_dist = 50.0, 500.0
        sentinel_rtt = 10000.0
        sentinel_dist = sentinel_rtt / THEORETICAL_SLOPE
        slope = (sentinel_dist - cutoff_dist) / (sentinel_rtt - cutoff_rtt)
        self.assertGreater(slope, 1.0 / THEORETICAL_SLOPE)
        d = hull_outer_distance(
            60.0, [10.0, 50.0], [100.0, cutoff_dist], cutoff_rtt=cutoff_rtt,
            baseline_slope=THEORETICAL_SLOPE, sentinel_rtt=sentinel_rtt,
        )
        self.assertAlmostEqual(d, cutoff_dist + slope * (60.0 - cutoff_rtt))

    def test_extension_reaches_sentinel_on_two_thirds_c_line(self):
        # At rtt = sentinel_rtt, the extension should land exactly on the
        # 2/3·c bound (the paper's "conservative constraint") — that's the
        # whole point of the sentinel.
        sentinel_rtt = 1000.0
        d = hull_outer_distance(
            sentinel_rtt, [10.0, 50.0], [100.0, 500.0],
            cutoff_rtt=50.0,
            baseline_slope=THEORETICAL_SLOPE,
            sentinel_rtt=sentinel_rtt,
        )
        self.assertAlmostEqual(d, sentinel_rtt / THEORETICAL_SLOPE)

    def test_large_sentinel_approaches_baseline_slope_limit(self):
        # As sentinel → ∞, slope → 1/baseline_slope (parallel to 2/3·c).
        d = hull_outer_distance(
            60.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=50.0,
            baseline_slope=THEORETICAL_SLOPE, sentinel_rtt=1e9,
        )
        self.assertAlmostEqual(d, 500.0 + 10.0 / THEORETICAL_SLOPE, places=2)

    def test_raises_when_rtt_exceeds_sentinel(self):
        # rtt past the sentinel would push the extension above the 2/3·c
        # bound — violating the speed-of-internet constraint that the
        # sentinel was placed to honor.
        with self.assertRaises(ValueError):
            hull_outer_distance(
                60.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=50.0,
                baseline_slope=THEORETICAL_SLOPE, sentinel_rtt=55.0,
            )

    def test_raises_when_sentinel_not_above_cutoff(self):
        # Misconfiguration: the sentinel must sit strictly to the right of
        # the cutoff for the smooth-transition construction to be defined.
        with self.assertRaises(ValueError):
            hull_outer_distance(
                60.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=50.0,
                baseline_slope=THEORETICAL_SLOPE, sentinel_rtt=50.0,
            )

    def test_below_leftmost_vertex_uses_line_through_origin(self):
        # Leftmost vertex (10, 100); rtt=5 lies on the line from (0, 0) through
        # the leftmost vertex → (100/10)·5 = 50.
        d = hull_outer_distance(
            5.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=1000.0
        )
        self.assertAlmostEqual(d, 50.0)

    def test_empty_hull_falls_back_to_baseline(self):
        d = hull_outer_distance(20.0, [], [], cutoff_rtt=0.0)
        self.assertAlmostEqual(d, 20.0 / THEORETICAL_SLOPE, places=2)


class TestHullInnerDistance(unittest.TestCase):
    def test_interpolates_between_vertices(self):
        # Two-vertex hull: (10, 100) → (50, 500). At rtt=30 → 300.
        d = hull_inner_distance(
            30.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=1000.0
        )
        self.assertAlmostEqual(d, 300.0)

    def test_holds_flat_above_cutoff(self):
        d = hull_inner_distance(
            60.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=50.0
        )
        self.assertAlmostEqual(d, 500.0)

    def test_below_leftmost_vertex_returns_zero(self):
        # No useful inner constraint at very low RTT.
        d = hull_inner_distance(
            5.0, [10.0, 50.0], [100.0, 500.0], cutoff_rtt=1000.0
        )
        self.assertEqual(d, 0.0)

    def test_empty_hull_returns_zero(self):
        d = hull_inner_distance(20.0, [], [], cutoff_rtt=0.0)
        self.assertEqual(d, 0.0)


class TestFitRttDistanceSpline(unittest.TestCase):
    def test_recovers_linear_relation(self):
        # 12 points on dist = 100·rtt; piecewise-linear fit should be near-perfect.
        rtts = np.linspace(5.0, 50.0, 12)
        distances = 100.0 * rtts
        knot_rtts, knot_dists, meta = fit_rtt_distance_spline(rtts, distances, n_knots=4)
        # Spline matches the linear data at every query.
        for r in (10.0, 25.0, 40.0):
            self.assertAlmostEqual(np.interp(r, knot_rtts, knot_dists), 100.0 * r, places=1)
        self.assertGreater(meta["r_squared"], 0.999)

    def test_raises_when_too_few_points(self):
        with self.assertRaises(SplineFitError):
            fit_rtt_distance_spline(np.array([10.0, 20.0]), np.array([100.0, 200.0]), n_knots=4)

    def test_enforces_monotonicity(self):
        # Slight dip in the middle; the post-fit pass should level it out.
        rtts = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        distances = np.array([100.0, 200.0, 150.0, 200.0, 500.0, 600.0, 700.0, 800.0])
        knot_rtts, knot_dists, _ = fit_rtt_distance_spline(rtts, distances, n_knots=4)
        for i in range(1, len(knot_dists)):
            self.assertGreaterEqual(knot_dists[i], knot_dists[i - 1])


class TestFindDeltaForCoverage(unittest.TestCase):
    def test_recovers_delta_for_parallel_band(self):
        # Centered spline dist = 50·rtt; band at ±100 km. The hardest probe to
        # cover is (rtt=10, dist=400): need 500/δ ≤ 400 → δ ≥ 1.25.
        rtts, distances = _band_samples()
        spline_rtts = np.linspace(10.0, 50.0, 5)
        spline_dists = 50.0 * spline_rtts
        delta, meta = find_delta_for_coverage(
            rtts, distances, spline_rtts, spline_dists,
            target_coverage=1.0, tolerance=0.0,
        )
        # δ must be at least 1.25 to enclose every band edge.
        self.assertGreaterEqual(delta, 1.25 - 1e-6)
        self.assertAlmostEqual(meta["actual_coverage"], 1.0)

    def test_raises_when_target_unreachable_with_invalid_spline(self):
        # Spline knots missing → SplineFitError.
        with self.assertRaises(SplineFitError):
            find_delta_for_coverage(
                np.array([10.0]), np.array([100.0]),
                spline_rtt_knots=np.array([]),
                spline_dist_knots=np.array([]),
                target_coverage=0.9,
            )


class TestOctantRTTModelFit(unittest.TestCase):
    def _fitted(self, fit_spline: bool = True) -> OctantRTTModel:
        rtts, distances = _band_samples()
        model = OctantRTTModel(anchor_ip="x")
        ok = model.fit(
            rtts, distances,
            cutoff_min_points=1,
            fit_spline=fit_spline,
            spline_n_knots=4,
            bin_size_ms=1000,
        )
        self.assertTrue(ok, msg=model.fit_message)
        return model

    def test_marks_fitted_and_stores_hull(self):
        m = self._fitted()
        self.assertTrue(m.fitted)
        self.assertGreater(len(m.hull_upper_rtts), 0)
        self.assertGreater(len(m.hull_lower_rtts), 0)

    def test_fit_returns_false_with_too_few_points(self):
        m = OctantRTTModel(anchor_ip="x")
        ok = m.fit(np.array([10.0]), np.array([100.0]))
        self.assertFalse(ok)
        self.assertFalse(m.fitted)

    def test_fit_spline_false_leaves_spline_unset(self):
        m = self._fitted(fit_spline=False)
        self.assertTrue(m.fitted)
        self.assertIsNone(m.spline_rtt_knots)

    def test_fit_spline_default_populates_knots(self):
        m = self._fitted(fit_spline=True)
        self.assertIsNotNone(m.spline_rtt_knots)
        self.assertIsNotNone(m.spline_dist_knots)

    def test_fit_drops_sub_baseline_rows(self):
        """Rows below the speed-of-internet line are dropped before fitting.

        Physical bound: rtt >= 0.01 ms/km · distance at 2/3·c. A probe at
        10 ms claiming 10000 km away implies signal speed > c — must be
        a mislabeled coordinate or measurement artifact.
        """
        rtts, distances = _band_samples()
        # Splice in one sub-baseline outlier: rtt=10 with dist=10000.
        bad_rtts = np.append(rtts, 10.0)
        bad_dists = np.append(distances, 10000.0)
        m = OctantRTTModel(anchor_ip="x")
        ok = m.fit(
            bad_rtts, bad_dists,
            cutoff_min_points=1, fit_spline=True, spline_n_knots=4, bin_size_ms=1000,
        )
        self.assertTrue(ok)
        # 10 band points kept; the (10, 10000) outlier dropped.
        self.assertEqual(m.n_measurements, 10)


class TestPredictDistanceBounds(unittest.TestCase):
    def _fitted(self) -> OctantRTTModel:
        rtts, distances = _band_samples()
        model = OctantRTTModel(anchor_ip="x")
        ok = model.fit(
            rtts, distances,
            cutoff_min_points=1, fit_spline=True, spline_n_knots=4, bin_size_ms=1000,
        )
        self.assertTrue(ok, msg=model.fit_message)
        return model

    def test_returns_hull_bounds_when_delta_is_none(self):
        m = self._fitted()
        inner, outer = m.predict_distance_bounds(20.0, delta=None)
        self.assertAlmostEqual(inner, 900.0)
        self.assertAlmostEqual(outer, 1100.0)

    def test_returns_hull_bounds_above_cutoff_even_with_delta(self):
        m = self._fitted()
        # cutoff_rtt is clamped to max(data_rtt)=50, so RTT=2000 is well above.
        inner, outer = m.predict_distance_bounds(2000.0, delta=1.2)
        self.assertGreaterEqual(outer, inner)

    def test_delta_band_clipped_by_hull(self):
        m = self._fitted()
        inner, outer = m.predict_distance_bounds(20.0, delta=1.2)
        # Spline at RTT=20 ≈ 1000; band would be [1000/1.2, 1000·1.2]≈[833, 1200];
        # clipped to hull [900, 1100].
        self.assertGreaterEqual(inner, 900.0 - 1e-6)
        self.assertLessEqual(outer, 1100.0 + 1e-6)

    def test_raises_when_unfitted(self):
        m = OctantRTTModel(anchor_ip="x")
        with self.assertRaises(OctantFitError):
            m.predict_distance_bounds(20.0)

    def test_predict_distance_raises_without_spline(self):
        # Fit without spline → predict_distance must raise SplineFitError.
        rtts, distances = _band_samples()
        m = OctantRTTModel(anchor_ip="x")
        self.assertTrue(m.fit(
            rtts, distances,
            cutoff_min_points=1, fit_spline=False, bin_size_ms=1000,
        ))
        with self.assertRaises(SplineFitError):
            m.predict_distance(20.0)

    def test_predict_distance_above_cutoff_uses_finite_sentinel_slope(self):
        # Hand-build a fitted model with cutoff_rtt=50 (so RTT=60 is above)
        # and spline value 1000 km at cutoff. Default sentinel at 10_000 ms
        # gives slope (10_000 / 0.01 - 1000) / (10_000 - 50) ≈ 100.4020 km/ms;
        # at rtt=60 (Δ=10): 1000 + 10 · 100.4020 ≈ 2004.02 km. The hull band
        # is wide enough not to clip.
        m = OctantRTTModel(
            anchor_ip="x",
            hull_upper_rtts=[10.0, 50.0],
            hull_upper_distances=[100.0, 1000.0 * 100],  # wide hull → no outer clip
            hull_lower_rtts=[10.0, 50.0],
            hull_lower_distances=[0.0, 0.0],  # zero inner → no inner clip
            cutoff_rtt=50.0,
            spline_rtt_knots=[10.0, 50.0],
            spline_dist_knots=[100.0, 1000.0],
            sentinel_rtt=10000.0,
            fitted=True,
        )
        cutoff_val = 1000.0
        sentinel_dist = 10000.0 / THEORETICAL_SLOPE
        slope = (sentinel_dist - cutoff_val) / (10000.0 - 50.0)
        expected = cutoff_val + slope * (60.0 - 50.0)
        self.assertAlmostEqual(m.predict_distance(60.0), expected, places=4)


class TestSerializationJSON(unittest.TestCase):
    def _fitted(self) -> OctantRTTModel:
        rtts, distances = _band_samples()
        m = OctantRTTModel(anchor_ip="anchor-a", anchor_lat=40.0, anchor_lon=-74.0)
        self.assertTrue(m.fit(
            rtts, distances,
            cutoff_min_points=1, fit_spline=True, spline_n_knots=4, bin_size_ms=1000,
        ))
        return m

    def test_save_writes_human_readable_json(self):
        m = self._fitted()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "octant.json"
            m.save(path)
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text())
        self.assertEqual(payload["anchor_ip"], "anchor-a")
        self.assertTrue(payload["fitted"])
        self.assertIsNotNone(payload["spline_rtt_knots"])

    def test_save_load_roundtrip(self):
        m = self._fitted()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "octant.json"
            m.save(path)
            loaded = OctantRTTModel.load(path)
        self.assertTrue(loaded.fitted)
        self.assertEqual(loaded.cutoff_rtt, m.cutoff_rtt)
        inner_a, outer_a = m.predict_distance_bounds(20.0)
        inner_b, outer_b = loaded.predict_distance_bounds(20.0)
        self.assertAlmostEqual(inner_a, inner_b)
        self.assertAlmostEqual(outer_a, outer_b)

    def test_unfitted_model_roundtrips_with_none_spline(self):
        m = OctantRTTModel(anchor_ip="x")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            m.save(path)
            loaded = OctantRTTModel.load(path)
        self.assertFalse(loaded.fitted)
        self.assertIsNone(loaded.spline_rtt_knots)

    def test_legacy_fields_are_absent(self):
        """Confirm the simplification actually removed the inert fields."""
        m = OctantRTTModel(anchor_ip="x")
        for field_name in (
            "cutoff_variant",
            "low_cutoff_rtt",
            "reliable_min_rtt",
            "reliable_max_rtt",
        ):
            self.assertFalse(hasattr(m, field_name), msg=field_name)
        # The wide RTT-array prediction helpers and refined-bounds helper
        # were dropped from the production surface.
        for method in ("predict_distance_array", "predict_bounds_array", "get_refined_bounds"):
            self.assertFalse(hasattr(m, method), msg=method)


class TestConstants(unittest.TestCase):
    def test_theoretical_slope_at_two_thirds_c(self):
        # 2·d / (2/3·c) → slope = 2 / (200 km/ms) = 0.01 ms/km.
        self.assertAlmostEqual(THEORETICAL_SLOPE, 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
