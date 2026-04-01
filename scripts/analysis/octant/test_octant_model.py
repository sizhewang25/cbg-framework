"""
Unit tests for Octant RTT-Distance Model

Tests cover:
- Convex hull extraction and bound computation
- Piecewise linear spline fitting for iterative refinement
- Delta search for coverage requirement
- OctantRTTModel class functionality
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path

# Import will be updated once module is implemented
from octant_model import (
    compute_convex_hull_bounds,
    hull_rtt_to_distance,
    fit_rtt_distance_spline,
    find_delta_for_coverage,
    OctantRTTModel,
    OctantFitError,
    SplineFitError,
    DeltaSearchError,
    DeltaSearchTimeout,
    THEORETICAL_SLOPE
)


class TestConvexHullBounds(unittest.TestCase):
    """Test convex hull extraction and bound computation."""

    def test_compute_hull_basic(self):
        """Hull extraction from simple scatter data produces upper and lower bounds."""
        # Create a simple dataset with clear upper and lower bounds
        np.random.seed(42)
        n_points = 100
        rtts = np.linspace(10, 100, n_points)
        # Distance roughly linear with RTT, plus noise
        distances = 100 * rtts + np.random.uniform(-500, 500, n_points)
        distances = np.maximum(distances, 100)  # Ensure positive

        result = compute_convex_hull_bounds(rtts, distances)

        self.assertIn('hull_upper_rtts', result)
        self.assertIn('hull_upper_distances', result)
        self.assertIn('hull_lower_rtts', result)
        self.assertIn('hull_lower_distances', result)
        self.assertIn('cutoff_rtt', result)
        # Hull should have at least 2 vertices
        self.assertGreaterEqual(len(result['hull_upper_rtts']), 2)
        self.assertGreaterEqual(len(result['hull_lower_rtts']), 2)

    def test_hull_upper_lower_separation(self):
        """Upper hull distances >= lower hull distances for same RTT."""
        # Create data where upper and lower bounds are clearly different
        rtts = np.array([10, 20, 30, 40, 50, 10, 20, 30, 40, 50])
        distances = np.array([1500, 2500, 3500, 4500, 5500,  # Upper points
                              500, 1000, 1500, 2000, 2500])   # Lower points

        result = compute_convex_hull_bounds(rtts, distances, cutoff_min_points=1)

        # For any RTT in range, upper distance should be >= lower distance
        test_rtt = 30.0
        upper_dist = hull_rtt_to_distance(
            test_rtt, result['hull_upper_rtts'], result['hull_upper_distances'],
            result['cutoff_rtt'], is_upper=True
        )
        lower_dist = hull_rtt_to_distance(
            test_rtt, result['hull_lower_rtts'], result['hull_lower_distances'],
            result['cutoff_rtt'], is_upper=False
        )
        self.assertGreaterEqual(upper_dist, lower_dist)

    def test_hull_with_cutoff(self):
        """Sparse data beyond cutoff uses conservative bounds."""
        # Create dense data at low RTT, sparse at high RTT
        rtts_dense = np.random.uniform(10, 50, 50)
        rtts_sparse = np.array([80, 90])  # Only 2 points
        rtts = np.concatenate([rtts_dense, rtts_sparse])

        distances_dense = 100 * rtts_dense + np.random.uniform(-200, 200, 50)
        distances_sparse = np.array([8000, 9000])
        distances = np.concatenate([distances_dense, distances_sparse])

        result = compute_convex_hull_bounds(rtts, distances, cutoff_min_points=5)

        # Cutoff should be detected (sparse region has < 5 points per bin)
        self.assertLess(result['cutoff_rtt'], 80)

    def test_hull_rtt_to_distance(self):
        """Interpolation along hull facets works correctly."""
        # Simple hull with known vertices
        hull_rtts = [10.0, 20.0, 30.0]
        hull_distances = [1000.0, 2000.0, 3000.0]
        cutoff_rtt = 50.0

        # Test interpolation at RTT=15 (between 10 and 20)
        dist = hull_rtt_to_distance(15.0, hull_rtts, hull_distances, cutoff_rtt)
        self.assertAlmostEqual(dist, 1500.0, delta=1.0)

        # Test at exact vertex
        dist = hull_rtt_to_distance(20.0, hull_rtts, hull_distances, cutoff_rtt)
        self.assertAlmostEqual(dist, 2000.0, delta=1.0)


class TestSplineFit(unittest.TestCase):
    """Test piecewise linear spline fitting for iterative refinement."""

    def test_spline_fit_basic(self):
        """Spline fits to data with reasonable R²."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 5000 + 50 * rtts + np.random.normal(0, 200, 100)

        knot_rtts, knot_dists, metadata = fit_rtt_distance_spline(rtts, distances, n_knots=10)

        self.assertIsNotNone(knot_rtts)
        self.assertGreaterEqual(len(knot_rtts), 2)
        self.assertIn('r_squared', metadata)
        self.assertGreater(metadata['r_squared'], 0.5)

    def test_spline_knots_monotonic(self):
        """Spline knot distances are non-decreasing with RTT."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 5000 + 50 * rtts + np.random.normal(0, 500, 100)

        knot_rtts, knot_dists, _ = fit_rtt_distance_spline(rtts, distances, n_knots=15)

        for i in range(1, len(knot_dists)):
            self.assertGreaterEqual(knot_dists[i], knot_dists[i - 1])

    def test_spline_fit_insufficient_data(self):
        """Raises SplineFitError with too few points."""
        rtts = np.array([10.0, 20.0, 30.0])
        distances = np.array([1000.0, 2000.0, 3000.0])

        with self.assertRaises(SplineFitError):
            fit_rtt_distance_spline(rtts, distances, n_knots=20)  # needs 23+ points


class TestDeltaSearch(unittest.TestCase):
    """Test delta search for coverage requirement."""

    def test_delta_search_finds_solution(self):
        """Finds delta achieving target coverage."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 5000 + 50 * rtts + np.random.normal(0, 500, 100)

        knot_rtts, knot_dists, _ = fit_rtt_distance_spline(rtts, distances, n_knots=10)

        delta, metadata = find_delta_for_coverage(
            rtts, distances, knot_rtts, knot_dists,
            target_coverage=0.80,
            tolerance=0.05,
            timeout_seconds=5.0
        )

        self.assertGreater(delta, 1.0)
        self.assertAlmostEqual(metadata['actual_coverage'], 0.80, delta=0.05)

    def test_delta_search_timeout(self):
        """Raises DeltaSearchTimeout when time exceeded."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 1000)
        distances = 5000 + 50 * rtts + np.random.normal(0, 500, 1000)
        knot_rtts, knot_dists, _ = fit_rtt_distance_spline(rtts, distances, n_knots=20)

        with self.assertRaises(DeltaSearchTimeout):
            find_delta_for_coverage(
                rtts, distances, knot_rtts, knot_dists,
                target_coverage=0.99999,
                timeout_seconds=-1.0,  # Negative = instant timeout
                max_iterations=1000000
            )

    def test_delta_search_no_solution(self):
        """Raises DeltaSearchError when exact coverage impossible within tolerance."""
        np.random.seed(123)
        rtts = np.linspace(10, 100, 50)
        distances = 5000 + 50 * rtts + np.random.normal(0, 100, 50)
        # Add extreme outliers
        rtts = np.append(rtts, [50, 50])
        distances = np.append(distances, [100, 100000])

        knot_rtts, knot_dists, _ = fit_rtt_distance_spline(rtts, distances, n_knots=10)

        with self.assertRaises((DeltaSearchError, DeltaSearchTimeout)):
            find_delta_for_coverage(
                rtts, distances, knot_rtts, knot_dists,
                target_coverage=0.95,
                tolerance=0.001,  # Very tight tolerance
                timeout_seconds=0.1,  # Short timeout
                max_iterations=5
            )


class TestOctantRTTModel(unittest.TestCase):
    """Test OctantRTTModel class."""

    def test_fit_basic(self):
        """Model fits successfully on valid data."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 100 * rtts + np.random.uniform(-300, 300, 100)

        model = OctantRTTModel(
            anchor_ip='192.168.1.1',
            anchor_lat=40.0,
            anchor_lon=-74.0
        )
        success = model.fit(rtts, distances)

        self.assertTrue(success)
        self.assertTrue(model.fitted)
        self.assertGreater(len(model.hull_upper_rtts), 0)
        self.assertGreater(len(model.hull_lower_rtts), 0)

    def test_predict_distance_bounds(self):
        """Returns (min, max) distance tuple."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 100 * rtts + np.random.uniform(-300, 300, 100)

        model = OctantRTTModel(
            anchor_ip='192.168.1.1',
            anchor_lat=40.0,
            anchor_lon=-74.0
        )
        model.fit(rtts, distances)

        # Predict bounds for RTT=50
        min_dist, max_dist = model.predict_distance_bounds(50.0)

        self.assertIsInstance(min_dist, float)
        self.assertIsInstance(max_dist, float)
        self.assertLess(min_dist, max_dist)
        # Bounds should be reasonable (around 5000 km for RTT=50, slope~100)
        self.assertGreater(min_dist, 0)
        self.assertLess(max_dist, 20000)

    def test_spline_fitted_and_monotonic(self):
        """Fitted model has monotonically non-decreasing spline knots."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 5000 + 50 * rtts + np.random.uniform(-500, 500, 100)

        model = OctantRTTModel(anchor_ip='192.168.1.1', anchor_lat=40.0, anchor_lon=-74.0)
        model.fit(rtts, distances)

        self.assertIsNotNone(model.spline_rtt_knots)
        self.assertIsNotNone(model.spline_dist_knots)
        self.assertGreaterEqual(len(model.spline_rtt_knots), 2)
        for i in range(1, len(model.spline_dist_knots)):
            self.assertGreaterEqual(model.spline_dist_knots[i], model.spline_dist_knots[i - 1])

    def test_predict_with_spline_in_reliable_region(self):
        """Delta band within reliable region yields both positive and negative constraints."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 5000 + 50 * rtts + np.random.uniform(-300, 300, 100)

        model = OctantRTTModel(anchor_ip='192.168.1.1', anchor_lat=40.0, anchor_lon=-74.0)
        model.fit(rtts, distances)

        # Query RTT in the middle of reliable region
        mid_rtt = (model.low_cutoff_rtt + model.cutoff_rtt) / 2
        min_dist, max_dist = model.predict_distance_bounds(mid_rtt, delta=1.5)
        self.assertGreater(min_dist, 0, "Negative constraint (inner radius) should be > 0 in reliable region")
        self.assertLess(min_dist, max_dist)

    def test_predict_bounds_below_low_cutoff(self):
        """Delta band below low_cutoff yields only positive constraint (inner radius = 0)."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 5000 + 50 * rtts + np.random.uniform(-300, 300, 100)

        model = OctantRTTModel(anchor_ip='192.168.1.1', anchor_lat=40.0, anchor_lon=-74.0)
        model.fit(rtts, distances)

        # Query RTT below low cutoff
        rtt_below = model.low_cutoff_rtt - 5.0
        if rtt_below > 0:
            min_dist, max_dist = model.predict_distance_bounds(rtt_below, delta=1.5)
            self.assertEqual(min_dist, 0.0, "No negative constraint below low cutoff")
            self.assertGreater(max_dist, 0.0, "Positive constraint should exist")

    def test_predict_bounds_above_cutoff(self):
        """Delta band above cutoff yields only positive constraint (inner radius = 0)."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 100)
        distances = 5000 + 50 * rtts + np.random.uniform(-300, 300, 100)

        model = OctantRTTModel(anchor_ip='192.168.1.1', anchor_lat=40.0, anchor_lon=-74.0)
        model.fit(rtts, distances)

        # Query RTT above high cutoff
        rtt_above = model.cutoff_rtt + 20.0
        min_dist, max_dist = model.predict_distance_bounds(rtt_above, delta=1.5)
        self.assertEqual(min_dist, 0.0, "No negative constraint above cutoff")
        self.assertGreater(max_dist, 0.0, "Positive constraint should exist")

    def test_serialization(self):
        """Save/load preserves model state."""
        np.random.seed(42)
        rtts = np.linspace(10, 100, 50)
        distances = 100 * rtts + np.random.uniform(-200, 200, 50)

        model = OctantRTTModel(
            anchor_ip='192.168.1.1',
            anchor_lat=40.0,
            anchor_lon=-74.0
        )
        model.fit(rtts, distances)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_model.pkl'
            model.save(filepath)

            loaded_model = OctantRTTModel.load(filepath)

            self.assertEqual(loaded_model.anchor_ip, model.anchor_ip)
            self.assertEqual(loaded_model.fitted, model.fitted)
            self.assertEqual(len(loaded_model.hull_upper_rtts), len(model.hull_upper_rtts))
            np.testing.assert_array_almost_equal(
                loaded_model.hull_upper_rtts, model.hull_upper_rtts
            )


if __name__ == '__main__':
    unittest.main()
