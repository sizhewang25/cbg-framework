"""
Unit tests for Octant RTT-Distance Model

Tests cover:
- Convex hull extraction and bound computation
- Polynomial fitting for iterative refinement
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
    fit_rtt_distance_polynomial,
    find_delta_for_coverage,
    OctantRTTModel,
    OctantFitError,
    PolynomialFitError,
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


class TestPolynomialFit(unittest.TestCase):
    """Test polynomial fitting for iterative refinement."""

    def test_polynomial_fit_basic(self):
        """Polynomial fits to data with reasonable error."""
        np.random.seed(42)
        # Create data that follows a quadratic relationship
        rtts = np.linspace(10, 100, 50)
        # distance = 50 * rtt + 0.5 * rtt^2 (quadratic)
        distances = 50 * rtts + 0.5 * rtts**2 + np.random.normal(0, 100, 50)

        coefficients, metadata = fit_rtt_distance_polynomial(rtts, distances, degree=2)

        self.assertIsNotNone(coefficients)
        self.assertEqual(len(coefficients), 3)  # degree 2 has 3 coefficients
        self.assertIn('r_squared', metadata)
        self.assertGreater(metadata['r_squared'], 0.9)  # Should fit well

    def test_polynomial_fit_insufficient_data(self):
        """Raises error with too few points."""
        rtts = np.array([10.0, 20.0])  # Only 2 points for degree 2
        distances = np.array([1000.0, 2000.0])

        with self.assertRaises(PolynomialFitError):
            fit_rtt_distance_polynomial(rtts, distances, degree=2)


class TestDeltaSearch(unittest.TestCase):
    """Test delta search for coverage requirement."""

    def test_delta_search_finds_solution(self):
        """Finds delta achieving target coverage."""
        np.random.seed(42)
        # Create data around a polynomial
        rtts = np.linspace(10, 100, 100)
        poly_true = np.array([500, 50, 0.5])  # 500 + 50*rtt + 0.5*rtt^2
        distances = np.polyval(poly_true[::-1], rtts) + np.random.normal(0, 200, 100)

        # Fit polynomial first
        coefficients, _ = fit_rtt_distance_polynomial(rtts, distances, degree=2)

        # Search for delta with 80% coverage
        delta, metadata = find_delta_for_coverage(
            rtts, distances, coefficients,
            target_coverage=0.80,
            tolerance=0.05,
            timeout_seconds=5.0
        )

        self.assertGreater(delta, 1.0)
        self.assertAlmostEqual(metadata['actual_coverage'], 0.80, delta=0.05)

    def test_delta_search_timeout(self):
        """Raises DeltaSearchTimeout when time exceeded."""
        np.random.seed(42)
        # Large dataset to make search take longer
        rtts = np.linspace(10, 100, 10000)
        distances = 100 * rtts + np.random.normal(0, 500, 10000)
        coefficients, _ = fit_rtt_distance_polynomial(rtts, distances, degree=2)

        # Test that timeout parameter is respected
        # Use negative timeout to ensure immediate timeout
        with self.assertRaises(DeltaSearchTimeout):
            find_delta_for_coverage(
                rtts, distances, coefficients,
                target_coverage=0.99999,  # Very tight target
                timeout_seconds=-1.0,  # Negative timeout = instant timeout
                max_iterations=1000000
            )

    def test_delta_search_no_solution(self):
        """Raises DeltaSearchError when exact coverage impossible within tolerance."""
        # Create data with outliers that make exact 95% coverage hard to achieve
        np.random.seed(123)
        rtts = np.linspace(10, 100, 20)
        distances = 100 * rtts + np.random.normal(0, 50, 20)
        # Add extreme outliers
        rtts = np.append(rtts, [50, 50])
        distances = np.append(distances, [100, 50000])  # One very close, one very far

        coefficients, _ = fit_rtt_distance_polynomial(rtts, distances, degree=1)

        # Require 100% coverage with 0 tolerance and limited iterations
        # This should fail because the outliers make it impossible
        with self.assertRaises((DeltaSearchError, DeltaSearchTimeout)):
            find_delta_for_coverage(
                rtts, distances, coefficients,
                target_coverage=0.95,
                tolerance=0.001,  # Very tight tolerance
                timeout_seconds=0.1,  # Short timeout
                max_iterations=5  # Very few iterations
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
