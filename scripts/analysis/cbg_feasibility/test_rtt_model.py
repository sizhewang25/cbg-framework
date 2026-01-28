"""
Unit tests for RTT-Distance Modeling Module

Tests cover:
- Haversine distance calculation
- Bestline fitting with synthetic data
- Edge cases (few points, outliers, invalid data)
- RTTDistanceModel class functionality
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path

from rtt_model import (
    haversine_distance,
    fit_bestline,
    fit_bestline_lp,
    rtt_to_distance,
    rtt_to_distance_fixed,
    RTTDistanceModel,
    EARTH_RADIUS_KM,
    THEORETICAL_SLOPE,
    SCIPY_AVAILABLE
)


class TestHaversineDistance(unittest.TestCase):
    """Test haversine distance calculations."""

    def test_same_point(self):
        """Distance between same point should be 0."""
        dist = haversine_distance(40.0, -74.0, 40.0, -74.0)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_known_distance_nyc_la(self):
        """Test NYC to LA distance (~3940 km)."""
        # NYC: 40.7128° N, 74.0060° W
        # LA: 34.0522° N, 118.2437° W
        dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        # Actual distance is approximately 3940 km
        self.assertAlmostEqual(dist, 3940, delta=50)

    def test_known_distance_seattle_miami(self):
        """Test Seattle to Miami distance (~4400 km)."""
        # Seattle: 47.6062° N, 122.3321° W
        # Miami: 25.7617° N, 80.1918° W
        dist = haversine_distance(47.6062, -122.3321, 25.7617, -80.1918)
        # Actual distance is approximately 4400 km
        self.assertAlmostEqual(dist, 4400, delta=100)

    def test_equator_distance(self):
        """Test distance along equator (1 degree ≈ 111 km)."""
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        # At equator, 1 degree of longitude ≈ 111.32 km
        self.assertAlmostEqual(dist, 111.32, delta=1)

    def test_symmetry(self):
        """Distance should be symmetric."""
        dist1 = haversine_distance(40.0, -74.0, 34.0, -118.0)
        dist2 = haversine_distance(34.0, -118.0, 40.0, -74.0)
        self.assertAlmostEqual(dist1, dist2, places=5)

    def test_vectorized(self):
        """Test that function works with numpy arrays implicitly."""
        # Single point test with floats
        dist = haversine_distance(0.0, 0.0, 0.0, 90.0)
        # Quarter of Earth's circumference
        expected = EARTH_RADIUS_KM * np.pi / 2
        self.assertAlmostEqual(dist, expected, delta=1)


class TestFitBestline(unittest.TestCase):
    """Test bestline fitting functionality."""

    def test_perfect_linear_data(self):
        """Test with perfectly linear data."""
        # Create perfect linear data: RTT = 0.01 * distance + 5
        # Use bin centers to get exact fit
        distances = np.array([150, 250, 350, 450, 550, 650, 750, 850])  # Bin centers for 100km bins
        rtts = 0.01 * distances + 5.0

        result = fit_bestline(distances, rtts, bin_size_km=100)

        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['slope'], 0.01, places=4)
        self.assertAlmostEqual(result['intercept'], 5.0, delta=0.6)  # Allow small offset due to binning
        self.assertAlmostEqual(result['r_squared'], 1.0, places=4)

    def test_noisy_data_lower_envelope(self):
        """Test that bestline captures lower envelope of noisy data."""
        np.random.seed(42)
        n_points = 200

        # Generate data with noise ABOVE the baseline
        distances = np.random.uniform(100, 1000, n_points)
        # True relationship: RTT = 0.01 * distance + 5
        # Add positive noise (RTT can only be inflated, not below physical limit)
        baseline_rtts = 0.01 * distances + 5.0
        noise = np.abs(np.random.normal(0, 10, n_points))  # Only positive noise
        rtts = baseline_rtts + noise

        result = fit_bestline(distances, rtts, bin_size_km=100, percentile=0.05)

        self.assertTrue(result['success'])
        # Slope should be close to 0.01 (within reasonable tolerance)
        self.assertAlmostEqual(result['slope'], 0.01, delta=0.005)
        # Intercept should be close to 5 (baseline)
        self.assertAlmostEqual(result['intercept'], 5.0, delta=3.0)

    def test_insufficient_bins(self):
        """Test failure when fewer than 3 bins."""
        # Only data in 2 bins
        distances = np.array([50, 60, 70, 150, 160])
        rtts = np.array([10, 11, 10.5, 15, 16])

        result = fit_bestline(distances, rtts, bin_size_km=100)

        self.assertFalse(result['success'])
        self.assertEqual(result['n_bins'], 2)
        self.assertIn('need at least 3', result['message'].lower())

    def test_empty_input(self):
        """Test with empty arrays."""
        result = fit_bestline(np.array([]), np.array([]))

        self.assertFalse(result['success'])
        self.assertEqual(result['n_bins'], 0)

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        result = fit_bestline(np.array([1, 2, 3]), np.array([1, 2]))

        self.assertFalse(result['success'])
        self.assertIn('same length', result['message'].lower())

    def test_negative_values_filtered(self):
        """Test that negative/zero values are filtered out."""
        distances = np.array([-100, 0, 100, 200, 300, 400, 500])
        rtts = np.array([5, 6, 10, 15, 20, 25, 30])

        result = fit_bestline(distances, rtts, bin_size_km=100)

        # Should still succeed with valid data
        self.assertTrue(result['success'])
        self.assertGreaterEqual(result['n_bins'], 3)

    def test_custom_bin_size(self):
        """Test with different bin sizes."""
        distances = np.array([50, 100, 150, 200, 250, 300, 350, 400])
        rtts = 0.01 * distances + 5.0

        # With 50km bins, should get more bins
        result_50 = fit_bestline(distances, rtts, bin_size_km=50)
        # With 200km bins, should get fewer bins
        result_200 = fit_bestline(distances, rtts, bin_size_km=200)

        self.assertGreater(result_50['n_bins'], result_200['n_bins'])

    def test_custom_percentile(self):
        """Test with different percentiles."""
        np.random.seed(42)
        distances = np.repeat(np.arange(100, 600, 100), 20)  # 20 points per bin
        baseline = 0.01 * distances + 5.0
        rtts = baseline + np.abs(np.random.normal(0, 5, len(distances)))

        result_5 = fit_bestline(distances, rtts, bin_size_km=100, percentile=0.05)
        result_50 = fit_bestline(distances, rtts, bin_size_km=100, percentile=0.50)

        # 5th percentile should give lower intercept than 50th
        self.assertLess(result_5['intercept'], result_50['intercept'])


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy not available")
class TestFitBestlineLP(unittest.TestCase):
    """Test LP-based bestline fitting (original CBG paper method)."""

    def test_perfect_linear_data(self):
        """Test LP with data exactly on a line."""
        # Data on line: RTT = 0.01 * distance + 5
        distances = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        rtts = 0.01 * distances + 5.0

        result = fit_bestline_lp(distances, rtts)

        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['slope'], 0.01, places=4)
        self.assertAlmostEqual(result['intercept'], 5.0, places=2)
        self.assertEqual(result['violations'], 0)

    def test_line_below_all_points(self):
        """Test that LP bestline lies below all data points."""
        np.random.seed(42)
        distances = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        # RTTs with noise ABOVE baseline
        baseline = 0.012 * distances + 8.0
        rtts = baseline + np.random.uniform(0, 20, len(distances))

        result = fit_bestline_lp(distances, rtts)

        self.assertTrue(result['success'])
        # Verify all points are on or above the line
        predicted = result['slope'] * distances + result['intercept']
        self.assertTrue(np.all(rtts >= predicted - 0.001))
        self.assertEqual(result['violations'], 0)

    def test_slope_above_baseline(self):
        """Test that LP slope is at least the baseline (2/3 speed of light)."""
        # Data where true relationship is at baseline (slope = 0.01)
        distances = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        # RTTs on the baseline with small positive offset
        rtts = THEORETICAL_SLOPE * distances + 2.0  # intercept = 2ms

        result = fit_bestline_lp(distances, rtts)

        self.assertTrue(result['success'])
        # Slope should be at least THEORETICAL_SLOPE
        self.assertGreaterEqual(result['slope'], THEORETICAL_SLOPE - 0.0001)

    def test_infeasible_faster_than_light_no_filter(self):
        """Test that LP fails when data suggests faster-than-light transmission (without filtering)."""
        # Data that would require slope below physical limit
        distances = np.array([100, 200, 300, 400, 500])
        # These RTTs suggest slope ~0.005 ms/km (faster than 2/3 c)
        rtts = np.array([1.5, 2.0, 2.5, 3.0, 3.5])

        # With filtering disabled, LP should fail - no valid bestline exists
        result = fit_bestline_lp(distances, rtts, filter_violations=False)

        self.assertFalse(result['success'])
        self.assertIn('infeasible', result['message'].lower())

    def test_filter_faster_than_light_violations(self):
        """Test that violations are filtered when filter_violations=True."""
        # Data that would require slope below physical limit
        # Min RTT for d=100 is 1.0ms (0.01*100), for d=200 is 2.0ms, etc.
        distances = np.array([100, 200, 300, 400, 500])
        # RTTs = [1.5, 2.0, 2.5, 3.0, 3.5]
        # Valid:   1.5>=1.0 (yes), 2.0>=2.0 (yes), 2.5>=3.0 (no), 3.0>=4.0 (no), 3.5>=5.0 (no)
        # So 3 points are violations, leaving only 2 valid points
        rtts = np.array([1.5, 2.0, 2.5, 3.0, 3.5])

        # With filtering enabled (default), 3 violations are filtered
        result = fit_bestline_lp(distances, rtts, filter_violations=True, filter_outliers=False)

        # Should fail because only 2 valid points remain (need at least 3)
        self.assertFalse(result['success'])
        self.assertEqual(result['n_filtered'], 3)
        self.assertIn('baseline: 3', result['message'].lower())
        # Check filter_stats for detailed breakdown
        self.assertEqual(result['filter_stats']['removed_below_baseline'], 3)

    def test_filter_partial_violations(self):
        """Test filtering of some speed-of-light violations while keeping valid data."""
        # Mix of valid and invalid data
        distances = np.array([100, 200, 300, 400, 500, 600, 700, 800])
        rtts = np.array([
            0.5,   # Invalid: too fast for 100km (needs ~1.0 ms min)
            2.5,   # Valid for 200km
            4.0,   # Valid for 300km
            5.0,   # Valid for 400km
            6.5,   # Valid for 500km
            8.0,   # Valid for 600km
            10.0,  # Valid for 700km
            12.0,  # Valid for 800km
        ])

        result = fit_bestline_lp(distances, rtts, filter_violations=True, filter_outliers=False)

        self.assertTrue(result['success'])
        self.assertEqual(result['n_filtered'], 1)  # Only the first point filtered
        self.assertIn('filtered', result['message'].lower())

    def test_filter_outliers(self):
        """Test that outliers beyond n_std are filtered."""
        np.random.seed(42)
        # Generate data with clear outliers
        distances = np.concatenate([
            np.full(20, 100),   # Bin 1: 20 points at 100km
            np.full(20, 200),   # Bin 2: 20 points at 200km
            np.full(20, 300),   # Bin 3: 20 points at 300km
        ])
        # RTTs with some extreme outliers
        rtts = np.concatenate([
            np.array([2.0] * 18 + [100.0, 150.0]),  # Two extreme outliers at 100km bin
            np.array([4.0] * 18 + [200.0, 250.0]),  # Two extreme outliers at 200km bin
            np.array([6.0] * 20),                    # No outliers at 300km bin
        ])

        # With outlier filtering enabled
        result = fit_bestline_lp(distances, rtts, filter_violations=False, filter_outliers=True, n_std=2.0)

        self.assertTrue(result['success'])
        # The extreme outliers should be filtered
        self.assertGreater(result['filter_stats']['removed_outliers'], 0)
        # Slope should be reasonable (not pulled up by outliers)
        self.assertLess(result['slope'], 0.05)  # Would be much higher without filtering

    def test_non_negative_intercept(self):
        """Test that LP intercept is non-negative."""
        distances = np.array([100, 200, 300, 400, 500, 600])
        # RTTs that might suggest negative intercept
        rtts = 0.015 * distances  # No intercept in true relationship

        result = fit_bestline_lp(distances, rtts)

        self.assertTrue(result['success'])
        # Intercept should be >= 0 per CBG paper
        self.assertGreaterEqual(result['intercept'], -0.001)

    def test_insufficient_points(self):
        """Test LP failure with fewer than 3 points."""
        distances = np.array([100, 200])
        rtts = np.array([5, 10])

        result = fit_bestline_lp(distances, rtts)

        self.assertFalse(result['success'])
        self.assertIn('at least 3', result['message'].lower())

    def test_empty_input(self):
        """Test LP with empty arrays."""
        result = fit_bestline_lp(np.array([]), np.array([]))

        self.assertFalse(result['success'])

    def test_realistic_network_data(self):
        """Test LP with realistic network measurement data."""
        # Simulate real measurements: some near physical limit, some delayed
        distances = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        rtts = np.array([
            2.5,   # near optimal for 100km
            4.0,   # near optimal for 200km
            8.0,   # delayed
            6.0,   # near optimal for 400km
            12.0,  # delayed
            9.0,   # near optimal for 600km
            15.0,  # delayed
            11.0,  # near optimal for 800km
            18.0,  # delayed
            14.0,  # near optimal for 1000km
        ])

        result = fit_bestline_lp(distances, rtts)

        self.assertTrue(result['success'])
        # All points should be on or above the line
        predicted = result['slope'] * distances + result['intercept']
        self.assertTrue(np.all(rtts >= predicted - 0.001))
        # Slope should be reasonable (between 0.01 and 0.03 ms/km)
        self.assertGreater(result['slope'], 0.005)
        self.assertLess(result['slope'], 0.05)


class TestRttToDistance(unittest.TestCase):
    """Test RTT to distance conversion functions."""

    def test_calibrated_conversion(self):
        """Test calibrated RTT to distance conversion."""
        slope = 0.01  # ms/km
        intercept = 5.0  # ms

        # RTT = 15ms should give distance = (15 - 5) / 0.01 = 1000 km
        dist = rtt_to_distance(15.0, slope, intercept)
        self.assertAlmostEqual(dist, 1000.0, places=1)

    def test_calibrated_below_intercept(self):
        """RTT below intercept should return 0."""
        dist = rtt_to_distance(3.0, slope=0.01, intercept=5.0)
        self.assertEqual(dist, 0.0)

    def test_calibrated_zero_slope(self):
        """Zero slope should return 0."""
        dist = rtt_to_distance(10.0, slope=0.0, intercept=5.0)
        self.assertEqual(dist, 0.0)

    def test_fixed_conversion(self):
        """Test fixed speed threshold conversion."""
        # RTT = 10ms, speed = 2/3 c = 200 km/ms
        # One-way time = 5ms, distance = 200 * 5 = 1000 km
        dist = rtt_to_distance_fixed(10.0, speed_fraction=2/3)
        self.assertAlmostEqual(dist, 1000.0, places=1)

    def test_fixed_vs_theoretical(self):
        """Compare fixed method with theoretical slope."""
        # At theoretical slope (2/3 c), conversions should roughly match
        rtt = 20.0
        dist_fixed = rtt_to_distance_fixed(rtt, speed_fraction=2/3)
        # Calibrated with theoretical slope and 0 intercept
        dist_calibrated = rtt_to_distance(rtt, slope=THEORETICAL_SLOPE, intercept=0.0)

        # Should be within 1% of each other
        self.assertAlmostEqual(dist_fixed, dist_calibrated, delta=dist_fixed * 0.01)


class TestRTTDistanceModel(unittest.TestCase):
    """Test RTTDistanceModel class."""

    def setUp(self):
        """Set up test model."""
        self.model = RTTDistanceModel(
            anchor_ip='192.168.1.1',
            anchor_lat=47.6062,
            anchor_lon=-122.3321
        )

    def test_model_initialization(self):
        """Test model is initialized correctly."""
        self.assertEqual(self.model.anchor_ip, '192.168.1.1')
        self.assertFalse(self.model.fitted)
        self.assertIsNone(self.model.slope)

    def test_model_fit_success(self):
        """Test successful model fitting."""
        # Use bin centers for cleaner fit
        distances = np.array([150, 250, 350, 450, 550, 650])
        rtts = 0.01 * distances + 5.0

        success = self.model.fit(distances, rtts, bin_size_km=100)

        self.assertTrue(success)
        self.assertTrue(self.model.fitted)
        self.assertAlmostEqual(self.model.slope, 0.01, places=4)
        self.assertAlmostEqual(self.model.intercept, 5.0, delta=0.6)

    def test_model_fit_failure(self):
        """Test model fitting failure."""
        distances = np.array([100, 110])  # Only 1 bin
        rtts = np.array([10, 11])

        success = self.model.fit(distances, rtts, bin_size_km=100)

        self.assertFalse(success)
        self.assertFalse(self.model.fitted)

    def test_model_predict_distance(self):
        """Test distance prediction."""
        self.model.slope = 0.01
        self.model.intercept = 5.0
        self.model.fitted = True

        dist = self.model.predict_distance(15.0)
        self.assertAlmostEqual(dist, 1000.0, places=1)

    def test_model_predict_unfitted(self):
        """Unfitted model should return None."""
        dist = self.model.predict_distance(15.0)
        self.assertIsNone(dist)

    def test_model_predict_rtt(self):
        """Test RTT prediction."""
        self.model.slope = 0.01
        self.model.intercept = 5.0
        self.model.fitted = True

        rtt = self.model.predict_rtt(1000.0)
        self.assertAlmostEqual(rtt, 15.0, places=1)

    def test_model_save_load(self):
        """Test model serialization."""
        # Fit the model first
        distances = np.array([100, 200, 300, 400, 500, 600])
        rtts = 0.01 * distances + 5.0
        self.model.fit(distances, rtts, bin_size_km=100)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_model.pkl'
            self.model.save(filepath)

            loaded_model = RTTDistanceModel.load(filepath)

            self.assertEqual(loaded_model.anchor_ip, self.model.anchor_ip)
            self.assertAlmostEqual(loaded_model.slope, self.model.slope, places=6)
            self.assertAlmostEqual(loaded_model.intercept, self.model.intercept, places=6)
            self.assertTrue(loaded_model.fitted)

    def test_model_to_dict(self):
        """Test dictionary conversion."""
        self.model.slope = 0.01
        self.model.intercept = 5.0
        self.model.fitted = True
        self.model.n_bins = 5

        d = self.model.to_dict()

        self.assertEqual(d['anchor_ip'], '192.168.1.1')
        self.assertEqual(d['slope'], 0.01)
        self.assertEqual(d['intercept'], 5.0)
        self.assertTrue(d['fitted'])

    def test_model_repr(self):
        """Test string representation."""
        self.model.slope = 0.01
        self.model.intercept = 5.0
        self.model.r_squared = 0.95
        self.model.n_bins = 5
        self.model.fitted = True

        repr_str = repr(self.model)

        self.assertIn('192.168.1.1', repr_str)
        self.assertIn('slope=', repr_str)


class TestIntegration(unittest.TestCase):
    """Integration tests with realistic data."""

    def test_realistic_scenario(self):
        """Test with realistic US network data."""
        np.random.seed(42)

        # Simulate probes at various distances from Seattle anchor
        n_probes = 100
        distances = np.random.uniform(100, 4000, n_probes)  # 100-4000 km

        # Realistic RTT model: ~0.01 ms/km baseline + processing delays
        # Plus some routing inefficiency (10-50% inflation)
        baseline_rtts = 0.01 * distances + 3.0  # 3ms processing baseline
        inflation = 1.0 + np.random.uniform(0.1, 0.5, n_probes)
        rtts = baseline_rtts * inflation

        # Create and fit model
        model = RTTDistanceModel(
            anchor_ip='45.77.211.82',
            anchor_lat=47.6095,
            anchor_lon=-122.3415
        )

        success = model.fit(distances, rtts, bin_size_km=200)

        self.assertTrue(success)
        # Slope should be reasonably close to 0.01
        self.assertAlmostEqual(model.slope, 0.01, delta=0.005)
        # R-squared should be decent
        self.assertGreater(model.r_squared, 0.8)

        # Test prediction
        test_rtt = 30.0  # ms
        predicted_dist = model.predict_distance(test_rtt)
        # Should be in reasonable range
        self.assertGreater(predicted_dist, 0)
        self.assertLess(predicted_dist, 5000)


if __name__ == '__main__':
    unittest.main()
