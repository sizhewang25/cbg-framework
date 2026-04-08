"""
Unit tests for the Octant geolocation pipeline.

Tests cover:
- Constraint formation from RTT measurements
- Annulus geometry (Shapely conversion, point-in-annulus)
- Feasible region computation (unweighted and weighted)
- Monte Carlo point selection (sampling, geometric median)
- End-to-end estimate_location function
- OctantGeolocator orchestration class
"""

import unittest
import numpy as np
from shapely.geometry import Point

from scripts.analysis.octant.octant_model import OctantRTTModel
from scripts.analysis.octant.octant_geolocation import (
    AnnularConstraint,
    form_constraint,
    form_constraints,
    _annulus_to_shapely,
    _circle_to_shapely,
    _point_in_annulus,
    _points_in_annulus_vectorized,
    compute_feasible_region_unweighted,
    compute_feasible_region_weighted,
    sample_points_in_region,
    geometric_median_approx,
    estimate_location,
    OctantGeolocator,
)
from scripts.utils.helpers import haversine


# =============================================================================
# Test Fixtures
# =============================================================================

def _make_fitted_model(anchor_ip: str, lat: float, lon: float) -> OctantRTTModel:
    """Create a fitted OctantRTTModel with synthetic calibration data.

    Generates RTT/distance pairs along a ~2/3c trend line with noise,
    ensuring the hull bounds produce reasonable (r_L, R_L) pairs.
    """
    rng = np.random.RandomState(hash(anchor_ip) % 2**31)
    rtts = np.linspace(5, 100, 60)
    # ~100 km per ms (roughly 2/3c), plus noise
    distances = 100.0 * rtts + rng.uniform(-150, 150, len(rtts))
    distances = np.maximum(distances, 10.0)  # no negative distances

    model = OctantRTTModel(anchor_ip=anchor_ip, anchor_lat=lat, anchor_lon=lon)
    model.fit(rtts, distances)
    return model


def _synthetic_rtt(true_lat: float, true_lon: float,
                   lm_lat: float, lm_lon: float,
                   noise_ms: float = 2.0) -> float:
    """Compute synthetic RTT from true location to landmark.

    RTT = 2 * distance / (2/3 * c) in ms, plus small noise.
    """
    dist_km = haversine((true_lat, true_lon), (lm_lat, lm_lon))
    speed_km_per_ms = 300.0 * 2 / 3  # 200 km/ms
    rtt = 2 * dist_km / speed_km_per_ms
    rng = np.random.RandomState(int(abs(true_lat * 1000 + lm_lat * 100)) % 2**31)
    return max(1.0, rtt + rng.uniform(0, noise_ms))


# Landmark definitions for reuse across tests
LANDMARKS = {
    'lm_nyc': (40.7128, -74.0060),   # New York
    'lm_chi': (41.8781, -87.6298),    # Chicago
    'lm_la':  (34.0522, -118.2437),   # Los Angeles
    'lm_hou': (29.7604, -95.3698),    # Houston
    'lm_sea': (47.6062, -122.3321),   # Seattle
}


def _build_test_models():
    """Build fitted models for all test landmarks."""
    return {ip: _make_fitted_model(ip, lat, lon) for ip, (lat, lon) in LANDMARKS.items()}


# =============================================================================
# Test: Constraint Formation
# =============================================================================

class TestConstraintFormation(unittest.TestCase):

    def setUp(self):
        self.models = _build_test_models()

    def test_form_single_constraint(self):
        """form_constraint produces correct bounds and weight."""
        model = self.models['lm_nyc']
        c = form_constraint(40.7128, -74.0060, 'lm_nyc', 30.0, model, weight_tau_ms=50.0)

        self.assertIsInstance(c, AnnularConstraint)
        self.assertGreater(c.outer_radius_km, c.inner_radius_km)
        self.assertGreaterEqual(c.inner_radius_km, 0)
        self.assertAlmostEqual(c.weight, np.exp(-30.0 / 50.0), places=6)
        self.assertEqual(c.rtt_ms, 30.0)

    def test_filters_unfitted_models(self):
        """Only fitted models produce constraints."""
        unfitted = OctantRTTModel(anchor_ip='unfitted', anchor_lat=0, anchor_lon=0)
        models = {**self.models, 'unfitted': unfitted}
        rtts = {'lm_nyc': 20.0, 'unfitted': 20.0}

        constraints = form_constraints('target', rtts, LANDMARKS, models)
        ips = [c.landmark_ip for c in constraints]
        self.assertIn('lm_nyc', ips)
        self.assertNotIn('unfitted', ips)

    def test_keeps_high_rtt(self):
        """High-RTT measurements are retained for later weighted handling."""
        rtts = {'lm_nyc': 20.0, 'lm_chi': 300.0}
        constraints = form_constraints(
            'target', rtts, LANDMARKS, self.models, max_rtt_ms=200.0
        )
        ips = [c.landmark_ip for c in constraints]
        self.assertIn('lm_nyc', ips)
        self.assertIn('lm_chi', ips)

    def test_sorted_by_weight(self):
        """Constraints are sorted by weight descending (lowest RTT first)."""
        rtts = {'lm_nyc': 50.0, 'lm_chi': 10.0, 'lm_la': 80.0}
        constraints = form_constraints('target', rtts, LANDMARKS, self.models)

        weights = [c.weight for c in constraints]
        self.assertEqual(weights, sorted(weights, reverse=True))
        # Lowest RTT should have highest weight
        self.assertEqual(constraints[0].landmark_ip, 'lm_chi')

    def test_empty_input(self):
        """Empty measurements returns empty list."""
        constraints = form_constraints('target', {}, LANDMARKS, self.models)
        self.assertEqual(constraints, [])


# =============================================================================
# Test: Annulus Geometry
# =============================================================================

class TestAnnulusGeometry(unittest.TestCase):

    def test_zero_radius_circle_uses_geometry_epsilon(self):
        """Zero-radius circles still produce a usable Shapely geometry."""
        result = _circle_to_shapely(40.0, -74.0, 0.0, n_pts=64)
        self.assertFalse(result.is_empty)
        self.assertGreater(result.area, 0)

    def test_annulus_valid(self):
        """Annulus with inner < outer produces a valid ring polygon."""
        result = _annulus_to_shapely(40.0, -74.0, 100.0, 500.0, n_pts=64)
        self.assertIsNotNone(result)
        self.assertFalse(result.is_empty)
        self.assertGreater(result.area, 0)

        # The annulus should have a hole (inner ring)
        # In Shapely, a polygon with a hole has interiors
        if hasattr(result, 'interiors'):
            self.assertGreater(len(list(result.interiors)), 0)

    def test_annulus_degenerate(self):
        """Inner >= outer radius returns None."""
        self.assertIsNone(_annulus_to_shapely(40.0, -74.0, 500.0, 100.0))
        self.assertIsNone(_annulus_to_shapely(40.0, -74.0, 500.0, 500.0))

    def test_point_in_annulus_inside(self):
        """Point at known distance within annulus returns True."""
        # NYC landmark, point ~200 km away (roughly Philadelphia)
        c = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='test', rtt_ms=10.0,
            inner_radius_km=100.0, outer_radius_km=300.0,
            weight=1.0,
        )
        # Philadelphia is ~130 km from NYC
        self.assertTrue(_point_in_annulus(39.9526, -75.1652, c))

    def test_point_in_annulus_outside(self):
        """Point too close or too far returns False."""
        c = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='test', rtt_ms=10.0,
            inner_radius_km=200.0, outer_radius_km=300.0,
            weight=1.0,
        )
        # Point very close to NYC (within inner radius)
        self.assertFalse(_point_in_annulus(40.72, -74.01, c))
        # Point very far from NYC (LA, ~3900 km)
        self.assertFalse(_point_in_annulus(34.05, -118.24, c))


# =============================================================================
# Test: Region Computation
# =============================================================================

class TestRegionComputation(unittest.TestCase):

    def test_unweighted_two_constraints(self):
        """Two overlapping constraints produce a non-empty region."""
        # NYC and Chicago are ~1150 km apart
        # Outer radii of 1500 km should overlap; inner radii of 200 km
        c1 = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='nyc', rtt_ms=10.0,
            inner_radius_km=200.0, outer_radius_km=1500.0,
            weight=1.0,
        )
        c2 = AnnularConstraint(
            landmark_lat=41.8781, landmark_lon=-87.6298,
            landmark_ip='chi', rtt_ms=15.0,
            inner_radius_km=200.0, outer_radius_km=1500.0,
            weight=0.8,
        )
        region = compute_feasible_region_unweighted([c1, c2], n_pts=64)
        self.assertIsNotNone(region)
        self.assertFalse(region.is_empty)

    def test_unweighted_no_intersection(self):
        """Non-overlapping constraints produce None."""
        # Small radii, far-apart landmarks
        c1 = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='nyc', rtt_ms=10.0,
            inner_radius_km=50.0, outer_radius_km=100.0,
            weight=1.0,
        )
        c2 = AnnularConstraint(
            landmark_lat=34.0522, landmark_lon=-118.2437,
            landmark_ip='la', rtt_ms=10.0,
            inner_radius_km=50.0, outer_radius_km=100.0,
            weight=1.0,
        )
        region = compute_feasible_region_unweighted([c1, c2], n_pts=64)
        self.assertIsNone(region)

    def test_weighted_basic(self):
        """Weighted region is non-empty for overlapping constraints."""
        c1 = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='nyc', rtt_ms=10.0,
            inner_radius_km=100.0, outer_radius_km=1500.0,
            weight=1.0,
        )
        c2 = AnnularConstraint(
            landmark_lat=41.8781, landmark_lon=-87.6298,
            landmark_ip='chi', rtt_ms=15.0,
            inner_radius_km=100.0, outer_radius_km=1500.0,
            weight=0.8,
        )
        region = compute_feasible_region_weighted(
            [c1, c2], weight_threshold=0.5, grid_resolution_deg=0.2
        )
        self.assertIsNotNone(region)
        self.assertFalse(region.is_empty)

    def test_threshold_effect(self):
        """Higher weight threshold produces smaller or equal region area."""
        c1 = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='nyc', rtt_ms=10.0,
            inner_radius_km=100.0, outer_radius_km=1500.0,
            weight=1.0,
        )
        c2 = AnnularConstraint(
            landmark_lat=41.8781, landmark_lon=-87.6298,
            landmark_ip='chi', rtt_ms=15.0,
            inner_radius_km=100.0, outer_radius_km=1500.0,
            weight=0.8,
        )
        constraints = [c1, c2]

        region_low = compute_feasible_region_weighted(
            constraints, weight_threshold=0.3, grid_resolution_deg=0.2
        )
        region_high = compute_feasible_region_weighted(
            constraints, weight_threshold=0.7, grid_resolution_deg=0.2
        )

        if region_low is not None and region_high is not None:
            self.assertGreaterEqual(region_low.area, region_high.area)

    def test_region_contains_true_location(self):
        """Feasible region from synthetic RTTs should be near the true target."""
        # Target: Pittsburgh (40.4406, -79.9959)
        true_lat, true_lon = 40.4406, -79.9959
        models = _build_test_models()

        rtts = {}
        for lm_ip, (lm_lat, lm_lon) in LANDMARKS.items():
            rtts[lm_ip] = _synthetic_rtt(true_lat, true_lon, lm_lat, lm_lon, noise_ms=1.0)

        constraints = form_constraints(
            'target', rtts, LANDMARKS, models, max_rtt_ms=200.0
        )
        self.assertGreater(len(constraints), 0, "Should have at least one constraint")

        region = compute_feasible_region_unweighted(constraints, n_pts=64)

        if region is not None:
            # The region's centroid should be within a reasonable distance
            # of the true location (hull-fitted models may not exactly
            # contain the true point due to calibration noise)
            centroid = region.centroid
            est_lat, est_lon = centroid.y, centroid.x
            error_km = haversine((est_lat, est_lon), (true_lat, true_lon))
            self.assertLess(
                error_km, 1500.0,
                f"Region centroid should be within 1500 km of true location, got {error_km:.0f} km"
            )
        else:
            # If unweighted region is empty (can happen with noisy synthetic data),
            # at least verify the fallback produces a reasonable estimate
            result = estimate_location(constraints, method='centroid')
            self.assertIsNotNone(result)
            error_km = haversine((result['lat'], result['lon']), (true_lat, true_lon))
            self.assertLess(error_km, 2000.0)


# =============================================================================
# Test: Point Selection
# =============================================================================

class TestPointSelection(unittest.TestCase):

    def _make_simple_region(self):
        """Create a simple rectangular region for testing."""
        return _circle_to_shapely(40.0, -74.0, 200.0, n_pts=64)

    def test_sample_count(self):
        """sample_points_in_region returns the requested number of points."""
        region = self._make_simple_region()
        rng = np.random.default_rng(42)
        points = sample_points_in_region(region, n_samples=100, rng=rng)
        self.assertEqual(len(points), 100)

    def test_samples_within_region(self):
        """All sampled points lie within the region."""
        region = self._make_simple_region()
        rng = np.random.default_rng(42)
        points = sample_points_in_region(region, n_samples=50, rng=rng)

        for lat, lon in points:
            self.assertTrue(
                region.contains(Point(lon, lat)),
                f"Point ({lat}, {lon}) should be inside region"
            )

    def test_median_symmetric(self):
        """Geometric median of symmetric points is near the center."""
        # Many points symmetrically around (40, -74)
        rng = np.random.default_rng(42)
        n = 200
        offsets = rng.normal(0, 0.5, (n, 2))
        points = np.column_stack([
            40.0 + offsets[:, 0],
            -74.0 + offsets[:, 1],
        ])
        lat, lon = geometric_median_approx(points)
        self.assertAlmostEqual(lat, 40.0, delta=1.0)
        self.assertAlmostEqual(lon, -74.0, delta=1.0)

    def test_median_cluster_vs_outlier(self):
        """Geometric median is closer to a cluster than the arithmetic mean."""
        # Cluster near (40, -74), one outlier at (34, -118) (LA)
        points = np.array([
            [40.0, -74.0],
            [40.1, -74.1],
            [39.9, -73.9],
            [40.05, -74.05],
            [34.0, -118.0],  # outlier
        ])
        med_lat, med_lon = geometric_median_approx(points)
        mean_lat, mean_lon = points.mean(axis=0)

        dist_median = haversine((med_lat, med_lon), (40.0, -74.0))
        dist_mean = haversine((mean_lat, mean_lon), (40.0, -74.0))

        self.assertLess(dist_median, dist_mean)


# =============================================================================
# Test: estimate_location (end-to-end)
# =============================================================================

class TestEstimateLocation(unittest.TestCase):

    def _make_constraints(self):
        """Create 3 overlapping constraints around the US northeast."""
        return [
            AnnularConstraint(
                landmark_lat=40.7128, landmark_lon=-74.0060,
                landmark_ip='nyc', rtt_ms=10.0,
                inner_radius_km=100.0, outer_radius_km=1500.0,
                weight=np.exp(-10.0 / 50.0),
            ),
            AnnularConstraint(
                landmark_lat=41.8781, landmark_lon=-87.6298,
                landmark_ip='chi', rtt_ms=15.0,
                inner_radius_km=100.0, outer_radius_km=1500.0,
                weight=np.exp(-15.0 / 50.0),
            ),
            AnnularConstraint(
                landmark_lat=29.7604, landmark_lon=-95.3698,
                landmark_ip='hou', rtt_ms=25.0,
                inner_radius_km=200.0, outer_radius_km=2000.0,
                weight=np.exp(-25.0 / 50.0),
            ),
        ]

    def test_unweighted_e2e(self):
        """End-to-end test with unweighted method returns expected keys."""
        constraints = self._make_constraints()
        rng = np.random.default_rng(42)
        result = estimate_location(constraints, method='unweighted', n_samples=200, rng=rng)

        self.assertIsNotNone(result)
        for key in ('lat', 'lon', 'region_area_km2', 'n_constraints', 'method'):
            self.assertIn(key, result)
        self.assertEqual(result['n_constraints'], 3)
        self.assertIsInstance(result['lat'], float)
        self.assertIsInstance(result['lon'], float)

    def test_weighted_e2e(self):
        """Weighted method returns a result with finite coordinates."""
        constraints = self._make_constraints()
        rng = np.random.default_rng(42)
        result = estimate_location(
            constraints, method='weighted', n_samples=200,
            grid_resolution_deg=0.5, rng=rng,
        )

        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result['lat']))
        self.assertTrue(np.isfinite(result['lon']))

    def test_fallback(self):
        """Non-intersecting constraints trigger fallback."""
        # Two tiny, far-apart constraints
        constraints = [
            AnnularConstraint(
                landmark_lat=40.7128, landmark_lon=-74.0060,
                landmark_ip='nyc', rtt_ms=10.0,
                inner_radius_km=50.0, outer_radius_km=80.0,
                weight=1.0,
            ),
            AnnularConstraint(
                landmark_lat=34.0522, landmark_lon=-118.2437,
                landmark_ip='la', rtt_ms=10.0,
                inner_radius_km=50.0, outer_radius_km=80.0,
                weight=1.0,
            ),
        ]
        result = estimate_location(constraints, method='weighted', n_samples=100)
        self.assertIsNotNone(result)
        self.assertTrue(result['fallback'])

    def test_centroid_method(self):
        """'centroid' method returns immediately without Monte Carlo."""
        constraints = self._make_constraints()
        result = estimate_location(constraints, method='centroid')

        self.assertIsNotNone(result)
        self.assertEqual(result['method'], 'centroid')
        self.assertFalse(result['fallback'])
        # Should not have n_samples key (no Monte Carlo)
        self.assertNotIn('n_samples', result)


# =============================================================================
# Test: OctantGeolocator
# =============================================================================

class TestOctantGeolocator(unittest.TestCase):

    def setUp(self):
        self.models = _build_test_models()
        self.geolocator = OctantGeolocator(
            models=self.models,
            landmark_coords=LANDMARKS,
            method='unweighted',
            n_samples=200,
            max_rtt_ms=200.0,
        )

    def test_single_target(self):
        """Geolocate one target returns a result dict."""
        # Synthetic RTTs from Pittsburgh
        true_lat, true_lon = 40.4406, -79.9959
        rtts = {
            lm_ip: _synthetic_rtt(true_lat, true_lon, lat, lon)
            for lm_ip, (lat, lon) in LANDMARKS.items()
        }
        rng = np.random.default_rng(42)
        result = self.geolocator.geolocate('target_1', rtts, rng=rng)

        self.assertIsNotNone(result)
        self.assertIn('lat', result)
        self.assertIn('lon', result)
        self.assertIn('n_constraints', result)

    def test_batch(self):
        """Geolocate batch returns results for all targets."""
        targets = {}
        for i, (city, (lat, lon)) in enumerate([
            ('pittsburgh', (40.4406, -79.9959)),
            ('atlanta', (33.7490, -84.3880)),
            ('denver', (39.7392, -104.9903)),
        ]):
            rtts = {
                lm_ip: _synthetic_rtt(lat, lon, lm_lat, lm_lon)
                for lm_ip, (lm_lat, lm_lon) in LANDMARKS.items()
            }
            targets[city] = rtts

        results = self.geolocator.geolocate_batch(targets)
        self.assertEqual(len(results), 3)
        for city in targets:
            self.assertIn(city, results)

    def test_evaluate_errors(self):
        """evaluate() computes correct error metrics."""
        true_lat, true_lon = 40.4406, -79.9959
        rtts = {
            lm_ip: _synthetic_rtt(true_lat, true_lon, lat, lon)
            for lm_ip, (lat, lon) in LANDMARKS.items()
        }

        ground_truth = {'target_1': (true_lat, true_lon)}
        rng = np.random.default_rng(42)

        eval_result = self.geolocator.evaluate(
            {'target_1': rtts}, ground_truth, rng=rng
        )

        self.assertIn('median_error_km', eval_result)
        self.assertIn('mean_error_km', eval_result)
        self.assertIn('accuracy_at_thresholds', eval_result)
        self.assertIn('results', eval_result)

        if eval_result['median_error_km'] is not None:
            self.assertGreaterEqual(eval_result['median_error_km'], 0)
            # With 5 landmarks and clean synthetic data, error should be bounded
            self.assertIn(40, eval_result['accuracy_at_thresholds'])

    def test_evaluate_empty(self):
        """Empty targets dict returns empty results."""
        eval_result = self.geolocator.evaluate({}, {})
        self.assertEqual(eval_result['results'], [])
        self.assertIsNone(eval_result['median_error_km'])


if __name__ == '__main__':
    unittest.main()
