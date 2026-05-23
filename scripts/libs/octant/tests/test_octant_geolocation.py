"""Unit tests for the octant geolocation pipeline.

`TestComputeFeasibleRegionWeightedFaceDecomposition` covers the paper-
faithful invariants of the weighted feasible region: representative-point
face weights, top-weight selection, disconnected results, inner-hole
exclusion, and a divergence case where a sparse-greedy variant would miss
support from low-weight annuli.
"""

import unittest
import numpy as np
from shapely.geometry import MultiPolygon, Point

from scripts.libs.octant.octant_model import OctantRTTModel
from scripts.libs.octant.octant_geolocation import (
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
    """Create a fitted OctantRTTModel with synthetic calibration data."""
    rng = np.random.RandomState(hash(anchor_ip) % 2**31)
    rtts = np.linspace(5, 100, 60)
    distances = 100.0 * rtts + rng.uniform(-150, 150, len(rtts))
    distances = np.maximum(distances, 10.0)

    model = OctantRTTModel(anchor_ip=anchor_ip, anchor_lat=lat, anchor_lon=lon)
    model.fit(rtts, distances)
    return model


def _synthetic_rtt(true_lat: float, true_lon: float,
                   lm_lat: float, lm_lon: float,
                   noise_ms: float = 2.0) -> float:
    dist_km = haversine((true_lat, true_lon), (lm_lat, lm_lon))
    speed_km_per_ms = 300.0 * 2 / 3
    rtt = 2 * dist_km / speed_km_per_ms
    rng = np.random.RandomState(int(abs(true_lat * 1000 + lm_lat * 100)) % 2**31)
    return max(1.0, rtt + rng.uniform(0, noise_ms))


LANDMARKS = {
    'lm_nyc': (40.7128, -74.0060),
    'lm_chi': (41.8781, -87.6298),
    'lm_la':  (34.0522, -118.2437),
    'lm_hou': (29.7604, -95.3698),
    'lm_sea': (47.6062, -122.3321),
}


def _build_test_models():
    return {ip: _make_fitted_model(ip, lat, lon) for ip, (lat, lon) in LANDMARKS.items()}


# =============================================================================
# Test: Constraint Formation
# =============================================================================

class TestConstraintFormation(unittest.TestCase):

    def setUp(self):
        self.models = _build_test_models()

    def test_form_single_constraint(self):
        model = self.models['lm_nyc']
        c = form_constraint(40.7128, -74.0060, 'lm_nyc', 30.0, model, weight_tau_ms=50.0)

        self.assertIsInstance(c, AnnularConstraint)
        self.assertGreater(c.outer_radius_km, c.inner_radius_km)
        self.assertGreaterEqual(c.inner_radius_km, 0)
        self.assertAlmostEqual(c.weight, np.exp(-30.0 / 50.0), places=6)
        self.assertEqual(c.rtt_ms, 30.0)

    def test_filters_unfitted_models(self):
        unfitted = OctantRTTModel(anchor_ip='unfitted', anchor_lat=0, anchor_lon=0)
        models = {**self.models, 'unfitted': unfitted}
        rtts = {'lm_nyc': 20.0, 'unfitted': 20.0}

        constraints = form_constraints('target', rtts, LANDMARKS, models)
        ips = [c.landmark_ip for c in constraints]
        self.assertIn('lm_nyc', ips)
        self.assertNotIn('unfitted', ips)

    def test_keeps_high_rtt(self):
        rtts = {'lm_nyc': 20.0, 'lm_chi': 300.0}
        constraints = form_constraints(
            'target', rtts, LANDMARKS, self.models, max_rtt_ms=200.0
        )
        ips = [c.landmark_ip for c in constraints]
        self.assertIn('lm_nyc', ips)
        self.assertIn('lm_chi', ips)

    def test_sorted_by_weight(self):
        rtts = {'lm_nyc': 50.0, 'lm_chi': 10.0, 'lm_la': 80.0}
        constraints = form_constraints('target', rtts, LANDMARKS, self.models)

        weights = [c.weight for c in constraints]
        self.assertEqual(weights, sorted(weights, reverse=True))
        self.assertEqual(constraints[0].landmark_ip, 'lm_chi')

    def test_empty_input(self):
        constraints = form_constraints('target', {}, LANDMARKS, self.models)
        self.assertEqual(constraints, [])


# =============================================================================
# Test: Annulus Geometry
# =============================================================================

class TestAnnulusGeometry(unittest.TestCase):

    def test_zero_radius_circle_uses_geometry_epsilon(self):
        result = _circle_to_shapely(40.0, -74.0, 0.0, n_pts=64)
        self.assertFalse(result.is_empty)
        self.assertGreater(result.area, 0)

    def test_annulus_valid(self):
        result = _annulus_to_shapely(40.0, -74.0, 100.0, 500.0, n_pts=64)
        self.assertIsNotNone(result)
        self.assertFalse(result.is_empty)
        self.assertGreater(result.area, 0)

        if hasattr(result, 'interiors'):
            self.assertGreater(len(list(result.interiors)), 0)

    def test_annulus_degenerate(self):
        self.assertIsNone(_annulus_to_shapely(40.0, -74.0, 500.0, 100.0))
        self.assertIsNone(_annulus_to_shapely(40.0, -74.0, 500.0, 500.0))

    def test_point_in_annulus_inside(self):
        c = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='test', rtt_ms=10.0,
            inner_radius_km=100.0, outer_radius_km=300.0,
            weight=1.0,
        )
        self.assertTrue(_point_in_annulus(39.9526, -75.1652, c))

    def test_point_in_annulus_outside(self):
        c = AnnularConstraint(
            landmark_lat=40.7128, landmark_lon=-74.0060,
            landmark_ip='test', rtt_ms=10.0,
            inner_radius_km=200.0, outer_radius_km=300.0,
            weight=1.0,
        )
        self.assertFalse(_point_in_annulus(40.72, -74.01, c))
        self.assertFalse(_point_in_annulus(34.05, -118.24, c))


# =============================================================================
# Test: Region Computation
# =============================================================================

class TestRegionComputation(unittest.TestCase):

    def test_unweighted_two_constraints(self):
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
        region = compute_feasible_region_weighted([c1, c2], weight_threshold=0.5)
        self.assertIsNotNone(region)
        self.assertFalse(region.is_empty)

    def test_threshold_effect(self):
        """Higher threshold needs more faces → equal or larger area."""
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

        region_low = compute_feasible_region_weighted(constraints, weight_threshold=0.3)
        region_high = compute_feasible_region_weighted(constraints, weight_threshold=0.9)

        if region_low is not None and region_high is not None:
            # Cumulative selection: more faces needed at higher threshold.
            self.assertGreaterEqual(region_high.area, region_low.area)

    def test_region_contains_true_location(self):
        true_lat, true_lon = 40.4406, -79.9959
        models = _build_test_models()

        rtts = {}
        for lm_ip, (lm_lat, lm_lon) in LANDMARKS.items():
            rtts[lm_ip] = _synthetic_rtt(true_lat, true_lon, lm_lat, lm_lon, noise_ms=1.0)

        constraints = form_constraints(
            'target', rtts, LANDMARKS, models, max_rtt_ms=200.0
        )
        self.assertGreater(len(constraints), 0)

        region = compute_feasible_region_unweighted(constraints, n_pts=64)

        if region is not None:
            centroid = region.centroid
            est_lat, est_lon = centroid.y, centroid.x
            error_km = haversine((est_lat, est_lon), (true_lat, true_lon))
            self.assertLess(error_km, 1500.0)
        else:
            result = estimate_location(constraints, method='centroid')
            self.assertIsNotNone(result)
            error_km = haversine((result['lat'], result['lon']), (true_lat, true_lon))
            self.assertLess(error_km, 2000.0)


# =============================================================================
# Test: compute_feasible_region_weighted — paper-faithful face decomposition
# =============================================================================

class TestComputeFeasibleRegionWeightedFaceDecomposition(unittest.TestCase):
    """Invariants the grid version did not satisfy."""

    def test_empty_input_returns_none(self):
        self.assertIsNone(compute_feasible_region_weighted([]))

    def test_single_annulus_returns_the_annulus(self):
        """One annulus → one face; result is (approximately) that annulus."""
        c = AnnularConstraint(
            landmark_lat=0.0, landmark_lon=0.0, landmark_ip='A',
            rtt_ms=0.0,
            inner_radius_km=0.0, outer_radius_km=222.0,
            weight=1.0,
        )
        region = compute_feasible_region_weighted([c], weight_threshold=0.5)
        self.assertIsNotNone(region)
        # 222 km ≈ 2° at the equator → unit disk in degree space, area ≈ π·2².
        self.assertAlmostEqual(region.area, np.pi * 4, delta=0.2)
        self.assertTrue(region.contains(Point(0.0, 0.0)))

    def test_two_overlapping_disks_default_threshold_keeps_only_lens(self):
        """Two equal-weight disks: lens has weight 2.0, crescents 1.0 each.
        Threshold 0.5 → target 1.0. Lens alone clears it; crescents excluded."""
        c1 = AnnularConstraint(
            landmark_lat=0.0, landmark_lon=0.0, landmark_ip='L',
            rtt_ms=0.0, inner_radius_km=0.0, outer_radius_km=222.0,
            weight=1.0,
        )
        c2 = AnnularConstraint(
            landmark_lat=0.0, landmark_lon=1.0, landmark_ip='R',
            rtt_ms=0.0, inner_radius_km=0.0, outer_radius_km=222.0,
            weight=1.0,
        )
        region = compute_feasible_region_weighted([c1, c2], weight_threshold=0.5)
        self.assertIsNotNone(region)
        # Lens is contained in both disks; crescent-only points are not.
        self.assertTrue(region.contains(Point(0.5, 0.0)))    # in both disks
        self.assertFalse(region.contains(Point(-1.5, 0.0)))  # only in L
        self.assertFalse(region.contains(Point(2.5, 0.0)))   # only in R

    def test_inner_disk_hole_is_excluded(self):
        """A face that lies inside an annulus's inner hole has weight 0
        and must be filtered out."""
        c = AnnularConstraint(
            landmark_lat=0.0, landmark_lon=0.0, landmark_ip='A',
            rtt_ms=0.0,
            inner_radius_km=222.0,   # 2° hole
            outer_radius_km=444.0,   # 4° outer
            weight=1.0,
        )
        region = compute_feasible_region_weighted([c], weight_threshold=0.5)
        self.assertIsNotNone(region)
        # Origin sits inside the hole → must not be in the region.
        self.assertFalse(region.contains(Point(0.0, 0.0)))
        # A point in the annulus body should be.
        self.assertTrue(region.contains(Point(3.0, 0.0)))

    def test_disconnected_result_is_multipolygon(self):
        """Two far-apart heavy overlaps → top-tier faces in two locations.
        With a high enough threshold, the selected set is disconnected."""
        # Pair 1 overlaps near (lat=0, lon=-0.5).
        # Pair 2 overlaps near (lat=0, lon=5.5).
        constraints = [
            AnnularConstraint(0.0, 0.0, 'A', 0.0, 0.0, 222.0, 1.0),
            AnnularConstraint(0.0, -1.0, 'B', 0.0, 0.0, 222.0, 1.0),
            AnnularConstraint(0.0, 5.0, 'C', 0.0, 0.0, 222.0, 1.0),
            AnnularConstraint(0.0, 6.0, 'D', 0.0, 0.0, 222.0, 1.0),
        ]
        # Σw = 4. Threshold 0.6 → target 2.4. Each pair-overlap face has
        # weight 2.0, so we need both → MultiPolygon with two parts.
        region = compute_feasible_region_weighted(constraints, weight_threshold=0.6)
        self.assertIsNotNone(region)
        self.assertEqual(region.geom_type, 'MultiPolygon')
        self.assertEqual(len(region.geoms), 2)
        # One point inside each overlap region.
        self.assertTrue(region.contains(Point(-0.5, 0.0)))
        self.assertTrue(region.contains(Point(5.5, 0.0)))

    def test_paper_vs_greedy_divergence(self):
        """Low-weight annuli still contribute to faces they cover.

        Setup: A(w=0.9) is a large disk at the origin; B(w=0.5) overlaps A
        on the west; C(w=0.4) and D(w=0.3) both cover an eastern lobe inside
        A, with D ⊂ C ⊂ A∪... — specifically D sits entirely inside both A
        and C.

        Face {A,C,D} has weight 1.6.
        Face {A,B}   has weight 1.4.
        Σw = 2.1 → threshold 0.5 ⇒ target 1.05. The {A,C,D} face alone
        clears the target and the algorithm stops there.

        A greedy "intersect annuli in weight order until cumulative weight
        ≥ target" variant would pick A∩B (cumulative 1.4 ≥ 1.05) — a
        completely different region. This test pins the paper algorithm.
        """
        constraints = [
            # A: big disk at origin
            AnnularConstraint(0.0,  0.0, 'A', 0.0, 0.0, 444.0, 0.9),
            # B: west of A
            AnnularConstraint(0.0, -2.5, 'B', 0.0, 0.0, 222.0, 0.5),
            # C: east of A
            AnnularConstraint(0.0,  2.5, 'C', 0.0, 0.0, 222.0, 0.4),
            # D: same center as C but smaller, so D ⊂ C
            AnnularConstraint(0.0,  2.5, 'D', 0.0, 0.0, 111.0, 0.3),
        ]
        region = compute_feasible_region_weighted(constraints, weight_threshold=0.5)
        self.assertIsNotNone(region)

        # Top face is the small D-disk (inside A and inside C).
        self.assertTrue(region.contains(Point(2.5, 0.0)))    # in A∩C∩D
        # Greedy would have selected A∩B near (lon=-2.5, lat=0).
        self.assertFalse(region.contains(Point(-2.5, 0.0)))  # would be A∩B
        # Plain interior of A (no extra coverage) — not in top face either.
        self.assertFalse(region.contains(Point(0.0, 0.0)))

    def test_disjoint_disks_high_threshold_accumulates_all(self):
        """Three disjoint disks. Σw = 1.0; threshold 0.9 → target 0.9.
        Take 0.5 (cum 0.5 < 0.9), take 0.3 (cum 0.8 < 0.9), take 0.2
        (cum 1.0 ≥ 0.9) → result is a MultiPolygon with all three."""
        constraints = [
            AnnularConstraint(0.0,  0.0, 'A', 0.0, 0.0, 111.0, 0.5),
            AnnularConstraint(0.0, 10.0, 'B', 0.0, 0.0, 111.0, 0.3),
            AnnularConstraint(0.0, 20.0, 'C', 0.0, 0.0, 111.0, 0.2),
        ]
        region = compute_feasible_region_weighted(constraints, weight_threshold=0.9)
        self.assertIsNotNone(region)
        self.assertEqual(region.geom_type, 'MultiPolygon')
        self.assertEqual(len(region.geoms), 3)


# =============================================================================
# Test: Point Selection
# =============================================================================

class TestPointSelection(unittest.TestCase):

    def _make_simple_region(self):
        return _circle_to_shapely(40.0, -74.0, 200.0, n_pts=64)

    def test_sample_count(self):
        region = self._make_simple_region()
        rng = np.random.default_rng(42)
        points = sample_points_in_region(region, n_samples=100, rng=rng)
        self.assertEqual(len(points), 100)

    def test_samples_within_region(self):
        region = self._make_simple_region()
        rng = np.random.default_rng(42)
        points = sample_points_in_region(region, n_samples=50, rng=rng)

        for lat, lon in points:
            self.assertTrue(region.contains(Point(lon, lat)))

    def test_sampling_is_deterministic_for_seeded_rng(self):
        region = self._make_simple_region()
        points_a = sample_points_in_region(region, n_samples=64, rng=np.random.default_rng(42))
        points_b = sample_points_in_region(region, n_samples=64, rng=np.random.default_rng(42))
        self.assertEqual(points_a.shape, points_b.shape)
        np.testing.assert_allclose(points_a, points_b)

    def test_median_symmetric(self):
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
        points = np.array([
            [40.0, -74.0],
            [40.1, -74.1],
            [39.9, -73.9],
            [40.05, -74.05],
            [34.0, -118.0],
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
        constraints = self._make_constraints()
        rng = np.random.default_rng(42)
        result = estimate_location(
            constraints, method='weighted', n_samples=200, rng=rng,
        )

        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result['lat']))
        self.assertTrue(np.isfinite(result['lon']))

    def test_fallback(self):
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
        # Disjoint annuli: weighted region still has positive-weight faces
        # (each annulus's own face), so the primary weighted path produces a
        # result and no fallback is triggered. Force fallback by asking for
        # unweighted on disjoint inputs.
        result = estimate_location(constraints, method='unweighted', n_samples=100)
        self.assertIsNotNone(result)
        self.assertTrue(result['fallback'])

    def test_centroid_method(self):
        constraints = self._make_constraints()
        result = estimate_location(constraints, method='centroid')

        self.assertIsNotNone(result)
        self.assertEqual(result['method'], 'centroid')
        self.assertFalse(result['fallback'])
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
        targets = {}
        for city, (lat, lon) in [
            ('pittsburgh', (40.4406, -79.9959)),
            ('atlanta', (33.7490, -84.3880)),
            ('denver', (39.7392, -104.9903)),
        ]:
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
            self.assertIn(40, eval_result['accuracy_at_thresholds'])

    def test_evaluate_empty(self):
        eval_result = self.geolocator.evaluate({}, {})
        self.assertEqual(eval_result['results'], [])
        self.assertIsNone(eval_result['median_error_km'])


if __name__ == '__main__':
    unittest.main()
