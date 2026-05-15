"""
Regression and smoke tests for shared Octant evaluation helpers.
"""

import unittest

import numpy as np
import pandas as pd

from scripts.libs.octant.octant_evaluation import run_octant_cbg
from scripts.libs.octant.octant_geolocation import estimate_location, form_constraints
from scripts.libs.octant.octant_model import OctantRTTModel
from scripts.utils.helpers import haversine


def _make_fitted_model(anchor_ip: str, lat: float, lon: float) -> OctantRTTModel:
    """Create a fitted synthetic Octant model for testing."""
    rng = np.random.RandomState(hash(anchor_ip) % 2**31)
    rtts = np.linspace(5, 100, 60)
    distances = 100.0 * rtts + rng.uniform(-150.0, 150.0, len(rtts))
    distances = np.maximum(distances, 10.0)

    model = OctantRTTModel(anchor_ip=anchor_ip, anchor_lat=lat, anchor_lon=lon)
    model.fit(rtts, distances)
    return model


def _synthetic_rtt(true_lat: float, true_lon: float, lm_lat: float, lm_lon: float) -> float:
    """Compute a synthetic RTT measurement from a true location to a landmark."""
    dist_km = haversine((true_lat, true_lon), (lm_lat, lm_lon))
    speed_km_per_ms = 300.0 * 2 / 3
    return max(1.0, 2 * dist_km / speed_km_per_ms + 1.0)


class TestOctantEvaluationHelpers(unittest.TestCase):

    def test_run_octant_cbg_matches_manual_geolocation_flow(self):
        """Shared helper should match the manual form_constraints + estimate flow."""
        landmarks = {
            'lm_nyc': (40.7128, -74.0060),
            'lm_chi': (41.8781, -87.6298),
            'lm_atl': (33.7490, -84.3880),
            'lm_den': (39.7392, -104.9903),
        }
        models = {
            ip: _make_fitted_model(ip, lat, lon)
            for ip, (lat, lon) in landmarks.items()
        }

        true_lat, true_lon = 39.9526, -75.1652  # Philadelphia
        probe_ip = 'test-probe'
        rows = []
        rtt_measurements = {}
        for anchor_ip, (lat, lon) in landmarks.items():
            rtt = _synthetic_rtt(true_lat, true_lon, lat, lon)
            rtt_measurements[anchor_ip] = rtt
            rows.append({
                'src_ip': probe_ip,
                'dst_ip': anchor_ip,
                'probe_latitude': true_lat,
                'probe_longitude': true_lon,
                'anchor_latitude': lat,
                'anchor_longitude': lon,
                'min_rtt': rtt,
            })
        df_asn = pd.DataFrame(rows)

        delta = 1.2
        anchor_coords = {anchor_ip: coords for anchor_ip, coords in landmarks.items()}
        constraints = form_constraints(
            probe_ip,
            rtt_measurements,
            anchor_coords,
            models,
            delta=delta,
        )
        expected = estimate_location(
            constraints,
            method='weighted',
            n_samples=5000,
            weight_threshold=0.5,
            grid_resolution_deg=0.25,
            n_pts=128,
            rng=np.random.default_rng(42),
            collect_benchmark=True,
        )

        results, all_radii, all_areas, benchmarks = run_octant_cbg(
            df_asn,
            models,
            delta,
            method_name='shared_helper_test',
            rng_seed=42,
        )

        self.assertEqual(len(results), 1)
        self.assertGreater(len(all_radii), 0)
        self.assertGreaterEqual(len(all_areas), 0)
        self.assertEqual(benchmarks['n_probes'], 1)

        result = results[0]
        self.assertIsNotNone(expected)
        self.assertAlmostEqual(result['estimated_lat'], expected['lat'], places=6)
        self.assertAlmostEqual(result['estimated_lon'], expected['lon'], places=6)
        self.assertAlmostEqual(
            result['error_km'],
            haversine((expected['lat'], expected['lon']), (true_lat, true_lon)),
            places=6,
        )
        self.assertEqual(result['geolocation_method'], expected['method'])


if __name__ == '__main__':
    unittest.main()
