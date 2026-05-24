"""Tests for scripts.vp_selection.pair_distances.

TDD-first: written before pair_distances.py exists. Defines the API contract
the implementation must satisfy.
"""

from __future__ import annotations

import unittest

from scripts.libs.cbg.rtt_model import haversine_distance
from scripts.vp_selection.pair_distances import compute_geodesic_distances


class TestComputeGeodesicDistances(unittest.TestCase):

    def test_empty_pool_returns_empty(self):
        self.assertEqual(compute_geodesic_distances({}), {})

    def test_single_vp_returns_empty(self):
        """One VP means zero pairs."""
        self.assertEqual(
            compute_geodesic_distances({"only": (0.0, 0.0)}),
            {},
        )

    def test_canonical_key_is_min_first(self):
        """Pair key is always (lex_min, lex_max) — never the reverse."""
        coords = {"b": (0.0, 0.0), "a": (10.0, 10.0)}
        distances = compute_geodesic_distances(coords)
        self.assertEqual(list(distances.keys()), [("a", "b")])
        self.assertNotIn(("b", "a"), distances)

    def test_no_self_distance_keys(self):
        coords = {"x": (0.0, 0.0), "y": (1.0, 1.0), "z": (2.0, 2.0)}
        distances = compute_geodesic_distances(coords)
        for a, b in distances:
            self.assertNotEqual(a, b)

    def test_pair_count_is_n_choose_2(self):
        coords = {chr(ord("a") + i): (float(i), float(i)) for i in range(5)}
        distances = compute_geodesic_distances(coords)
        # 5 choose 2 = 10
        self.assertEqual(len(distances), 10)

    def test_matches_haversine_helper(self):
        """Wrapper uses the same haversine as RTTDistanceModel — no rounding loss,
        no swapped lat/lon."""
        coords = {
            "london": (51.5074, -0.1278),
            "paris": (48.8566, 2.3522),
            "nyc": (40.7128, -74.0060),
        }
        distances = compute_geodesic_distances(coords)
        expected_london_paris = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        expected_london_nyc = haversine_distance(51.5074, -0.1278, 40.7128, -74.0060)
        expected_nyc_paris = haversine_distance(40.7128, -74.0060, 48.8566, 2.3522)
        self.assertAlmostEqual(distances[("london", "paris")], expected_london_paris, places=6)
        self.assertAlmostEqual(distances[("london", "nyc")], expected_london_nyc, places=6)
        self.assertAlmostEqual(distances[("nyc", "paris")], expected_nyc_paris, places=6)

    def test_known_london_paris_distance(self):
        """Independent sanity: London-Paris is ~344 km; we should be within 1 km."""
        coords = {
            "london": (51.5074, -0.1278),
            "paris": (48.8566, 2.3522),
        }
        distances = compute_geodesic_distances(coords)
        self.assertAlmostEqual(distances[("london", "paris")], 344.0, delta=1.0)

    def test_known_nyc_la_distance(self):
        """NYC-LA is ~3936 km via spherical (R=6371) haversine; within 10 km
        of the commonly-cited 3940-3944 km ellipsoidal references."""
        coords = {
            "nyc": (40.7128, -74.0060),
            "la": (34.0522, -118.2437),
        }
        distances = compute_geodesic_distances(coords)
        self.assertAlmostEqual(distances[("la", "nyc")], 3940.0, delta=10.0)

    def test_symmetry_via_canonical_lookup(self):
        """Whichever order we ask for a pair, we get the same distance back."""
        coords = {"x": (40.0, -74.0), "y": (48.0, 2.0)}
        distances = compute_geodesic_distances(coords)
        a, b = sorted(("x", "y"))
        self.assertIn((a, b), distances)
        # there's only one canonical entry; symmetry is by construction
        self.assertEqual(len(distances), 1)


if __name__ == "__main__":
    unittest.main()
