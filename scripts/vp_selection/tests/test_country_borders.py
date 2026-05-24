"""Tests for scripts.vp_selection.country_borders.

TDD-first. Unit tests use synthetic polygons so they don't depend on the
Natural Earth shapefile being downloaded; real-shapefile behaviour is verified
in the agreement smoke run.
"""

from __future__ import annotations

import unittest

from shapely.geometry import Point, Polygon, box

from scripts.vp_selection.country_borders import (
    nearest_border_distance_km,
    precompute_landmark_country_distances,
)


def _square_polygons() -> dict[str, Polygon]:
    """Three non-overlapping 1°×1° boxes for synthetic tests."""
    return {
        "A": box(0.0, 0.0, 1.0, 1.0),      # equator, prime meridian
        "B": box(10.0, 0.0, 11.0, 1.0),    # 10° east of A, same latitude
        "C": box(0.0, 20.0, 1.0, 21.0),    # 20° north of A
    }


class TestNearestBorderDistance(unittest.TestCase):

    def test_landmark_inside_polygon_returns_zero(self):
        polygons = _square_polygons()
        # Landmark at (0.5° lat, 0.5° lon) is inside box A
        self.assertEqual(
            nearest_border_distance_km((0.5, 0.5), "A", polygons),
            0.0,
        )

    def test_landmark_far_from_polygon_returns_positive(self):
        polygons = _square_polygons()
        # Landmark at (0.5° lat, 5.5° lon): ~4.5° east of A's east edge
        d = nearest_border_distance_km((0.5, 5.5), "A", polygons)
        self.assertGreater(d, 0.0)

    def test_distance_scales_with_distance(self):
        polygons = _square_polygons()
        # Landmark at (0.5° lat, 5.5° lon) vs (0.5° lat, 15.5° lon) — both
        # outside A, but the second is much further
        d_near = nearest_border_distance_km((0.5, 5.5), "A", polygons)
        d_far = nearest_border_distance_km((0.5, 15.5), "A", polygons)
        self.assertLess(d_near, d_far)

    def test_unknown_country_raises(self):
        polygons = _square_polygons()
        with self.assertRaises(KeyError):
            nearest_border_distance_km((0.0, 0.0), "ZZ", polygons)

    def test_distance_to_eastern_edge_matches_haversine_approx(self):
        """Landmark at (0.5°, 5.5°) — 4.5° east of box A's eastern edge (at
        lon=1.0). At the equator, 1° lon ≈ 111.32 km, so ~501 km expected.
        Allow 5% tolerance for projection."""
        polygons = {"A": box(0.0, 0.0, 1.0, 1.0)}
        d = nearest_border_distance_km((0.5, 5.5), "A", polygons)
        expected_km = 4.5 * 111.32  # ≈ 501
        self.assertAlmostEqual(d, expected_km, delta=0.05 * expected_km)


class TestPrecomputeLandmarkCountryDistances(unittest.TestCase):

    def test_returns_n_landmarks_times_n_countries_entries(self):
        polygons = _square_polygons()
        landmarks = {
            "L1": (0.0, 0.0),
            "L2": (0.5, 10.5),  # inside B
            "L3": (5.0, 5.0),   # outside everything
        }
        table = precompute_landmark_country_distances(
            landmarks=landmarks,
            country_iso2s=["A", "B", "C"],
            polygons=polygons,
        )
        self.assertEqual(len(table), 3 * 3)
        for (lm_id, cc), v in table.items():
            self.assertIn(lm_id, landmarks)
            self.assertIn(cc, polygons)
            self.assertGreaterEqual(v, 0.0)

    def test_landmark_inside_country_is_zero(self):
        polygons = _square_polygons()
        landmarks = {"inside_B": (0.5, 10.5)}
        table = precompute_landmark_country_distances(
            landmarks=landmarks,
            country_iso2s=["B"],
            polygons=polygons,
        )
        self.assertEqual(table[("inside_B", "B")], 0.0)

    def test_skips_unknown_country_silently(self):
        """If a requested country isn't in `polygons`, skip rather than KeyError
        — caller may pass a superset country list (e.g. all ISO_A2 codes from
        anchor metadata, including codes Natural Earth uses different aliases for)."""
        polygons = {"A": box(0.0, 0.0, 1.0, 1.0)}
        landmarks = {"L": (5.0, 5.0)}
        table = precompute_landmark_country_distances(
            landmarks=landmarks,
            country_iso2s=["A", "MISSING"],
            polygons=polygons,
        )
        self.assertIn(("L", "A"), table)
        self.assertNotIn(("L", "MISSING"), table)


if __name__ == "__main__":
    unittest.main()
