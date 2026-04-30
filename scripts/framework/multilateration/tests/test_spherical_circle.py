"""Tests for spherical-circle multilateration."""

from __future__ import annotations

import unittest

from scripts.framework.multilateration.spherical_circle import (
    SphericalCircleMultilateration,
)
from scripts.framework.multilateration.tests.helpers import circle

CAP_RADIUS_KM = 138.9936583057  # 1.25 degrees on a 6371 km Earth.
TWO_CAP_LAT_DEG = 1.1456584648  # Crossing latitude for 1.25 degree caps 1 degree apart.


class TestSphericalCircleMultilateration(unittest.TestCase):
    def test_empty_input_fails(self):
        result = SphericalCircleMultilateration().multilaterate([])

        self.assertFalse(result.success)
        self.assertIsNone(result.vertices)
        self.assertEqual(result.circles_used, [])

    def test_single_circle_returns_four_cardinal_vertices(self):
        result = SphericalCircleMultilateration().multilaterate([
            circle("a", radius_km=111.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual(len(result.vertices), 4)
        self.assertEqual([c.vp_ip for c in result.circles_used], ["a"])

        # The single-circle fallback samples cardinal points using the framework's
        # local tangent-plane approximation. 111 km is 111/6378137 radians, or
        # about 0.997 degrees.
        expected = [
            (0.0, 0.997),
            (0.997, 0.0),
            (0.0, -0.997),
            (-0.997, 0.0),
        ]
        for actual, want in zip(result.vertices, expected):
            self.assertAlmostEqual(actual[0], want[0], places=3)
            self.assertAlmostEqual(actual[1], want[1], places=3)

    def test_two_overlapping_circles_return_two_crossing_vertices(self):
        result = SphericalCircleMultilateration().multilaterate([
            circle("west", lon=0.0, radius_km=CAP_RADIUS_KM),
            circle("east", lon=1.0, radius_km=CAP_RADIUS_KM),
        ])

        self.assertTrue(result.success)
        self.assertEqual([c.vp_ip for c in result.circles_used], ["west", "east"])
        self.assertEqual(len(result.vertices), 2)

        vertices = sorted(result.vertices)
        self.assertAlmostEqual(vertices[0][0], -TWO_CAP_LAT_DEG, places=5)
        self.assertAlmostEqual(vertices[0][1], 0.5, places=5)
        self.assertAlmostEqual(vertices[1][0], TWO_CAP_LAT_DEG, places=5)
        self.assertAlmostEqual(vertices[1][1], 0.5, places=5)

    def test_three_circles_filter_to_northern_crossing_vertex(self):
        result = SphericalCircleMultilateration().multilaterate([
            circle("west", lon=0.0, radius_km=CAP_RADIUS_KM),
            circle("east", lon=1.0, radius_km=CAP_RADIUS_KM),
            circle("north", lat=TWO_CAP_LAT_DEG, lon=0.5, radius_km=100.0),
        ])

        self.assertTrue(result.success)
        self.assertEqual([c.vp_ip for c in result.circles_used], ["west", "east", "north"])
        self.assertEqual(len(result.vertices), 1)
        self.assertAlmostEqual(result.vertices[0][0], TWO_CAP_LAT_DEG, places=5)
        self.assertAlmostEqual(result.vertices[0][1], 0.5, places=5)

    def test_non_intersecting_circles_fail(self):
        result = SphericalCircleMultilateration().multilaterate([
            circle("a", lon=0.0, radius_km=50.0),
            circle("b", lon=10.0, radius_km=50.0),
        ])

        self.assertFalse(result.success)
        self.assertEqual([c.vp_ip for c in result.circles_used], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
