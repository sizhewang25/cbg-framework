"""Shared test helpers for centroid wrapper tests."""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon

from scripts.framework.types import MultilatResult


def successful_vertices(vertices: list[tuple[float, float]]) -> MultilatResult:
    """Build a successful spherical-circle-style vertex result."""
    return MultilatResult(vertices=vertices, success=True)


def successful_region(region) -> MultilatResult:
    """Build a successful planar-region result."""
    return MultilatResult(region=region, success=True)


def failed_vertices(vertices: list[tuple[float, float]]) -> MultilatResult:
    """Build a failed multilateration result carrying ignored vertices."""
    return MultilatResult(vertices=vertices, success=False)


def rectangle_region() -> Polygon:
    """Return a 4x2 rectangle in Shapely (lon, lat) coordinates."""
    return Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])


def polygon_with_asymmetric_hole() -> Polygon:
    """Return a polygon whose hole changes the boundary-vertex mean."""
    return Polygon(
        [(0, 0), (6, 0), (6, 4), (0, 4)],
        [[(1, 1), (1, 2), (2, 2), (2, 1)]],
    )


def two_rectangle_multipolygon() -> MultiPolygon:
    """Return two equal rectangles separated in longitude."""
    return MultiPolygon([
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(10, 0), (12, 0), (12, 2), (10, 2)]),
    ])


def unequal_area_multipolygon() -> MultiPolygon:
    """Return two disconnected rectangles with 1:2 area ratio."""
    return MultiPolygon([
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(10, 0), (14, 0), (14, 2), (10, 2)]),
    ])


def assert_point_almost_equal(testcase, point, expected, places: int = 7) -> None:
    """Assert two (lat, lon) points match component-wise."""
    testcase.assertIsNotNone(point)
    testcase.assertAlmostEqual(point[0], expected[0], places=places)
    testcase.assertAlmostEqual(point[1], expected[1], places=places)
