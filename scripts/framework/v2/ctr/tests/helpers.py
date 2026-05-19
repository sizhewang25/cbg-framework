"""Shared test helpers for v2 ctr wrapper tests."""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon

from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.types import Coord, Error


def successful_vertices(vertices: list[tuple[float, float]]) -> MTLResult:
    """Build a successful spherical-style vertex result.

    Accepts (lat, lon) tuples and wraps them as Coord — matches what a
    CircleMTLMethod returns for the spherical branch.
    """
    return MTLResult(
        success=True,
        intersection=[Coord(lat, lon) for lat, lon in vertices],
    )


def successful_region(region) -> MTLResult:
    """Build a successful planar-region result (Shapely geometry)."""
    return MTLResult(success=True, intersection=region)


def failed_vertices(vertices: list[tuple[float, float]]) -> MTLResult:
    """Build a failed multilateration result carrying ignored vertices.

    Mirrors v1's failed_vertices: success=False; the carried intersection
    should not be consumed by CTR (defensive guards return EMPTY_REGION).
    """
    return MTLResult(
        success=False,
        error=Error.NO_INTERSECTION,
        intersection=[Coord(lat, lon) for lat, lon in vertices],
    )


def rectangle_region() -> Polygon:
    """4x2 rectangle in Shapely (lon, lat) coordinates."""
    return Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])


def polygon_with_asymmetric_hole() -> Polygon:
    """Polygon whose hole changes the boundary-vertex mean."""
    return Polygon(
        [(0, 0), (6, 0), (6, 4), (0, 4)],
        [[(1, 1), (1, 2), (2, 2), (2, 1)]],
    )


def two_rectangle_multipolygon() -> MultiPolygon:
    """Two equal rectangles separated in longitude."""
    return MultiPolygon([
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(10, 0), (12, 0), (12, 2), (10, 2)]),
    ])


def unequal_area_multipolygon() -> MultiPolygon:
    """Two disconnected rectangles with 1:2 area ratio."""
    return MultiPolygon([
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(10, 0), (14, 0), (14, 2), (10, 2)]),
    ])


def assert_coord_almost_equal(
    testcase, coord: Coord, expected: tuple[float, float], places: int = 7,
) -> None:
    """Assert Coord matches an expected (lat, lon) tuple component-wise."""
    testcase.assertIsNotNone(coord)
    testcase.assertAlmostEqual(coord.lat, expected[0], places=places)
    testcase.assertAlmostEqual(coord.lon, expected[1], places=places)
