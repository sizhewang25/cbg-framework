"""Boundary-vertex mean centroid (v2 port).

Coordinate mean of feasible-region boundary vertices. Mirrors v1's
BoundaryVertexMeanCentroid; the only changes are the MTLResult input shape,
the CTRResult output shape, and Error-coded failure paths.
"""

from __future__ import annotations

from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry

from scripts.framework.geometry import arithmetic_mean_centroid
from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.registry import register_ctr
from scripts.framework.v2.types import Coord, Error


@register_ctr("boundary_vertex_mean")
class BoundaryVertexMeanCTR(CTRMethod):
    """Coordinate mean of boundary vertices.

    Always the finite-set arithmetic mean of vertices, independent of count.
    For Shapely regions: extracts exterior and interior ring vertices, then
    averages them. This is a boundary-vertex mean, not an area centroid.
    """

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        if not mtl.success:
            return CTRResult(success=False, error=Error.EMPTY_REGION)

        intersection = mtl.intersection

        if isinstance(intersection, BaseGeometry):
            if intersection.is_empty:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            coords = _extract_vertex_coords(intersection)
            if not coords:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            lat, lon = arithmetic_mean_centroid(coords)
            return CTRResult(success=True, tg_coord=Coord(lat, lon))

        if isinstance(intersection, list):
            if not intersection:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            verts = [(c.lat, c.lon) for c in intersection]
            lat, lon = arithmetic_mean_centroid(verts)
            return CTRResult(success=True, tg_coord=Coord(lat, lon))

        return CTRResult(success=False, error=Error.EMPTY_REGION)


def _extract_vertex_coords(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Extract (lat, lon) boundary vertex coords from a Shapely geometry.

    Shapely stores coordinates as (x, y) = (lon, lat).
    """
    if isinstance(geom, MultiPolygon):
        all_coords: list[tuple[float, float]] = []
        for poly in geom.geoms:
            all_coords.extend(_polygon_boundary_coords(poly))
        return all_coords
    if hasattr(geom, "exterior"):
        return _polygon_boundary_coords(geom)
    return []


def _ring_coords(ring) -> list[tuple[float, float]]:
    coords = list(ring.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(lat, lon) for lon, lat in coords]


def _polygon_boundary_coords(poly) -> list[tuple[float, float]]:
    coords = _ring_coords(poly.exterior)
    for interior in poly.interiors:
        coords.extend(_ring_coords(interior))
    return coords
