"""Phase 4 variant: Boundary Vertex Mean Centroid.

Simple average of feasible-region boundary vertices.

Wraps:
  - scripts/framework/geometry.py :: polygon_centroid()  (>2 vertices)
  - scripts/framework/geometry.py :: get_middle_intersection()  (2 vertices)
"""

from __future__ import annotations

from typing import Optional, Tuple

from scripts.framework.centroid import BaseCentroid
from scripts.framework.geometry import get_middle_intersection, polygon_centroid
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult


@register_centroid("boundary_vertex_mean")
class BoundaryVertexMeanCentroid(BaseCentroid):
    """Coordinate mean of boundary vertices.

    For vertex lists: simple coordinate average (polygon_centroid).
    For 2 vertices: geodetic midpoint (get_middle_intersection).
    For Shapely regions: extracts exterior and interior ring vertices, then
    averages them. This is a boundary-vertex mean, not an area centroid.
    """

    name = "boundary_vertex_mean"

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        if not result.success:
            return None

        # Vertex-based path (from spherical_circle multilateration)
        if result.vertices is not None:
            n = len(result.vertices)
            if n > 2:
                return polygon_centroid(result.vertices)
            if n == 2:
                return get_middle_intersection(result.vertices)
            if n == 1:
                return result.vertices[0]
            return None

        # Shapely region path (from planar multilateration)
        if result.region is not None:
            coords = _extract_vertex_coords(result.region)
            if coords:
                return polygon_centroid(coords)

        return None


def _extract_vertex_coords(geom) -> list:
    """Extract (lat, lon) boundary vertex coordinates from a Shapely geometry.

    Shapely stores coordinates as (x, y) = (lon, lat).
    """
    from shapely.geometry import MultiPolygon

    if isinstance(geom, MultiPolygon):
        all_coords = []
        for poly in geom.geoms:
            all_coords.extend(_polygon_boundary_coords(poly))
        return all_coords
    if hasattr(geom, "exterior"):
        return _polygon_boundary_coords(geom)
    return []


def _ring_coords(ring) -> list:
    """Return ring coordinates without the duplicate closing vertex."""
    coords = list(ring.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(lat, lon) for lon, lat in coords]


def _polygon_boundary_coords(poly) -> list:
    """Return exterior plus interior-ring vertices for a Shapely polygon."""
    coords = _ring_coords(poly.exterior)
    for interior in poly.interiors:
        coords.extend(_ring_coords(interior))
    return coords
