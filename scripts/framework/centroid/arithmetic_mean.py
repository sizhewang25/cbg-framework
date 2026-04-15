"""Phase 4 variant: Arithmetic Mean Centroid (Million-Scale CBG).

Simple average of intersection vertex coordinates.

Wraps:
  - scripts/utils/helpers.py :: polygon_centroid()  (>2 vertices)
  - scripts/utils/helpers.py :: get_middle_intersection()  (2 vertices)
"""

from __future__ import annotations

from typing import Optional, Tuple

from scripts.framework.centroid import BaseCentroid
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult
from scripts.utils.helpers import get_middle_intersection, polygon_centroid


@register_centroid("arithmetic_mean")
class ArithmeticMeanCentroid(BaseCentroid):
    """Arithmetic mean of intersection points.

    For vertex lists: simple coordinate average (polygon_centroid).
    For 2 vertices: geodetic midpoint (get_middle_intersection).
    For Shapely regions: extract boundary vertices, then average.
    """

    name = "arithmetic_mean"

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        if not result.success:
            return None

        # Vertex-based path (from spherical multilateration)
        if result.vertices is not None:
            n = len(result.vertices)
            if n > 2:
                return polygon_centroid(result.vertices)
            if n == 2:
                return get_middle_intersection(result.vertices)
            if n == 1:
                return result.vertices[0]
            return None

        # Shapely region path (from shapely/weighted multilateration)
        if result.region is not None:
            coords = _extract_vertex_coords(result.region)
            if coords:
                return polygon_centroid(coords)

        return None


def _extract_vertex_coords(geom) -> list:
    """Extract (lat, lon) vertex coordinates from a Shapely geometry.

    Shapely stores coordinates as (x, y) = (lon, lat).
    """
    from shapely.geometry import MultiPolygon

    if isinstance(geom, MultiPolygon):
        all_coords = []
        for poly in geom.geoms:
            all_coords.extend(
                [(lat, lon) for lon, lat in poly.exterior.coords]
            )
        return all_coords
    if hasattr(geom, "exterior"):
        return [(lat, lon) for lon, lat in geom.exterior.coords]
    return []
