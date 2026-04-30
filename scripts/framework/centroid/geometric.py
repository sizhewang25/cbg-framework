"""Phase 4 variant: Geometric Centroid (area-weighted).

Uses Shapely's .centroid — the area-weighted center of mass of the
intersection polygon. Guaranteed to be inside convex polygons.

Reference: centroid_comparison.py :: compute_shapely_centroid()
"""

from __future__ import annotations

from typing import Optional, Tuple

from shapely.geometry import Polygon as ShapelyPolygon

from scripts.framework.centroid import BaseCentroid
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult


@register_centroid("geometric_centroid")
class GeometricCentroid(BaseCentroid):
    """Area-weighted centroid via Shapely.
    https://shapely.readthedocs.io/en/2.1.1/reference/shapely.centroid.html

    For Shapely regions: uses .centroid directly (area-weighted center of mass).
    For vertex lists: builds a Shapely polygon first, then uses .centroid.
    """

    name = "geometric_centroid"

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        if not result.success:
            return None

        # Shapely region path (from planar multilateration)
        if result.region is not None:
            c = result.region.centroid
            return (c.y, c.x)  # Shapely: y=lat, x=lon

        # Vertex list path (from spherical_circle multilateration)
        if result.vertices is not None and len(result.vertices) >= 3:
            # Convert (lat, lon) → Shapely (lon, lat)
            coords = [(lon, lat) for lat, lon in result.vertices]
            try:
                poly = ShapelyPolygon(coords)
                if poly.is_valid and not poly.is_empty:
                    c = poly.centroid
                    return (c.y, c.x)
            except Exception:
                pass

        # Fallback for < 3 vertices
        if result.vertices:
            if len(result.vertices) == 2:
                from scripts.utils.helpers import get_middle_intersection

                return get_middle_intersection(result.vertices)
            if len(result.vertices) == 1:
                return result.vertices[0]

        return None
