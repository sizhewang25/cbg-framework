"""Phase 4 variant: Geometric Centroid (area-weighted).

Uses Shapely's .centroid — the area-weighted center of mass of the
intersection polygon. Guaranteed to be inside convex polygons.

Reference: centroid_comparison.py :: compute_shapely_centroid()
"""

from __future__ import annotations

from typing import Optional, Tuple

from scripts.framework.centroid import BaseCentroid
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult


@register_centroid("geometric_centroid")
class GeometricCentroid(BaseCentroid):
    """Area-weighted centroid via Shapely.
    https://shapely.readthedocs.io/en/2.1.1/reference/shapely.centroid.html

    Only accepts Shapely region inputs. It intentionally does not convert
    unordered spherical crossing vertices into polygons.
    """

    name = "geometric_centroid"

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        if not result.success:
            return None

        # Shapely region path (from planar multilateration)
        if result.region is not None:
            c = result.region.centroid
            return (c.y, c.x)  # Shapely: y=lat, x=lon

        return None
