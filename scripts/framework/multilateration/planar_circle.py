"""Phase 2 variant: `planar_circle`.

Approximates circles as 100-point polygons in `(lon, lat)` degree space and
computes their intersection via Shapely. Returns a Shapely geometry.

Pattern from: evaluate_million_scale.py:58 and octant_geolocation.py:150
"""

from __future__ import annotations

import math
from functools import reduce
from typing import List

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from scripts.framework.multilateration import BaseMultilateration
from scripts.framework.registry import register_multilateration
from scripts.framework.types import CircleConstraint, MultilatResult


def _circle_to_planar_polygon(
    clat: float,
    clon: float,
    radius_km: float,
    n_pts: int = 100,
):
    """Convert a geographic circle to a Shapely polygon in (lon, lat) space.

    Uses degree-based approximation accounting for latitude compression.
    Consistent with octant_geolocation.py:150 (endpoint=False).
    """
    km_per_deg_lat = 111.0
    km_per_deg_lon = max(111.0 * math.cos(math.radians(clat)), 1.0)
    r_lat = radius_km / km_per_deg_lat
    r_lon = radius_km / km_per_deg_lon
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    lons = clon + r_lon * np.cos(angles)
    lats = clat + r_lat * np.sin(angles)
    return ShapelyPolygon(zip(lons, lats))  # Shapely: (x, y) = (lon, lat)


@register_multilateration("planar_circle")
class PlanarCircleMultilateration(BaseMultilateration):
    """Planar polygon intersection of RTT constraint circles.

    Each circle is approximated as a Shapely polygon with n_pts vertices in
    raw `(lon, lat)` degree space. The intersection of all polygons forms the
    feasible region.
    """

    name = "planar_circle"

    def __init__(self, n_pts: int = 100):
        self.n_pts = n_pts

    def multilaterate(self, circles: List[CircleConstraint]) -> MultilatResult:
        if not circles:
            return MultilatResult(success=False)

        polys = []
        for c in circles:
            p = _circle_to_planar_polygon(
                c.vp_lat,
                c.vp_lon,
                c.radius_km,
                self.n_pts,
            )
            if p.is_valid and not p.is_empty:
                polys.append(p)

        if not polys:
            return MultilatResult(circles_used=circles, success=False)

        region = reduce(lambda a, b: a.intersection(b), polys)

        if region.is_empty:
            return MultilatResult(circles_used=circles, success=False)

        # Reject degenerate geometries (lines, points)
        if region.geom_type not in ("Polygon", "MultiPolygon"):
            return MultilatResult(circles_used=circles, success=False)

        return MultilatResult(
            region=region,
            circles_used=circles,
            success=True,
        )
