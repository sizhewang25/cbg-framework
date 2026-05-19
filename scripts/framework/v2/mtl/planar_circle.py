"""PlanarCircleMTL — Shapely polygon-intersection multilateration.

Approximates each outer disk as an `n_pts`-vertex polygon in (lon, lat) degree
space (accounting for latitude compression) and intersects them via Shapely.
Returns a `Polygon` or `MultiPolygon`. Disk-only — annular `lower_km` is
ignored.

Lifts `_circle_to_planar_polygon` from
scripts/framework/multilateration/planar_circle.py.
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import CircleMTLMethod, MTLResult
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Error


def _circle_to_planar_polygon(
    clat: float,
    clon: float,
    radius_km: float,
    n_pts: int,
) -> ShapelyPolygon:
    km_per_deg_lat = 111.0
    km_per_deg_lon = max(111.0 * math.cos(math.radians(clat)), 1.0)
    r_lat = radius_km / km_per_deg_lat
    r_lon = radius_km / km_per_deg_lon
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    lons = clon + r_lon * np.cos(angles)
    lats = clat + r_lat * np.sin(angles)
    return ShapelyPolygon(zip(lons, lats))


@register_mtl("planar_circle")
class PlanarCircleMTL(CircleMTLMethod):
    """Planar polygon intersection of disk constraints."""

    def __init__(self, n_pts: int = 100) -> None:
        self.n_pts = n_pts

    def _multilaterate(self, results: list[LTDResult]) -> MTLResult:
        if not results:
            return MTLResult(success=False, error=Error.INSUFFICIENT_DATA)

        polys = []
        for r in results:
            p = _circle_to_planar_polygon(
                r.vp_coord.lat,
                r.vp_coord.lon,
                r.tg_distance.upper_km,
                self.n_pts,
            )
            if p.is_valid and not p.is_empty:
                polys.append(p)

        if not polys:
            return MTLResult(success=False, error=Error.EMPTY_REGION)

        region = reduce(lambda a, b: a.intersection(b), polys)

        if region.is_empty:
            return MTLResult(success=False, error=Error.EMPTY_REGION)
        if region.geom_type not in ("Polygon", "MultiPolygon"):
            return MTLResult(success=False, error=Error.DEGENERATE_REGION)

        return MTLResult(success=True, intersection=region)
