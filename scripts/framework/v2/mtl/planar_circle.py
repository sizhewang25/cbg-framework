"""PlanarCircleMTL — Shapely polygon-intersection multilateration.

Core idea: approximate each RTT disk as a polygon in degree space, then take the Shapely intersection of all of them. Disk-only

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

from scripts.framework.geometry import filter_redundant_outer_disks
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
    """
    It's a planar approximation: fine at equatorial latitudes and short distances, 
    increasingly wrong as you move poleward or as radii get large.
    """
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
    """Planar polygon intersection of disk constraints.

    `enable_circle_filter` runs the redundant-disk preprocessing from
    `filter_redundant_outer_disks` before polygonization. For pure disk
    intersection this is a no-op on the result (A ⊇ B ⇒ A ∩ B = B), purely
    a speedup. Default on for consistency with the annular MTLs.
    """

    def __init__(self, n_pts: int = 64, enable_circle_filter: bool = True) -> None:
        self.n_pts = n_pts
        self.enable_circle_filter = enable_circle_filter

    def _multilaterate(self, results: list[LTDResult]) -> MTLResult:
        if not results:
            return MTLResult(success=False, error=Error.INSUFFICIENT_DATA)

        if self.enable_circle_filter:
            centers = [(r.vp_coord.lat, r.vp_coord.lon) for r in results]
            radii = [r.tg_distance.upper_km for r in results]
            keep = filter_redundant_outer_disks(centers, radii)
            results = [results[k] for k in keep]

        # VPs that survived the filter and decide the region (recording only).
        participating = tuple(r.vp_id for r in results)

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
            return MTLResult(success=False, error=Error.EMPTY_REGION,
                             participating_vp_ids=participating)

        # Chained pairwise intersection
        # Output is a Shapely geometry
        # Since intersection is associative and commutative, 
        # the final geometry is the set of points contained in every polygon
        # - the common intersection of all disks.
        # If any pairwise step produces an empty geometry, all subsequent intersections stay empty
        region = reduce(lambda a, b: a.intersection(b), polys)

        if region.is_empty:
            return MTLResult(success=False, error=Error.EMPTY_REGION,
                             participating_vp_ids=participating)
        if region.geom_type not in ("Polygon", "MultiPolygon"):
            return MTLResult(success=False, error=Error.DEGENERATE_REGION,
                             participating_vp_ids=participating)

        return MTLResult(success=True, intersection=region,
                         participating_vp_ids=participating)
