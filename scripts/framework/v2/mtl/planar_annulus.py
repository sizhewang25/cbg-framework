"""PlanarAnnulusMTL — Octant unweighted annular intersection.

For each constraint, intersects all outer disks then subtracts the union of
all inner disks. With disk-only inputs (`lower_km == 0`) this degenerates to
the same result as PlanarCircleMTL, but the family base is AnnulusMTLMethod
because the value of this method comes from honouring `lower_km > 0`.

Wraps scripts/libs/octant_simple/octant_geolocation.compute_feasible_region_unweighted.
The wrapped function reads only `landmark_lat`, `landmark_lon`,
`inner_radius_km`, and `outer_radius_km`; `rtt_ms` and `weight` are unused —
we pass placeholders.
"""

from __future__ import annotations

from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import AnnulusMTLMethod, MTLResult
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Error
from scripts.libs.octant_simple.octant_geolocation import (
    AnnularConstraint,
    compute_feasible_region_unweighted,
)


@register_mtl("planar_annulus")
class PlanarAnnulusMTL(AnnulusMTLMethod):
    """Octant unweighted feasible-region intersection."""

    def __init__(self, n_pts: int = 64) -> None:
        self.n_pts = n_pts

    def _multilaterate(self, results: list[LTDResult]) -> MTLResult:
        if not results:
            return MTLResult(success=False, error=Error.INSUFFICIENT_DATA)

        constraints = [
            AnnularConstraint(
                landmark_lat=r.vp_coord.lat,
                landmark_lon=r.vp_coord.lon,
                landmark_ip=str(r.vp_id) if r.vp_id is not None else "",
                rtt_ms=0.0,
                inner_radius_km=r.tg_distance.lower_km,
                outer_radius_km=r.tg_distance.upper_km,
                weight=1.0,
            )
            for r in results
        ]

        region = compute_feasible_region_unweighted(constraints, n_pts=self.n_pts)

        if region is None or region.is_empty:
            return MTLResult(success=False, error=Error.EMPTY_REGION)
        if region.geom_type not in ("Polygon", "MultiPolygon"):
            return MTLResult(success=False, error=Error.DEGENERATE_REGION)

        return MTLResult(success=True, intersection=region)
