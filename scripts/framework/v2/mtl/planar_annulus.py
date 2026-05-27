"""PlanarAnnulusMTL — Octant unweighted annular intersection.

For each constraint, intersects all outer disks then subtracts the union of
all inner disks. With disk-only inputs (`lower_km == 0`) this degenerates to
the same result as PlanarCircleMTL, but the family base is AnnulusMTLMethod
because the value of this method comes from honouring `lower_km > 0`.

Wraps scripts/libs/octant/octant_geolocation.compute_feasible_region_unweighted.
The wrapped function reads only `landmark_lat`, `landmark_lon`,
`inner_radius_km`, and `outer_radius_km`; `rtt_ms` and `weight` are unused —
we pass placeholders.
"""

from __future__ import annotations

from scripts.framework.geometry import filter_redundant_outer_disks
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import AnnulusMTLMethod, MTLResult
from scripts.framework.v2.mtl._annulus_common import (
    annular_constraint_from_ltd,
    wrap_region_as_mtl_result,
)
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Error
from scripts.libs.octant.octant_geolocation import (
    compute_feasible_region_unweighted,
)


@register_mtl("planar_annulus")
class PlanarAnnulusMTL(AnnulusMTLMethod):
    """Octant unweighted feasible-region intersection.

    `enable_circle_filter` drops constraints whose *outer* disk fully contains
    another's outer disk before the annular intersection. Heuristic: the
    engulfing constraint is non-binding on the outer-disk intersection and is
    typically a wide-RTT VP whose inner disk is the most likely to falsely
    exclude the truth on dense local fleets (see AS7018 NA collapse). The
    smallest outer disk in any chain is always kept.
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

        constraints = [
            annular_constraint_from_ltd(r, rtt_ms=0.0, weight=1.0)
            for r in results
        ]

        region = compute_feasible_region_unweighted(constraints, n_pts=self.n_pts)
        return wrap_region_as_mtl_result(region)
