"""Phase 2 variant: `planar_annulus` (Octant).

Intersects all outer disks, then subtracts all inner disks.
Result = intersection(all outer disks) - union(all inner disks)

Unlike `planar_circle` which treats all constraints as disks (ignores
inner_radius_km), this method uses the full annular constraint when
inner_radius_km > 0.

Wraps: scripts/libs/octant/octant_geolocation.py :: compute_feasible_region_unweighted()
"""

from __future__ import annotations

from typing import List

from scripts.framework.multilateration import BaseMultilateration
from scripts.framework.registry import register_multilateration
from scripts.framework.types import CircleConstraint, MultilatResult
from scripts.libs.octant.octant_geolocation import (
    AnnularConstraint,
    compute_feasible_region_unweighted,
)


@register_multilateration("planar_annulus")
class PlanarAnnulusMultilateration(BaseMultilateration):
    """Octant `planar_annulus` intersection.

    For each constraint, builds an outer disk and an inner disk.
    The feasible region is the intersection of all outer disks minus
    the union of all inner disks.

    Only meaningful with bounded_spline distance (which produces
    annular constraints with inner_radius_km > 0). For disk constraints
    (inner_radius_km = 0), this degenerates to the same result as
    planar_circle multilateration.
    """

    name = "planar_annulus"

    def __init__(self, n_pts: int = 100):
        self.n_pts = n_pts

    def multilaterate(self, circles: List[CircleConstraint]) -> MultilatResult:
        if not circles:
            return MultilatResult(success=False)

        # Convert CircleConstraint → AnnularConstraint.
        annular = [
            AnnularConstraint(
                landmark_lat=c.vp_lat,
                landmark_lon=c.vp_lon,
                landmark_ip=c.vp_ip,
                rtt_ms=c.rtt_ms,
                inner_radius_km=c.inner_radius_km,
                outer_radius_km=c.radius_km,
                weight=c.weight,
            )
            for c in circles
        ]

        region = compute_feasible_region_unweighted(annular, n_pts=self.n_pts)

        if region is None:
            return MultilatResult(circles_used=circles, success=False)

        return MultilatResult(
            region=region,
            circles_used=circles,
            success=True,
        )
