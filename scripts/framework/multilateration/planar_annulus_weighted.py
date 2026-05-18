"""Phase 2 variant: `planar_annulus_weighted` (Octant, fused Phase 2+3).

Grid-based weight accumulation over annular constraints. Points with
cumulative weight exceeding a threshold form the feasible region.

Wraps: scripts/libs/octant/octant_geolocation.py :: compute_feasible_region_weighted()
Only compatible with bounded_spline distance (requires inner_radius_km > 0).
"""

from __future__ import annotations

from typing import List

from scripts.framework.multilateration import BaseMultilateration
from scripts.framework.registry import register_multilateration
from scripts.framework.types import CircleConstraint, MultilatResult
from scripts.libs.octant.octant_geolocation import (
    AnnularConstraint,
    compute_feasible_region_weighted,
)


@register_multilateration("planar_annulus_weighted")
class PlanarAnnulusWeightedMultilateration(BaseMultilateration):
    """Octant weighted planar-annulus grid multilateration.

    For each grid point, accumulates weights from constraints whose annulus
    contains the point. Points above threshold × max_weight form the region.

    Note: This method has built-in filtering. Phase 2 should typically be 'none'.
    """

    name = "planar_annulus_weighted"

    def __init__(
        self,
        weight_threshold: float = 0.5,
        grid_resolution_deg: float = 0.25,
    ):
        self.weight_threshold = weight_threshold
        self.grid_resolution_deg = grid_resolution_deg

    def multilaterate(self, circles: List[CircleConstraint]) -> MultilatResult:
        if not circles:
            return MultilatResult(success=False)

        # Convert CircleConstraint → AnnularConstraint for octant API
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

        region = compute_feasible_region_weighted(
            annular,
            weight_threshold=self.weight_threshold,
            grid_resolution_deg=self.grid_resolution_deg,
        )

        if region is None:
            return MultilatResult(circles_used=circles, success=False)

        return MultilatResult(
            region=region,
            circles_used=circles,
            success=True,
        )
