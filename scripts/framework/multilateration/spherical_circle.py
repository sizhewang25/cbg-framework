"""Phase 2 variant: `spherical_circle` (Million-Scale CBG).

Exact spherical geometry: computes pairwise great-circle crossing points
and filters to points inside all circles. Returns a vertex list.

Uses the framework-owned corrected copy of the Million-Scale helper. Unlike the
legacy helper, redundant-circle preprocessing is optional so the framework's
filtering phase controls whether it happens.
"""

from __future__ import annotations

from typing import List

from scripts.framework.multilateration import BaseMultilateration
from scripts.framework.registry import register_multilateration
from scripts.framework.types import CircleConstraint, MultilatResult
from scripts.framework.geometry import circle_intersections


@register_multilateration("spherical_circle")
class SphericalCircleMultilateration(BaseMultilateration):
    """`spherical_circle` intersection (IMC 2012 original).

    Computes pairwise great-circle crossing points, then filters to points that
    lie inside all circles. The pipeline's filtering phase owns redundant-circle
    removal; set `preprocess=True` only for legacy-style internal preprocessing.
    """

    name = "spherical_circle"

    def __init__(self, speed_threshold: float = 2 / 3, preprocess: bool = False):
        self.speed_threshold = speed_threshold
        self.preprocess = preprocess

    def multilaterate(self, circles: List[CircleConstraint]) -> MultilatResult:
        if not circles:
            return MultilatResult(success=False)

        legacy_tuples = [c.to_legacy_tuple() for c in circles]
        # circle_intersections returns (filtered_vertex_points, used_circles)
        points, used_set = circle_intersections(
            legacy_tuples,
            speed_threshold=self.speed_threshold,
            preprocess=self.preprocess,
        )

        # Rebuild circles_used from the returned set
        used_keys = {(t[0], t[1], t[2]) for t in used_set}
        circles_used = [
            c
            for c in circles
            if (c.vp_lat, c.vp_lon, c.rtt_ms) in used_keys
        ]

        if not points:
            return MultilatResult(circles_used=circles_used, success=False)

        return MultilatResult(
            vertices=points,
            circles_used=circles_used,
            success=True,
        )
