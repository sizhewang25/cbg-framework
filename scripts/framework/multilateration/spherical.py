"""Phase 3 variant: Spherical Circle Intersection (Million-Scale CBG).

Exact spherical geometry: computes pairwise great-circle crossing points
and filters to points inside all circles. Returns a vertex list.

Wraps: scripts/utils/helpers.py :: circle_intersections()
Reference: run_million_scale_cbg() in evaluate_million_scale.py:174
"""

from __future__ import annotations

from typing import List

from scripts.framework.multilateration import BaseMultilateration
from scripts.framework.registry import register_multilateration
from scripts.framework.types import CircleConstraint, MultilatResult
from scripts.utils.helpers import circle_intersections


@register_multilateration("spherical")
class SphericalMultilateration(BaseMultilateration):
    """Spherical circle intersection (IMC 2012 original).

    Computes pairwise great-circle crossing points, then filters to
    points that lie inside ALL circles. Internally calls
    circle_preprocessing (idempotent if Phase 2 already filtered).
    """

    name = "spherical"

    def __init__(self, speed_threshold: float = 2 / 3):
        self.speed_threshold = speed_threshold

    def multilaterate(self, circles: List[CircleConstraint]) -> MultilatResult:
        if not circles:
            return MultilatResult(success=False)

        legacy_tuples = [c.to_legacy_tuple() for c in circles]
        # circle_intersections returns (filtered_vertex_points, used_circles_set)
        points, used_set = circle_intersections(
            legacy_tuples, speed_threshold=self.speed_threshold
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
