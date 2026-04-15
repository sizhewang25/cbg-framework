"""Phase 2 variant: Redundant Circle Removal (Million-Scale CBG).

Pairwise containment check — removes the larger circle when one fully
contains another, keeping the tightest bound.

Wraps: scripts/utils/helpers.py :: circle_preprocessing()
"""

from __future__ import annotations

from typing import List

from scripts.framework.filtering import BaseFilter
from scripts.framework.registry import register_filtering
from scripts.framework.types import CircleConstraint
from scripts.utils.helpers import circle_preprocessing


@register_filtering("redundant_circle")
class RedundantCircleFilter(BaseFilter):
    """Remove circles that fully contain other circles.

    When circle A fully contains circle B (i.e., A's radius > center_distance + B's radius),
    circle A is removed because B provides a tighter constraint.
    """

    name = "redundant_circle"

    def __init__(self, speed_threshold: float = 2 / 3):
        self.speed_threshold = speed_threshold

    def filter(self, circles: List[CircleConstraint]) -> List[CircleConstraint]:
        if len(circles) <= 1:
            return list(circles)

        legacy_tuples = [c.to_legacy_tuple() for c in circles]
        kept_set = circle_preprocessing(
            legacy_tuples, speed_threshold=self.speed_threshold
        )

        # Match back: circle_preprocessing returns set of (lat, lon, rtt, d, r).
        # Use (lat, lon, rtt) as composite key to identify kept constraints.
        kept_keys = {(t[0], t[1], t[2]) for t in kept_set}
        return [
            c
            for c in circles
            if (c.vp_lat, c.vp_lon, c.rtt_ms) in kept_keys
        ]
