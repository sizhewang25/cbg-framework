"""Phase 2 variant: No filtering (passthrough)."""

from __future__ import annotations

from typing import List

from scripts.framework.filtering import BaseFilter
from scripts.framework.registry import register_filtering
from scripts.framework.types import CircleConstraint


@register_filtering("none")
class NoFilter(BaseFilter):
    """Passthrough — returns all circles unmodified."""

    name = "none"

    def filter(self, circles: List[CircleConstraint]) -> List[CircleConstraint]:
        return list(circles)
