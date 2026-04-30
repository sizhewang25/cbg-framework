"""Optional preprocessing: Constraint Filtering.

Base class for all filtering variants.
Each variant takes a list of CircleConstraints and returns a filtered subset.
"""

from __future__ import annotations

from typing import List

from scripts.framework.types import CircleConstraint


class BaseFilter:
    """Abstract base for constraint filtering."""

    name: str = "base"

    def filter(self, circles: List[CircleConstraint]) -> List[CircleConstraint]:
        """Filter circle constraints, removing erroneous or redundant ones.

        Args:
            circles: List of CircleConstraint from Phase 1.

        Returns:
            Filtered list (same type, potentially fewer items).
        """
        raise NotImplementedError
