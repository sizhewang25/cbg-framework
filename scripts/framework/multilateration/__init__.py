"""Phase 3: Multilateration (Region Formation).

Base class for all multilateration variants.
Each variant intersects constraints to form a feasible region.
"""

from __future__ import annotations

from typing import List

from scripts.framework.types import CircleConstraint, MultilatResult


class BaseMultilateration:
    """Abstract base for multilateration / region formation."""

    name: str = "base"

    def multilaterate(self, circles: List[CircleConstraint]) -> MultilatResult:
        """Intersect circle constraints to form a feasible region.

        Args:
            circles: Filtered list of CircleConstraint from Phase 2.

        Returns:
            MultilatResult with either vertices or Shapely region set.
        """
        raise NotImplementedError
