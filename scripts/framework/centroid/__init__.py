"""Phase 4: Centroid Selection.

Base class for all centroid variants.
Each variant collapses a feasible region into a single (lat, lon) estimate.
"""

from __future__ import annotations

from typing import Optional, Tuple

from scripts.framework.types import MultilatResult


class BaseCentroid:
    """Abstract base for single-point estimation from a region."""

    name: str = "base"

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        """Select a representative point from the multilateration result.

        Args:
            result: MultilatResult from Phase 3 (vertices or Shapely region).

        Returns:
            (lat, lon) tuple or None if no valid point can be selected.
        """
        raise NotImplementedError
