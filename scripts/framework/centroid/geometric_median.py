"""Phase 4 variant: snapped geometric median.

Samples points inside the feasible region, computes the continuous geometric
median with framework-owned Weiszfeld iterations, then snaps to the nearest
sampled feasible point.
This keeps the final estimate inside the sampled region while preserving most
of the speed advantage of continuous geometric-median approximation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from scripts.framework.centroid import BaseCentroid
from scripts.framework.geometry import (
    continuous_geometric_median,
    nearest_sample_point,
    sample_points_in_region,
)
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult


@register_centroid("geometric_median")
class GeometricMedianCentroid(BaseCentroid):
    """Continuous geometric median snapped to a sampled feasible point."""

    name = "geometric_median"

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        if not result.success:
            return None

        # Shapely region path (from planar multilateration)
        if result.region is not None:
            points = sample_points_in_region(
                result.region, n_samples=self.n_samples, rng=self.rng,
            )
            if len(points) >= 2:
                median = continuous_geometric_median(points)
                return nearest_sample_point(points, median)
            if len(points) == 1:
                return (float(points[0, 0]), float(points[0, 1]))
            # Fallback: region too small for sampling; keep point feasible.
            point = result.region.representative_point()
            return (point.y, point.x)

        # Vertex list path (from spherical_circle multilateration)
        if result.vertices is not None and len(result.vertices) >= 2:
            points = np.array(result.vertices, dtype=float)
            median = continuous_geometric_median(points)
            return nearest_sample_point(points, median)
        if result.vertices is not None and len(result.vertices) == 1:
            return result.vertices[0]

        return None
