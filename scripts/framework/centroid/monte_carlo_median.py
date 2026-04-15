"""Phase 4 variant: Monte Carlo Geometric Median (Octant).

Samples points inside the feasible region via Sobol QMC rejection sampling,
then computes the geometric median (minimizes sum of Euclidean distances).

More robust than arithmetic mean or area-weighted centroid for irregular
or elongated feasible regions.

Wraps:
  - scripts/analysis/octant/octant_geolocation.py :: sample_points_in_region()
  - scripts/analysis/octant/octant_geolocation.py :: geometric_median_approx()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from scripts.framework.centroid import BaseCentroid
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult
from scripts.analysis.octant.octant_geolocation import (
    geometric_median_approx,
    sample_points_in_region,
)


@register_centroid("monte_carlo_median")
class MonteCarloMedianCentroid(BaseCentroid):
    """Geometric median via Monte Carlo sampling.

    For Shapely regions: samples n_samples points via Sobol QMC,
    then computes geometric median using geom-median library.

    For vertex lists: computes geometric median directly on the vertices
    (no sampling needed — vertices are the point set).
    """

    name = "monte_carlo_median"

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        if not result.success:
            return None

        # Shapely region path (from shapely/unweighted_annulus/weighted_grid)
        if result.region is not None:
            points = sample_points_in_region(
                result.region, n_samples=self.n_samples, rng=self.rng,
            )
            if len(points) >= 2:
                return geometric_median_approx(points)
            if len(points) == 1:
                return (float(points[0, 0]), float(points[0, 1]))
            # Fallback: region too small for sampling
            c = result.region.centroid
            return (c.y, c.x)

        # Vertex list path (from spherical multilateration)
        if result.vertices is not None and len(result.vertices) >= 2:
            points = np.array(result.vertices, dtype=float)
            return geometric_median_approx(points)
        if result.vertices is not None and len(result.vertices) == 1:
            return result.vertices[0]

        return None
