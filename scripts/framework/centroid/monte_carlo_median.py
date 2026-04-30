"""Phase 4 variant: Monte Carlo sampled medoid (Octant).

Samples points inside the feasible region via Sobol QMC rejection sampling,
then selects the sampled point with minimum total distance to all samples.

More robust than arithmetic mean or area-weighted centroid for irregular
or elongated feasible regions. Unlike a continuous geometric median, the final
point is guaranteed to be one of the sampled feasible points.

Wraps:
  - scripts/analysis/octant/octant_geolocation.py :: sample_points_in_region()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from scripts.framework.centroid import BaseCentroid
from scripts.framework.geometry import sample_points_in_region, sampled_medoid
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult


@register_centroid("monte_carlo_median")
class MonteCarloMedianCentroid(BaseCentroid):
    """Octant-faithful sampled medoid via Monte Carlo sampling.

    For Shapely regions: samples n_samples points via Sobol QMC,
    then selects the sampled point minimizing total distance to all samples.

    For vertex lists: selects the medoid of the vertices directly
    (no sampling needed; vertices are the point set).
    """

    name = "monte_carlo_median"

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
            if len(points) >= 1:
                return sampled_medoid(points)
            # Fallback: region too small for sampling; keep point feasible.
            point = result.region.representative_point()
            return (point.y, point.x)

        # Vertex list path (from spherical_circle multilateration)
        if result.vertices is not None and len(result.vertices) >= 1:
            points = np.array(result.vertices, dtype=float)
            return sampled_medoid(points)

        return None
