"""Snapped geometric median centroid (v2 port).

Samples points inside the feasible region, computes the continuous geometric
median via Weiszfeld iterations, then snaps to the nearest sampled feasible
point.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry.base import BaseGeometry

from scripts.framework.geometry import (
    continuous_geometric_median,
    nearest_sample_point,
    sample_points_in_region,
)
from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.registry import register_ctr
from scripts.framework.v2.types import Coord, Error


@register_ctr("geometric_median")
class GeometricMedianCTR(CTRMethod):
    """Continuous geometric median snapped to a sampled feasible point."""

    def __init__(self, n_samples: int = 1000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        if not mtl.success:
            return CTRResult(success=False, error=Error.EMPTY_REGION)

        intersection = mtl.intersection

        if isinstance(intersection, BaseGeometry):
            if intersection.is_empty:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            points = sample_points_in_region(
                intersection, n_samples=self.n_samples, rng=self.rng,
            )
            if len(points) >= 2:
                median = continuous_geometric_median(points)
                lat, lon = nearest_sample_point(points, median)
                return CTRResult(success=True, tg_coord=Coord(lat, lon))
            if len(points) == 1:
                return CTRResult(
                    success=True,
                    tg_coord=Coord(float(points[0, 0]), float(points[0, 1])),
                )
            # Region too small for sampling; keep point feasible.
            point = intersection.representative_point()
            return CTRResult(success=True, tg_coord=Coord(point.y, point.x))

        if isinstance(intersection, list):
            n = len(intersection)
            if n == 0:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            verts = [(c.lat, c.lon) for c in intersection]
            if n == 1:
                return CTRResult(success=True, tg_coord=Coord(verts[0][0], verts[0][1]))
            points = np.array(verts, dtype=float)
            median = continuous_geometric_median(points)
            lat, lon = nearest_sample_point(points, median)
            return CTRResult(success=True, tg_coord=Coord(lat, lon))

        return CTRResult(success=False, error=Error.EMPTY_REGION)
