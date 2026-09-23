"""Monte Carlo sampled medoid CTR (v2 port).

Octant-faithful: samples points inside the feasible region via Sobol QMC
rejection, then selects the sampled point minimizing total distance to all
samples (the discrete L1-medoid).
"""

from __future__ import annotations

import numpy as np
from shapely.geometry.base import BaseGeometry

from scripts.framework.geometry import sample_points_in_region, sampled_medoid
from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.registry import register_ctr
from scripts.framework.v2.types import Coord, Error


@register_ctr("monte_carlo_medoid")
class MonteCarloMedoidCTR(CTRMethod):
    """Octant-faithful sampled medoid via Monte Carlo sampling.

    For Shapely regions: samples n_samples points via Sobol QMC, then selects
    the sampled point minimizing total distance to all samples.

    For vertex lists: selects the medoid of the vertices directly (no sampling
    needed; vertices are the point set).
    """

    #: Default seed, so a run is reproducible whether or not anything upstream
    #: seeds it. `sample_points_in_region` builds its Sobol sampler with
    #: `scramble=True` and draws the scramble seed from this generator, so an
    #: unseeded default made the sampled medoid move run to run -- measured at
    #: ~2 km of jitter on a 1-degree region at n_samples=1024. That is small
    #: against an H3-4 cell, which is why it was invisible in the accuracy
    #: tables, but it made every error_km figure irreproducible at the metre
    #: precision those CSVs are written to.
    #:
    #: `benchmark/v2/runner.py` REPLACES `self.rng` per target when the run
    #: carries a `base_seed`, deriving each from `SeedSequence([base_seed,
    #: target_index])`. That still wins and is the stronger guarantee -- it
    #: survives a change in target order, which a single generator advancing
    #: across targets does not. This default is the floor for the case where
    #: nothing sets one: a config with no `seed:` key, or a direct caller.
    #:
    #: Matches `GeometricMedianCTR`, the other sampling CTR, which has defaulted
    #: to 42 all along. Pass `seed=None` explicitly to opt back into OS entropy
    #: -- the only reason to want that is measuring the sampler's own variance.
    DEFAULT_SEED = 42

    def __init__(
        self, n_samples: int = 1000, seed: int | None = DEFAULT_SEED
    ) -> None:
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
            if len(points) >= 1:
                lat, lon = sampled_medoid(points)
                return CTRResult(success=True, tg_coord=Coord(lat, lon))
            # Region too small for sampling; keep point feasible.
            point = intersection.representative_point()
            return CTRResult(success=True, tg_coord=Coord(point.y, point.x))

        if isinstance(intersection, list):
            if len(intersection) == 0:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            verts = [(c.lat, c.lon) for c in intersection]
            points = np.array(verts, dtype=float)
            lat, lon = sampled_medoid(points)
            return CTRResult(success=True, tg_coord=Coord(lat, lon))

        return CTRResult(success=False, error=Error.EMPTY_REGION)
