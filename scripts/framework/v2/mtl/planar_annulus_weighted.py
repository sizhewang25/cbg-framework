"""PlanarAnnulusWeightedMTL — Octant weighted face decomposition.

Each constraint contributes `exp(-rtt_ms / weight_tau_ms)` to faces of the
planar arrangement of all annulus boundaries. The top-weighted faces are
unioned until their cumulative weight clears `weight_threshold * Σwᵢ`,
yielding the feasible region.

`highest_weight_only=True` bypasses the cumulative-weight aggregation and
returns just the single top-weighted face — deterministic, always Polygon,
and avoids the MultiPolygon-discards-weights failure mode where downstream
area-biased sampling drifts off the densest overlap. `weight_threshold`
has no effect in that branch.

The original RTT is required for the exponential weight. v2's `LTDResult`
does not yet carry latency; this wrapper reads `getattr(r, "latency", None)`
so it stays forward-compatible. Once `LTDResult.latency` lands, the happy
path works without further code changes. Until then, the wrapper short-
circuits with `Error.INSUFFICIENT_DATA` whenever any result lacks latency.

Wraps scripts/libs/octant/octant_geolocation.compute_feasible_region_weighted.
"""

from __future__ import annotations

import math

from scripts.framework.geometry import filter_redundant_outer_disks
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import AnnulusMTLMethod, MTLResult
from scripts.framework.v2.mtl._annulus_common import (
    annular_constraint_from_ltd,
    wrap_region_as_mtl_result,
)
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Error
from scripts.libs.octant.octant_geolocation import (
    compute_feasible_region_weighted,
)


@register_mtl("planar_annulus_weighted")
class PlanarAnnulusWeightedMTL(AnnulusMTLMethod):
    """Octant weighted feasible region via planar face decomposition.

    `enable_circle_filter` drops constraints whose outer disk fully contains
    another's outer disk before the weighted face decomposition. Same heuristic
    as PlanarAnnulusMTL; the cumulative weight Σwᵢ used by the threshold is
    recomputed over the kept constraints.
    """

    def __init__(
        self,
        weight_threshold: float = 0.5,
        weight_tau_ms: float = 50.0,
        n_pts: int = 64,
        enable_circle_filter: bool = True,
        highest_weight_only: bool = False,
    ) -> None:
        self.weight_threshold = weight_threshold
        self.weight_tau_ms = weight_tau_ms
        self.n_pts = n_pts
        self.enable_circle_filter = enable_circle_filter
        self.highest_weight_only = highest_weight_only

    def _multilaterate(self, results: list[LTDResult]) -> MTLResult:
        if not results:
            return MTLResult(success=False, error=Error.INSUFFICIENT_DATA)

        if self.enable_circle_filter:
            centers = [(r.vp_coord.lat, r.vp_coord.lon) for r in results]
            radii = [r.tg_distance.upper_km for r in results]
            keep = filter_redundant_outer_disks(centers, radii)
            results = [results[k] for k in keep]

        constraints = []
        for r in results:
            rtt_ms = getattr(r, "latency", None)
            if rtt_ms is None:
                return MTLResult(success=False, error=Error.INSUFFICIENT_DATA)
            constraints.append(
                annular_constraint_from_ltd(
                    r,
                    rtt_ms=float(rtt_ms),
                    weight=math.exp(-float(rtt_ms) / self.weight_tau_ms),
                )
            )

        region = compute_feasible_region_weighted(
            constraints,
            weight_threshold=self.weight_threshold,
            n_pts=self.n_pts,
            highest_weight_only=self.highest_weight_only,
        )
        return wrap_region_as_mtl_result(region)
