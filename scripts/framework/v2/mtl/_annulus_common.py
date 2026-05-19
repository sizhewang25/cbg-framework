"""Shared boilerplate for AnnulusMTLMethod wrappers.

Both PlanarAnnulusMTL and PlanarAnnulusWeightedMTL convert LTDResult to
octant_simple's AnnularConstraint and wrap the returned Shapely geometry
in an MTLResult with the same error mapping. Those two operations live
here so the wrapper classes can focus on the per-constraint weight
derivation and which feasibility function they call.
"""

from __future__ import annotations

from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.types import Error
from scripts.libs.octant_simple.octant_geolocation import AnnularConstraint


def annular_constraint_from_ltd(
    r: LTDResult,
    *,
    rtt_ms: float,
    weight: float,
) -> AnnularConstraint:
    """Translate an LTDResult into an octant_simple AnnularConstraint.

    `rtt_ms` and `weight` are kwargs-only because their values are
    wrapper-specific: unweighted passes placeholder (0.0, 1.0); weighted
    derives both from the LTD result's latency.
    """
    return AnnularConstraint(
        landmark_lat=r.vp_coord.lat,
        landmark_lon=r.vp_coord.lon,
        landmark_ip=str(r.vp_id) if r.vp_id is not None else "",
        rtt_ms=rtt_ms,
        inner_radius_km=r.tg_distance.lower_km,
        outer_radius_km=r.tg_distance.upper_km,
        weight=weight,
    )


def wrap_region_as_mtl_result(region) -> MTLResult:
    """Map a Shapely region (or None) to the appropriate MTLResult.

    None/empty → EMPTY_REGION. Non-(Polygon|MultiPolygon) → DEGENERATE_REGION.
    Otherwise success with `intersection=region`.
    """
    if region is None or region.is_empty:
        return MTLResult(success=False, error=Error.EMPTY_REGION)
    if region.geom_type not in ("Polygon", "MultiPolygon"):
        return MTLResult(success=False, error=Error.DEGENERATE_REGION)
    return MTLResult(success=True, intersection=region)
