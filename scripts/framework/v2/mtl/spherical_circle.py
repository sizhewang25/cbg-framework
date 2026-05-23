"""SphericalCircleMTL — exact spherical multilateration (Million-Scale CBG / IMC 2012).

Computes pairwise great-circle crossing points of the outer disks, then filters
to points that lie inside every disk. Returns vertices as `list[Coord]`.

Annular `tg_distance.lower_km` from the LTD stage is silently ignored — this
is a CircleMTLMethod and only consumes the outer radius. An AnnulusLTDModel
can be paired with it (CBGModel allows the degraded composition), but the
inner-bound information is lost. Use a planar-annulus method to preserve it.

Wraps scripts/framework/geometry.circle_intersections (unchanged from v1).
"""

from __future__ import annotations

from scripts.framework.geometry import EARTH_RADIUS_KM, circle_intersections
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import CircleMTLMethod, MTLResult
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Coord, Error


@register_mtl("spherical_circle")
class SphericalCircleMTL(CircleMTLMethod):
    """`spherical_circle` intersection (IMC 2012 original).

    `preprocess=True` enables the legacy redundant-circle preprocessing inside
    `circle_intersections`. The default is False because v2 expects upstream
    filtering to handle that.
    """

    def __init__(self, speed_ratio: float = 2 / 3, enable_circle_filter: bool = False) -> None:
        self.speed_ratio = speed_ratio
        self.enable_circle_filter = enable_circle_filter

    def _multilaterate(self, results: list[LTDResult]) -> MTLResult:
        if not results:
            return MTLResult(success=False, error=Error.INSUFFICIENT_DATA)

        # circle_intersections wants (lat, lon, rtt_ms, radius_km, radius_rad)
        legacy_tuples = [
            (
                r.vp_coord.lat,
                r.vp_coord.lon,
                r.latency,
                r.tg_distance.upper_km,
                r.tg_distance.upper_km / EARTH_RADIUS_KM,
            )
            for r in results
        ]
        points, _used = circle_intersections(
            legacy_tuples,
            speed_threshold=self.speed_ratio,
            preprocess=self.enable_circle_filter,
        )

        if not points:
            return MTLResult(success=False, error=Error.NO_INTERSECTION)

        vertices = [Coord(lat=lat, lon=lon) for lat, lon in points]
        return MTLResult(success=True, intersection=vertices)
