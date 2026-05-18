"""Speed-of-Internet LTD — theoretical 2/3c model (IMC 2012).

radius_km = rtt_to_km(rtt, speed_threshold). At the default speed_threshold = 2/3,
this is 100 * rtt_ms. Stateless: `_fit` ignores `samples` and always succeeds.

Wraps scripts/framework/geometry.rtt_to_km.
"""

from __future__ import annotations

from scripts.framework.geometry import rtt_to_km
from scripts.framework.v2.ltd.base import (
    CircleLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId


@register_ltd("speed_of_internet")
class SpeedOfInternetLTD(CircleLTDModel):
    """Theoretical speed-of-Internet RTT-to-distance model."""

    def __init__(
        self,
        speed_threshold: float = 2 / 3,
        max_rtt_ms: float = float("inf"),
    ) -> None:
        self.speed_threshold = speed_threshold
        self.max_rtt_ms = max_rtt_ms

    def _fit(self, samples: list[FitSample]) -> FittingResult:
        return FittingResult(success=True)

    def _predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult:
        if latency > self.max_rtt_ms:
            return LTDResult(
                success=False,
                error=Error.RTT_OUT_OF_RANGE,
                vp_id=vp_id,
                vp_coord=vp_coord,
            )
        radius_km = float(rtt_to_km(latency, speed_threshold=self.speed_threshold))
        return LTDResult(
            success=True,
            vp_id=vp_id,
            vp_coord=vp_coord,
            tg_distance=Distance(upper_km=radius_km),
        )
