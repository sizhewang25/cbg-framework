"""Speed-of-Internet LTD — universal theoretical RTT-to-distance model.

The simplest distance estimator: a single ratio applied to a fraction of c.
No per-VP state, no calibration, no cutoffs.

    radius_km = speed_ratio · c · rtt_ms / 2

The /2 converts round-trip latency to one-way distance. At the classic CBG
default `speed_ratio = 2/3`, this is `100 · rtt_ms` km. Throughout this module
`latency` denotes RTT in milliseconds.

Wraps scripts/framework/geometry.rtt_to_km.
"""

from __future__ import annotations

from typing import List

from scripts.framework.geometry import rtt_to_km
from scripts.framework.v2.ltd.base import (
    CircleLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Latency, VpId


@register_ltd("speed_of_internet")
class SpeedOfInternetLTD(CircleLTDModel):
    """Universal theoretical speed-of-Internet RTT-to-distance model.

    The only hyperparameter is `speed_ratio` — the fraction of c at which the
    signal is assumed to propagate. Stateless: `_fit` is a no-op.
    """

    def __init__(self, speed_ratio: float = 2 / 3) -> None:
        self.speed_ratio = speed_ratio

    def _fit(self, samples: List[FitSample]) -> FittingResult:
        return FittingResult(success=True)

    def _predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult:
        # `latency` is the RTT in ms; rtt_to_km handles the round-trip→one-way /2.
        radius_km = float(rtt_to_km(latency, speed_threshold=self.speed_ratio))
        return LTDResult(
            success=True,
            vp_id=vp_id,
            vp_coord=vp_coord,
            latency=latency,
            tg_distance=Distance(upper_km=radius_km),
        )
