"""Speed-of-Internet LTD — theoretical 2/3c model with optional empirical calibration.

radius_km = rtt_to_km(rtt, speed_threshold) = 150 · speed_threshold · rtt_ms.
At the theoretical default speed_threshold = 2/3, this is 100 · rtt_ms.

`_fit(samples)` calibrates `speed_threshold` from observed (latency, true distance)
pairs: for each sample, the smallest threshold that contains true_distance within
the predicted radius is `d / (150 · rtt)`. The fitted threshold is the
`target_coverage`-th quantile of those per-sample ratios, i.e. the smallest value
that upper-bounds `target_coverage` of the observed distances. With no samples
the constructor value is left untouched (theoretical mode).

Wraps scripts/framework/geometry.rtt_to_km +
scripts/libs/cbg_feasibility/rtt_model.haversine_distance.
"""

from __future__ import annotations

from typing import List

import numpy as np

from scripts.framework.geometry import rtt_to_km
from scripts.framework.v2.ltd.base import (
    CircleLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.cbg_feasibility.rtt_model import haversine_distance

# rtt_to_km(rtt, st) = st * rtt * c / 2 with c = 300 km/ms → 150 * st * rtt_ms.
_RADIUS_PER_RTT_PER_THRESHOLD = 150.0


@register_ltd("speed_of_internet")
class SpeedOfInternetLTD(CircleLTDModel):
    """Speed-of-Internet RTT-to-distance model.

    Theoretical mode (default): radius = 100 * rtt at speed_threshold = 2/3.
    Empirical mode: call fit(samples); speed_threshold gets recalibrated so
    that `target_coverage` of (rtt, distance) sample pairs land inside the
    predicted radius.
    """

    def __init__(
        self,
        speed_threshold: float = 2 / 3,
        max_rtt_ms: float = float("inf"),
        target_coverage: float = 0.95,
    ) -> None:
        self.speed_threshold = speed_threshold
        self.max_rtt_ms = max_rtt_ms
        self.target_coverage = target_coverage

    def _fit(self, samples: List[FitSample]) -> FittingResult:
        if not samples:
            # No data — keep the theoretical default; predict still works.
            return FittingResult(
                success=True,
                args={
                    "calibrated": False,
                    "speed_threshold": self.speed_threshold,
                    "samples_used": 0,
                },
            )

        ratios: List[float] = []
        for s in samples:
            rtt = float(s.latency)
            if rtt <= 0:
                continue
            d_km = haversine_distance(
                s.vp_coord.lat,
                s.vp_coord.lon,
                s.probe_coord.lat,
                s.probe_coord.lon,
            )
            ratios.append(d_km / (_RADIUS_PER_RTT_PER_THRESHOLD * rtt))

        if not ratios:
            return FittingResult(
                success=False,
                error=Error.INSUFFICIENT_DATA,
                args={"reason": "no samples with rtt > 0"},
            )

        calibrated = float(np.quantile(np.asarray(ratios), self.target_coverage))
        self.speed_threshold = calibrated
        return FittingResult(
            success=True,
            args={
                "calibrated": True,
                "speed_threshold": calibrated,
                "samples_used": len(ratios),
                "target_coverage": self.target_coverage,
            },
        )

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
                latency=latency,
            )
        radius_km = float(rtt_to_km(latency, speed_threshold=self.speed_threshold))
        return LTDResult(
            success=True,
            vp_id=vp_id,
            vp_coord=vp_coord,
            latency=latency,
            tg_distance=Distance(upper_km=radius_km),
        )
