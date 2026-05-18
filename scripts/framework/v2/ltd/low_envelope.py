"""LowEnvelopeLTD — per-anchor LP best-line RTT-to-distance model (Vanilla CBG).

radius_km = (rtt - intercept) / slope, fit per-VP. Constructor accepts pre-fitted
RTTDistanceModel instances keyed by VpId. `_fit(samples)` ignores samples in
this migration step (real fit-from-samples is a follow-up); it succeeds if any
pre-fitted state is present and reports INSUFFICIENT_DATA otherwise.

Wraps scripts/libs/cbg_feasibility/rtt_model.py :: RTTDistanceModel.
"""

from __future__ import annotations

from typing import Optional

from scripts.framework.v2.ltd.base import (
    CircleLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.cbg_feasibility.rtt_model import RTTDistanceModel


@register_ltd("low_envelope")
class LowEnvelopeLTD(CircleLTDModel):
    """Per-anchor LP best-line RTT-to-distance model."""

    def __init__(
        self,
        models: Optional[dict[VpId, RTTDistanceModel]] = None,
        max_rtt_ms: float = float("inf"),
    ) -> None:
        self.models: dict[VpId, RTTDistanceModel] = dict(models) if models else {}
        self.max_rtt_ms = max_rtt_ms

    def _fit(self, samples: list[FitSample]) -> FittingResult:
        if self.models:
            return FittingResult(success=True, args={"models": self.models})
        return FittingResult(success=False, error=Error.INSUFFICIENT_DATA)

    def _predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult:
        submodel = self.models.get(vp_id)
        if submodel is None or not submodel.fitted:
            return LTDResult(
                success=False,
                error=Error.VP_NOT_FITTED,
                vp_id=vp_id,
                vp_coord=vp_coord,
                latency=latency,
            )
        if latency > self.max_rtt_ms:
            return LTDResult(
                success=False,
                error=Error.RTT_OUT_OF_RANGE,
                vp_id=vp_id,
                vp_coord=vp_coord,
                latency=latency,
            )
        radius_km = submodel.predict_distance(latency)
        if radius_km is None or radius_km <= 0:
            return LTDResult(
                success=False,
                error=Error.NUMERICAL_FAILURE,
                vp_id=vp_id,
                vp_coord=vp_coord,
                latency=latency,
            )
        return LTDResult(
            success=True,
            vp_id=vp_id,
            vp_coord=vp_coord,
            latency=latency,
            tg_distance=Distance(upper_km=float(radius_km)),
        )
