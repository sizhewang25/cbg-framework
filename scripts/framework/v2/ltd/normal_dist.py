"""NormalDistLTD — pooled-normal RTT-to-distance model (Laki et al. 2011).

A single (mu(d), sigma(d), k) shared across all VPs — the pooled-normal claim.
Maps an RTT to an annular Distance:

    lower_km = max(0, mu(rtt) - k * sigma(rtt))
    upper_km = max(0, mu(rtt) + k * sigma(rtt))

with k calibrated empirically (see notes/2026-05-17-spotter-normality-check.md).
Out-of-range RTTs return Error.RTT_OUT_OF_RANGE — the polynomial mu/sigma fits
are not safe to extrapolate.

Constructor accepts a pre-fitted SpotterRTTModel. `_fit(samples)` ignores
samples in this migration step (real fit-from-samples is a follow-up).

Wraps scripts/libs/spotter/spotter_model.py :: SpotterRTTModel.
"""

from __future__ import annotations

from typing import Optional

from scripts.framework.v2.ltd.base import (
    AnnulusLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.spotter.spotter_model import SpotterRTTModel


@register_ltd("normal_dist")
class NormalDistLTD(AnnulusLTDModel):
    """Pooled-normal RTT-to-distance model with empirical k."""

    def __init__(
        self,
        model: Optional[SpotterRTTModel] = None,
        max_rtt_ms: float = float("inf"),
    ) -> None:
        self.model = model
        self.max_rtt_ms = max_rtt_ms

    def _fit(self, samples: list[FitSample]) -> FittingResult:
        if self.model is not None and self.model.fitted:
            return FittingResult(success=True, args={"model": self.model})
        return FittingResult(success=False, error=Error.INSUFFICIENT_DATA)

    def _predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult:
        if self.model is None or not self.model.fitted:
            return LTDResult(
                success=False,
                error=Error.VP_NOT_FITTED,
                vp_id=vp_id,
                vp_coord=vp_coord,
            )
        if latency > self.max_rtt_ms:
            return LTDResult(
                success=False,
                error=Error.RTT_OUT_OF_RANGE,
                vp_id=vp_id,
                vp_coord=vp_coord,
            )
        bounds = self.model.predict_distance_bounds(latency)
        if bounds is None:
            return LTDResult(
                success=False,
                error=Error.RTT_OUT_OF_RANGE,
                vp_id=vp_id,
                vp_coord=vp_coord,
            )
        inner_km, outer_km = bounds
        if outer_km <= inner_km:
            return LTDResult(
                success=False,
                error=Error.DEGENERATE_REGION,
                vp_id=vp_id,
                vp_coord=vp_coord,
            )
        return LTDResult(
            success=True,
            vp_id=vp_id,
            vp_coord=vp_coord,
            tg_distance=Distance(
                upper_km=float(outer_km), lower_km=float(inner_km)
            ),
        )
