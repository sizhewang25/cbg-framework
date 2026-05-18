"""BoundedSplineLTD — per-anchor Octant spline + shared delta band (Octant).

Per-VP OctantRTTModel produces annular bounds (inner, outer) for each RTT.
A shared delta is applied across all VPs; the v1 fit(df_asn=df) path computed
delta from data — preserved here as a constructor kwarg until the deferred
fit-from-samples follow-up.

_predict catches exceptions from OctantRTTModel.predict_distance_bounds (the
common failure is a fitted model with no spline) and maps them to
NUMERICAL_FAILURE — distinct from VP_NOT_FITTED (unfitted) and RTT_OUT_OF_RANGE
(latency cutoff). v1 silently skipped both; v2 surfaces the distinction.

Wraps scripts/libs/octant/octant_model.py :: OctantRTTModel.
"""

from __future__ import annotations

import logging
from typing import Optional

from scripts.framework.v2.ltd.base import (
    AnnulusLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.octant.octant_model import OctantRTTModel

logger = logging.getLogger(__name__)


@register_ltd("bounded_spline")
class BoundedSplineLTD(AnnulusLTDModel):
    """Per-anchor Octant spline RTT-to-distance model with annular bounds."""

    def __init__(
        self,
        models: Optional[dict[VpId, OctantRTTModel]] = None,
        delta: Optional[float] = None,
        max_rtt_ms: float = float("inf"),
    ) -> None:
        self.models: dict[VpId, OctantRTTModel] = dict(models) if models else {}
        self.delta = delta
        self.max_rtt_ms = max_rtt_ms

    def _fit(self, samples: list[FitSample]) -> FittingResult:
        if self.models:
            return FittingResult(
                success=True, args={"models": self.models, "delta": self.delta}
            )
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
            )
        if latency > self.max_rtt_ms:
            return LTDResult(
                success=False,
                error=Error.RTT_OUT_OF_RANGE,
                vp_id=vp_id,
                vp_coord=vp_coord,
            )
        try:
            inner_km, outer_km = submodel.predict_distance_bounds(
                latency, delta=self.delta
            )
        except Exception as exc:
            logger.debug(
                "Octant predict_distance_bounds failed for %s at RTT %.3f ms: %s",
                vp_id,
                latency,
                exc,
            )
            return LTDResult(
                success=False,
                error=Error.NUMERICAL_FAILURE,
                vp_id=vp_id,
                vp_coord=vp_coord,
            )
        inner_km = max(0.0, float(inner_km))
        outer_km = max(0.0, float(outer_km))
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
            tg_distance=Distance(upper_km=outer_km, lower_km=inner_km),
        )
