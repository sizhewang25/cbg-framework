"""LowEnvelopeLTD — per-anchor LP best-line RTT-to-distance model (Vanilla CBG).

radius_km = (rtt - intercept) / slope, fit per VP. The constructor takes only
hyperparameters; the per-VP RTTDistanceModel submodels are built inside `_fit`
from the FitSamples (distances computed via haversine over the sample coords).
Upstream callers never instantiate library-level submodel objects.

Wraps scripts/libs/cbg_feasibility/rtt_model.py :: RTTDistanceModel.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from scripts.framework.v2.ltd.base import (
    CircleLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.cbg_feasibility.rtt_model import RTTDistanceModel, haversine_distance


@register_ltd("low_envelope")
class LowEnvelopeLTD(CircleLTDModel):
    """Per-anchor LP best-line RTT-to-distance model."""

    def __init__(
        self,
        max_rtt_ms: float = float("inf"),
        bin_size_km: float = 100.0,
    ) -> None:
        self.max_rtt_ms = max_rtt_ms
        self.bin_size_km = bin_size_km
        self._submodels: Dict[VpId, RTTDistanceModel] = {}

    def _fit(self, samples: list[FitSample]) -> FittingResult:
        if not samples:
            return FittingResult(success=False, error=Error.INSUFFICIENT_DATA)

        by_vp: Dict[VpId, Dict] = defaultdict(
            lambda: {"rtts": [], "distances": [], "vp_coord": None}
        )
        for s in samples:
            d = haversine_distance(
                s.vp_coord.lat, s.vp_coord.lon, s.probe_coord.lat, s.probe_coord.lon
            )
            bucket = by_vp[s.vp_id]
            bucket["rtts"].append(float(s.latency))
            bucket["distances"].append(d)
            bucket["vp_coord"] = s.vp_coord

        new_submodels: Dict[VpId, RTTDistanceModel] = {}
        for vp_id, data in by_vp.items():
            vp_coord: Coord = data["vp_coord"]
            model = RTTDistanceModel(
                anchor_ip=str(vp_id),
                anchor_lat=vp_coord.lat,
                anchor_lon=vp_coord.lon,
            )
            try:
                model.fit(
                    distances=np.array(data["distances"], dtype=float),
                    rtts=np.array(data["rtts"], dtype=float),
                    method="lp",
                    bin_size_km=self.bin_size_km,
                )
            except Exception:
                pass  # .fitted reflects failure; we still store the model
            new_submodels[vp_id] = model

        self._submodels = new_submodels
        fitted_vps = [vp for vp, m in new_submodels.items() if m.fitted]
        return FittingResult(
            success=True,
            args={"vps_fitted": fitted_vps, "vps_attempted": list(new_submodels)},
        )

    def _predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult:
        submodel = self._submodels.get(vp_id)
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
