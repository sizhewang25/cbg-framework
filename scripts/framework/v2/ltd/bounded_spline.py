"""BoundedSplineLTD — per-anchor Octant spline + shared delta band (Octant).

Per-VP OctantRTTModel produces annular bounds (inner, outer) for each RTT;
a shared delta is applied across all VPs to widen the band to a target
coverage. The constructor takes only hyperparameters; submodels and the
shared delta are built inside `_fit` from FitSamples (distances computed
via haversine over the sample coords).

_predict catches exceptions from OctantRTTModel.predict_distance_bounds (the
common failure is a fitted model with no spline) and maps them to
NUMERICAL_FAILURE — distinct from VP_NOT_FITTED (unfitted) and
RTT_OUT_OF_RANGE (latency cutoff).

Wraps scripts/libs/octant/octant_model.py :: OctantRTTModel + find_delta_for_coverage.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from scripts.framework.v2.ltd.base import (
    AnnulusLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.cbg_feasibility.rtt_model import haversine_distance
from scripts.libs.octant.octant_model import OctantRTTModel, find_delta_for_coverage

logger = logging.getLogger(__name__)


@register_ltd("bounded_spline")
class BoundedSplineLTD(AnnulusLTDModel):
    """Per-anchor Octant spline RTT-to-distance model with annular bounds."""

    def __init__(
        self,
        max_rtt_ms: float = float("inf"),
        target_coverage: float = 0.80,
        cutoff_variant: str = "high_only",
        cutoff_min_points: int = 5,
        fit_spline: bool = True,
        spline_n_knots: int = 4,
        bin_size_ms: float = 5.0,
    ) -> None:
        self.max_rtt_ms = max_rtt_ms
        self.target_coverage = target_coverage
        self.cutoff_variant = cutoff_variant
        self.cutoff_min_points = cutoff_min_points
        self.fit_spline = fit_spline
        self.spline_n_knots = spline_n_knots
        self.bin_size_ms = bin_size_ms
        self._submodels: Dict[VpId, OctantRTTModel] = {}
        self._delta: Optional[float] = None

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

        new_submodels: Dict[VpId, OctantRTTModel] = {}
        pooled_rtts: list[float] = []
        pooled_dists: list[float] = []

        for vp_id, data in by_vp.items():
            vp_coord: Coord = data["vp_coord"]
            rtts = np.array(data["rtts"], dtype=float)
            dists = np.array(data["distances"], dtype=float)
            model = OctantRTTModel(
                anchor_ip=str(vp_id),
                anchor_lat=vp_coord.lat,
                anchor_lon=vp_coord.lon,
                cutoff_variant=self.cutoff_variant,
            )
            try:
                model.fit(
                    rtts,
                    dists,
                    cutoff_min_points=self.cutoff_min_points,
                    fit_spline=self.fit_spline,
                    spline_n_knots=self.spline_n_knots,
                    bin_size_ms=self.bin_size_ms,
                )
            except Exception:
                pass
            new_submodels[vp_id] = model

            if model.fitted and model.spline_rtt_knots is not None:
                pooled_rtts.extend(rtts.tolist())
                pooled_dists.extend(dists.tolist())

        delta: Optional[float] = None
        fitted_with_spline = [
            m
            for m in new_submodels.values()
            if m.fitted and m.spline_rtt_knots is not None
        ]
        if pooled_rtts and fitted_with_spline:
            ref = fitted_with_spline[0]
            try:
                delta, _ = find_delta_for_coverage(
                    np.array(pooled_rtts),
                    np.array(pooled_dists),
                    np.array(ref.spline_rtt_knots),
                    np.array(ref.spline_dist_knots),
                    target_coverage=self.target_coverage,
                )
            except Exception as exc:
                logger.debug("Delta search failed: %s — falling back to hull bounds", exc)

        self._submodels = new_submodels
        self._delta = delta
        fitted_vps = [vp for vp, m in new_submodels.items() if m.fitted]
        return FittingResult(
            success=True,
            args={
                "vps_fitted": fitted_vps,
                "vps_attempted": list(new_submodels),
                "delta": delta,
            },
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
        try:
            inner_km, outer_km = submodel.predict_distance_bounds(
                latency, delta=self._delta
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
                latency=latency,
            )
        inner_km = max(0.0, float(inner_km))
        outer_km = max(0.0, float(outer_km))
        if outer_km <= inner_km:
            return LTDResult(
                success=False,
                error=Error.DEGENERATE_REGION,
                vp_id=vp_id,
                vp_coord=vp_coord,
                latency=latency,
            )
        return LTDResult(
            success=True,
            vp_id=vp_id,
            vp_coord=vp_coord,
            latency=latency,
            tg_distance=Distance(upper_km=outer_km, lower_km=inner_km),
        )
