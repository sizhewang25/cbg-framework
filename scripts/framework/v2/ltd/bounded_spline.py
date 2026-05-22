"""BoundedSplineLTD — per-VP Octant spline + delta band (Octant).

Per-VP `OctantRTTModel` produces annular bounds (inner, outer) for each RTT;
a per-VP `delta` widens the spline-anchored band to a target coverage of
that VP's own training points. Each VP carries its own δ; nothing is pooled
across VPs.

`_fit` builds the submodels and runs `find_delta_for_coverage` once per
fitted VP using only that VP's data and spline. `_predict` looks up the
per-VP δ and asks the submodel for `predict_distance_bounds(rtt, delta)`.

Wraps scripts/libs/octant_simple/octant_model.py — the simplified mirror
that drops the multi-variant cutoff (only `high_only` behavior survives)
and the unused RTT-array prediction helpers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import numpy as np

from scripts.framework.v2.ltd.base import (
    AnnulusLTDModel,
    FitSample,
    FittingResult,
    LTDResult,
)
from scripts.framework.v2.registry import register_ltd
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.cbg.rtt_model import haversine_distance
from scripts.libs.octant_simple.octant_model import (
    OctantRTTModel,
    find_delta_for_coverage,
)

logger = logging.getLogger(__name__)


@register_ltd("bounded_spline")
class BoundedSplineLTD(AnnulusLTDModel):
    """Per-anchor Octant spline RTT-to-distance model with per-VP delta band."""

    def __init__(
        self,
        target_coverage: float = 0.9,
        cutoff_min_points: int = 5,
        fit_spline: bool = True,
        spline_n_knots: int = 4,
        bin_size_ms: float = 5.0,
        sentinel_rtt: float = 10000.0,
    ) -> None:
        self.target_coverage = target_coverage
        self.cutoff_min_points = cutoff_min_points
        self.fit_spline = fit_spline
        self.spline_n_knots = spline_n_knots
        self.bin_size_ms = bin_size_ms
        self.sentinel_rtt = sentinel_rtt
        self._submodels: Dict[VpId, OctantRTTModel] = {}
        self._deltas: Dict[VpId, float] = {}

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
        new_deltas: Dict[VpId, float] = {}

        for vp_id, data in by_vp.items():
            vp_coord: Coord = data["vp_coord"]
            rtts = np.array(data["rtts"], dtype=float)
            dists = np.array(data["distances"], dtype=float)
            model = OctantRTTModel(
                anchor_ip=str(vp_id),
                anchor_lat=vp_coord.lat,
                anchor_lon=vp_coord.lon,
                sentinel_rtt=self.sentinel_rtt,
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
                pass  # .fitted reflects failure; submodel is still stored.
            new_submodels[vp_id] = model

            if model.fitted and model.spline_rtt_knots is not None:
                try:
                    delta, _ = find_delta_for_coverage(
                        rtts,
                        dists,
                        np.array(model.spline_rtt_knots),
                        np.array(model.spline_dist_knots),
                        target_coverage=self.target_coverage,
                    )
                    new_deltas[vp_id] = delta
                except Exception as exc:
                    logger.debug(
                        "Per-VP delta search failed for %s: %s — falling back to hull bounds",
                        vp_id,
                        exc,
                    )

        self._submodels = new_submodels
        self._deltas = new_deltas
        fitted_vps = [vp for vp, m in new_submodels.items() if m.fitted]
        return FittingResult(
            success=True,
            args={
                "vps_fitted": fitted_vps,
                "vps_attempted": list(new_submodels),
                "deltas": dict(new_deltas),
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
        delta = self._deltas.get(vp_id)
        try:
            inner_km, outer_km = submodel.predict_distance_bounds(latency, delta=delta)
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
