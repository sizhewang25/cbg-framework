"""NormalDistLTD — pooled-normal RTT-to-distance model (Laki et al. 2011).

A single (mu(d), sigma(d)) shared across all VPs — the pooled-normal claim.
Maps an RTT to an annular Distance:

    lower_km = max(0, mu(rtt) - k * sigma(rtt))
    upper_km = min(max(0, mu(rtt) + k * sigma(rtt)), rtt / THEORETICAL_SLOPE)

With the default `target_coverage=None`, k = 1.0 and the band reproduces the
paper's published Figure 3a (Laki et al. 2011). Setting `target_coverage`
switches to a calibrated k = quantile(|z|, target_coverage) — the Spotter
analogue of Octant's δ-search (see bounded_spline.py for the parallel knob).
Above `cutoff_rtt` (the right edge of the last dense RTT bin) mu and sigma are
held flat at the cutoff value — Octant-style graceful degradation when the
deg-3 / deg-2 polynomial extrapolation would otherwise diverge in the sparse
tail. The 2/3*c clip on the outer bound keeps low-RTT predictions inside the
physical envelope.

The constructor takes only hyperparameters; the SpotterRTTModel is built inside
`_fit` from all samples pooled (no per-VP partitioning — that's the point).

Wraps scripts/libs/spotter/spotter_model.py :: SpotterRTTModel.
"""

from __future__ import annotations

import logging
from typing import Optional

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
from scripts.libs.spotter.spotter_model import SpotterRTTModel

logger = logging.getLogger(__name__)


@register_ltd("normal_dist")
class NormalDistLTD(AnnulusLTDModel):
    """Pooled-normal RTT-to-distance model with the paper's +/-sigma band."""

    def __init__(
        self,
        n_bins: int = 40,
        min_per_bin: int = 30,
        deg_mu: int = 3,
        deg_sigma: int = 2,
        bin_size_ms: float = 5.0,
        cutoff_min_points: int = 30,
        sentinel_rtt: float = 10000.0,
        target_coverage: Optional[float] = None,
    ) -> None:
        self.n_bins = n_bins
        self.min_per_bin = min_per_bin
        self.deg_mu = deg_mu
        self.deg_sigma = deg_sigma
        self.bin_size_ms = bin_size_ms
        self.cutoff_min_points = cutoff_min_points
        self.sentinel_rtt = sentinel_rtt
        self.target_coverage = target_coverage
        self._model: Optional[SpotterRTTModel] = None

    def _fit(self, samples: list[FitSample]) -> FittingResult:
        if not samples:
            return FittingResult(success=False, error=Error.INSUFFICIENT_DATA)

        rtts = np.array([float(s.latency) for s in samples], dtype=float)
        dists = np.array(
            [
                haversine_distance(
                    s.vp_coord.lat,
                    s.vp_coord.lon,
                    s.probe_coord.lat,
                    s.probe_coord.lon,
                )
                for s in samples
            ],
            dtype=float,
        )

        model = SpotterRTTModel(sentinel_rtt=self.sentinel_rtt)
        try:
            model.fit(
                rtts,
                dists,
                n_bins=self.n_bins,
                min_per_bin=self.min_per_bin,
                deg_mu=self.deg_mu,
                deg_sigma=self.deg_sigma,
                target_coverage=self.target_coverage,
                bin_size_ms=self.bin_size_ms,
                cutoff_min_points=self.cutoff_min_points,
            )
        except Exception:
            return FittingResult(success=False, error=Error.NUMERICAL_FAILURE)

        if not model.fitted:
            return FittingResult(
                success=False,
                error=Error.NUMERICAL_FAILURE,
                args={"fit_message": model.fit_message},
            )

        self._model = model
        return FittingResult(
            success=True,
            args={
                "rtt_min": model.rtt_min,
                "rtt_max": model.rtt_max,
                "cutoff_rtt": model.cutoff_rtt,
            },
        )

    def _predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult:
        if self._model is None or not self._model.fitted:
            return LTDResult(
                success=False,
                error=Error.VP_NOT_FITTED,
                vp_id=vp_id,
                vp_coord=vp_coord,
                latency=latency,
            )
        try:
            bounds = self._model.predict_distance_bounds(latency)
        except Exception as exc:
            logger.debug(
                "Spotter predict_distance_bounds failed for %s at RTT %.3f ms: %s",
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
        if bounds is None:
            return LTDResult(
                success=False,
                error=Error.RTT_OUT_OF_RANGE,
                vp_id=vp_id,
                vp_coord=vp_coord,
                latency=latency,
            )
        inner_km, outer_km = bounds
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
            tg_distance=Distance(
                upper_km=float(outer_km), lower_km=float(inner_km)
            ),
        )
