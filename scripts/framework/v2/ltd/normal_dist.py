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
        per_vp_offset: bool = False,
        offset_span_ms: float = 25.0,
        offset_step_ms: float = 0.1,
    ) -> None:
        self.n_bins = n_bins
        self.min_per_bin = min_per_bin
        self.deg_mu = deg_mu
        self.deg_sigma = deg_sigma
        self.bin_size_ms = bin_size_ms
        self.cutoff_min_points = cutoff_min_points
        self.sentinel_rtt = sentinel_rtt
        self.target_coverage = target_coverage
        self.per_vp_offset = per_vp_offset
        self.offset_span_ms = offset_span_ms
        self.offset_step_ms = offset_step_ms
        self._model: Optional[SpotterRTTModel] = None
        self._deltas: dict[VpId, float] = {}

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
        args = {
            "rtt_min": model.rtt_min,
            "rtt_max": model.rtt_max,
            "cutoff_rtt": model.cutoff_rtt,
        }

        self._deltas = {}
        if self.per_vp_offset:
            self._deltas = self._fit_offsets(samples, model, rtts, dists)
            if self._deltas:
                vals = np.fromiter(self._deltas.values(), dtype=float)
                args["n_vps_with_offset"] = int(vals.size)
                args["delta_ms_p05"] = float(np.quantile(vals, 0.05))
                args["delta_ms_p50"] = float(np.median(vals))
                args["delta_ms_p95"] = float(np.quantile(vals, 0.95))

        return FittingResult(success=True, args=args)

    def _fit_offsets(
        self,
        samples: list[FitSample],
        model: SpotterRTTModel,
        rtts: np.ndarray,
        dists: np.ndarray,
    ) -> dict[VpId, float]:
        """One additive RTT offset per VP: argmin_d SSE(dist - mu(rtt - d)).

        Tests R4 of notes/2026-09-19-normal-dist-wrong-for-operator-hypergiant.md,
        which measures that the per-VP residual is not exchangeable noise but a
        reproducible property of each VP's position in a fixed topology, and that
        its shape is a constant additive RTT offset: one scalar per VP removes
        ~90% of the landmark effect (eta^2 10.8% -> 1.2% on as01), with the
        per-VP mean residual correlating -0.93 with the fitted offset.

        Physically the offset absorbs fixed access/backhaul latency or a fixed
        detour to the VP's backbone ingress -- delay that is present on every
        path from that VP and carries no distance information. Subtracting it
        before evaluating mu/sigma is therefore the *propagation-relevant* RTT,
        which is also why `_predict` applies it before the 2/3*c envelope rather
        than after.

        Grid search rather than a derivative method: mu is a deg-3 polynomial
        composed with a clip, so the SSE is piecewise-smooth and not reliably
        unimodal, and a 0.1 ms grid over +/-25 ms is cheap enough at this scale.
        Deliberately mirrors the estimator in
        scripts/libs/cbg_feasibility/spotter_assumption_breakdown.py so the
        fitted offsets are comparable with that note's reported numbers.

        LEAKAGE: `samples` is the training split only -- the framework calls
        `fit` per fold with train data -- so the offsets inherit the same
        leakage-safety as the pooled polynomials. VPs unseen at fit time get no
        entry and fall back to offset 0 (the pooled model) at predict time.
        """
        grid = np.arange(
            -self.offset_span_ms,
            self.offset_span_ms + self.offset_step_ms,
            self.offset_step_ms,
        )
        # Rebuild the same validity mask `_fit` applied, so offsets are fitted
        # on exactly the rows the pooled polynomials saw.
        raw_rtt = np.array([float(s.latency) for s in samples], dtype=float)
        raw_dist = np.array(
            [
                haversine_distance(
                    s.vp_coord.lat, s.vp_coord.lon, s.probe_coord.lat, s.probe_coord.lon
                )
                for s in samples
            ],
            dtype=float,
        )
        valid = (
            np.isfinite(raw_rtt)
            & np.isfinite(raw_dist)
            & (raw_rtt > 0)
            & (raw_dist > 0)
        )
        vp_ids = np.array([s.vp_id for s in samples], dtype=object)[valid]
        r_all, s_all = raw_rtt[valid], raw_dist[valid]

        deltas: dict[VpId, float] = {}
        for vp_id in set(vp_ids.tolist()):
            m = vp_ids == vp_id
            r, s = r_all[m], s_all[m]
            if r.size < self.min_per_bin:
                # Too few rows to estimate a stable offset; the pooled model is
                # the safer answer, and 0 is exactly that.
                continue
            shifted = np.clip(
                r[None, :] - grid[:, None], model.rtt_min, model.rtt_max
            )
            resid = s[None, :] - np.polyval(model.p_mu, shifted)
            best = int(np.argmin(np.einsum("ij,ij->i", resid, resid)))
            deltas[vp_id] = float(grid[best])
        return deltas

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
        # Per-VP offset: subtract this VP's fixed non-propagation delay before
        # consulting the pooled model. Applied BEFORE the 2/3*c envelope inside
        # predict_distance_bounds, because the corrected value is the
        # propagation-relevant RTT -- bounding the speed of light with delay
        # that is known not to be propagation would be the wrong envelope.
        # Clipped into the calibration range for the same reason the model
        # clamps internally: a shifted RTT can land outside it, and
        # extrapolating a deg-3 polynomial there is exactly what `cutoff_rtt`
        # exists to prevent. Unseen VPs get 0.0 and so behave as pooled.
        eval_latency = latency
        if self._deltas:
            delta = self._deltas.get(vp_id, 0.0)
            if delta:
                eval_latency = Latency(
                    float(
                        np.clip(
                            float(latency) - delta,
                            self._model.rtt_min,
                            self._model.rtt_max,
                        )
                    )
                )
        try:
            bounds = self._model.predict_distance_bounds(eval_latency)
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
        # The band is what geometric MTLs consume; (mu, sigma) is the
        # distribution it was derived from, carried alongside for
        # DensityMTLMethod. Attached rather than re-derived downstream because
        # the band is not invertible: inner clamps at 0, outer is clipped by
        # 2/3*c, and the half-width is k*sigma for an internal k.
        #
        # A None here is not an error for this result -- the geometric path is
        # unaffected -- so the bounds are returned either way and only a
        # density MTL will notice the absence.
        mu_sigma = self._model.predict_mu_sigma(eval_latency)
        mu_km = float(mu_sigma[0]) if mu_sigma is not None else None
        sigma_km = float(mu_sigma[1]) if mu_sigma is not None else None
        return LTDResult(
            success=True,
            vp_id=vp_id,
            vp_coord=vp_coord,
            latency=latency,
            tg_distance=Distance(
                upper_km=float(outer_km),
                lower_km=float(inner_km),
                mu_km=mu_km,
                sigma_km=sigma_km,
            ),
        )
