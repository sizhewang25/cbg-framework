"""Phase 1 variant: Spotter pooled normal (Laki et al. 2011).

Single (mu(d), sigma(d), k) shared across all anchors -- Spotter's central
landmark-independence claim. Maps to an annular CircleConstraint:

    inner_km = max(0, mu(rtt) - k * sigma(rtt))
    outer_km = max(0, mu(rtt) + k * sigma(rtt))

with k calibrated empirically as quantile(|z|, target_coverage) on the
calibration set rather than picked from a normal table. See
notes/2026-05-17-spotter-normality-check.md for why empirical: the pooled
z distribution is leptokurtic (sigma_z ~= 0.89 on probes->anchors), so the
parametric k = Phi^-1(1 - alpha/2) is wrong by ~12 %. Out-of-range RTTs
return no constraint -- the deg-3 mu(d) / deg-2 sigma(d) polynomials are not
safe to extrapolate.

Wraps: scripts/libs/spotter/spotter_model.py :: SpotterRTTModel
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.framework.distance import BaseDistance
from scripts.framework.registry import register_distance
from scripts.framework.types import CircleConstraint
from scripts.libs.spotter.spotter_model import SpotterRTTModel

logger = logging.getLogger(__name__)


@register_distance("spotter")
class SpotterDistance(BaseDistance):
    """Spotter pooled-normal RTT -> distance model.

    Produces annular constraints with a single (mu, sigma, k) shared across
    all anchors. Two fit modes:
      (a) fit(model=SpotterRTTModel)  -- pass a pre-fitted model.
      (b) fit(df_asn=df, target_coverage=...)  -- calibrate from columns
          `min_rtt` and `distance_km`.

    CAVEAT. The pooled (= landmark-independent) assumption holds on
    anchors->anchors data (sigma_z ~ 0.96) but FAILS on probes->anchors due
    to consumer last-mile heterogeneity (per-anchor Q-Q S-shapes). See the
    note for the panel-by-panel mechanism. Use as a benchmark baseline that
    deliberately refuses per-VP calibration.
    """

    name = "spotter"

    def __init__(
        self,
        target_coverage: float = 0.95,
        weight_tau_ms: float = 50.0,
        max_rtt_ms: float = float("inf"),
    ):
        self.target_coverage = target_coverage
        self.weight_tau_ms = weight_tau_ms
        self.max_rtt_ms = max_rtt_ms
        self.model: Optional[SpotterRTTModel] = None

    def fit(
        self,
        df_asn=None,
        model: Optional[SpotterRTTModel] = None,
        target_coverage: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Fit or load the pooled Spotter model.

        Args:
            model: Pre-fitted SpotterRTTModel. Takes priority.
            df_asn: DataFrame with columns `min_rtt` (ms) and `distance_km`.
                Calibrates a fresh SpotterRTTModel.
            target_coverage: Overrides constructor default if provided.
            **kwargs: Forwarded to SpotterRTTModel.fit (n_bins, min_per_bin,
                deg_mu, deg_sigma).
        """
        if model is not None:
            self.model = model
            return
        if df_asn is not None:
            tc = self.target_coverage if target_coverage is None else target_coverage
            rtt = np.asarray(df_asn["min_rtt"], dtype=float)
            dist = np.asarray(df_asn["distance_km"], dtype=float)
            fresh = SpotterRTTModel()
            fresh.fit(rtt, dist, target_coverage=tc, **kwargs)
            self.model = fresh

    def estimate(
        self,
        measurements: Dict[str, float],
        anchor_coords: Dict[str, Tuple[float, float]],
    ) -> List[CircleConstraint]:
        circles: List[CircleConstraint] = []
        if self.model is None or not self.model.fitted:
            return circles
        for vp_ip, rtt in measurements.items():
            if vp_ip not in anchor_coords:
                continue
            if rtt > self.max_rtt_ms:
                continue
            bounds = self.model.predict_distance_bounds(rtt)
            if bounds is None:
                continue
            inner_km, outer_km = bounds
            if outer_km <= inner_km:
                continue
            lat, lon = anchor_coords[vp_ip]
            weight = float(np.exp(-rtt / self.weight_tau_ms))
            circles.append(
                CircleConstraint(
                    vp_lat=lat,
                    vp_lon=lon,
                    vp_ip=vp_ip,
                    rtt_ms=rtt,
                    radius_km=outer_km,
                    inner_radius_km=inner_km,
                    weight=weight,
                )
            )
        return circles
