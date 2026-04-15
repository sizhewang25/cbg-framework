"""Phase 1 variant: Bounded Spline (Octant).

Per-anchor spline + shared delta band → annular constraints (inner, outer radius).

Wraps: scripts/analysis/octant/octant_model.py :: OctantRTTModel.predict_distance_bounds()
Reference: run_octant_cbg() in octant_evaluation.py, form_constraint() in octant_geolocation.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.framework.distance import BaseDistance
from scripts.framework.registry import register_distance
from scripts.framework.types import CircleConstraint


@register_distance("bounded_spline")
class BoundedSplineDistance(BaseDistance):
    """Octant bounded spline RTT → distance model.

    Produces annular constraints with inner and outer radius bounds.
    Weight = exp(-rtt / tau) for downstream weighted methods.

    Requires fitting before use. Two modes:
      (a) fit(models={ip: OctantRTTModel}, delta=float)  — pass pre-fitted
      (b) fit(df_asn=df)  — delegate to fit_octant_models()
    """

    name = "bounded_spline"

    def __init__(
        self,
        weight_tau_ms: float = 50.0,
        max_rtt_ms: float = float("inf"),
    ):
        self.weight_tau_ms = weight_tau_ms
        self.max_rtt_ms = max_rtt_ms
        self.models: Dict = {}  # {anchor_ip: OctantRTTModel}
        self.delta: Optional[float] = None  # shared delta for spline band

    def fit(
        self,
        df_asn=None,
        models=None,
        delta=None,
        target_coverage: float = 0.80,
        **kwargs,
    ) -> None:
        """Fit or load per-anchor Octant spline models.

        Args:
            models: Pre-fitted dict {anchor_ip: OctantRTTModel}. Takes priority.
            delta: Shared delta for spline distance band.
            df_asn: DataFrame for fitting. Delegates to fit_octant_models().
            target_coverage: Target coverage for delta search (used with df_asn).
        """
        if models is not None:
            self.models = models
            if delta is not None:
                self.delta = delta
            return
        if df_asn is not None:
            from scripts.analysis.octant.octant_evaluation import fit_octant_models

            fitted_models, computed_delta = fit_octant_models(
                df_asn, target_coverage=target_coverage, **kwargs
            )
            self.models = fitted_models
            self.delta = computed_delta if delta is None else delta

    def estimate(
        self,
        measurements: Dict[str, float],
        anchor_coords: Dict[str, Tuple[float, float]],
    ) -> List[CircleConstraint]:
        circles = []
        for vp_ip, rtt in measurements.items():
            if vp_ip not in anchor_coords or vp_ip not in self.models:
                continue
            if rtt > self.max_rtt_ms:
                continue
            model = self.models[vp_ip]
            if not model.fitted:
                continue
            inner_km, outer_km = model.predict_distance_bounds(
                rtt, delta=self.delta
            )
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
                    inner_radius_km=max(0.0, inner_km),
                    weight=weight,
                )
            )
        return circles
