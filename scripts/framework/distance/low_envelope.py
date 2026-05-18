"""Phase 1 variant: Lower Envelope (Vanilla CBG).

Per-anchor LP bestline inversion: radius = (rtt - intercept) / slope.

Wraps: scripts/libs/cbg_feasibility/rtt_model.py :: RTTDistanceModel
Reference: run_vanilla_cbg() in evaluate_million_scale.py:287
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from scripts.framework.distance import BaseDistance
from scripts.framework.registry import register_distance
from scripts.framework.types import CircleConstraint


@register_distance("low_envelope")
class LowEnvelopeDistance(BaseDistance):
    """Per-anchor LP bestline RTT → distance model.

    Requires fitting before use. Two modes:
      (a) fit(models={anchor_ip: RTTDistanceModel, ...})  — pass pre-fitted models
      (b) fit(df_asn=df)  — delegate to fit_lp_models() from evaluate_million_scale.py
    """

    name = "low_envelope"

    def __init__(self, max_rtt_ms: float = float("inf")):
        self.max_rtt_ms = max_rtt_ms
        self.models: Dict = {}  # {anchor_ip: RTTDistanceModel}

    def fit(self, df_asn=None, models=None, **kwargs) -> None:
        """Fit or load per-anchor LP bestline models.

        Args:
            models: Pre-fitted dict {anchor_ip: RTTDistanceModel}. Takes priority.
            df_asn: DataFrame with columns: dst_ip, src_ip, distance_km, min_rtt,
                    anchor_latitude, anchor_longitude, anchor_city.
                    Delegates to fit_lp_models() from evaluate_million_scale.py.
        """
        if models is not None:
            self.models = models
            return
        if df_asn is not None:
            from scripts.libs.million_scale.evaluate_million_scale import (
                fit_lp_models,
            )

            self.models = fit_lp_models(df_asn)

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
            radius_km = model.predict_distance(rtt)
            if radius_km is None or radius_km <= 0:
                continue
            lat, lon = anchor_coords[vp_ip]
            circles.append(
                CircleConstraint(
                    vp_lat=lat,
                    vp_lon=lon,
                    vp_ip=vp_ip,
                    rtt_ms=rtt,
                    radius_km=radius_km,
                )
            )
        return circles
