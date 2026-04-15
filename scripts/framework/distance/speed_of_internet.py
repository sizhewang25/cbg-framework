"""Phase 1 variant: Speed-of-Internet (Million-Scale CBG).

Theoretical 2/3c model: radius_km = rtt_to_km(rtt, speed_threshold=2/3).
At 2/3c this equals 100 * rtt_ms.

Wraps: scripts/utils/helpers.py :: rtt_to_km()
Reference: run_million_scale_cbg() in evaluate_million_scale.py:123
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from scripts.framework.distance import BaseDistance
from scripts.framework.registry import register_distance
from scripts.framework.types import CircleConstraint
from scripts.utils.helpers import rtt_to_km


@register_distance("speed_of_internet")
class SpeedOfInternetDistance(BaseDistance):
    """Theoretical speed-of-Internet model (IMC 2012).

    Converts RTT to distance using a fixed fraction of the speed of light.
    Default: 2/3c → radius = 100 × RTT.
    """

    name = "speed_of_internet"

    def __init__(
        self,
        speed_threshold: float = 2 / 3,
        max_rtt_ms: float = float("inf"),
    ):
        self.speed_threshold = speed_threshold
        self.max_rtt_ms = max_rtt_ms

    def estimate(
        self,
        measurements: Dict[str, float],
        anchor_coords: Dict[str, Tuple[float, float]],
    ) -> List[CircleConstraint]:
        circles = []
        for vp_ip, rtt in measurements.items():
            if vp_ip not in anchor_coords:
                continue
            if rtt > self.max_rtt_ms:
                continue
            lat, lon = anchor_coords[vp_ip]
            radius_km = rtt_to_km(rtt, speed_threshold=self.speed_threshold)
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
