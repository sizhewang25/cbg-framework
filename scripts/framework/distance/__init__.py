"""Phase 1: RTT-to-Distance Estimation.

Base class for all distance estimation variants.
Each variant converts RTT measurements into CircleConstraint objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from scripts.framework.types import CircleConstraint


class BaseDistance:
    """Abstract base for RTT → radius estimation."""

    name: str = "base"

    def fit(self, df_asn=None, **kwargs) -> None:
        """Optional calibration step (e.g., fitting LP models).

        Override in subclasses that require per-anchor model fitting.
        No-op by default.
        """

    def estimate(
        self,
        measurements: Dict[str, float],
        anchor_coords: Dict[str, Tuple[float, float]],
    ) -> List[CircleConstraint]:
        """Convert RTT measurements to circle constraints for one probe.

        Args:
            measurements: {anchor_ip: min_rtt_ms}
            anchor_coords: {anchor_ip: (lat, lon)}

        Returns:
            List of CircleConstraint, one per valid anchor.
        """
        raise NotImplementedError
