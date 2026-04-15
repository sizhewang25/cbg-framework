"""CBGPipeline — composable 4-phase geolocation pipeline.

Usage:
    pipe = CBGPipeline.from_config(
        distance="speed_of_internet",
        filtering="redundant_circle",
        multilateration="spherical",
        centroid="arithmetic_mean",
    )
    location, circles = pipe.geolocate(measurements, anchor_coords)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from scripts.framework.types import CircleConstraint


# Incompatible (multilateration, distance) pairs
_INCOMPATIBLE_MULTILAT_DISTANCE = {
    ("weighted_grid", "speed_of_internet"),
    ("weighted_grid", "low_envelope"),
}


class CBGPipeline:
    """Composable 4-phase CBG geolocation pipeline.

    Phases:
        1. distance     — RTT → CircleConstraint list
        2. filtering    — remove erroneous/redundant constraints
        3. multilateration — intersect constraints → region
        4. centroid     — select single point from region

    Fallback: when centroid selection fails, uses closest VP by min RTT.
    """

    def __init__(self, distance, filtering, multilateration, centroid):
        self.distance = distance
        self.filtering = filtering
        self.multilateration = multilateration
        self.centroid = centroid

    def geolocate(
        self,
        measurements: Dict[str, float],
        anchor_coords: Dict[str, Tuple[float, float]],
    ) -> Tuple[Optional[Tuple[float, float]], List[CircleConstraint]]:
        """Run the full 4-phase pipeline for one probe.

        Args:
            measurements: {anchor_ip: min_rtt_ms}
            anchor_coords: {anchor_ip: (lat, lon)}

        Returns:
            (location_or_None, circles_used)
            location is (lat, lon) or None if all phases fail.
        """
        # Phase 1: Distance estimation
        circles = self.distance.estimate(measurements, anchor_coords)
        if not circles:
            return None, []

        # Phase 2: Filtering
        filtered = self.filtering.filter(circles)
        if not filtered:
            return None, circles

        # Phase 3: Multilateration
        result = self.multilateration.multilaterate(filtered)

        # Phase 4: Centroid selection
        location = self.centroid.select(result)

        # Fallback: closest VP by min RTT
        if location is None and filtered:
            closest = min(filtered, key=lambda c: c.rtt_ms)
            location = (closest.vp_lat, closest.vp_lon)

        used = result.circles_used if result.circles_used else filtered
        return location, used

    def geolocate_batch(
        self,
        targets: Dict[str, Dict[str, float]],
        anchor_coords: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[Optional[Tuple[float, float]], List[CircleConstraint]]]:
        """Run the pipeline for multiple probes.

        Args:
            targets: {probe_ip: {anchor_ip: min_rtt_ms}}
            anchor_coords: {anchor_ip: (lat, lon)}

        Returns:
            {probe_ip: (location, circles_used)}
        """
        return {
            ip: self.geolocate(rtts, anchor_coords)
            for ip, rtts in targets.items()
        }

    @classmethod
    def from_config(
        cls,
        distance: str = "speed_of_internet",
        filtering: str = "redundant_circle",
        multilateration: str = "spherical",
        centroid: str = "arithmetic_mean",
        distance_kwargs: Optional[Dict[str, Any]] = None,
        filtering_kwargs: Optional[Dict[str, Any]] = None,
        multilateration_kwargs: Optional[Dict[str, Any]] = None,
        centroid_kwargs: Optional[Dict[str, Any]] = None,
    ) -> CBGPipeline:
        """Create a pipeline from string names (registry lookup).

        Args:
            distance: Name registered in DISTANCE_REGISTRY.
            filtering: Name registered in FILTERING_REGISTRY.
            multilateration: Name registered in MULTILATERATION_REGISTRY.
            centroid: Name registered in CENTROID_REGISTRY.
            *_kwargs: Optional keyword arguments for each component's __init__.

        Returns:
            Configured CBGPipeline instance.

        Raises:
            ValueError: If an incompatible combination is requested.
            KeyError: If a name is not found in its registry.
        """
        from scripts.framework.registry import (
            CENTROID_REGISTRY,
            DISTANCE_REGISTRY,
            FILTERING_REGISTRY,
            MULTILATERATION_REGISTRY,
        )

        # Validate compatibility
        if (multilateration, distance) in _INCOMPATIBLE_MULTILAT_DISTANCE:
            raise ValueError(
                f"multilateration={multilateration!r} requires 'bounded_spline' "
                f"distance, got {distance!r}"
            )
        if multilateration == "weighted_grid" and filtering != "none":
            warnings.warn(
                f"weighted_grid has built-in filtering; filtering={filtering!r} "
                f"is redundant. Consider filtering='none'.",
                stacklevel=2,
            )

        return cls(
            distance=DISTANCE_REGISTRY[distance](**(distance_kwargs or {})),
            filtering=FILTERING_REGISTRY[filtering](**(filtering_kwargs or {})),
            multilateration=MULTILATERATION_REGISTRY[multilateration](
                **(multilateration_kwargs or {})
            ),
            centroid=CENTROID_REGISTRY[centroid](**(centroid_kwargs or {})),
        )

    def __repr__(self) -> str:
        return (
            f"CBGPipeline("
            f"distance={self.distance.name!r}, "
            f"filtering={self.filtering.name!r}, "
            f"multilateration={self.multilateration.name!r}, "
            f"centroid={self.centroid.name!r})"
        )
