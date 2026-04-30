"""CBGPipeline — composable CBG geolocation pipeline.

Usage:
    pipe = CBGPipeline.from_config(
        distance="speed_of_internet",
        filtering="redundant_circle",
        multilateration="spherical",
        centroid="arithmetic_mean",
    )
    location, circles = pipe.geolocate(measurements, anchor_coords)

    result = pipe.geolocate_with_metadata(measurements, anchor_coords)
    if result.fallback_used:
        ...
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from scripts.framework.types import CircleConstraint, GeolocationResult


# Incompatible (multilateration, distance) pairs
_INCOMPATIBLE_MULTILAT_DISTANCE = {
    ("weighted_grid", "speed_of_internet"),
    ("weighted_grid", "low_envelope"),
}


class CBGPipeline:
    """Composable CBG geolocation pipeline.

    Core phases:
        1. distance     — RTT → CircleConstraint list
        2. multilateration — intersect constraints → region
        3. centroid     — select single point from region

    Optional preprocessing:
        filtering — remove erroneous/redundant constraints before multilateration.
        Use filtering="none" or filtering=None to disable it.

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
        result = self.geolocate_with_metadata(measurements, anchor_coords)
        return result.location, result.circles_used

    def geolocate_with_metadata(
        self,
        measurements: Dict[str, float],
        anchor_coords: Dict[str, Tuple[float, float]],
    ) -> GeolocationResult:
        """Run the full pipeline and return explicit success/fallback metadata.

        Unlike `geolocate()`, this method distinguishes a real multilateration
        result from closest-VP fallback. Use it for benchmark metrics.
        """
        # Phase 1: Distance estimation
        circles = self.distance.estimate(measurements, anchor_coords)
        if not circles:
            return GeolocationResult(
                location=None,
                fallback_reason="no_constraints",
            )

        # Optional preprocessing: Filtering
        filtered = self.filtering.filter(circles)
        if not filtered:
            return GeolocationResult(
                location=None,
                circles_used=circles,
                all_circles=circles,
                fallback_reason="filtering_removed_all_constraints",
            )

        # Phase 2: Multilateration
        multilat_result = self.multilateration.multilaterate(filtered)

        # Phase 3: Centroid selection
        location = self.centroid.select(multilat_result)
        centroid_success = location is not None

        # Fallback: closest VP by min RTT
        fallback_used = False
        fallback_reason = None
        if location is None and filtered:
            closest = min(filtered, key=lambda c: c.rtt_ms)
            location = (closest.vp_lat, closest.vp_lon)
            fallback_used = True
            fallback_reason = (
                "centroid_failed"
                if multilat_result.success
                else "multilateration_failed"
            )

        used = multilat_result.circles_used if multilat_result.circles_used else filtered
        return GeolocationResult(
            location=location,
            circles_used=used,
            all_circles=circles,
            filtered_circles=filtered,
            multilateration_success=multilat_result.success,
            centroid_success=centroid_success,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

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
        filtering: Optional[str] = "redundant_circle",
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
            filtering: Name registered in FILTERING_REGISTRY, or None for "none".
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

        filtering = filtering or "none"

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
