"""Shared data types for the CBG geolocation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


EARTH_RADIUS_KM = 6371.0


@dataclass
class CircleConstraint:
    """A single VP's constraint on target location.

    For disk constraints (Million-Scale, Vanilla): inner_radius_km = 0.
    For annular constraints (Octant): inner_radius_km > 0.
    """

    vp_lat: float
    vp_lon: float
    vp_ip: str
    rtt_ms: float
    radius_km: float  # outer bound (distance upper limit)
    inner_radius_km: float = 0.0  # 0 = full disk, >0 = annulus
    weight: float = 1.0  # for weighted methods (e.g., exp(-rtt/tau))

    @property
    def radius_rad(self) -> float:
        """Angular radius in radians for spherical intersection math."""
        return self.radius_km / EARTH_RADIUS_KM

    def to_legacy_tuple(self) -> tuple:
        """Convert to (lat, lon, rtt, d, r) for helpers.py functions."""
        return (self.vp_lat, self.vp_lon, self.rtt_ms, self.radius_km, self.radius_rad)


@dataclass
class MultilatResult:
    """Output of Phase 3 multilateration.

    Exactly one of `vertices` or `region` is set when success=True.
    - vertices: list of (lat, lon) from spherical intersection
    - region: Shapely Polygon/MultiPolygon from shapely/weighted methods
    """

    vertices: Optional[list] = None
    region: Any = None  # Shapely geometry
    circles_used: list = field(default_factory=list)
    success: bool = False


@dataclass
class GeolocationResult:
    """Full pipeline result for one target.

    `location` may be present even when `multilateration_success` is false if
    the closest-VP fallback was used. Callers that need intersection-rate or
    availability metrics should use the explicit metadata fields instead of
    inferring state from `location is not None`.
    """

    location: Optional[Tuple[float, float]]
    circles_used: list = field(default_factory=list)
    all_circles: list = field(default_factory=list)
    filtered_circles: list = field(default_factory=list)
    multilateration_success: bool = False
    centroid_success: bool = False
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
