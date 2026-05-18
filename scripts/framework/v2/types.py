"""Shared types for the v2 CBG framework.

Coord, Distance, Latency, VpId are passed between stages.
Error and GeoStatus enumerate the failure / status vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

Latency = NewType("Latency", float)
VpId = NewType("VpId", str)


@dataclass(frozen=True)
class Coord:
    lat: float
    lon: float


@dataclass(frozen=True)
class Distance:
    """One distance constraint, possibly annular.

    lower_km == 0 means a full disk of radius upper_km.
    lower_km > 0 means an annulus with the given inner and outer radii.
    """

    upper_km: float
    lower_km: float = 0.0

    def __post_init__(self) -> None:
        if self.upper_km < 0:
            raise ValueError(f"upper_km must be non-negative, got {self.upper_km}")
        if self.lower_km < 0:
            raise ValueError(f"lower_km must be non-negative, got {self.lower_km}")
        if self.lower_km > self.upper_km:
            raise ValueError(
                f"lower_km ({self.lower_km}) must not exceed upper_km ({self.upper_km})"
            )

    @property
    def is_annular(self) -> bool:
        return self.lower_km > 0.0


class Error(Enum):
    """Failure reasons produced by stages or the composed pipeline."""

    INSUFFICIENT_DATA = "insufficient_data"
    NUMERICAL_FAILURE = "numerical_failure"
    VP_NOT_FITTED = "vp_not_fitted"
    RTT_OUT_OF_RANGE = "rtt_out_of_range"
    NO_INTERSECTION = "no_intersection"
    INSUFFICIENT_CONSTRAINTS = "insufficient_constraints"
    EMPTY_REGION = "empty_region"
    DEGENERATE_REGION = "degenerate_region"
    ALL_PHASES_FAILED = "all_phases_failed"


class GeoStatus(Enum):
    """Top-level outcome of CBGModel.geolocate."""

    SUCCESS = "success"
    FALLBACK = "fallback"
    ERROR = "error"
