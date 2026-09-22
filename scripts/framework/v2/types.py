"""Shared types for the v2 CBG framework.

Coord, Distance, Latency, VpId are passed between stages.
Error and GeoStatus enumerate the failure / status vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType, Optional

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

    `mu_km` / `sigma_km` are the OPTIONAL distribution the bounds were derived
    from, for LTDs whose model is a distribution over distance rather than a
    pair of limits (today: NormalDistLTD, i.e. Spotter's f_d = N(mu, sigma^2)).
    They exist because the bounds are **not invertible** back to the
    distribution: `lower_km` is clamped at 0, `upper_km` is clipped by the
    2/3*c envelope, and the band half-width is `k * sigma` for an LTD-internal
    `k`. A density MTL that re-derived mu = (lower+upper)/2 would therefore be
    silently wrong exactly where the clipping bit. Geometric MTLs ignore both
    fields; only a DensityMTLMethod reads them.
    """

    upper_km: float
    lower_km: float = 0.0
    mu_km: Optional[float] = None
    sigma_km: Optional[float] = None

    def __post_init__(self) -> None:
        if self.upper_km < 0:
            raise ValueError(f"upper_km must be non-negative, got {self.upper_km}")
        if self.lower_km < 0:
            raise ValueError(f"lower_km must be non-negative, got {self.lower_km}")
        if self.lower_km > self.upper_km:
            raise ValueError(
                f"lower_km ({self.lower_km}) must not exceed upper_km ({self.upper_km})"
            )
        # sigma == 0 is rejected rather than tolerated: it divides the z-score
        # in every density MTL, and a zero-width Gaussian is not a weaker
        # constraint but an undefined one.
        if self.sigma_km is not None and self.sigma_km <= 0:
            raise ValueError(f"sigma_km must be positive when set, got {self.sigma_km}")
        if self.mu_km is not None and self.mu_km < 0:
            raise ValueError(f"mu_km must be non-negative when set, got {self.mu_km}")

    @property
    def is_annular(self) -> bool:
        return self.lower_km > 0.0

    @property
    def has_distribution(self) -> bool:
        """True when both mu and sigma are present, i.e. a density MTL can run."""
        return self.mu_km is not None and self.sigma_km is not None


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
