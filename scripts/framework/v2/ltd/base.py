"""Latency-to-distance model interface.

Two family bases — CircleLTDModel and AnnulusLTDModel — encode at the type
level whether a model produces disk constraints (lower_km == 0) or annular
constraints (lower_km may be > 0). CBGModel uses isinstance against these
bases to validate compatibility with the multilateration stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Optional

from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId


def _require_positive_latency(latency: Latency) -> None:
    """Guard: RTT must be strictly positive for any LTD prediction to make sense.

    A zero or negative latency indicates a bug in the caller's data pipeline
    (or an upstream parse failure), not a transient prediction failure, so we
    raise rather than return an LTDResult(success=False). Centralized in the
    base so every concrete wrapper inherits the same invariant.
    """
    if latency <= 0:
        raise ValueError(
            f"latency (RTT in ms) must be strictly positive, got {latency!r}"
        )


@dataclass(frozen=True)
class FitSample:
    """One training observation."""

    vp_id: VpId
    vp_coord: Coord
    probe_coord: Coord
    latency: Latency


@dataclass(frozen=True)
class FittingResult:
    """Outcome of LTDModel.fit. `args` carries fitted parameters (shape is
    model-specific: a per-VP dict for CircleLTDModel/AnnulusLTDModel
    partitioning by VP, a single struct for a future global model).
    `method` is auto-stamped by LTDModel.fit with the concrete class name."""

    success: bool
    error: Optional[Error] = None
    args: Optional[dict] = None
    method: Optional[str] = None


@dataclass(frozen=True)
class LTDResult:
    """Single per-VP latency-to-distance prediction.

    When success is True, vp_id / vp_coord / tg_distance are populated.
    When success is False, error is set and the geometry fields may be None.
    `latency` is echoed from the call on both paths so downstream stages
    (e.g. MTL weight = exp(-rtt/tau)) don't have to thread the input
    latency separately through the pipeline.
    `method` is auto-stamped by LTDModel.predict with the concrete class name.
    """

    success: bool
    error: Optional[Error] = None
    vp_id: Optional[VpId] = None
    vp_coord: Optional[Coord] = None
    latency: Optional[Latency] = None
    tg_distance: Optional[Distance] = None
    method: Optional[str] = None


class LTDModel(ABC):
    """Abstract latency-to-distance model.

    Two shapes both implement this interface:
      * Per-VP (mainstream): one submodel per VpId, partitioned at fit time.
      * Global (future):     one model pooled across all VPs.

    Concrete classes choose. Callers don't need to know which.

    Subclasses implement `_fit` and `_predict`; the public `fit` / `predict`
    wrappers stamp `method=type(self).__name__` onto the returned result.

    Do not subclass LTDModel directly. Subclass CircleLTDModel or
    AnnulusLTDModel so that compatibility with the multilateration stage
    is expressed in the type system.
    """

    @abstractmethod
    def _fit(self, samples: list[FitSample]) -> FittingResult: ...

    @abstractmethod
    def _predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult: ...

    def fit(self, samples: list[FitSample]) -> FittingResult:
        return replace(self._fit(samples), method=type(self).__name__)

    def predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult:
        _require_positive_latency(latency)
        return replace(
            self._predict(vp_id, vp_coord, latency),
            method=type(self).__name__,
        )

    def predict_all(
        self,
        obs: list[tuple[VpId, Coord, Latency]],
    ) -> list[LTDResult]:
        return [self.predict(vp_id, vp_coord, lat) for vp_id, vp_coord, lat in obs]


class CircleLTDModel(LTDModel, ABC):
    """Produces disk constraints (Distance.lower_km is always 0)."""


class AnnulusLTDModel(LTDModel, ABC):
    """Produces possibly-annular constraints (Distance.lower_km may be > 0)."""
