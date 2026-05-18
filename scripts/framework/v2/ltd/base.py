"""Latency-to-distance model interface.

Two family bases — CircleLTDModel and AnnulusLTDModel — encode at the type
level whether a model produces disk constraints (lower_km == 0) or annular
constraints (lower_km may be > 0). CBGModel uses isinstance against these
bases to validate compatibility with the multilateration stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId


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
    partitioning by VP, a single struct for a future global model)."""

    success: bool
    error: Optional[Error] = None
    args: Optional[dict] = None


@dataclass(frozen=True)
class LTDResult:
    """Single per-VP latency-to-distance prediction.

    When success is True, vp_id / vp_coord / tg_distance are populated.
    When success is False, error is set and the geometry fields may be None.
    """

    success: bool
    error: Optional[Error] = None
    vp_id: Optional[VpId] = None
    vp_coord: Optional[Coord] = None
    tg_distance: Optional[Distance] = None


class LTDModel(ABC):
    """Abstract latency-to-distance model.

    Two shapes both implement this interface:
      * Per-VP (mainstream): one submodel per VpId, partitioned at fit time.
      * Global (future):     one model pooled across all VPs.

    Concrete classes choose. Callers don't need to know which.

    Do not subclass LTDModel directly. Subclass CircleLTDModel or
    AnnulusLTDModel so that compatibility with the multilateration stage
    is expressed in the type system.
    """

    @abstractmethod
    def fit(self, samples: list[FitSample]) -> FittingResult: ...

    @abstractmethod
    def predict(
        self,
        vp_id: VpId,
        vp_coord: Coord,
        latency: Latency,
    ) -> LTDResult: ...

    def predict_all(
        self,
        obs: list[tuple[VpId, Coord, Latency]],
    ) -> list[LTDResult]:
        return [self.predict(vp_id, vp_coord, lat) for vp_id, vp_coord, lat in obs]


class CircleLTDModel(LTDModel, ABC):
    """Produces disk constraints (Distance.lower_km is always 0)."""


class AnnulusLTDModel(LTDModel, ABC):
    """Produces possibly-annular constraints (Distance.lower_km may be > 0)."""
