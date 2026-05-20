"""Multilateration interface.

Two family bases — CircleMTLMethod and AnnulusMTLMethod — encode whether a
method consumes disk-only constraints or requires annular constraints.
CBGModel validates the pairing against LTDModel at composition time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Optional, Union

from shapely.geometry.base import BaseGeometry

from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.types import Coord, Error

# Output region representation. Planar methods (e.g. Shapely-based) return
# a Polygon / MultiPolygon. Spherical methods return a list of (lat, lon)
# vertices on the unit sphere. Centroid methods dispatch on the runtime type.
Intersection = Union[BaseGeometry, list[Coord], None]


@dataclass(frozen=True)
class MTLResult:
    """Outcome of multilateration over a set of distance constraints.

    `method` is auto-stamped by MTLMethod.multilaterate with the concrete
    class name."""

    success: bool
    error: Optional[Error] = None
    intersection: Intersection = None
    method: Optional[str] = None


class MTLMethod(ABC):
    """Abstract multilateration method.

    Subclasses implement `_multilaterate`; the public `multilaterate`
    wrapper stamps `method=type(self).__name__` onto the returned result.

    Do not subclass MTLMethod directly. Subclass CircleMTLMethod or
    AnnulusMTLMethod so that the compatibility requirement against the
    LTDModel stage is expressed in the type system.
    """

    @abstractmethod
    def _multilaterate(self, results: list[LTDResult]) -> MTLResult: ...

    def multilaterate(self, results: list[LTDResult]) -> MTLResult:
        return replace(self._multilaterate(results), method=type(self).__name__)


class CircleMTLMethod(MTLMethod, ABC):
    """Reads only `tg_distance.upper_km` from each LTDResult.

    Accepts either a CircleLTDModel (the native pairing) or an
    AnnulusLTDModel (the inner bound is discarded — a deliberate
    degradation that CBGModel allows).
    """


class AnnulusMTLMethod(MTLMethod, ABC):
    """Designed around annular constraints (lower_km may be > 0).

    Must be paired with an AnnulusLTDModel: lower_km is part of the
    method's information budget, so pairing with a CircleLTDModel —
    which always emits lower_km=0 — would silently strip the method
    of what makes it different from a Circle MTL. CBGModel rejects
    that pairing.
    """
