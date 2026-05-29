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

    Accepts any LTDModel (CircleLTDModel natively, AnnulusLTDModel with the
    inner bound silently discarded — a deliberate degradation that lets
    annulus LTDs be benchmarked against disk MTLs).
    """


class AnnulusMTLMethod(MTLMethod, ABC):
    """Reads both `tg_distance.lower_km` and `tg_distance.upper_km`.

    Native pairing is with AnnulusLTDModel. Circle LTDs (a subclass of
    AnnulusLTDModel with `lower_km` always 0) are also legal: the annular
    region collapses to a disk and the wrapper still produces a polygon —
    useful when downstream stages (e.g. GeometricCentroidCTR) require a
    polygon-shape output that the spherical Circle MTLs don't emit.
    """
