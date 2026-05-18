"""Multilateration interface.

Two family bases — CircleMTLMethod and AnnulusMTLMethod — encode whether a
method consumes disk-only constraints or requires annular constraints.
CBGModel validates the pairing against LTDModel at composition time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    """Outcome of multilateration over a set of distance constraints."""

    success: bool
    error: Optional[Error] = None
    intersection: Intersection = None


class MTLMethod(ABC):
    """Abstract multilateration method.

    Do not subclass MTLMethod directly. Subclass CircleMTLMethod or
    AnnulusMTLMethod so that the compatibility requirement against the
    LTDModel stage is expressed in the type system.
    """

    @abstractmethod
    def multilaterate(self, results: list[LTDResult]) -> MTLResult: ...


class CircleMTLMethod(MTLMethod, ABC):
    """Consumes disk constraints. Must be paired with a CircleLTDModel."""


class AnnulusMTLMethod(MTLMethod, ABC):
    """Requires annular constraints. Must be paired with an AnnulusLTDModel."""
