"""Centroid selection interface.

Centroid is orthogonal to the Circle/Annulus axis. Concrete implementations
dispatch on the runtime type of MTLResult.intersection (BaseGeometry vs
list[Coord]).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Optional

from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.types import Coord, Error


@dataclass(frozen=True)
class CTRResult:
    """Outcome of selecting a single point from a feasible region.

    `method` is auto-stamped by CTRMethod.select_centroid with the concrete
    class name."""

    success: bool
    error: Optional[Error] = None
    tg_coord: Optional[Coord] = None
    method: Optional[str] = None


class CTRMethod(ABC):
    """Abstract centroid selection method.

    Subclasses implement `_select_centroid`; the public `select_centroid`
    wrapper stamps `method=type(self).__name__` onto the returned result."""

    @abstractmethod
    def _select_centroid(self, mtl: MTLResult) -> CTRResult: ...

    def select_centroid(self, mtl: MTLResult) -> CTRResult:
        return replace(self._select_centroid(mtl), method=type(self).__name__)
