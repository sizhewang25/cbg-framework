"""Centroid selection interface.

Centroid is orthogonal to the Circle/Annulus axis. Concrete implementations
dispatch on the runtime type of MTLResult.intersection (BaseGeometry vs
list[Coord]).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.types import Coord, Error


@dataclass(frozen=True)
class CTRResult:
    """Outcome of selecting a single point from a feasible region."""

    success: bool
    error: Optional[Error] = None
    tg_coord: Optional[Coord] = None


class CTRMethod(ABC):
    """Abstract centroid selection method."""

    @abstractmethod
    def select_centroid(self, mtl: MTLResult) -> CTRResult: ...
