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
from scripts.framework.v2.types import Coord, Error, VpId

# Output region representation. Planar methods (e.g. Shapely-based) return
# a Polygon / MultiPolygon. Spherical methods return a list of (lat, lon)
# vertices on the unit sphere. Density methods return the credible region as a
# list of cell centres, which reuses the same `list[Coord]` shape so existing
# coord-consuming CTRs keep working (in a degraded, density-blind way).
# Centroid methods dispatch on the runtime type.
Intersection = Union[BaseGeometry, list[Coord], None]


@dataclass(frozen=True)
class DensityField:
    """A discretised posterior over target position, plus the constraints for it.

    The output of a DensityMTLMethod, and the thing that makes Spotter's
    §III-B point-estimate options (argmax / distribution mean / region centre)
    expressible as CTRs rather than baked into the MTL.

    `log_density` is **unnormalised** and carries an arbitrary additive
    constant: with Gaussian f_d the `Σ log σᵢ` term of Eq. (2) depends only on
    the measured RTTs, which are fixed for one target, so it is constant across
    cells and dropped. Anything comparing cells (argmax, credible mass, a
    weighted mean) is unaffected; anything wanting absolute probability is not
    available and should not be invented from this.

    There is deliberately no per-VP constraint list. An earlier version carried
    `(coord, mu_km, sigma_km)` triples so a CTR could refine the grid argmax by
    continuous optimisation (`min Σ ((s_i(x) − µ_i)/σ_i)²`). That CTR was
    removed, and with only `density_argmax` consuming the field, the triples had
    no reader — a field nothing reads is a field that goes stale. Re-add it with
    the solver if the exact MAP is wanted again.
    """

    cells: tuple[Coord, ...]
    log_density: tuple[float, ...]
    grid: Optional[str] = None
    resolution: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.cells) != len(self.log_density):
            raise ValueError(
                f"cells ({len(self.cells)}) and log_density "
                f"({len(self.log_density)}) must be the same length"
            )

    def probabilities(self) -> list[float]:
        """Normalised cell probabilities, max-shifted before exponentiating.

        Subtracting the maximum is not cosmetic: with ~100 VPs the summed
        z-squared runs to thousands, and `exp` of that underflows every cell to
        zero, which turns a well-determined posterior into 0/0. Shifting makes
        the best cell exactly 1.0 and everything else a ratio to it, which is
        all any caller here needs.
        """
        import math

        if not self.log_density:
            return []
        top = max(self.log_density)
        weights = [math.exp(v - top) for v in self.log_density]
        total = sum(weights)
        if total <= 0:
            return [0.0] * len(weights)
        return [w / total for w in weights]


@dataclass(frozen=True)
class MTLResult:
    """Outcome of multilateration over a set of distance constraints.

    `method` is auto-stamped by MTLMethod.multilaterate with the concrete
    class name.

    `participating_vp_ids`, when set, lists the VPs whose constraints survived
    the method's redundant-disk filter and were actually fed to the
    intersection / feasible-region computation — i.e. the VPs that *decide* the
    region. None means the method did not record participation (older callers);
    an empty tuple means no constraint participated. Recorded on both success
    and failure so the deciding set is recoverable even when the region is
    empty."""

    success: bool
    error: Optional[Error] = None
    intersection: Intersection = None
    method: Optional[str] = None
    participating_vp_ids: Optional[tuple[VpId, ...]] = None
    # Set only by DensityMTLMethod. None for every geometric method, which is
    # what a density-aware CTR checks before refusing to run.
    density: Optional[DensityField] = None


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


class DensityMTLMethod(MTLMethod, ABC):
    """Evaluates a posterior over position instead of intersecting constraints.

    Reads `tg_distance.mu_km` / `.sigma_km` — so it pairs only with an LTD whose
    predictions carry a distribution (`Distance.has_distribution`), today
    NormalDistLTD alone. Pairing it with a bounds-only LTD is a configuration
    error and the concrete method reports INSUFFICIENT_DATA rather than
    inventing a sigma.

    Why this is a third family rather than an AnnulusMTLMethod with extra
    output: the annulus families answer "which points satisfy every
    constraint", a set question whose answer is a region, and one violated
    constraint empties it. This family answers "how plausible is each point",
    a ranking whose answer is a field, where a badly-fitting constraint lowers
    a score instead of vetoing it. Those are different objectives, they fail
    differently, and the type system should not let a caller assume one and
    get the other.

    Contract additions on top of MTLMethod:
      * `MTLResult.density` is populated on success.
      * `MTLResult.intersection` is the credible region as cell centres
        (`list[Coord]`), so density-blind CTRs still degrade rather than crash.
    """
