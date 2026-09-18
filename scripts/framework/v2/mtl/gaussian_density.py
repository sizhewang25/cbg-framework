"""GaussianDensityMTL — Spotter's joint probability surface (Laki et al. 2011, §III).

The faithful reading of the paper's multilateration step, as opposed to the
annular-intersection stack `spotter_cbg` currently borrows from Octant. See
notes/2026-09-18-spotter-mtl-fidelity-gap.md for why that stack is not Spotter.

## What the paper actually asks for

Eq. (2) makes the target's position a *product* of per-landmark densities, each
`g_i` isotropic in the great-circle distance `s_i` from landmark `i`, with
`f_d = N(mu(d), sigma(d)^2)`. Taking logs at a candidate position `x`:

    log P(x) = Σ_i [ −log σ_i − (s_i(x) − µ_i)² / (2 σ_i²) ] + const

`Σ log σ_i` depends only on the measured RTTs, which are fixed for one target,
so it is constant in `x` and drops out. What is left is the whole objective:

    log P(x) = −½ Σ_i z_i²,     z_i = (s_i(x) − µ_i) / σ_i

Three properties of this that the annular stack does not have, and that are the
reason this class exists:

* **No constraint can veto.** A badly-fitting landmark lowers a score; it cannot
  empty the answer. `EMPTY_REGION` and `EXCLUSIVE_REGION` are not outcomes this
  method can produce, which is the point — §IV-C's claim is precisely that
  Spotter is "less prone to measurement errors" than CBG's and Octant's strict
  constraints.
* **Near-landmark space is low density, not excluded.** The annular path
  subtracts inner disks; here `s_i << µ_i` is simply an unlikely position.
* **σ is load-bearing.** A tight landmark dominates via `1/σ_i`. Counting or
  intersecting bands discards exactly this, which is the whole reason the LTD
  fits `σ(d)` at all.

## Discretisation

§V-A-2 divides the globe into HTM cells and evaluates (2) per cell. We
substitute H3, which the v3 analysis tree is already keyed on, so a density
surface and an answer-space region are quantised the same way.

Evaluated **coarse-to-fine**: a global pass at `coarse_resolution`, then repeated
descent into the children of the top-scoring cells until `resolution`. The
paper sanctions this directly — §V-A-2 notes that "the hierarchical structure of
HTM provides flexibility in choosing the cell size", which is what a quadtree
is for. It also keeps the cost sane: a single global H3-4 pass is 288,122 cells,
while `coarse_resolution=2` is 5,882 followed by two descents over a few hundred.

**The pruning is the one place this is an approximation, and it is recorded as
one.** `log_density` then covers the refined neighbourhood rather than the
globe, so `DensityField.probabilities()` normalises over retained cells and will
overstate confidence if a genuine secondary mode was dropped. Two mitigations,
both on by default: the top `top_k` cells are expanded by `neighbor_ring` before
descending, so a mode sitting just across a cell boundary survives; and setting
`coarse_resolution >= resolution` switches to a single global pass with no
pruning at all, which is the honest setting when a full surface is wanted.

## Output

`MTLResult.density` carries the field. `MTLResult.intersection` carries the
credible region — the smallest set of cells whose normalised mass reaches
`credible_mass` — as cell centres, which is §III-B's "union of the most probable
cells according to a required confidence level". Reusing the `list[Coord]` shape
means a density-blind CTR (`boundary_vertex_mean`) still degrades to a mean over
those centres instead of crashing; the point estimate the paper describes comes
from a density-aware CTR (`density_argmax`, `density_mean`, `density_mle`).
"""

from __future__ import annotations

import math
from typing import Optional

import h3
import numpy as np

from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import DensityField, DensityMTLMethod, MTLResult
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Coord, Error, VpId

EARTH_RADIUS_KM = 6371.0

#: Below this many distribution-carrying constraints the surface is not worth
#: evaluating: one Gaussian ring is isotropic (the paper says so in §III-A), and
#: two leave a two-point ambiguity. Three is the first count that can pin a
#: position, matching `INSUFFICIENT_CONSTRAINTS` elsewhere in the framework.
MIN_CONSTRAINTS = 3


def _haversine_km_to_many(
    lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Great-circle distance from one point to an array of points, in km.

    `arcsin(sqrt(...))` rather than `arctan2`: the arcsin form is clamped below
    against the rounding that can push the radicand a hair above 1 for
    antipodal pairs, which would otherwise return NaN and poison the whole
    accumulated z-sum for that cell.
    """
    p0, l0 = math.radians(lat0), math.radians(lon0)
    p1, l1 = np.radians(lats), np.radians(lons)
    dp, dl = p1 - p0, l1 - l0
    a = np.sin(dp / 2.0) ** 2 + math.cos(p0) * np.cos(p1) * np.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def _global_cells(resolution: int) -> list[str]:
    """Every H3 cell at `resolution`, via the 122 base cells."""
    if resolution == 0:
        return list(h3.get_res0_cells())
    out: list[str] = []
    for base in h3.get_res0_cells():
        out.extend(h3.cell_to_children(base, resolution))
    return out


def _cell_coords(cells: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ll = [h3.cell_to_latlng(c) for c in cells]
    return (
        np.fromiter((x[0] for x in ll), dtype=float, count=len(ll)),
        np.fromiter((x[1] for x in ll), dtype=float, count=len(ll)),
    )


def _log_density(
    lats: np.ndarray,
    lons: np.ndarray,
    constraints: list[tuple[Coord, float, float]],
) -> np.ndarray:
    """−½ Σ z² over cells, accumulated one constraint at a time.

    Accumulating in place rather than building an (n_cells × n_vps) matrix: a
    global H3-4 pass with ~130 VPs would be a 300 MB intermediate, and nothing
    here needs the per-constraint residuals after they are summed.
    """
    acc = np.zeros(lats.shape, dtype=float)
    for coord, mu, sigma in constraints:
        s = _haversine_km_to_many(coord.lat, coord.lon, lats, lons)
        z = (s - mu) / sigma
        acc += z * z
    return -0.5 * acc


@register_mtl("gaussian_density")
class GaussianDensityMTL(DensityMTLMethod):
    """Spotter's joint density surface over an H3 grid.

    Args:
        resolution: H3 resolution the field is reported at.
        coarse_resolution: resolution of the global first pass. Set it equal to
            (or above) `resolution` for a single global pass with no pruning.
        top_k: cells carried forward from each level before descending.
        neighbor_ring: rings of H3 neighbours added around each carried cell, so
            a mode just across a cell boundary is not pruned. 0 disables.
        credible_mass: mass the reported credible region must reach.
    """

    def __init__(
        self,
        resolution: int = 4,
        coarse_resolution: int = 2,
        top_k: int = 64,
        neighbor_ring: int = 1,
        credible_mass: float = 0.95,
    ) -> None:
        if not 0 <= resolution <= 15:
            raise ValueError(f"resolution must be in [0, 15], got {resolution}")
        if not 0 <= coarse_resolution <= 15:
            raise ValueError(
                f"coarse_resolution must be in [0, 15], got {coarse_resolution}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if neighbor_ring < 0:
            raise ValueError(f"neighbor_ring must be >= 0, got {neighbor_ring}")
        if not 0.0 < credible_mass <= 1.0:
            raise ValueError(
                f"credible_mass must be in (0, 1], got {credible_mass}"
            )
        self.resolution = resolution
        self.coarse_resolution = coarse_resolution
        self.top_k = top_k
        self.neighbor_ring = neighbor_ring
        self.credible_mass = credible_mass

    def _multilaterate(self, results: list[LTDResult]) -> MTLResult:
        constraints: list[tuple[Coord, float, float]] = []
        participating: list[VpId] = []
        for r in results:
            d = r.tg_distance
            if r.vp_coord is None or d is None or not d.has_distribution:
                continue
            constraints.append((r.vp_coord, float(d.mu_km), float(d.sigma_km)))
            participating.append(r.vp_id)

        if len(constraints) < MIN_CONSTRAINTS:
            # Distinguishes "the LTD carries no distribution" (a composition
            # error — a bounds-only LTD was paired with a density MTL) from
            # "too few VPs answered", because the fix differs.
            error = (
                Error.INSUFFICIENT_DATA
                if not constraints
                else Error.INSUFFICIENT_CONSTRAINTS
            )
            return MTLResult(
                success=False,
                error=error,
                participating_vp_ids=tuple(participating),
            )

        cells, logp = self._evaluate(constraints)
        if cells is None or logp is None or not len(cells):
            return MTLResult(
                success=False,
                error=Error.NUMERICAL_FAILURE,
                participating_vp_ids=tuple(participating),
            )

        lats, lons = _cell_coords(cells)
        field = DensityField(
            cells=tuple(Coord(float(a), float(b)) for a, b in zip(lats, lons)),
            log_density=tuple(float(v) for v in logp),
            constraints=tuple(constraints),
            grid="h3",
            resolution=self.resolution,
        )
        return MTLResult(
            success=True,
            intersection=self._credible_region(field),
            density=field,
            participating_vp_ids=tuple(participating),
        )

    def _evaluate(
        self, constraints: list[tuple[Coord, float, float]]
    ) -> tuple[Optional[list[str]], Optional[np.ndarray]]:
        """Coarse global pass, then descend into the best cells' children."""
        start = min(self.coarse_resolution, self.resolution)
        cells = _global_cells(start)
        lats, lons = _cell_coords(cells)
        logp = _log_density(lats, lons, constraints)

        res = start
        while res < self.resolution:
            order = np.argsort(logp)[::-1][: self.top_k]
            carried: set[str] = set()
            for i in order:
                cell = cells[int(i)]
                if self.neighbor_ring > 0:
                    carried.update(h3.grid_disk(cell, self.neighbor_ring))
                else:
                    carried.add(cell)
            children: list[str] = []
            for cell in carried:
                children.extend(h3.cell_to_children(cell, res + 1))
            if not children:
                # Nothing to descend into: report the level actually reached
                # rather than an empty field.
                break
            cells = children
            lats, lons = _cell_coords(cells)
            logp = _log_density(lats, lons, constraints)
            res += 1

        finite = np.isfinite(logp)
        if not finite.any():
            return None, None
        if not finite.all():
            cells = [c for c, ok in zip(cells, finite) if ok]
            logp = logp[finite]
        return cells, logp

    def _credible_region(self, field: DensityField) -> list[Coord]:
        """Cell centres of the smallest set reaching `credible_mass`.

        Always at least the top cell, so a posterior sharp enough that one cell
        carries the whole mass still yields a region rather than an empty list.
        """
        probs = field.probabilities()
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        picked: list[Coord] = []
        cumulative = 0.0
        for i in order:
            picked.append(field.cells[i])
            cumulative += probs[i]
            if cumulative >= self.credible_mass:
                break
        return picked
