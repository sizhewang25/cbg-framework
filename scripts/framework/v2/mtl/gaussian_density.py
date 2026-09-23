"""GaussianDensityMTL — Spotter's joint probability surface (Laki et al. 2011, §III).

The faithful reading of the paper's multilateration step, as opposed to the
annular-intersection stack borrowed from Octant. On the as0* configs this is
what `spotter_cbg` now composes; the Octant-geometry version it replaced is
still scored alongside it as `spotter_hybrid_cbg`. See
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

§V-A-2 divides the globe into HTM cells and evaluates (2) per cell. We use
**HEALPix**, which shares HTM's two load-bearing properties — aperture-4
subdivision, with the four children exactly tiling the parent — and adds exactly
equal-area cells, which HTM does not have (its cells vary by 110% at every
level, against HEALPix's 0%).

This replaced H3, and the reason is not aesthetic. H3 is aperture-7, hexagons
cannot tile hexagons, and a parent's six outer children straddle its boundary:
measured over 400,000 random points, **7.12% land in an H3 res-3 cell that is
not a child of their own res-2 cell**. So the coarse-to-fine descent below had
holes at every level, and `neighbor_ring` was *patching* that rather than
providing the margin it is documented as. Under HEALPix the descent's candidate
set is exactly the refinement of the cells it carried.

Evaluated **coarse-to-fine**: a global pass at `coarse_resolution`, then
repeated descent into the children of the top-scoring cells until `resolution`.
The paper sanctions this directly — §V-A-2 notes that "the hierarchical
structure of HTM provides flexibility in choosing the cell size", which is what
a quadtree is for. It also keeps the cost sane: a single global nside-128 pass
is 196,608 cells at ~1,085 ms per target, against 3,072 at `coarse_resolution=16`
plus ~104 cells per descent level, which measures at ~27.5 ms.

**The pruning is the one place this is an approximation, and it is recorded as
one.** `log_density` covers the refined neighbourhood rather than the globe, so
a genuine secondary mode dropped at a coarse level is gone. Two mitigations,
both on by default: the top `top_k` cells are expanded by `neighbor_ring` before
descending, so a mode sitting just across a cell boundary survives; and setting
`coarse_resolution >= resolution` switches to a single global pass with no
pruning at all, which is the honest setting when a full surface is wanted.
`top_k` and `neighbor_ring` are not guesses — `scripts/benchmark/v2/cli.py
mtl-basin-miss` sweeps them against an exhaustive global pass, and the shipped
values are a zero-miss row.

## Output

`MTLResult.density` carries the field. `MTLResult.intersection` carries the same
cells as coordinates, so a density-blind CTR (`boundary_vertex_mean`) degrades
to a mean over them instead of crashing; the point estimate the paper describes
comes from a density-aware CTR (`density_argmax`).

It used to carry a `credible_mass` sub-region instead — §III-B's "union of the
most probable cells according to a required confidence level". That was removed
because it could not mean what it said. The mass is normalised over the retained
cells, not the globe, so pruning made "95%" a statement about a neighbourhood;
and `credible_mass = 1.0` did not even mean "all of them", because
`probabilities()` max-shifts before exponentiating and the partial sum reaches
1.0 in float64 after a handful of cells — measured, a sharp posterior over 104
cells selected **1**. Nothing read the region either: `density_argmax` takes the
maximum of `log_density` directly. Reporting the retained field, whose exact
area is `len(cells) * pixel_area_km2(resolution)` because the cells are
equal-area, is the honest version of what was there.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import DensityField, DensityMTLMethod, MTLResult
from scripts.framework.v2.registry import register_mtl
from scripts.framework.v2.types import Coord, Error, VpId
from scripts.libs.healpix import grid as HP

#: Imported from the grid rather than redeclared, so the tessellation and the
#: haversine that measures distances across it cannot disagree.
EARTH_RADIUS_KM = HP.EARTH_RADIUS_KM

#: The only accepted `grid`. A name rather than a bare flag because the config
#: should state which tessellation produced a number, and because adding HTM
#: later should be a new branch here rather than a silent change of meaning.
GRID_HEALPIX = "healpix"

#: Below this many distribution-carrying constraints the surface is not worth
#: evaluating: one Gaussian ring is isotropic (the paper says so in §III-A), and
#: two leave a two-point ambiguity. Three is the first count that can pin a
#: position, matching `INSUFFICIENT_CONSTRAINTS` elsewhere in the framework.
MIN_CONSTRAINTS = 3

#: Largest nside either resolution may name.
#:
#: A budget, not a grid limit — HEALPix itself goes far higher. The global grid
#: is built **eagerly in `__init__`**, i.e. inside `CBGModel.from_config`,
#: before any fold output exists, so an over-large value is an OOM with no
#: artifacts and no instrumentation to attribute it. At 24 bytes per cell (an
#: int64 id plus two float64 coordinates) nside 512 is 3.1 M cells and ~75 MB;
#: nside 1024 would be 12.6 M and ~302 MB. Both are plausible typos for 128,
#: and both pass `validate_nside` since every power of two does.
MAX_NSIDE = 512

#: Bytes per cell in a cached global grid — one int64 id, two float64 degrees.
_BYTES_PER_CELL = 24


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


#: Global first-pass grids, keyed by `(scheme, nside)`, shared across instances.
#:
#: Process-lifetime and bounded: a run uses one or two resolutions, and the
#: largest plausible entry (nside 128, 196,608 cells) is 4.7 MB. Shared rather
#: than per-instance so that constructing many models -- the test suite, a sweep
#: over combos -- pays the build once.
#:
#: The key carries the scheme even though there is only one. The grid's identity
#: is `(scheme, nside, order)`, and a bare nside is sufficient only because the
#: order is a module constant in `scripts.libs.healpix.grid`. Naming the scheme
#: keeps that from being an invisible assumption if a second one is ever added.
_GLOBAL_GRID_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _global_grid(nside: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The global grid at `nside` as (cells, lats, lons), memoised.

    All three arrays are marked read-only. They are shared by every instance
    and every target, so an in-place write would not corrupt one result -- it
    would silently corrupt all of them. Nothing writes to them today; this
    makes a future attempt fail loudly instead.
    """
    key = (GRID_HEALPIX, HP.validate_nside(nside))
    hit = _GLOBAL_GRID_CACHE.get(key)
    if hit is None:
        cells = np.arange(HP.npix(nside), dtype=np.int64)
        centres = HP.pix2ang(cells, nside)
        lats = np.ascontiguousarray(centres[:, 0])
        lons = np.ascontiguousarray(centres[:, 1])
        for arr in (cells, lats, lons):
            arr.flags.writeable = False
        hit = (cells, lats, lons)
        _GLOBAL_GRID_CACHE[key] = hit
    return hit


def _log_density(
    lats: np.ndarray,
    lons: np.ndarray,
    constraints: list[tuple[Coord, float, float]],
) -> np.ndarray:
    """−½ Σ z² over candidate positions (lats, lons), one constraint at a time.

    The candidates are HEALPix cell centres; the constraints come from vantage
    points.

    Accumulating in place rather than building an (n_cells × n_vps) matrix: a
    global nside-128 pass with ~130 VPs would be a 204 MB intermediate, and
    nothing here needs the per-constraint residuals after they are summed.
    """
    acc = np.zeros(lats.shape, dtype=float)
    for coord, mu, sigma in constraints:
        # s is a candidate-to-landmark distance set.
        s = _haversine_km_to_many(coord.lat, coord.lon, lats, lons)
        z = (s - mu) / sigma
        acc += z * z
    return -0.5 * acc


@register_mtl("gaussian_density")
class GaussianDensityMTL(DensityMTLMethod):
    """Spotter's joint density surface over a HEALPix grid.

    Args:
        grid: Tessellation name. Required, and only `"healpix"` is accepted.
            Deliberately has no default: the combos that ran on H3 stored
            `resolution: 4` and no `grid`, and 4 is a legal nside, so a default
            would let a stale payload replay as a 192-cell globe rather than
            failing. See the class docstring's Discretisation section.
        resolution: nside the field is reported at. 128 is 196,608 cells,
            50.9 km nominal.
        coarse_resolution: nside of the global first pass. Set it equal to (or
            above) `resolution` for a single global pass with no pruning.
        top_k: cells carried forward from each level before descending.
        neighbor_ring: rings of HEALPix neighbours added around each carried
            cell, so a mode just across a cell boundary is not pruned. 0
            disables. Validated together with `top_k` by `mtl-basin-miss`.
    """

    def __init__(
        self,
        *,
        grid: str,
        resolution: int = HP.DEFAULT_NSIDE,
        coarse_resolution: int = 16,
        top_k: int = 8,
        neighbor_ring: int = 1,
    ) -> None:
        if grid != GRID_HEALPIX:
            raise ValueError(
                f"grid must be {GRID_HEALPIX!r}, got {grid!r}. H3 was removed: "
                f"its aperture-7 subdivision does not nest geometrically, so the "
                f"coarse-to-fine descent below lost 7.12% of candidate cells at "
                f"every level."
            )
        self.grid = grid
        self.resolution = self._validate_nside("resolution", resolution)
        self.coarse_resolution = self._validate_nside(
            "coarse_resolution", coarse_resolution
        )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if neighbor_ring < 0:
            raise ValueError(f"neighbor_ring must be >= 0, got {neighbor_ring}")
        self.top_k = top_k
        self.neighbor_ring = neighbor_ring
        # Built EAGERLY, and that is the whole point -- see `_coarse_grid`.
        self._coarse = _global_grid(min(coarse_resolution, resolution))

    @staticmethod
    def _validate_nside(name: str, value: int) -> int:
        """A power of two, at or below `MAX_NSIDE`."""
        try:
            nside = HP.validate_nside(value)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        if nside > MAX_NSIDE:
            mb = HP.npix(nside) * _BYTES_PER_CELL / 1e6
            raise ValueError(
                f"{name} must be <= {MAX_NSIDE}, got {nside}: its global grid is "
                f"{HP.npix(nside):,} cells (~{mb:.0f} MB), built eagerly before "
                f"any output is written"
            )
        return nside

    @property
    def cell_area_km2(self) -> float:
        """Area of one reported cell. Exact, because the cells are equal-area."""
        return HP.pixel_area_km2(self.resolution)

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

        lats, lons, logp = self._evaluate(constraints)
        if lats is None or logp is None or not len(lats):
            return MTLResult(
                success=False,
                error=Error.NUMERICAL_FAILURE,
                participating_vp_ids=tuple(participating),
            )

        field = DensityField(
            cells=tuple(Coord(float(a), float(b)) for a, b in zip(lats, lons)),
            log_density=tuple(float(v) for v in logp),
            grid=GRID_HEALPIX,
            resolution=self.resolution,
        )
        return MTLResult(
            success=True,
            intersection=list(field.cells),
            density=field,
            participating_vp_ids=tuple(participating),
        )

    def _coarse_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The global first-pass grid. Built in `__init__`, never here.

        It depends only on `min(coarse_resolution, resolution)`, both fixed at
        construction, so building it per target would be pure waste.

        **Built eagerly specifically so it stays out of the instrumentation**,
        and that reason survived the move to HEALPix even though its original
        numbers did not. Under H3 the argument was cost: 346 ms to build the
        res-4 grid, which a lazy build would have charged to the first target's
        `cm("mtl")` block. `pix2ang` is vectorised, so the same grid is now
        27 ms and the default `coarse_resolution=16` is 3,072 cells in well
        under a millisecond.

        What replaces it is the **import**. `astropy_healpix` costs ~26 MB of
        RSS and ~0.33 s, paid the first time `_global_grid` runs. `runner.py`
        constructs the model between `rss_after_inputs` and
        `measure_block("fit")`, so eager keeps that out of every per-stage
        column: those are deltas against a baseline sampled inside the block
        (`instrument.py`), so `fit_*` and all `{ltd,mtl,ctr}_*_peak_bytes` are
        unaffected. Only the absolute marks `rss_after_fit_bytes` and
        `run_peak_rss_bytes` carry it.

        Do **not** move the astropy import to module scope to "fix" that.
        `framework/v2/__init__.py` imports this module, so every combo would
        pay the 26 MB and no previously collected run would remain comparable.

        The per-instance construction cost is amortised by `_global_grid`'s
        module-level cache, so a sweep or a test suite that builds many models
        pays for each distinct nside once.
        """
        return self._coarse

    def _evaluate(
        self, constraints: list[tuple[Coord, float, float]]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Coarse global pass, then descend into the best cells' children.

        Returns `(lats, lons, log_density)` rather than pixel ids: the caller
        only ever wanted the coordinates, and returning the ids made it
        recompute the centres -- a full duplicate of the most expensive step,
        per target.

        **The nside advances with the cells, and nothing can check that it
        did.** Every HEALPix id is a legal id at some nside, so calling
        `pix2ang` or `disk` with the wrong one yields centres that are silently
        displaced rather than an exception, which `error_km` then absorbs. H3's
        opaque string ids raised instead. `test_gaussian_density.py` pins the
        invariant that each descended cell degrades back into the set it came
        from.
        """
        nside = min(self.coarse_resolution, self.resolution)
        cells, lats, lons = self._coarse_grid()
        logp = _log_density(lats, lons, constraints)

        while nside < self.resolution:
            top = cells[np.argsort(logp)[::-1][: self.top_k]]
            carried = (
                HP.disk(top, self.neighbor_ring, nside)
                if self.neighbor_ring > 0
                else np.unique(top)
            )
            # `disk` already returns unique cells, so overlapping rings need no
            # deduplication here, and `children` of a non-empty set is never
            # empty -- there is no "nothing to descend into" case to guard.
            cells = HP.children(carried).ravel()
            nside *= 2
            centres = HP.pix2ang(cells, nside)
            lats, lons = centres[:, 0], centres[:, 1]
            logp = _log_density(lats, lons, constraints)

        finite = np.isfinite(logp)
        if not finite.any():
            return None, None, None
        if not finite.all():
            # Fancy indexing copies, so the cached coarse arrays are never
            # aliased into the returned field.
            lats, lons, logp = lats[finite], lons[finite], logp[finite]
        return lats, lons, logp
