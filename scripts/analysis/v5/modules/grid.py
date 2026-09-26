"""The grid: HEALPix, NESTED, at one resolution. Ported from v4's `healpix.py`.

In v5's vocabulary a **grid** is one HEALPix pixel (and, by extension, the
partition those pixels form). v4 called these "cells"; v5 reserves *cell* for
the Voronoi cell of the seeds, so nothing in this module says cell.

Why HEALPix and not H3: exactly equal-area, so a grid count converts to an
area and the quantisation floor is the same everywhere; aperture-4 with exact
nesting, so in NESTED ordering the parent of `pix` is `pix >> 2`. The nesting
property is what made `ring0` monotone across resolutions where H3
produced 705 violations on this repo's data. **v5 now runs one resolution**,
so that monotonicity is no longer exercised here -- the equal-area argument is
what still carries weight.

`grid_km` is the **nominal grid distance**, `sqrt(area)`: 50.9 km at nside
128. It is one number with two jobs -- the grid pitch, and the
complete-linkage diameter that groups sites into seeds -- so that both
partitions of the answer space are built at one tolerance.
"""

from __future__ import annotations

import warnings

import numpy as np

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

#: Working resolution: 2,594 km^2 grids, 50.9 km nominal.
DEFAULT_NSIDE = 128

#: The resolutions v5 builds. One rung: the cell partition is the granularity
#: dial now, not a ladder of grids.
NSIDE_LADDER: tuple[int, ...] = (128,)

#: Fixed, not a parameter: NESTED is what makes a parent a bit shift away, and
#: `ang2pix`/`neighbours` are written against it.
_ORDER = "nested"

def _healpix(nside: int):
    # Lazy: astropy is a heavy import and the areas are closed-form.
    from astropy_healpix import HEALPix

    return HEALPix(nside=nside, order=_ORDER)


def validate_nside(nside: int) -> int:
    """`nside` must be a positive power of two for NESTED ids to nest."""
    n = int(nside)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"nside must be a positive power of two, got {nside}")
    return n


def n_grids(nside: int) -> int:
    """Total grids: `12 * nside**2` (196,608 at nside=128)."""
    return 12 * validate_nside(nside) ** 2


def grid_area_km2(nside: int) -> float:
    """Grid area in km^2 -- exact and identical for every grid at this nside."""
    return 4.0 * np.pi * EARTH_RADIUS_KM**2 / n_grids(nside)


def grid_km(nside: int) -> float:
    """Nominal grid distance, `sqrt(area)` -- 50.9 km at nside=128."""
    return float(np.sqrt(grid_area_km2(nside)))


def ang2pix(lat_deg, lon_deg, nside: int = DEFAULT_NSIDE) -> np.ndarray:
    """NESTED grid id per `(lat, lon)` in degrees."""
    import astropy.units as u

    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    pix = _healpix(validate_nside(nside)).lonlat_to_healpix(lon * u.deg, lat * u.deg)
    return np.asarray(pix, dtype=np.int64)


def pix2ang(pix, nside: int = DEFAULT_NSIDE) -> np.ndarray:
    """Grid centres as an `(N, 2)` array of `(lat, lon)` degrees.

    Longitude is normalised to `[-180, 180)`: astropy returns `[0, 360)`, so a
    Chicago grid would otherwise arrive as 271.77 rather than -88.23.
    """
    p = np.asarray(pix, dtype=np.int64).ravel()
    if p.size == 0:
        return np.zeros((0, 2), dtype=float)
    lon, lat = _healpix(validate_nside(nside)).healpix_to_lonlat(p)
    lon = np.asarray(lon.to_value("deg"), dtype=float)
    lat = np.asarray(lat.to_value("deg"), dtype=float)
    return np.column_stack([lat, ((lon + 180.0) % 360.0) - 180.0])


def neighbours(pix, nside: int) -> np.ndarray:
    """The 8 neighbours of each grid, as an `(N, 8)` array.

    Absent neighbours are `-1`: 24 grids at every nside sit at a corner of the
    12-face base tessellation and have 7. `astropy_healpix` warns on them,
    which is noise here, so the warning is suppressed.
    """
    from astropy_healpix import neighbours as _nb

    p = np.asarray(pix, dtype=np.int64).ravel()
    if p.size == 0:
        return np.zeros((0, 8), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*invalid value.*neighbours.*")
        out = _nb(p, validate_nside(nside), order=_ORDER)
    return np.asarray(out, dtype=np.int64).T


#: Neighbour tables, one per nside, built on first use. The table is
#: `n_grids(nside)` by 8 int32 -- 6.3 MB at nside 128 -- and every
#: `grid_offset` call reuses it, which is the only thing here worth keeping.
_NEIGHBOUR_TABLES: dict[int, np.ndarray] = {}


def _neighbour_table(nside: int) -> np.ndarray:
    """Every grid's neighbours at this nside, as one `(n_grids, 8)` array."""
    tbl = _NEIGHBOUR_TABLES.get(nside)
    if tbl is None:
        tbl = neighbours(np.arange(n_grids(nside), dtype=np.int64), nside).astype(np.int32)
        _NEIGHBOUR_TABLES[nside] = tbl
    return tbl


def grid_offset(a, b, nside: int) -> np.ndarray:
    """Exact grid steps separating each `(a, b)` pair. Never gives up.

    `grid_offset == k` means k grids out, for any k: the whole grid axis, at
    the resolution the grid itself provides. `classify` is what decides where
    to stop grading, banding this into `ring0` / `ring1` / `ring2` / `beyond`;
    that cap belongs to the metric, not here.

    `-1` is unreachable for a real pair: the neighbour graph is connected --
    at nside 128 its diameter is 315 -- so every pair resolves. A caller that
    sees `-1` passed something that was never a grid id.

    Breadth-first, grown over one shared neighbour table per **distinct `a`**
    rather than per pair. Per pair is what an earlier capped version did, and
    it does not survive the cap coming off: at offset 68 the disk a pair
    sweeps is some 15,000 grids. Each source stops as soon as it has reached
    every `b` asked of it.
    """
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a and b must be the same length, got {a.shape} vs {b.shape}")
    nside = validate_nside(nside)
    out = np.full(a.shape, -1, dtype=np.int64)
    if a.size == 0:
        return out

    tbl = _neighbour_table(nside)
    # One `dist` is live at a time and it is not cached across sources, so
    # memory stays flat however many distinct `a` a caller passes.
    for src in np.unique(a):
        rows = np.flatnonzero(a == src)
        wanted = b[rows]
        dist = np.full(tbl.shape[0], -1, dtype=np.int32)
        dist[src] = 0
        frontier = np.array([src], dtype=np.int64)
        k = 0
        while frontier.size and (dist[wanted] < 0).any():
            k += 1
            nbrs = np.unique(tbl[frontier].ravel())
            nbrs = nbrs[nbrs >= 0]
            nbrs = nbrs[dist[nbrs] < 0]
            dist[nbrs] = k
            frontier = nbrs
        out[rows] = dist[wanted]
    return out


def ring_grids(pix: int, nside: int, max_ring: int) -> list[list[int]]:
    """The neighbourhood of one grid, split by ring: `[[pix], ring 1, ring 2]`.

    Same breadth-first growth as `grid_offset`, the same treatment of the
    absent neighbours `neighbours` reports as `-1`, and the same guarantee
    that the rings are disjoint -- a grid already reached at ring `k-1` is not
    re-listed at ring `k`.

    The **set** form of `grid_offset`: that answers "how far is b from a",
    this answers "which grids are k steps out from a". `test_grid`
    cross-checks the two -- every grid this returns at ring `k` must make
    `grid_offset` answer `k`.

    `max_ring` is **required**, with no default. Where to stop is a property
    of the metric and lives in `classify.MAX_RING`, not here; grid.py cannot
    import classify without a cycle, and a local default would be a second
    copy of a number that must not drift.

    **Only ever called with a small `max_ring`.** It grows a disk, so the cost
    is quadratic in the radius: at offset 68 -- a real `grid_offset` value on
    these runs -- the disk is some 15,000 grids. `map_mtl` draws the
    neighbourhood at `classify.MAX_RING` and reports anything further as a
    number rather than a region, which is why that is not a problem here.

    Ring 1 holds **7** rather than 8 grids at the 24 base-face corner grids of
    every nside, and ring 2 is correspondingly short. Callers must read the
    lengths rather than assume 8 and 16.
    """
    n = validate_nside(nside)
    seed = int(pix)
    rings: list[list[int]] = [[seed]]
    if max_ring < 1:
        return rings

    reached = {seed}
    shell = {seed}
    for _ in range(1, int(max_ring) + 1):
        if not shell:
            rings.append([])
            continue
        block = neighbours(np.fromiter(shell, dtype=np.int64), n).ravel()
        new = {int(c) for c in block if c >= 0} - reached
        reached |= new
        shell = new
        rings.append(sorted(new))
    return rings

def grid_rings(pix, nside: int, *, step: int = 8) -> list[np.ndarray]:
    """One `(V, 2)` `(lon, lat)`-degree boundary ring per grid, for drawing.

    **`(lon, lat)`, the opposite order from `pix2ang`**, because that is what
    plotting wants. `step` points per edge so curvature shows. Each ring is made
    contiguous in longitude so a grid straddling the antimeridian does not smear
    across the map. Ported from v4's `cell_rings`.
    """
    p = np.asarray(pix, dtype=np.int64).ravel()
    if p.size == 0:
        return []
    lon, lat = _healpix(validate_nside(nside)).boundaries_lonlat(p, step=step)
    lon = np.asarray(lon.to_value("deg"), dtype=float)
    lat = np.asarray(lat.to_value("deg"), dtype=float)
    rings = []
    for i in range(p.size):
        lo = ((lon[i] + 180.0) % 360.0) - 180.0
        lo = lo[0] + ((lo - lo[0] + 180.0) % 360.0) - 180.0
        rings.append(np.column_stack([lo, lat[i]]))
    return rings


def describe(nside: int) -> dict:
    """Static facts about the grid at this nside, for a `meta.json` block."""
    n = validate_nside(nside)
    return {
        "scheme": "healpix",
        "nside": n,
        "order": _ORDER,
        "n_grids": n_grids(n),
        "grid_area_km2": round(grid_area_km2(n), 3),
        "grid_km": round(grid_km(n), 3),
        "grid_km_note": "sqrt(area); HEALPix grids are exactly equal-area",
    }
