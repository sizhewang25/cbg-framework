"""HEALPix grid primitives, shared by the framework and the analysis layers.

The framework's density MTL and `analysis/v4`'s laddered accuracy metric both
quantise the globe the same way, and neither may import the other -- the
framework is a leaf with respect to analysis, a rule
`scripts/analysis/v3/tests/test_layering.py` states and enforces. So the grid
math lives here, in `scripts/libs/`, which both are already allowed to depend on.

## NESTED is not a parameter

In NESTED ordering the parent of `pix` is `pix >> 2` and its four children are
`pix << 2 | {0,1,2,3}`, and those four exactly tile the parent. That is the
whole reason this grid was chosen over H3: H3 is aperture-7, hexagons cannot
tile hexagons, and a parent's six outer children straddle its boundary.
Measured over 400,000 random points, **7.12% land in an H3 res-3 cell that is
not a child of their own res-2 cell** -- so a coarse-to-fine descent on H3 has
holes at every level, and a containment ladder is not even well-formed.

RING ordering would break `degrade` and `children` and therefore everything
built on them, so `_ORDER` is a module constant. Do not make it an argument
without also re-keying every cache whose identity currently assumes it.

## Why `EARTH_RADIUS_KM` is defined here

It is deliberately *not* imported from `scripts.libs.cbg.rtt_model`, which
defines the same constant. That module pulls in `scipy.optimize.linprog`, and
importing it for one float costs 59 MB of RSS and 0.34 s -- more than the
`astropy_healpix` import this module works to keep lazy. `gaussian_density.py`
imports the constant from here rather than keeping its own, so the grid and the
haversine that measures distances across it cannot disagree.
"""

from __future__ import annotations

import warnings

import numpy as np

#: Mean Earth radius. See the module docstring for why this is not imported.
EARTH_RADIUS_KM = 6371.0

#: Working resolution: 196,608 cells of 2,594 km^2, 50.9 km nominal.
DEFAULT_NSIDE = 128

#: Fixed, not a parameter -- see the module docstring.
_ORDER = "nested"


def _healpix(nside: int):
    # Lazy: astropy_healpix costs ~26 MB and ~0.33 s to import, and the areas,
    # the ladder and `children` are all closed-form. A caller that only needs
    # those should not pay for it.
    from astropy_healpix import HEALPix

    return HEALPix(nside=nside, order=_ORDER)


def validate_nside(nside: int) -> int:
    """`nside` must be a positive power of two for NESTED ids to nest.

    The `int()` cast comes **first**, deliberately. A YAML round-trip or a
    pandas column can hand this a `128.0`, and `nside & (nside - 1)` on a float
    raises `TypeError` -- an error about bitwise operands, which says nothing
    about what the caller did wrong.
    """
    n = int(nside)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"nside must be a positive power of two, got {nside!r}")
    return n


def npix(nside: int) -> int:
    """Total cells: `12 * nside**2` (196,608 at nside=128)."""
    return 12 * validate_nside(nside) ** 2


def pixel_area_km2(nside: int) -> float:
    """Cell area in km^2 -- exact and identical for every cell at this nside.

    The equality is the useful part: a cell *count* converts to an area without
    knowing which cells. H3 res-4 varies by 33% and HTM by 110%, so on those
    grids the same count means different things in different places.
    """
    return 4.0 * np.pi * EARTH_RADIUS_KM**2 / npix(nside)


def nominal_cell_km(nside: int) -> float:
    """Cell pitch as `sqrt(area)` -- 50.9 km at nside=128.

    A pitch, not a radius. The largest distance from a cell centre to its own
    boundary is roughly the half-diagonal, `sqrt(2)/2` of this, so a tolerance
    derived from quantisation alone should use that rather than the pitch.
    """
    return float(np.sqrt(pixel_area_km2(nside)))


def ang2pix(lat_deg, lon_deg, nside: int = DEFAULT_NSIDE) -> np.ndarray:
    """NESTED cell id per `(lat, lon)` in degrees."""
    import astropy.units as u

    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    pix = _healpix(validate_nside(nside)).lonlat_to_healpix(lon * u.deg, lat * u.deg)
    return np.asarray(pix, dtype=np.int64)


def pix2ang(pix, nside: int = DEFAULT_NSIDE) -> np.ndarray:
    """Cell centres as an `(N, 2)` array of `(lat, lon)` degrees.

    **`(lat, lon)`, the opposite order from `cell_rings`.** Coordinates are
    written to disk as lat-then-lon, while plotting wants lon-then-lat. Two
    orders in one module is a swap-bug magnet, and the swap is silent --
    haversine goes through sin/cos and returns a plausible number either way --
    so the round trip `ang2pix(pix2ang(p)) == p` is pinned in the tests.

    Longitude is normalised to `[-180, 180)`. Not cosmetic:
    `healpix_to_lonlat` returns an astropy `Longitude` wrapped to `[0, 360)`,
    so a Chicago cell arrives as 271.77 rather than -88.23, and any map picking
    its projection centre from `lon.mean()` renders the wrong hemisphere.
    """
    p = np.asarray(pix, dtype=np.int64).ravel()
    if p.size == 0:
        return np.zeros((0, 2), dtype=float)
    lon, lat = _healpix(validate_nside(nside)).healpix_to_lonlat(p)
    lon = np.asarray(lon.to_value("deg"), dtype=float)
    lat = np.asarray(lat.to_value("deg"), dtype=float)
    return np.column_stack([lat, ((lon + 180.0) % 360.0) - 180.0])


def degrade(pix, nside_from: int, nside_to: int) -> np.ndarray:
    """Coarsen NESTED ids by bit shift -- a whole ladder in one operation.

    Equals geometric re-binning exactly, which is what makes the ladder free;
    the tests assert it against `ang2pix` at every rung rather than trusting
    the algebra.
    """
    validate_nside(nside_from)
    validate_nside(nside_to)
    if nside_to > nside_from:
        raise ValueError(
            f"nside_to ({nside_to}) must not exceed nside_from ({nside_from})"
        )
    shift = 2 * int(np.log2(nside_from // nside_to))
    return np.asarray(pix, dtype=np.int64) >> shift


def children(pix, levels: int = 1) -> np.ndarray:
    """The `4**levels` descendants of each cell, as an `(N, 4**levels)` array.

    The exact inverse of `degrade`: `degrade(children(p, k), nside*2**k, nside)`
    returns `p`, and every child's centre re-bins to `p` at the parent's nside.
    Both are asserted in the tests, because this function is the one place the
    aperture-4 nesting is relied on numerically.

    **The returned ids are valid at `nside * 2**levels`, not at the parent's
    nside.** Nothing here can check that -- every HEALPix id is a legal id at
    some nside -- so a caller that forgets to advance its nside gets cell
    centres that are silently displaced rather than an exception. That is the
    one hazard this grid has that H3's opaque string ids did not.
    """
    k = int(levels)
    if k < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    p = np.asarray(pix, dtype=np.int64).reshape(-1, 1)
    return (p << (2 * k)) | np.arange(4**k, dtype=np.int64)[None, :]


def neighbours(pix, nside: int) -> np.ndarray:
    """The 8 neighbours of each cell, as an `(N, 8)` array.

    Absent neighbours are `-1`, and they are real: **24 cells at every nside**
    sit at a corner of the 12-face base tessellation and have 7 rather than 8.
    That is 0.012% at nside=128 but **half the grid at nside=2** (24 of 48), so
    the `-1`s are a case to handle at every resolution, not a rare edge.
    `astropy_healpix` warns on them, which is noise here, so it is suppressed
    and the `-1`s are left for callers to filter. The direct analogue is H3's
    12 pentagons.

    Transposed from `astropy_healpix`, which returns `(8, N)`: callers want one
    row per input cell.
    """
    from astropy_healpix import neighbours as _nb

    p = np.asarray(pix, dtype=np.int64).ravel()
    if p.size == 0:
        return np.zeros((0, 8), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*invalid value.*neighbours.*")
        out = _nb(p, validate_nside(nside), order=_ORDER)
    return np.asarray(out, dtype=np.int64).T


def disk(pix, k: int, nside: int) -> np.ndarray:
    """Every cell within `k` steps of any of `pix`, including `pix` itself.

    H3's `grid_disk`. `astropy_healpix` has no k-ring primitive -- `neighbours`
    only ever returns the immediate ring -- so this grows breadth-first, one
    `neighbours` call per ring over the current shell rather than over
    everything reached so far.

    Returns a sorted unique array, so overlapping disks collapse: two adjacent
    seeds at `k=1` give 12 cells, not 2x9. Callers that expand a top-k set
    therefore need no deduplication of their own.

    `k=0` is the seeds, deduplicated.
    """
    nside = validate_nside(nside)
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    reached = np.unique(np.asarray(pix, dtype=np.int64).ravel())
    shell = reached
    for _ in range(k):
        nbrs = neighbours(shell, nside).ravel()
        nbrs = np.unique(nbrs[nbrs >= 0])
        shell = np.setdiff1d(nbrs, reached, assume_unique=True)
        if shell.size == 0:
            break
        reached = np.union1d(reached, shell)
    return reached


def cell_rings(pix, nside: int, *, step: int = 8) -> list[np.ndarray]:
    """One `(V, 2)` `(lon, lat)`-degree boundary ring per cell, for drawing.

    `step` points per edge so the cell's curvature on the sphere shows. Each
    ring is made contiguous in longitude: a cell straddling the antimeridian
    comes back with mixed-sign longitudes, and drawing that as one polygon
    smears it across the whole map. Every vertex is placed within half a turn
    of the first, which puts a straddling ring slightly outside [-180, 180] --
    correct for plotting, and what cartopy expects.
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
        "n_cells": npix(n),
        "cell_area_km2": round(pixel_area_km2(n), 3),
        "nominal_cell_km": round(nominal_cell_km(n), 3),
        "nominal_cell_km_note": "sqrt(area); HEALPix cells are exactly equal-area",
    }
