"""The grid: HEALPix, NESTED, laddered. Ported from v4's `healpix.py`.

In v5's vocabulary a **grid** is one HEALPix pixel at a rung (and, by
extension, the partition those pixels form). v4 called these "cells"; v5
reserves *cell* for the landmass-bounded Voronoi cell of the seeds, so nothing
in this module says cell.

Why HEALPix and not H3 is argued in v4's module of the same role and holds
unchanged: aperture-4 with exact nesting, so `ring == 0` is monotone
non-increasing as grids grow (H3 produced 705 violations on this repo's data);
exactly equal-area; and in NESTED ordering the parent of `pix` is `pix >> 2`.

`grid_km` is the **nominal grid distance**, `sqrt(area)`: 50.9 km at nside 128,
407 km at nside 16. It is one number with three jobs in v5 -- the grid pitch,
the complete-linkage diameter that groups sites into seeds, and the distance
the landmass is buffered by -- so that both partitions of the answer space are
built at one tolerance.
"""

from __future__ import annotations

import warnings

import numpy as np

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

#: Working resolution: 2,594 km^2 grids, 50.9 km nominal.
DEFAULT_NSIDE = 128

#: The ladder, finest first. Each step is one 4-to-1 subdivision.
NSIDE_LADDER: tuple[int, ...] = (128, 64, 32, 16)

#: Fixed, not a parameter: RING ordering would break `degrade`.
_ORDER = "nested"

#: How many rings out `ring_distance` grades before answering -1 (beyond).
MAX_RING = 2


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


def degrade(pix, nside_from: int, nside_to: int) -> np.ndarray:
    """Coarsen NESTED ids by bit shift. Equals geometric re-binning exactly."""
    validate_nside(nside_from)
    validate_nside(nside_to)
    if nside_to > nside_from:
        raise ValueError(
            f"nside_to ({nside_to}) must not exceed nside_from ({nside_from})"
        )
    shift = 2 * int(np.log2(nside_from // nside_to))
    return np.asarray(pix, dtype=np.int64) >> shift


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


def ring_distance(a, b, nside: int, max_ring: int = MAX_RING) -> np.ndarray:
    """Grid steps separating each `(a, b)` pair, or `-1` past `max_ring`.

    0 is the same grid, 1 one of `a`'s 8 neighbours. `-1` is "beyond",
    deliberately not a large number: once a prediction has left the
    neighbourhood, how far is `pred_dist_to_tg_km`'s question.

    Local by construction, which is the bound an unclipped Voronoi partition
    lacks. Grown breadth-first one ring at a time; unchanged from v4.
    """
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a and b must be the same length, got {a.shape} vs {b.shape}")
    nside = validate_nside(nside)

    out = np.full(a.shape, -1, dtype=np.int64)
    out[a == b] = 0
    if max_ring < 1:
        return out

    # `frontier` is everything reached, `shell` the grids whose neighbours are
    # still unexplored -- tracking both keeps the interior from re-expanding.
    frontier = [{int(x)} for x in a]
    shell = [{int(x)} for x in a]
    for k in range(1, max_ring + 1):
        pending = [i for i in range(a.size) if out[i] == -1 and shell[i]]
        if not pending:
            break
        flat = np.fromiter((c for i in pending for c in shell[i]), dtype=np.int64)
        counts = [len(shell[i]) for i in pending]
        nbrs = neighbours(flat, nside)
        pos = 0
        for i, n in zip(pending, counts):
            block = nbrs[pos : pos + n].ravel()
            pos += n
            new = {int(c) for c in block if c >= 0} - frontier[i]
            if int(b[i]) in new:
                out[i] = k
            frontier[i] |= new
            shell[i] = new
    return out


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
