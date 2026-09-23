"""HEALPix, the only grid v4 knows.

v3 carried a `Grid` abstraction over H3 and HEALPix. v4 does not, and that is
the point of it being a separate package: the laddered accuracy metric v4 exists
to compute **cannot be expressed on H3 at all**.

## Why not H3

H3 is aperture-7, and hexagons cannot tile hexagons: a parent has 7 children but
the 6 outer ones straddle its boundary, so the children's geometric union is not
the parent. `h3.cell_to_parent` is exact on the *index* and is not a geometric
container — v3's own answer-space metadata reports the fallout as
`parent_lineage_disagreements`.

The consequence for a laddered metric is fatal rather than cosmetic. Measured on
this repo's real predictions (1,269 targets x 6 methods x 3 datasets), asking
"do the prediction and the truth share a cell at resolution r" produced **705
cases where they shared a cell at r but not at r-1**. It shows up in aggregate,
not just per target: `vanilla_cbg` on as01 scored 0.594 at res 1 and 0.659 at
res 2 — higher than its own parent. So "this method achieves res-3 accuracy" is
not a well-formed claim under H3, because agreeing at res 3 does not imply
agreeing at res 2.

Re-binning coordinates and walking `cell_to_parent` also disagree with each
other, on 552 of 5,906 targets at res 2. There is no single right answer to pick.

## Why HEALPix

Aperture-4 with the four children exactly tiling the parent, so containment is
transitive and the ladder is monotone by construction — 0 violations on the same
data. Two further properties earn it the job over HTM (Spotter's own grid, which
is also aperture-4 and also measured 0 violations):

* **Exactly equal-area.** Every cell at an nside has identical area. H3 res-4
  varies by 33% (`cell_area_km2_max_over_min = 1.3255`), so "res-4 accuracy"
  already means slightly different things in different parts of the map.
* **The ladder is a bit shift.** In NESTED ordering the parent of `pix` is
  `pix >> 2`, so one `ang2pix` pass at the finest rung yields every coarser rung
  for free and with no re-projection. `degrade` is that shift, and
  `test_healpix.py` pins it against independent re-binning.

`nside=128` is the working resolution: 196,608 cells of 2,594 km², nominally
50.9 km across, which is the metro granularity prior work puts near 40 km.
The ladder runs 128 -> 16 (50.9 km -> 407 km).

RING ordering would break `degrade` and therefore the whole package; `_ORDER` is
fixed at "nested" and is not a parameter.
"""

from __future__ import annotations

import warnings

import numpy as np

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

#: Working resolution: 2,594 km^2 cells, 50.9 km nominal.
DEFAULT_NSIDE = 128

#: The accuracy ladder, finest first. Each step is one 4-to-1 subdivision, so
#: `degrade` walks it by shifting 2 bits per rung.
NSIDE_LADDER: tuple[int, ...] = (128, 64, 32, 16)

#: Fixed, not a parameter — see the module docstring.
_ORDER = "nested"

#: How many rings out the classification metric grades before giving up. Ring 0
#: is "same cell"; beyond `MAX_RING` the prediction is reported as unplaced
#: rather than as a large ring, because the count stops being informative once
#: it exceeds the neighbourhood the metric is asking about.
MAX_RING = 2


def _healpix(nside: int):
    # Lazy: astropy is a heavy import and a caller that only needs the areas
    # (which are closed-form) should not pay for it.
    from astropy_healpix import HEALPix

    return HEALPix(nside=nside, order=_ORDER)


def validate_nside(nside: int) -> int:
    """`nside` must be a positive power of two for NESTED ids to nest."""
    n = int(nside)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"nside must be a positive power of two, got {nside}")
    return n


def npix(nside: int) -> int:
    """Total cells: `12 * nside**2` (196,608 at nside=128)."""
    return 12 * validate_nside(nside) ** 2


def pixel_area_km2(nside: int) -> float:
    """Cell area in km^2 — exact and identical for every cell at this nside."""
    return 4.0 * np.pi * EARTH_RADIUS_KM**2 / npix(nside)


def nominal_cell_km(nside: int) -> float:
    """Cell pitch as `sqrt(area)` — 50.9 km at nside=128.

    This is the merge scale: the distance below which two targets are treated as
    one place.
    """
    return float(np.sqrt(pixel_area_km2(nside)))


def ladder_for(nside: int) -> tuple[int, ...]:
    """The rungs at or coarser than `nside`, finest first.

    Downward-only, like v3's `coarsening_ladder`. A rung finer than the answer
    space was built at would be scoring against cells no class was ever defined
    on.
    """
    n = validate_nside(nside)
    return (n,) + tuple(x for x in NSIDE_LADDER if x < n)


def ang2pix(lat_deg, lon_deg, nside: int = DEFAULT_NSIDE) -> np.ndarray:
    """NESTED cell id per `(lat, lon)` in degrees."""
    import astropy.units as u

    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    pix = _healpix(validate_nside(nside)).lonlat_to_healpix(lon * u.deg, lat * u.deg)
    return np.asarray(pix, dtype=np.int64)


def pix2ang(pix, nside: int = DEFAULT_NSIDE) -> np.ndarray:
    """Cell centres as an `(N, 2)` array of `(lat, lon)` degrees.

    **`(lat, lon)`, the opposite order from `cell_rings`.** A seed is written to
    `seeds.csv` as `seed_lat` then `seed_lon`; rings are `(lon, lat)` because
    that is what plotting wants. Two orders in one module is a swap-bug magnet,
    so `test_healpix.py` pins the round trip `ang2pix(pix2ang(p)) == p`.

    Longitude is normalised to `[-180, 180)`. Not cosmetic: `healpix_to_lonlat`
    returns an astropy `Longitude` wrapped to `[0, 360)`, so a Chicago cell
    arrives as 271.77 rather than -88.23. Nothing would crash — haversine goes
    through sin/cos — but any map that picks a projection centre from
    `lon.mean()` would render in the wrong hemisphere.
    """
    p = np.asarray(pix, dtype=np.int64).ravel()
    if p.size == 0:
        return np.zeros((0, 2), dtype=float)
    lon, lat = _healpix(validate_nside(nside)).healpix_to_lonlat(p)
    lon = np.asarray(lon.to_value("deg"), dtype=float)
    lat = np.asarray(lat.to_value("deg"), dtype=float)
    return np.column_stack([lat, ((lon + 180.0) % 360.0) - 180.0])


def degrade(pix, nside_from: int, nside_to: int) -> np.ndarray:
    """Coarsen NESTED ids by bit shift — the whole ladder in one operation.

    Valid only for NESTED ordering with both nsides powers of two and
    `nside_to <= nside_from`. Equals geometric re-binning exactly, which is the
    property that makes the ladder free; `test_healpix.py` asserts it against
    `ang2pix` at every rung rather than trusting the algebra.
    """
    validate_nside(nside_from)
    validate_nside(nside_to)
    if nside_to > nside_from:
        raise ValueError(
            f"nside_to ({nside_to}) must not exceed nside_from ({nside_from})"
        )
    shift = 2 * int(np.log2(nside_from // nside_to))
    return np.asarray(pix, dtype=np.int64) >> shift


def neighbours(pix, nside: int) -> np.ndarray:
    """The 8 neighbours of each cell, as an `(N, 8)` array.

    Absent neighbours are `-1`, and they are real: **24 cells at every nside**
    sit at a corner of the 12-face base tessellation and have 7 rather than 8
    (0.012% at nside=128, 0.78% at nside=16). `astropy_healpix` warns on them,
    which is noise here rather than information, so it is suppressed and the
    `-1`s are left for `ring_distance` to filter. The direct analogue is H3's 12
    pentagons.

    Transposed from `astropy_healpix`, which returns `(8, N)`: every caller here
    wants one row per input cell.
    """
    from astropy_healpix import neighbours as _nb

    p = np.asarray(pix, dtype=np.int64).ravel()
    if p.size == 0:
        return np.zeros((0, 8), dtype=np.int64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*invalid value.*neighbours.*")
        out = _nb(p, validate_nside(nside), order=_ORDER)
    return np.asarray(out, dtype=np.int64).T


def ring_distance(
    a, b, nside: int, max_ring: int = MAX_RING
) -> np.ndarray:
    """How many cell steps separate each `(a, b)` pair, or `-1` past `max_ring`.

    0 means the same cell. 1 means `b` is one of `a`'s 8 neighbours. `-1` means
    "further than `max_ring`", deliberately not a large number: the metric asks
    whether a prediction landed in the target's neighbourhood, and once the
    answer is no, how far beyond is what `error_km` is for.

    This is the bound that nearest-seed assignment lacked. A Voronoi partition
    over K seeds labels every point on Earth, so a prediction 2,360 km away in
    the Canadian Arctic was credited to Seattle merely for being marginally
    closer to it than to Omaha. Ring distance is local by construction: no
    number of far-away cells can make a distant cell adjacent.

    Grown breadth-first one ring at a time, which costs `max_ring`
    `neighbours` calls rather than a global BFS -- fine because `max_ring` is 2.
    `astropy_healpix.neighbours` only ever returns the immediate ring, so there
    is no k-ring primitive to call instead (H3's `grid_disk` takes any k).
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

    # `frontier` is the set already reached, `shell` the cells whose neighbours
    # are still unexplored. Tracking both is what keeps the growth from
    # re-expanding the interior at every step.
    frontier = [{int(x)} for x in a]
    shell = [{int(x)} for x in a]
    for k in range(1, max_ring + 1):
        pending = [i for i in range(a.size) if out[i] == -1 and shell[i]]
        if not pending:
            break
        flat = np.fromiter(
            (c for i in pending for c in shell[i]), dtype=np.int64
        )
        counts = [len(shell[i]) for i in pending]
        # neighbours() returns -1 for absent neighbours
        nbrs = neighbours(flat, nside)
        pos = 0
        for i, n in zip(pending, counts):
            block = nbrs[pos : pos + n].ravel()
            pos += n
            # remove reached duplicates from new neighbors
            new = {int(c) for c in block if c >= 0} - frontier[i]
            if int(b[i]) in new:
                out[i] = k
            frontier[i] |= new
            shell[i] = new
    return out


def occupied_cells_by_nside(
    lat_deg, lon_deg, nsides: tuple[int, ...] = NSIDE_LADDER
) -> dict[int, int]:
    """Distinct occupied cell count at each rung, in one `ang2pix` pass.

    A steep climb toward fine cells means the point set only separates at
    intra-metro scales; a flat curve means genuinely distinct metros.
    """
    ordered = sorted({int(n) for n in nsides}, reverse=True)
    if not ordered:
        return {}
    finest = ordered[0]
    pix = ang2pix(lat_deg, lon_deg, finest)
    return {n: int(np.unique(degrade(pix, finest, n)).size) for n in ordered}


def cell_rings(pix, nside: int, *, step: int = 8) -> list[np.ndarray]:
    """One `(V, 2)` `(lon, lat)`-degree boundary ring per cell, for drawing.

    `step` points per edge so the cell's curvature on the sphere shows. Each
    ring is made contiguous in longitude: a cell straddling the antimeridian
    comes back with mixed-sign longitudes, and drawing that as one polygon
    smears it across the whole map. Every vertex is placed within half a turn of
    the first, which puts a straddling ring slightly outside [-180, 180] —
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
