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

RING ordering would break `degrade` and therefore the whole package; the order
is fixed at "nested" in `scripts.libs.healpix.grid` and is not a parameter.

## Where the primitives live

`validate_nside`, `npix`, `pixel_area_km2`, `nominal_cell_km`, `ang2pix`,
`pix2ang`, `degrade`, `neighbours`, `children`, `disk`, `cell_rings` and
`describe` are **re-exported from `scripts.libs.healpix.grid`**, which is the
one implementation. They moved there when the framework's density MTL needed
the same grid: the framework may not import `scripts.analysis`, so a grid
shared by both belongs in `scripts/libs/` -- the rule
`scripts/analysis/v3/tests/test_layering.py` states for the CSV contract.

What stays here is the part that is about *this metric* rather than about the
grid: the ladder, the ring-distance rule, and the occupancy curve.
"""

from __future__ import annotations

import numpy as np

from scripts.libs.healpix.grid import (  # noqa: F401  (re-exported)
    DEFAULT_NSIDE,
    ang2pix,
    cell_rings,
    children,
    degrade,
    describe,
    disk,
    neighbours,
    nominal_cell_km,
    npix,
    pix2ang,
    pixel_area_km2,
    validate_nside,
)

#: The accuracy ladder, finest first. Each step is one 4-to-1 subdivision, so
#: `degrade` walks it by shifting 2 bits per rung.
NSIDE_LADDER: tuple[int, ...] = (128, 64, 32, 16)

#: How many rings out the classification metric grades before giving up. Ring 0
#: is "same cell"; beyond `MAX_RING` the prediction is reported as unplaced
#: rather than as a large ring, because the count stops being informative once
#: it exceeds the neighbourhood the metric is asking about.
MAX_RING = 2


def ladder_for(nside: int) -> tuple[int, ...]:
    """The rungs at or coarser than `nside`, finest first.

    Downward-only, like v3's `coarsening_ladder`. A rung finer than the answer
    space was built at would be scoring against cells no class was ever defined
    on.
    """
    n = validate_nside(nside)
    return (n,) + tuple(x for x in NSIDE_LADDER if x < n)


def ring_distance(a, b, nside: int, max_ring: int = MAX_RING) -> np.ndarray:
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
