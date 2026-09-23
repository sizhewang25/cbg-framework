"""Cartopy primitives for the v4 maps — basemap, lattice, cell fills.

v3 has a module of the same name and this is **not** a port of it. Two thirds of
that file (`SeedVoronoi`, `seed_voronoi`, `_projected_frame_box`,
`_VORONOI_SEGMENT_M`, `_EDGE_SAMPLES`) exists to draw a nearest-seed decision
boundary, and v4 has no such boundary to draw: correctness is *containment in a
cell*, so the class boundary is the cell edge, which `healpix.cell_rings` already
returns. The geometry v3 had to reconstruct in an azimuthal-equidistant
projection and then re-segmentize at 200 km is, here, simply the grid.

What is left is the part that was never about Voronoi: a frame, a basemap, and
two ways of inking cells.

## The lattice

`frame_cells` is the one piece with no v3 counterpart. v3 drew **only occupied**
cells and said so ("the full 288,122-cell grid would be both unreadable and
pointless at continental scale") — true of a global frame, and false of a
continental one: the US mainland window holds 5,740 cells at nside 128 and 92 at
nside 16, both drawable as a single `PolyCollection`.

Drawing the unoccupied cells is the point rather than decoration. The claim v4
rests on is that the grid is **fixed, not fitted** — a class boundary falls where
HEALPix falls, not where the targets are sparse — and the empty cells are the
only direct evidence of it. Occupied cells alone look exactly like a clustering,
which is the reading the answer space must not invite.
"""

from __future__ import annotations

import numpy as np

from scripts.analysis.v4.modules import healpix as H

#: Continental-US window, matching v3's `mapping.US_MAINLAND_EXTENT` and
#: `plot_targets_vps.US_MAINLAND_EXTENT`, so the frame is the repo's one frame.
US_MAINLAND_EXTENT = (-125.0, -66.0, 24.0, 50.0)

#: Fraction of the data span added as padding when the extent is auto-derived.
_PAD_FRAC = 0.08

# ---- ink ---------------------------------------------------------------------
# Slots 1 and 2 of the dataviz reference theme. Validated as a pair with
# `validate_palette.js --mode light --pairs all` (all-pairs because a map is a
# choropleth form, where any two marks can end up adjacent): all five checks
# pass, worst CVD dE 24.7 (protan) / 32.7 (tritan), normal-vision dE 33.6.
#
# Blue for the VP side is also v3's `map_bipartite._VP_COLOR`, unchanged, so the
# two packages' maps do not disagree about which side is which.

#: Target cells — the classes. Filled, because they are the figure's subject.
TARGET_FILL = "#eb6834"

#: Edge of a target cell. Its own hue darkened, so a target cell *without* a VP
#: still reads as bounded; a target cell *with* one takes the blue VP outline
#: over the top and the difference is visible without consulting the legend.
TARGET_EDGE = "#b8451c"

#: VP cells — outline only, never filled. There are 4x as many VP cells as
#: target cells on these meshes (80 against 18 at nside 128), so two fills of
#: equal weight put the ink on the wrong side. Outline-vs-fill also gives the
#: overlap a compositional reading — orange inside blue — with no third hue and
#: no alpha blending, whose result is neither predictable nor validatable.
VP_EDGE = "#2a78d6"

#: The unoccupied grid. `#dedcd3` is a half-step below the design system's
#: `_GRID`: at nside 128 the frame carries 5,740 of these and the surface has to
#: survive them.
LATTICE_EDGE = "#dedcd3"

OCEAN = "#eaf2f8"
LAND = "#f8f7f3"
COASTLINE = "#a5a39c"
BORDERS = "#c3c2b7"
STATES = "#e6e4dd"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"

#: Marker area, in points^2, for **both** node kinds. One size deliberately: a
#: target site and a VP are two categories, not two magnitudes, so they separate
#: by shape and ink and must not also differ in size.
MARKER_AREA = 14.0


def auto_extent(lats, lons) -> tuple[float, float, float, float]:
    """Data bounding box padded by `_PAD_FRAC`, clipped to valid lon/lat."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    # A degenerate span (every target in one metro) would give a zero-width
    # extent, so floor the pad at half a degree.
    lon_pad = max((lon_max - lon_min) * _PAD_FRAC, 0.5)
    lat_pad = max((lat_max - lat_min) * _PAD_FRAC, 0.5)
    return (
        max(lon_min - lon_pad, -180.0),
        min(lon_max + lon_pad, 180.0),
        max(lat_min - lat_pad, -90.0),
        min(lat_max + lat_pad, 90.0),
    )


def _angular_radius_deg(extent: tuple[float, float, float, float]) -> float:
    """Great-circle radius, in degrees, of a cone covering the whole frame.

    Measured rather than approximated from the lon/lat diagonal: a degree of
    longitude is not a degree of distance, so `hypot(dlon, dlat)/2` overshoots
    badly at high latitude and — more dangerously — can undershoot for a wide
    low-latitude box. The extremum over a lon/lat box is attained on its
    boundary, so sampling the perimeter and taking the max is exact up to the
    sampling step, which the caller's own padding then covers.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    # Odd, so each edge's midpoint is sampled. The extremum often sits exactly
    # there — for a whole-sphere box an even count misses the antipode and
    # returns 178.6 rather than 180, which would then skip the full-sphere
    # shortcut in `frame_cells`.
    t = np.linspace(0.0, 1.0, 65)
    lon_edge = lon_min + (lon_max - lon_min) * t
    lat_edge = lat_min + (lat_max - lat_min) * t
    ones = np.ones_like(t)
    lons = np.concatenate([lon_edge, lon_edge, ones * lon_min, ones * lon_max])
    lats = np.concatenate([ones * lat_min, ones * lat_max, lat_edge, lat_edge])

    c_lon = np.radians((lon_min + lon_max) / 2.0)
    c_lat = np.radians((lat_min + lat_max) / 2.0)
    p_lon, p_lat = np.radians(lons), np.radians(lats)
    cos_sep = np.sin(c_lat) * np.sin(p_lat) + np.cos(c_lat) * np.cos(p_lat) * np.cos(
        p_lon - c_lon
    )
    return float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))).max())


def frame_cells(
    nside: int, extent: tuple[float, float, float, float], *, pad_cells: float = 2.0
) -> np.ndarray:
    """Every cell of the grid whose centre lies in (a padded) `extent`.

    The padding is in **cells**, not degrees, so it tracks the rung: without it
    the cells straddling the frame edge — whose centres sit just outside — are
    dropped, and the lattice stops short of the border in a ragged fringe that
    reads as a feature. Two cells of slack puts the fringe off-screen at every
    rung, and the axes clip what hangs over.

    Selection is by **centre**, which is what makes the padding necessary and is
    also what makes it cheap: no polygon clipping, one vectorised mask.

    A frame crossing the antimeridian is refused rather than mis-drawn: the mask
    below is a plain interval test, so `lon_min > lon_max` would silently select
    nothing.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    if lon_min >= lon_max:
        raise ValueError(
            f"extent longitudes must increase, got {lon_min} >= {lon_max}; "
            f"a frame crossing the antimeridian is not supported"
        )
    nside = H.validate_nside(nside)

    pad = pad_cells * H.nominal_cell_km(nside) / 111.19
    radius = _angular_radius_deg(extent) + pad

    if radius >= 180.0:
        idx = np.arange(H.npix(nside), dtype=np.int64)
    else:
        import astropy.units as u
        from astropy_healpix import HEALPix

        idx = np.asarray(
            HEALPix(nside=nside, order="nested").cone_search_lonlat(
                ((lon_min + lon_max) / 2.0) * u.deg,
                ((lat_min + lat_max) / 2.0) * u.deg,
                radius=radius * u.deg,
            ),
            dtype=np.int64,
        )

    centres = H.pix2ang(idx, nside)
    keep = (
        (centres[:, 1] >= lon_min - pad)
        & (centres[:, 1] <= lon_max + pad)
        & (centres[:, 0] >= lat_min - pad)
        & (centres[:, 0] <= lat_max + pad)
    )
    return idx[keep]


def draw_basemap(ax) -> None:
    """Ocean, land, states, coastline, borders — bottom to top."""
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN)
    ax.add_feature(cfeature.LAND, facecolor=LAND)
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor=STATES)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor=BORDERS)
    ax.spines["geo"].set_edgecolor(BORDERS)
    ax.spines["geo"].set_linewidth(0.6)


def draw_cells(
    ax,
    cell_ids,
    nside: int,
    *,
    facecolor="none",
    edgecolor: str,
    linewidth: float,
    alpha: float = 1.0,
    step: int = 8,
    zorder: int = 3,
) -> int:
    """Cells as one `PolyCollection`. Returns the count drawn.

    One collection rather than a patch per cell: the lattice alone is 5,740
    polygons at nside 128, where the per-artist overhead dominates everything
    else the figure does.

    Rings come back from `healpix.cell_rings` already in `(lon, lat)` degrees and
    already made contiguous across the antimeridian, so nothing here re-wraps
    longitude.
    """
    import cartopy.crs as ccrs
    from matplotlib.collections import PolyCollection

    cell_ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    if cell_ids.size == 0:
        return 0
    rings = H.cell_rings(cell_ids, nside, step=step)
    ax.add_collection(
        PolyCollection(
            rings,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )
    )
    return len(rings)


def draw_lattice(
    ax, nside: int, extent: tuple[float, float, float, float], *, zorder: int = 2
) -> int:
    """The unoccupied grid across the whole frame. Returns the count drawn.

    `step` and `linewidth` both track the rung. A nside-128 cell is ~19 px wide
    in a continental panel, so its four edges need no curvature (`step=2` is the
    corners) and a hairline; an nside-16 cell is ~150 px and gets both. The
    coarse rungs are also where the lattice is legible enough to be worth
    reading, which is why they get the heavier line rather than a uniform one.
    """
    fine = nside >= 64
    return draw_cells(
        ax,
        frame_cells(nside, extent),
        nside,
        facecolor="none",
        edgecolor=LATTICE_EDGE,
        linewidth=0.18 if fine else 0.35,
        step=2 if fine else 8,
        zorder=zorder,
    )
