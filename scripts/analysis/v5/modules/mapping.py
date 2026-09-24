"""Cartopy primitives for the v5 map: basemap, grid lattice, grid fills, cells.

Ported from v4's `mapping.py`, with its "cell" drawing renamed to *grid* (v5's
word for a HEALPix pixel) and one addition v4 could not have: `draw_cells`,
which inks the landmass-bounded Voronoi cells. v4 had no partition beside the
grid to draw -- correctness was containment in a grid -- so v3's Voronoi layer
was deleted there. v5 grades against both partitions, so both go on the map.

Ink is v4's validated pair: `TARGET_FILL` for TG grids (filled), and
`LANDMASS_EDGE` -- v4's VP blue, free here because v5's answer space has no VP
side -- for the buffered landmass outline.
"""

from __future__ import annotations

import numpy as np

from scripts.analysis.v5.modules import grid as G

#: Continental-US window, the repo's one frame.
US_MAINLAND_EXTENT = (-125.0, -66.0, 24.0, 50.0)

_PAD_FRAC = 0.08

#: TG grids -- occupied grids, filled because they are one partition's subject.
TARGET_FILL = "#eb6834"
TARGET_EDGE = "#b8451c"

#: The buffered landmass: outline only, dashed, so it reads as a limit rather
#: than a region. v4's `VP_EDGE`; the pair was validated all-pairs against
#: `TARGET_FILL` (worst CVD dE 24.7).
LANDMASS_EDGE = "#2a78d6"

#: The unoccupied grid lattice.
LATTICE_EDGE = "#dedcd3"

OCEAN = "#eaf2f8"
LAND = "#f8f7f3"
COASTLINE = "#a5a39c"
BORDERS = "#c3c2b7"
STATES = "#e6e4dd"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"

#: Cell boundaries: ink, not a hue. The cells are the second partition, drawn
#: as lines over the grid fill so neither hides the other.
CELL_EDGE = INK_2

#: Marker area, points^2. One size for both kinds: a site and a seed are two
#: categories, not two magnitudes.
MARKER_AREA = 14.0


def auto_extent(lats, lons) -> tuple[float, float, float, float]:
    """Data bounding box padded by `_PAD_FRAC`, clipped to valid lon/lat."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
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

    Measured on the box perimeter (where the extremum lies) rather than
    approximated from the lon/lat diagonal. Odd sample count so each edge's
    midpoint is sampled.
    """
    lon_min, lon_max, lat_min, lat_max = extent
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


def frame_grids(
    nside: int, extent: tuple[float, float, float, float], *, pad_grids: float = 2.0
) -> np.ndarray:
    """Every grid whose centre lies in (a padded) `extent`.

    Padding is in grids, so it tracks the rung and the lattice runs off the
    frame edge rather than stopping in a ragged fringe. A frame crossing the
    antimeridian is refused rather than mis-drawn.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    if lon_min >= lon_max:
        raise ValueError(
            f"extent longitudes must increase, got {lon_min} >= {lon_max}; "
            f"a frame crossing the antimeridian is not supported"
        )
    nside = G.validate_nside(nside)
    pad = pad_grids * G.grid_km(nside) / 111.19
    radius = _angular_radius_deg(extent) + pad

    if radius >= 180.0:
        idx = np.arange(G.n_grids(nside), dtype=np.int64)
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
    centres = G.pix2ang(idx, nside)
    keep = (
        (centres[:, 1] >= lon_min - pad)
        & (centres[:, 1] <= lon_max + pad)
        & (centres[:, 0] >= lat_min - pad)
        & (centres[:, 0] <= lat_max + pad)
    )
    return idx[keep]


def draw_basemap(ax) -> None:
    """Ocean, land, states, coastline, borders -- bottom to top."""
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN)
    ax.add_feature(cfeature.LAND, facecolor=LAND)
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor=STATES)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor=BORDERS)
    ax.spines["geo"].set_edgecolor(BORDERS)
    ax.spines["geo"].set_linewidth(0.6)


def draw_grids(
    ax,
    grid_ids,
    nside: int,
    *,
    facecolor="none",
    edgecolor: str,
    linewidth: float,
    alpha: float = 1.0,
    step: int = 8,
    zorder: int = 3,
) -> int:
    """Grids as one `PolyCollection` (the lattice alone is 5,740 polygons at
    nside 128). Returns the count drawn."""
    import cartopy.crs as ccrs
    from matplotlib.collections import PolyCollection

    grid_ids = np.asarray(grid_ids, dtype=np.int64).ravel()
    if grid_ids.size == 0:
        return 0
    rings = G.grid_rings(grid_ids, nside, step=step)
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
    """The unoccupied grid across the frame. Hairline and straight edges at the
    fine rungs (a nside-128 grid is ~19 px), heavier and curved at coarse ones."""
    fine = nside >= 64
    return draw_grids(
        ax,
        frame_grids(nside, extent),
        nside,
        facecolor="none",
        edgecolor=LATTICE_EDGE,
        linewidth=0.18 if fine else 0.35,
        step=2 if fine else 8,
        zorder=zorder,
    )


def draw_geometry(ax, geom, *, edgecolor: str, linewidth: float, linestyle="-", zorder=4) -> None:
    """A `(lon, lat)` shapely (Multi)Polygon as outline only."""
    import cartopy.crs as ccrs

    ax.add_geometries(
        [geom],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def draw_cells(ax, polygons: dict, *, linewidth: float = 0.9, zorder: int = 4) -> int:
    """The landmass-bounded Voronoi cells, as boundaries. Returns the count."""
    for geom in polygons.values():
        draw_geometry(ax, geom, edgecolor=CELL_EDGE, linewidth=linewidth, zorder=zorder)
    return len(polygons)
