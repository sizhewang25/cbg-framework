"""Shared cartopy primitives for the v3 map commands.

Split out of `map_answer_space.py` when a second map (`map_bipartite.py`) needed
the same layers. Everything here was already load-bearing in that module and is
moved **verbatim**, so `answer_space_map.png` is byte-identical across the split
— the same bargain `venn.py` made when `diagram/` was extracted from it.

Three kinds of thing live here:

* **The frame.** `_auto_extent` and `US_MAINLAND_EXTENT`, so every map derives
  its window the same way.
* **The class partition.** `seed_voronoi` and its three measured constants. This
  is the classifier's own top-1 decision boundary (`classify.py` labels a
  prediction by `argmin` over great-circle distance to the K seeds), not
  decoration, so both maps draw the identical geometry rather than each
  approximating it.
* **The layers.** `draw_basemap` / `draw_voronoi` / `draw_cells`, thin wrappers
  whose defaults reproduce the answer-space map exactly. A caller changes a
  colour by naming it, never by re-implementing the layer.

Plus `great_circle_segments`, which only the flow map needs but which is pure
spherical geometry and belongs beside the rest.

`matplotlib.use("Agg")` is selected here, before `pyplot` is imported, which is
why the importing modules do not have to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.analysis.v3.modules.answer_space import _unit_vectors  # noqa: E402

#: Continental-US window, matching `plot_targets_vps.US_MAINLAND_EXTENT`.
US_MAINLAND_EXTENT = (-125.0, -66.0, 24.0, 50.0)

#: Fraction of the data span added as padding when the extent is auto-derived.
_PAD_FRAC = 0.08


def _auto_extent(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float, float, float]:
    """Data bounding box padded by `_PAD_FRAC`, clipped to valid lon/lat."""
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    # A degenerate span (all targets in one metro) would give a zero-width
    # extent, so floor the pad at half a degree.
    lon_pad = max((lon_max - lon_min) * _PAD_FRAC, 0.5)
    lat_pad = max((lat_max - lat_min) * _PAD_FRAC, 0.5)
    return (
        max(lon_min - lon_pad, -180.0),
        min(lon_max + lon_pad, 180.0),
        max(lat_min - lat_pad, -90.0),
        min(lat_max + lat_pad, 90.0),
    )


def _seed_colors(n: int) -> list:
    """One colour per seed, cycling `tab20`.

    Identity cue, not a scale: with K > 20 (as7018 has 27) two seeds can share
    a colour, so colour distinguishes *neighbouring* cells rather than naming a
    class globally.
    """
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


@dataclass(frozen=True)
class SeedVoronoi:
    """The nearest-seed partition of the frame, ready to draw.

    `cells` are shapely polygons living in `crs`, deliberately **not**
    unprojected to lon/lat. Handing cartopy the projected geometry together with
    `crs=` is what makes a bisector — straight in the projection — get drawn as
    the curve it actually is in PlateCarree; unprojecting the vertices here and
    inking straight lon/lat segments would discard that. Cartopy alone only
    densifies ~1.2x though, so the edges are pre-split at
    `_VORONOI_SEGMENT_M` first.

    `seed_index` is the position in `seeds` each cell belongs to, so a cell can
    be coloured to match its seed. It is not the identity permutation: cells
    outside the frame are dropped, and GEOS does not return them in seed order.
    """

    cells: list
    seed_index: np.ndarray
    crs: Any


#: Fraction of the projected frame grown around the clip box. The padding is
#: what keeps the clip-box edges off-screen: without it the outermost cells are
#: cut exactly at the frame, and their outlines draw a spurious "boundary"
#: tracing the map border.
_VORONOI_PAD_FRAC = 0.25

#: Points sampled along each frame edge before projecting. A projected frame
#: edge is curved, so its four corners alone understate the box it needs.
_EDGE_SAMPLES = 25

#: Max cell-edge length, in projection metres, before handing cells to cartopy.
#: A Voronoi edge is a single straight segment thousands of km long, and cartopy
#: only densifies it ~1.2x while reprojecting, so the inked line cuts the corner
#: of the curve it should follow. Measured on as01: without this the drawn line
#: sits up to 54.8 km (21 px) off the true bisector; at 200 km that falls to
#: 8.2 km (3 px), which is the projection's own floor — 100 km and 25 km give no
#: further gain, so this is the cheap end of a saturated curve.
_VORONOI_SEGMENT_M = 200_000.0


def _projected_frame_box(extent, crs, *, pad_frac: float):
    """Padded bounding box of the map frame, in `crs`'s coordinates."""
    import cartopy.crs as ccrs
    from shapely.geometry import box

    lon_min, lon_max, lat_min, lat_max = extent
    t = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
    lon_span = lon_min + (lon_max - lon_min) * t
    lat_span = lat_min + (lat_max - lat_min) * t
    edge = np.full(_EDGE_SAMPLES, 1.0)
    lons = np.concatenate([lon_span, lon_span, edge * lon_min, edge * lon_max])
    lats = np.concatenate([edge * lat_min, edge * lat_max, lat_span, lat_span])

    p = crs.transform_points(ccrs.PlateCarree(), lons, lats)
    x, y = p[:, 0], p[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    x, y = x[finite], y[finite]
    dx = max((x.max() - x.min()) * pad_frac, 1.0)
    dy = max((y.max() - y.min()) * pad_frac, 1.0)
    return box(x.min() - dx, y.min() - dy, x.max() + dx, y.max() + dy)


def seed_voronoi(
    seeds, extent, *, pad_frac: float = _VORONOI_PAD_FRAC
) -> SeedVoronoi | None:
    """Nearest-seed Voronoi partition of the frame — the top-1 decision boundary.

    Computed in an **azimuthal-equidistant projection** centred on the seed mean,
    not in raw lon/lat, because the classifier ranks seeds by great-circle
    distance and a degree is not a distance: at 38 N one degree of longitude is
    87.6 km against 111.2 km of latitude, rising to a 1.49x anisotropy at 48 N.
    Euclidean-in-degrees therefore over-weights north-south separation and tilts
    every bisector. Measured against true great-circle nearest-seed labelling on
    the `h3-4` runs, raw lon/lat misassigns ~10.5% of the frame's area (drawn
    line displaced up to ~210 km); this projection misassigns 0.35-0.38%, with
    the residual sitting a median 0.7-0.9 km from the true boundary, which is
    sub-pixel here.

    Returns `None` when there is nothing to draw: `K < 2` has no boundary at all,
    and a frame that no cell reaches (every seed far off-screen) yields no cells.

    Seeds are used in full, including any outside `extent` — unlike
    `voronoi.build_landmass_voronoi`, which filters seeds to its landmass. An
    off-frame seed still shapes an on-frame boundary, so dropping it would move
    lines that are visible.
    """
    import cartopy.crs as ccrs
    import shapely

    from scripts.visualization.cluster.voronoi import clipped_voronoi_cells

    lat = np.asarray(seeds["seed_lat"], dtype=float)
    lon = np.asarray(seeds["seed_lon"], dtype=float)
    if lat.size < 2:
        return None

    aeqd = ccrs.AzimuthalEquidistant(
        central_longitude=float(lon.mean()), central_latitude=float(lat.mean())
    )
    pts = aeqd.transform_points(ccrs.PlateCarree(), lon, lat)

    clip = _projected_frame_box(extent, aeqd, pad_frac=pad_frac)
    if clip is None:
        return None

    cells = clipped_voronoi_cells(pts[:, 0], pts[:, 1], clip, crs=None)
    if cells.empty:
        return None
    return SeedVoronoi(
        cells=[shapely.segmentize(g, _VORONOI_SEGMENT_M) for g in cells.geometry],
        seed_index=cells["seed_index"].to_numpy(dtype=int),
        crs=aeqd,
    )


# ---- layers -----------------------------------------------------------------
# Defaults reproduce the answer-space map exactly, so a caller changes a colour
# by naming it rather than by re-implementing the layer.


def draw_basemap(
    ax,
    *,
    ocean: str = "#eaf2f8",
    land: str = "#f6f4ef",
    coastline: str = "#999999",
    borders: str = "#cccccc",
) -> None:
    """Ocean, land, coastline and borders, in that order."""
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.OCEAN, facecolor=ocean)
    ax.add_feature(cfeature.LAND, facecolor=land)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=coastline)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor=borders)


def draw_voronoi(
    ax,
    partition: SeedVoronoi,
    *,
    edgecolor: str = "red",
    linewidth: float = 0.5,
    linestyle: str = "--",
    zorder: int = 3,
) -> None:
    """The class-region boundaries: outlines only, never filled.

    Same choice as `_plot_voronoi_underlay` in plot_ground_truth_clusters.py and
    `VORONOI_LINE` in cluster_world_map.js, so partitions stay comparable across
    figures and the basemap stays legible. The geometry is handed over projected,
    with `crs=partition.crs`, which is what lets cartopy re-curve each bisector.
    """
    ax.add_geometries(
        list(partition.cells),
        crs=partition.crs,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def draw_cells(
    ax,
    grid,
    cell_ids,
    resolution: int,
    *,
    colors: list | None = None,
    facecolor: Any = None,
    step: int = 8,
    edgecolor: str = "#333333",
    linewidth: float = 0.6,
    linestyle: str = "-",
    alpha: float = 0.75,
    zorder: int = 2,
) -> int:
    """Occupied grid cells as filled or outlined polygons. Returns the count.

    `colors` gives one fill per cell (the answer-space map's per-seed identity
    cue); `facecolor` gives every cell the same one (`"none"` for an outline-only
    layer). Exactly one of the two is meaningful, and `colors` wins.

    Rings come back **ragged** from `Grid.cell_boundaries` — 4 sides x `step` for
    a HEALPix quad, 6 for an H3 hexagon, 5 for one of its 12 pentagons — already
    in `(lon, lat)` degrees and already made contiguous across the antimeridian.
    So nothing here may assume a rectangular array or re-wrap longitudes.
    """
    import cartopy.crs as ccrs
    from matplotlib.patches import Polygon

    rings = grid.cell_boundaries(np.asarray(cell_ids), resolution, step=step)
    for i, ring in enumerate(rings):
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor=colors[i] if colors is not None else facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                zorder=zorder,
            )
        )
    return len(rings)


# ---- great-circle polylines --------------------------------------------------

#: Vertices per drawn edge, endpoints included. A great circle over the widest
#: measured edge here (4,387 km on as01) departs from its straight lon/lat chord
#: by ~90 km at the midpoint, so the chord is not usable; 9 vertices bring the
#: residual under a pixel at continental scale, and more buys nothing visible.
DEFAULT_SEGMENT_POINTS = 9


def great_circle_segments(
    lat_a, lon_a, lat_b, lon_b, *, n_points: int = DEFAULT_SEGMENT_POINTS
) -> np.ndarray:
    """Great-circle polylines for paired endpoints, as `(E, n_points, 2)` lon/lat.

    Interpolation is a slerp between the endpoints' unit vectors, so the path is
    the true great circle with no projection entering — the same projection-free
    stance as `answer_space.pairwise_km`.

    Done here rather than by handing cartopy `ccrs.Geodetic()` because that
    transform densifies per artist and is far too slow at tens of thousands of
    edges; the output of this is drawn in PlateCarree, where it is already the
    right curve. Coincident endpoints (a VP measuring a target at its own
    coordinate) slerp to a zero-length path, and are interpolated linearly
    instead of dividing by `sin(0)`.

    Longitudes follow `grid.ring_lonlat`'s rule — every vertex within half a turn
    of its polyline's first — so an edge crossing the antimeridian is drawn as one
    line slightly outside [-180, 180] rather than smeared across the map.
    """
    a = _unit_vectors(lat_a, lon_a)
    b = _unit_vectors(lat_b, lon_b)
    if a.shape != b.shape:
        raise ValueError(f"endpoint arrays disagree: {a.shape} vs {b.shape}")

    omega = np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))
    sin_omega = np.sin(omega)
    degenerate = sin_omega < 1e-12
    safe = np.where(degenerate, 1.0, sin_omega)[:, None]

    t = np.linspace(0.0, 1.0, int(n_points))[None, :]
    w_a = np.where(degenerate[:, None], 1.0 - t, np.sin((1.0 - t) * omega[:, None]) / safe)
    w_b = np.where(degenerate[:, None], t, np.sin(t * omega[:, None]) / safe)

    pts = w_a[:, :, None] * a[:, None, :] + w_b[:, :, None] * b[:, None, :]
    norm = np.linalg.norm(pts, axis=2, keepdims=True)
    pts = pts / np.where(norm == 0.0, 1.0, norm)

    lat = np.degrees(np.arcsin(np.clip(pts[:, :, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(pts[:, :, 1], pts[:, :, 0]))
    lon = lon[:, :1] + ((lon - lon[:, :1] + 180.0) % 360.0) - 180.0
    return np.stack([lon, lat], axis=2)
