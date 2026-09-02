"""Static map of the target answer space: class regions, cells, targets, seeds.

Renders what `build-answer-space` produced, so the §7.3 quantization can be
looked at rather than inferred from `meta.json`. Four layers, bottom to top:
the occupied grid cells (filled, one colour per seed), the nearest-seed Voronoi
boundaries (red dashed, no fill), the targets that occupy them, and each cell's
seed centroid.

The Voronoi layer is the **decision boundary the accuracy numbers are computed
against**, not decoration: `classify.py` labels each prediction by `argmin` over
great-circle distances to the K seeds, so top-1 classification is nearest-seed
labelling and its boundary is the Voronoi diagram of the seeds. `margin_km` in
`seeds.csv` is the scalar summary of the same thing — half the distance to the
nearest other seed, i.e. how far a prediction may drift before it is scored
wrong. Drawing it makes the operator reading direct: a prediction landing
anywhere inside one of these regions is scored as that region's class.

Note the two layers are at wildly different scales and that is the point. A
grid cell is ~45 km; a class region is hundreds of km wide. The answer space is
a coarse nearest-seed partition whose *seeds* are placed by a fine quantizer.

**There is no clustering algorithm here.** Occupied cell and seed are in
one-to-one correspondence: two targets share a class iff the grid puts them in
the same cell. That is worth stating on the figure, because the benchmark also
ships an older `clusters/` answer space built by radius-capped complete-linkage
agglomeration, and the two are not comparable.

What the map is for is the straddle cost (§7.3, §10): grid lines fall where the
grid falls, not where targets are sparse, so a facility group spanning one is
quantized into two adjacent cells and two classes. That reads instantly as two
touching filled cells and is hard to believe from a scalar.

Only *occupied* cells are drawn. At the default h3 `res=4` a cell is ~45 km, so
the full 288,122-cell grid (196,608 at HEALPix `nside=128`) would be both
unreadable and pointless at continental scale.

Reuses the repo's cartopy conventions from
`scripts/visualization/cluster/plot_ground_truth_clusters.py` and
`scripts/visualization/cluster/plot_targets_vps.py`.

Command: `plot-answer-space`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402

from scripts.analysis.v3.modules.grid import (  # noqa: E402
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    get_grid,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.answer_space import (  # noqa: E402
    AnswerSpace,
    load_answer_space,
)
from scripts.analysis.v3.modules.paths import (  # noqa: E402
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    discover_runs,
    resolve_run,
)

#: Continental-US window, matching `plot_targets_vps.US_MAINLAND_EXTENT`.
US_MAINLAND_EXTENT = (-125.0, -66.0, 24.0, 50.0)

MAP_PNG = "answer_space_map.png"

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

    lat = np.asarray(seeds["centroid_lat"], dtype=float)
    lon = np.asarray(seeds["centroid_lon"], dtype=float)
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


def plot_answer_space(
    space: AnswerSpace,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    cell_step: int = 8,
    target_size: float = 7.0,
    voronoi: bool = True,
) -> Path:
    """Draw class regions, occupied cells, their targets, and their seeds.

    `cell_step` is the number of interpolated points per cell edge; >1 follows
    the cell's curvature on the sphere instead of drawing a straight-line quad.

    `voronoi` adds the nearest-seed partition — the classifier's own top-1
    decision boundary. On by default; see `seed_voronoi`.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Polygon

    seeds = space.seeds
    assignments = space.assignments
    grid = get_grid(str(seeds["grid_scheme"].iloc[0]))
    resolution = int(seeds["grid_resolution"].iloc[0])

    if extent is None:
        extent = _auto_extent(
            assignments["target_lat"].to_numpy(), assignments["target_lon"].to_numpy()
        )

    fig = plt.figure(figsize=(14.0, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8")
    ax.add_feature(cfeature.LAND, facecolor="#f6f4ef")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#cccccc")

    colors = _seed_colors(len(seeds))

    # --- layer 0: nearest-seed class regions -------------------------------
    # Boundaries only, no fill — same choice as `_plot_voronoi_underlay` in
    # plot_ground_truth_clusters.py and VORONOI_LINE in cluster_world_map.js, so
    # partitions stay comparable across figures and the basemap stays legible.
    # zorder 3 keeps the lines crisp above the occupied cells (zorder 2) but
    # under the targets and seeds.
    partition = seed_voronoi(seeds, extent) if voronoi else None
    if partition is not None:
        ax.add_geometries(
            list(partition.cells),
            crs=partition.crs,
            facecolor="none",
            edgecolor="red",
            linewidth=0.5,
            linestyle="--",
            zorder=3,
        )

    # --- layer 1: occupied cells -------------------------------------------
    # Rings come back ragged (4-sided x cell_step for HEALPix, 6 for an H3
    # hexagon, 5 for one of its 12 pentagons), already in (lon, lat) degrees and
    # already made contiguous across the antimeridian — so nothing here may
    # assume a rectangular array or re-wrap longitudes.
    rings = grid.cell_boundaries(
        seeds["cell_id"].to_numpy(), resolution, step=cell_step
    )

    for i, ring in enumerate(rings):
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor=colors[i],
                edgecolor="#333333",
                linewidth=0.6,
                alpha=0.75,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
        )

    # --- layer 2: targets ---------------------------------------------------
    ax.scatter(
        assignments["target_lon"].to_numpy(),
        assignments["target_lat"].to_numpy(),
        s=target_size,
        c="#111111",
        marker="o",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        transform=ccrs.PlateCarree(),
        zorder=4,
        label=f"Target ({len(assignments):,})",
    )

    # --- layer 3: seeds -----------------------------------------------------
    # Drawn last so the seed-vs-targets offset — the intra-seed spread that
    # floors any error-distance figure — stays visible. Deliberately smaller
    # than a cell: at continental scale a 51 km cell is only a few pixels, and
    # an oversized marker would hide both the fill and its own targets.
    ax.scatter(
        seeds["centroid_lon"].to_numpy(),
        seeds["centroid_lat"].to_numpy(),
        s=14,
        c="#d62728",
        marker="x",
        linewidths=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label=f"Region ({len(seeds)})",
    )

    # The partition gets no legend entry: it is explained in the caption, and
    # `add_geometries` takes no label so it would need a proxy artist anyway.
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    pitch = grid.nominal_cell_km(resolution)
    n_singleton = int((seeds["n_targets"] == 1).sum())
    boundary_note = ""
    if partition is not None:
        boundary_note = (
            f" Red dashed lines are the top-1 decision boundary — a prediction "
            f"landing anywhere in a region is scored as that region's class; "
            f"median margin {float(seeds['margin_km'].median()):.0f} km."
        )
    ax.set_title(title or "Target answer space", fontsize=12)
    ax.annotate(
        f"{grid.name} {grid.resolution_arg}={resolution} (~{pitch:.0f} km cells) · "
        f"K={len(seeds)} occupied "
        f"cells = {len(seeds)} classes ({n_singleton} singleton) · "
        f"{len(assignments):,} targets\n"
        f"One occupied cell is exactly one class: targets share a class iff they "
        f"fall in the same cell. No clustering algorithm; cells are the quantizer."
        f"{boundary_note}",
        xy=(0.5, -0.03),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-answer-space")
    def plot_answer_space_cmd(
        run_id: str = typer.Option(
            None, help="Run to plot. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Plot every run under --outputs-root."
        ),
        answer_space: Path = typer.Option(
            None,
            help="Answer-space dir (from build-answer-space). Defaults to this "
                 "run's target-answer-space/<grid>-<resolution>/ under "
                 "--analysis-root.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        us_only: bool = typer.Option(
            False, "--us-only", help="Clamp the view to the continental US."
        ),
        extent: tuple[float, float, float, float] = typer.Option(
            (None, None, None, None),
            "--extent",
            help="LON_MIN LON_MAX LAT_MIN LAT_MAX. Overrides --us-only.",
        ),
        target_size: float = typer.Option(7.0, help="Target dot size."),
        no_voronoi: bool = typer.Option(
            False,
            "--no-voronoi",
            help="Skip the nearest-seed Voronoi overlay. On by default: it is "
                 "the classifier's own top-1 decision boundary, so the figure "
                 "is incomplete without it.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Static map of the answer space: occupied cells, targets, seeds.

        Writes answer_space_map.png into the answer-space dir.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and answer_space is not None:
            raise typer.BadParameter("--answer-space cannot be combined with --all-runs")

        chosen_extent: tuple[float, float, float, float] | None = None
        if extent and all(v is not None for v in extent):
            chosen_extent = tuple(float(v) for v in extent)  # type: ignore[assignment]
        elif us_only:
            chosen_extent = US_MAINLAND_EXTENT

        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        jobs = (
            [(r, None) for r in runs]
            if answer_space is not None
            else [(r, x) for r in runs for x in resolutions]
        )
        for run, want_res in jobs:
            space_dir = answer_space or run.answer_space_dir(
                root=analysis_root, grid=g.name, resolution=want_res
            )
            space = load_answer_space(space_dir)
            space_grid = str(space.seeds["grid_scheme"].iloc[0])
            space_res = int(space.seeds["grid_resolution"].iloc[0])
            label = f"{space_grid} {get_grid(space_grid).resolution_arg}={space_res}"
            out = plot_answer_space(
                space,
                space_dir / MAP_PNG,
                extent=chosen_extent,
                title=f"{run.run_id} — target answer space ({label})",
                target_size=target_size,
                voronoi=not no_voronoi,
            )
            typer.echo(
                f"{run.run_id}: {label} · K={space.n_seeds} classes "
                f"over {len(space.assignments):,} targets -> {out}"
            )
