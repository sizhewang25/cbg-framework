"""Static map of the target answer space: occupied HEALPix cells + targets.

Renders what `build-answer-space` produced, so the §7.3 quantization can be
looked at rather than inferred from `meta.json`. Three layers, bottom to top:
occupied grid cells (filled, one colour per seed), the targets that occupy
them, and each cell's seed centroid.

**There is no clustering algorithm here.** Occupied cell and seed are in
one-to-one correspondence: two targets share a class iff `ang2pix` returns the
same cell id. That is worth stating on the figure, because the benchmark also
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

from pathlib import Path

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


def plot_answer_space(
    space: AnswerSpace,
    out_path: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    cell_step: int = 8,
    target_size: float = 7.0,
) -> Path:
    """Draw occupied cells, their targets, and their seeds onto one map.

    `cell_step` is the number of interpolated points per cell edge; >1 follows
    the cell's curvature on the sphere instead of drawing a straight-line quad.
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

    # --- layer 1: occupied cells -------------------------------------------
    # Rings come back ragged (4-sided x cell_step for HEALPix, 6 for an H3
    # hexagon, 5 for one of its 12 pentagons), already in (lon, lat) degrees and
    # already made contiguous across the antimeridian — so nothing here may
    # assume a rectangular array or re-wrap longitudes.
    rings = grid.cell_boundaries(
        seeds["cell_id"].to_numpy(), resolution, step=cell_step
    )

    colors = _seed_colors(len(seeds))
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
        label=f"target ({len(assignments):,})",
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
        label=f"seed ({len(seeds)})",
    )

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    pitch = grid.nominal_cell_km(resolution)
    n_singleton = int((seeds["n_targets"] == 1).sum())
    ax.set_title(title or "Target answer space", fontsize=12)
    ax.annotate(
        f"{grid.name} {grid.resolution_arg}={resolution} (~{pitch:.0f} km cells) · "
        f"K={len(seeds)} occupied "
        f"cells = {len(seeds)} classes ({n_singleton} singleton) · "
        f"{len(assignments):,} targets\n"
        f"One occupied cell is exactly one class: targets share a class iff they "
        f"fall in the same cell. No clustering algorithm; cells are the quantizer.",
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
            )
            typer.echo(
                f"{run.run_id}: {label} · K={space.n_seeds} classes "
                f"over {len(space.assignments):,} targets -> {out}"
            )
