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

The frame, the partition and the three layers live in `modules/mapping.py`,
shared with `map_bipartite.py`, and are **re-exported here** so this module
stays the one import name for the answer-space map. Reuses the repo's cartopy
conventions from `scripts/visualization/cluster/plot_ground_truth_clusters.py`
and `scripts/visualization/cluster/plot_targets_vps.py`.

Command: `plot-answer-space`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    get_grid,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.answer_space import (
    AnswerSpace,
    load_answer_space,
)
from scripts.analysis.v3.modules.mapping import (
    _EDGE_SAMPLES,
    _PAD_FRAC,
    _VORONOI_PAD_FRAC,
    _VORONOI_SEGMENT_M,
    US_MAINLAND_EXTENT,
    SeedVoronoi,
    _auto_extent,
    _projected_frame_box,
    _seed_colors,
    draw_basemap,
    draw_cells,
    draw_voronoi,
    seed_voronoi,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    discover_runs,
    resolve_run,
)

#: Re-exported for callers and tests that have always imported them from here.
__all__ = [
    "MAP_PNG",
    "US_MAINLAND_EXTENT",
    "SeedVoronoi",
    "_EDGE_SAMPLES",
    "_PAD_FRAC",
    "_VORONOI_PAD_FRAC",
    "_VORONOI_SEGMENT_M",
    "_auto_extent",
    "_projected_frame_box",
    "_seed_colors",
    "plot_answer_space",
    "register",
    "seed_voronoi",
]

MAP_PNG = "answer_space_map.png"


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
    decision boundary. On by default; see `mapping.seed_voronoi`.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

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
    draw_basemap(ax)

    colors = _seed_colors(len(seeds))

    # --- layer 0: nearest-seed class regions -------------------------------
    # zorder 3 keeps the lines crisp above the occupied cells (zorder 2) but
    # under the targets and seeds.
    partition = seed_voronoi(seeds, extent) if voronoi else None
    if partition is not None:
        draw_voronoi(ax, partition)

    # --- layer 1: occupied cells -------------------------------------------
    draw_cells(
        ax, grid, seeds["cell_id"].to_numpy(), resolution, colors=colors, step=cell_step
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
