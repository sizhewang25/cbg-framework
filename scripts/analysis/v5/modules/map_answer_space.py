"""The answer space on a map: both partitions, one panel per rung.

v4 drew the grid alone, because the grid was its only partition. v5 grades
every prediction against two, so the map draws both at each rung:

* the **grid** -- the full HEALPix lattice across the frame, TG grids filled;
* the **cells** -- Voronoi cells of the seeds, drawn as boundaries and bounded
  by the **landmass**, whose buffered outline is drawn dashed. Outside that
  outline a prediction is `outland`.

Sites are dots and seeds are crosses: where a cross sits between two dots, two
sites within one `grid_km` were grouped into one seed (EWR and JFK at
nside 128), and the cell boundary follows the seed, not the sites.

## Why all four rungs

`grid_km` is the tolerance dial for both partitions at once -- the grid pitch,
the seed diameter and the landmass buffer are the same number -- so how the
two coarsen together is the figure. On as01 the grids go 18 -> 18 -> 17 -> 15
and the seeds 18 -> 18 -> 16 -> 12 while the landmass swells from 51 km to
407 km past the border. A single rung is a different, smaller figure, so there
is no `--nside`.

## The cells are drawn, not scored, from polygons

`classify` labels a prediction by great-circle nearest seed. The polygons here
are a planar Voronoi in the landmass projection; `cells.agreement_with_nearest_seed`
measures how often they agree (99.4-99.7% of inland points on the three
meshes) and the manifest records it per rung.

Command: `plot-answer-space`. Needs `build-answer-space`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analysis.v5.modules import cells as CL
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import mapping as M
from scripts.analysis.v5.modules.answer_space import (
    META_JSON,
    SWEEP_CSV,
    AnswerSpace,
    load_answer_space,
)
from scripts.analysis.v5.modules.landmass import load_landmass
from scripts.analysis.v5.modules.paths import ANSWER_SPACE_KIND, MissingArtifactError, RunPaths

FIGURE_PNG = "answer_space_map.healpix.png"
FIGURE_MANIFEST = "answer_space_map.healpix.manifest.json"

_NROWS, _NCOLS = 2, 2
_FIGSIZE = (14.0, 7.6)


def load_rung(run: RunPaths, nside: int, *, analysis_root: Path | None = None) -> AnswerSpace:
    d = run.answer_space_dir(nside, root=analysis_root)
    if not (d / META_JSON).exists():
        raise MissingArtifactError(
            f"{d / META_JSON} missing; run `build-answer-space --run-id {run.run_id}` first"
        )
    return load_answer_space(d)


def rung_counts(space: AnswerSpace, polygons: dict, agreement: float) -> dict:
    """The numbers a panel title carries, and the manifest repeats."""
    grids_multi = int((space.grids["n_sites"] > 1).sum())
    return {
        "nside": space.nside,
        "grid_km": round(space.grid_km, 1),
        "n_sites": int(space.meta["n_sites"]),
        "n_grids": int(space.meta["n_grids"]),
        "n_grids_with_several_sites": grids_multi,
        "n_seeds": int(space.meta["n_seeds"]),
        "n_sites_merged": int(space.meta["n_sites_merged"]),
        "n_cells_drawn": len(polygons),
        "landmass_buffer_km": space.meta["landmass"]["buffer_km"],
        "cell_polygon_agreement": round(agreement, 4),
    }


def draw_panel(ax, space: AnswerSpace, extent) -> dict:
    """One rung. Returns its counts."""
    import cartopy.crs as ccrs

    landmass = load_landmass(space.grid_km)
    polygons = CL.cell_polygons(space.seeds, landmass)
    agreement = CL.agreement_with_nearest_seed(space.seeds, landmass, polygons)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    M.draw_basemap(ax)
    M.draw_lattice(ax, space.nside, extent)
    M.draw_grids(
        ax,
        space.grids["grid_id"].to_numpy(),
        space.nside,
        facecolor=M.TARGET_FILL,
        edgecolor=M.TARGET_EDGE,
        linewidth=0.5,
        alpha=0.80,
        zorder=3,
    )
    M.draw_cells(ax, polygons, zorder=4)
    M.draw_geometry(
        ax,
        landmass.to_lonlat(landmass.geometry),
        edgecolor=M.LANDMASS_EDGE,
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        zorder=5,
    )
    ax.scatter(
        space.sites["site_lon"], space.sites["site_lat"],
        s=M.MARKER_AREA, c=M.INK, marker="o", linewidths=0,
        transform=ccrs.PlateCarree(), zorder=6,
    )
    ax.scatter(
        space.seeds["seed_lon"], space.seeds["seed_lat"],
        s=M.MARKER_AREA * 2.2, c=M.INK, marker="x", linewidths=0.9,
        transform=ccrs.PlateCarree(), zorder=7,
    )

    counts = rung_counts(space, polygons, agreement)
    ax.set_title(
        f"nside {counts['nside']} · grid_km {counts['grid_km']:.0f}\n"
        f"{counts['n_grids']} TG grids ({counts['n_grids_with_several_sites']} hold >1 site) · "
        f"{counts['n_seeds']} seeds from {counts['n_sites']} sites · "
        f"landmass +{counts['landmass_buffer_km']:.0f} km",
        fontsize=8.5,
        color=M.INK,
        linespacing=1.4,
        pad=5,
    )
    return counts


def _legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=M.TARGET_FILL, edgecolor=M.TARGET_EDGE, alpha=0.80, label="TG grid"),
        Line2D([], [], color=M.CELL_EDGE, lw=0.9, label="Cell boundary (serving region)"),
        Line2D([], [], color=M.LANDMASS_EDGE, lw=1.1, ls=(0, (4, 2)), label="Landmass (+grid_km)"),
        Line2D([], [], ls="", marker="o", ms=4.2, color=M.INK, label="Site"),
        Line2D([], [], ls="", marker="x", ms=5.5, mew=0.9, color=M.INK, label="Seed"),
        Patch(facecolor="none", edgecolor=M.LATTICE_EDGE, label="HEALPix lattice (empty)"),
    ]


_CAPTION = (
    "Two partitions at one tolerance, grid_km. ring grades a prediction against the "
    "grid; cell_label against the cells: true / wrong by nearest seed inside the "
    "dashed landmass, outland beyond it."
)


def render(
    run: RunPaths,
    out_dir: Path,
    *,
    extent: tuple[float, float, float, float] = M.US_MAINLAND_EXTENT,
    nsides: tuple[int, ...] = G.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> tuple[Path, list[dict]]:
    """Draw every rung into one figure. Returns `(png_path, per-rung counts)`."""
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = sorted({G.validate_nside(n) for n in nsides}, reverse=True)
    fig = plt.figure(figsize=_FIGSIZE)
    counts = []
    for i, nside in enumerate(rungs):
        ax = fig.add_subplot(_NROWS, _NCOLS, i + 1, projection=ccrs.PlateCarree())
        counts.append(draw_panel(ax, load_rung(run, nside, analysis_root=analysis_root), extent))

    handles = _legend_handles()
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5,
        frameon=False, bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(f"{run.run_id} — the TG answer space: grids and cells", fontsize=12, y=0.985)
    fig.text(0.5, 0.055, _CAPTION, ha="center", fontsize=8, color=M.MUTED)
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.20, wspace=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / FIGURE_PNG
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png, counts


def _manifest(run: RunPaths, counts: list[dict], extent) -> str:
    body = {
        "figure": FIGURE_PNG,
        "csv": SWEEP_CSV,
        "run": run.run_id,
        "source": run.source,
        "setup": run.setup,
        "extent": dict(zip(("lon_min", "lon_max", "lat_min", "lat_max"), extent)),
        "rungs": counts,
        "encoding": {
            "tg_grid": f"{M.TARGET_FILL} fill, {M.TARGET_EDGE} edge",
            "lattice": f"{M.LATTICE_EDGE}, every grid in the frame",
            "cell": f"{M.CELL_EDGE} boundary: Voronoi cell of a seed, clipped to the landmass",
            "landmass": f"{M.LANDMASS_EDGE} dashed: US mainland buffered by grid_km",
            "site": "dot, one per distinct TG coordinate",
            "seed": "cross, spherical centroid of its grouped sites",
        },
        "cell_polygons": (
            "planar Voronoi in EPSG:5070, edges densified before reprojection; "
            "cell_polygon_agreement is the share of sampled inland points whose "
            "drawn cell is their great-circle nearest seed, the rule classify scores"
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_run(
    run: RunPaths,
    *,
    extent: tuple[float, float, float, float] = M.US_MAINLAND_EXTENT,
    nsides: tuple[int, ...] = G.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> Path:
    """Write the figure and manifest into `answer-space/`, beside `sweep.csv`:
    one level above the rung directories, because the figure spans the ladder."""
    out_dir = run.analysis_dir(ANSWER_SPACE_KIND, root=analysis_root)
    png, counts = render(run, out_dir, extent=extent, nsides=nsides, analysis_root=analysis_root)
    (out_dir / FIGURE_MANIFEST).write_text(_manifest(run, counts, extent))
    return png


def auto_extent_for_run(
    run: RunPaths, *, nside: int = G.DEFAULT_NSIDE, analysis_root: Path | None = None
) -> tuple[float, float, float, float]:
    """Frame from the run's own sites, padded."""
    space = load_rung(run, nside, analysis_root=analysis_root)
    return M.auto_extent(
        np.asarray(space.sites["site_lat"]), np.asarray(space.sites["site_lon"])
    )
