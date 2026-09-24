"""The answer space on a map, one panel per rung of the ladder.

What v3's `answer_space_map.png` was for, redrawn for a grid where correctness is
containment. The two figures are not versions of each other:

* v3 drew **one** resolution and put the classifier's nearest-seed Voronoi
  partition over it in red dashes, because that partition *was* the decision
  boundary and no cell edge was. v4 has no partition to draw — the cell edge is
  the boundary — so the ink that carried it is free.
* v3 drew **occupied cells only**. v4 draws the whole lattice across the frame,
  which is affordable at continental scale (5,740 cells at nside 128) and is the
  only direct evidence for the claim the package rests on: the grid is fixed, not
  fitted. See `mapping.frame_cells`.

## Why all four rungs

A single-nside map throws away the thing v4 exists to measure. Resolution is the
tolerance dial, and the ladder is where the dataset's geometry shows itself: on
as01 the target cells go 18 -> 18 -> 17 -> 15 as the grid coarsens, and the number
of cells merging more than one distinct operator site goes 2 -> 2 -> 3 -> 5. This
is the figure behind `occupancy_by_resolution.healpix.csv`, the same relationship
v3's `distance_cdf.png` had to its two scalars.

So there is no `--nside`: a subset of the ladder is not this figure. The rungs
come from `healpix.NSIDE_LADDER`, and the 2x2 comes from there being four of them.

## The site dots are load-bearing

These meshes carry ~20 IP replicas per coordinate — 399 targets over 20 distinct
sites on as01 — so a per-target scatter would draw 20 marks on one pixel and say
nothing. Plotting **distinct coordinates** makes the merge legible directly: a
cell holding two dots is two operator sites quantized into one class, which is
the cost of the grid and the number `n_merged_cells` reports.

## Encoding

Target cells filled, VP cells outlined, both composing without a third hue. The
reasoning and the validated hexes are in `mapping.py`; the short version is that
VP cells outnumber target cells 4:1, so two fills of equal weight put the ink on
the wrong side of the figure.

Command: `plot-answer-space`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules import sites as S
from scripts.analysis.v4.modules import mapping as M
from scripts.analysis.v4.modules.bipartite import (
    OCCUPANCY_CSV,
    TG_CELLS_CSV,
    VP_CELLS_CSV,
)
from scripts.analysis.v4.modules.bipartite import BY_RESOLUTION_CSV
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths

FIGURE_PNG = "answer_space_map.healpix.png"
FIGURE_MANIFEST = "answer_space_map.healpix.manifest.json"

#: Panel grid. Four rungs, so 2x2 — a 1x4 row would give each panel a twentieth
#: of the page's width and a 4x1 column would leave the US frame's 2.3:1 aspect
#: swimming in margin.
_NROWS, _NCOLS = 2, 2

#: Inches. Sized so one panel is ~6.7 in wide, where a nside-128 cell is ~19 px:
#: below that the finest rung's target cells stop being visible at all.
_FIGSIZE = (14.0, 7.4)


def load_rung(run: RunPaths, nside: int, *, analysis_root: Path | None = None):
    """`(occupancy, target_cells, vp_cells)` for one rung of one run."""
    d = run.bipartite_dir(nside, root=analysis_root)
    frames = {}
    for name in (OCCUPANCY_CSV, TG_CELLS_CSV, VP_CELLS_CSV):
        path = d / name
        if not path.exists():
            raise MissingArtifactError(
                f"{path} missing; run `build-bipartite --run-id {run.run_id}` first"
            )
        frames[name] = pd.read_csv(path)
    return frames[OCCUPANCY_CSV], frames[TG_CELLS_CSV], frames[VP_CELLS_CSV]


def distinct_sites(targets: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct target **coordinate**, not per target.

    ~20 replicas share a coordinate on these meshes, so the per-target frame
    would put 20 coincident marks on one pixel. Deduplicating is what lets a
    two-dot cell be read as two merged sites.

    Keyed on `sites.SITE_COLUMNS` rather than on a literal pair, so this figure
    and the tables that count sites cannot come to disagree about what one is.
    The run id is not needed here: a figure is drawn for one run at a time, so
    every row already shares it.
    """
    return targets.drop_duplicates(list(S.SITE_COLUMNS))


def count_merged_cells(targets: pd.DataFrame) -> int:
    """Cells holding more than one distinct target coordinate.

    The quantization cost at this rung, as a scalar: every one of these is two
    or more operator sites that the grid has collapsed into a single class, and
    that no method can be asked to tell apart.
    """
    if targets.empty:
        return 0
    per_cell = distinct_sites(targets).groupby("cell_id").size()
    return int((per_cell > 1).sum())


def rung_counts(
    run: RunPaths, nside: int, *, analysis_root: Path | None = None
) -> dict:
    """The four numbers a panel title carries, and the manifest repeats."""
    occ, targets, vps = load_rung(run, nside, analysis_root=analysis_root)
    return {
        "nside": int(nside),
        "cell_km": round(H.nominal_cell_km(nside), 1),
        "n_target_cells": int((occ["n_targets"] > 0).sum()),
        "n_shared_cells": int(((occ["n_targets"] > 0) & (occ["n_vps"] > 0)).sum()),
        "n_vp_cells": int((occ["n_vps"] > 0).sum()),
        "n_sites": int(len(distinct_sites(targets))),
        "n_vps": int(len(vps)),
        "n_merged_cells": count_merged_cells(targets),
    }


def draw_panel(
    ax,
    run: RunPaths,
    nside: int,
    extent: tuple[float, float, float, float],
    *,
    analysis_root: Path | None = None,
) -> dict:
    """One rung. Returns its counts, so the caller need not reload the rung."""
    import cartopy.crs as ccrs

    occ, targets, vps = load_rung(run, nside, analysis_root=analysis_root)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    M.draw_basemap(ax)
    M.draw_lattice(ax, nside, extent)

    M.draw_cells(
        ax,
        occ.loc[occ["n_targets"] > 0, "cell_id"].to_numpy(),
        nside,
        facecolor=M.TARGET_FILL,
        edgecolor=M.TARGET_EDGE,
        linewidth=0.5,
        alpha=0.80,
        zorder=3,
    )
    # Over the fill, so a target cell that also holds a VP takes the blue border
    # and one that does not keeps its own. That difference is the figure's
    # answer to `share_of_target_cells_with_a_vp`.
    M.draw_cells(
        ax,
        occ.loc[occ["n_vps"] > 0, "cell_id"].to_numpy(),
        nside,
        facecolor="none",
        edgecolor=M.VP_EDGE,
        linewidth=1.1,
        zorder=4,
    )

    sites = distinct_sites(targets)
    ax.scatter(
        vps["vp_lon"],
        vps["vp_lat"],
        s=M.MARKER_AREA,
        c=M.INK_2,
        marker="^",
        alpha=0.85,
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.scatter(
        sites["target_lon"],
        sites["target_lat"],
        s=M.MARKER_AREA,
        c=M.INK,
        marker="o",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    counts = rung_counts(run, nside, analysis_root=analysis_root)
    ax.set_title(
        f"nside {counts['nside']} · {counts['cell_km']:.0f} km cells\n"
        f"{counts['n_target_cells']} target cells "
        f"({counts['n_shared_cells']} hold a VP) · "
        f"{counts['n_vp_cells']} VP cells · "
        f"{counts['n_merged_cells']} cell(s) merge >1 site",
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
        Patch(
            facecolor=M.TARGET_FILL,
            edgecolor=M.TARGET_EDGE,
            alpha=0.80,
            label="Target cell (a class)",
        ),
        Patch(facecolor="none", edgecolor=M.VP_EDGE, lw=1.1, label="VP cell"),
        Line2D([], [], ls="", marker="o", ms=4.2, color=M.INK, label="Target site"),
        Line2D([], [], ls="", marker="^", ms=4.2, color=M.INK_2, label="Vantage point"),
        Patch(
            facecolor="none", edgecolor=M.LATTICE_EDGE, label="HEALPix lattice (empty)"
        ),
    ]


#: Said on the figure rather than left to the manifest, because it is the one
#: reading the picture does not enforce on its own: a map of filled cells looks
#: like the output of a clustering, and this answer space is not one.
_CAPTION = (
    "The lattice is fixed by the grid, not fitted to the data: a class boundary "
    "falls where HEALPix falls. Two sites in one cell are one class; a "
    "prediction outside a target's own cell scores ring >= 1."
)


def render(
    run: RunPaths,
    out_dir: Path,
    *,
    extent: tuple[float, float, float, float] = M.US_MAINLAND_EXTENT,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> tuple[Path, list[dict]]:
    """Draw every rung into one figure. Returns `(png_path, per-rung counts)`."""
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = sorted({H.validate_nside(n) for n in nsides}, reverse=True)
    fig = plt.figure(figsize=_FIGSIZE)
    counts = []
    for i, nside in enumerate(rungs):
        ax = fig.add_subplot(_NROWS, _NCOLS, i + 1, projection=ccrs.PlateCarree())
        counts.append(
            draw_panel(ax, run, nside, extent, analysis_root=analysis_root)
        )

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=len(_legend_handles()),
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        f"{run.run_id} — the HEALPix answer space, across the ladder",
        fontsize=12,
        y=0.985,
    )
    fig.text(0.5, 0.055, _CAPTION, ha="center", fontsize=8, color=M.MUTED)
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.18, wspace=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / FIGURE_PNG
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png, counts


def _manifest(run: RunPaths, counts: list[dict], extent) -> str:
    body = {
        "figure": FIGURE_PNG,
        # No CSV twin is written: `build-bipartite` already emits the merged
        # curve one directory up, and every number on this figure comes from it.
        # A second copy would be a thing to keep in sync for no new information.
        "csv": f"../{BY_RESOLUTION_CSV}",
        "run": run.run_id,
        "source": run.source,
        "setup": run.setup,
        "extent": {
            "lon_min": extent[0],
            "lon_max": extent[1],
            "lat_min": extent[2],
            "lat_max": extent[3],
        },
        "rungs": counts,
        "encoding": {
            "target_cell": f"{M.TARGET_FILL} fill, {M.TARGET_EDGE} edge",
            "vp_cell": f"{M.VP_EDGE} outline, never filled",
            "overlap": (
                "compositional — a cell holding both reads as the orange fill "
                "inside the blue outline. No third hue and no alpha blend, "
                "whose result is neither predictable nor validatable."
            ),
            "lattice": f"{M.LATTICE_EDGE}, every cell in the frame",
            "markers": (
                "one size for both kinds: a target site and a VP are two "
                "categories, not two magnitudes"
            ),
            "sites_not_targets": (
                "the dots are distinct target COORDINATES, not targets — these "
                "meshes carry ~20 IP replicas per coordinate, so a per-target "
                "scatter would draw 20 marks on one pixel"
            ),
        },
        "palette_validated": (
            "dataviz validate_palette.js --mode light --pairs all on "
            f"{M.TARGET_FILL},{M.VP_EDGE}: all five checks pass, worst CVD "
            "dE 24.7 (protan) / 32.7 (tritan), normal-vision dE 33.6. "
            "All-pairs rather than adjacent because a map is a choropleth form."
        ),
        "no_decision_boundary": (
            "v3 drew a red nearest-seed Voronoi partition here because that "
            "partition WAS the classifier's boundary. Under containment the "
            "cell edge is the boundary, so there is nothing further to draw."
        ),
        "reading_caveat": (
            "occupied cells alone would look like the output of a clustering. "
            "The empty lattice is the evidence that they are not: the grid is "
            "fixed before any target is seen."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_run(
    run: RunPaths,
    *,
    extent: tuple[float, float, float, float] = M.US_MAINLAND_EXTENT,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> Path:
    """Write the figure and its manifest into the run's `bipartite-graph/`.

    **Beside the bipartite artifacts, not the answer-space ones**, even though
    the figure is named for the answer space. Every number it draws comes from
    `cell_occupancy.csv` / `target_cells.csv` / `vp_cells.csv`, and the target
    cells there *are* the answer space's classes — one occupied cell is exactly
    one class, so `build-bipartite` alone is enough and `build-answer-space` is
    not a dependency. Putting the figure where its inputs are keeps that true
    rather than implying a dependency it does not have.

    One level above the rung directories, beside `occupancy_by_resolution`,
    because the figure spans the ladder and no single rung owns it.
    """
    out_dir = run.analysis_dir("bipartite-graph", root=analysis_root)
    png, counts = render(
        run, out_dir, extent=extent, nsides=nsides, analysis_root=analysis_root
    )
    (out_dir / FIGURE_MANIFEST).write_text(_manifest(run, counts, extent))
    return png


def auto_extent_for_run(
    run: RunPaths, *, nside: int = H.DEFAULT_NSIDE, analysis_root: Path | None = None
) -> tuple[float, float, float, float]:
    """Frame derived from the run's own targets and VPs, padded.

    The default frame is the continental US because that is where these meshes
    live, but it is a *choice* and a run outside it would otherwise render an
    empty map. `--auto-extent` picks this instead.
    """
    _, targets, vps = load_rung(run, nside, analysis_root=analysis_root)
    lats = np.concatenate([targets["target_lat"], vps["vp_lat"]])
    lons = np.concatenate([targets["target_lon"], vps["vp_lon"]])
    return M.auto_extent(lats, lons)
