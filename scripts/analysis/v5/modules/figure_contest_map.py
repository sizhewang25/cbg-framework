"""Where two methods disagree, site by site: one map per mesh.

The outcome bars say Spotter places 63.7% of predictions in the right cell and
Octant-Hull 63.1%, which reads as a tie. It is not a tie -- it is two methods
that are right about different places. This figure draws that: one panel per
mesh, one marker per site, shaped and coloured by which of the two methods put
more of the site's ~20 replicas in the correct cell.

## Four categories, no threshold

`tied`, `<A> wins`, `<B> wins`, `neither`. A site's category comes from
comparing the two correct counts directly -- see `contest.categorize`, and the
Los Angeles regression it exists to prevent. The label under each marker is
`k_a | k_b | n`, so the margin behind every category is on the page and a
reader can re-derive the category from the drawing.

`neither` is kept separate from `tied` on purpose. A site both methods missed
entirely is concordant, like a tie, but it says something different about the
answer space -- it is a place neither estimator reaches -- and folding the two
together would inflate the tie count with failures.

## Colour is the method's, not the category's

`<A> wins` takes method A's hue from `methods.LABEL_HUES` and `<B> wins`
method B's, so this figure, the peripherality boxplots and the error CDF all
paint Spotter the same red and Octant-Hull the same green. `tied` is grey and
`neither` is hollow: neither is a method, so neither takes a method's colour.

## No statistics on the figure

The exploratory prototype carried a per-panel box with the category counts and
the McNemar p-value. Both are gone from the drawing and live in the manifest
and the CSV twin instead. A p-value in a corner invites the reading that the
map *is* the test, and the test is over sites while the map is over places.

McNemar is exact and over sites -- the discordant sites are the sample -- and
is reported with the smallest p that many discordant sites could have
produced. On these meshes only `as01` resolves.

## What this figure is not evidence for

It compares two methods on the unbounded cell metric, which the evaluation
section argues is broken: a nearest-seed verdict credits a prediction at any
distance. The figure shows *what that metric rewards*. Read it beside the
error-distance figure, never instead of it.

Command: `plot-contest-map`. Needs `build-answer-space` and `classify` on
every run. Writes `_cross/contest-map/<datasets>[@<arm>]/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.analysis.v5.modules import cells as CL  # noqa: E402
from scripts.analysis.v5.modules import classify as C  # noqa: E402
from scripts.analysis.v5.modules import contest as CT  # noqa: E402
from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import figure_outcome_map as FOM  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules import mapping as M  # noqa: E402
from scripts.analysis.v5.modules.map_answer_space import load_rung  # noqa: E402
from scripts.analysis.v5.modules.methods import (  # noqa: E402
    method_colors,
    method_label,
    method_term_table,
)
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402

#: `_cross/<KIND>/<datasets>[@<arm>]/`.
KIND = "contest-map"

#: Which rung's answer space and `*_tgs.parquet` are read. Off the ladder
#: rather than spelled, so a second rung cannot desync seeds from labels.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: The pair the paper argues about. A default, not a constraint -- the module
#: takes any two scored methods.
DEFAULT_METHOD_A = "spotter_cbg"
DEFAULT_METHOD_B = "octant_cbg_hull"

#: The same frame as the outcome maps, so the two can be laid side by side and
#: a site found in both. Every site on these meshes is well inside it; the
#: margin is there because the cells are not.
DEFAULT_EXTENT: tuple[float, float, float, float] = FOM.DEFAULT_EXTENT

#: `category -> (marker, area in points^2)`. Four shapes, not four colours
#: alone: the figure must survive greyscale printing and colour-vision
#: deficiency, and the two decisive categories take the two shapes hardest to
#: confuse at this size. `neither` is the only unfilled one.
CATEGORY_MARKER: dict[str, tuple[str, float]] = {
    CT.TIED: ("o", 30.0),
    CT.A_WINS: ("^", 36.0),
    CT.B_WINS: ("s", 28.0),
    CT.NEITHER: ("X", 34.0),
}

#: `neither`'s fill. Neutral, and darker than `tied`'s: neither category is a
#: method, so neither takes a method's hue, and the two separate by lightness
#: as well as by shape. White was tried first and vanished into `mapping.LAND`
#: (`#f8f7f3`), which is where nearly every site is.
NEITHER_FILL = M.INK_2

MARKER_EDGE_PT = 0.6

#: The site label's drawn box, in **inches**. Measured, not guessed:
#: `20|20|20` at `LABEL_FONT_PT` renders 0.398 x 0.090 in. Kept in inches
#: rather than degrees because a point size is absolute and a degree is not --
#: `label_scale` turns these into degrees for whatever frame and panel width
#: the figure is actually drawn at, so narrowing the panel makes the label
#: bigger in map units and the relaxation knows it.
LABEL_W_IN = 0.398
LABEL_H_IN = 0.090
LABEL_FONT_PT = 6.4

#: Where a label sits before anything pushes it, in label heights: directly
#: below its marker, by slightly under one box so the two do not touch.
LABEL_OFFSET_UNITS = (0.0, -0.95)

#: All four in units of the label box, so they are read as geometry rather
#: than as tuning.
#:
#: Two axis-aligned unit boxes clear each other once their centres are 1.0
#: apart *along one axis*, but the relaxation separates in the Euclidean
#: metric, where a pair can sit 1.0 apart diagonally and still overlap in
#: both. `MIN_SEP` is therefore the box diagonal, sqrt(2), rounded up: it
#: over-separates a side-by-side pair by 40% and guarantees the diagonal one,
#: and over-separated labels are the failure worth having.
#:
#: `OBSTACLE_SEP` cannot take the same treatment. The Euclidean bound that
#: cleared a marker vertically would push labels 6.5 deg clear of markers
#: *horizontally*, where 2.9 deg suffices, and the map has no room for that.
#: 0.85 is the compromise: it clears a marker below a label (the common case,
#: since a label starts directly under its own) and pushes a little harder
#: than it needs to sideways.
MIN_SEP_UNITS = 1.45
OBSTACLE_SEP_UNITS = 0.85
FRAME_PAD_UNITS = 0.55
LEADER_MIN_UNITS = 0.45


PNG_NAME = "contest_map.{pair}.png"
CSV_NAME = "contest_map.{pair}.csv"
MANIFEST_NAME = "contest_map.{pair}.manifest.json"

#: Panel geometry. Height follows the frame's aspect, so the figure carries no
#: empty band above or below the maps at any `--extent`.
_PANEL_W_IN = 5.4
_TITLE_IN = 0.55
_LEGEND_IN = 0.42

_Z_CELLS = 4.0
_Z_LEADER = 5.0
_Z_MARKER = 6.0
_Z_LABEL = 7.0

_HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
_LEADER_HALO = [pe.withStroke(linewidth=1.6, foreground="white")]


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/contest-map/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def pair_slug(method_a: str, method_b: str) -> str:
    """`SPO-vs-OCT-H` -- the terms, so the filename names the comparison."""
    return f"{method_label(method_a)}-vs-{method_label(method_b)}".replace(" ", "_")


def category_ink(method_a: str, method_b: str) -> dict[str, str]:
    """`category -> facecolor`, with each win taking its own method's hue."""
    hues = method_colors([method_a, method_b])
    return {
        CT.TIED: M.MUTED,
        CT.A_WINS: hues[method_a],
        CT.B_WINS: hues[method_b],
        CT.NEITHER: NEITHER_FILL,
    }


def category_names(method_a: str, method_b: str) -> dict[str, str]:
    """`category -> legend label`, in the methods' terms."""
    return {
        CT.TIED: "Tied",
        CT.A_WINS: f"{method_label(method_a)} wins",
        CT.B_WINS: f"{method_label(method_b)} wins",
        CT.NEITHER: "Neither",
    }


# -- loading --------------------------------------------------------------


@dataclass(frozen=True)
class ContestData:
    """The scored frames and seeds for one method pair over several meshes."""

    run_ids: list[str]
    method_a: str
    method_b: str
    frames: dict[tuple[str, str], pd.DataFrame]
    seeds: dict[str, pd.DataFrame]
    nside: int


def load(
    runs: list[RunPaths],
    *,
    method_a: str = DEFAULT_METHOD_A,
    method_b: str = DEFAULT_METHOD_B,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> ContestData:
    """Both methods' `*_tgs.parquet` and each run's seeds.

    Both methods must be scored in every run. A mesh carrying one of them
    would contribute a panel of half-contests, and the pooled counts the
    manifest reports would be over a population that changes per panel.
    """
    if method_a == method_b:
        raise ValueError(f"a contest needs two methods, got {method_a!r} twice")
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    seeds: dict[str, pd.DataFrame] = {}
    for run in runs:
        seeds[run.run_id] = load_rung(run, nside, analysis_root=analysis_root).seeds
        for m in (method_a, method_b):
            path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=m)
            if not path.exists():
                have = FOM.scored_methods(run, nside, analysis_root=analysis_root)
                raise MissingArtifactError(
                    f"{m} is not scored in {run.run_id} at nside={nside} "
                    f"({path} missing; have {have}). Run "
                    f"`classify --run-id {run.run_id}` first."
                )
            frames[(run.run_id, m)] = pd.read_parquet(path, columns=list(CT.TG_COLUMNS))
    return ContestData(
        run_ids=[r.run_id for r in runs],
        method_a=method_a,
        method_b=method_b,
        frames=frames,
        seeds=seeds,
        nside=int(nside),
    )


# -- the numbers ----------------------------------------------------------


def contest_table(data: ContestData) -> pd.DataFrame:
    """One row per `(run_id, site_id)`, in panel order. The whole figure."""
    parts = [
        CT.site_contest(
            data.frames[(run_id, data.method_a)],
            data.frames[(run_id, data.method_b)],
            run_id=run_id,
            method_a=data.method_a,
            method_b=data.method_b,
        )
        for run_id in data.run_ids
    ]
    return pd.concat(parts, ignore_index=True)


def panel_counts(table: pd.DataFrame, data: ContestData) -> list[dict]:
    """Per-panel category counts and McNemar, then the pooled line last.

    The pooled line is reported and immediately qualified: the three meshes
    share facilities -- 65 sites at 43 distinct coordinates -- so pooled sites
    are not independent and the pooled p-value is optimistic. See the manifest.
    """
    common = {
        "method_a": data.method_a,
        "method_b": data.method_b,
        "label_a": method_label(data.method_a),
        "label_b": method_label(data.method_b),
    }
    out = [
        CT.contest_counts(
            table[table["run_id"] == run_id],
            run_id=run_id,
            dataset=cross.short_dataset(run_id),
            **common,
        )
        for run_id in data.run_ids
    ]
    out.append(
        CT.contest_counts(
            table, run_id="__pooled__", dataset=cross.dataset_slug(data.run_ids), **common
        )
    )
    return out


#: Panel-level columns, broadcast onto every site row of that panel. The twin
#: is one table rather than two so a single `groupby("dataset")` reproduces
#: both the drawing and the statistic behind it.
_PANEL_COLUMNS = (
    "n_sites", f"n_{CT.TIED}", f"n_{CT.A_WINS}", f"n_{CT.B_WINS}", f"n_{CT.NEITHER}",
    "mcnemar_b", "mcnemar_c", "mcnemar_n_discordant", "mcnemar_p", "mcnemar_p_floor",
)


def csv_columns() -> list[str]:
    """The CSV twin's columns: what the panel concludes, then each site."""
    return [
        "run_id", "dataset", "method_a", "method_b", "label_a", "label_b",
        *_PANEL_COLUMNS,
        "site_id", "tg_lat", "tg_lon", "tg_seed_id",
        "k_a", "n_a", "k_b", "n_b", "n_tgs", "category",
    ]


def build_csv(table: pd.DataFrame, counts: list[dict]) -> pd.DataFrame:
    """The per-site table with its panel's counts joined onto every row.

    The pooled line is deliberately absent: it is not a site, and a row that
    is not a site in a per-site table is a row somebody will sum. It is in the
    manifest, and `groupby("dataset")` over this table reproduces it.
    """
    panel = pd.DataFrame(
        [c for c in counts if c["run_id"] != "__pooled__"]
    )[["run_id", *_PANEL_COLUMNS]]
    out = table.merge(panel, on="run_id", how="left", validate="many_to_one")
    return out.reindex(columns=csv_columns())


# -- drawing --------------------------------------------------------------


def label_scale(
    extent: tuple[float, float, float, float], panel_width: float = _PANEL_W_IN
) -> np.ndarray:
    """Degrees -> label boxes. The space the relaxation and its audit share.

    A panel preserves aspect, so degrees per inch is the same along both axes
    and one factor converts both sides of the label box. Narrow the panel and
    the box grows in degrees, which is exactly the effect that makes a
    three-column figure impossible to label at paper width.
    """
    deg_per_in = (extent[1] - extent[0]) / float(panel_width)
    return np.array([1.0 / (LABEL_W_IN * deg_per_in), 1.0 / (LABEL_H_IN * deg_per_in)])


def label_crowding(
    anchors: np.ndarray,
    placed: np.ndarray,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    panel_width: float = _PANEL_W_IN,
) -> dict:
    """How well the relaxation did, in label boxes. For the manifest.

    The separations it *achieved*, not the ones it was asked for. A dense
    cluster can make the two constraints unsatisfiable -- as03's north-east
    puts six sites inside three degrees -- and when that happens the figure
    should say so with a number rather than leave the reader to squint.

    `n_overlapping_label_pairs` is the one that matters and is measured in the
    Chebyshev metric, because that is what overlapping boxes means and it is
    what the eye sees; the relaxation itself separates in Euclidean, which is
    why `MIN_SEP_UNITS` is the diagonal.
    """
    scale = label_scale(extent, panel_width)
    a = np.asarray(anchors, dtype=float) * scale
    p = np.asarray(placed, dtype=float) * scale
    if len(a) < 2:
        return {"min_label_sep_units": None, "n_overlapping_label_pairs": 0,
                "min_label_marker_units": None}
    sep = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    np.fill_diagonal(sep, np.inf)
    dx = np.abs(p[:, None, 0] - p[None, :, 0])
    dy = np.abs(p[:, None, 1] - p[None, :, 1])
    overlap = (dx < 1.0) & (dy < 1.0)
    np.fill_diagonal(overlap, False)
    marker = np.linalg.norm(p[:, None] - a[None, :], axis=-1)
    return {
        "min_label_sep_units": round(float(sep.min()), 3),
        "n_overlapping_label_pairs": int(overlap.sum() // 2),
        "min_label_marker_units": round(float(marker.min()), 3),
    }


def place_site_labels(
    anchors: np.ndarray,
    extent: tuple[float, float, float, float],
    panel_width: float = _PANEL_W_IN,
) -> np.ndarray:
    """Label positions for the site markers, relaxed in units of a label box.

    `FOM.place_labels` separates points isotropically, and a site label is
    about four times wider than it is tall: one separation that clears two
    labels side by side would fling them apart vertically, and one that sits
    right vertically would leave them overlapping horizontally. So the whole
    relaxation runs in `label_scale`'s space, where a label is a unit box and
    a single isotropic rule is the correct one, and the result is scaled back.

    The anchors are passed as obstacles as well: every marker repels every
    label, including its own, which is what keeps a count off a neighbouring
    site's marker.
    """
    a = np.asarray(anchors, dtype=float)
    scale = label_scale(extent, panel_width)
    scaled = a * scale
    placed = FOM.place_labels(
        scaled,
        (
            extent[0] * scale[0], extent[1] * scale[0],
            extent[2] * scale[1], extent[3] * scale[1],
        ),
        min_sep=MIN_SEP_UNITS,
        pad=FRAME_PAD_UNITS,
        offset=LABEL_OFFSET_UNITS,
        obstacles=scaled,
        obstacle_sep=OBSTACLE_SEP_UNITS,
    )
    return placed / scale


def draw_panel(
    ax,
    rows: pd.DataFrame,
    polygons: dict,
    *,
    method_a: str,
    method_b: str,
    title: str,
    extent: tuple[float, float, float, float],
    panel_width: float = _PANEL_W_IN,
    labels: bool = True,
) -> dict:
    """One mesh's map. Returns how crowded the labels ended up.

    `labels=False` draws the markers alone. The per-site counts are eight
    characters at a fixed point size, so on a narrow panel they cannot be
    separated at all -- see `render` -- and the spatial pattern, which is what
    the figure argues, survives without them. The counts are in the CSV twin
    either way.
    """
    import cartopy.crs as ccrs

    ink = category_ink(method_a, method_b)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    M.draw_basemap(ax)
    M.draw_cells(ax, polygons, zorder=_Z_CELLS)

    for category in CT.CATEGORIES:
        hit = rows[rows["category"] == category]
        if not len(hit):
            continue
        marker, area = CATEGORY_MARKER[category]
        ax.scatter(
            hit["tg_lon"], hit["tg_lat"], s=area, marker=marker,
            facecolors=ink[category], edgecolors=M.INK,
            linewidths=MARKER_EDGE_PT,
            transform=ccrs.PlateCarree(), zorder=_Z_MARKER,
        )

    if not labels or not len(rows):
        ax.set_title(title, fontsize=8.5)
        return {"n_labels": 0}
    anchors = np.c_[rows["tg_lon"].to_numpy(), rows["tg_lat"].to_numpy()]
    placed = place_site_labels(anchors, extent, panel_width)
    scale = label_scale(extent, panel_width)
    for (ax0, ay0), (lx, ly), row in zip(anchors, placed, rows.itertuples()):
        moved = np.hypot((lx - ax0) * scale[0], (ly - ay0) * scale[1])
        if moved > LEADER_MIN_UNITS:
            ax.plot(
                [ax0, lx], [ay0, ly], lw=0.5, color=M.INK_2, alpha=0.9,
                transform=ccrs.PlateCarree(), zorder=_Z_LEADER,
                path_effects=_LEADER_HALO,
            )
        ax.text(
            lx, ly, f"{row.k_a}|{row.k_b}|{row.n_tgs}", fontsize=LABEL_FONT_PT,
            color=M.INK, ha="center", va="center", transform=ccrs.PlateCarree(),
            zorder=_Z_LABEL, path_effects=_HALO,
        )
    ax.set_title(title, fontsize=8.5)
    return {
        "n_labels": int(len(rows)),
        **label_crowding(anchors, placed, extent, panel_width),
    }


def legend_handles(method_a: str, method_b: str) -> list[Line2D]:
    """One proxy per category, in `contest.CATEGORIES` order."""
    ink = category_ink(method_a, method_b)
    names = category_names(method_a, method_b)
    return [
        Line2D(
            [], [], linestyle="none", marker=CATEGORY_MARKER[c][0],
            markerfacecolor=ink[c], markeredgecolor=M.INK, markeredgewidth=MARKER_EDGE_PT,
            markersize=np.sqrt(CATEGORY_MARKER[c][1]) * 1.15, label=names[c],
        )
        for c in CT.CATEGORIES
    ]


def figure_title(method_a: str, method_b: str, labels: bool = True) -> str:
    """The one line of prose on the figure.

    The label key is half of it, so it goes when the labels do -- a key to
    something that is not drawn is worse than no key.
    """
    a, b = method_label(method_a), method_label(method_b)
    title = f"{a} vs {b} per site on the cell axis"
    if labels:
        title += f"  ·  label = {a} correct | {b} correct | total"
    return title


def render(
    data: ContestData,
    table: pd.DataFrame,
    out_png: Path,
    *,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    ncols: int | None = None,
    panel_width: float = _PANEL_W_IN,
    labels: bool = True,
    dpi: int = 200,
) -> tuple[Path, list[dict]]:
    """The meshes in reading order, `ncols` panels to a row.

    `ncols` defaults to one row of everything, which is how the figure is read
    on screen. `panel_width` and `labels` exist because that default cannot be
    printed: a site label is eight characters at a fixed point size, and

    * three panels side by side is a 16-inch figure, so scaling it to a
      two-column paper's 7-inch `\textwidth` takes the labels under 3 pt;
    * authoring it at 7 inches instead gives each panel 2.3 inches, where one
      label spans 11 degrees of a 65-degree frame and twenty of them cannot be
      separated at all.

    The two ways out, and both are one argument: `ncols=1` stacks the meshes
    so each panel keeps a printable width (at the cost of a tall figure), or
    `labels=False` drops the counts to the CSV twin and keeps the row. The
    spatial pattern is what this figure argues; the margins are the error
    figures' job.

    Cells are built once per run: the planar Voronoi is the slowest thing on
    the page and it belongs to the answer space, not to a method.
    """
    import cartopy.crs as ccrs

    n = len(data.run_ids)
    ncols = n if ncols is None else int(ncols)
    if ncols < 1:
        raise ValueError(f"ncols must be at least 1, got {ncols}")
    nrows = -(-n // ncols)
    lon_span = extent[1] - extent[0]
    lat_span = extent[3] - extent[2]
    panel_h = panel_width * lat_span / lon_span
    total_h = panel_h * nrows + _TITLE_IN * nrows + _LEGEND_IN
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_width * ncols, total_h),
        squeeze=False, subplot_kw={"projection": ccrs.PlateCarree()},
    )
    crowding = []
    for k, run_id in enumerate(data.run_ids):
        polygons = CL.cell_polygons(data.seeds[run_id], extent)
        drawn = draw_panel(
            axes[k // ncols, k % ncols],
            table[table["run_id"] == run_id],
            polygons,
            method_a=data.method_a,
            method_b=data.method_b,
            title=cross.short_dataset(run_id),
            extent=extent,
            panel_width=panel_width,
            labels=labels,
        )
        crowding.append({"dataset": cross.short_dataset(run_id), **drawn})
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].set_axis_off()

    fig.subplots_adjust(
        left=0.01, right=0.99, wspace=0.03, hspace=0.12,
        bottom=_LEGEND_IN / total_h, top=1.0 - _TITLE_IN / total_h,
    )
    fig.legend(
        handles=legend_handles(data.method_a, data.method_b),
        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(CT.CATEGORIES),
        frameon=False, fontsize=9, handletextpad=0.4, columnspacing=2.2,
    )
    fig.suptitle(
        figure_title(data.method_a, data.method_b, labels), fontsize=11, y=0.995
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png, crowding


def _manifest(
    data: ContestData,
    counts: list[dict],
    crowding: list[dict],
    *,
    extent: tuple[float, float, float, float],
    panel_width: float,
    labels: bool,
    png_name: str,
    csv_name: str,
) -> str:
    ink = category_ink(data.method_a, data.method_b)
    names = category_names(data.method_a, data.method_b)
    body = {
        "figure": png_name,
        "csv": csv_name,
        "method_a": data.method_a,
        "method_b": data.method_b,
        "datasets": cross.dataset_slug(data.run_ids),
        "run_ids": data.run_ids,
        "arm": cross.arm(data.run_ids),
        "grid": G.describe(data.nside),
        "source_nside": data.nside,
        "extent": dict(zip(("lon_min", "lon_max", "lat_min", "lat_max"), extent)),
        "panel_width_in": float(panel_width),
        "site_labels_drawn": bool(labels),
        "panel_order": [cross.short_dataset(r) for r in data.run_ids],
        "method_terms": method_term_table([data.method_a, data.method_b]),
        "panels": counts,
        "label_crowding": crowding,
        "encoding": {
            "site": (
                "one marker per (run_id, site_id) at the site's TG coordinate; "
                "~20 IP replicas share that coordinate exactly"
            ),
            "category": {
                c: f"marker {CATEGORY_MARKER[c][0]!r}, fill {ink[c]} -- {names[c]}"
                for c in CT.CATEGORIES
            },
            "label": (
                "k_a | k_b | n: correct counts for method_a and method_b over "
                "the site's n replicas. Relaxed apart in units of a label box "
                "and joined back to its marker once displaced. Absent when "
                "site_labels_drawn is false, in which case the counts are in "
                "the CSV twin only"
            ),
            "cell": f"{M.CELL_EDGE} boundary: Voronoi cell of a seed, unbounded",
        },
        "policy": {
            "correct": (
                "cell_label == 'correct' AND status.solved_mask. A FALLBACK row "
                "carries the shortest-ping baseline's coordinate, so counting it "
                "credits the method with an answer it declined to give."
            ),
            "category": (
                "Direct comparison of the two counts; 'neither' when both are "
                "zero. No threshold anywhere. A majority rule (correct_frac >= "
                "0.5) reads Octant-Hull's 9 of 20 at Los Angeles as a loss and "
                "paints the site an exclusive Spotter win, on a site where "
                "Octant-Hull is the more accurate method."
            ),
            "site_key": (
                "(run_id, site_id). site_id is assigned per run, so pooling on "
                "it alone merges the meshes -- it reported 20 sites where there "
                "are 34."
            ),
            "mcnemar": (
                "Exact two-sided sign test over the discordant sites (the ones "
                "one method wins outright); ties and double failures carry no "
                "direction and are excluded. mcnemar_p_floor is the smallest p "
                "that many discordant sites could produce -- read it beside the "
                "p-value, or an underpowered mesh looks like a null result."
            ),
            "pooled": (
                "The '__pooled__' panel is reported and is not independent: the "
                "three meshes share facilities (65 sites at 43 distinct "
                "coordinates), so its p-value is optimistic. It is absent from "
                "the CSV twin, which is a per-site table."
            ),
        },
        "known_limitation": (
            "label_crowding reports, per panel and in units of a label box, "
            "the separations the relaxation achieved. n_overlapping_label_pairs "
            "is the one to read: 0 means no two labels overlap. Where the sites "
            "are dense the label-to-marker clearance is the constraint that "
            "gives first -- as03's north-east puts six sites inside three "
            "degrees -- so a label there can come to rest closer to a marker "
            "than OBSTACLE_SEP_UNITS asks for. More relaxation sweeps do not "
            "fix it; it is unsatisfiable, not unconverged. --ncols 1 gives each "
            "panel the full figure width and is the real remedy."
        ),
        "caveat": (
            "This compares two methods on the unbounded cell metric, which "
            "credits a prediction at any distance from its target. The figure "
            "shows what that metric rewards; it is not evidence that either "
            "method is accurate. Read it beside the error-distance figures."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    method_a: str = DEFAULT_METHOD_A,
    method_b: str = DEFAULT_METHOD_B,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    ncols: int | None = None,
    panel_width: float = _PANEL_W_IN,
    labels: bool = True,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> Path:
    """PNG, CSV twin and manifest for one method pair. Returns the PNG."""
    nside = G.validate_nside(nside)
    data = load(
        runs, method_a=method_a, method_b=method_b, nside=nside, analysis_root=analysis_root
    )
    table = contest_table(data)
    counts = panel_counts(table, data)
    pair = pair_slug(method_a, method_b)
    names = {k: v.format(pair=pair) for k, v in
             zip(("png", "csv", "man"), (PNG_NAME, CSV_NAME, MANIFEST_NAME))}
    out_dir = output_dir(data.run_ids, analysis_root=analysis_root)
    build_csv(table, counts).to_csv(out_dir / names["csv"], index=False)
    png, crowding = render(
        data, table, out_dir / names["png"], extent=extent, ncols=ncols,
        panel_width=panel_width, labels=labels,
    )
    (out_dir / names["man"]).write_text(
        _manifest(
            data, counts, crowding, extent=extent, panel_width=panel_width,
            labels=labels, png_name=names["png"], csv_name=names["csv"],
        )
    )
    return png
