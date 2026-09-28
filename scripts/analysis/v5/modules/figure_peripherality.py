"""How far out the sites each method wins sit: two boxes, one axis.

The contest map shows *where* the two methods win. This reduces that to one
number per site -- its distance from the run's seed-cloud centroid -- and
contrasts the two win categories on a single axis.

The finding it backs: Spotter's wins are exclusively peripheral. Not mostly,
exclusively -- its nearest win on these meshes is 1,018 km out, and its lower
quartile sits above Octant-Hull's median. Octant-Hull wins from 98 km out and
its box spans nearly the whole range. Two boxes make that a shape rather than
a pair of quoted numbers.

## What is measured, from what, to what

Great-circle, from a site's **seed** to the spherical centroid of **all** that
run's seeds. Both ends are points of the answer space, which is what makes the
number coherent: it says where a serving region sits, not where a target
happens to sit inside one. Sites grouped under one seed share a value -- as01
has 18 seeds over 20 sites.

The origin is the seeds' and not the site set's: a mesh whose sites happen to
cluster would otherwise move the thing it is being measured against. It is
found by `geodesy.spherical_centroid`, which is how `seeds` places each seed
over the sites it groups -- one notion of centre in the package.

Not in EPSG:5070. The partition is defined by great-circle nearest seed and
`classify` uses no projection; the plane exists only so `cells` can draw the
partition, and peripherality is a property of the partition rather than of
the drawing. See `contest.seed_cloud_centroid_km` for what the difference measures out
at -- small enough that no claim here turns on it, which is the point.

## The axis is normalised, and the kilometres are in the twin

Min-max over **all 65 pooled sites**, including the tied and neither ones that
are not drawn -- so the axis is "position within the observed range of site
peripherality" and 0 and 1 are real sites rather than round numbers. A reader
cannot quote a normalised axis, so the CSV twin carries every site's raw
kilometres alongside, and the manifest carries the two constants and the
per-category quantiles in both units.

## Two boxes, not four

`tied` and `neither` are in the twin and off the figure. They are concordant
outcomes and carry no direction, and drawing four boxes to compare two of them
puts the contrast this figure exists for on a third of the page.

## The seed, or the cell it owns

A seed's distance is not the same as its cell's. A peripheral seed can own a
large cell reaching back in toward the centre: as01's Seattle seed is 2,349 km
out and its cell begins at 1,241. Measured, the two rank the 65 sites only at
Spearman 0.87 -- a real difference, not a rounding one.

It is not what produces the contrast. The inward reach is the same size in
both categories (median 404 km where Spotter wins, 333 where Octant-Hull
does), and the claim holds under either measure: by cell reach, Spotter never
wins a site whose region begins within 770 km of the centre, while
Octant-Hull wins one whose region *contains* it. The seed distance is the
simpler quantity and the one the contest map keys on, so it is the one drawn.

## What this does not separate

On a CONUS answer space, distance from the centroid is nearly collinear with
"coastal". This figure cannot tell the two apart and must not be read as
having done so. And 21 sites against 20 is the whole sample: the three meshes
share facilities, so even that is not 41 independent observations.

Command: `plot-peripherality`. Needs `build-answer-space` and `classify` on
every run. Writes `_cross/peripherality/<datasets>[@<arm>]/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import contest as CT  # noqa: E402
from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules import methods  # noqa: E402
from scripts.analysis.v5.modules.methods import method_colors, method_label  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

#: `_cross/<KIND>/<datasets>[@<arm>]/`.
KIND = "peripherality"

SOURCE_NSIDE = CT.SOURCE_NSIDE
DEFAULT_METHOD_A = CT.DEFAULT_METHOD_A
DEFAULT_METHOD_B = CT.DEFAULT_METHOD_B

#: The two decisive categories, drawn top to bottom in this order. `tied` and
#: `neither` are concordant and are in the twin only -- see the docstring.
DRAWN: tuple[str, str] = (CT.A_WINS, CT.B_WINS)

#: The column the axis is built from, and its normalised twin.
DISTANCE_COL = "centroid_km"
NORM_COL = "centroid_norm"

#: Quantiles the manifest reports, in both units. The minimum is the headline
#: -- "no Spotter win is closer in than this" -- so it is reported beside them
#: rather than left to the whisker.
QUANTILES = (0.25, 0.5, 0.75)

PNG_NAME = "peripherality.{pair}.png"
CSV_NAME = "peripherality.{pair}.csv"
MANIFEST_NAME = "peripherality.{pair}.manifest.json"

#: Sized for a paper column, following `figure_outcome_bars`: two boxes need
#: no more height than this, and a taller figure is white space above and
#: below them.
#:
#: The width is set by the x-label, not by the boxes. `tight_layout` fits the
#: label's *height* and lets a long one run off both sides, and the figure is
#: saved at a fixed canvas, so it is silently clipped -- 4.6 in lost the last
#: character. `test_the_axis_label_fits` measures it rather than trusting this
#: number, so shortening or lengthening the label fails loudly.
_FIG_W = 5.3
_FIG_H = 1.75

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

_BOX_H = 0.55
_BOX_EDGE_PT = 0.9
_MEDIAN_PT = 1.6
_WHISKER_PT = 0.9

#: Outliers as small open circles in the method's own hue: they are sites, not
#: errors, and on a sample of twenty they are worth seeing individually.
_FLIER_SIZE = 3.0

#: The figure's only text. The repo's axis labels are sentence case.
_XLABEL = "normalized distance from the TG's seed to the centroid of all seeds"


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/peripherality/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def pair_slug(method_a: str, method_b: str) -> str:
    return f"{method_label(method_a)}-vs-{method_label(method_b)}".replace(" ", "_")


def category_names(method_a: str, method_b: str) -> dict[str, str]:
    return {
        CT.A_WINS: f"{method_label(method_a)} wins",
        CT.B_WINS: f"{method_label(method_b)} wins",
    }


def category_ink(method_a: str, method_b: str) -> dict[str, str]:
    """Each box takes its own method's hue, as on the contest map."""
    hues = method_colors([method_a, method_b])
    return {CT.A_WINS: hues[method_a], CT.B_WINS: hues[method_b]}


# -- the numbers ----------------------------------------------------------


def normalise(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add `centroid_norm`, min-max over **every** pooled site.

    Over all four categories, not just the drawn two: the axis then means
    "position within the range of site peripherality this answer space
    actually has", and both endpoints are real sites. Normalising over the
    drawn subset instead would move the axis whenever a category changed.
    """
    d = table[DISTANCE_COL]
    lo, hi = float(d.min()), float(d.max())
    span = hi - lo
    if span <= 0:
        raise ValueError(
            f"every site is {lo:.0f} km from the centroid; there is nothing to "
            f"normalise and nothing to draw"
        )
    out = table.copy()
    out[NORM_COL] = (d - lo) / span
    return out, {
        "min_km": round(lo, 1),
        "max_km": round(hi, 1),
        "over_n_sites": int(len(table)),
        "over_categories": list(CT.CATEGORIES),
    }


def category_stats(table: pd.DataFrame, category: str) -> dict:
    """One category's shape, in kilometres and normalised both.

    `min` is reported explicitly rather than left to a whisker: "no Spotter win
    is closer in than 1,020 km" is the claim, and a whisker is a drawing.

    """
    rows = table[table["category"] == category]
    out = {"category": category, "n_sites": int(len(rows))}
    if not len(rows):
        return out
    for col, unit in ((DISTANCE_COL, "km"), (NORM_COL, "norm")):
        out[f"min_{unit}"] = round(float(rows[col].min()), 4)
        for q in QUANTILES:
            out[f"p{int(q * 100)}_{unit}"] = round(float(rows[col].quantile(q)), 4)
        out[f"max_{unit}"] = round(float(rows[col].max()), 4)
    return out


def csv_columns() -> list[str]:
    """The twin's columns: the normalisation, then every site behind it."""
    return [
        "run_id", "dataset", "method_a", "method_b", "label_a", "label_b",
        "norm_min_km", "norm_max_km",
        "site_id", "tg_lat", "tg_lon", "tg_seed_id",
        "k_a", "k_b", "n_tgs", "category", "drawn",
        DISTANCE_COL, NORM_COL,
    ]


def build_csv(table: pd.DataFrame, norm: dict) -> pd.DataFrame:
    """Every pooled site, drawn or not.

    All 65 rather than the 41 the figure draws: the normalisation spans all of
    them, so a twin holding only the drawn ones could not reproduce its own
    axis.
    """
    out = table.copy()
    out["norm_min_km"] = norm["min_km"]
    out["norm_max_km"] = norm["max_km"]
    out["drawn"] = out["category"].isin(DRAWN)
    return out.reindex(columns=csv_columns())


# -- drawing --------------------------------------------------------------


def render(
    data: CT.ContestData,
    table: pd.DataFrame,
    out_png: Path,
    *,
    fig_size: tuple[float, float] = (_FIG_W, _FIG_H),
    dpi: int = 150,
) -> Path:
    """Two horizontal boxes on one normalised axis. No title -- the paper's
    caption names the figure, as `figure_outcome_bars` settled."""
    ink = category_ink(data.method_a, data.method_b)
    names = category_names(data.method_a, data.method_b)
    series = [table.loc[table["category"] == c, NORM_COL].to_numpy() for c in DRAWN]

    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    # Position 1 is the bottom in matplotlib, and `DRAWN` reads top-down, so
    # the axis is inverted rather than the data reordered.
    bp = ax.boxplot(
        series, vert=False, widths=_BOX_H, patch_artist=True, showfliers=True,
        medianprops={"color": _SURFACE, "linewidth": _MEDIAN_PT},
        whiskerprops={"color": _INK_2, "linewidth": _WHISKER_PT},
        capprops={"color": _INK_2, "linewidth": _WHISKER_PT},
    )
    for box, flier, category in zip(bp["boxes"], bp["fliers"], DRAWN):
        box.set_facecolor(ink[category])
        box.set_edgecolor(_INK_2)
        box.set_linewidth(_BOX_EDGE_PT)
        flier.set(
            marker="o", markersize=_FLIER_SIZE, markerfacecolor="none",
            markeredgecolor=ink[category], markeredgewidth=0.8, linestyle="none",
        )

    ax.set_yticks(range(1, len(DRAWN) + 1), [names[c] for c in DRAWN])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.set_xlabel(_XLABEL, fontsize=8.5, color=_INK)
    ax.tick_params(labelsize=8.5, colors=_INK, length=3, width=0.7)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=_SURFACE)
    plt.close(fig)
    return out_png


def _manifest(
    data: CT.ContestData,
    table: pd.DataFrame,
    norm: dict,
    *,
    png_name: str,
    csv_name: str,
) -> str:
    a, b = method_label(data.method_a), method_label(data.method_b)
    ink = category_ink(data.method_a, data.method_b)
    body = {
        "figure": png_name,
        "csv": csv_name,
        "subject": (
            f"How far from its answer space's centre each site sits, for the "
            f"sites {a} wins and the sites {b} wins"
        ),
        "subject_note": (
            "the figure draws no title of its own -- the paper's caption names it"
        ),
        "method_a": data.method_a,
        "method_b": data.method_b,
        "datasets": cross.dataset_slug(data.run_ids),
        "run_ids": data.run_ids,
        "grid": G.describe(data.nside),
        "source_nside": data.nside,
        "method_terms": {
            m: {"term": method_label(m), "name": methods.METHOD_TERMS.get(method_label(m))}
            for m in (data.method_a, data.method_b)
        },
        "normalisation": norm,
        "seed_cloud_centre": {
            rid: dict(zip(("lat", "lon"), CT.seed_cloud_centre(
                data.seeds[rid]["seed_lat"], data.seeds[rid]["seed_lon"]
            )))
            for rid in data.run_ids
        },
        "drawn": list(DRAWN),
        "stats": [category_stats(table, c) for c in CT.CATEGORIES],
        "encoding": {
            "box": {c: ink[c] for c in DRAWN},
            "axis": (
                "distance to the run's seed-cloud centroid, min-max normalised "
                "over every pooled site; linear. The raw kilometres are in the "
                "CSV twin, per site."
            ),
            "flier": "open circle in the category's hue, one per outlying site",
        },
        "policy": {
            "origin": (
                "From a site's seed to geodesy.spherical_centroid over the "
                "run's *seed* positions -- both ends are points of the answer "
                "space, and sites sharing a seed share a value. The origin is "
                "the seeds' and not the sites': a mesh whose sites cluster "
                "would otherwise move it. Spherical, not EPSG:5070 -- "
                "the cell partition is great-circle nearest seed and the plane "
                "exists only to draw it. The planar answer differs by a median "
                "5 km here and ranks the sites identically."
            ),
            "normalisation": (
                "Over all four categories' sites, so the endpoints are real "
                "sites and the axis does not move when a category does. "
                "Normalising over the drawn two alone would rescale the figure "
                "every time a site changed hands."
            ),
            "seed_or_cell": (
                "The axis is the seed's distance, not its cell's. A peripheral "
                "seed can own a large cell reaching back toward the centre -- "
                "as01's Seattle seed is 2,349 km out and its cell begins at "
                "1,241 -- and the two rank the sites only at Spearman 0.87. It "
                "does not produce the contrast: the inward reach is the same "
                "size in both categories (median 404 km where A wins, 333 where "
                "B does), and under cell reach A still never wins inside 770 km "
                "while B wins a site whose cell contains the centre."
            ),
            "twin": (
                "Every pooled site is in the CSV, `drawn` false for the tied "
                "and neither ones. A twin holding only the drawn rows could not "
                "reproduce its own axis."
            ),
        },
        "caveat": (
            f"Distance from the centroid is nearly collinear with 'coastal' on "
            f"a CONUS answer space; this figure does not separate the two. The "
            f"sample is {len(table[table['category'] == CT.A_WINS])} sites "
            f"against {len(table[table['category'] == CT.B_WINS])}, and the "
            f"three meshes share facilities, so they are not independent."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    method_a: str = DEFAULT_METHOD_A,
    method_b: str = DEFAULT_METHOD_B,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> Path:
    """PNG, CSV twin and manifest for one method pair. Returns the PNG."""
    nside = G.validate_nside(nside)
    data = CT.load(
        runs, method_a=method_a, method_b=method_b, nside=nside,
        analysis_root=analysis_root,
    )
    table, norm = normalise(CT.contest_table(data))
    pair = pair_slug(method_a, method_b)
    names = {k: v.format(pair=pair) for k, v in
             zip(("png", "csv", "man"), (PNG_NAME, CSV_NAME, MANIFEST_NAME))}
    out_dir = output_dir(data.run_ids, analysis_root=analysis_root)
    build_csv(table, norm).to_csv(out_dir / names["csv"], index=False)
    png = render(data, table, out_dir / names["png"])
    (out_dir / names["man"]).write_text(
        _manifest(data, table, norm, png_name=names["png"], csv_name=names["csv"])
    )
    return png
