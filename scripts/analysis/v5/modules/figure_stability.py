"""How consistent each method is across the replicas of one site.

Two figures over the same three pooled meshes, written side by side so a
paper can place them independently. They do **not** share a denominator: the
success ratio counts the 61 Voronoi cells, the spread counts the 65 sites.
See `UNIT` for why.

* `paired_ratio_of_success` -- the distribution over **cells** of the share of
  a cell's targets the method put in the correct cell. Spotter is
  all-or-nothing on 57 of 61 cells; Octant-Hull on 35, splitting the other 26.
* `paired_std_grid_offset` -- the standard deviation of `pred_dist_to_tg_grid`
  across a site's targets, in grid steps. Spotter's median is 0.31 grids
  against Octant-Hull's 1.22, and it is perfectly consistent on 21 sites
  against 5.

## The ratio figure is a CDF, after two failed violins

A success ratio lives on [0, 1] and these pile up on both ends: 61 of
Spotter's 65 sites sit at exactly 0 or exactly 1. A violin over that was
wrong twice -- Scott's bandwidth invented a waist as wide as Octant-Hull's
where Spotter has four split sites, and bounding the KDE by clipping the drawn
body left a shape that did not read as a violin at all.

A share of 65 sites concentrated on two values has no shape a smoother can be
trusted with, so it is drawn as an empirical CDF: exact, no bins, no
bandwidth, no choices. The claim is in the geometry. The jump at 0 is the
share of sites the method missed entirely, the jump at 1 the share it swept,
and the two together are its unanimity rate -- 93.8% against 60.0%. Between
them Spotter's curve is flat and Octant-Hull's climbs, and that flatness *is*
the finding rather than a stand-in for it.

## The one claim here that is about the estimator

Every other figure in this family compares two methods on the unbounded cell
metric, which the evaluation section argues is broken. This one does not:
answering ~20 byte-identical coordinates the same way is a property of the
estimator, and it would read the same under any scoring rule.
`paired_std_grid_offset` does not mention `cell_label` at all.

## Spread of what, and in what units

The standard deviation of the grid *error distance* across a site's targets.
It is the spread of the error magnitude rather than of the answers, so two
replicas five grids out in opposite directions read as perfect agreement --
a known limitation, measured and accepted rather than overlooked. See
`contest.offset_spread` for the number it costs.

Grid steps, not kilometres. A spread in kilometres invites comparison against
an error distance, and this figure is deliberately not about accuracy: a site
whose twenty replicas all sit 38 grids out scores 0, perfectly stable and
consistently ~1,900 km wrong. as01 has such a site.

## And the part that cuts the other way

Spotter is *more often* perfectly consistent and *worse* when it is not: its
per-site spread runs to 15.8 grids where Octant-Hull's stops at 6.5. The
whiskers are the 5th and 95th percentiles and no outliers are drawn, so
**that tail is not on the page**. It is in the twin and the manifest as
`spread_max`, and anyone quoting "more stable" from this figure alone would be
quoting the middle of a distribution whose tail says the opposite.

## Solved rows only, stated once

Both panels use `status.solved_mask`: a FALLBACK row carries the shortest-ping
baseline's coordinate, so its offset describes the baseline's consistency
rather than the method's. On these meshes Spotter and Octant-Hull never fall
back, so the two readings coincide -- the rule is fixed here because it will
not always, and Vanilla is the method it bites.

A site with fewer than two solved rows has no spread to report and is absent
from the spread figure rather than drawn at zero, which would read as perfect
agreement. The count of such sites is in the manifest.

Command: `plot-stability`. Needs `build-answer-space` and `classify` on every
run. Writes a PNG, CSV twin and manifest per figure into
`_cross/stability/<datasets>[@<arm>]/`.
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
KIND = "stability"

SOURCE_NSIDE = CT.SOURCE_NSIDE
DEFAULT_METHOD_A = CT.DEFAULT_METHOD_A
DEFAULT_METHOD_B = CT.DEFAULT_METHOD_B

#: The quantity each figure draws, per row of its own table.
RATIO_COL = "success_ratio"
SPREAD_COL = CT.OFFSET_SD

#: The two figures, drawn and filed separately so a paper can place them
#: apart. They share a substrate and a claim, not a canvas.
RATIO = "ratio_of_success"
SPREAD = "std_grid_offset"
FIGURES: tuple[str, ...] = (RATIO, SPREAD)

#: What each figure counts, and they differ on purpose.
#:
#: The success ratio is counted per **cell**, because a cell is the unit the
#: metric grades in: `correct` means the prediction's nearest seed is the
#: target's seed, so two sites that share a seed are one question and not
#: two. Complete linkage merges sites within one `grid_km`, which on the
#: as0* meshes makes 61 cells out of 65 sites.
#:
#: The spread is counted per **site**, because it asks whether a method
#: answers *identical coordinates* identically. Two sites in one cell are up
#: to ~51 km apart, so pooling them would score a legitimate difference of
#: input as inconsistency.
UNIT = {RATIO: "cell", SPREAD: "site"}
UNIT_KEY = {RATIO: "tg_seed_id", SPREAD: "site_id"}

PNG_NAME = "paired_{figure}.{pair}.png"
CSV_NAME = "paired_{figure}.{pair}.csv"
MANIFEST_NAME = "paired_{figure}.{pair}.manifest.json"

#: Sized for a paper column. `test_the_axis_labels_fit` measures the labels
#: against this canvas: the figure is saved at a fixed size, so a label that
#: overruns is silently clipped rather than shrinking the axes.
_FIG_W = 4.0
_FIG_H = 3.0

#: The spread figure's whiskers. Percentiles, **not** matplotlib's default
#: 1.5x IQR, and no outliers past them: the whisker ends are p5 and p95 and
#: must not be read as the extremes. `spread_min`/`spread_max` carry those.
SPREAD_WHIS = (5.0, 95.0)

#: Quantiles every manifest reports. `SPREAD_WHIS` is in here on purpose: the
#: drawn whisker ends have to be readable as numbers, or the figure states a
#: bound nothing else does.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"


#: The ratio axis. A share, so it is the unit interval and nothing else.
RATIO_BOUNDS = (0.0, 1.0)

_CDF_PT = 1.8
_BOX_W = 0.45
_EDGE_PT = 0.9
_MEDIAN_PT = 1.6
_LABEL_PT = 8.5

#: Axis labels, in the wording the paper uses. Title case, unlike the repo's
#: other figures -- these are the strings the section was written against.
#: The spread is in grid steps; "Grid Error Distance" carries that implicitly,
#: and the manifest states it outright.
_RATIO_LABEL = "Fraction of Correct Predictions per Cell"
_RATIO_Y_LABEL = "Fraction of Cells"
_SPREAD_LABEL = "Std. Dev. of Grid Error Distance"


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/stability/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def pair_slug(method_a: str, method_b: str) -> str:
    """`spo_vs_octh` -- the terms, lowercased and stripped to word characters.

    Differs from `figure_contest_map.pair_slug`'s `SPO-vs-OCT-H`, which names
    artifacts already written. Worth unifying; not worth renaming committed
    output as a side effect of adding a figure.
    """
    terms = [
        "".join(ch for ch in method_label(m) if ch.isalnum()).lower()
        for m in (method_a, method_b)
    ]
    return "_vs_".join(terms)


def validate_figure(figure: str) -> str:
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure!r}; pick from {list(FIGURES)}")
    return figure


def artifact_names(figure: str, method_a: str, method_b: str) -> dict[str, str]:
    """`{"png": ..., "csv": ..., "man": ...}` for one figure of one pair."""
    validate_figure(figure)
    pair = pair_slug(method_a, method_b)
    return {
        key: template.format(figure=figure, pair=pair)
        for key, template in zip(
            ("png", "csv", "man"), (PNG_NAME, CSV_NAME, MANIFEST_NAME)
        )
    }


# -- the numbers ----------------------------------------------------------


def ratio_rows(
    targets: pd.DataFrame, method_a: str, method_b: str
) -> pd.DataFrame:
    """One row per `(run_id, tg_seed_id, method)`: the share it placed right.

    Keyed on the **cell**, not the site. `correct` means the prediction's
    nearest seed is the target's seed, so sites sharing a seed are one
    question; counting them separately would weight a merged facility twice
    and measure something the metric does not grade.

    Built from `contest.paired_targets` rather than from the per-site
    contest, so the aggregation happens once, at the right key.
    """
    out = []
    for suffix, method in (("a", method_a), ("b", method_b)):
        by_cell = targets.groupby(["run_id", "dataset", "tg_seed_id"], sort=True)
        part = pd.DataFrame(
            {
                "k": by_cell[f"{suffix}_correct"].sum().astype(int),
                "n_tgs": by_cell.size().astype(int),
                "n_sites": by_cell["site_id"].nunique().astype(int),
            }
        ).reset_index()
        part["method"] = method
        part["method_label"] = method_label(method)
        part[RATIO_COL] = part["k"] / part["n_tgs"]
        part["unanimous"] = (part["k"] == 0) | (part["k"] == part["n_tgs"])
        out.append(part)
    return pd.concat(out, ignore_index=True).reindex(columns=csv_columns(RATIO))


def spread_rows(
    table: pd.DataFrame, method_a: str, method_b: str
) -> pd.DataFrame:
    """One row per `(run_id, site_id, method)`: how far its answers scatter.

    Keyed on the **site**: the question is whether a method answers identical
    coordinates identically, and two sites sharing a cell are not identical
    coordinates.
    """
    out = []
    for suffix, method in (("a", method_a), ("b", method_b)):
        part = table[
            ["run_id", "dataset", "site_id", "tg_lat", "tg_lon", "tg_seed_id", "category"]
        ].copy()
        part["method"] = method
        part["method_label"] = method_label(method)
        part["n_tgs"] = table[f"n_{suffix}"]
        part["n_solved"] = table[f"n_solved_{suffix}"]
        part[SPREAD_COL] = table[f"offset_sd_{suffix}"]
        out.append(part)
    return pd.concat(out, ignore_index=True).reindex(columns=csv_columns(SPREAD))


def csv_columns(figure: str = RATIO) -> list[str]:
    """One figure's twin: its own key, then only what that figure rests on.

    The two keys differ -- see `UNIT` -- and the twins say which they are by
    carrying it. A twin that held both keys, or columns its figure never
    drew, invites a number to be quoted from the wrong file.
    """
    validate_figure(figure)
    if figure == RATIO:
        return [
            "run_id", "dataset", "tg_seed_id", "method", "method_label",
            "k", "n_tgs", "n_sites", RATIO_COL, "unanimous",
        ]
    return [
        "run_id", "dataset", "site_id", "tg_lat", "tg_lon", "tg_seed_id",
        "method", "method_label", "category", "n_tgs", "n_solved", SPREAD_COL,
    ]


def method_stats(rows: pd.DataFrame, figure: str, method: str) -> dict:
    """One method's shape on one figure, in that figure's own unit.

    `n_units` rather than `n_sites`, because the two figures do not count the
    same thing: the ratio is per cell and the spread per site. The quantiles
    include `SPREAD_WHIS`, so the drawn whisker ends are readable as numbers
    -- with `spread_min` and `spread_max` beside them, since the whiskers are
    not the range.
    """
    validate_figure(figure)
    mine = rows[rows["method"] == method]
    n = int(len(mine))
    out = {
        "method": method,
        "method_label": method_label(method),
        "unit": UNIT[figure],
        "n_units": n,
    }
    if figure == RATIO:
        unanimous = int(mine["unanimous"].sum())
        out["n_unanimous"] = unanimous
        out["unanimity_rate"] = round(unanimous / n, 4) if n else None
        out["n_split"] = n - unanimous
        out["n_cells_over_several_sites"] = int((mine["n_sites"] > 1).sum())
        src, key = mine[RATIO_COL], "ratio"
    else:
        spread = mine[SPREAD_COL].dropna()
        out["n_units_without_spread"] = int(mine[SPREAD_COL].isna().sum())
        out["n_units_zero_spread"] = int((spread == 0).sum())
        src, key = spread, "spread"
    if len(src):
        out[f"{key}_min"] = round(float(src.min()), 4)
        for q in QUANTILES:
            out[f"{key}_p{int(q * 100)}"] = round(float(src.quantile(q)), 4)
        out[f"{key}_max"] = round(float(src.max()), 4)
    return out


# -- drawing --------------------------------------------------------------


def _series(rows: pd.DataFrame, methods_: list[str], col: str, *, dropna: bool) -> list:
    out = []
    for m in methods_:
        v = rows.loc[rows["method"] == m, col]
        out.append(v.dropna().to_numpy() if dropna else v.to_numpy())
    return out


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The empirical CDF as a staircase: `(x, share of sites at or below x)`.

    Exact. No bins and no bandwidth -- which is the point: this figure was
    drawn twice as a violin and misled both times, once because Scott's
    bandwidth invented density in the middle and once because clipping a KDE
    to the unit interval left a shape that did not read as a violin. A share
    of 65 sites piled on two values has no shape a smoother can be trusted
    with; a staircase has no choices in it.
    """
    lo, hi = RATIO_BOUNDS
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return (
        np.concatenate([[lo], x, [hi]]),
        np.concatenate([[0.0], y, [1.0]]),
    )


def draw_ratio(ax, series: list, inks: list[str], labels: list[str]) -> None:
    """One step curve per method: the share of sites at or below each ratio.

    Everything the claim needs is in the two endpoints and the middle. The
    jump at 0 is the share of sites a method missed entirely, the jump at 1
    the share it swept, and the two together are its unanimity rate -- so
    Spotter's curve is flat across the middle where Octant-Hull's climbs, and
    that flatness *is* the finding rather than a stand-in for it.
    """
    for values, ink, label in zip(series, inks, labels):
        x, y = ecdf(values)
        ax.step(x, y, where="post", color=ink, linewidth=_CDF_PT, label=label)
    ax.set_xlim(RATIO_BOUNDS[0] - 0.02, RATIO_BOUNDS[1] + 0.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(_RATIO_LABEL, fontsize=_LABEL_PT, color=_INK)
    ax.set_ylabel(_RATIO_Y_LABEL, fontsize=_LABEL_PT, color=_INK)
    ax.legend(fontsize=_LABEL_PT, frameon=False, loc="upper left")


def draw_spread(ax, series: list, inks: list[str]) -> None:
    """A box per method; whiskers at `SPREAD_WHIS`, no outliers.

    The whisker ends are the 5th and 95th percentiles, not the extremes and
    not matplotlib's 1.5x IQR. Nothing is drawn past them, so Spotter's worst
    site -- 15.8 grids against Octant-Hull's 6.5 -- is off the page. That is
    the half of this claim that runs the other way, and it survives only in
    `spread_max` in the twin and the manifest.
    """
    bp = ax.boxplot(
        series, widths=_BOX_W, patch_artist=True, showfliers=False, whis=SPREAD_WHIS,
        medianprops={"color": _SURFACE, "linewidth": _MEDIAN_PT},
        whiskerprops={"color": _INK_2, "linewidth": _EDGE_PT},
        capprops={"color": _INK_2, "linewidth": _EDGE_PT},
    )
    for box, ink in zip(bp["boxes"], inks):
        box.set_facecolor(ink)
        box.set_edgecolor(_INK_2)
        box.set_linewidth(_EDGE_PT)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(_SPREAD_LABEL, fontsize=_LABEL_PT, color=_INK)


def _finish(ax, labels: list[str] | None) -> None:
    if labels is not None:
        ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.tick_params(labelsize=_LABEL_PT, colors=_INK, length=3, width=0.7)
    ax.grid(True, color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.7)


def render(
    rows: pd.DataFrame,
    methods_: list[str],
    figure: str,
    out_png: Path,
    *,
    fig_size: tuple[float, float] = (_FIG_W, _FIG_H),
    dpi: int = 150,
) -> Path:
    """One figure, one panel, no title -- the paper's caption names it."""
    validate_figure(figure)
    inks = [method_colors(methods_)[m] for m in methods_]
    labels = [method_label(m) for m in methods_]

    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor(_SURFACE)
    if figure == RATIO:
        draw_ratio(ax, _series(rows, methods_, RATIO_COL, dropna=False), inks, labels)
        _finish(ax, None)
    else:
        draw_spread(ax, _series(rows, methods_, SPREAD_COL, dropna=True), inks)
        _finish(ax, labels)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=_SURFACE)
    plt.close(fig)
    return out_png


#: What each figure is of, for its manifest. The paper's caption says this;
#: the figure does not.
SUBJECTS = {
    RATIO: (
        "The share of one cell's targets each method places in the correct "
        "cell, over the pooled cells"
    ),
    SPREAD: (
        "The spread of each method's grid error distance across the ~20 "
        "replicas of one site, in grid steps, over the pooled sites"
    ),
}


def _manifest(
    data: CT.ContestData,
    rows: pd.DataFrame,
    figure: str,
    *,
    png_name: str,
    csv_name: str,
) -> str:
    methods_ = [data.method_a, data.method_b]
    stats = [method_stats(rows, figure, m) for m in methods_]
    body = {
        "figure": png_name,
        "csv": csv_name,
        "kind": figure,
        "unit": UNIT[figure],
        "unit_note": (
            "The success ratio counts cells and the spread counts sites, and "
            "they differ on purpose. `correct` means the prediction's nearest "
            "seed is the target's seed, so sites sharing a seed are one "
            "question the metric grades once. The spread asks whether a "
            "method answers identical coordinates identically, and two sites "
            "in one cell are up to one grid_km apart -- pooling them would "
            "score a difference of input as inconsistency."
        ),
        "companion": [n for n in FIGURES if n != figure],
        "subject": SUBJECTS[figure],
        "subject_note": (
            "the figure draws no title of its own -- the paper's caption names it"
        ),
        "method_a": data.method_a,
        "method_b": data.method_b,
        "datasets": cross.dataset_slug(data.run_ids),
        "run_ids": data.run_ids,
        "arm": cross.arm(data.run_ids),
        "grid": G.describe(data.nside),
        "source_nside": data.nside,
        "method_terms": {
            m: {"term": method_label(m), "name": methods.METHOD_TERMS.get(method_label(m))}
            for m in methods_
        },
        "encoding": (
            {
                "quantity": RATIO_COL,
                "axis": _RATIO_LABEL,
                "mark": "step curve, one per method, in that method's hue",
                "axes": (
                    "x: the per-site success ratio. y: the share of sites at "
                    "or below it -- an empirical CDF, exact, with no bins and "
                    "no bandwidth."
                ),
                "reading": (
                    "The jump at 0 is the share of sites the method missed "
                    "entirely and the jump at 1 the share it swept; together "
                    "they are its unanimity rate. The curve is flat between "
                    "them exactly to the extent the method is all-or-nothing."
                ),
            }
            if figure == RATIO
            else {
                "quantity": SPREAD_COL,
                "axis": _SPREAD_LABEL,
                "mark": "boxplot, one per method",
                "definition": (
                    f"Sample standard deviation (ddof={CT.OFFSET_SD_DDOF}) of "
                    f"pred_dist_to_tg_grid across a site's solved targets, in "
                    f"grid steps rather than kilometres so it is not read as "
                    f"an accuracy. A site whose replicas all sit 38 grids out "
                    f"scores 0: perfectly stable and consistently wrong."
                ),
                "known_limitation": (
                    "The spread of the error magnitude, not of the answers: "
                    "two replicas five grids out in opposite directions read "
                    "as perfect agreement. Measured against the spread of the "
                    "prediction cloud itself, the two rank the 65 sites at "
                    "Spearman 0.90 and agree on the conclusion, and differ on "
                    "11 sites that read as perfectly consistent here while "
                    "their predictions were up to two grids apart. The simpler "
                    "statistic was chosen on those terms."
                ),
                "whiskers": (
                    f"percentiles {SPREAD_WHIS[0]:.0f} and {SPREAD_WHIS[1]:.0f}, "
                    f"not 1.5x IQR and not the extremes. Nothing is drawn beyond "
                    f"them: no outliers. Read spread_min and spread_max for the "
                    f"range -- Spotter's worst site is off the page."
                ),
            }
        ),
        "stats": stats,
        "stats_note": (
            "This figure's statistics only, in its own unit -- see `unit`. "
            "The two figures no longer share a denominator, so reporting both "
            "in one manifest would put 61 cells beside 65 sites with nothing "
            "saying which was which."
        ),
        "policy": {
            "solved_mask": (
                "Both panels use status.solved_mask. A FALLBACK row carries the "
                "shortest-ping baseline's coordinate, so its offset describes "
                "the baseline's consistency, not the method's. SPO and OCT-H "
                "never fall back on these meshes, so the two readings coincide; "
                "the rule is fixed because it will not always."
            ),
            "unanimity": (
                "A cell is unanimous when k is 0 or n -- every target in it "
                "went the same way. Against the cell's total, not its solved "
                "count: a method that declined half a cell did not agree with "
                "itself about the other half."
            ),
            "spread": (
                f"contest.offset_spread over a site's solved targets: the "
                f"sample standard deviation (ddof={CT.OFFSET_SD_DDOF}) of "
                f"pred_dist_to_tg_grid. A site with fewer than two solved rows "
                f"has no spread to report and is absent rather than drawn at "
                f"zero, which would read as perfect agreement; "
                f"n_sites_without_spread counts them."
            ),
        },
        "caveat": (
            "This is the one claim in the family that is about the estimator "
            "rather than the metric -- answering identical coordinates "
            "identically would read the same under any scoring rule, and "
            "paired_std_grid_offset does not use cell_label at all. It cuts "
            "both ways: the method perfectly consistent on more sites is also "
            "the worse of the two on its worst, 15.8 grids against 6.5, and "
            "with whiskers at p5/p95 and no outliers that tail is not drawn."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    method_a: str = DEFAULT_METHOD_A,
    method_b: str = DEFAULT_METHOD_B,
    figures: list[str] | None = None,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> list[Path]:
    """PNG, CSV twin and manifest per figure. Returns the PNGs."""
    for figure in figures or FIGURES:
        validate_figure(figure)
    nside = G.validate_nside(nside)
    data = CT.load(
        runs, method_a=method_a, method_b=method_b, nside=nside,
        analysis_root=analysis_root,
    )
    wanted = list(figures or FIGURES)
    rows = {}
    if RATIO in wanted:
        rows[RATIO] = ratio_rows(CT.paired_targets(data), method_a, method_b)
    if SPREAD in wanted:
        rows[SPREAD] = spread_rows(CT.contest_table(data), method_a, method_b)
    out_dir = output_dir(data.run_ids, analysis_root=analysis_root)
    written: list[Path] = []
    for figure in wanted:
        names = artifact_names(figure, method_a, method_b)
        rows[figure].to_csv(out_dir / names["csv"], index=False)
        png = render(rows[figure], [method_a, method_b], figure, out_dir / names["png"])
        (out_dir / names["man"]).write_text(
            _manifest(
                data, rows[figure], figure, png_name=names["png"], csv_name=names["csv"]
            )
        )
        written.append(png)
    return written
