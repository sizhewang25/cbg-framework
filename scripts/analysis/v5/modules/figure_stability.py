"""How consistent each method is across the replicas of one site.

Two figures over the same 65 pooled sites, one observation per site per
method, written side by side so a paper can place them independently:

* `paired_ratio_of_success` -- the share of a site's targets the method put in
  the correct cell. Spotter is all-or-nothing on 61 of 65 sites; Octant-Hull
  on 39, splitting the other 26.
* `paired_std_grid_offset` -- the standard deviation of `pred_dist_to_tg_grid`
  across a site's targets, in grid steps. Spotter's median is 0.31 grids
  against Octant-Hull's 1.22, and it is perfectly consistent on 21 sites
  against 5.

## The violin is bounded, and that takes doing

A success ratio lives on [0, 1] and these pile up on both ends: 61 of
Spotter's 65 sites sit at exactly 0 or exactly 1. A Gaussian KDE over that
spills past both bounds -- density at 1.15 on a share is a drawing that cannot
be true -- and clipping the body afterwards leaves a chopped silhouette that
does not read as a violin.

So the density is estimated by **reflection**: the sample is mirrored about 0
and about 1 before the KDE, and evaluated only on [0, 1]. That is bounded by
construction, keeps its mass, and comes out the shape a violin should be.

The bandwidth is `_VIOLIN_BW` rather than Scott's rule, which is not a
cosmetic choice: Scott's smears Spotter's four split sites into a waist as
wide as Octant-Hull's twenty-six, so the default violin denies the very claim
the figure is drawn for.

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

#: The long-format columns, one row per `(run_id, site_id, method)`.
RATIO_COL = "success_ratio"
SPREAD_COL = CT.OFFSET_SD

#: The two figures, drawn and filed separately so a paper can place them
#: apart. They share a substrate and a claim, not a canvas.
RATIO = "ratio_of_success"
SPREAD = "std_grid_offset"
FIGURES: tuple[str, ...] = (RATIO, SPREAD)

PNG_NAME = "paired_{figure}.{pair}.png"
CSV_NAME = "paired_{figure}.{pair}.csv"
MANIFEST_NAME = "paired_{figure}.{pair}.manifest.json"

#: Sized for a paper column. One figure, two categories, so it is narrow; the
#: height is set by the y-label, which `test_the_axis_label_fits` measures.
_FIG_W = 3.0
_FIG_H = 3.1

#: The spread figure's whiskers. Percentiles, **not** matplotlib's default
#: 1.5x IQR, and no outliers past them: the whisker ends are p5 and p95 and
#: must not be read as the extremes. `spread_min`/`spread_max` carry those.
SPREAD_WHIS = (5.0, 95.0)

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

_VIOLIN_W = 0.72

#: KDE bandwidth for the ratio violin. Scott's rule, the default, spreads
#: each of 65 points over about a fifth of the unit interval, which fills the
#: waist with density that is not there: Spotter is split on 4 sites and the
#: default violin draws it as though it were split on twenty.
_VIOLIN_BW = 0.12

#: Where the ratio violin's density is evaluated, and how finely.
_VIOLIN_BOUNDS = (0.0, 1.0)
_VIOLIN_POINTS = 201
_BOX_W = 0.45
_EDGE_PT = 0.9
_MEDIAN_PT = 1.6
_LABEL_PT = 8.5

_RATIO_LABEL = "share of the site's targets in the correct cell"
_SPREAD_LABEL = "std. dev. of grid error distance (grid steps)"


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


def long_table(table: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    """`contest_table`'s two methods unstacked: one row per site per method.

    Long rather than wide because both panels plot one series per method and
    the CSV twin should be read the way the figure is drawn.
    """
    out = []
    for suffix, method in (("a", method_a), ("b", method_b)):
        part = table[
            ["run_id", "dataset", "site_id", "tg_lat", "tg_lon", "tg_seed_id", "category"]
        ].copy()
        part["method"] = method
        part["method_label"] = method_label(method)
        part["k"] = table[f"k_{suffix}"]
        part["n_tgs"] = table[f"n_{suffix}"]
        part["n_solved"] = table[f"n_solved_{suffix}"]
        part[SPREAD_COL] = table[f"offset_sd_{suffix}"]
        part[RATIO_COL] = part["k"] / part["n_tgs"]
        part["unanimous"] = (part["k"] == 0) | (part["k"] == part["n_tgs"])
        out.append(part)
    return pd.concat(out, ignore_index=True).reindex(columns=list(LONG_COLUMNS))


#: Every long-format column, in order. Each figure's twin is a subset.
LONG_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "site_id", "tg_lat", "tg_lon", "tg_seed_id",
    "method", "method_label", "category",
    "k", "n_tgs", "n_solved", RATIO_COL, "unanimous", SPREAD_COL,
)

_KEY_COLUMNS = (
    "run_id", "dataset", "site_id", "tg_lat", "tg_lon", "tg_seed_id",
    "method", "method_label", "category",
)


def csv_columns(figure: str = RATIO) -> list[str]:
    """One figure's twin: the site key, then only what that figure rests on.

    Not the whole long table in both twins. A twin that carries columns its
    figure never drew invites a number to be quoted from the wrong file.
    """
    validate_figure(figure)
    if figure == RATIO:
        return [*_KEY_COLUMNS, "k", "n_tgs", RATIO_COL, "unanimous"]
    return [*_KEY_COLUMNS, "n_tgs", "n_solved", SPREAD_COL]


def method_stats(long: pd.DataFrame, method: str) -> dict:
    """One method's shape on both panels, and the unanimity rate behind (a)."""
    rows = long[long["method"] == method]
    spread = rows[SPREAD_COL].dropna()
    n = int(len(rows))
    unanimous = int(rows["unanimous"].sum())
    out = {
        "method": method,
        "method_label": method_label(method),
        "n_sites": n,
        "n_unanimous": unanimous,
        "unanimity_rate": round(unanimous / n, 4) if n else None,
        "n_split": n - unanimous,
        "n_sites_without_spread": int(rows[SPREAD_COL].isna().sum()),
        "n_sites_zero_spread": int((spread == 0).sum()),
    }
    for col, key, src in ((RATIO_COL, "ratio", rows[RATIO_COL]), (SPREAD_COL, "spread", spread)):
        if not len(src):
            continue
        out[f"{key}_min"] = round(float(src.min()), 4)
        for q in (0.25, 0.5, 0.75):
            out[f"{key}_p{int(q * 100)}"] = round(float(src.quantile(q)), 4)
        out[f"{key}_max"] = round(float(src.max()), 4)
    return out


# -- drawing --------------------------------------------------------------


def _series(long: pd.DataFrame, methods_: list[str], col: str, *, dropna: bool) -> list:
    out = []
    for m in methods_:
        v = long.loc[long["method"] == m, col]
        out.append(v.dropna().to_numpy() if dropna else v.to_numpy())
    return out


def violin_stats(values: np.ndarray, *, bw: float = _VIOLIN_BW) -> dict:
    """A bounded density for `ax.violin`, estimated by reflection.

    A success ratio lives on [0, 1] and these pile up on both ends. A Gaussian
    KDE over that spills past both bounds, and clipping the drawn body
    afterwards leaves a chopped shape that does not read as a violin. So the
    sample is mirrored about each bound before the KDE and the result is
    evaluated only inside them: bounded by construction, mass preserved (hence
    the factor of three), and the right shape.

    The summary values are the sample's own, never the density's.
    """
    from scipy.stats import gaussian_kde

    lo, hi = _VIOLIN_BOUNDS
    mirrored = np.concatenate([values, 2 * lo - values, 2 * hi - values])
    coords = np.linspace(lo, hi, _VIOLIN_POINTS)
    return {
        "coords": coords,
        "vals": gaussian_kde(mirrored, bw_method=bw)(coords) * 3.0,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": [float(np.quantile(values, q)) for q in (0.25, 0.75)],
    }


def draw_ratio(ax, series: list, inks: list[str]) -> None:
    """A violin per method, bounded to the unit interval by reflection.

    A method unanimous *and* uniform on every site has no variance at all,
    which `gaussian_kde` cannot take -- and that is the success case for this
    claim, so it is drawn as a bar rather than raised.
    """
    for pos, values, ink in zip(range(1, len(series) + 1), series, inks):
        if np.ptp(values) == 0:
            ax.hlines(values[0], pos - _VIOLIN_W / 2, pos + _VIOLIN_W / 2,
                      color=ink, linewidth=3.0)
            continue
        part = ax.violin(
            [violin_stats(values)], positions=[pos], widths=_VIOLIN_W,
            showmedians=True, showextrema=False,
        )
        for body in part["bodies"]:
            body.set_facecolor(ink)
            body.set_edgecolor(_INK_2)
            body.set_linewidth(_EDGE_PT)
            body.set_alpha(1.0)
        part["cmedians"].set_color(_SURFACE)
        part["cmedians"].set_linewidth(_MEDIAN_PT)
        part["cquantiles"].set_color(_SURFACE)
        part["cquantiles"].set_linewidth(_EDGE_PT)
    ax.set_ylim(_VIOLIN_BOUNDS[0] - 0.05, _VIOLIN_BOUNDS[1] + 0.05)
    ax.set_ylabel(_RATIO_LABEL, fontsize=_LABEL_PT, color=_INK)


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


def _finish(ax, labels: list[str]) -> None:
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.tick_params(labelsize=_LABEL_PT, colors=_INK, length=3, width=0.7)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.7)


def render(
    long: pd.DataFrame,
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
        draw_ratio(ax, _series(long, methods_, RATIO_COL, dropna=False), inks)
    else:
        draw_spread(ax, _series(long, methods_, SPREAD_COL, dropna=True), inks)
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
        "The share of one site's ~20 replicas each method places in the "
        "correct cell, over the pooled sites"
    ),
    SPREAD: (
        "The spread of each method's grid error distance across the ~20 "
        "replicas of one site, in grid steps, over the pooled sites"
    ),
}


def _manifest(
    data: CT.ContestData,
    long: pd.DataFrame,
    figure: str,
    *,
    png_name: str,
    csv_name: str,
) -> str:
    methods_ = [data.method_a, data.method_b]
    stats = [method_stats(long, m) for m in methods_]
    body = {
        "figure": png_name,
        "csv": csv_name,
        "kind": figure,
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
                "mark": "violin, one per method",
                "density": (
                    f"Gaussian KDE, bandwidth {_VIOLIN_BW}, bounded to "
                    f"{list(_VIOLIN_BOUNDS)} by reflecting the sample about "
                    f"each bound. Scott's default bandwidth smears the four "
                    f"split Spotter sites into a waist as wide as Octant-Hull's "
                    f"twenty-six, which denies the claim the figure is drawn for."
                ),
                "lines": "white: median, and the quartiles",
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
            "Both figures' statistics, in both manifests: they are one claim "
            "over one substrate and a reader comparing them should not have to "
            "open two files. The CSV twin is the other way round -- it carries "
            "only the columns this figure drew, so a number cannot be quoted "
            "from a file that did not draw it."
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
                "A site is unanimous when k is 0 or n -- every target went the "
                "same way. Against the site's total, not its solved count: a "
                "method that declined half a site did not agree with itself "
                "about the other half."
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
    long = long_table(CT.contest_table(data), method_a, method_b)
    out_dir = output_dir(data.run_ids, analysis_root=analysis_root)
    written: list[Path] = []
    for figure in figures or FIGURES:
        names = artifact_names(figure, method_a, method_b)
        long.reindex(columns=csv_columns(figure)).to_csv(
            out_dir / names["csv"], index=False
        )
        png = render(long, [method_a, method_b], figure, out_dir / names["png"])
        (out_dir / names["man"]).write_text(
            _manifest(
                data, long, figure, png_name=names["png"], csv_name=names["csv"]
            )
        )
        written.append(png)
    return written
