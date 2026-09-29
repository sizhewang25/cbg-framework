"""Who is nearest on each TG: the error CDF, paired. An UpSet plot.

The error CDF is **unpaired**. Each curve is one method's marginal
distribution, so it cannot say whether the TGs where OCT-H is close are the
same TGs where SOI is close, or which method was closest on any one of them.
Two methods with identical curves can win on disjoint halves of the roster.
This figure is the pairing: on every TG, which methods got the lowest error.

## A champion is within 1 km of the best

On each TG, `best` is the smallest `pred_dist_to_tg_km` among the methods that
answered, and a method is a **champion** when its own error is at most
`best + tie_km` (default 1 km). Every champion counts in full, so a tie credits
each tied method -- no fractional shares, no tie-break.

The tolerance is not cosmetic. S-P predicts the shortest-ping VP's coordinate
and SOI very often returns that same VP, so the two errors differ by float
noise (~0.006 km). A zero tolerance would split those TGs between them at
random; 1 km names them for what they are, a tie. It is the same margin as
`cohort_overlap.DEFAULT_MARGIN_KM`, and the artifacts carry it in their names
(`tie-1km`), so a second tolerance cannot overwrite the first.

## Unanswered rows never win

`status.solved_mask` again, via `figure_error_cdf.error_matrix`. A FALLBACK row
carries the shortest-ping VP's coordinate, so letting it compete would credit
VAN with S-P's wins on exactly the TGs VAN refused. S-P's all-BASELINE rows are
answered. S-P answers every TG, so on the published roster every TG has at
least one champion and the `(none)` combination is empty; it is still counted,
not assumed, for a `--method` selection without S-P.

## Two bar charts, two different quantities

* **Intersections** (top): one bar per *exact* champion combination -- "OCT-H
  and nobody else", "SOI and S-P and nobody else". These partition the TGs and
  sum to 100%. Single-method columns take the method's hue; tie columns are
  neutral grey, so a tie reads as its own outcome rather than as a win for
  whichever colour it happens to borrow. S-P is the error CDF's baseline ink
  (`_INK_2`, not its `LABEL_HUES` blue), hatched where the CDF dashes, so the
  pair of figures names the reference the same way.
* **Set sizes** (left): each method's champion share, ties included. These
  overlap and do **not** sum to 100% -- SOI's set is mostly inside S-P's.

The axis titles differ on purpose. A method with no champion TG keeps its row
and prints 0.0 -- the reason this figure is drawn here rather than by
`upsetplot`, which drops an empty category silently (SPO on pro-as01).

Columns are sorted largest first, so position is data-dependent: compare
columns across figures by their dots, never by x.

## Pooling

Every run's error matrix is stacked (`<run_id>::<tg_id>`), then judged. The
contest is row-wise, so judging after stacking equals judging per run and
concatenating. A micro-average: a dataset weighs by its TG count. Coverage is
strict -- the error CDF's guards and remedies.

## Counts are TGs; sites are the effective n

~20 IP replicas share a site and its VP geometry, so twenty TGs at one site
are closer to one observation than to twenty. Both tables carry `n_sites`
beside every TG count (distinct `sites.site_key`); a count whose site twin is
much smaller is one observation wearing a large number.

Command: `plot-champion-upset`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import figure_error_cdf as E  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules.methods import (  # noqa: E402
    method_colors,
    method_label,
    method_order,
    method_term_table,
)
from scripts.analysis.v5.modules.paths import CLASSIFY_KIND, RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import SHORTEST_PING  # noqa: E402

#: Within this many km of the best error is a tie, not a loss.
DEFAULT_TIE_KM = 1.0

PER_RUN = E.PER_RUN
POOLED = E.POOLED
LAYOUTS = E.LAYOUTS
SOURCE_NSIDE = E.SOURCE_NSIDE

#: Past eight methods there are up to 255 columns, which no reader traces.
MAX_METHODS = 8

#: What a TG with no champion is called in the tables.
NONE_LABEL = "(none)"

#: Joins the terms of a combination: `SOI+S-P`.
MEMBER_SEP = "+"

#: Separator in the pooled index. `::` occurs in neither a run id nor a TG id.
RUN_KEY_SEP = "::"

STEM = "champion_upset"
KINDS: tuple[str, ...] = ("png", "intersections", "sets", "membership", "manifest")
_SUFFIX = {
    "png": "png",
    "intersections": "intersections.csv",
    "sets": "sets.csv",
    "membership": "membership.csv",
    "manifest": "manifest.json",
}

#: The two bar charts' axis names. Different words, because one partitions the
#: TGs and the other does not.
INTERSECTION_LABEL = "Exact combination (% of TGs)"
SET_LABEL = "Champion share (%)"

#: Canvas (inches). Wider than the CDF's paper column: a dozen columns of dots
#: will not fit four inches.
FIGSIZE: tuple[float, float] = (6.5, 3.8)

#: Tie columns and absent dots. Neutral ink, never a method hue.
TIE_INK = E._MUTED
ABSENT_DOT = "#dcdbd4"
ROW_SHADE = "#f4f3ef"

#: The baseline's bars are hatched: the CDF marks S-P as the reference with a
#: dash, and a bar has no dash to give.
BASELINE_HATCH = "////"

#: S-P's ink, taken from the error CDF rather than from `methods.LABEL_HUES`,
#: so the baseline is the same dark grey in both figures of the pair. Read off
#: `_curve_style` instead of copied, so the two cannot drift.
BASELINE_INK = E._curve_style(SHORTEST_PING, {})["color"]


def upset_colors(methods) -> dict[str, str]:
    """`methods.method_colors`, with the baseline in the error CDF's ink."""
    colors = method_colors(methods)
    if SHORTEST_PING in colors:
        colors[SHORTEST_PING] = BASELINE_INK
    return colors


def _method_style(method: str, colors: dict[str, str]) -> dict:
    """One method's bar: its hue, or the hatched baseline."""
    if method == SHORTEST_PING:
        return {"color": "white", "edgecolor": colors[method], "hatch": BASELINE_HATCH,
                "linewidth": 0.8}
    return {"color": colors[method], "edgecolor": colors[method], "linewidth": 0.0}

INTERSECTION_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "tie_km", "members", "n_methods", "n_tgs", "share", "n_sites",
)
SET_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "tie_km", "method", "method_label", "is_baseline",
    "n_tgs", "n_champion", "share", "n_sole", "n_sites", "n_sites_champion",
)


def tie_slug(tie_km: float) -> str:
    """`1.0 -> "tie-1km"`, `0.5 -> "tie-0.5km"`."""
    return f"tie-{float(tie_km):g}km"


def artifact_names(layout: str, tie_km: float = DEFAULT_TIE_KM) -> dict[str, str]:
    """`kind -> filename` for one layout at one tolerance."""
    parts = [STEM, tie_slug(tie_km)]
    if layout == POOLED:
        parts.insert(1, "pooled")
    stem = ".".join(parts)
    return {kind: f"{stem}.{_SUFFIX[kind]}" for kind in KINDS}


def validate_tie(tie_km: float) -> float:
    tie = float(tie_km)
    if not np.isfinite(tie) or tie < 0:
        raise ValueError(f"tie_km must be a finite distance >= 0; got {tie_km}")
    return tie


# ---- the contest ------------------------------------------------------------


def champion_mask(errors: pd.DataFrame, tie_km: float = DEFAULT_TIE_KM) -> pd.DataFrame:
    """Per TG and method: is this error within `tie_km` of the TG's best?

    NaN -- unanswered -- is never a champion, and a TG no method answered is
    all-False, so it lands in the `(none)` combination rather than vanishing.
    """
    tie = validate_tie(tie_km)
    best = errors.min(axis=1, skipna=True)
    return errors.le(best + tie, axis=0).astype(bool)


def combination(mask: pd.DataFrame) -> pd.Series:
    """Per TG, its champions' terms joined in column order: `SOI+S-P`."""
    labels = [method_label(m) for m in mask.columns]
    values = mask.to_numpy(dtype=bool)
    return pd.Series(
        [MEMBER_SEP.join(l for l, hit in zip(labels, row) if hit) or NONE_LABEL for row in values],
        index=mask.index,
        name="members",
    )


def intersection_table(mask: pd.DataFrame, sites: pd.Series) -> pd.DataFrame:
    """One row per non-empty exact combination, largest first. Sums to `n`."""
    combo = combination(mask)
    degree = mask.sum(axis=1)
    frame = pd.DataFrame({"members": combo, "n_methods": degree, "site": sites.reindex(mask.index)})
    grouped = frame.groupby("members", sort=False).agg(
        n_methods=("n_methods", "first"),
        n_tgs=("n_methods", "size"),
        n_sites=("site", "nunique"),
    )
    table = grouped.reset_index()
    table["share"] = (table["n_tgs"] / len(mask)).round(4) if len(mask) else 0.0
    return (
        table.sort_values(["n_tgs", "n_methods", "members"], ascending=[False, True, True])
        .reset_index(drop=True)[["members", "n_methods", "n_tgs", "share", "n_sites"]]
    )


def set_table(mask: pd.DataFrame, sites: pd.Series) -> pd.DataFrame:
    """One row per method in column order: its champion count, ties included."""
    site = sites.reindex(mask.index)
    sole = mask.sum(axis=1) == 1
    n = len(mask)
    rows = []
    for method in mask.columns:
        hit = mask[method]
        rows.append(
            {
                "method": method,
                "method_label": method_label(method),
                "is_baseline": method == SHORTEST_PING,
                "n_tgs": int(n),
                "n_champion": int(hit.sum()),
                "share": round(float(hit.mean()), 4) if n else 0.0,
                "n_sole": int((hit & sole).sum()),
                "n_sites": int(site.nunique()),
                "n_sites_champion": int(site[hit].nunique()),
            }
        )
    return pd.DataFrame(rows)


# ---- loading ----------------------------------------------------------------


def load_run(
    run: RunPaths,
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """One run's error matrix, columns in `TERM_ORDER`, and its sites."""
    errors, sites = E.error_matrix(run, nside, methods=methods, analysis_root=analysis_root)
    order = method_order(errors.columns)
    if len(order) > MAX_METHODS:
        raise ValueError(
            f"{run.run_id} scores {len(order)} methods ({order}); an UpSet over more "
            f"than {MAX_METHODS} has up to {2 ** len(order) - 1} columns. Pick a "
            f"subset with --method."
        )
    return errors[order], sites


def stack_runs(
    by_run: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Every run's matrix as one population: `(errors, sites, run_id per row)`.

    Strict coverage and disjoint TG ids, with the error CDF's remedies: a
    method missing from one run would lose every contest there, and a shared
    TG id would sit in the denominator twice.
    """
    if not by_run:
        raise ValueError("pooling needs at least one run")
    common = cross.guard_common_methods(
        {rid: set(e.columns) for rid, (e, _) in by_run.items()}, remedy=E.REMEDY_COMMON
    )
    cross.guard_disjoint_tgs(
        {rid: set(e.index) for rid, (e, _) in by_run.items()}, remedy=E.REMEDY_DISJOINT
    )
    order = method_order(common)
    errors, sites, origin = [], [], []
    for rid, (e, s) in by_run.items():
        keys = [f"{rid}{RUN_KEY_SEP}{t}" for t in e.index]
        errors.append(e[order].set_axis(keys))
        sites.append(s.reindex(e.index).set_axis(keys))
        origin.append(pd.Series(rid, index=keys, name="run_id"))
    return pd.concat(errors), pd.concat(sites), pd.concat(origin)


# ---- drawing ----------------------------------------------------------------


def _bar_style(members: list[str], colors: dict[str, str]) -> dict:
    """A single-method column is that method's; a tie is neutral."""
    if len(members) == 1:
        return _method_style(members[0], colors)
    return {"color": TIE_INK, "edgecolor": TIE_INK, "linewidth": 0.0}


def plot_upset(
    mask: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    figsize: tuple[float, float] = FIGSIZE,
    dpi: int = 300,
) -> Path:
    """Intersection bars over a dot matrix, set-size bars to its left.

    Drawn on three plain axes rather than through `upsetplot`, which drops a
    category with no members -- and a method that never wins is a result.
    """
    import matplotlib as mpl

    order = list(mask.columns)
    labels = [method_label(m) for m in order]
    colors = upset_colors(order)
    n = len(mask)
    values = mask.to_numpy(dtype=bool)

    # Exact combinations, largest first; ties broken by degree then by the
    # dot pattern, so the layout is a function of the data alone.
    patterns: dict[tuple[bool, ...], int] = {}
    for row in map(tuple, values):
        patterns[row] = patterns.get(row, 0) + 1
    columns = sorted(
        patterns.items(),
        key=lambda kv: (-kv[1], sum(kv[0]), tuple(not b for b in kv[0])),
    )
    n_cols, n_rows = len(columns), len(order)
    totals = values.sum(axis=0)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(E._SURFACE)
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=(1.25, max(n_cols, 3) * 0.42),
        height_ratios=(1.7, max(n_rows, 2) * 0.24),
        wspace=0.20, hspace=0.04, top=0.86,
    )
    ax_bar = fig.add_subplot(grid[0, 1])
    ax_dot = fig.add_subplot(grid[1, 1], sharex=ax_bar)
    ax_set = fig.add_subplot(grid[1, 0], sharey=ax_dot)
    # The row names are the matrix's tick labels and sit in the gap to its
    # left, over the set-size axis; drawn above it, or the longest bar cuts
    # its own method's name.
    ax_dot.set_zorder(2)
    ax_dot.patch.set_alpha(0.0)
    ax_set.set_zorder(1)
    ys = np.arange(n_rows)[::-1]  # first method on top

    # -- intersections
    for x, (pattern, count) in enumerate(columns):
        members = [m for m, hit in zip(order, pattern) if hit]
        pct = 100.0 * count / n if n else 0.0
        ax_bar.bar(x, pct, width=0.62, zorder=2, **_bar_style(members, colors))
        ax_bar.annotate(
            f"{pct:.1f}", xy=(x, pct), xytext=(0, 1.5), textcoords="offset points",
            ha="center", va="bottom", fontsize=E._GUIDE_PT, color=E._INK_2,
        )
    ax_bar.set_ylabel(INTERSECTION_LABEL, fontsize=E._GUIDE_PT + 0.5, color=E._INK_2)
    top = max((100.0 * c / n for _, c in columns), default=1.0) if n else 1.0
    ax_bar.set_ylim(0, top * 1.14)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.tick_params(axis="y", colors=E._MUTED, labelsize=E._GUIDE_PT)
    ax_bar.grid(True, axis="y", color=E._GRID, linewidth=0.5, zorder=0)
    ax_bar.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax_bar.spines[side].set_visible(False)
    ax_bar.spines["left"].set_color(E._AXIS)

    # -- dot matrix
    for k, y in enumerate(ys):
        if k % 2 == 0:
            ax_dot.axhspan(y - 0.5, y + 0.5, color=ROW_SHADE, zorder=0, linewidth=0)
    for x, (pattern, _) in enumerate(columns):
        hit_y = [ys[k] for k, hit in enumerate(pattern) if hit]
        miss_y = [ys[k] for k, hit in enumerate(pattern) if not hit]
        ax_dot.scatter([x] * len(miss_y), miss_y, s=16, color=ABSENT_DOT, zorder=2, linewidths=0)
        if len(hit_y) > 1:
            ax_dot.plot([x, x], [min(hit_y), max(hit_y)], color=E._INK, linewidth=1.3, zorder=3)
        ax_dot.scatter([x] * len(hit_y), hit_y, s=16, color=E._INK, zorder=4, linewidths=0)
    ax_dot.set_xlim(-0.6, n_cols - 0.4)
    ax_dot.set_ylim(-0.5, n_rows - 0.5)
    ax_dot.set_yticks(ys)
    ax_dot.set_yticklabels(labels)
    ax_dot.tick_params(axis="y", left=False, labelleft=True, pad=3, labelsize=E._TICK_PT)
    for tick, method in zip(ax_dot.get_yticklabels(), order):
        tick.set_color(colors[method])
        tick.set_fontweight("bold")
    ax_dot.tick_params(axis="x", bottom=False, labelbottom=False)
    for side in ax_dot.spines.values():
        side.set_visible(False)

    # -- set sizes, growing leftwards into the margin
    for k, (method, y) in enumerate(zip(order, ys)):
        pct = 100.0 * totals[k] / n if n else 0.0
        ax_set.barh(y, pct, height=0.58, zorder=2, **_method_style(method, colors))
        ax_set.annotate(
            f"{pct:.1f}", xy=(pct, y), xytext=(-2, 0), textcoords="offset points",
            ha="right", va="center", fontsize=E._GUIDE_PT, color=E._INK_2,
        )
    widest = 100.0 * totals.max() / n if n and totals.size else 1.0
    ax_set.set_xlim(max(widest, 1.0) * 1.35, 0)
    # Axis on top: the space above the set bars is empty anyway (the grid's
    # top-left cell), so the ticks and name live there instead of adding a
    # strip under the figure.
    ax_set.xaxis.tick_top()
    ax_set.xaxis.set_label_position("top")
    ax_set.set_xlabel(SET_LABEL, fontsize=E._GUIDE_PT + 0.5, color=E._INK_2, labelpad=3)
    ax_set.tick_params(axis="y", left=False, labelleft=False)
    ax_set.tick_params(axis="x", colors=E._MUTED, labelsize=E._GUIDE_PT, pad=1.5)
    ax_set.grid(True, axis="x", color=E._GRID, linewidth=0.5, zorder=0)
    ax_set.set_axisbelow(True)
    for side in ("bottom", "left", "right"):
        ax_set.spines[side].set_visible(False)
    ax_set.spines["top"].set_color(E._AXIS)

    # Anchored to the gridspec's own top rather than to `suptitle`'s default,
    # which floats an inch above the bars at this aspect.
    fig.text(0.5, 0.955, title, ha="center", va="bottom", fontsize=E._TITLE_PT,
             fontweight="bold", color=E._INK)
    fig.text(0.5, 0.925, subtitle, ha="center", va="bottom", fontsize=E._SUBTITLE_PT,
             color=E._INK_2)

    with mpl.rc_context({"hatch.color": E._INK_2, "hatch.linewidth": 0.6}):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=E._SURFACE)
    plt.close(fig)
    return out_path


# ---- artifacts --------------------------------------------------------------


def _manifest(
    layout: str,
    *,
    names: dict[str, str],
    mask: pd.DataFrame,
    sets: pd.DataFrame,
    run_ids: list[str],
    origin: pd.Series,
    tie_km: float,
    nside: int,
) -> str:
    degree = mask.sum(axis=1)
    order = list(mask.columns)
    per_run = {r: int(c) for r, c in origin.value_counts().sort_index().items()}
    body: dict = {
        "figure": names["png"],
        "tables": {k: names[k] for k in ("intersections", "sets", "membership")},
        "layout": layout,
        "runs": list(run_ids),
        "dataset": cross.dataset_slug(run_ids),
        "methods": order,
        "method_terms": method_term_table(order),
        "baseline": SHORTEST_PING,
        "dist_column": E.DIST_COLUMN,
        "tie_km": float(tie_km),
        "rule": (
            f"per TG, best = min {E.DIST_COLUMN} over the methods that answered; a "
            f"method is a champion when its error <= best + {tie_km:g} km. Every "
            "champion counts in full -- a tie credits each tied method."
        ),
        "row_policy": (
            "status.solved_mask via figure_error_cdf.error_matrix: FALLBACK and "
            "ERROR rows (and answered rows with no distance) never compete, so a "
            "fallback cannot win with the shortest-ping VP's coordinate. S-P's "
            "BASELINE rows are answered."
        ),
        "readings": {
            "intersections": (
                "exact combinations -- these methods champion and no other. They "
                "partition the TGs and sum to 100%."
            ),
            "sets": (
                "each method's champion share, ties included. They overlap and do "
                "not sum to 100%."
            ),
            "column_order": (
                "largest first, so a combination's x position is data-dependent; "
                "compare figures by their dots."
            ),
        },
        "n_tgs": int(len(mask)),
        "n_tgs_per_run": per_run,
        "share_single_champion": round(float((degree == 1).mean()), 4) if len(mask) else None,
        "share_tied": round(float((degree > 1).mean()), 4) if len(mask) else None,
        "n_no_champion": int((degree == 0).sum()),
        "n_champion_per_method": {
            r["method"]: int(r["n_champion"]) for _, r in sets.iterrows()
        },
        "empty_sets": [r["method"] for _, r in sets.iterrows() if r["n_champion"] == 0] or None,
        "site_caveat": (
            "~20 IP replicas share a site and its VP geometry, so TG counts overstate "
            "independent observations. n_sites (distinct run_id|lat,lon) is the "
            "lower bound beside every count."
        ),
        "source_rung": {
            "nside": int(nside),
            "note": f"{E.DIST_COLUMN} is identical at every rung; this only names the file read.",
        },
    }
    if layout == POOLED:
        body["pooling"] = {
            "rule": (
                "micro-pool: every run's error matrix stacked (<run_id>::<tg_id>), "
                "then judged row by row -- identical to judging each run and "
                "concatenating. A dataset weighs by its TG count."
            ),
            "largest_share": (
                round(max(per_run.values()) / len(mask), 4) if per_run and len(mask) else None
            ),
            "coverage": (
                "strict -- a method absent from any run is refused, and overlapping "
                "TG ids are refused."
            ),
        }
    return json.dumps(body, indent=2) + "\n"


def _write(
    errors: pd.DataFrame,
    sites: pd.Series,
    origin: pd.Series,
    out_dir: Path,
    layout: str,
    *,
    run_ids: list[str],
    tie_km: float,
    nside: int,
    subtitle: str,
) -> dict[str, Path]:
    """Both tables, the per-TG audit, the PNG and the manifest for one layout."""
    names = artifact_names(layout, tie_km)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = champion_mask(errors, tie_km)
    run_col = "+".join(sorted(run_ids))
    dataset = cross.dataset_slug(run_ids)

    def stamp(table: pd.DataFrame) -> pd.DataFrame:
        table.insert(0, "run_id", run_col)
        table.insert(1, "dataset", dataset)
        table.insert(2, "tie_km", float(tie_km))
        return table

    inter = stamp(intersection_table(mask, sites))
    sets = stamp(set_table(mask, sites))
    written = {k: out_dir / names[k] for k in KINDS}
    inter[list(INTERSECTION_COLUMNS)].to_csv(written["intersections"], index=False)
    sets[list(SET_COLUMNS)].to_csv(written["sets"], index=False)

    audit = errors.rename(columns=lambda m: f"{m}_km").copy()
    audit.insert(0, "run_id", origin.reindex(errors.index).to_numpy())
    audit.insert(1, "tg_id", [str(k).split(RUN_KEY_SEP, 1)[-1] for k in errors.index])
    audit["best_km"] = errors.min(axis=1, skipna=True)
    for m in mask.columns:
        audit[f"{m}_champion"] = mask[m]
    audit["n_champions"] = mask.sum(axis=1)
    audit["members"] = combination(mask)
    audit.to_csv(written["membership"], index=False)

    plot_upset(
        mask, written["png"],
        title="Lowest error per TG", subtitle=subtitle,
    )
    written["manifest"].write_text(
        _manifest(
            layout, names=names, mask=mask, sets=sets, run_ids=run_ids,
            origin=origin, tie_km=tie_km, nside=nside,
        )
    )
    return written


def _subtitle(name: str, n: int, tie_km: float) -> str:
    return f"{name} · n={n:,} TGs · champion = within {tie_km:g} km of the lowest error"


def build_for_run(
    run: RunPaths,
    *,
    nside: int = SOURCE_NSIDE,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    tie_km: float = DEFAULT_TIE_KM,
) -> dict[str, Path]:
    """One run's artifact set, written into `classify/` beside `error_cdf.png`."""
    tie_km = validate_tie(tie_km)
    errors, sites = load_run(run, nside, methods=methods, analysis_root=analysis_root)
    origin = pd.Series(run.run_id, index=errors.index, name="run_id")
    return _write(
        errors, sites, origin,
        run.analysis_dir(CLASSIFY_KIND, root=analysis_root),
        PER_RUN,
        run_ids=[run.run_id], tie_km=tie_km, nside=nside,
        subtitle=_subtitle(run.run_id, len(errors), tie_km),
    )


def build_for_runs(
    runs: list[RunPaths],
    *,
    layouts: tuple[str, ...] = (PER_RUN,),
    nside: int = SOURCE_NSIDE,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    tie_km: float = DEFAULT_TIE_KM,
) -> list[dict[str, Path]]:
    """Render the requested layouts; one artifact set per figure, layout-major."""
    ordered = tuple(dict.fromkeys(layouts)) or (PER_RUN,)
    unknown = [x for x in ordered if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layout {unknown}; pick from {list(LAYOUTS)}")
    nside = G.validate_nside(nside)
    tie_km = validate_tie(tie_km)
    run_ids = [r.run_id for r in runs]

    out: list[dict[str, Path]] = []
    for layout in ordered:
        if layout == PER_RUN:
            out.extend(
                build_for_run(
                    run, nside=nside, methods=methods,
                    analysis_root=analysis_root, tie_km=tie_km,
                )
                for run in runs
            )
            continue
        by_run = {
            run.run_id: load_run(run, nside, methods=methods, analysis_root=analysis_root)
            for run in runs
        }
        errors, sites, origin = stack_runs(by_run)
        out.append(
            _write(
                errors, sites, origin,
                cross.cross_dir(run_ids, analysis_root=analysis_root),
                POOLED,
                run_ids=run_ids, tie_km=tie_km, nside=nside,
                subtitle=_subtitle(
                    f"{cross.dataset_slug(run_ids).upper()} pooled", len(errors), tie_km
                ),
            )
        )
    return out
