"""Error distance against class error: two band figures, one renderer.

§2.4(a) claims accuracy and error distance are different metrics and that the
disagreement is itself a finding. The accuracy table reports one axis and the
error CDF the other; these are the only figures that put both on the same
point, so the claim becomes readable rather than rhetorical — and each method
turns out to have its own signature in the pair.

One band per class-error level, one thin line per target inside it, x = the
coordinate error on a log axis.

## Two y modes, because the two quantities are not the same

* **`cells`** — `seeds_crossed`, how many class boundaries lie between the true
  cell and the predicted one. "Two cells away" is a statement about the answer
  space's local density, which is the nearest-seed snapping story §8.1 asks
  about.
* **`rank`** — the true class's **top-cell index**: 1 if it is the cell closest
  to the estimate, 2 if one other cell is closer, and so on. This is
  the quantity the reported metric is built on, and the 1-indexing is what
  makes it read directly — `index <= N` *is* top-N, where the underlying
  `tg_seed_rank < N` needed translating. Band 1 is top-1 accuracy and the
  cumulative share through band 3 is top-3.

SCHEMA.md warns these get confused and that they order the methods
differently — on as01, 42% of wrong rows are at rank 1 while 67% are one
boundary away, and by rank Octant-Spline leads SoI while by crossings SoI
leads. They therefore live in one module, so the distinction is documented
once and the two figures share an x axis, a renderer and a denominator; the CLI
exposes them as separate commands so each figure has its own name.

## Density is drawn, not binned and not jittered

Each point is a vertical line spanning its band, in the method's hue at low
alpha, so coincident values darken by overplotting. No bin width to choose and
no random offset to mislead — an x position on the figure is an x position in
the data.

The one limit: alpha accumulation saturates at roughly 1/alpha coincident lines
(about 7 at alpha 0.15). Shortest-Ping and SoI have exact repeated errors,
because many targets share a VP coordinate and the answer *is* that
coordinate, so their darkest stripes stop distinguishing 7 coincident points
from 30. The band's row-share label carries the count, so this costs
within-band shading detail and no reported number.

## The denominator is every target, so band 0 is the accuracy

The correct band — 0 for `cells`, 1 for `rank` — means the method named the
true class: `seeds_crossed == 0` and `tg_seed_rank == 0` are the same event
(the crossing matrix is >= 1 off the diagonal), and both are top-1 correctness. `topn_accuracy.csv` divides by
*every* target and counts fallbacks as failures, so this figure must too —
dividing by solved rows instead would make that band disagree with the
accuracy the paper reports. On as02 Vanilla, 123 band-0 rows over 412 targets is 0.298,
its top-1 accuracy exactly; over its 337 solved rows it would read 0.365 and
match nothing.

The bands therefore sum to `1 - fallback_rate` rather than to 1, and the
shortfall is labelled: Vanilla's four bands total 81.8% and the missing 18.2%
is where its fallbacks went.

## One run per figure, or two ways of putting several together

Passing two or more `--run-id` renders a cross-dataset figure, and `--layout`
picks which question it answers. They are different questions, not two
renderings of one, so both are kept and each writes its own file.

**`pooled` (the default)** merges the runs' targets into one population and
draws this module's own six-panel grid over it — same renderer, so a pooled
panel is read exactly like a single-run one. The operator runs' target sets are
**disjoint** (399 + 412 + 458 distinct ids, checked by
`guard_disjoint_targets` rather than assumed), so the union is a 1,269-target
population and not a double count; this is the same pooling `plot-venn`
performs over the same three runs. It is the fleet-wide answer, and it is the
form where the density encoding finally has enough targets to read as a
distribution rather than as a handful of stripes.

A pooled share is a **micro**-average — pick a target at random from the fleet
— so it is target-weighted and as03 carries 36% of it. The subtitle says so,
and the manifest's `weighting` block reports the macro-average (the mean of the
three datasets' shares) beside it. On as01/02/03 the two agree to within
0.6 pp on every method, so the weighting is a caveat that has been measured
rather than a distortion to worry about.

**`compare`** keeps each run separate: one panel per (method, dataset), methods
down the rows and datasets across the columns, sharing one error axis and one
band stack. Each panel's counts, shares and medians come from that run's own
`load_points` / `band_table`, so a panel is identical to the same panel in that
run's own figure. It answers whether a method's signature survives a change of
dataset, and on as01/02/03 it does not uniformly: Octant-Hull's correct band
runs 72.9% / 65.0% / 50.2% while its one-cell-out band runs
21.6% / 28.6% / 40.2%, so the same method degrades by sliding one band up
rather than by scattering.

`--all-runs` stays per-run either way: which datasets belong in one figure is a
claim about comparability (§7.3), so it is named rather than discovered, and
mixing the two run families is refused by `cross.guard_one_setup`.

Commands: `plot-error-vs-cells`, `plot-error-vs-rank`. One run writes to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`;
several write to `outputs/analysis/v3/_cross/error-vs-class/<dataset-set>/`,
with the comparison grid suffixed `_by_dataset`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import cross, io
from scripts.analysis.v3.modules.answer_space import (
    load_answer_space,
    seed_crossing_matrix,
)
from scripts.analysis.v3.modules.confusion import load_scored
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.labels import (
    PUBLISHED_METHODS,
    short_label,
)
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
    _SURFACE,
    method_colors,
)
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    RunPaths,
    discover_runs,
    grid_slug,
    resolve_run,
)

#: The error axis is shared with `plot-error-cdf` on purpose — same column, same
#: row filter, same bounds — so a reader can put the figures side by side and
#: read one x position across all of them. That module owns the error axis;
#: these borrow it rather than declaring bounds free to drift.
from scripts.analysis.v3.modules.figure_error_cdf import (  # noqa: E402
    DEFAULT_X_MAX_KM,
    ERROR_COLUMN,
    X_MIN_KM,
)


@dataclass(frozen=True)
class YMode:
    """One class-error axis: where its values come from and how it is numbered."""

    key: str
    #: Column on the scored frame, or `None` when the values are walked from the
    #: answer space rather than read off a row.
    column: str | None
    #: What the raw value is shifted by before it is drawn or tabulated.
    #:
    #: `cells` counts boundaries, so 0 is "no boundary crossed" and 0-indexing
    #: is the only honest numbering. `rank` is an *index into* the cells ordered
    #: by distance from the estimate, so it is 1-indexed: index 1 means the true
    #: cell is the nearest one. That makes the axis read straight off top-N —
    #: `index <= N` is exactly top-N — where `rank < N` needed translating.
    index_base: int
    #: Bands drawn. The top one is a bucket whenever anything exceeds it.
    n_bands: int
    axis_label: str
    stem: str
    title: str

    @property
    def levels(self) -> range:
        return range(self.index_base, self.index_base + self.n_bands)

    @property
    def top_level(self) -> int:
        return self.index_base + self.n_bands - 1


CELLS = YMode(
    key="cells",
    column=None,
    index_base=0,
    n_bands=4,
    axis_label="Cells away from the true class",
    stem="error_vs_cells",
    title="Coordinate error against class distance",
)

RANK = YMode(
    key="rank",
    column="tg_seed_rank",
    index_base=1,
    n_bands=4,
    axis_label="Top cell index of the true class",
    stem="error_vs_rank",
    title="Coordinate error against top-cell index",
)

MODES: dict[str, YMode] = {CELLS.key: CELLS, RANK.key: RANK}

#: Rows are contiguous grid cells of height 1, so the bottom row's lower edge
#: *is* the x axis and the separators between rows read as a grid rather than as
#: bands floating at a tick. Everything below is a fraction of one row.
#:
#: The band sits on its row's floor and stops short of the ceiling: the strip
#: above it is the gutter the median readout occupies. Without that gutter the
#: number would land inside the row above and name the wrong band.
#:
#: There is deliberately no gutter *below*. A band resting on its separator
#: reads as sitting on the row's floor, and it makes the bottom band rest on the
#: x axis itself rather than hovering a tenth of a row above it.
BAND_BOTTOM = 0.0
BAND_TOP = 0.70
MEDIAN_TEXT_Y = 0.72

#: Inches of figure height per band. This is the knob that compacts the y
#: direction: the fractions above are shares of a row, so shrinking the row
#: shortens the band *and* its gutter together, where trading one fraction
#: against the other can only move height from one to the other.
#:
#: At 0.50 the gutter is 0.30 x 0.50 = 0.15 in, against ~0.10 in for the 7 pt
#: median readout — the floor this cannot go below without the number touching
#: the band.
ROW_INCHES = 0.50

#: Inches per panel row for the title, x tick labels and padding.
PANEL_CHROME_INCHES = 1.7

#: Per-line alpha. Low enough that ~7 coincident targets read as solid and a
#: lone one is still visible. Past that the band saturates; see the module
#: docstring.
LINE_ALPHA = 0.15
LINE_WIDTH = 0.9


def load_points(
    run: RunPaths,
    mode: YMode,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """One row per (method, placed target), plus each method's row counts.

    `counts[method]["n_targets"]` is every scored row including fallbacks, and
    it is the denominator every share in this figure uses — see the module
    docstring for why solved rows would be the wrong one.
    """
    cls_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    chosen = list(methods) if methods else list(PUBLISHED_METHODS)

    crossings = pos = None
    if mode.column is None:
        space = load_answer_space(
            run.answer_space_dir(root=analysis_root, grid=grid, resolution=resolution)
        )
        crossings = seed_crossing_matrix(space.seeds)
        pos = {int(s): i for i, s in enumerate(space.seeds["seed_id"].to_numpy())}

    frames: list[pd.DataFrame] = []
    counts: dict[str, dict] = {}
    for method in chosen:
        df = load_scored(cls_dir, method)
        n_targets = int(len(df))
        solved = df.loc[io.solved_mask(df)]
        if mode.column is None:
            placed = solved.loc[solved["pred_seed_id"] >= 0]
            values = np.array(
                [
                    crossings[pos[int(t)], pos[int(p)]]
                    for t, p in zip(placed["tg_seed_id"], placed["pred_seed_id"])
                ],
                dtype=int,
            )
        else:
            placed = solved.loc[solved[mode.column] >= 0]
            values = placed[mode.column].to_numpy(dtype=int)
        counts[method] = {
            "n_targets": n_targets,
            "n_solved": int(len(solved)),
            "n_placed": int(len(placed)),
        }
        if placed.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "method_label": short_label(method),
                    "target_id": placed["target_id"].to_numpy(),
                    "error_km": placed[ERROR_COLUMN].to_numpy(dtype=float),
                    "class_error": values,
                    "level": np.clip(
                        values + mode.index_base, mode.index_base, mode.top_level
                    ),
                }
            )
        )
    if not frames:
        raise ValueError(
            f"no placed rows for {chosen} on {run.run_id} at "
            f"{grid_slug(grid, resolution)}; run `classify` first"
        )
    return pd.concat(frames, ignore_index=True), counts


def band_table(
    points: pd.DataFrame, counts: dict[str, dict], mode: YMode
) -> pd.DataFrame:
    """Per (method, band): count, share of **all** targets, error percentiles.

    `share` is over `n_targets`, so `share` at level 0 is the method's top-1
    accuracy and the bands sum to `1 - fallback_rate`. `cumulative_share` makes
    the rank mode's top-N reading direct: through level 2 it is `accuracy_top3`.
    """
    rows: list[dict] = []
    for method, g in points.groupby("method", sort=False):
        n_targets = counts.get(method, {}).get("n_targets", len(g)) or 1
        running = 0
        for level in mode.levels:
            gl = g.loc[g["level"] == level]
            running += len(gl)
            rows.append(
                {
                    "method": method,
                    "method_label": short_label(method),
                    "y_mode": mode.key,
                    "level": level,
                    "n": len(gl),
                    "n_targets": n_targets,
                    "share": round(len(gl) / n_targets, 4),
                    "cumulative_share": round(running / n_targets, 4),
                    "error_km_p25": round(float(gl["error_km"].quantile(0.25)), 3)
                    if len(gl)
                    else np.nan,
                    "error_km_p50": round(float(gl["error_km"].median()), 3)
                    if len(gl)
                    else np.nan,
                    "error_km_p75": round(float(gl["error_km"].quantile(0.75)), 3)
                    if len(gl)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def level_labels(mode: YMode, max_observed: int) -> list[str]:
    """Band tick labels: bare integers, with a `+` on the top one when it buckets.

    Nothing but the number. What the first band *means* is carried by the axis
    label and the footnote, so the ticks stay a clean integer scale rather than
    one annotated value beside three plain ones.

    `max_observed` is the raw value, so it is shifted by `index_base` before
    being compared with the top band.
    """
    top = mode.top_level
    labels = [str(i) for i in mode.levels if i != top]
    labels.append(f"{top}+" if max_observed + mode.index_base > top else str(top))
    return labels


def band_span(mode: YMode, level: int) -> tuple[float, float]:
    """`(bottom, top)` of a band in axes data units.

    Row `k = level - index_base` occupies `[k, k+1]`; the band rests on that
    row's floor and stops below its ceiling, leaving the gutter that holds the
    median readout.
    """
    k = level - mode.index_base
    return k + BAND_BOTTOM, k + BAND_TOP


def band_centre(mode: YMode, level: int) -> float:
    """Where the band's tick label goes — the band's middle, not the row's.

    The tick names the data, so it lines up with the data rather than with the
    grid cell that also contains the readout gutter.
    """
    lo, hi = band_span(mode, level)
    return (lo + hi) / 2


def _fmt_km(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


def panel_header(bands: pd.DataFrame) -> str:
    """`n = 412 · 18.2% fell back` — the panel's denominator and its shortfall.

    The bands sum to `1 - fallback_rate` rather than to 1, so the shortfall has
    to be named somewhere or the reader is left to infer it from percentages
    that do not add up. Read off the band table rather than recomputed, so it
    cannot disagree with the shares beside it.
    """
    if bands.empty:
        return ""
    n_targets = int(bands["n_targets"].iloc[0])
    drawn = int(bands["n"].sum())
    missing = n_targets - drawn
    header = f"n = {n_targets}"
    if missing and n_targets:
        header += f" · {missing / n_targets * 100:.1f}% fell back"
    return header


def _draw_panel(
    ax,
    *,
    group: pd.DataFrame,
    bands: pd.DataFrame,
    mode: YMode,
    hue: str,
    y_labels: list[str],
    min_x_km: float,
    max_x_km: float,
    title: str | None = None,
    title_color: str | None = None,
    header: str | None = None,
) -> None:
    """One panel's band stack: separators, target lines, medians, shares.

    Shared by the per-run figure (one panel per method) and the cross-dataset
    grid (one panel per method x dataset), so the two cannot drift in band
    geometry, alpha, median rule or share denominator — the whole point of the
    cross figure is that a band in it is the same object as a band in the
    per-run figure.

    `group` is this panel's points and `bands` its rows of `band_table`.
    """
    from matplotlib.ticker import FuncFormatter

    ax.set_facecolor(_SURFACE)
    shares: dict[int, float] = {}

    # Row separators first, so the grid sits under the data.
    for k in range(mode.n_bands + 1):
        ax.axhline(k, color=_C_GRID, linewidth=0.8, zorder=1)

    by_level = bands.set_index("level")
    for level in mode.levels:
        lo, hi = band_span(mode, level)
        gl = group.loc[group["level"] == level]
        if len(gl):
            # One line per target, no binning and no jitter: density comes
            # from lines landing on the same x, so a dark stripe is a real
            # cluster of targets rather than a bin that happens to be wide.
            ax.vlines(
                np.clip(gl["error_km"].to_numpy(dtype=float), min_x_km, max_x_km),
                lo,
                hi,
                color=hue,
                alpha=LINE_ALPHA,
                linewidth=LINE_WIDTH,
                zorder=2,
            )
        if level not in by_level.index:
            continue
        row = by_level.loc[level]
        p50 = float(row["error_km_p50"])
        if np.isfinite(p50):
            # Exactly the band's height, like every other line in it — the
            # median is one of the targets, not an annotation layer.
            ax.vlines(p50, lo, hi, color=_C_INK, linewidth=1.5, zorder=4)
            ax.annotate(
                _fmt_km(p50),
                xy=(p50, level - mode.index_base + MEDIAN_TEXT_Y),
                ha="center",
                va="bottom",
                fontsize=7,
                color=_C_INK,
                zorder=5,
            )
        shares[level] = float(row["share"])

    if header:
        ax.annotate(
            header,
            xy=(0.015, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7.5,
            color=_C_MUTED,
            family="monospace",
        )
    if title:
        ax.set_title(
            title,
            fontsize=10.5,
            fontweight="bold",
            color=title_color or hue,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(min_x_km, max_x_km)
    # Exactly the stack of rows: no padding, so the first row rests on the
    # x axis and the last closes the panel.
    ax.set_ylim(0, mode.n_bands)
    ax.set_yticks([band_centre(mode, lv) for lv in mode.levels])
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.grid(True, axis="x", which="major", color=_C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    ax.tick_params(colors=_C_MUTED, labelsize=8.5)

    # Shares hang off a twin y axis rather than off annotations at the panel
    # edge: placed text lands in the gap *between* panels, where it reads as
    # belonging to either, and inside the panel it collides with the long
    # tail on as01/as03 (Shortest-Ping's p95 is 3,871 km, ~93% along the log
    # axis). A right-hand tick is aligned with its band and cannot overlap.
    share_ax = ax.twinx()
    share_ax.set_ylim(ax.get_ylim())
    share_ax.set_yticks([band_centre(mode, lv) for lv in mode.levels])
    share_ax.set_yticklabels(
        [f"{shares.get(lv, 0.0) * 100:.1f}%" for lv in mode.levels],
        fontsize=7.5,
    )
    share_ax.tick_params(axis="y", length=0, colors=_C_INK_2)
    for tick in share_ax.get_yticklabels():
        tick.set_family("monospace")
    for side in ("top", "right", "left", "bottom"):
        share_ax.spines[side].set_visible(False)


def footnote_text(mode: YMode) -> str:
    """What the lines, the black rule and the percentages are.

    Shared by both figures so the per-run and cross-dataset renderings make the
    same promises about the same marks.
    """
    # Only the rank mode's bands accumulate into a reported metric: its index is
    # 1-based, so summing bands 1..N is top-N. Crossings do not have that
    # property (SCHEMA.md), so the clause is not printed on that figure.
    cumulative_note = (
        f", and bands 1-{mode.index_base + 2} sum to top-{mode.index_base + 2}"
        if mode is RANK
        else ""
    )
    return (
        "One line per target, no binning: a dark stripe is targets sharing an "
        "error. Black rule and figure above each band are its median error.\n"
        "Percentages on the right are shares of all targets, so the bottom "
        f"band is the method's top-1 accuracy{cumulative_note} and the bands "
        "sum to 100% minus fallbacks."
    )


def panel_methods(points: pd.DataFrame) -> list[str]:
    """Published order first, then anything `--method` added, in first-seen order."""
    methods = [m for m in PUBLISHED_METHODS if m in set(points["method"])]
    return methods + [m for m in points["method"].unique() if m not in methods]


def plot_bands(
    points: pd.DataFrame,
    table: pd.DataFrame,
    out_path: Path,
    *,
    mode: YMode,
    title: str,
    subtitle: str,
    max_observed: int,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> Path:
    """Small multiples, one panel per method, shared log x and shared bands.

    One panel per method rather than six hues in one frame: the bands would
    overlap into one another and the panel already carries the method's
    identity, so colour never has to separate six series here and the
    palette's all-pairs margin is not called on.
    """
    methods = panel_methods(points)
    colors = method_colors(methods)
    labels = level_labels(mode, max_observed)

    n_cols = 3
    n_rows = int(np.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            4.9 * n_cols,
            ROW_INCHES * mode.n_bands * n_rows
            + PANEL_CHROME_INCHES * n_rows
            + 1.2,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    by_method = {m: g for m, g in points.groupby("method", sort=False)}

    for idx, ax in enumerate(axes.flat):
        if idx >= len(methods):
            ax.set_visible(False)
            continue
        method = methods[idx]
        bands = table.loc[table["method"] == method]
        _draw_panel(
            ax,
            group=by_method[method],
            bands=bands,
            mode=mode,
            hue=colors[method],
            y_labels=labels,
            min_x_km=min_x_km,
            max_x_km=max_x_km,
            title=short_label(method),
            header=panel_header(bands),
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=0.995)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9.5, color=_C_INK_2)
    fig.text(
        0.5,
        0.075,
        "Error distance to the raw target (km)",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=_C_INK_2,
    )
    fig.supylabel(mode.axis_label, fontsize=10.5, color=_C_INK_2)
    fig.text(
        0.5,
        0.008,
        footnote_text(mode),
        ha="center",
        va="bottom",
        fontsize=8,
        color=_C_MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.115, 0.99, 0.945))
    # `tight_layout` spaces the columns for the widest tick label, which leaves
    # the share column sitting midway between two panels where it reads as
    # belonging to either. Tightened afterwards so each panel's shares hug its
    # own right spine.
    fig.subplots_adjust(wspace=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def build(
    run: RunPaths,
    *,
    mode: YMode,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict]:
    """Render and return `(png, points, table, manifest)`."""
    points, counts = load_points(
        run,
        mode,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    table = band_table(points, counts, mode)
    max_observed = int(points["class_error"].max())
    out_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    png = plot_bands(
        points,
        table,
        out_dir / f"{mode.stem}.png",
        mode=mode,
        title=mode.title,
        subtitle=f"{run.run_id} · {grid_slug(grid, resolution)}",
        max_observed=max_observed,
        max_x_km=max_x_km,
        min_x_km=min_x_km,
    )
    manifest = {
        "run_id": run.run_id,
        "grid": {"scheme": grid, "resolution": resolution},
        "y_mode": mode.key,
        "y_axis": mode.axis_label,
        "methods": [m for m in PUBLISHED_METHODS if m in set(points["method"])],
        "error_column": ERROR_COLUMN,
        "index_base": mode.index_base,
        "bands_drawn": list(mode.levels),
        "max_raw_value_observed": max_observed,
        "top_band_is_a_bucket": max_observed + mode.index_base > mode.top_level,
        "n_lines_drawn": int(len(points)),
        "counts": counts,
        "share_denominator": (
            "n_targets — every scored row, fallbacks included. Level 0 is "
            "therefore the method's top-1 accuracy exactly, matching "
            "topn_accuracy.csv, and the bands sum to 1 - fallback_rate rather "
            "than to 1. Dividing by solved rows would make level 0 disagree "
            "with the accuracy the paper reports (as02 Vanilla: 0.365 vs 0.298)."
        ),
        "correct_band": mode.index_base,
        "correct_band_meaning": (
            "the method named the true class. seeds_crossed == 0 and "
            "tg_seed_rank == 0 are the same event, since the crossing matrix is "
            ">= 1 off the diagonal, and both are top-1 correctness. The rank mode "
            "is 1-indexed, so its correct band is 1 and `index <= N` is top-N: "
            "cumulative share through band 3 is accuracy_top3."
        ),
        "density_encoding": (
            f"one {LINE_WIDTH} pt line per target at alpha {LINE_ALPHA}, no "
            "binning and no jitter, so an x position on the figure is an x "
            "position in the data. Accumulation saturates near "
            f"{int(1 / LINE_ALPHA)} coincident lines; Shortest-Ping and SoI hit "
            "that because many targets share a VP coordinate and the answer is "
            "that coordinate. The band's share label carries the count."
        ),
        "row_policy": (
            "solved rows with a placeable class error are drawn (FALLBACK "
            "excluded per §7.2; Shortest-Ping's all-BASELINE rows counted as "
            "solved), but every share divides by n_targets"
        ),
    }
    return png, points, table, manifest


# ---- cross-dataset ----------------------------------------------------------

#: This module's artifact kind under `_cross/`, the sibling of `venn`'s
#: `"venn-diagram"` and `pareto`'s `"cost-accuracy"`. One kind for both y modes:
#: the filenames are stemmed by mode (`error_vs_rank` / `error_vs_cells`), so
#: they coexist in one directory the way the per-run figures do.
CROSS_KIND = "error-vs-class"

#: Inches per panel row in the cross grid, against `PANEL_CHROME_INCHES` in the
#: per-run figure. Much smaller because the grid shares its x axis down each
#: column and carries the method name as a row label rather than a per-panel
#: title, so a row needs padding and nothing else. The bands themselves are
#: `ROW_INCHES` tall in both figures — that is what makes a band in one
#: comparable to a band in the other.
CROSS_ROW_CHROME_INCHES = 0.42

#: The two cross-dataset layouts.
#:
#: * `POOLED` merges the runs' targets into one population and draws the
#:   per-run figure's own six-panel grid over it. The three operator runs have
#:   **disjoint** target sets, so their union is a population rather than a
#:   double count — the same pooling `plot-venn` does over the same three runs.
#:   Read it as the fleet-wide answer to "how often is each method right, and
#:   how wrong is it when it is not".
#: * `COMPARE` keeps each run separate, one panel per (method, dataset), and
#:   answers whether a method's signature survives a change of dataset.
#:
#: They are different questions rather than a rendering choice, so both are
#: kept and each writes its own file.
POOLED = "pooled"
COMPARE = "compare"
LAYOUTS: tuple[str, ...] = (POOLED, COMPARE)


def layout_stem(mode: YMode, layout: str) -> str:
    """Filename stem per layout, so the two coexist in one directory.

    The pooled figure keeps the bare stem because it is the cross-dataset
    figure a reader wants by default; the comparison grid is suffixed with what
    makes it different.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    return mode.stem if layout == POOLED else f"{mode.stem}_by_dataset"


def load_cross_points(
    runs: dict[str, RunPaths],
    mode: YMode,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Every run's points and bands in one frame each, keyed by `dataset`.

    Each run goes through the same `load_points` / `band_table` pair the
    per-run figure uses, so a `dataset` share here is the same number as the
    share in that run's own figure. This loader never pools: the returned
    `table` is per-dataset, and the pooled layout re-derives its own bands from
    the pooled `points` and `pool_counts`. Both layouts therefore start from
    one read of the data.
    """
    points: list[pd.DataFrame] = []
    tables: list[pd.DataFrame] = []
    counts: dict[str, dict] = {}
    for run_id in sorted(runs, key=lambda r: cross.short_dataset(r)):
        run = runs[run_id]
        dataset = cross.short_dataset(run_id)
        pts, cnt = load_points(
            run,
            mode,
            analysis_root=analysis_root,
            grid=grid,
            resolution=resolution,
            methods=methods,
        )
        tbl = band_table(pts, cnt, mode)
        points.append(pts.assign(dataset=dataset, run_id=run_id))
        tables.append(tbl.assign(dataset=dataset, run_id=run_id))
        counts[dataset] = {"run_id": run_id, "methods": cnt}
    return (
        pd.concat(points, ignore_index=True),
        pd.concat(tables, ignore_index=True),
        counts,
    )


def pool_counts(counts: dict[str, dict]) -> dict[str, dict]:
    """Per method, the runs' row counts added up into one population.

    `n_targets` is the sum over the runs that *have* that method, so a method
    missing from one run is scored on the targets it was actually run on rather
    than penalized for the rest — the same rule `pareto.dataset_lines` applies
    when a variant is absent. All six published variants exist on all three
    operator runs today, so this is a guard rather than a live case; the
    manifest records the per-method dataset list so it cannot go unnoticed.

    Pooling is legitimate here because the runs' target sets are **disjoint**:
    as01/02/03 draw 399, 412 and 458 distinct targets, so the union is a
    1,269-target population and not a double count. `guard_disjoint_targets`
    checks that rather than trusting it.

    What pooling *does* do is weight by target count: as03 carries 36% of the
    pooled denominator. So a pooled share is a micro-average — "pick a target
    at random from the fleet" — and not the mean of the three datasets'
    accuracies. The pooled manifest reports both so the gap is visible.
    """
    pooled: dict[str, dict] = {}
    for dataset, entry in counts.items():
        for method, cnt in entry["methods"].items():
            acc = pooled.setdefault(
                method,
                {"n_targets": 0, "n_solved": 0, "n_placed": 0, "datasets": []},
            )
            for key in ("n_targets", "n_solved", "n_placed"):
                acc[key] += int(cnt.get(key, 0))
            acc["datasets"].append(dataset)
    for cnt in pooled.values():
        cnt["datasets"] = sorted(cnt["datasets"])
    return pooled


def guard_disjoint_targets(points: pd.DataFrame) -> dict[str, int]:
    """Refuse to pool runs that share a `target_id`.

    A shared target would be counted once per run in the pooled denominator and
    drawn twice in the band, so the pooled share would stop being a rate over a
    population. The three operator runs are disjoint by construction — each
    draws its targets from its own AS — but that is a property of the data, and
    a re-run with an overlapping target list would otherwise pool silently.
    """
    if "dataset" not in points.columns:
        return {}
    per_method = points.groupby("method")["target_id"]
    dupes = {
        str(method): int(ids.duplicated().sum())
        for method, ids in per_method
        if ids.duplicated().any()
    }
    if dupes:
        worst = max(dupes.items(), key=lambda kv: kv[1])
        raise typer.BadParameter(
            f"the selected runs share targets ({worst[1]} repeated ids on "
            f"{worst[0]}, {len(dupes)} methods affected), so pooling them would "
            "count those targets once per run in the denominator and draw them "
            f"twice in the band. Use --layout {COMPARE}, which keeps each run's "
            "numbers separate."
        )
    return dupes


def cross_grid_shape(points: pd.DataFrame) -> tuple[int, int]:
    """`(n_rows, n_cols)` of the cross grid: methods down, datasets across.

    Not the transpose. The figure answers "does this method's pattern survive a
    change of dataset", which is a within-method read, so a method owns a row
    and its datasets sit side by side in it.
    """
    return len(panel_methods(points)), int(points["dataset"].nunique())


def cross_figsize(n_rows: int, n_cols: int, mode: YMode) -> tuple[float, float]:
    """Inches for the cross grid.

    Width per column is the per-run figure's panel width, so the log x axis is
    the same length in both and an x position can be read across them. Height
    per row is `ROW_INCHES * n_bands` — also the per-run figure's — plus a
    smaller chrome allowance, since the grid shares its x axis down each column
    and names the method in a row label rather than a per-panel title.
    """
    return (
        4.9 * n_cols,
        (ROW_INCHES * mode.n_bands + CROSS_ROW_CHROME_INCHES) * n_rows + 1.6,
    )


def plot_cross_bands(
    points: pd.DataFrame,
    table: pd.DataFrame,
    out_path: Path,
    *,
    mode: YMode,
    title: str,
    subtitle: str,
    max_observed: int,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> Path:
    """One panel per (method, dataset): methods down the rows, datasets across.

    ## Why this orientation

    The per-run figure answers "what does each method's error/class pattern
    look like"; this one answers "does that pattern survive a change of
    dataset", which is a *within-method, across-dataset* read. So a method owns
    a row and its three datasets sit side by side in it, left to right. The
    transpose would put the same panels on the page and make the cross-dataset
    comparison a vertical scan across two intervening rows.

    Three columns rather than six also keeps the panel width — and therefore
    the log x axis — identical to the per-run figure, which is the whole basis
    for reading an x position across the two. Six method columns would need
    ~3 in panels and could not label five decades without collisions.

    ## What is shared and what is not

    x and y are shared across every panel: same error axis, same band stack,
    same top-band bucket label, so any two panels are directly comparable. The
    numbers are not shared — each panel's `n`, shares and medians are its own
    run's, and each panel prints its own `n` because the runs differ in size
    (399 / 412 / 458 targets).
    """
    methods = panel_methods(points)
    datasets = sorted(points["dataset"].unique())
    colors = method_colors(methods)
    labels = level_labels(mode, max_observed)

    n_rows, n_cols = cross_grid_shape(points)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=cross_figsize(n_rows, n_cols, mode),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    groups = {k: g for k, g in points.groupby(["method", "dataset"], sort=False)}

    for r, method in enumerate(methods):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            bands = table.loc[
                (table["method"] == method) & (table["dataset"] == dataset)
            ]
            _draw_panel(
                ax,
                group=groups.get(
                    (method, dataset), points.iloc[0:0]
                ),
                bands=bands,
                mode=mode,
                hue=colors[method],
                y_labels=labels,
                min_x_km=min_x_km,
                max_x_km=max_x_km,
                # The dataset names the column, so it is a header on the top
                # row only and in ink — it is not a series, and painting it in
                # the row's method hue would imply it were one.
                title=dataset.upper() if r == 0 else None,
                title_color=_C_INK,
                header=panel_header(bands),
            )
            if c == 0:
                # The method rides on the left in its own hue, once per row,
                # rather than once per panel: the row *is* the method, and
                # repeating the name three times across it would say so three
                # times while crowding out the band tick labels.
                ax.set_ylabel(
                    short_label(method),
                    fontsize=10.5,
                    fontweight="bold",
                    color=colors[method],
                    labelpad=8,
                )

    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=0.997)
    fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=9.5, color=_C_INK_2)
    fig.text(
        0.5,
        0.048,
        "Error distance to the raw target (km)",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=_C_INK_2,
    )
    # The method names occupy the per-axes ylabel slot, so the class-error axis
    # label goes outside them at figure level.
    fig.supylabel(mode.axis_label, fontsize=10.5, color=_C_INK_2, x=0.004)
    fig.text(
        0.5,
        0.006,
        footnote_text(mode),
        ha="center",
        va="bottom",
        fontsize=8,
        color=_C_MUTED,
    )
    fig.tight_layout(rect=(0.03, 0.062, 0.99, 0.968))
    fig.subplots_adjust(wspace=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def weighting_check(
    per_dataset: pd.DataFrame, pooled: pd.DataFrame, mode: YMode
) -> dict[str, dict]:
    """Per method, the pooled correct-band share against the datasets' mean.

    A pooled share is a micro-average: pick a target at random from the fleet.
    The mean of the three datasets' shares is a macro-average: pick a dataset,
    then a target. They differ whenever accuracy correlates with dataset size,
    and the difference is the whole content of the "pooling weights as03
    heaviest" caveat — so it is measured and recorded rather than warned about.
    """
    correct_d = per_dataset.loc[per_dataset["level"] == mode.index_base]
    correct_p = pooled.loc[pooled["level"] == mode.index_base].set_index("method")
    out: dict[str, dict] = {}
    for method, g in correct_d.groupby("method"):
        if method not in correct_p.index:
            continue
        micro = float(correct_p.loc[method, "share"])
        macro = float(g["share"].mean())
        out[str(method)] = {
            "pooled_micro": round(micro, 4),
            "dataset_macro_mean": round(macro, 4),
            "delta": round(micro - macro, 4),
            "per_dataset": {
                str(d): round(float(v), 4)
                for d, v in zip(g["dataset"], g["share"])
            },
        }
    return out


def build_cross(
    runs: dict[str, RunPaths],
    *,
    mode: YMode,
    layouts: tuple[str, ...] | list[str] = (POOLED,),
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
    out_dir: Path | None = None,
) -> list[dict]:
    """Render the requested cross-dataset layouts from **one** read of the data.

    Returns one entry per layout: `{"layout", "stem", "png", "points",
    "table", "manifest"}`. The loader runs once whatever is asked for, since
    reading and walking the answer space is the expensive part and both layouts
    consume the same rows.
    """
    unknown = [lay for lay in layouts if lay not in LAYOUTS]
    if unknown:
        raise typer.BadParameter(
            f"unknown layout(s) {unknown}; expected any of {list(LAYOUTS)}"
        )
    points, per_dataset, counts = load_cross_points(
        runs,
        mode,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    max_observed = int(points["class_error"].max())
    slug = grid_slug(grid, resolution)
    datasets = sorted(points["dataset"].unique())
    target_dir = out_dir or cross.cross_dir(analysis_root, runs.keys(), kind=CROSS_KIND)

    common = {
        "run_ids": sorted(runs),
        "datasets": datasets,
        "grid": {"scheme": grid, "resolution": resolution},
        "y_mode": mode.key,
        "y_axis": mode.axis_label,
        "methods": panel_methods(points),
        "error_column": ERROR_COLUMN,
        "index_base": mode.index_base,
        "bands_drawn": list(mode.levels),
        "max_raw_value_observed": max_observed,
        "top_band_is_a_bucket": max_observed + mode.index_base > mode.top_level,
        "n_lines_drawn": int(len(points)),
    }

    built: list[dict] = []
    for layout in layouts:
        stem = layout_stem(mode, layout)
        if layout == POOLED:
            guard_disjoint_targets(points)
            pooled_counts = pool_counts(counts)
            table = band_table(points, pooled_counts, mode)
            n_pooled = max(
                (c["n_targets"] for c in pooled_counts.values()), default=0
            )
            png = plot_bands(
                points,
                table,
                target_dir / f"{stem}.{slug}.png",
                mode=mode,
                title=mode.title,
                # "target-weighted" rides on the figure rather than only in the
                # manifest: the shares here are a micro-average over the merged
                # population, so the largest run carries the most weight, and a
                # reader comparing this to a per-dataset number needs to know
                # which average they are looking at.
                subtitle=(
                    f"{' + '.join(datasets)} pooled · {slug} · "
                    f"{n_pooled:,} targets · shares are target-weighted"
                ),
                max_observed=max_observed,
                max_x_km=max_x_km,
                min_x_km=min_x_km,
            )
            manifest = {
                **common,
                "cross_layout": layout,
                "layout": (
                    "the per-run figure's own six-panel grid, one panel per "
                    "method, drawn over the runs' merged targets. Identical "
                    "renderer, so a pooled panel is read exactly like a "
                    "single-run one."
                ),
                "pooling": (
                    "the runs' targets are merged into one population. Their "
                    "target sets are disjoint (399 + 412 + 458 distinct ids), "
                    "checked by guard_disjoint_targets rather than assumed, so "
                    "the union is a population and not a double count — the "
                    "same pooling plot-venn performs over these three runs. "
                    "n_targets per method is the sum over the runs that carry "
                    "that method."
                ),
                "share_denominator": (
                    f"{n_pooled} pooled targets — every scored row, fallbacks "
                    "included. A share here is a micro-average over the fleet, "
                    "so it is target-weighted: as03 carries 36% of it. See "
                    "weighting for the macro-average beside it."
                ),
                "weighting": weighting_check(per_dataset, table, mode),
                "counts": {"pooled": pooled_counts, "per_dataset": counts},
            }
        else:
            table = per_dataset
            png = plot_cross_bands(
                points,
                table,
                target_dir / f"{stem}.{slug}.png",
                mode=mode,
                title=mode.title,
                subtitle=(
                    f"{' + '.join(datasets)} · {slug} · "
                    "rows are methods, columns are datasets"
                ),
                max_observed=max_observed,
                max_x_km=max_x_km,
                min_x_km=min_x_km,
            )
            manifest = {
                **common,
                "cross_layout": layout,
                "layout": (
                    "one panel per (method, dataset): methods down the rows in "
                    "PUBLISHED_METHODS order, datasets across the columns. x "
                    "and y are shared by every panel, including the top band's "
                    "bucket label, so any two panels are directly comparable."
                ),
                "pooling": (
                    "nothing is pooled. Each panel's counts, shares and medians "
                    "come from that run's own load_points/band_table, so a "
                    "panel here is identical to the same panel in the run's own "
                    "figure."
                ),
                "share_denominator": (
                    "each panel divides by its own run's n_targets — every "
                    "scored row, fallbacks included — so the correct band is "
                    "that run's top-1 accuracy from topn_accuracy.csv and the "
                    "bands sum to 1 - fallback_rate."
                ),
                "counts": counts,
            }
        built.append(
            {
                "layout": layout,
                "stem": stem,
                "png": png,
                "points": points,
                "table": table,
                "manifest": manifest,
            }
        )
    return built


# ---- CLI --------------------------------------------------------------------


def _run_mode(
    mode: YMode,
    *,
    run_id: list[str] | None,
    all_runs: bool,
    grid: str,
    resolution: list[int],
    sweep: bool,
    method: list[str] | None,
    max_x_km: float,
    min_x_km: float,
    outputs_root: Path,
    analysis_root: Path,
    allow_mixed_setups: bool = False,
    out_dir: Path | None = None,
    layout: list[str] | None = None,
) -> None:
    """Shared body of both commands — they differ only in their `YMode`."""
    if all_runs and run_id:
        raise typer.BadParameter("pass --run-id or --all-runs, not both")
    ids = list(run_id or [])
    if not ids and not all_runs:
        raise typer.BadParameter("pass at least one --run-id, or --all-runs")

    g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)

    # Two or more --run-id means one cross-dataset grid, matching `plot-venn`'s
    # convention. --all-runs stays per-run: which datasets belong in one figure
    # is a claim about comparability (§7.3 declines the operator/public
    # head-to-head), so it has to be named rather than discovered.
    if len(ids) > 1:
        runs = {rid: resolve_run(rid, outputs_root) for rid in ids}
        cross.guard_one_setup(runs, allow_mixed=allow_mixed_setups)
        layouts = list(layout) if layout else [POOLED]
        for res in resolutions:
            slug = grid_slug(g.name, res)
            for built in build_cross(
                runs,
                mode=mode,
                layouts=layouts,
                analysis_root=analysis_root,
                grid=g.name,
                resolution=res,
                methods=list(method) if method else None,
                max_x_km=max_x_km,
                min_x_km=min_x_km,
                out_dir=out_dir,
            ):
                png, table, stem = built["png"], built["table"], built["stem"]
                manifest = built["manifest"]
                built["points"].to_csv(
                    png.parent / f"{stem}_points.{slug}.csv", index=False
                )
                table.to_csv(png.parent / f"{stem}_bands.{slug}.csv", index=False)
                (png.parent / f"{stem}.{slug}.manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                correct = table.loc[table["level"] == mode.index_base]
                if built["layout"] == POOLED:
                    top = correct.set_index("method_label")["share"]
                    best = top.idxmax()
                    summary = (
                        f"{int(correct['n_targets'].max()):,} pooled targets · "
                        f"band-{mode.index_base} best {best} {top[best]:.3f}"
                    )
                else:
                    summary = "band-{} best per dataset: {}".format(
                        mode.index_base,
                        ", ".join(
                            f"{d} {g_.set_index('method_label')['share'].idxmax()} "
                            f"{g_['share'].max():.3f}"
                            for d, g_ in correct.groupby("dataset", sort=True)
                        ),
                    )
                typer.echo(
                    f"{'+'.join(manifest['datasets'])} {slug} {mode.key} "
                    f"[{built['layout']}] · "
                    f"{len(manifest['methods'])} methods x "
                    f"{len(manifest['datasets'])} datasets · "
                    f"{manifest['n_lines_drawn']} lines · {summary} -> {png}"
                )
        return

    if layout:
        raise typer.BadParameter(
            "--layout applies to the cross-dataset figures (two or more "
            "--run-id); a single run has one layout"
        )
    if out_dir is not None:
        raise typer.BadParameter(
            "--out-dir applies to the cross-dataset grid (two or more --run-id); "
            "a single run always writes back into its own target-cls-accuracy/"
        )
    runs_list = (
        discover_runs(outputs_root)
        if all_runs
        else [resolve_run(ids[0], outputs_root)]
    )
    for run in runs_list:
        for res in resolutions:
            png, points, table, manifest = build(
                run,
                mode=mode,
                analysis_root=analysis_root,
                grid=g.name,
                resolution=res,
                methods=list(method) if method else None,
                max_x_km=max_x_km,
                min_x_km=min_x_km,
            )
            points.to_csv(png.parent / f"{mode.stem}_points.csv", index=False)
            table.to_csv(png.parent / f"{mode.stem}_bands.csv", index=False)
            (png.parent / f"{mode.stem}_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            top = table.loc[table["level"] == mode.index_base].set_index(
                "method_label"
            )["share"]
            best = top.idxmax()
            typer.echo(
                f"{run.run_id} {grid_slug(g.name, res)} {mode.key} · "
                f"{manifest['n_lines_drawn']} lines · "
                f"band-{mode.index_base} share best {best} {top[best]:.3f} -> {png}"
            )


def register(app: typer.Typer) -> None:
    #: Two commands rather than one with a `--y` switch: each figure has its own
    #: name in `--help` and its own config sub-block, and the two answer
    #: different questions (SCHEMA.md §"Fields that are not accurate for the
    #: obvious reading"). They share `_run_mode`, so the axes cannot diverge.
    @app.command("plot-error-vs-cells")
    def plot_error_vs_cells_cmd(
        run_id: list[str] = typer.Option(
            None,
            "--run-id",
            help="Run to render (repeatable). One run writes back into that run's "
            "target-cls-accuracy/; two or more render one cross-dataset grid "
            "into _cross/error-vs-class/. Omit with --all-runs.",
        ),
        all_runs: bool = typer.Option(
            False,
            "--all-runs",
            help="Render every run under --outputs-root, each on its own "
            "(never pooled into the cross grid).",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        max_x_km: float = typer.Option(
            DEFAULT_X_MAX_KM, "--max-x-km", help="Upper bound of the log x axis (km)."
        ),
        min_x_km: float = typer.Option(
            X_MIN_KM, "--min-x-km", help="Lower bound of the log x axis (km)."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
        layout: list[str] = typer.Option(
            None,
            "--layout",
            help=f"Cross-dataset layout (repeatable): {POOLED!r} merges the "
            f"runs' targets into one population and draws the usual six-panel "
            f"grid over it; {COMPARE!r} keeps them separate, one panel per "
            f"(method, dataset). Default: {POOLED!r}. Ignored for a single run.",
        ),
        allow_mixed_setups: bool = typer.Option(
            False,
            "--allow-mixed-setups",
            help="Permit one cross grid over both run families. They swap the "
            "VP/target roles (SCHEMA.md §7), so the default refuses.",
        ),
        out_dir: Path = typer.Option(
            None, help="Override the cross-dataset output directory."
        ),
    ) -> None:
        """Coordinate error against cells away from the true class.

        One run writes error_vs_cells.{png,_points.csv,_bands.csv,
        _manifest.json}} into that run's
        target-cls-accuracy/<grid>-<resolution>/. Two or more write one
        grid of methods x datasets into
        _cross/error-vs-class/<dataset-set>/, grid-slugged.
        Needs `classify`.
        """
        _run_mode(
            CELLS,
            run_id=run_id,
            all_runs=all_runs,
            grid=grid,
            resolution=resolution,
            sweep=sweep,
            method=method,
            max_x_km=max_x_km,
            min_x_km=min_x_km,
            outputs_root=outputs_root,
            analysis_root=analysis_root,
            allow_mixed_setups=allow_mixed_setups,
            out_dir=out_dir,
            layout=list(layout) if layout else None,
        )

    @app.command("plot-error-vs-rank")
    def plot_error_vs_rank_cmd(
        run_id: list[str] = typer.Option(
            None,
            "--run-id",
            help="Run to render (repeatable). One run writes back into that run's "
            "target-cls-accuracy/; two or more render one cross-dataset grid "
            "into _cross/error-vs-class/. Omit with --all-runs.",
        ),
        all_runs: bool = typer.Option(
            False,
            "--all-runs",
            help="Render every run under --outputs-root, each on its own "
            "(never pooled into the cross grid).",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        max_x_km: float = typer.Option(
            DEFAULT_X_MAX_KM, "--max-x-km", help="Upper bound of the log x axis (km)."
        ),
        min_x_km: float = typer.Option(
            X_MIN_KM, "--min-x-km", help="Lower bound of the log x axis (km)."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
        layout: list[str] = typer.Option(
            None,
            "--layout",
            help=f"Cross-dataset layout (repeatable): {POOLED!r} merges the "
            f"runs' targets into one population and draws the usual six-panel "
            f"grid over it; {COMPARE!r} keeps them separate, one panel per "
            f"(method, dataset). Default: {POOLED!r}. Ignored for a single run.",
        ),
        allow_mixed_setups: bool = typer.Option(
            False,
            "--allow-mixed-setups",
            help="Permit one cross grid over both run families. They swap the "
            "VP/target roles (SCHEMA.md §7), so the default refuses.",
        ),
        out_dir: Path = typer.Option(
            None, help="Override the cross-dataset output directory."
        ),
    ) -> None:
        """Coordinate error against how many classes outrank the true one.

        `tg_seed_rank < N` is top-N, so each band's cumulative share is that
        top-N accuracy: level 0 is top-1, through level 2 is top-3.

        One run writes error_vs_rank.{png,_points.csv,_bands.csv,
        _manifest.json}} into that run's
        target-cls-accuracy/<grid>-<resolution>/. Two or more write one
        grid of methods x datasets into
        _cross/error-vs-class/<dataset-set>/, grid-slugged.
        Needs `classify`.
        """
        _run_mode(
            RANK,
            run_id=run_id,
            all_runs=all_runs,
            grid=grid,
            resolution=resolution,
            sweep=sweep,
            method=method,
            max_x_km=max_x_km,
            min_x_km=min_x_km,
            outputs_root=outputs_root,
            analysis_root=analysis_root,
            allow_mixed_setups=allow_mixed_setups,
            out_dir=out_dir,
            layout=list(layout) if layout else None,
        )
