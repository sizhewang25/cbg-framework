"""Error-distance CDF, one curve per method, log x.

The companion to the outcome bars. Those say *where* a prediction landed --
in the cell, one ring out, two, further, or never answered. They cannot say
*how far off* it was, and the two can disagree: a method that is rarely in the
right cell may still be consistently close, and a method that is often in the
right cell may have a catastrophic tail. The disagreement is the finding, so
the package needs both halves.

`classify` has computed the distance all along -- `error_km`, one value per
target -- and `accuracy.csv` already publishes two order statistics from it.
Nothing drew the distribution those two numbers summarise.

## This figure does not depend on the grid

`error_km` is prediction to the ground-truth coordinate. The answer space
defines the *classes*; it has no business inside a distance that has its own
ground truth. So the number is identical at every rung, and measurably so:
across healpix-128/64/32/16 on all three meshes, `error_km` and every count
and percentile derived from it are byte-identical.

That is why the artifacts carry **no `healpix-<n>` in their names** and land in
the rung-free parent of the rung directories. v3 could only aspire to this --
its docstring asks that re-quantizing "must leave this figure bit-identical",
which is a property to be checked. Here there is no seed-routed error column to
be tempted by (v3's `error_to_tg_seed_km` has no v4 counterpart, because `ring`
answers that question instead), so the independence is structural and the
filename can assert it. `TestGridFree` re-checks the premise against real runs.

## Unanswered rows are excluded, and that is not a new policy

A `FALLBACK` row still carries a coordinate -- the shortest-ping VP's -- so
filtering on NaN will not drop it, and pooling it would pull a variant's curve
toward the baseline exactly where the variant failed. `classify.solved_mask` is
the shared predicate, and `classify.summarize` already applies it before taking
`error_km_p50`/`p90`. Those rows are also exactly the outcome bars' grey "no
answer" segment (`n_failed = (~solved).sum()`), so the population behind a
curve here is that figure's *non-grey* stack, method for method.

The consequence has to be visible, because it is not uniform: on the three
meshes pooled, every method's curve rests on all 1,269 targets except
`vanilla_cbg`, whose 275 fallbacks leave it at 994. Hence the `plotted/total`
column in the percentile box.

That predicate also carries the case a hand-written filter gets wrong:
Shortest-Ping's rows are all `BASELINE`, never `SUCCESS`, so
`status == "SUCCESS"` would silently empty the baseline curve.

## Pooling concatenates rows; it cannot average percentiles

`--layout pooled` draws one curve per method over every selected run's solved
rows at once. A micro-pool: the rows are concatenated, so a dataset weighs by
its target count, and the pooled `n_solved` is the sum of the runs'.

Averaging the runs' published p50s instead is not an approximation, it is a
different quantity, and on this data it reverses the leader: true pooled p50
puts `million_scale_cbg` first at 96.1 km, while the mean of its three runs'
p50s reads 192.2 and hands first place to `octant_cbg_hull` at 126.6. It errs
in both directions, too -- `vanilla_cbg` is 198.6 pooled against 171.3
averaged -- so there is not even a sign to correct for.

A method enters only if **every** run scored it, and target ids are checked
disjoint, for the reason the pooled outcome bars give: one axis, one
denominator.

## The baseline is dark grey and dashed, in both layouts

Every other v4 figure paints Shortest-Ping in its own hue. Here it is
recessive, because the CDF's job is to show the variants *against* a reference
rather than to compare seven peers.

The grey is `_INK_2`, **not** `methods.OTHER_HUE`, and the difference is not
cosmetic. v3 claims a dash is enough to tell its baseline from the
unpublished-method bucket -- but v3's `_C_MUTED` and `_C_OTHER` are the *same
hex*, `#898781`, so the claim rests on the dash alone. That was survivable
while nothing occupied the bucket. It was not survivable here: `spotter_h3_cbg`
was scored on all three meshes and fell through `LABEL_HUES` to `OTHER_HUE`, so
porting v3's choice put a solid `#898781` curve on the same axis as a dashed
`#898781` one, tracking it closely for much of the range. A darker grey keeps
the baseline recessive while making the two separable by colour as well as by
dash.

That arm is now parked (`outputs/benchmark/_parked/`) and the bucket is empty
again, which changes nothing here: the choice holds whether or not anything
currently occupies it, and going back to `OTHER_HUE` would only reinstate a
collision the next unpublished arm would find.

v3 also had to drop the grey-dashed baseline entirely in its cross-run views,
where a dash meant "traffic-weighted". v4 has no weighted arm, so the
convention holds in both layouts and colour stays free to mean the method.

Command: `plot-error-cdf`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v4.modules import classify as C  # noqa: E402
from scripts.analysis.v4.modules import cross
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.methods import method_colors, method_label
from scripts.analysis.v4.modules.paths import (
    MissingArtifactError,
    RunPaths,
    grid_slug,
)

#: The column this figure is about. A constant so the "distance to the raw
#: target, never through the seed" decision is greppable.
ERROR_COLUMN = "error_km"

#: Only what the curve and the percentile box need, so a 13-column parquet is
#: read three columns wide.
READ_COLUMNS: tuple[str, ...] = ("target_id", "status", ERROR_COLUMN)

#: Reported per method in the box under the legend.
#:
#: v3's set with **75 swapped for 90**. 90 has to be here: with 50 it is one of
#: the two percentiles `accuracy.csv` publishes, so carrying both is what lets
#: a reader join this CSV to the accuracy table and check they agree. Five
#: columns is also the width v3 already measured as fitting the box.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 90, 95)

#: Reference verticals (km). Neutral ink, **not** green/orange/red: green is
#: the Octant family and red is Spotter in this palette, so coloured guides
#: would read as series.
THRESHOLDS_KM: tuple[int, ...] = (100, 500, 1000)

#: Fixed x range, both bounds, so per-run and pooled figures are read on one
#: axis rather than several auto-fitted ones.
#:
#: The floor is 0.1 km rather than 1 km because sub-kilometre errors are not
#: rare here, and the observed minimum across all three meshes is 0.106 km --
#: a 1 km floor would flatten the left tail of exactly the curves that earned
#: it. Nothing on this data is clamped; the manifest records the count that
#: *would* be, so a future run cannot start clipping silently.
X_MIN_KM = 0.1
DEFAULT_X_MAX_KM = 10_000.0

PER_RUN = "per-run"
POOLED = "pooled"
LAYOUTS: tuple[str, ...] = (PER_RUN, POOLED)

#: `layout -> (png, csv, manifest)`. No `{slug}`: the figure does not vary with
#: the rung, so the name does not pretend it might. The pooled triple takes a
#: `.pooled.` infix, matching `outcome_bars.pooled.*`, so the two layouts sort
#: together and share one prefix to glob.
NAMES: dict[str, tuple[str, str, str]] = {
    PER_RUN: ("error_cdf.png", "error_cdf.csv", "error_cdf.manifest.json"),
    POOLED: (
        "error_cdf.pooled.png",
        "error_cdf.pooled.csv",
        "error_cdf.pooled.manifest.json",
    ),
}

#: Which rung's parquets to read. Any of them would do -- that is the point --
#: so the finest is the default and `--nside` exists mainly so a caller can
#: demonstrate the output does not change.
SOURCE_NSIDE = H.NSIDE_LADDER[0]

#: Summed when runs are pooled. One list, so a count added to `load_errors`
#: cannot be silently left per-run.
COUNT_KEYS: tuple[str, ...] = (
    "n_targets", "n_solved", "n_failed", "n_no_distance",
)

#: The CDF's way out of a pooling refusal. `--layout compare` is the outcome
#: bars' answer and does not exist here.
REMEDY_COMMON = "--layout per-run to keep each dataset on its own figure."
REMEDY_DISJOINT = (
    "Use --layout per-run, which keeps each dataset on its own figure."
)

#: Printed where a percentile has no value, i.e. a method that answered
#: nothing. An em dash, not "nan" and not 0.
_NO_VALUE = "\u2014"

#: Neutral ink. Declared per figure module, as every v4 figure does. Note
#: `_MUTED` is byte-for-byte `methods.OTHER_HUE`, which is why the baseline
#: takes `_INK_2` instead -- see `_curve_style`.
_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

#: The row and distance policies, printed under every layout's axes.
FOOTNOTE = (
    "Unanswered rows excluded: a FALLBACK coordinate is the shortest-ping "
    "VP's, so its error is the baseline's wearing a variant's name.\nThose are "
    "the outcome bars' “no answer” segment — the n below is that "
    "figure's non-grey stack. Distance is prediction to the raw\ntarget; the "
    "grid never enters it, which is why this figure is the same at every rung."
)


# ---- loading ----------------------------------------------------------------


def scored_methods(cls_dir: Path) -> list[str]:
    """Method ids with a `*_cells.parquet` in `cls_dir`.

    Order is irrelevant and therefore not imposed: the curves are ranked by
    their own p50 further down. `euler.membership.available_methods` does the
    same glob but also applies a display order, and reaching for it would drag
    the Euler layout solver into a module that draws lines.
    """
    suffix = C.CELLS_PARQUET.format(method="")
    return sorted(
        p.name[: -len(suffix)] for p in Path(cls_dir).glob(f"*{suffix}")
    )


def _load_cells(
    run: RunPaths, method: str, nside: int, *, analysis_root: Path | None = None
) -> pd.DataFrame:
    """One run's per-target scored rows for one method, three columns wide."""
    path = run.cls_accuracy_dir(nside, root=analysis_root) / C.CELLS_PARQUET.format(
        method=method
    )
    if not path.exists():
        raise MissingArtifactError(
            f"{path} missing; run `classify --run-id {run.run_id}` first"
        )
    return pd.read_parquet(path, columns=list(READ_COLUMNS))


def load_errors(
    run: RunPaths,
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> dict[str, dict]:
    """Per method: the solved rows' errors, the row counts, and the target ids.

    `n_solved` here equals `accuracy.csv`'s by construction rather than by
    coincidence -- both come from `classify.solved_mask` on the same frame.

    The ids are the **whole** scored population, not the solved subset, because
    what the pooled layout guards against is one target reaching two runs'
    denominators, and `n_targets` is a denominator.
    """
    cls_dir = run.cls_accuracy_dir(nside, root=analysis_root)
    chosen = list(methods) if methods else scored_methods(cls_dir)
    if not chosen:
        raise MissingArtifactError(
            f"{cls_dir} holds no *_cells.parquet; run "
            f"`classify --run-id {run.run_id}` first"
        )
    out: dict[str, dict] = {}
    for method in chosen:
        df = _load_cells(run, method, nside, analysis_root=analysis_root)
        solved = C.solved_mask(df)
        values = df.loc[solved, ERROR_COLUMN].to_numpy(dtype=float)
        finite = np.isfinite(values)
        out[method] = {
            "errors": values[finite],
            "target_ids": set(df["target_id"]),
            "n_targets": int(len(df)),
            "n_solved": int(solved.sum()),
            "n_failed": int((~solved).sum()),
            # A row the method answered that still has no distance. Zero on
            # every run today; carried so it cannot start happening quietly.
            "n_no_distance": int((~finite).sum()),
        }
    return out


def load_per_run(
    runs: list[RunPaths],
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> dict[str, dict[str, dict]]:
    """`{run_id: load_errors(run)}`, read once.

    Split out because the pooled layout needs both the pooled population and
    each run's target count for its manifest, and reading every parquet twice
    to get them would be the only cost of keeping `pooled_errors` a one-liner.
    """
    return {
        run.run_id: load_errors(
            run, nside, methods=methods, analysis_root=analysis_root
        )
        for run in runs
    }


def pool(loaded: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Every run's rows for a method concatenated into one population.

    A micro-pool: nothing is averaged. The percentiles come out of
    `percentile_table` reading the concatenation, which is the only way to get
    them right -- see the module docstring for the 96 vs 192 km case.
    """
    common = cross.guard_common_methods(
        {rid: set(entries) for rid, entries in loaded.items()},
        remedy=REMEDY_COMMON,
    )
    cross.guard_disjoint_targets(
        {
            rid: set().union(*(entries[m]["target_ids"] for m in common))
            for rid, entries in loaded.items()
        },
        remedy=REMEDY_DISJOINT,
    )
    out: dict[str, dict] = {}
    for method in common:
        parts = [entries[method] for entries in loaded.values()]
        out[method] = {
            "errors": np.concatenate([p["errors"] for p in parts]),
            "target_ids": set().union(*(p["target_ids"] for p in parts)),
            **{k: int(sum(p[k] for p in parts)) for k in COUNT_KEYS},
        }
    return out


def pooled_errors(
    runs: list[RunPaths],
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> dict[str, dict]:
    """`pool` over `load_per_run` -- the whole pooled population in one call."""
    return pool(
        load_per_run(runs, nside, methods=methods, analysis_root=analysis_root)
    )


# ---- the table --------------------------------------------------------------

#: The sort cascade, best first. p50 is the headline and is *drawn* (the dashed
#: horizontal at y=0.5), so the legend reads top-to-bottom in the order the
#: curves cross that visible line. p90 breaks a p50 tie on the tail, then the
#: larger population wins, then the id -- so the order is total and stable.
_RANK_KEYS: tuple[str, ...] = ("error_km_p50", "error_km_p90", "n_solved", "method")
_RANK_ASC: tuple[bool, ...] = (True, True, False, True)


def percentile_table(loaded: dict[str, dict]) -> pd.DataFrame:
    """One row per method: the counts plus the reported percentiles, ranked.

    Interpolation is **pandas' default (linear) via `Series.quantile`**, and
    the rounding is 3dp, because that is exactly what `classify.summarize`
    does. The two files sit one directory apart and both call the column
    `error_km_p50`; they have to agree digit for digit, and the way to
    guarantee that is to run the same call rather than a compatible one.
    """
    rows: list[dict] = []
    for method, entry in loaded.items():
        values = pd.Series(entry["errors"], dtype=float)
        row = {
            "method": method,
            "method_label": method_label(method),
            "is_baseline": method == C.SHORTEST_PING,
            **{k: int(entry[k]) for k in COUNT_KEYS},
            "n_plotted": int(len(values)),
        }
        for p in PERCENTILES:
            row[f"error_km_p{p}"] = (
                round(float(values.quantile(p / 100)), 3) if len(values) else np.nan
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    # Rank on the rounded values the CSV prints, so the order always follows
    # from the numbers in front of the reader.
    return table.sort_values(
        list(_RANK_KEYS), ascending=list(_RANK_ASC)
    ).reset_index(drop=True)


def curve_order(table: pd.DataFrame) -> list[str]:
    """The ranked method ids -- legend order, and the percentile box's."""
    return list(table["method"])


# ---- drawing ----------------------------------------------------------------


def _cdf(values: np.ndarray, min_x_km: float = X_MIN_KM):
    """`(x, y)` for an empirical CDF, x clamped up to the log floor.

    The clamp is a rendering concession -- a log axis cannot show 0 -- and it
    is applied *here only*. `percentile_table` reads the unclamped values, so a
    reported percentile is never the floor in disguise.
    """
    xs = np.sort(np.maximum(values, min_x_km))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def _draw_guides(ax, min_x_km: float, max_x_km: float) -> None:
    """The 100/500/1,000 km verticals and the median line, under everything."""
    for km in THRESHOLDS_KM:
        if min_x_km < km < max_x_km:
            ax.axvline(km, color=_GRID, linestyle=":", linewidth=1.2, zorder=1)
            # Labels ride under the top spine, rotated, rather than along the
            # baseline: on a log axis 500 and 1,000 km sit a third of a decade
            # apart, so horizontal labels there collide with each other and
            # with whichever curve happens to be low at that x.
            ax.annotate(
                f"{km}",
                xy=(km, 0.995),
                xytext=(-3, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=_MUTED,
                ha="right",
                va="top",
                rotation=90,
                zorder=1,
            )
    ax.axhline(0.5, color=_GRID, linestyle="--", linewidth=1.2, zorder=1)


def _style_axes(ax, min_x_km: float, max_x_km: float) -> None:
    """Log x on the fixed range, 0-1 y, and the package's quiet spines."""
    from matplotlib.ticker import FuncFormatter

    ax.set_xscale("log")
    # `%g` rather than a ScalarFormatter, which renders the 0.1 km decade as a
    # bare "0" -- an axis claiming the curve starts at zero when it starts at
    # 100 m.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(min_x_km, max_x_km)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Error distance to the raw target (km)", fontsize=10.5, color=_INK_2)
    ax.set_ylabel("Fraction of targets answered", fontsize=10.5, color=_INK_2)
    ax.grid(True, which="both", color=_GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelsize=9)


def _curve_style(method: str, colors: dict[str, str]) -> dict:
    """Colour is the method; the baseline is recessive grey and dashed."""
    if method == C.SHORTEST_PING:
        # _INK_2, not _MUTED: _MUTED is byte-for-byte `methods.OTHER_HUE`, so
        # a baseline drawn in it shares a colour with every unpublished arm.
        return {"color": _INK_2, "linestyle": "--", "linewidth": 2.2, "zorder": 3}
    return {
        "color": colors[method],
        "linestyle": "-",
        "linewidth": 2.0,
        "zorder": 2,
    }


def _percentile_box(ax, table: pd.DataFrame, legend) -> None:
    """The monospace percentile table, hung under `legend`.

    Needs a drawn canvas, so the legend's extent is real. Row order is the
    legend's, which is what stops the two halves of the same figure from
    disagreeing about which method leads -- v3's box sorts by p50 while its
    legend does not.

    The `plotted/total` column is the only place a reader learns that a curve
    was drawn over a subset: pooled, `vanilla_cbg` reads 994/1269 while every
    other method reads 1269/1269.
    """
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    header = (
        f"{'':<14}{'plotted':>11}"
        + "".join(f"{'p' + str(p):>7}" for p in PERCENTILES)
    )
    lines = [header]
    for _, row in table.iterrows():
        counts = f"{int(row['n_plotted'])}/{int(row['n_targets'])}"
        cells = "".join(
            f"{row[f'error_km_p{p}']:>7.0f}"
            if np.isfinite(row[f"error_km_p{p}"])
            else f"{'—':>7}"
            for p in PERCENTILES
        )
        lines.append(f"{row['method_label'][:14]:<14}{counts:>11}{cells}")
    ax.text(
        0.02,
        box.ymin - 0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color=_INK_2,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor=_SURFACE, edgecolor=_GRID, alpha=0.95),
    )


def plot_cdf(
    loaded: dict[str, dict],
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    min_x_km: float = X_MIN_KM,
    max_x_km: float = DEFAULT_X_MAX_KM,
    figsize: tuple[float, float] = (8.6, 6.6),
    dpi: int = 200,
) -> Path:
    """One panel, one curve per method, log x. Both layouts draw through here.

    v3 needed three plot functions because its cross-run views spent line style
    on the dataset type. v4 has no traffic-weighted arm, so per-run and pooled
    differ only in which rows went in and what the subtitle says.

    Honest limit: CDF curves **cross**, so no single ranking is true across the
    whole axis -- a method can lead at p50 and trail badly at p90. The legend
    is ordered by p50 and the percentile box sits directly under it in the same
    order, showing p5 through p95, so the reader can see both where the order
    comes from and where it stops holding.
    """
    from matplotlib.lines import Line2D

    order = curve_order(table)
    colors = method_colors(order)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    _draw_guides(ax, min_x_km, max_x_km)

    # Variants first, baseline last, so the grey dashed reference reads on top
    # of them. Draw order is therefore not legend order, which is why the
    # legend is built from explicit handles below.
    variants = [m for m in order if m != C.SHORTEST_PING]
    for method in variants + [m for m in order if m == C.SHORTEST_PING]:
        values = loaded[method]["errors"]
        if len(values) == 0:
            continue
        xs, ys = _cdf(values, min_x_km)
        ax.plot(xs, ys, alpha=0.95, gid=method, **_curve_style(method, colors))

    _style_axes(ax, min_x_km, max_x_km)

    handles = [
        Line2D(
            [],
            [],
            label=(
                f"{method_label(m)} (baseline)"
                if m == C.SHORTEST_PING
                else method_label(m)
            ),
            **{k: v for k, v in _curve_style(m, colors).items() if k != "zorder"},
        )
        for m in order
    ]
    legend = ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(_INK_2)

    ax.set_title(title, fontsize=13, fontweight="bold", color=_INK, pad=16)
    ax.annotate(
        subtitle,
        xy=(0.5, 1.005),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=_INK_2,
    )
    ax.annotate(
        FOOTNOTE,
        xy=(0.5, -0.135),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=7.5,
        color=_MUTED,
    )

    fig.canvas.draw()
    _percentile_box(ax, table, legend)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


# ---- artifacts --------------------------------------------------------------

#: The CSV twin's columns, in order. `run_id` and `dataset` lead so a per-run
#: and a pooled file concatenate into one frame, the way the outcome bars'
#: two layouts do.
CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "method", "method_label", "is_baseline",
    *COUNT_KEYS, "n_plotted",
    *[f"error_km_p{p}" for p in PERCENTILES],
)


def _clamped(loaded: dict[str, dict], min_x_km: float) -> dict[str, int]:
    """Per method, how many drawn points the log floor moved. Empty is good."""
    return {
        m: int((e["errors"] < min_x_km).sum())
        for m, e in loaded.items()
        if (e["errors"] < min_x_km).any()
    }


def _manifest(
    layout: str,
    table: pd.DataFrame,
    loaded: dict[str, dict],
    *,
    run_ids: list[str],
    nside: int,
    png_name: str,
    csv_name: str,
    min_x_km: float,
    max_x_km: float,
    per_run: dict[str, int] | None = None,
) -> str:
    body: dict = {
        "figure": png_name,
        "csv": csv_name,
        "layout": layout,
        "runs": list(run_ids),
        "dataset": cross.dataset_slug(run_ids),
        "arm": cross.arm(run_ids),
        "methods": curve_order(table),
        "baseline": C.SHORTEST_PING,
        "error_column": ERROR_COLUMN,
        "percentiles": list(PERCENTILES),
        "source_rung": {
            "nside": int(nside),
            "slug": grid_slug(nside),
            "note": (
                "which rung's *_cells.parquet was read. It does not affect the "
                "output: error_km is prediction-to-target and is identical at "
                "every rung, which is why neither the figure nor its name "
                "carries a grid."
            ),
        },
        "grid_free": (
            "verified across healpix-128/64/32/16 on all three meshes: "
            "error_km, and every count and percentile derived from it, are "
            "byte-identical. The artifacts therefore live in the rung-free "
            "parent of the healpix-<n>/ directories rather than four times over."
        ),
        "row_policy": (
            "solved rows only, via classify.solved_mask -- FALLBACK and ERROR "
            "excluded, and Shortest-Ping's all-BASELINE rows counted as solved "
            "so the baseline curve is drawn. n_plotted therefore equals "
            "accuracy.csv's n_solved, and n_failed is the outcome bars' "
            "'no answer' segment."
        ),
        "distance_policy": (
            "error_km: prediction to the ground-truth coordinate. The answer "
            "space defines the classes and stays out of the distance; 'how far "
            "from the class centre' is what ring answers instead."
        ),
        "percentile_policy": (
            "pandas Series.quantile (linear) rounded to 3dp -- the same call "
            "classify.summarize makes, so error_km_p50/p90 here and in "
            "accuracy.csv agree digit for digit."
        ),
        "curve_order": (
            "error_km_p50 ascending, then p90, then n_solved descending, then "
            "method id -- ranked on the rounded values the CSV prints. CDF "
            "curves cross, so this is the order they reach the median line, "
            "not a ranking that holds across the whole axis; the percentile "
            "box shows p5..p95 in the same order so the limit is visible."
        ),
        "baseline_encoding": (
            "dark grey (_INK_2) and dashed rather than its own hue: the "
            "figure shows variants against a reference. Deliberately NOT "
            "methods.OTHER_HUE, which is the same hex as this module's "
            "_MUTED -- any unpublished arm takes that bucket, and v3's "
            "claim that a dash alone separates them holds only while the "
            "bucket is empty. It is empty today (spotter_h3_cbg occupied it "
            "until it was parked), which is not something the encoding "
            "should depend on. No traffic-weighted arm claims the dash "
            "either, so the convention holds in both layouts."
        ),
        "x_axis": {
            "scale": "log",
            "min_km": min_x_km,
            "max_km": max_x_km,
            "fixed_note": (
                "both bounds fixed rather than auto-fitted, so the per-run and "
                "pooled figures are read on one axis"
            ),
            "n_clamped_to_floor": _clamped(loaded, min_x_km),
            "clamp_note": (
                f"a log axis cannot render 0, so the drawn curve clamps errors "
                f"below {min_x_km} km up to the floor. Empty on as01/02/03, "
                f"whose observed minimum is 0.106 km. Applied to the curve "
                f"only -- the CSV reads unclamped values, so no reported "
                f"percentile can be the floor in disguise."
            ),
        },
        "counts": {
            m: {k: int(e[k]) for k in COUNT_KEYS} for m, e in loaded.items()
        },
    }
    if layout == POOLED:
        total = int(table["n_targets"].max()) if len(table) else 0
        body["pooling"] = {
            "rule": (
                "micro-pool: each run's solved rows are concatenated into one "
                "population, so a dataset weighs by its target count and the "
                "pooled n_solved is the sum of the runs'."
            ),
            "runs": per_run or {},
            "n_targets": total,
            "largest_share": (
                round(max(per_run.values()) / total, 4)
                if per_run and total
                else None
            ),
            "percentiles": (
                "TRUE pooled quantiles over the concatenated per-target "
                "errors, never a mean of the runs' published percentiles. On "
                "these three meshes averaging reverses the leader: pooled p50 "
                "puts million_scale_cbg first at 96.1 km, the mean of its "
                "runs' p50s reads 192.2 and hands first place to "
                "octant_cbg_hull at 126.6."
            ),
            "coverage": (
                "strict -- a method absent from any run is refused rather than "
                "pooled over the runs that carry it, so every curve on the "
                "axis rests on the same datasets. Overlapping target ids are "
                "refused for the same reason."
            ),
            "reading_caveat": (
                "this is a method's accuracy on THIS target mix, not in "
                "general. Use --layout per-run to see per-dataset divergence."
            ),
        }
    return json.dumps(body, indent=2) + "\n"


def _write(
    loaded: dict[str, dict],
    out_dir: Path,
    layout: str,
    *,
    run_ids: list[str],
    nside: int,
    subtitle: str,
    min_x_km: float,
    max_x_km: float,
    per_run: dict[str, int] | None = None,
) -> Path:
    """Table, CSV twin, PNG and manifest for one layout. Returns the PNG."""
    png_name, csv_name, manifest_name = NAMES[layout]
    table = percentile_table(loaded)
    table.insert(0, "run_id", "+".join(sorted(run_ids)))
    table.insert(1, "dataset", cross.dataset_slug(run_ids))
    table[[c for c in CSV_COLUMNS if c in table.columns]].to_csv(
        out_dir / csv_name, index=False
    )
    png = plot_cdf(
        loaded,
        table,
        out_dir / png_name,
        title="Error distance to the raw target",
        subtitle=subtitle,
        min_x_km=min_x_km,
        max_x_km=max_x_km,
    )
    (out_dir / manifest_name).write_text(
        _manifest(
            layout,
            table,
            loaded,
            run_ids=run_ids,
            nside=nside,
            png_name=png_name,
            csv_name=csv_name,
            min_x_km=min_x_km,
            max_x_km=max_x_km,
            per_run=per_run,
        )
    )
    return png


def _n_targets(loaded: dict[str, dict]) -> int:
    """The population behind the panel. One number, because every method was
    scored on the same roster -- `classify` writes one row per target per
    method."""
    return max((e["n_targets"] for e in loaded.values()), default=0)


def build_for_run(
    run: RunPaths,
    *,
    nside: int = SOURCE_NSIDE,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    min_x_km: float = X_MIN_KM,
    max_x_km: float = DEFAULT_X_MAX_KM,
) -> Path:
    """One run's CDF, written beside its `healpix-<n>/` rungs."""
    loaded = load_errors(run, nside, methods=methods, analysis_root=analysis_root)
    return _write(
        loaded,
        run.cls_accuracy_root(root=analysis_root),
        PER_RUN,
        run_ids=[run.run_id],
        nside=nside,
        subtitle=(
            f"{run.run_id} · n={_n_targets(loaded):,} targets "
            f"· solved rows only"
        ),
        min_x_km=min_x_km,
        max_x_km=max_x_km,
    )


def build_for_runs(
    runs: list[RunPaths],
    *,
    layouts: tuple[str, ...] = (PER_RUN,),
    nside: int = SOURCE_NSIDE,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    min_x_km: float = X_MIN_KM,
    max_x_km: float = DEFAULT_X_MAX_KM,
) -> list[Path]:
    """Render the requested layouts; returns the PNG paths, layout-major.

    `per-run` writes one figure per run into that run's own tree; `pooled`
    writes a single figure into the cross-dataset directory, beside the
    outcome bars and the Euler diagrams for the same dataset set.
    """
    ordered = tuple(dict.fromkeys(layouts)) or (PER_RUN,)
    unknown = [x for x in ordered if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layout {unknown}; pick from {list(LAYOUTS)}")
    nside = H.validate_nside(nside)
    run_ids = [r.run_id for r in runs]

    out: list[Path] = []
    for layout in ordered:
        if layout == PER_RUN:
            out.extend(
                build_for_run(
                    run,
                    nside=nside,
                    methods=methods,
                    analysis_root=analysis_root,
                    min_x_km=min_x_km,
                    max_x_km=max_x_km,
                )
                for run in runs
            )
            continue
        by_run = load_per_run(
            runs, nside, methods=methods, analysis_root=analysis_root
        )
        loaded = pool(by_run)
        per_run = {rid: _n_targets(e) for rid, e in by_run.items()}
        label = cross.dataset_slug(run_ids).upper()
        out.append(
            _write(
                loaded,
                cross.cross_dir(run_ids, analysis_root=analysis_root),
                POOLED,
                run_ids=run_ids,
                nside=nside,
                subtitle=(
                    f"{label} pooled · n={_n_targets(loaded):,} targets "
                    f"· solved rows only"
                ),
                min_x_km=min_x_km,
                max_x_km=max_x_km,
                per_run=per_run,
            )
        )
    return out
