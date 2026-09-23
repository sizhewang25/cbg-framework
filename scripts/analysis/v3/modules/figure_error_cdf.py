"""Error-distance CDF, one curve per method, log x.

The companion to the classification tables: §2.4(a) says accuracy and error
distance can disagree and that the disagreement is itself a finding, so the
paper needs both. This is the error half.

## The distance is to the raw target, never through the seed

`classify` writes two error columns per row and this figure reads
`error_to_target_km` — prediction to the ground-truth coordinate. The
alternative, `error_to_tg_seed_km`, routes through the class seed, and because
seeds sit at cell centres that would add the row's `cell_offset_km` (p50 16-20
km at `h3-4`) to *every* answer including the correct ones. The grid exists to
define the classes; it has no business inside a distance that has its own
ground truth. One useful consequence: re-quantizing the answer space changes
accuracy and must leave this figure bit-identical.

## Fallbacks are excluded, and the baseline is not a fallback

A `FALLBACK` row's coordinate is the shortest-ping VP's, so its error is the
baseline's error wearing a variant's name. Pooling it would pull every
variant's curve toward the baseline exactly where the variant failed — the
comparison RQ2 rests on. `io.solved_mask` is the shared predicate, so this
figure's per-method `n` equals `topn_accuracy.csv`'s `n_solved` by
construction.

That predicate also carries the case a hand-written filter gets wrong:
Shortest-Ping's rows are all `BASELINE`, never `SUCCESS`, so
`status == "SUCCESS"` would drop the baseline curve entirely.

## The baseline is grey and dashed

Every other figure in this paper paints Shortest-Ping with its own variant hue.
Here it is grey and dashed instead, because the CDF's job is to show the
variants *against* a reference rather than to compare seven peers, and a
recessive baseline is the convention the v2 plotter established. The dash
carries the distinction on its own, so the grey cannot be confused with the
`_C_OTHER` bucket a non-published method would land in.

That holds for the `per-run` layout only. The cross-run layouts spend line
style on the **dataset type** — mesh solid, traffic-weighted dashed — so a grey
dashed baseline there would read as "weighted Shortest-Ping". The baseline takes
its own variant hue instead, solid or dashed by kind like every other method,
and is still drawn last.

## Pooled and compare: the outcome bars' two cross-run views

`--layout pooled` draws each dataset type's curve per method over every
selected run's solved rows at once; `--layout compare` draws one panel per
dataset with its mesh and weighted curves overlaid. Rows come from
`headline_table.row_plan`, so each curve is the population of a headline-table
row by construction. A pooled curve is a micro-pool — the runs' rows
concatenated, so a dataset weighs by its target count — and a method enters it
only if every run of that kind scored it. A kind with no run (the weighted
campaign, today) draws no curve: `PROVISIONAL_WEIGHTED` holds rates, and a rate
has no distance distribution.

Command: `plot-error-cdf`. `per-run` writes to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`;
`pooled` / `compare` write to `outputs/analysis/v3/_cross/accuracy-table/
<dataset-set>/`, beside `plot-outcome-bars`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import cross, io
from scripts.analysis.v3.modules import headline_table as H
from scripts.analysis.v3.modules.classify import (
    SHORTEST_PING,
    denominator_mismatch,
)
from scripts.analysis.v3.modules.confusion import load_scored
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.labels import (
    PREFERRED_ORDER,
    PUBLISHED_METHODS,
    short_label,
)
from scripts.analysis.v3.modules.diagram.common.membership import (
    available_methods,
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
    MissingArtifactError,
    RunPaths,
    discover_runs,
    grid_slug,
    resolve_run,
)

FIGURE_PNG = "error_cdf.png"
CURVE_CSV = "error_cdf_percentiles.csv"
MANIFEST_JSON = "error_cdf_manifest.json"

#: The column this figure is about. Named as a constant so the "not through the
#: seed" decision is greppable rather than a string literal in a plot call.
ERROR_COLUMN = "error_to_target_km"

#: Reported per method beside each curve.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)

#: Reference verticals (km). Neutral ink, **not** the v2 plotter's
#: green/orange/red — green is the Octant family and red is Spotter in this
#: paper's palette, so coloured guides would read as series.
THRESHOLDS_KM: tuple[int, ...] = (100, 500, 1000)

#: Fixed x range, both bounds, so the three operator runs' curves are read on
#: one axis rather than three auto-fitted ones.
#:
#: The floor is 0.1 km rather than the v2 plotter's 1 km because sub-kilometre
#: errors are **not** rare here — 8 to 20 rows per method across as01/02/03,
#: up to 4.4% of a run's targets, with an observed minimum of 0.135 km. A 1 km
#: floor flattens the left tail of exactly the curves that earned it and pins
#: their p5 to the clamp. 0.1 km sits below every observed value on this data,
#: so nothing is clamped and the figure reports what was measured; the manifest
#: records the count that *would* be clamped so a future run cannot start
#: clipping silently.
X_MIN_KM = 0.1
DEFAULT_X_MAX_KM = 10_000.0

PER_RUN = "per-run"
POOLED = "pooled"
COMPARE = "compare"
LAYOUTS: tuple[str, ...] = (PER_RUN, POOLED, COMPARE)

#: Line style is the dataset type in the cross-run layouts; colour stays the
#: method. That is why those layouts drop the per-run figure's grey dashed
#: baseline: a dash there would read as "traffic-weighted Shortest-Ping".
KIND_LINESTYLE = {H.MESH: "-", H.WEIGHTED: "--"}

#: The percentile box's kind column. Short, because the box is monospace and
#: sits inside the axes.
KIND_SHORT = {H.MESH: "mesh", H.WEIGHTED: "wtd"}


def load_errors(
    run: RunPaths,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Per method, the solved rows' error-to-target distances plus row counts.

    Returns `(errors, counts)`. `counts[method]` carries `n_total`, `n_solved`
    and `n_fallback`, so the figure can print the denominator beside a curve
    that was drawn over a subset of the targets.
    """
    errors, counts, _ = _load_solved(
        run,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    return errors, counts


def _load_solved(
    run: RunPaths,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict], set]:
    """`load_errors` plus the run's `target_id`s, every row and every method.

    The ids are the whole scored population rather than the solved subset,
    because what the pooled layouts guard against is one target reaching two
    runs' denominators, and `n_total` is a denominator.
    """
    cls_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    # Default to the methods this run actually scored, not to the full
    # published set. An explicit --method still raises on a missing parquet --
    # that was asked for by name -- but the default must degrade, because a run
    # that benchmarked a subset of combos is normal (a materialization test, a
    # partial sweep) and every other consumer already tolerates it via
    # `available_methods` (breakdown.py, pni_sping.py, venn.py). Order follows
    # PUBLISHED_METHODS, matching this module's other intersection.
    chosen = (
        list(methods)
        if methods
        else [m for m in PUBLISHED_METHODS if m in set(available_methods(cls_dir))]
    )
    if not chosen:
        raise MissingArtifactError(
            f"{cls_dir} holds no *_seed_distances.parquet; run `classify` first"
        )
    errors: dict[str, np.ndarray] = {}
    counts: dict[str, dict] = {}
    target_ids: set = set()
    for method in chosen:
        df = load_scored(cls_dir, method)
        target_ids.update(df["target_id"])
        solved = io.solved_mask(df)
        values = df.loc[solved, ERROR_COLUMN].to_numpy(dtype=float)
        errors[method] = values[np.isfinite(values)]
        counts[method] = {
            "n_total": int(len(df)),
            "n_solved": int(solved.sum()),
            "n_fallback": int((df["status"] == "FALLBACK").sum()),
        }
    # Each curve is normalized by its own n, so two populations on one axis
    # read as one distribution. Stale artifacts only — `classify` no longer
    # writes them — hence a warning rather than a refusal to draw.
    problem = denominator_mismatch({m: c["n_total"] for m, c in counts.items()})
    if problem:
        warnings.warn(f"{cls_dir}: {problem}", stacklevel=3)
    return errors, counts, target_ids


def method_order(methods) -> list[str]:
    """Published order first, then anything else alphabetically.

    The baseline is drawn *last* despite sorting first, so its grey dashed line
    sits on top of the variants rather than under them — see `plot_error_cdf`.
    """
    known = [m for m in PREFERRED_ORDER if m in set(methods)]
    return known + sorted(set(methods) - set(known))


def percentile_table(
    errors: dict[str, np.ndarray], counts: dict[str, dict]
) -> pd.DataFrame:
    """One row per method: counts plus the reported percentiles.

    Interpolation is numpy's **default (linear)**, matching
    `classify.topn_summary`. That is a compatibility constraint rather than a
    preference: both files sit in one directory and both call the column
    `error_km_p50`, and `topn_accuracy.csv` is what the paper's accuracy table
    reads. The v2 plotter used `method="nearest"` so a quoted percentile named
    a real target and lined up with the MTL world-map viewer's bookmarks —
    a good reason there, and worth 0.7 to 1.3 km of disagreement with the
    paper's own error column here, which is not a trade worth making.
    """
    rows: list[dict] = []
    for method in method_order(errors):
        values = errors[method]
        row = {
            "method": method,
            "method_label": short_label(method),
            "is_baseline": method == SHORTEST_PING,
            **counts.get(method, {}),
            "n_plotted": int(len(values)),
        }
        for p in PERCENTILES:
            row[f"error_km_p{p}"] = (
                round(float(np.percentile(values, p)), 3)
                if len(values)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cdf(
    values: np.ndarray, min_x_km: float = X_MIN_KM
) -> tuple[np.ndarray, np.ndarray]:
    """`(x, y)` for an empirical CDF, x clamped up to the log floor.

    The clamp is a rendering concession — a log axis cannot show 0 — and it is
    applied *here only*. `percentile_table` reads the unclamped values, so a
    reported percentile is never the floor in disguise.
    """
    xs = np.sort(np.maximum(values, min_x_km))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def plot_error_cdf(
    errors: dict[str, np.ndarray],
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
    figsize: tuple[float, float] = (8.6, 6.4),
) -> Path:
    """One panel, one curve per method, log x.

    Colour is the variant, except the baseline which is grey and dashed. Six
    curves share one coordinate space, which is the case where the palette's
    all-pairs CVD margin only *warns* — so identity is carried twice over: the
    legend names every curve and the percentile table beside it is the table
    view the contrast rule asks for.
    """
    order = method_order(errors)
    colors = method_colors(order)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_SURFACE)
    _draw_guides(ax, min_x_km, max_x_km)

    # --- curves; baseline last so it reads on top -------------------------
    variants = [m for m in order if m != SHORTEST_PING]
    for method in variants + ([SHORTEST_PING] if SHORTEST_PING in order else []):
        values = errors[method]
        if len(values) == 0:
            continue
        xs, ys = _cdf(values, min_x_km)
        is_baseline = method == SHORTEST_PING
        ax.plot(
            xs,
            ys,
            color=_C_MUTED if is_baseline else colors[method],
            linestyle="--" if is_baseline else "-",
            linewidth=2.2 if is_baseline else 2.0,
            alpha=0.95,
            label=f"{short_label(method)} (baseline)"
            if is_baseline
            else short_label(method),
            zorder=3 if is_baseline else 2,
        )

    _style_axes(ax, min_x_km, max_x_km)

    legend = ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(_C_INK_2)

    ax.set_title(title, fontsize=13, fontweight="bold", color=_C_INK, pad=16)
    ax.annotate(
        subtitle,
        xy=(0.5, 1.005),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=_C_INK_2,
    )
    _footnote(ax, FOOTNOTE)

    # --- percentile table, below the legend -------------------------------
    fig.canvas.draw()
    _percentile_box(ax, table, legend)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


#: The row and distance policies, printed under every layout's axes.
FOOTNOTE = (
    "Fallbacks excluded (§7.2): their coordinate is the shortest-ping VP's, so "
    "their error is the baseline's.\nDistance is to the raw target, not through "
    "the class seed, so the grid's quantization stays out of it."
)


def _footnote(ax, text: str, *, y: float = -0.13) -> None:
    ax.annotate(
        text,
        xy=(0.5, y),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color=_C_MUTED,
    )


def _percentile_box(ax, table: pd.DataFrame, legend, *, by_kind: bool = False) -> None:
    """The monospace percentile table, hung under `legend`.

    Needs a drawn canvas, so the legend's extent is real. `by_kind` adds a
    dataset-type column and orders the rows mesh-first, each kind by p50, so a
    method's mesh and weighted rows can be read against the dash that tells
    their curves apart.
    """
    leg_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    kind_cell = (lambda row: f"{KIND_SHORT[row['kind']]:<6}") if by_kind else (lambda row: "")
    header = (
        f"{'':<14}"
        + (f"{'':<6}" if by_kind else "")
        + f"{'solved':>8}"
        + "".join(f"{'p' + str(p):>7}" for p in PERCENTILES)
    )
    rows = (
        table.assign(_k=table["kind"].map({k: i for i, k in enumerate(H.KINDS)}))
        .sort_values(["_k", "error_km_p50"])
        if by_kind
        else table.sort_values("error_km_p50")
    )
    lines = [header]
    for _, row in rows.iterrows():
        counts = f"{int(row['n_plotted'])}/{int(row['n_total'])}"
        cells = "".join(
            f"{row[f'error_km_p{p}']:>7.0f}"
            if np.isfinite(row[f"error_km_p{p}"])
            else f"{'—':>7}"
            for p in PERCENTILES
        )
        lines.append(
            f"{short_label(row['method'])[:14]:<14}{kind_cell(row)}{counts:>8}{cells}"
        )
    ax.text(
        0.02,
        leg_bbox.ymin - 0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color=_C_INK_2,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor=_SURFACE, edgecolor=_C_GRID, alpha=0.95),
    )


def _draw_guides(ax, min_x_km: float, max_x_km: float) -> None:
    """The 100/500/1,000 km verticals and the median line, under everything."""
    for km in THRESHOLDS_KM:
        if min_x_km < km < max_x_km:
            ax.axvline(km, color=_C_GRID, linestyle=":", linewidth=1.2, zorder=1)
            # Labels ride just under the top spine, rotated, rather than along
            # the baseline: on a log axis 500 and 1,000 km sit a third of a
            # decade apart, so horizontal labels there overlap each other and
            # whichever curve happens to be low at that x.
            ax.annotate(
                f"{km}",
                xy=(km, 0.995),
                xytext=(-3, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=_C_MUTED,
                ha="right",
                va="top",
                rotation=90,
                zorder=1,
            )
    ax.axhline(0.5, color=_C_GRID, linestyle="--", linewidth=1.2, zorder=1)


def _style_axes(ax, min_x_km: float, max_x_km: float, *, labels: bool = True) -> None:
    """Log x on the fixed range, 0-1 y, and the paper's quiet spines.

    `labels` is false on a comparison panel that is not in the left column or
    bottom row, where the shared axis already carries the name.
    """
    from matplotlib.ticker import FuncFormatter

    ax.set_xscale("log")
    # `%g` rather than `ScalarFormatter(set_scientific(False))`, which renders the
    # 0.1 km decade as a bare "0" — an axis that claims the curve starts at zero
    # when it starts at 100 m.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(min_x_km, max_x_km)
    ax.set_ylim(0, 1)
    if labels:
        ax.set_xlabel(
            "Error distance to the raw target (km)", fontsize=10.5, color=_C_INK_2
        )
        ax.set_ylabel("Fraction of solved targets", fontsize=10.5, color=_C_INK_2)
    ax.grid(True, which="both", color=_C_GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    ax.tick_params(colors=_C_MUTED, labelsize=9)


def build(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> tuple[Path, pd.DataFrame, dict]:
    """Render the figure and return `(png_path, percentile_table, manifest)`."""
    errors, counts = load_errors(
        run,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    if not errors:
        raise ValueError(
            f"no methods found for {run.run_id} at {grid_slug(grid, resolution)}; "
            "run `classify` first"
        )
    table = percentile_table(errors, counts)
    out_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    png = plot_error_cdf(
        errors,
        table,
        out_dir / FIGURE_PNG,
        title="Error distance to the raw target",
        subtitle=f"{run.run_id} · {grid_slug(grid, resolution)} · solved rows only",
        max_x_km=max_x_km,
        min_x_km=min_x_km,
    )
    clamped = {
        m: int((v < min_x_km).sum())
        for m, v in errors.items()
        if (v < min_x_km).any()
    }
    manifest = {
        "run_id": run.run_id,
        "grid": {"scheme": grid, "resolution": resolution},
        "methods": method_order(errors),
        "baseline": SHORTEST_PING,
        "error_column": ERROR_COLUMN,
        "percentiles": list(PERCENTILES),
        "x_axis": {
            "scale": "log",
            "min_km": min_x_km,
            "max_km": max_x_km,
            "fixed_note": (
                "both bounds are fixed rather than auto-fitted, so the operator "
                "runs' curves are read on one axis"
            ),
            "n_clamped_to_floor": clamped,
            "clamp_note": (
                f"a log axis cannot render 0, so the curve clamps errors below "
                f"{min_x_km} km up to the floor. n_clamped_to_floor is empty on "
                "as01/02/03, whose observed minimum is 0.135 km. The clamp is "
                "applied to the drawn curve only — error_cdf_percentiles.csv "
                "reads the unclamped values, so no reported percentile can be "
                "the floor in disguise."
            ),
        },
        "row_policy": (
            "solved rows only, via io.solved_mask — FALLBACK excluded (§7.2), and "
            "Shortest-Ping's all-BASELINE rows counted as solved so the baseline "
            "curve is drawn. Per-method n therefore equals topn_accuracy.csv's "
            "n_solved."
        ),
        "distance_policy": (
            "error_to_target_km: prediction to the ground-truth coordinate. NOT "
            "error_to_tg_seed_km, which routes through the cell-centre seed and "
            "would add cell_offset_km (p50 16-20 km at h3-4) to every answer, "
            "correct ones included."
        ),
        "baseline_encoding": (
            "grey dashed rather than its variant hue: the figure shows variants "
            "against a reference, and the dash keeps the grey distinct from the "
            "_C_OTHER bucket a non-published method would use"
        ),
        "counts": counts,
    }
    return png, table, manifest


# ---- cross-run layouts: pooled and compare ----------------------------------

#: Summed across runs when a kind is pooled. Kept as the one list both the
#: pooling and its test read, so a count added to `_load_solved` cannot be
#: silently left per-run.
COUNT_KEYS: tuple[str, ...] = ("n_total", "n_solved", "n_fallback")


def pool_runs(
    runs: list[RunPaths],
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> dict:
    """One dataset type's runs concatenated into one curve per method.

    A **micro-pool**: each run's solved rows go into one array, so a dataset
    weighs by its target count and the pooled `n_solved` is the sum of the
    runs' `topn_accuracy.csv` `n_solved` — the same pooling `headline_table`
    applies to the counts the outcome bars draw.

    A method enters only if **every** run scored it. Concatenating a method
    that ran on two of three datasets would draw it over a different
    population from its neighbours on the same axis; it is dropped instead and
    named in `methods_absent_in_some_runs`.

    Targets are checked disjoint across the runs, because one shared id would
    sit in the pooled denominator twice. Only *within* a kind: a
    traffic-weighted run is its mesh twin filtered, so its targets overlap the
    mesh ones by design.
    """
    loaded = {
        run.run_id: _load_solved(
            run,
            analysis_root=analysis_root,
            grid=grid,
            resolution=resolution,
            methods=methods,
        )
        for run in runs
    }
    if len(loaded) > 1:
        cross.guard_disjoint_targets(
            {run_id: ids for run_id, (_, _, ids) in loaded.items()},
            remedy="Use --layout compare, which keeps each dataset on its own panel.",
        )
    method_sets = [set(errors) for errors, _, _ in loaded.values()]
    common = set.intersection(*method_sets) if method_sets else set()
    union = set.union(*method_sets) if method_sets else set()
    errors = {
        m: np.concatenate([e[m] for e, _, _ in loaded.values()])
        for m in method_order(common)
    }
    counts = {
        m: {key: sum(c[m][key] for _, c, _ in loaded.values()) for key in COUNT_KEYS}
        for m in errors
    }
    return {
        "errors": errors,
        "counts": counts,
        "methods_absent_in_some_runs": method_order(union - common),
        "per_run_counts": {run_id: c for run_id, (_, c, _) in loaded.items()},
    }


def curve_entries(
    plan: list[dict],
    scope: str,
    runs_by_id: dict[str, RunPaths],
    **load,
) -> list[dict]:
    """One entry per `row_plan` row of `scope`: its kind, its runs, its curves.

    A row with no runs is **pending** — the traffic-weighted campaign before it
    is collected. It carries no curves rather than a placeholder:
    `headline_table.PROVISIONAL_WEIGHTED` holds top-1 rates, and a rate has no
    distance distribution to draw.
    """
    entries: list[dict] = []
    for row in plan:
        if row["scope"] != scope:
            continue
        runs = [runs_by_id[r] for r in row["run_ids"]]
        pooled = (
            pool_runs(runs, **load)
            if runs
            else {
                "errors": {},
                "counts": {},
                "methods_absent_in_some_runs": [],
                "per_run_counts": {},
            }
        )
        entries.append(
            {
                "kind": row["kind"],
                "dataset": row["dataset"],
                "datasets": list(row["datasets"]),
                "run_ids": list(row["run_ids"]),
                "pending": not runs,
                **pooled,
            }
        )
    return entries


def curve_table(entries: list[dict]) -> pd.DataFrame:
    """`percentile_table` per entry, stacked, with the kind and dataset attached.

    The per-entry arithmetic is `percentile_table`'s own, so a one-run mesh row
    here and the per-run CSV are the same numbers.
    """
    frames = []
    for entry in entries:
        if entry["pending"]:
            continue
        table = percentile_table(entry["errors"], entry["counts"])
        table.insert(0, "kind", entry["kind"])
        table.insert(1, "kind_label", H.KIND_LABELS[entry["kind"]])
        table.insert(2, "dataset", entry["dataset"] or ";".join(entry["datasets"]))
        table.insert(3, "run_ids", ";".join(entry["run_ids"]))
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _draw_kind_curves(ax, entries: list[dict], min_x_km: float) -> list[str]:
    """Every `(kind, method)` curve in `entries`; returns the methods drawn.

    Colour is the method, the baseline's own variant hue included; line style
    is the kind. Baselines go last, over every kind's variants, so they read on
    top as they do in the per-run figure. Each line's `gid` is `kind:method`,
    which is what the tests read back.
    """
    drawn = method_order({m for e in entries for m in e["errors"]})
    colors = method_colors(drawn)
    ordered = sorted(entries, key=lambda e: H.KINDS.index(e["kind"]))
    passes = [
        (e, m) for e in ordered for m in method_order(e["errors"]) if m != SHORTEST_PING
    ] + [(e, SHORTEST_PING) for e in ordered if SHORTEST_PING in e["errors"]]
    for entry, method in passes:
        values = entry["errors"][method]
        if len(values) == 0:
            continue
        xs, ys = _cdf(values, min_x_km)
        is_baseline = method == SHORTEST_PING
        ax.plot(
            xs,
            ys,
            color=colors[method],
            linestyle=KIND_LINESTYLE[entry["kind"]],
            linewidth=2.2 if is_baseline else 2.0,
            alpha=0.95,
            zorder=3 if is_baseline else 2,
            gid=f"{entry['kind']}:{method}",
        )
    return drawn


def _method_handles(methods: list[str]):
    """Solid swatches in the method hue — the colour key, style-free."""
    from matplotlib.lines import Line2D

    colors = method_colors(methods)
    return [
        Line2D(
            [], [], color=colors[m], linewidth=2.0,
            label=f"{short_label(m)} (baseline)" if m == SHORTEST_PING else short_label(m),
        )
        for m in methods
    ]


def _kind_handles():
    """Neutral lines in each kind's style — the line-style key, colour-free.

    Both kinds are always listed, pending or not: this is the key to a fixed
    encoding, and an entry that comes and goes with the data makes the encoding
    look data-dependent. Whether a kind was collected is the caption's to say.
    """
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [], [], color=_C_MUTED, linestyle=KIND_LINESTYLE[k], linewidth=2.0,
            label=H.KIND_LABELS[k].lower(),
        )
        for k in H.KINDS
    ]


def pending_note(entries: list[dict]) -> str:
    """The caption line naming what was not drawn or drawn over fewer datasets.

    The second case is a pooled kind that covers only some of the datasets its
    mesh twin pools — `headline_table`'s `(1 of 3 ASes)` row. Its curve is a
    different population from the solid one beside it, and the percentile
    box's `n` alone is too quiet a place to say so.
    """
    coverage = max((len(e["datasets"]) for e in entries), default=0)
    partial = [
        f"{H.KIND_LABELS[e['kind']].lower()} pools {len(e['datasets'])} of "
        f"{coverage} datasets ({', '.join(d.upper() for d in e['datasets'])})"
        for e in entries
        if e["dataset"] is None and not e["pending"] and len(e["datasets"]) < coverage
    ]
    missing = [e for e in entries if e["pending"]]
    if not missing:
        return ("Partial: " + "; ".join(partial) + ".") if partial else ""
    by_kind: dict[str, list[str]] = {}
    for e in missing:
        by_kind.setdefault(e["kind"], []).append(
            str(e["dataset"]).upper() if e["dataset"] else "pooled"
        )
    parts = [
        f"{H.KIND_LABELS[k].lower()} not collected"
        + ("" if names == ["pooled"] else f" for {', '.join(names)}")
        for k, names in by_kind.items()
    ]
    note = "No curve drawn: " + "; ".join(parts) + "."
    return note + (" Partial: " + "; ".join(partial) + "." if partial else "")


def plot_pooled(
    entries: list[dict],
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
    figsize: tuple[float, float] = (8.6, 6.8),
) -> Path:
    """One panel: each kind's pooled curve per method, mesh solid, weighted dashed.

    The per-run figure's layout — method key upper left with the percentile
    box under it — plus a line-style key lower right, the corner a CDF leaves
    empty (large error, low fraction).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_SURFACE)
    _draw_guides(ax, min_x_km, max_x_km)
    methods = _draw_kind_curves(ax, entries, min_x_km)
    _style_axes(ax, min_x_km, max_x_km)

    method_key = ax.legend(
        handles=_method_handles(methods), loc="upper left", fontsize=8.5, frameon=False
    )
    ax.add_artist(method_key)
    kind_key = ax.legend(
        handles=_kind_handles(), loc="lower right", fontsize=8.5, frameon=False,
        handlelength=2.6,
    )
    for legend in (method_key, kind_key):
        for text in legend.get_texts():
            text.set_color(_C_INK_2)

    ax.set_title(title, fontsize=13, fontweight="bold", color=_C_INK, pad=16)
    ax.annotate(
        subtitle, xy=(0.5, 1.005), xycoords="axes fraction", ha="center",
        va="bottom", fontsize=9.5, color=_C_INK_2,
    )
    note = pending_note(entries)
    _footnote(ax, FOOTNOTE + (f"\n{note}" if note else ""))

    fig.canvas.draw()
    if not table.empty:
        _percentile_box(ax, table, method_key, by_kind=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def plot_compare(
    entries: list[dict],
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
    panel_inches: tuple[float, float] = (5.0, 4.6),
) -> Path:
    """One panel per dataset, its mesh and weighted curves overlaid.

    Both axes are shared, and fixed anyway, so a curve's position compares
    across panels directly. No percentile box: three of them would crowd
    panels this size, and the CSV twin carries every number.
    """
    datasets = list(dict.fromkeys(e["dataset"] for e in entries))
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(panel_inches[0] * len(datasets), panel_inches[1]),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor=_SURFACE,
    )
    axes = axes[0]
    methods: set[str] = set()
    for ax, dataset in zip(axes, datasets):
        panel = [e for e in entries if e["dataset"] == dataset]
        ax.set_facecolor(_SURFACE)
        _draw_guides(ax, min_x_km, max_x_km)
        methods.update(_draw_kind_curves(ax, panel, min_x_km))
        _style_axes(ax, min_x_km, max_x_km, labels=False)
        mesh = next((e for e in panel if e["kind"] == H.MESH), None)
        n = max((c["n_total"] for c in mesh["counts"].values()), default=0) if mesh else 0
        ax.set_title(f"{str(dataset).upper()}  ·  n={n:,}", fontsize=10, color=_C_INK, pad=8)
    axes[0].set_ylabel("Fraction of solved targets", fontsize=10.5, color=_C_INK_2)
    fig.supxlabel("Error distance to the raw target (km)", fontsize=10.5, color=_C_INK_2)
    fig.tight_layout()
    # `tight_layout` knows nothing of figure-level text, so the band above the
    # panels is reserved by hand: title, subtitle, then the key, top down.
    fig.subplots_adjust(top=0.78)

    handles = _method_handles(method_order(methods)) + _kind_handles()
    legend = fig.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, 0.865),
        ncol=len(handles), frameon=False, fontsize=8.5, handlelength=2.6,
        columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_color(_C_INK_2)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=0.99)
    fig.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=9.5, color=_C_INK_2)
    note = pending_note(entries)
    fig.text(
        0.5, -0.02, FOOTNOTE.replace("\n", " ") + (f"\n{note}" if note else ""),
        ha="center", va="top", fontsize=8, color=_C_MUTED,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def build_cross(
    mesh_runs: dict[str, RunPaths],
    weighted_runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    layouts: tuple[str, ...] = (POOLED,),
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> tuple[dict[str, tuple[Path, pd.DataFrame]], dict]:
    """Render the cross-run layouts; `({layout: (png, table)}, manifest)`.

    Rows come from `headline_table.row_plan`, the plan the headline table and
    the outcome bars print from, so this figure's curves are those rows'
    populations by construction — and an unpaired weighted run or two runs of
    one dataset are refused with that plan's own message.
    """
    plan = H.row_plan(mesh_runs, weighted_runs)
    runs_by_id = {**mesh_runs, **{r.run_id: r for r in weighted_runs.values()}}
    load = dict(
        analysis_root=analysis_root, grid=grid, resolution=resolution, methods=methods
    )
    out_dir = cross.cross_dir(analysis_root, sorted(mesh_runs), kind=H.CROSS_KIND)
    slug = grid_slug(grid, resolution)
    label = "+".join(cross.short_dataset(r) for r in mesh_runs)

    # The pooled layout's single-dataset case falls back to its DATASET rows,
    # the rule `figure_outcome_bars.outcome_table` uses: `row_plan` suppresses
    # a one-dataset aggregate because it would *be* that dataset.
    pooled_scope = (
        H.AGGREGATE if any(r["scope"] == H.AGGREGATE for r in plan) else H.DATASET
    )
    scopes = {POOLED: pooled_scope, COMPARE: H.DATASET}
    stems = {POOLED: "error_cdf", COMPARE: "error_cdf_by_dataset"}

    rendered: dict[str, tuple[Path, pd.DataFrame]] = {}
    described: list[dict] = []
    for layout in layouts:
        entries = curve_entries(plan, scopes[layout], runs_by_id, **load)
        table = curve_table(entries)
        png = out_dir / f"{stems[layout]}.{slug}.png"
        common = dict(
            title="Error distance to the raw target, mesh vs traffic-weighted",
            max_x_km=max_x_km,
            min_x_km=min_x_km,
        )
        if layout == POOLED:
            plot_pooled(
                entries, table, png,
                subtitle=f"{label} pooled · {slug} · solved rows only", **common,
            )
        else:
            plot_compare(
                entries, png,
                subtitle=f"one panel per dataset · {slug} · solved rows only", **common,
            )
        rendered[layout] = (png, table)
        for e in entries:
            described.append(
                {
                    "layout": layout,
                    "kind": e["kind"],
                    "dataset": e["dataset"],
                    "datasets": e["datasets"],
                    "run_ids": e["run_ids"],
                    "pending": e["pending"],
                    "methods": list(e["errors"]),
                    "methods_absent_in_some_runs": e["methods_absent_in_some_runs"],
                    "counts": e["counts"],
                    "per_run_counts": e["per_run_counts"],
                    "n_clamped_to_floor": {
                        m: int((v < min_x_km).sum())
                        for m, v in e["errors"].items()
                        if (v < min_x_km).any()
                    },
                }
            )

    manifest = {
        "mesh_runs": sorted(mesh_runs),
        "weighted_runs": {d: r.run_id for d, r in sorted(weighted_runs.items())},
        "grid": {"scheme": grid, "resolution": resolution},
        "layouts": list(layouts),
        "baseline": SHORTEST_PING,
        "error_column": ERROR_COLUMN,
        "percentiles": list(PERCENTILES),
        "x_axis": {"scale": "log", "min_km": min_x_km, "max_km": max_x_km},
        "curves": described,
        "encoding": (
            "colour = method, in the paper's fixed variant hues with the baseline "
            "in its own; line style = dataset type, mesh solid and traffic-weighted "
            "dashed. The per-run layout's grey dashed baseline is not used here, "
            "because a dash means traffic-weighted."
        ),
        "pooling": (
            "micro-pool: a kind's solved rows are concatenated across its runs, so "
            "each dataset weighs by its target count and pooled n_solved is the sum "
            "of the runs' topn_accuracy.csv n_solved. Target ids are checked "
            "disjoint within a kind (cross.guard_disjoint_targets); never across "
            "kinds, since traffic-weighted is its mesh twin filtered."
        ),
        "method_policy": (
            "a method enters a kind's pooled curve only if every run of that kind "
            "scored it; the rest are listed per curve under "
            "methods_absent_in_some_runs"
        ),
        "row_policy": (
            "solved rows only, via io.solved_mask — FALLBACK excluded (§7.2), and "
            "Shortest-Ping's all-BASELINE rows counted as solved."
        ),
        "distance_policy": (
            "error_to_target_km: prediction to the ground-truth coordinate, not "
            "through the cell-centre seed."
        ),
        "pending_note": (
            "a row_plan row with no run draws no curve. "
            "headline_table.PROVISIONAL_WEIGHTED holds top-1 rates, not distances, "
            "so the placeholder the table prints has no curve to give here."
        ),
    }
    return rendered, manifest


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-error-cdf")
    def plot_error_cdf_cmd(
        run_id: list[str] = typer.Option(
            None,
            "--run-id",
            help="Runs to render (repeatable): one figure each under per-run; the "
            "mesh runs, one per dataset, under pooled/compare.",
        ),
        all_runs: bool = typer.Option(
            False,
            "--all-runs",
            help="Render every run found under --outputs-root (per-run only).",
        ),
        layout: list[str] = typer.Option(
            [],
            "--layout",
            help=f"{PER_RUN} (one figure per run), {POOLED} (each dataset type "
            f"pooled over the runs, mesh solid vs weighted dashed) or {COMPARE} "
            f"(one panel per dataset). Repeatable; default {PER_RUN}.",
        ),
        weighted_run_id: list[str] = typer.Option(
            None,
            "--weighted-run-id",
            help="Traffic-weighted twin, as <dataset>=<run_id> (repeatable), drawn "
            "dashed by pooled/compare. None are collected yet.",
        ),
        allow_mixed_setups: bool = typer.Option(
            False,
            "--allow-mixed-setups",
            help="Let pooled/compare span both run setups (SCHEMA.md §7).",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None,
            "--method",
            help="Restrict to these method ids (repeatable). Default: the six "
            "published variants.",
        ),
        max_x_km: float = typer.Option(
            DEFAULT_X_MAX_KM, "--max-x-km", help="Upper bound of the log x axis (km)."
        ),
        min_x_km: float = typer.Option(
            X_MIN_KM,
            "--min-x-km",
            help="Lower bound of the log x axis (km). Errors below it are clamped "
            "up in the drawn curve only; the percentile CSV is unclamped.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Error-distance CDF per method, log x, fallbacks excluded.

        per-run writes error_cdf.png + error_cdf_percentiles.csv + a manifest into
        target-cls-accuracy/<grid>-<resolution>/. pooled / compare write
        error_cdf[_by_dataset].<grid>.{png,csv} + a manifest into
        _cross/accuracy-table/<dataset-set>/, beside the outcome bars. Needs
        `classify` on every run.
        """
        unknown = [x for x in layout if x not in LAYOUTS]
        if unknown:
            raise typer.BadParameter(f"unknown --layout {unknown}; pick from {list(LAYOUTS)}")
        layouts = tuple(dict.fromkeys(layout)) or (PER_RUN,)
        cross_layouts = tuple(x for x in layouts if x != PER_RUN)
        if weighted_run_id and not cross_layouts:
            raise typer.BadParameter(
                f"--weighted-run-id is drawn by --layout {POOLED} or {COMPARE}; the "
                f"{PER_RUN} figure has no weighted curve"
            )
        if cross_layouts and all_runs:
            raise typer.BadParameter(
                f"--layout {'/'.join(cross_layouts)} needs explicit --run-id: which "
                "datasets belong in one figure is the caller's call, as for "
                "plot-outcome-bars"
            )
        if all_runs and run_id:
            raise typer.BadParameter("pass --run-id or --all-runs, not both")
        runs = (
            discover_runs(outputs_root)
            if all_runs
            else [resolve_run(rid, outputs_root) for rid in (run_id or [])]
        )
        if not runs:
            raise typer.BadParameter("pass at least one --run-id, or --all-runs")

        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
        if cross_layouts:
            mesh_runs = {run.run_id: run for run in runs}
            weighted_runs = {
                dataset: resolve_run(rid, outputs_root)
                for dataset, rid in H.parse_weighted_pairs(weighted_run_id or []).items()
            }
            cross.guard_one_setup(
                {**mesh_runs, **{r.run_id: r for r in weighted_runs.values()}},
                allow_mixed=allow_mixed_setups,
            )
            for res in resolutions:
                rendered, manifest = build_cross(
                    mesh_runs,
                    weighted_runs,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=res,
                    methods=list(method) if method else None,
                    layouts=cross_layouts,
                    max_x_km=max_x_km,
                    min_x_km=min_x_km,
                )
                for png, table in rendered.values():
                    table.to_csv(png.with_suffix(".csv"), index=False)
                out_dir = next(iter(rendered.values()))[0].parent
                slug = grid_slug(g.name, res)
                (out_dir / f"error_cdf.{slug}.manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                pending = sorted(
                    {c["kind"] for c in manifest["curves"] if c["pending"]}
                )
                typer.echo(
                    f"{'+'.join(sorted(mesh_runs))} {slug} · layouts "
                    f"{', '.join(rendered)}"
                    + (f" · pending: {', '.join(pending)}" if pending else "")
                    + f" -> {', '.join(p.name for p, _ in rendered.values())} in {out_dir}"
                )
        if PER_RUN not in layouts:
            return
        for run in runs:
            for res in resolutions:
                png, table, manifest = build(
                    run,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=res,
                    methods=list(method) if method else None,
                    max_x_km=max_x_km,
                    min_x_km=min_x_km,
                )
                out_dir = png.parent
                table.to_csv(out_dir / CURVE_CSV, index=False)
                (out_dir / MANIFEST_JSON).write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                best = table.loc[table["error_km_p50"].idxmin()]
                typer.echo(
                    f"{run.run_id} {grid_slug(g.name, res)} · "
                    f"{len(manifest['methods'])} methods · "
                    f"best p50 {best['error_km_p50']:.0f} km "
                    f"({best['method_label']}) -> {png}"
                )
