"""Per-stage cost per method: where does each variant spend, and what does the
pipeline actually cost.

`pareto.py` collapses a variant to one cost scalar so it can sit on an accuracy
frontier. That is the right shape for "is it worth it" and the wrong shape for
"why" — it cannot say that Octant's memory is a Sobol grid in CTR while
Vanilla's is polygon geometry in MTL. This figure is the decomposition, and
`cost.combo_stage_costs` is the API it was written for (`cost.py:3` named this
module as its owner before it existed).

## Grouped, never stacked

The legacy `plot_phase_memory.py` / `plot_phase_runtime.py` stacked per-stage
percentiles. `test_cost.py:166-213` proves that bar has no statistic under it:
stacking p50s of 1/10/100 gives 111 when no target costs 111, and for memory
`sum >= max` at every stat, so a stacked memory bar overstates the true peak by
construction. Only the *mean* is exactly additive, and only under `sum` — one
corner of the parameter space. Rather than ship a `--layout stacked` valid in
that corner alone, this figure does not stack at all.

So the three stages are drawn side by side, and the honest composite is a
separate mark: the **PIPELINE tick**, `cost_stats(per_target_cost(...))` —
reduced per target *before* the percentile, `max` for memory and `sum` for
runtime. It lands on the tallest bar for memory (the pipeline high-water mark
is whichever stage peaked) and above all three for runtime (stages run in
sequence). A reader who tries to eyeball the total by adding bars is corrected
by the tick.

## Boxes, not bars — because the axis is log

The span is ~4 orders of magnitude on real data (Vanilla's 6 KB CTR against
Octant's 25 MB one), so the axis has to be log. A bar encodes magnitude with
*length from a baseline*, and a log axis has no zero to anchor to: the bar's
base becomes whatever `ylim` happens to be, and its length encodes nothing.
A box encodes a *range between two positions*, which stays valid — and on a log
axis equal vertical distance is equal ratio, which is the quantity that matters
here (MTL's 11x heap/alloc gap, Octant's 6x MTL gap over Vanilla).

Boxes rather than single dots because the choice of statistic was itself a
parameter, and a poor one: p50 answers "which method costs more" while p95
answers "how much must I provision", they can rank differently, and emitting a
near-identical artifact per stat multiplies files without adding information.
The box shows p5 / p25 / p50 / p75 / p95 at once, so `--cost-stat` is gone.

## Four boxes: the pipeline is a distribution too

The fourth box is the per-target reduce — `max` for memory, `sum` for runtime —
and it is drawn as a box rather than a reference line because it has its own
spread that is *not* derivable from the three stage boxes. The p95 of a
per-target max is not the max of the three p95s; no target need be at the 95th
percentile of every stage at once. Drawing it as a line would make the one
quantity a reader actually provisions against the least legible thing on the
figure.

## Colour is the stage, not the variant

The second figure in this package to make that departure, after
`figure_outcome_bars` — same reason: the question is "which stage", so the
stage must be what hue encodes, and the variant is the x position.

Stage is *ordinal* (LTD -> MTL -> CTR is a fixed pipeline order), so `STAGE_INK`
is a single-hue ordinal ramp rather than three categorical hues — steps 250 /
450 / 650 of the blue ramp. Validated with `validate_palette.js --ordinal`:
monotone lightness, all adjacent dL >= 0.06, light end 2.06:1 against the
surface, hue spread 3 degrees. An earlier hand-picked gray-to-navy ramp failed
both the lightness band and the chroma floor.

## Two denominators, on purpose

`cost._scaled_stages` fills a null stage to 0 — correct, because a CTR that
never ran on a FALLBACK target cost nothing. That zero is fine inside the
pipeline reduce (`max(ltd, mtl, 0)` is genuinely that target's peak) but it
cannot be *plotted*: on `as7018-ripe-mesh-reciprocal` `vanilla_cbg` is null on
10 of 45 CTR rows, so 22% zeros would put p25 at 0, and 0 has no position on a
log axis.

So the two blocks use different row sets, deliberately:

  * a **stage** box is drawn over the rows where that stage ran, because "what
    CTR costs when CTR runs" is the meaningful distribution;
  * the **pipeline** box keeps every row, because dropping the fallback targets
    would discard the cheap ones and inflate the figure an operator sizes from.

This is the one place the shared-denominator property is knowingly broken, so
each box carries its own `n` and the caption says so.

"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules import cost as cost_mod
from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.cost import (
    COST_ROWS,
    COST_SPECS,
    PIPELINE,
    REDUCERS,
    STAGE_LABELS,
    STAGES,
    CostSpec,
    combo_stage_costs,
)
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.labels import (
    PREFERRED_ORDER,
    label_for,
)
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
    _SURFACE,
)
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths

DPI = 200

#: Stage -> ink. Monotone lightness up the pipeline so the LTD -> MTL -> CTR
#: order is recoverable in greyscale, and deliberately NOT the six variant hues
#: from `palette._VARIANT_HUES`: in this figure hue is the stage, and reusing a
#: variant hue would collide with every other figure in the paper, where the
#: same hue is a method.
STAGE_INK = {
    "ltd": "#86b6ef",   # blue 250 — the light end still clears 2:1 on surface
    "mtl": "#2a78d6",   # blue 450
    "ctr": "#104281",   # blue 650
}

#: A within-method span guide was tried and removed: the dots sit at
#: `x +/- width`, so a rule at the group centre joins none of them and reads as
#: a stem hanging off MTL. The three dots in a column already show the span.

#: The pipeline mark is a rule, not a dot: it is a different kind of quantity
#: (a per-target reduce, not a marginal) and must not read as a fourth stage.
#: Red against the blue ramp so form *and* hue separate it.
PIPELINE_INK = "#e34948"

#: Marker diameter (pt). The pipeline rule usually lands on the dearest stage's
#: dot under `max`, so overlapping marks get a surface ring to stay countable.
DOT_SIZE = 9.0
RING_INK = "#ffffff"

def artifact_name(cost_key: str, ext: str, *, rows: str, reduce: str) -> str:
    """`phase_cost_memory_heap.all.max.png`.

    Every axis that changes the numbers is in the name. `cost_stat` used to be
    one and no longer is — the box shows the whole distribution, so there is one
    artifact per (cost, rows, reduce) instead of one per statistic.
    `pareto.artifact_name` omits rows/reduce entirely, so its sweeps overwrite
    each other; that defect is not inherited.
    """
    return f"phase_cost_{cost_key}.{rows}.{reduce}.{ext}"


def collect(
    run: RunPaths,
    methods: list[str],
    spec: CostSpec,
    *,
    rows: str,
    reduce: str,
) -> pd.DataFrame:
    """Long frame: one row per (method, stage), plus the pipeline row.

    Carries the full box block (p5/p25/p50/p75/p95) and each block's own `n`,
    because the two kinds of block use different row sets — see the module
    docstring. A stage is summarised over the rows where it ran; the pipeline
    over every row, since a fallback target's zero-cost CTR is a real datum
    there rather than a missing one.
    """
    records: list[dict] = []
    keys = ("p5", "p25", "p50", "p75", "p90", "p95", "mean", "min", "max", "n")
    for method in methods:
        df = cost_mod.load_cost_frame(run, method, spec, rows=rows)
        scaled = cost_mod.per_stage_cost(df, spec)
        pipeline = cost_mod.per_target_cost(df, spec, reduce=reduce)
        cost_mod.require_measured(
            cost_mod.cost_stats(pipeline), spec, run_id=run.run_id, method=method
        )
        n_rows = len(df)

        for stage, col in zip(STAGES, spec.stage_cols):
            ran = pd.to_numeric(df[col], errors="coerce").notna().to_numpy()
            block = cost_mod.cost_stats(scaled[stage].to_numpy()[ran])
            records.append({
                "method": method, "label": label_for(method), "stage": stage,
                "n_rows": n_rows, "n_null": int((~ran).sum()),
                **{k: block[k] for k in keys},
            })

        block = cost_mod.cost_stats(pipeline)
        records.append({
            "method": method, "label": label_for(method), "stage": PIPELINE,
            "n_rows": n_rows, "n_null": 0,
            **{k: block[k] for k in keys},
        })
    return pd.DataFrame.from_records(records)


def _order_methods(methods) -> list[str]:
    """`PREFERRED_ORDER` first, then anything else alphabetically.

    Identity order, not a ranking by the quantity plotted — the same variant
    occupies the same x slot in every figure of a sweep.
    """
    known = [m for m in PREFERRED_ORDER if m in set(methods)]
    rest = sorted(set(methods) - set(known))
    return known + rest


#: Box geometry. Whiskers are p5/p95 rather than a Tukey fence, because the
#: fence is defined off the IQR and these distributions are heavily skewed —
#: CTR is near-constant, MTL has a long right tail. A stated percentile pair is
#: legible; a computed fence would silently mean something different per stage.
BOX_WIDTH = 0.19
WHIS_PCT = ("p5", "p95")


def _bxp_stat(row, stat_keys=WHIS_PCT) -> dict:
    lo, hi = stat_keys
    return {
        "med": row["p50"], "q1": row["p25"], "q3": row["p75"],
        "whislo": row[lo], "whishi": row[hi], "fliers": [],
    }


def plot(
    df: pd.DataFrame,
    spec: CostSpec,
    out_path: Path,
    *,
    rows: str,
    reduce: str,
    run_id: str,
    title: str | None = None,
) -> None:
    methods = _order_methods(df["method"].unique())
    x = np.arange(len(methods), dtype=float)
    blocks = (*STAGES, PIPELINE)
    # Four boxes centred on the method's tick: offsets -1.5..+1.5 half-widths.
    offsets = {b: (i - 1.5) * BOX_WIDTH for i, b in enumerate(blocks)}

    fig, ax = plt.subplots(
        figsize=(1.55 * len(methods) + 2.8, 4.6), dpi=DPI, facecolor=_SURFACE
    )
    ax.set_facecolor(_SURFACE)

    by = {(r["method"], r["stage"]): r for _, r in df.iterrows()}
    any_partial = False

    for block in blocks:
        is_pipeline = block == PIPELINE
        face = PIPELINE_INK if is_pipeline else STAGE_INK[block]
        stats, positions = [], []
        for j, m in enumerate(methods):
            row = by[(m, block)]
            if not np.isfinite(row["p50"]) or row["p50"] <= 0:
                continue
            if row["n_null"]:
                any_partial = True
            stats.append(_bxp_stat(row))
            positions.append(x[j] + offsets[block])
        if not stats:
            continue
        art = ax.bxp(
            stats, positions=positions, widths=BOX_WIDTH * 0.82,
            showfliers=False, patch_artist=True, manage_ticks=False, zorder=3,
        )
        for box in art["boxes"]:
            box.set_facecolor(_SURFACE if is_pipeline else face)
            box.set_edgecolor(face)
            box.set_linewidth(1.6 if is_pipeline else 0.9)
        for key in ("whiskers", "caps"):
            for ln in art[key]:
                ln.set_color(face)
                ln.set_linewidth(1.0)
        for ln in art["medians"]:
            # The median must survive on a dark fill and on the open pipeline
            # box alike, so it takes surface ink inside a filled box.
            ln.set_color(face if is_pipeline else RING_INK)
            ln.set_linewidth(1.8)
        label = STAGE_LABELS[block]
        ax.plot([], [], marker="s", linestyle="none", markersize=8,
                markerfacecolor=_SURFACE if is_pipeline else face,
                markeredgecolor=face, markeredgewidth=1.6, label=label)

    ax.set_yscale("log")
    finite = df[["p5", "p95"]].to_numpy().ravel()
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if len(finite):
        ax.set_ylim(finite.min() / 2.2, finite.max() * 2.2)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylabel(spec.axis_label, fontsize=10.5, color=_C_INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m) for m in methods], fontsize=9, rotation=18, ha="right")
    ax.grid(True, axis="y", which="major", color=_C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_C_MUTED, labelsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=4,
              fontsize=8.5, frameon=False)

    fig.suptitle(
        title or f"{spec.title_noun.capitalize()} per stage — {run_id}",
        fontsize=11.5, fontweight="bold", color=_C_INK,
    )

    n_rows = int(df["n_rows"].iloc[0]) if len(df) else 0
    caption = (
        f"{spec.key} over {rows} rows ({n_rows} targets). Box p25-p75, whisker "
        f"p5-p95, line p50. The pipeline box is the per-target {reduce} across "
        f"LTD/MTL/CTR, reduced before the percentile — stage boxes are marginals "
        f"and do not compose into it."
    )
    if any_partial:
        partial = [f"{by[(m, st)]['label']} {STAGE_LABELS[st].split()[0]} "
                   f"n={int(by[(m, st)]['n'])}/{n_rows}"
                   for m in methods for st in STAGES if by[(m, st)]["n_null"]]
        caption += (" A stage is summarised over the rows where it ran ("
                    + "; ".join(partial) + "); the pipeline box keeps every row.")
    ax.annotate(
        "\n".join(textwrap.wrap(caption, 104)),
        xy=(0.5, -0.40), xycoords="axes fraction",
        ha="center", va="top", fontsize=8, color=_C_MUTED, linespacing=1.4,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)


def register(app) -> None:
    import typer

    from scripts.analysis.v3.modules.paths import DEFAULT_ANALYSIS_ROOT, DEFAULT_OUTPUTS_ROOT, resolve_run

    @app.command("plot-phase-cost")
    def plot_phase_cost_cmd(
        run_id: str = typer.Option(..., "--run-id", help="Run to decompose."),
        cost: str = typer.Option(
            "memory_heap", "--cost", help=f"Cost channel: {list(COST_SPECS)}."
        ),
        cost_rows: str = typer.Option(
            "all", "--cost-rows",
            help="Which target rows the cost is over. 'all' matches accuracy_topN's "
                 "denominator; 'solved' drops FALLBACK rows.",
        ),
        memory_reduce: str = typer.Option(
            "max", "--memory-reduce",
            help="How the pipeline rule combines LTD/MTL/CTR. 'max' is the "
                 "high-water mark; 'sum' the no-release upper bound. Runtime "
                 "always sums.",
        ),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict the method pool (repeatable)."
        ),
        title: str = typer.Option(None, "--title", help="Override the figure title."),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
        out_dir: Path = typer.Option(None, "--out-dir", help="Override the artifact directory."),
    ) -> None:
        """Per-stage cost per method, with the true pipeline reduce beside it.

        Four box-and-whisker blocks per method (LTD, MTL, CTR, pipeline) on a
        log axis. Never stacked: stacking per-stage percentiles has no statistic
        under it, and for memory it overstates the peak by construction. There
        is no `--cost-stat` — the box shows p5/p25/p50/p75/p95 at once.
        """
        if cost not in COST_SPECS:
            raise typer.BadParameter(f"--cost must be one of {list(COST_SPECS)}")
        if cost_rows not in COST_ROWS:
            raise typer.BadParameter(f"--cost-rows must be one of {list(COST_ROWS)}")
        if memory_reduce not in REDUCERS:
            raise typer.BadParameter(f"--memory-reduce must be one of {list(REDUCERS)}")

        spec = COST_SPECS[cost]
        cost_mod.warn_if_deprecated(spec, echo=typer.echo)
        reduce = "sum" if spec.key == "runtime" else memory_reduce

        run = resolve_run(run_id, outputs_root)
        methods = _order_methods(list(method) if method else run.combo_ids)
        if not methods:
            raise MissingArtifactError(f"{run_id}: no combos found under {run.setup_dir}")

        df = collect(run, methods, spec, rows=cost_rows, reduce=reduce)

        target_dir = Path(out_dir) if out_dir else run.analysis_dir("phase-cost", root=analysis_root)
        png = target_dir / artifact_name(spec.key, "png", rows=cost_rows, reduce=reduce)
        csv = target_dir / artifact_name(spec.key, "csv", rows=cost_rows, reduce=reduce)
        meta = target_dir / artifact_name(spec.key, "json", rows=cost_rows, reduce=reduce)

        plot(df, spec, png, rows=cost_rows, reduce=reduce, run_id=run_id, title=title)
        df.to_csv(csv, index=False)
        meta.write_text(json.dumps({
            "run_id": run_id,
            "cost": {
                "key": spec.key, "rows": cost_rows,
                "reduce": reduce, "unit": spec.unit, "stage_cols": list(spec.stage_cols),
            },
            "methods": methods,
            "n_rows": int(df["n_rows"].iloc[0]) if len(df) else 0,
            "nulls_by_stage": {
                m: {s: int(df[(df.method == m) & (df.stage == s)]["n_null"].iloc[0]) for s in STAGES}
                for m in methods
            },
        }, indent=2) + "\n")
        typer.echo(f"Wrote {png}")
