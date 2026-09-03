"""Cost model — which columns are a cost, how they compose, and in what unit.

Owns *the cost model*; `pareto.py` and `phase_cost.py` own the figures that
read it. Three commands now depend on the policies below, so they live here
rather than beside any one of them.

The per-target runtime and memory the benchmark records in `targets.parquet`
(`{ltd,mtl,ctr}_{ms,alloc_peak_bytes,rss_peak_bytes}` — see ../SCHEMA.md §3)
are per-*stage* peaks and durations. Turning them into a number means deciding
how stages compose, and the two channels do not compose alike:

1. **Memory reduces across stages with `max`, not `sum`.** `instrument.py`
   resets tracemalloc *inside* each stage and the RSS sampler returns a delta,
   so the columns are per-stage peaks: `max` is the pipeline high-water mark,
   `sum` the no-release upper bound. Summing also triples the ~41.6 KB
   tracemalloc pedestal, which is bookkeeping rather than work. Runtime
   genuinely sums.
2. **`memory_rss` is degenerate at p50** — the sampler is 5 ms, so fast stages
   floor at one 4096-byte page — but graded at p95. Hence `cost_stat`.
3. **Reduce across stages per target, then percentile.** Never the reverse. A
   quantile does not distribute over a sum (per-stage medians 1/10/100 sum to
   111 and no target costs 111), and for `max` there is no statistic at which
   the reverse order is correct. `per_stage_cost` exists so a caller can hold
   the three stages apart *without* losing this discipline: it stops one step
   short of the reduce, and `per_target_cost` is defined as its reduction, so
   the null policy has exactly one definition.

Callers own the row denominator. `combo_cost` / `combo_stage_costs` are the
only readers of `io.load_folds` here, and they apply one `rows` filter for
every stage — different denominators per stage is a real defect the legacy
`plot_phase_*` scripts shipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths

_BYTES_PER_MB = 1024 * 1024

#: The three instrumented pipeline stages, in execution order.
STAGES: tuple[str, str, str] = ("ltd", "mtl", "ctr")

#: The derived fourth mark: `per_target_cost`, i.e. the *correct* composition
#: of the three stages. Not a stage — a `stage` column value, so a reader can
#: check it against its three siblings without reshaping.
PIPELINE = "pipeline"

STAGE_LABELS: dict[str, str] = {
    "ltd": "LTD (distance)",
    "mtl": "MTL (multilateration)",
    "ctr": "CTR (centroid)",
    PIPELINE: "pipeline (per-target)",
}


@dataclass(frozen=True)
class CostSpec:
    """One cost axis: which columns, how to reduce them, and in what unit."""

    key: str
    stage_cols: tuple[str, str, str]
    reduce: str  # "sum" (runtime) | "max" (memory)
    scale: float
    unit: str
    axis_label: str
    supports_throughput: bool
    #: Prose name for the figure title; `key` is the CLI flag value and reads
    #: as an identifier rather than a quantity.
    title_noun: str


#: `reduce` is the field `plot_accuracy_cost_box._COST_SPECS` lacks — it sums
#: memory across stages, which double-counts the per-stage tracemalloc pedestal
#: and assumes nothing is released between stages.
COST_SPECS: dict[str, CostSpec] = {
    "runtime": CostSpec(
        key="runtime",
        stage_cols=("ltd_ms", "mtl_ms", "ctr_ms"),
        reduce="sum",
        scale=1.0,
        unit="ms",
        axis_label="Per-target runtime (ms)",
        supports_throughput=True,
        title_noun="per-target runtime",
    ),
    "memory_alloc": CostSpec(
        key="memory_alloc",
        stage_cols=(
            "ltd_alloc_peak_bytes",
            "mtl_alloc_peak_bytes",
            "ctr_alloc_peak_bytes",
        ),
        reduce="max",
        scale=1.0 / _BYTES_PER_MB,
        unit="MB",
        axis_label="Per-target peak tracemalloc (MB)",
        supports_throughput=False,
        title_noun="peak memory (tracemalloc)",
    ),
    "memory_rss": CostSpec(
        key="memory_rss",
        stage_cols=("ltd_rss_peak_bytes", "mtl_rss_peak_bytes", "ctr_rss_peak_bytes"),
        reduce="max",
        scale=1.0 / _BYTES_PER_MB,
        unit="MB",
        axis_label="Per-target peak sampled RSS (MB)",
        supports_throughput=False,
        title_noun="peak memory (sampled RSS)",
    ),
}

COST_STATS: tuple[str, ...] = ("p50", "p75", "p90", "p95", "mean")
COST_ROWS: tuple[str, ...] = ("all", "solved")

#: Valid `reduce` values. Named for the operation, not the channel: runtime
#: validates against this too, so the old `MEMORY_REDUCERS` was a misnomer.
REDUCERS: tuple[str, ...] = ("max", "sum")

_STAT_QUANTILE = {"p50": 50.0, "p75": 75.0, "p90": 90.0, "p95": 95.0}


def _scaled_stages(df: pd.DataFrame, spec: CostSpec) -> list[np.ndarray] | None:
    """Null-filled, `spec.scale`-scaled per-stage vectors, or None if the run
    was never instrumented.

    The single definition of the null policy, shared by `per_target_cost` and
    `per_stage_cost`.

    Nulls fill to 0 because a null stage never ran — a skipped CTR on a
    FALLBACK row cost nothing. The one exception is the LTD column, which
    `TARGETS_SCHEMA` declares non-nullable precisely because LTD always runs:
    if *it* is entirely null the run was not instrumented, which is absence of
    measurement rather than absence of work. That returns None so the caller
    can refuse to plot; filling it to 0 would manufacture a free method that
    then dominates the entire frontier.
    """
    missing = [c for c in spec.stage_cols if c not in df.columns]
    if missing:
        raise MissingArtifactError(
            f"cost columns {missing} absent. The pre-{'c7ee30a'} schema named these "
            f"`{{ltd,mtl,ctr}}_peak_bytes`; re-run the benchmark to get the "
            f"dual-channel columns this analysis needs."
        )

    out: list[np.ndarray] = []
    for idx, col in enumerate(spec.stage_cols):
        s = pd.to_numeric(df[col], errors="coerce")
        if idx == 0 and s.isna().all():
            return None
        out.append(s.fillna(0.0).to_numpy(dtype=float) * spec.scale)
    return out


def per_stage_cost(df: pd.DataFrame, spec: CostSpec) -> pd.DataFrame:
    """One column per stage, one row per target, in `spec.unit`.

    The sibling of `per_target_cost`, sharing its null policy exactly and
    deliberately stopping one step earlier: it does **not** reduce across
    stages. `per_target_cost(df, spec)` is this frame reduced row-wise by
    `spec.reduce`, and a test pins that identity so the two cannot drift.

    Columns are the short stage names (`cost.STAGES`), not the source column
    names, so a caller never has to know whether it asked for `_ms`,
    `_alloc_peak_bytes` or `_rss_peak_bytes`.

    An uninstrumented run NaNs **every** stage, not just LTD. Returning real
    MTL/CTR columns beside a NaN LTD would let a figure draw two bars for a run
    that was never measured, which reads as a working measurement.
    """
    stages = _scaled_stages(df, spec)
    if stages is None:
        return pd.DataFrame(
            {s: np.full(len(df), np.nan) for s in STAGES}, index=df.index
        )
    return pd.DataFrame(dict(zip(STAGES, stages)), index=df.index)


def per_target_cost(
    df: pd.DataFrame, spec: CostSpec, *, reduce: str | None = None
) -> np.ndarray:
    """One cost per target, in `spec.unit`.

    Order matters and is the one thing this function exists to pin: null-fill
    each stage, reduce **across stages per target**, and only then let the
    caller percentile the result. A median does not distribute over a sum; the
    same holds for `max`.

    Null policy is `_scaled_stages`'; an uninstrumented LTD column returns
    all-NaN so the caller can refuse to plot it.
    """
    reduce = reduce or spec.reduce
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {list(REDUCERS)}; got {reduce!r}")
    stages = _scaled_stages(df, spec)
    if stages is None:
        return np.full(len(df), np.nan)
    stack = np.vstack(stages) if stages else np.zeros((1, len(df)))
    return stack.sum(axis=0) if reduce == "sum" else stack.max(axis=0)


def cost_stats(values: np.ndarray) -> dict[str, float]:
    """p50/p75/p90/p95/mean/min/max/n over the finite values; NaN in -> NaN out."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {k: float("nan") for k in ("p50", "p75", "p90", "p95", "mean", "min", "max")}
        out["n"] = 0
        return out
    out = {k: float(np.percentile(arr, q)) for k, q in _STAT_QUANTILE.items()}
    out["mean"] = float(arr.mean())
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["n"] = int(arr.size)
    return out


def _rows_for(
    run: RunPaths, combo_id: str, spec: CostSpec, *, rows: str
) -> pd.DataFrame:
    """`io.load_folds` for one combo, filtered to `rows`. One denominator.

    `columns` is passed explicitly so the nested `ltd_predictions` /
    `mtl_participants` columns are never read (they dominate the file size)
    while `load_folds`' K-fold disjointness check still runs.
    """
    if rows not in COST_ROWS:
        raise ValueError(f"rows must be one of {list(COST_ROWS)}; got {rows!r}")
    cols = ["target_id", "status", *spec.stage_cols]
    df = io.load_folds(run, combo_id, columns=cols)
    if rows == "solved":
        df = df[df["status"].isin(io.CBG_SUCCESS_STATUSES)]
    return df


def combo_cost(
    run: RunPaths,
    combo_id: str,
    spec: CostSpec,
    *,
    rows: str = "all",
    reduce: str | None = None,
) -> tuple[dict[str, float], int]:
    """Cost stats for one combo, pooled over folds. Returns (stats, n_rows_used)."""
    df = _rows_for(run, combo_id, spec, rows=rows)
    return cost_stats(per_target_cost(df, spec, reduce=reduce)), len(df)


def combo_stage_costs(
    run: RunPaths,
    combo_id: str,
    spec: CostSpec,
    *,
    rows: str = "all",
    reduce: str | None = None,
) -> tuple[dict[str, dict[str, float]], int, dict[str, int]]:
    """Per-stage *and* pipeline cost stats for one combo, pooled over folds.

    Returns `(stats_by_stage, n_rows, nulls_by_stage)`.

    `stats_by_stage` has four keys — `ltd`, `mtl`, `ctr`, and `PIPELINE`. The
    last is `cost_stats(per_target_cost(...))`, the correct composition
    (`sum` for runtime, `max` for memory) computed per target *before* any
    percentile. Putting it beside the three marginals is the point: the three
    stage stats are comparable marginals over one target set, but their sum is
    not a statistic of anything.

    One `load_folds` call and one `rows` filter, so all four blocks share one
    denominator by construction. That is the audited defect — the legacy bars
    came from `_stat_block`'s per-column `dropna()` over the SUCCESS+FALLBACK
    subset (`scripts/benchmark/v2/cli.py:268-283`) while the reference line
    coerced nulls to 0 over every row: two denominators, live on `vanilla_cbg`
    at 21.4% FALLBACK rows with a null `ctr_ms`.

    `nulls_by_stage` counts the per-target nulls filled to 0 per stage. It is
    what makes the shared denominator legible: a reader can see that CTR's p50
    is low because a fifth of its rows are structural zeros, not because CTR
    is fast.
    """
    df = _rows_for(run, combo_id, spec, rows=rows)
    stages = per_stage_cost(df, spec)

    nulls = {
        stage: int(pd.to_numeric(df[col], errors="coerce").isna().sum())
        for stage, col in zip(STAGES, spec.stage_cols)
    }
    stats = {stage: cost_stats(stages[stage].to_numpy()) for stage in STAGES}
    stats[PIPELINE] = cost_stats(per_target_cost(df, spec, reduce=reduce))
    return stats, len(df), nulls
