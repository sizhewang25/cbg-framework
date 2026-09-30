"""Cost model -- which columns are a cost, how they compose, and in what unit.

Ported from v3 (`scripts/analysis/v3/modules/cost.py`). Owns *the cost model*;
`figure_cost_box.py` owns the figure that reads it. The policies are v3's,
unchanged; what moved is the loader (v5's `classify.load_method_frame`, in v5
names) and the answered predicate (`status.solved_mask`).

The per-target runtime and memory the benchmark records in `targets.parquet`
(`{ltd,mtl,ctr}_{ms,alloc_peak_bytes,heap_peak_bytes}`) are per-*stage* peaks
and durations. Turning them into one number per TG means deciding how stages
compose, and the channels do not compose alike:

1. **Memory reduces across stages with `max`, not `sum`.** `instrument.py`
   resets tracemalloc *inside* each stage and the samplers return deltas, so
   the columns are per-stage peaks: `max` is the pipeline high-water mark,
   `sum` the no-release upper bound. Runtime genuinely sums.
2. **`memory_alloc` and `memory_heap` do not measure the same thing and must
   never be combined.** tracemalloc sees Python objects and NumPy but is blind
   to Shapely/GEOS C allocations; the heap channel (mallinfo2) sees GEOS and
   large NumPy buffers but is blind to pymalloc-satisfied small objects. They
   overlap on malloc-backed NumPy, so summing double-counts and `max` discards
   the pymalloc side. Report them separately.
3. **Reduce across stages per target, then percentile.** Never the reverse. A
   quantile does not distribute over a sum (per-stage medians 1/10/100 sum to
   111 and no target costs 111), and for `max` there is no statistic at which
   the reverse order is correct. `per_stage_cost` stops one step short of the
   reduce, and `per_target_cost` is defined as its reduction, so the null
   policy has exactly one definition.

**Not ported: `memory_rss`.** v3 kept it selectable, deprecated, to read
archived runs. It is degenerate at every stat (glibc heap reuse collapses a
per-stage RSS delta to one page after warmup) and NULL on every run carrying
the heap channel -- all of v5's. A v5 caller has nothing to read with it.

**Not a per-target cost: the LTD fit.** `fit_ms` / `fit_*_peak_bytes` are in
`run.json`, once per `(combo, fold)`, and amortise over the fold's TGs. They are
not in `targets.parquet` and do not enter anything here.

**Shortest-Ping has no cost.** It is not a combo -- no LTD, MTL or CTR, no
`targets.parquet` -- so there is nothing to measure. `load_cost_frame` refuses
it rather than drawing a free baseline.

Callers own the row denominator. `load_cost_frame` applies one `rows` filter
for every stage -- different denominators per stage is a real defect the
legacy v2 `plot_phase_*` scripts shipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules.classify import load_method_frame
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask

_BYTES_PER_MB = 1024 * 1024

#: The three instrumented pipeline stages, in execution order.
STAGES: tuple[str, str, str] = ("ltd", "mtl", "ctr")

#: The derived fourth mark: `per_target_cost`, the *correct* composition of the
#: three stages. Not a stage -- a `stage` column value, so a reader can check it
#: against its three siblings without reshaping.
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
    #: Prose name for a figure title or manifest; `key` reads as an identifier.
    title_noun: str


COST_SPECS: dict[str, CostSpec] = {
    "runtime": CostSpec(
        key="runtime",
        stage_cols=("ltd_ms", "mtl_ms", "ctr_ms"),
        reduce="sum",
        scale=1.0,
        unit="ms",
        axis_label="Runtime per TG (ms)",
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
        axis_label="Peak memory per TG (MB, tracemalloc)",
        title_noun="peak memory (tracemalloc)",
    ),
    "memory_heap": CostSpec(
        key="memory_heap",
        stage_cols=(
            "ltd_heap_peak_bytes",
            "mtl_heap_peak_bytes",
            "ctr_heap_peak_bytes",
        ),
        reduce="max",
        scale=1.0 / _BYTES_PER_MB,
        unit="MB",
        axis_label="Peak memory per TG (MB, libc heap)",
        title_noun="peak memory (libc heap)",
    ),
}

#: The memory channels, for a `--memory` choice. Runtime is not one.
MEMORY_SPECS: tuple[str, ...] = ("memory_heap", "memory_alloc")

#: `all` is the default: a method pays for a TG it gave up on, and the cost of
#: that attempt is real. `solved` is `status.solved_mask`.
COST_ROWS: tuple[str, ...] = ("all", "solved")

#: Valid `reduce` values. Named for the operation, not the channel.
REDUCERS: tuple[str, ...] = ("max", "sum")

#: The quantiles `cost_stats` reports. p5/p95 are the box figure's whiskers,
#: p25/p75 its hinges.
STAT_QUANTILES: dict[str, float] = {
    "p5": 5.0, "p25": 25.0, "p50": 50.0, "p75": 75.0, "p90": 90.0, "p95": 95.0,
}
STAT_KEYS: tuple[str, ...] = (*STAT_QUANTILES, "mean", "min", "max", "n")


def _scaled_stages(df: pd.DataFrame, spec: CostSpec) -> list[np.ndarray] | None:
    """Null-filled, `spec.scale`-scaled per-stage vectors, or None if the run
    was never instrumented.

    The single definition of the null policy, shared by `per_target_cost` and
    `per_stage_cost`.

    Nulls fill to 0 because a null stage never ran -- a skipped CTR on a
    FALLBACK row cost nothing. The one exception is the LTD column: LTD always
    runs, so if *it* is entirely null the run was not instrumented, which is
    absence of measurement rather than absence of work. That returns None so the
    caller can refuse to plot; filling it to 0 would manufacture a free method.
    """
    missing = [c for c in spec.stage_cols if c not in df.columns]
    if missing:
        raise MissingArtifactError(
            f"cost columns {missing} absent. The pre-c7ee30a schema named these "
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

    Shares `per_target_cost`'s null policy exactly and deliberately stops one
    step earlier: it does **not** reduce across stages. `per_target_cost(df,
    spec)` is this frame reduced row-wise by `spec.reduce`, and a test pins that
    identity so the two cannot drift.

    An uninstrumented run NaNs **every** stage, not just LTD. Real MTL/CTR
    columns beside a NaN LTD would read as a working measurement.
    """
    stages = _scaled_stages(df, spec)
    if stages is None:
        return pd.DataFrame({s: np.full(len(df), np.nan) for s in STAGES}, index=df.index)
    return pd.DataFrame(dict(zip(STAGES, stages)), index=df.index)


def per_target_cost(
    df: pd.DataFrame, spec: CostSpec, *, reduce: str | None = None
) -> np.ndarray:
    """One cost per target, in `spec.unit`.

    Null-fill each stage, reduce **across stages per target**, and only then let
    the caller percentile the result. An uninstrumented LTD column returns
    all-NaN so the caller can refuse to plot it.
    """
    reduce = reduce or spec.reduce
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {list(REDUCERS)}; got {reduce!r}")
    stages = _scaled_stages(df, spec)
    if stages is None:
        return np.full(len(df), np.nan)
    stack = np.vstack(stages)
    return stack.sum(axis=0) if reduce == "sum" else stack.max(axis=0)


def cost_stats(values: np.ndarray) -> dict[str, float]:
    """p5/p25/p50/p75/p90/p95/mean/min/max/n over the finite values.

    NaN in -> NaN out, and `n` is the count of finite values, which is what lets
    a caller see that a block is empty rather than zero."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out: dict[str, float] = {k: float("nan") for k in STAT_KEYS if k != "n"}
        out["n"] = 0
        return out
    out = {k: float(np.percentile(arr, q)) for k, q in STAT_QUANTILES.items()}
    out["mean"] = float(arr.mean())
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["n"] = int(arr.size)
    return out


def require_measured(stats: dict[str, float], spec: CostSpec, *, run_id: str, method: str) -> None:
    """Raise when a channel's columns are present but hold no measurements.

    `_scaled_stages` NaNs an uninstrumented run so a figure can *refuse* rather
    than draw a box for something never measured. This is the refusal.
    """
    if stats.get("n", 0) > 0 and np.isfinite(stats.get("p50", np.nan)):
        return
    alt = "memory_heap" if spec.key != "memory_heap" else "memory_alloc"
    raise MissingArtifactError(
        f"{run_id}/{method}: cost channel {spec.key!r} has no measurements -- "
        f"{list(spec.stage_cols)} are present but entirely NULL. Try {alt}."
    )


def load_cost_frame(
    run: RunPaths, method: str, specs: tuple[CostSpec, ...], *, rows: str = "all"
) -> pd.DataFrame:
    """One row per evaluated TG for `method`, with every spec's raw stage columns.

    Every spec is read in **one** load and filtered by **one** `rows` mask, so
    runtime and memory share a denominator by construction. The columns are
    unscaled and still nullable -- pair with `per_stage_cost` /
    `per_target_cost` for the scaled, null-filled views.

    A TG id seen twice means the K-fold test sets overlap; v3's `load_folds`
    refused that and so does this, since the duplicate would count twice.
    """
    if rows not in COST_ROWS:
        raise ValueError(f"rows must be one of {list(COST_ROWS)}; got {rows!r}")
    if method == SHORTEST_PING:
        raise ValueError(
            f"{SHORTEST_PING} has no cost: it is not a combo (no LTD/MTL/CTR, no "
            f"targets.parquet), so there is nothing measured to draw."
        )
    cols = tuple(dict.fromkeys(c for spec in specs for c in spec.stage_cols))
    df = load_method_frame(run, method, columns=cols)
    dup = df["tg_id"].duplicated()
    if dup.any():
        raise ValueError(
            f"{run.run_id}/{method}: {int(dup.sum())} TG ids appear in more than one "
            f"fold (e.g. {df.loc[dup, 'tg_id'].head(3).tolist()}); the K-fold test "
            f"sets overlap."
        )
    if rows == "solved":
        df = df[solved_mask(df)]
    return df.reset_index(drop=True)


def stage_cost_table(df: pd.DataFrame, spec: CostSpec) -> dict[str, dict[str, float]]:
    """Per-stage *and* pipeline stats over one frame: `{ltd, mtl, ctr, pipeline}`.

    The three stage blocks are comparable marginals over one target set; their
    sum is not a statistic of anything. `pipeline` is the correct composition,
    computed per target before any percentile. Each stage block also carries
    `n_null`, the rows filled to 0 -- what makes a low CTR p50 legible as
    structural zeros on FALLBACK rows rather than a fast CTR.
    """
    stages = per_stage_cost(df, spec)
    out: dict[str, dict[str, float]] = {}
    for stage, col in zip(STAGES, spec.stage_cols):
        block = cost_stats(stages[stage].to_numpy())
        block["n_null"] = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
        out[stage] = block
    out[PIPELINE] = cost_stats(per_target_cost(df, spec))
    out[PIPELINE]["n_null"] = 0
    return out
