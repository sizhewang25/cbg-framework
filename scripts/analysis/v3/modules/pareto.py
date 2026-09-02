"""Cost/accuracy Pareto analysis — `plot-pareto`.

The v3 layer already answers *how accurate* each CBG variant is
(`classify` -> `topn_accuracy.csv`). This module answers what that accuracy
costs, by joining it to the per-target runtime and memory the benchmark already
records in `targets.parquet` (`{ltd,mtl,ctr}_{ms,alloc_peak_bytes,rss_peak_bytes}`
— see ../SCHEMA.md §3).

**Cross-dataset by construction.** A single run's accuracy number hides how
much a variant's accuracy moves between datasets, which on the operator runs is
the dominant effect: `spotter_cbg` varies 6pp across as01/02/03 while
`million_scale_cbg` swings 30pp. So every method is drawn as a *vertical spread*
over the datasets' accuracy scalars rather than one point, and
`accuracy_range` is a first-class column of the emitted CSV.

Four cost-model policies this module exists to enforce, each measured rather
than assumed (see `README.md` for the numbers):

1. **Cost is over every target, not just solved ones** (`cost_rows="all"`).
   `ctr_ms` is null on exactly the FALLBACK rows, so a solved-only cost is
   ~22% higher on `vanilla_cbg` — and `accuracy_topN`'s denominator is every
   target, so the cost denominator must be too.
2. **Memory reduces across stages with `max`, not `sum`.** `instrument.py`
   resets tracemalloc *inside* each stage and the RSS sampler returns a delta,
   so the columns are per-stage peaks: `max` is the pipeline high-water mark,
   `sum` the no-release upper bound. Summing also triples the ~41.6 KB
   tracemalloc pedestal, which is bookkeeping rather than work. Runtime
   genuinely sums.
3. **`memory_rss` is degenerate at p50** — the sampler is 5 ms, so fast stages
   floor at one 4096-byte page — but graded at p95. Hence `cost_stat`.
4. **Costs can cluster inside their own measurement floor.** On the memory
   axis the three heavy variants sit within 160 bytes of each other at
   ~24.09 MB, so their x ordering is noise. The figure shows each variant's
   cost *range* rather than only its median for exactly that reason — read the
   horizontal extent, not the ordering, when the marks overlap.

Shortest-Ping is scored through the same answer space as every variant
(`classify.score_shortest_ping`) but is analytical — it has no row in any
`targets.parquet` and runs no geometry. Its cost is therefore recorded as
exactly 0, in `shortest_ping_row` alone, so the policy is auditable in one place.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.classify import SHORTEST_PING, TOPN_CSV
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    MissingArtifactError,
    RunPaths,
    grid_slug,
)
from scripts.analysis.v3.modules.venn import PREFERRED_ORDER, label_for


def short_label(method: str) -> str:
    """`venn.label_for` with the trailing " CBG" dropped.

    Every method but Shortest-Ping is a CBG variant and the figure says so, so
    repeating it up to 16 times only makes the labels wide enough to collide.
    """
    label = label_for(method)
    return label[: -len(" CBG")] if label.endswith(" CBG") else label

# ---------------------------------------------------------------------------
# § cost model
# ---------------------------------------------------------------------------

_BYTES_PER_MB = 1024 * 1024


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

DEFAULT_COST = "runtime"
DEFAULT_COST_STAT = "p50"
COST_STATS: tuple[str, ...] = ("p50", "p75", "p90", "p95", "mean")
COST_ROWS: tuple[str, ...] = ("all", "solved")
MEMORY_REDUCERS: tuple[str, ...] = ("max", "sum")

_STAT_QUANTILE = {"p50": 50.0, "p75": 75.0, "p90": 90.0, "p95": 95.0}


def per_target_cost(
    df: pd.DataFrame, spec: CostSpec, *, reduce: str | None = None
) -> np.ndarray:
    """One cost per target, in `spec.unit`.

    Order matters and is the one thing this function exists to pin: null-fill
    each stage, reduce **across stages per target**, and only then let the
    caller percentile the result. `plot_phase_runtime._total_runtime_stat`
    documents why for sums (a median does not distribute over a sum); the same
    holds for `max`.

    Nulls fill to 0 because a null stage never ran — a skipped CTR on a FALLBACK
    row cost nothing. The one exception is the LTD column, which
    `TARGETS_SCHEMA` declares non-nullable precisely because LTD always runs: if
    *it* is entirely null the run was not instrumented, which is absence of
    measurement rather than absence of work. That returns all-NaN so the caller
    can refuse to plot it; filling it to 0 would manufacture a free method that
    then dominates the entire frontier.
    """
    reduce = reduce or spec.reduce
    if reduce not in MEMORY_REDUCERS:
        raise ValueError(f"reduce must be one of {list(MEMORY_REDUCERS)}; got {reduce!r}")
    missing = [c for c in spec.stage_cols if c not in df.columns]
    if missing:
        raise MissingArtifactError(
            f"cost columns {missing} absent. The pre-{'c7ee30a'} schema named these "
            f"`{{ltd,mtl,ctr}}_peak_bytes`; re-run the benchmark to get the "
            f"dual-channel columns this analysis needs."
        )

    stages: list[np.ndarray] = []
    for idx, col in enumerate(spec.stage_cols):
        s = pd.to_numeric(df[col], errors="coerce")
        if idx == 0 and s.isna().all():
            return np.full(len(df), np.nan)
        stages.append(s.fillna(0.0).to_numpy(dtype=float))

    stack = np.vstack(stages) if stages else np.zeros((1, len(df)))
    total = stack.sum(axis=0) if reduce == "sum" else stack.max(axis=0)
    return total * spec.scale


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


def combo_cost(
    run: RunPaths,
    combo_id: str,
    spec: CostSpec,
    *,
    rows: str = "all",
    reduce: str | None = None,
) -> tuple[dict[str, float], int]:
    """Cost stats for one combo, pooled over folds. Returns (stats, n_rows_used).

    `columns` is passed explicitly so the nested `ltd_predictions` /
    `mtl_participants` columns are never read (they dominate the file size) while
    `load_folds`' K-fold disjointness check still runs.
    """
    if rows not in COST_ROWS:
        raise ValueError(f"rows must be one of {list(COST_ROWS)}; got {rows!r}")
    cols = ["target_id", "status", *spec.stage_cols]
    df = io.load_folds(run, combo_id, columns=cols)
    if rows == "solved":
        df = df[df["status"].isin(io.CBG_SUCCESS_STATUSES)]
    return cost_stats(per_target_cost(df, spec, reduce=reduce)), len(df)


# ---------------------------------------------------------------------------
# § accuracy input
# ---------------------------------------------------------------------------

ACC_PREFIX = "accuracy_top"


def available_top_ns(df: pd.DataFrame) -> list[int]:
    """The Ns a `topn_accuracy.csv` actually carries, ascending."""
    ns = []
    for c in df.columns:
        if c.startswith(ACC_PREFIX):
            suffix = c[len(ACC_PREFIX) :]
            if suffix.isdigit():
                ns.append(int(suffix))
    return sorted(ns)


def load_accuracy(path: Path, *, top_n: int) -> pd.DataFrame:
    """`topn_accuracy.csv` restricted to one N, indexed by method.

    An explicit input (mirroring `classify`'s `--answer-space`) so the figure is
    provably the same number as the published accuracy table rather than a
    re-derivation that could drift from it. `accuracy_topN` already counts
    FALLBACK rows as failures (paper §7.2); this module does not re-apply that
    policy.
    """
    path = Path(path)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `cli classify` for this run/grid")
    df = pd.read_csv(path)
    col = f"{ACC_PREFIX}{top_n}"
    if col not in df.columns:
        have = available_top_ns(df)
        raise MissingArtifactError(
            f"{path} has no {col!r}; it reports top-N {have}. `classify` only writes "
            f"the Ns it was asked for — re-run with `classify --topn "
            f"{','.join(str(n) for n in sorted(set(have) | {top_n}))}`."
        )
    dup = df["method"].duplicated()
    if dup.any():
        raise ValueError(
            f"{path}: method(s) {df.loc[dup, 'method'].tolist()} appear more than once; "
            f"which row is the accuracy is ambiguous"
        )
    if df["n_targets"].nunique() > 1:
        warnings.warn(
            f"{path}: n_targets differs across methods "
            f"({sorted(df['n_targets'].unique().tolist())}); accuracies over different "
            f"denominators are not comparable on one frontier",
            stacklevel=2,
        )
    keep = ["method", "n_targets", "n_solved", "n_fallback", "fallback_rate", col]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.rename(columns={col: "accuracy"}).set_index("method")


# ---------------------------------------------------------------------------
# § baseline policy — the one place Shortest-Ping's cost is decided
# ---------------------------------------------------------------------------


def shortest_ping_row(run_id: str, accuracy_row: pd.Series, spec: CostSpec) -> dict:
    """Shortest-Ping's row, with every cost field exactly 0.

    Shortest-Ping has no row in any `targets.parquet`: its estimate is the
    lowest-RTT VP's own coordinate, read from `eval_source` by
    `classify.score_shortest_ping`. It runs no distance model, no
    multilateration and no centroid step — the whole method is an `argmin` over
    RTTs the measurement already produced — so it is charged nothing on either
    cost axis.

    That makes it the frontier's anchor at `(0, its accuracy)`, which is the
    point: a CBG variant earns a place on the frontier only by being *more
    accurate than free*. Verified non-degenerate on the operator runs — four CBG
    variants survive at top-1, two at top-3.
    """
    return {
        "run_id": run_id,
        "method": SHORTEST_PING,
        "label": label_for(SHORTEST_PING),
        "is_baseline": True,
        "accuracy": float(accuracy_row["accuracy"]),
        "n_targets": int(accuracy_row.get("n_targets", 0) or 0),
        "n_fallback": 0,
        "fallback_rate": 0.0,
        "cost": 0.0,
        "cost_p50": 0.0,
        "cost_p95": 0.0,
        "cost_min": 0.0,
        "cost_max": 0.0,
        "cost_n": int(accuracy_row.get("n_targets", 0) or 0),
        "cost_basis": "analytical",
        "fit_ms": 0.0,
        "fit_ms_per_target": 0.0,
        "n_targets_per_fold": float("nan"),
    }


# ---------------------------------------------------------------------------
# § assembly
# ---------------------------------------------------------------------------


def load_cost_accuracy(
    accuracy_csvs: dict[str, Path],
    runs: dict[str, RunPaths],
    *,
    top_n: int,
    spec: CostSpec,
    cost_stat: str = DEFAULT_COST_STAT,
    cost_rows: str = "all",
    reduce: str | None = None,
    amortize_fit: bool = False,
    methods: list[str] | None = None,
    strict_membership: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """The unified loader: one row per `(run_id, method)` across every dataset.

    Accuracy comes from each dataset's own `topn_accuracy.csv`; cost from that
    dataset's `targets.parquet`. Returns the long frame plus the list of skip
    notes, which the manifest records so a missing method is never silent.
    """
    if cost_stat not in COST_STATS:
        raise ValueError(f"cost_stat must be one of {list(COST_STATS)}; got {cost_stat!r}")
    if amortize_fit and spec.key != "runtime":
        raise ValueError(
            "--amortize-fit applies to runtime only: a one-time fit's *peak* memory "
            "is not additive across targets, so amortizing it per target would be "
            "meaningless."
        )

    rows: list[dict] = []
    notes: list[str] = []
    for run_id, csv_path in accuracy_csvs.items():
        run = runs[run_id]
        acc = load_accuracy(csv_path, top_n=top_n)
        cfg = io.load_run_configs(run)
        on_disk = set(run.combo_ids)

        wanted = list(acc.index) if methods is None else [m for m in methods if m in acc.index]
        if methods is not None:
            for m in methods:
                if m not in acc.index:
                    notes.append(f"{run_id}: {m!r} requested but absent from {csv_path.name}")

        for method in wanted:
            arow = acc.loc[method]
            if method == SHORTEST_PING:
                rows.append(shortest_ping_row(run_id, arow, spec))
                continue
            if method not in on_disk:
                msg = (
                    f"{run_id}: {method!r} is in {csv_path.name} but has no combo dir on "
                    f"disk, so it has no cost — skipped"
                )
                if strict_membership:
                    raise MissingArtifactError(msg)
                warnings.warn(msg, stacklevel=2)
                notes.append(msg)
                continue

            try:
                stats, n_used = combo_cost(
                    run, method, spec, rows=cost_rows, reduce=reduce
                )
            except MissingArtifactError as exc:
                raise MissingArtifactError(f"{run_id}/{method}: {exc}") from exc

            sub = cfg[cfg["combo_id"] == method] if "combo_id" in cfg.columns else cfg.iloc[:0]
            fit_ms = float(sub["fit_ms"].mean()) if len(sub) else float("nan")
            n_per_fold = float(sub["n_targets"].mean()) if len(sub) else float("nan")
            fit_per_target = (
                fit_ms / n_per_fold
                if np.isfinite(fit_ms) and np.isfinite(n_per_fold) and n_per_fold > 0
                else float("nan")
            )

            cost = stats[cost_stat]
            if amortize_fit and np.isfinite(fit_per_target):
                cost = cost + fit_per_target

            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "label": label_for(method),
                    "is_baseline": False,
                    "accuracy": float(arow["accuracy"]),
                    "n_targets": int(arow.get("n_targets", 0) or 0),
                    "n_fallback": int(arow.get("n_fallback", 0) or 0),
                    "fallback_rate": float(arow.get("fallback_rate", float("nan"))),
                    "cost": float(cost),
                    "cost_p50": stats["p50"],
                    "cost_p95": stats["p95"],
                    "cost_min": stats["min"],
                    "cost_max": stats["max"],
                    "cost_n": stats["n"],
                    "cost_basis": "instrumented"
                    if np.isfinite(stats["p50"])
                    else "unmeasured",
                    "fit_ms": fit_ms,
                    "fit_ms_per_target": fit_per_target,
                    "n_targets_per_fold": n_per_fold,
                }
            )

    if not rows:
        raise MissingArtifactError(
            "no (dataset, method) pair had both an accuracy row and cost data"
        )
    return pd.DataFrame(rows), notes


def aggregate_methods(long: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Collapse the long frame to one row per method.

    Accuracy becomes a *spread* over datasets — `accuracy_range` is the
    cross-dataset stability metric this whole figure exists to show — and cost
    becomes its median across datasets, since cost varies too (`octant_cbg_hull`
    ranges 1875-5027 ms). The per-dataset accuracies survive as their own
    columns, which is what the figure's one-line-per-dataset encoding plots.

    `partial_coverage` marks a method that is missing from at least one dataset:
    it is still plotted, but the affected dataset's polyline skips it rather
    than interpolating across a gap it never measured.
    """
    n_datasets = long["run_id"].nunique()
    out_rows: list[dict] = []
    for method, g in long.groupby("method", sort=False):
        accs = g["accuracy"].to_numpy(dtype=float)
        costs = g["cost"].to_numpy(dtype=float)
        row = {
            "method": method,
            "label": g["label"].iloc[0],
            "is_baseline": bool(g["is_baseline"].iloc[0]),
            "n_datasets": int(g["run_id"].nunique()),
            "partial_coverage": bool(g["run_id"].nunique() < n_datasets),
        }
        for run_id, gr in g.groupby("run_id", sort=True):
            row[f"{ACC_PREFIX}{top_n}_{run_id}"] = float(gr["accuracy"].iloc[0])
        row.update(
            accuracy_median=float(np.median(accs)),
            accuracy_min=float(np.min(accs)),
            accuracy_max=float(np.max(accs)),
            accuracy_range=float(np.max(accs) - np.min(accs)),
            cost=float(np.median(costs)),
            cost_min=float(np.min(costs)),
            cost_max=float(np.max(costs)),
            cost_p95_max=float(g["cost_p95"].max()),
            cost_basis=g["cost_basis"].iloc[0],
            fit_ms_mean=float(g["fit_ms"].mean()),
            fit_ms_per_target=float(g["fit_ms_per_target"].mean()),
            n_targets_per_fold=float(g["n_targets_per_fold"].mean()),
        )
        out_rows.append(row)

    wide = pd.DataFrame(out_rows)

    # Cost order is the figure's reading order (cheap on the left), so the CSV
    # rows and the plotted x positions agree.
    return wide.sort_values("cost", kind="stable", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# § figure
# ---------------------------------------------------------------------------

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Fixed variant -> hue, assigned by **identity** (`venn.PREFERRED_ORDER`) and
#: never by cost rank, so `--method` cannot repaint the survivors.
#:
#: Validated with the dataviz skill's `validate_palette.js` against the
#: reference 8-hue categorical theme, `--pairs all` on white — the right check
#: here, since every variant is visible at once. Every 6/7/8-slot prefix of that
#: theme FAILS (green vs orange is dE 3.2 under protanopia), and an exhaustive
#: search over its hues found exactly two passing 6-subsets; this is the better
#: one, worst dE 6.9 (deutan) / 7.6 (tritan). Orange is the hue that had to go.
#:
#: dE 6.9 sits in the 6-8 band that is legal *only* with secondary encoding.
#: That is satisfied three times over: each variant owns its own x column (they
#: never interleave spatially), the legend names every one, and the CSV is the
#: table view. Aqua/yellow/magenta are also below 3:1 on white (2.82/2.17/2.69),
#: a contrast WARN that obliges visible labels or a table view — the legend and
#: CSV again. No 6-subset of this theme clears 3:1 for all six (only five hues
#: do), so at six variants that is unavoidable rather than a shortcut; the 2 px
#: cost line and ringed >=8 px markers give each variant more ink than a dot.
_VARIANT_HUES: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Past the palette's capacity a 9th series is never a generated hue — it folds
#: into one "other" bucket. as7018 carries 11 ablation arms on top of the five
#: published variants, and they belong in that bucket.
_C_OTHER = "#898781"

_C_LINE = "#898781"  # the per-dataset polylines: neutral, because hue is taken
_C_GRID = "#e1e0d9"
_C_AXIS = "#c3c2b7"
_C_INK = "#0b0b0b"
_C_INK_2 = "#52514e"
_C_MUTED = "#898781"
_SURFACE = "#ffffff"

#: Shape *and* linestyle carry the dataset, so a polyline and its vertices agree
#: and three overlapping grey lines stay traceable. Cycled in sorted run order
#: so a dataset keeps its symbol across every figure in a sweep.
_DATASET_MARKERS = ("o", "s", "^", "D", "v", "P", "X")
_DATASET_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)))


def _short_dataset(run_id: str) -> str:
    """`as01-260728-260802` -> `as01`; anything else unchanged.

    The date range is identical across the runs being compared (it is what
    makes them comparable), so printing it three times in a legend costs width
    and carries no information. Only a trailing all-numeric tail is stripped,
    so `as7018_us_test01` survives intact.
    """
    parts = run_id.split("-")
    if len(parts) > 1 and all(p.isdigit() for p in parts[1:]):
        return parts[0]
    return run_id


def _build_label_hues() -> dict[str, str]:
    """Display label -> hue, fixed once from `venn.PREFERRED_ORDER`.

    Keyed on the *label* rather than the combo id so `octant_cbg_spl` and
    `octant_cbg` land on one hue: they are one paper variant whose id differs
    per run, which is why `venn.LABELS` already maps both onto
    "Octant-Spline CBG". Two hues would invent a distinction the runs do not
    contain. That aliasing is also what makes the six published variants fit the
    six validated hues exactly.
    """
    hues: dict[str, str] = {}
    for method in PREFERRED_ORDER:
        label = label_for(method)
        if label in hues:
            continue
        if len(hues) < len(_VARIANT_HUES):
            hues[label] = _VARIANT_HUES[len(hues)]
    return hues


#: Computed once, at import, from a constant order — never from the methods
#: present in a given call. This is what makes colour stable under `--method`.
_LABEL_HUES: dict[str, str] = _build_label_hues()


def method_colors(methods) -> dict[str, str]:
    """Variant -> hue, stable under filtering.

    Each hue is pinned to a variant's *identity* via `_LABEL_HUES`, which is
    built from a fixed order at import time. Filtering the method pool with
    `--method` therefore cannot repaint the survivors — colour follows the
    entity, never its rank in the current selection, and the same variant is
    the same colour in every figure of a sweep.

    Anything `venn.PREFERRED_ORDER` does not name — as7018's 11 ablation arms —
    folds into the single `_C_OTHER` bucket rather than being handed a generated
    hue, because no palette distinguishes 17 series.
    """
    return {m: _LABEL_HUES.get(label_for(m), _C_OTHER) for m in methods}


def dataset_lines(
    wide: pd.DataFrame, long: pd.DataFrame
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per dataset, its `(cost, accuracy)` polyline in ascending cost order.

    Each variant contributes one vertex, at the variant's median cost (shared by
    every dataset) and that dataset's own accuracy. Connecting them turns each
    dataset into its own cost/accuracy curve, so a reader can check whether the
    ranking holds per dataset rather than only in the median.

    A variant a dataset never measured is **skipped** for that dataset only —
    its neighbours join directly. Interpolating across the gap would draw a
    measurement that does not exist.
    """
    cost_of = dict(zip(wide["method"], wide["cost"]))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run_id, g in long.groupby("run_id", sort=True):
        pts = [
            (cost_of[m], a)
            for m, a in zip(g["method"], g["accuracy"])
            if m in cost_of and np.isfinite(cost_of[m]) and np.isfinite(a)
        ]
        pts.sort(key=lambda t: t[0])
        if not pts:
            out[run_id] = (np.array([]), np.array([]))
            continue
        xs, ys = zip(*pts)
        out[run_id] = (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    return out


def plot_pareto(
    wide: pd.DataFrame,
    long: pd.DataFrame,
    out_path: Path,
    *,
    spec: CostSpec,
    top_n: int,
    subtitle: str,
    title: str | None = None,
    x_label: str | None = None,
) -> Path:
    """Accuracy vs cost: colour per CBG variant, symbol per dataset.

    Three encodings, each carrying exactly one thing:

    * **Colour** is the variant. Every variant sits at one x — its median cost
      across the datasets — with a horizontal line spanning the cost range it
      actually occupied.
    * **Symbol** is the dataset. One marker per dataset at that variant's x,
      placed at the accuracy the variant scored *on that dataset*, so the
      vertical scatter within a colour is its cross-dataset stability.
    * **A grey polyline per dataset** connects that dataset's markers in cost
      order, giving each dataset its own cost/accuracy curve. Top-left is the
      most cost-effective corner.

    The x axis is `symlog`: Shortest-Ping's cost is exactly 0, which a log axis
    cannot render, and symlog keeps one continuous axis rather than splitting the
    figure across a break.
    """
    from matplotlib.lines import Line2D

    datasets = sorted(long["run_id"].unique())
    marker_of = {d: _DATASET_MARKERS[i % len(_DATASET_MARKERS)] for i, d in enumerate(datasets)}
    style_of = {d: _DATASET_LINESTYLES[i % len(_DATASET_LINESTYLES)] for i, d in enumerate(datasets)}
    color_of = method_colors(list(wide["method"]))

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ax.set_facecolor(_SURFACE)

    # --- layer 1: one polyline per dataset ---------------------------------
    for run_id, (xs, ys) in dataset_lines(wide, long).items():
        if len(xs) < 2:
            continue
        ax.plot(
            xs, ys, color=_C_LINE, linestyle=style_of[run_id], linewidth=1.2,
            alpha=0.9, zorder=2, solid_capstyle="round",
        )

    # --- layer 2: per-variant cost line + per-dataset markers --------------
    acc_by_method = {m: g for m, g in long.groupby("method", sort=False)}
    for _, row in wide.iterrows():
        x = row["cost"]
        if not np.isfinite(x):
            continue
        color = color_of.get(row["method"], _C_OTHER)

        lo, hi = row["cost_min"], row["cost_max"]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot(
                [lo, hi], [row["accuracy_median"]] * 2,
                color=color, linewidth=2.0, alpha=0.85, zorder=4,
                solid_capstyle="butt",
            )

        for _, d in acc_by_method.get(row["method"], long.iloc[:0]).iterrows():
            ax.plot(
                [x], [d["accuracy"]],
                marker=marker_of[d["run_id"]], markersize=9,
                color=color, markeredgecolor=_SURFACE, markeredgewidth=2.0,
                linestyle="none", zorder=6,
            )

    # --- axes --------------------------------------------------------------
    costs = pd.to_numeric(wide["cost"], errors="coerce")
    positive = costs[np.isfinite(costs) & (costs > 0)]
    if len(positive):
        lt = float(positive.min())
        ax.set_xscale("symlog", linthresh=lt, linscale=0.6)
        ax.set_xlim(left=-lt * 0.12, right=float(positive.max()) * 1.9)
    ax.set_xlabel(
        x_label or f"{spec.axis_label}  —  cheaper → left",
        fontsize=11, color=_C_INK_2,
    )
    ax.set_ylabel(f"Top-{top_n} classification accuracy", fontsize=11, color=_C_INK_2)
    ax.grid(True, which="major", color=_C_GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_C_MUTED, labelsize=9)

    # --- legends: colour = variant, symbol + linestyle = dataset -----------
    seen: set[str] = set()
    variant_handles: list[Line2D] = []
    for method in [m for m in PREFERRED_ORDER if m in set(wide["method"])] + [
        m for m in wide["method"] if m not in PREFERRED_ORDER
    ]:
        hue = color_of.get(method, _C_OTHER)
        if hue == _C_OTHER:
            continue
        label = short_label(method)
        if label in seen:  # the octant-spline alias shares a hue and a row
            continue
        seen.add(label)
        variant_handles.append(
            Line2D([], [], color=hue, linewidth=2.0, marker="o", markersize=8,
                   markeredgecolor=_SURFACE, markeredgewidth=2.0, label=label)
        )
    n_other = sum(1 for m in wide["method"] if color_of.get(m) == _C_OTHER)
    if n_other:
        variant_handles.append(
            Line2D([], [], color=_C_OTHER, linewidth=2.0, marker="o", markersize=8,
                   markeredgecolor=_SURFACE, markeredgewidth=2.0,
                   label=f"other ({n_other}, see CSV)")
        )

    leg1 = ax.legend(
        handles=variant_handles, loc="upper left", bbox_to_anchor=(0.0, -0.10),
        fontsize=8.5, frameon=False, ncol=min(len(variant_handles), 4),
        title="CBG variant", alignment="left",
    )
    leg1.get_title().set_fontsize(8.5)
    leg1.get_title().set_color(_C_INK_2)
    ax.add_artist(leg1)

    dataset_handles = [
        Line2D([], [], color=_C_LINE, linestyle=style_of[d], linewidth=1.2,
               marker=marker_of[d], markersize=8, markerfacecolor=_C_MUTED,
               markeredgecolor=_SURFACE, markeredgewidth=2.0, label=_short_dataset(d))
        for d in datasets
    ]
    leg2 = ax.legend(
        handles=dataset_handles, loc="upper right", bbox_to_anchor=(1.0, -0.10),
        fontsize=8.5, frameon=False, ncol=min(len(datasets), 4),
        title="Dataset (symbol + line)", alignment="right",
    )
    leg2.get_title().set_fontsize(8.5)
    leg2.get_title().set_color(_C_INK_2)

    fig.suptitle(
        title or f"Top-{top_n} accuracy vs {spec.title_noun}",
        fontsize=13, fontweight="bold", color=_C_INK,
    )
    ax.annotate(
        subtitle,
        xy=(0.5, -0.30), xycoords="axes fraction", ha="center", va="top",
        fontsize=8, color=_C_MUTED, wrap=True,
    )

    fig.subplots_adjust(bottom=0.32, top=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# § output naming
# ---------------------------------------------------------------------------

#: Cross-dataset artifacts have no per-run home (`RunPaths.analysis_dir` is
#: run-scoped), so they get a sibling directory. The leading underscore means it
#: can never collide with a `run_id`.
CROSS_DIRNAME = "_cross"


def dataset_set_slug(run_ids) -> str:
    """A stable directory name for one *set* of datasets.

    The dataset set is a parameter of every number in these artifacts, so it
    has to appear in the path — the cost channel, grid, top-N and fit policy are
    all in the filename, but without this a run over `as7018_us_test01` writes
    the same `pareto_runtime.healpix-128.top1.csv` as a run over as01+as02+as03
    and silently replaces it. Long sets are truncated and hashed so the name
    stays a usable directory while still being unique.
    """
    import hashlib

    short = sorted(_short_dataset(r) for r in run_ids)
    slug = "+".join(short)
    if len(slug) <= 60:
        return slug
    digest = hashlib.sha1("+".join(sorted(run_ids)).encode()).hexdigest()[:8]
    return f"{len(short)}sets-{digest}"


def cross_dir(analysis_root: Path | None = None, run_ids=None) -> Path:
    base = (analysis_root or DEFAULT_ANALYSIS_ROOT) / CROSS_DIRNAME / "cost-accuracy"
    if run_ids is not None:
        base = base / dataset_set_slug(run_ids)
    base.mkdir(parents=True, exist_ok=True)
    return base


def artifact_name(
    *,
    cost_key: str,
    grid: str,
    resolution: int,
    top_n: int,
    ext: str,
    amortized: bool = False,
    x: str = "cost",
) -> str:
    """`pareto_runtime.healpix-128.top1.csv`.

    Four axes parameterize these numbers and every one of them is in the name,
    because any missing axis means a sweep silently overwrites its own output:
    the cost channel and x-presentation in the stem, the quantization via
    `grid_slug`, the top-N as a suffix (`venn.artifact_name`'s convention), and
    the fit policy as an optional marker.
    """
    stem = f"pareto_{cost_key}" + ("-throughput" if x == "throughput" else "")
    parts = [stem, grid_slug(grid, resolution), f"top{top_n}"]
    if amortized:
        parts.append("amortized")
    return ".".join(parts) + f".{ext}"


def to_throughput(wide: pd.DataFrame, long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remap cost (ms) -> throughput (targets/s) for presentation only.

    `1000/cost` is monotone decreasing, so every variant's *ordering* is simply
    mirrored and each dataset's polyline keeps its shape, read right-to-left.

    A zero cost is *infinite* throughput, which no finite axis can place, so it
    becomes NaN and drops out of the figure — Shortest-Ping is therefore absent
    from a throughput view. It keeps its exact `inf` in the CSV's
    `throughput_per_s`, so the information is not lost, and the caller warns.
    Mapping it to `inf` here instead would propagate into the axis limits and
    raise.
    """
    def conv(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ("cost", "cost_min", "cost_max"):
            if col in out.columns:
                with np.errstate(divide="ignore"):
                    out[col] = np.where(out[col] > 0, 1000.0 / out[col], np.nan)
        if {"cost_min", "cost_max"} <= set(out.columns):
            lo = np.minimum(out["cost_min"], out["cost_max"])
            hi = np.maximum(out["cost_min"], out["cost_max"])
            out["cost_min"], out["cost_max"] = lo, hi
        return out

    return conv(wide), conv(long)


#: Column order — this *is* the CSV contract.
_CSV_ORDER = (
    "method", "label", "is_baseline", "n_datasets", "partial_coverage",
    "accuracy_median", "accuracy_min", "accuracy_max", "accuracy_range",
    "cost", "cost_min", "cost_max", "cost_p95_max", "cost_basis",
    "throughput_per_s",
    "fit_ms_mean", "fit_ms_per_target", "n_targets_per_fold", "cost_includes_fit",
)


def order_columns(wide: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """`_CSV_ORDER`, with the per-dataset accuracy columns kept after `label`."""
    per_ds = sorted(c for c in wide.columns if c.startswith(f"{ACC_PREFIX}{top_n}_"))
    head = ["method", "label", "is_baseline", "n_datasets", "partial_coverage"]
    rest = [c for c in _CSV_ORDER if c not in head and c in wide.columns]
    extra = [c for c in wide.columns if c not in head + per_ds + rest]
    return wide[head + per_ds + rest + extra]


# ---------------------------------------------------------------------------
# § CLI
# ---------------------------------------------------------------------------

def _guard_one_setup(runs: dict[str, RunPaths], *, allow_mixed: bool) -> None:
    """Refuse to pool `anchors_to_probes` with `probes_to_anchors`.

    SCHEMA.md §7: the two run families swap the VP and target roles, so their
    costs and accuracies describe different experiments. Pooling them onto one
    frontier would compare a 134-VP fleet against a 53-VP one as though the
    difference were the variant's.
    """
    setups = sorted({r.setup for r in runs.values()})
    if len(setups) > 1 and not allow_mixed:
        import typer

        by_setup = {
            s: sorted(rid for rid, r in runs.items() if r.setup == s) for s in setups
        }
        raise typer.BadParameter(
            f"selected runs span {len(setups)} setups: "
            + "; ".join(f"{s} = {v}" for s, v in by_setup.items())
            + ". These swap the VP/target roles (SCHEMA.md §7), so one frontier over "
            "both would not be a like-for-like comparison. Pass --allow-mixed-setups "
            "to override, or select runs from one setup."
        )


def register(app) -> None:
    import typer

    from scripts.analysis.v3.modules.grid import (
        DEFAULT_GRID,
        GRID_HELP,
        RESOLUTION_HELP,
        SWEEP_HELP,
        resolve_cli_grid,
    )
    from scripts.analysis.v3.modules.paths import (
        DEFAULT_OUTPUTS_ROOT,
        discover_runs,
        resolve_run,
    )

    @app.command("plot-pareto")
    def plot_pareto_cmd(
        run_id: list[str] = typer.Option(
            None, "--run-id",
            help="Dataset to include (repeatable). Omit with --all-runs.",
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Include every run under --outputs-root."
        ),
        accuracy_csv: list[Path] = typer.Option(
            None, "--accuracy-csv",
            help="topn_accuracy.csv per run, in --run-id order (repeatable). "
                 "Defaults to each run's target-cls-accuracy/<grid>-<res>/ copy.",
        ),
        top_n: int = typer.Option(1, "--top-n", help="N for accuracy_top<N>."),
        cost: str = typer.Option(
            DEFAULT_COST, "--cost", help=f"Cost axis: {list(COST_SPECS)}."
        ),
        cost_stat: str = typer.Option(
            DEFAULT_COST_STAT, "--cost-stat",
            help=f"Per-target statistic: {list(COST_STATS)}. memory_rss is "
                 "degenerate at p50 (4 KB page floor) — use p95 there.",
        ),
        cost_rows: str = typer.Option(
            "all", "--cost-rows",
            help="Which target rows the cost is over. 'all' matches "
                 "accuracy_topN's denominator; 'solved' drops FALLBACK rows and "
                 "reads ~22% higher on variants that fall back.",
        ),
        memory_reduce: str = typer.Option(
            "max", "--memory-reduce",
            help="How memory combines across LTD/MTL/CTR. 'max' is the pipeline "
                 "high-water mark; 'sum' the no-release upper bound.",
        ),
        x: str = typer.Option(
            "cost", "--x", help="x presentation: 'cost' or 'throughput' (runtime only)."
        ),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict the method pool (repeatable)."
        ),
        amortize_fit: bool = typer.Option(
            False, "--amortize-fit",
            help="Add fit_ms/n_targets_per_fold to each per-target runtime. Off by "
                 "default: the fold size is an experimental artifact, not a "
                 "deployment denominator.",
        ),
        allow_mixed_setups: bool = typer.Option(
            False, "--allow-mixed-setups",
            help="Permit pooling runs whose setup differs (see SCHEMA.md §7).",
        ),
        strict_membership: bool = typer.Option(
            False, "--strict-membership",
            help="Raise instead of warn when a method is in the CSV but not on disk.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
        out_dir: Path = typer.Option(
            None, "--out-dir", help="Override the artifact directory."
        ),
    ) -> None:
        """Pareto frontier of classification accuracy against runtime or memory.

        Merges several datasets so each variant is drawn as a spread over their
        accuracy scalars — cross-dataset stability, which one run cannot show.
        Shortest-Ping is included at cost 0, so a CBG variant reaches the
        frontier only by being more accurate than free.
        """
        ids = list(run_id or [])
        if all_runs == bool(ids):
            raise typer.BadParameter("pass exactly one of --run-id (repeatable) or --all-runs")
        if accuracy_csv and all_runs:
            raise typer.BadParameter(
                "--accuracy-csv cannot be combined with --all-runs: the files must pair "
                "positionally with --run-id"
            )
        if accuracy_csv and len(accuracy_csv) != len(ids):
            raise typer.BadParameter(
                f"got {len(accuracy_csv)} --accuracy-csv for {len(ids)} --run-id; "
                "pass one per run, in the same order"
            )
        if cost not in COST_SPECS:
            raise typer.BadParameter(f"--cost must be one of {list(COST_SPECS)}")
        if cost_stat not in COST_STATS:
            raise typer.BadParameter(f"--cost-stat must be one of {list(COST_STATS)}")
        if cost_rows not in COST_ROWS:
            raise typer.BadParameter(f"--cost-rows must be one of {list(COST_ROWS)}")
        if memory_reduce not in MEMORY_REDUCERS:
            raise typer.BadParameter(f"--memory-reduce must be one of {list(MEMORY_REDUCERS)}")
        if x not in ("cost", "throughput"):
            raise typer.BadParameter("--x must be 'cost' or 'throughput'")

        spec = COST_SPECS[cost]
        if x == "throughput" and not spec.supports_throughput:
            raise typer.BadParameter(
                f"--x throughput is meaningless for --cost {cost} (1/MB is not a rate); "
                "it applies to runtime only"
            )
        if amortize_fit and spec.key != "runtime":
            raise typer.BadParameter(
                "--amortize-fit applies to runtime only: a one-time fit's peak memory is "
                "not additive across targets"
            )
        if spec.key == "memory_rss" and cost_stat == "p50":
            typer.echo(
                "warning: memory_rss at p50 is floored at one 4096-byte page for every "
                "fast stage (5 ms sampler) and cannot rank methods — prefer "
                "--cost-stat p95."
            )

        g, resolutions = resolve_cli_grid(grid, list(resolution), sweep=sweep)

        if all_runs:
            found = discover_runs(outputs_root)
            runs = {r.run_id: r for r in found}
        else:
            runs = {rid: resolve_run(rid, outputs_root) for rid in ids}
        _guard_one_setup(runs, allow_mixed=allow_mixed_setups)

        reduce = memory_reduce if spec.key != "runtime" else "sum"
        target_dir = out_dir or cross_dir(analysis_root, runs.keys())

        for res in resolutions:
            if accuracy_csv:
                csvs = {rid: Path(p) for rid, p in zip(ids, accuracy_csv)}
            else:
                csvs = {
                    rid: r.cls_accuracy_dir(root=analysis_root, grid=g.name, resolution=res)
                    / TOPN_CSV
                    for rid, r in runs.items()
                }

            long, notes = load_cost_accuracy(
                csvs, runs,
                top_n=top_n, spec=spec, cost_stat=cost_stat, cost_rows=cost_rows,
                reduce=reduce, amortize_fit=amortize_fit,
                methods=list(method) if method else None,
                strict_membership=strict_membership,
            )
            wide = aggregate_methods(long, top_n=top_n)
            wide["cost_includes_fit"] = bool(amortize_fit)
            with np.errstate(divide="ignore"):
                wide["throughput_per_s"] = (
                    np.where(wide["cost"] > 0, 1000.0 / wide["cost"], np.inf)
                    if spec.key == "runtime"
                    else np.nan
                )

            colors = method_colors(list(wide["method"]))
            n_other = sum(1 for m in wide["method"] if colors[m] == _C_OTHER)
            if n_other:
                typer.echo(
                    f"warning: {len(wide)} methods but only {len(_VARIANT_HUES)} "
                    f"distinguishable hues — {n_other} drawn in one grey 'other' "
                    f"series (all identities are in the CSV). Use --method to pick "
                    f"the variants you want coloured."
                )

            name_kw = dict(
                cost_key=spec.key, grid=g.name, resolution=res, top_n=top_n,
                amortized=amortize_fit, x=x,
            )
            csv_path = target_dir / artifact_name(ext="csv", **name_kw)
            png_path = target_dir / artifact_name(ext="png", **name_kw)
            json_path = target_dir / artifact_name(ext="json", **name_kw)

            order_columns(wide, top_n=top_n).to_csv(csv_path, index=False)

            subtitle = (
                f"{len(runs)} dataset(s): {', '.join(sorted(runs))} · "
                f"answer space {grid_slug(g.name, res)} · top-{top_n} · "
                f"cost = {spec.key} {cost_stat} over {cost_rows} rows, "
                f"{reduce}-reduced across LTD/MTL/CTR · "
                f"fit {'amortized in' if amortize_fit else 'excluded (one-time per combo)'}"
                f"\ntop-left = most cost-effective · vertical spread within one "
                f"colour = that variant's cross-dataset stability · horizontal bar "
                f"= its cost range across datasets"
            )
            if len({r.setup for r in runs.values()}) > 1:
                subtitle += " · WARNING: pooled across differing setups"

            if x == "throughput":
                zero = wide.loc[wide["cost"] <= 0, "method"].tolist()
                if zero:
                    typer.echo(
                        f"note: {', '.join(zero)} cost 0, i.e. infinite throughput, "
                        f"which a throughput axis cannot place — omitted from this "
                        f"figure (exact value kept in the CSV's throughput_per_s). "
                        f"Use --x cost to see it."
                    )
            pw, pl = (to_throughput(wide, long) if x == "throughput" else (wide, long))
            plot_pareto(
                pw, pl, png_path,
                spec=spec, top_n=top_n, subtitle=subtitle,
                x_label=(
                    "Throughput (targets / s)  —  faster → right"
                    if x == "throughput" else None
                ),
            )

            json_path.write_text(
                json.dumps(
                    {
                        "datasets": {rid: str(csvs[rid]) for rid in sorted(csvs)},
                        "setups": {rid: r.setup for rid, r in sorted(runs.items())},
                        "grid": g.name,
                        "resolution": res,
                        "top_n": top_n,
                        "cost": {
                            "key": spec.key,
                            "stat": cost_stat,
                            "rows": cost_rows,
                            "reduce": reduce,
                            "unit": spec.unit,
                            "stage_cols": list(spec.stage_cols),
                        },
                        "x": x,
                        "fit_policy": (
                            "amortized into per-target cost"
                            if amortize_fit
                            else "excluded — one-time per combo, reported separately"
                        ),
                        "shortest_ping": {
                            "cost": 0.0,
                            "basis": "analytical",
                            "note": "no targets.parquet row; argmin over RTTs already "
                                    "measured, so no distance model, multilateration "
                                    "or centroid step is charged",
                        },
                        "encoding": {
                            "colour": "CBG variant identity",
                            "symbol": "dataset",
                            "line": "one polyline per dataset, cost-ascending",
                            "variant_hues": dict(_LABEL_HUES),
                            "n_folded_to_other": int(n_other),
                        },
                        "methods_skipped": notes,
                        "fallback_policy": (
                            "accuracy_topN already counts FALLBACK as failure (paper "
                            "§7.2); this module does not re-apply that policy"
                        ),
                    },
                    indent=2,
                )
            )
            typer.echo(
                f"{grid_slug(g.name, res)} top-{top_n} {spec.key}: "
                f"{len(wide)} methods x {len(runs)} dataset(s) -> "
                f"{csv_path.name}, {png_path.name}"
            )
