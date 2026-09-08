"""§8.1's headline table: datasets down the rows, methods across the columns.

`table-accuracy` already emits the same numbers as one row per (run, method).
That shape is right for re-analysis and wrong for the paper: §8.1 compares
*methods within a dataset*, so the methods have to be adjacent columns a reader
can scan in one line. This module transposes it and adds the two things a paper
table needs and a data table does not — a best-in-row mark, and a row for every
dataset variant the section promises whether or not it has data yet.

Nothing is recomputed. `accuracy_rows` reads each run's `topn_accuracy.csv`, so
this table and `table-accuracy`'s cannot disagree.

## Rows are (kind, scope), and the weighted kind is reserved

§8.1's story line compares the two campaigns at *dataset-type* granularity —
"in Proprietary Mesh Unicast: Octant Hull, Octant Spline, SoI CBG" against the
same ranking on the traffic-weighted set — so the dataset type owns the row and
the per-AS numbers sit beneath it as the breakdown. Rows run `MESH (3 ASes)`,
`· AS01`, `· AS02`, `· AS03`, `TRAFFIC-WEIGHTED (0 of 3 ASes)`, `· AS01`, …

An earlier shape interleaved the two kinds per dataset (`AS01 MESH`,
`AS01 WEIGHTED`, `AS02 MESH`, …) so a weighted row read as a filter applied to
the row directly above it. That optimized for a per-AS comparison the paper does
not make, and it left the fleet-wide number — the one the next subsection is
about — for the reader to average in their head.

Every weighted row is **empty today**: no run in
`outputs/benchmark/v2/` carries traffic weights (`has_weight: false` on all
three operator runs) and the canonical source CSVs have no weight column, so
the data has not been collected rather than merely not analyzed.

The rows are emitted anyway, marked pending. Two reasons. The table's shape is
a claim about what the section compares, and a table that silently omits half
its rows reads as though the mesh numbers were the whole comparison. And
`--weighted-run-id` is the only thing missing: name the pairing and the row
fills in, with no change here.

The pairing is **explicit** — `--weighted-run-id as01=<run_id>` — rather than
inferred from the run's name. `short_dataset`, which supplies the dataset key
everywhere else in this layer, strips only an all-numeric trailing tail, so
none of the plausible weighted run names reduce to their dataset:
`as01-weighted-260728` and `as01-260728-260802-weighted` both come back
unchanged, and `as01w-260728-260802` reduces to `as01w`. Guessing here would
put a weighted row under the wrong dataset, or silently under none, on a name
nobody has chosen yet.

## The aggregate row is a micro-average, and its winner may not be unanimous

The leading row of each kind pools the targets and scores once — a target-count
micro-average, "pick a target at random from the fleet" — rather than averaging
the three datasets' rates. The two differ by at most 0.0064 here, and the
manifest reports both, so the weighting is measured rather than waved away.

Two consequences the table has to carry rather than hide. A pooled winner can
lead the population while leading only one of the datasets in it: at top-3
Octant-Hull takes the pooled row at 89.1% having led as02 alone, since as01
goes to Octant-Spline and as03 to Shortest-Ping and SoI. That cell gets a `†`
and a footnote naming what it lost. And a method absent from one run is pooled
over the runs that carry it, so its denominator is smaller than the row's
printed `n`; that cell gets a `‡` and its own count.

**Only the aggregate is computed.** Dataset rows print `accuracy_topN` verbatim
from `topn_accuracy.csv`, never a rate recomputed from counts — three of the
thirty-six cells round differently the two ways (as01 Spotter reads 38.9%
verbatim and 38.8% reconstructed), and this table agreeing with
`table-accuracy`'s to the printed digit is the invariant the module exists
under. The aggregate has no such source of truth, so it is the one row that
sums counts; `accuracy_is_reconstructed` says so per cell.

## Best-in-row is marked with a tie rule, not with an argmax

Bolding the row maximum alone asserts a ranking the sample size does not
support. On as03 the gap from Octant-Hull (50.2%) to Spotter (47.4%) is 0.028
against a standard error of 0.023 on 458 targets; on as02 the baseline leads
SoI by 0.003. So the mark is "within one standard error of the row's best",
computed on the best cell's own rate, and the caption states it. Several
methods can therefore be marked in one row, which is the honest rendering of a
near-tie and is exactly what happens on as03 at top-3, where Shortest-Ping and
SoI both reach 92.6% and beat Octant-Hull.

Rates print as percentages to one decimal (see `pct`); the CSV twin and the
manifest keep the fractions.

Command: `table-headline`. Writes to
`outputs/analysis/v3/_cross/accuracy-table/<dataset-set>/`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules.accuracy_table import accuracy_rows
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.cross import (
    cross_dir,
    guard_disjoint_targets,
    guard_one_setup,
    short_dataset,
)
from scripts.analysis.v3.modules.diagram.common.labels import (
    PUBLISHED_METHODS,
    short_label,
)
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    grid_slug,
    resolve_run,
)

#: Same family as `table-accuracy` — one dataset set, one directory.
CROSS_KIND = "accuracy-table"

MESH = "mesh"
WEIGHTED = "weighted"

#: Group order. Mesh first because it is the superset: the weighted campaign
#: keeps only the flows and target locations carrying the top 95% of traffic, so
#: it reads as that population filtered.
KINDS: tuple[str, ...] = (MESH, WEIGHTED)

#: Printed name of each kind. `WEIGHTED` alone is ambiguous beside the
#: best-in-row tie rule, which is also a weighting.
KIND_LABELS = {MESH: "MESH", WEIGHTED: "TRAFFIC-WEIGHTED"}

AGGREGATE = "aggregate"
DATASET = "dataset"

#: Marks a cell whose denominator is smaller than its row's `n`, because the
#: method is absent from one of the pooled runs. `‡` rather than `*`, which
#: would collide with markdown's bold.
INCOMPLETE_MARK = "‡"

#: Marks a pooled winner that does not also lead every dataset it pools.
NON_UNANIMOUS_MARK = "†"

#: Prefix on a breakdown row. A middle dot rather than an en or em dash: `—` is
#: already `PENDING`, and one glyph must not spell both "indented under the row
#: above" and "not measured".
DATASET_PREFIX = "·"

#: Printed in every cell of a reserved row. An em dash rather than `0` or `nan`,
#: because "not measured" and "measured as zero" are the two readings this
#: table must never conflate.
PENDING = "—"

#: Body table vs appendix table. Top-1 is the operator's decision; top-3 is
#: where the ranking changes hands (the baseline wins as03 at top-3 and loses it
#: at top-1), which is a §8.1 claim that needs the numbers printed somewhere.
BODY_TOP_N = 1
APPENDIX_TOP_N = 3

#: Marks a value that did not come out of this pipeline. `§` rather than `*`,
#: which markdown reads as emphasis — a cell ending `99.3%*` beside a bolded
#: neighbour renders unpredictably.
PROVISIONAL_MARK = "§"

#: **Hard-coded placeholder numbers. Not produced by this layer.**
#:
#: Supplied by hand from an earlier run so §8.1's mesh-vs-traffic-weighted
#: comparison can be laid out and reviewed before the weighted campaign is
#: collected. Every one of these is provisional and must be deleted the moment
#: `--weighted-run-id` has a real run to point at — a paired run wins over this
#: table, so the deletion is a cleanup rather than a switch-over.
#:
#: Two things they do **not** come with, and which are therefore not invented:
#:
#: * **No denominator.** Only rates were given, so `n_targets` stays NaN. That
#:   propagates honestly — `best_in_row` cannot compute a standard error without
#:   an `n`, so a provisional row is never marked best, and the table's bolding
#:   never rests on a number nobody can reproduce.
#: * **No top-3.** Only top-1 was given, so the appendix table's weighted row
#:   stays reserved. Keying this on the top-N is what stops a top-1 figure being
#:   printed under a top-3 heading.
#:
#: Fallback is 0.0 on all six, as reported.
PROVISIONAL_WEIGHTED: dict[int, dict[str, dict[str, float]]] = {
    BODY_TOP_N: {
        SHORTEST_PING: {"accuracy": 0.993, "fallback_rate": 0.0},
        "million_scale_cbg": {"accuracy": 0.993, "fallback_rate": 0.0},
        "vanilla_cbg": {"accuracy": 0.763, "fallback_rate": 0.0},
        "octant_cbg_hull": {"accuracy": 0.987, "fallback_rate": 0.0},
        "octant_cbg_spl": {"accuracy": 0.956, "fallback_rate": 0.0},
        "spotter_cbg": {"accuracy": 0.727, "fallback_rate": 0.0},
    }
}

PROVISIONAL_NOTE = (
    "the TRAFFIC-WEIGHTED row is a hard-coded placeholder from an earlier run, "
    "not a result of this pipeline. It carries no denominator, so it is never "
    "marked best and its `n` is blank, and it exists only at top-1. Delete "
    "`PROVISIONAL_WEIGHTED` and pass `--weighted-run-id <dataset>=<run_id>` once "
    "the campaign is collected."
)


def provisional_for(kind: str, scope: str, top_n: int, n_runs: int) -> dict | None:
    """The placeholder block for a row, or `None` when the row is real.

    A row only qualifies when it is the traffic-weighted **aggregate**, has no
    run of its own, and the top-N is one the placeholder covers. A real paired
    run therefore always wins, without a flag to remember.
    """
    if kind != WEIGHTED or scope != AGGREGATE or n_runs:
        return None
    return PROVISIONAL_WEIGHTED.get(top_n)



def row_label(
    kind: str,
    scope: str,
    dataset: str | None,
    n_runs: int,
    n_expected: int,
    *,
    grouped: bool = True,
) -> str:
    """The row's printed name, derived from its coverage rather than declared.

    An aggregate that spans two of three datasets must not print `(3 ASes)` —
    a reader would compare its `n` against the mesh row's as though the two were
    the same population. So the count comes from `(n_runs, n_expected)` every
    time, and the fully-reserved row reads `(0 of 3 ASes)` by the same rule that
    makes a partial one read `(1 of 3 ASes)`.

    `grouped` is false when the aggregate above was suppressed (one dataset, see
    `row_plan`). The indent prefix then has nothing to indent under and the kind
    is nowhere on the page, so the breakdown row carries it itself — two rows
    both reading `· AS01` name neither campaign.
    """
    if scope == DATASET:
        name = str(dataset).upper()
        if not grouped:
            return f"{name} {KIND_LABELS[kind]}"
        return f"{DATASET_PREFIX} {name}"
    span = f"{n_expected} ASes" if n_runs == n_expected else f"{n_runs} of {n_expected} ASes"
    return f"{KIND_LABELS[kind]} ({span})"


def row_plan(
    mesh_runs: dict[str, RunPaths], weighted_runs: dict[str, RunPaths]
) -> list[dict]:
    """The rows to print: kind-major, each group led by its aggregate.

    `mesh_runs` is keyed by `run_id` and its dataset comes from `short_dataset`;
    `weighted_runs` is keyed by **dataset** already, because the caller stated
    the pairing (see this module's docstring for why it cannot be inferred).

    Every row carries the *set* of runs feeding it rather than a single
    `run_id`, because "which runs are in this row" is what decides whether it is
    pending, what its denominator is, and whether its winner is unanimous. A
    breakdown row is the degenerate one-run case (`n_expected == 1`), which is
    what lets pending stay one rule — `n_runs == 0` — across both scopes.

    The aggregate is **suppressed for a single dataset**: a micro-average over
    one dataset is that dataset, and two identical rows invite a reader to hunt
    for the difference between them.

    A weighted run naming a dataset with no mesh run is an error rather than a
    new row: the table's premise is that the two are the same target population
    filtered, so a weighted row with nothing above it has no comparison to make.
    """
    order: list[str] = []
    mesh_by_dataset: dict[str, str] = {}
    for run_id in mesh_runs:
        dataset = short_dataset(run_id)
        if dataset in mesh_by_dataset:
            raise typer.BadParameter(
                f"--run-id {mesh_by_dataset[dataset]!r} and {run_id!r} both reduce "
                f"to dataset {dataset!r}; this table gets one MESH row per dataset. "
                "Dropping one silently would leave it out of the row list while it "
                "still reached the pooled denominator. Pick one, or run the command "
                "twice."
            )
        mesh_by_dataset[dataset] = run_id
        order.append(dataset)

    unpaired = sorted(set(weighted_runs) - set(mesh_by_dataset))
    if unpaired:
        raise typer.BadParameter(
            f"weighted run(s) paired to {unpaired}, which have no mesh run in this "
            "table. The weighted campaign is the mesh one filtered to the "
            "top-95%-traffic flows, so its row is only readable beneath its mesh "
            f"twin — pass the matching --run-id, or re-pair to one of "
            f"{sorted(mesh_by_dataset)}."
        )

    def run_for(kind: str, dataset: str) -> str | None:
        if kind == MESH:
            return mesh_by_dataset[dataset]
        return getattr(weighted_runs.get(dataset), "run_id", None)

    n_expected = len(order)
    plan: list[dict] = []
    for kind in KINDS:
        if n_expected > 1:
            run_ids = [r for d in order if (r := run_for(kind, d)) is not None]
            datasets = [d for d in order if run_for(kind, d) is not None]
            plan.append(
                {
                    "kind": kind,
                    "scope": AGGREGATE,
                    "dataset": None,
                    "datasets": datasets,
                    "run_ids": run_ids,
                    "run_id": None,
                    "n_expected": n_expected,
                }
            )
        for dataset in order:
            run_id = run_for(kind, dataset)
            plan.append(
                {
                    "kind": kind,
                    "scope": DATASET,
                    "dataset": dataset,
                    "datasets": [dataset] if run_id else [],
                    "run_ids": [run_id] if run_id else [],
                    "run_id": run_id,
                    "n_expected": 1,
                }
            )
    for index, entry in enumerate(plan):
        entry["row_index"] = index
        entry["label"] = row_label(
            entry["kind"],
            entry["scope"],
            entry["dataset"],
            len(entry["run_ids"]),
            entry["n_expected"],
            grouped=n_expected > 1,
        )
    return plan


def best_in_row(accuracy: np.ndarray, n_targets) -> np.ndarray:
    """Which cells are within one standard error of the row's best.

    The error is the *best* cell's binomial standard error, not each cell's, so
    one threshold applies across the row and the mark answers a single question:
    "is this method indistinguishable from the winner here?". A cell within that
    band is marked alongside the maximum, which is why several methods can be
    bold in one row.

    `n_targets` is a scalar for a row whose methods share a denominator, or one
    value per cell where they do not — a pooled row scores a method absent from
    one run over the runs that carry it, so the winner's own `n` is the one the
    band must come from. A scalar broadcasts, so this is a widening and the
    scalar behaviour is unchanged.

    `n_targets <= 0` or an all-missing row marks nothing — there is no winner to
    be near.
    """
    acc = np.asarray(accuracy, dtype=float)
    counts = np.broadcast_to(np.asarray(n_targets, dtype=float), acc.shape)
    finite = np.isfinite(acc)
    if not finite.any():
        return np.zeros(acc.shape, dtype=bool)
    winner = int(np.nanargmax(np.where(finite, acc, -np.inf)))
    n_winner = counts[winner]
    if not np.isfinite(n_winner) or n_winner <= 0:
        return np.zeros(acc.shape, dtype=bool)
    top = float(acc[winner])
    se = math.sqrt(max(top * (1.0 - top), 0.0) / float(n_winner))
    return finite & (acc >= top - se)


def method_counts(source: pd.DataFrame | None, method: str, *, top_n: int) -> dict | None:
    """One method's outcome counts in one run: correct / wrong / fallback / error.

    `topn_accuracy.csv` publishes `n_targets`, `n_solved`, `n_fallback` and
    `n_error` as counts but `accuracy_topN` only as a rate rounded to four
    decimals, so the correct count is reconstructed as
    `round(accuracy_topN * n_targets)`.

    That is exact rather than nearly exact. At four decimals and `n <= 458` the
    widest a rate's rounding interval can be is under a tenth of a target, so
    exactly one integer maps to the published rate. It is also confirmed against
    a per-target source — `overlap_membership.top{1,3}.csv`, a boolean
    correctness matrix in the same directory, whose column sums equal these
    counts on all thirty-six cells. That file is a `plot-venn` output, so it is
    used to check this and never to compute it: depending on it would make the
    headline table need a figure command to have run first.

    The four outcomes partition the target set — `n_targets == n_solved +
    n_fallback + n_error` holds on every run — and `n_error` is zero everywhere
    today but is carried separately so a future non-zero cannot be absorbed into
    "wrong", which would read as a scoring failure rather than a run failure.
    """
    if source is None or method not in source.index:
        return None
    row = source.loc[method]
    n_targets = int(row["n_targets"])
    n_solved = int(row["n_solved"])
    n_fallback = int(row["n_fallback"])
    n_error = int(row["n_error"])
    rate = float(row[f"accuracy_top{top_n}"])
    n_correct = int(round(rate * n_targets))
    return {
        "n_targets": n_targets,
        "n_correct": n_correct,
        "n_wrong": n_solved - n_correct,
        "n_fallback": n_fallback,
        "n_error": n_error,
        "accuracy": rate,
        "fallback_rate": float(row["fallback_rate"]),
    }


def pool_method_counts(per_run: list[dict | None], datasets: list[str] | None = None) -> dict | None:
    """Several runs' counts for one method, added into a single population.

    The sum is over the runs that *carry* the method, following `pool_counts` in
    the band figures: a method absent from one run is scored on the targets it
    actually ran on rather than penalised for the rest. The caller compares the
    result's `n_targets` against the row's own to decide whether the cell needs
    marking.

    Rates are recomputed from the summed counts, which is what makes this a
    micro-average — the mean of the runs' rates would weight a 399-target
    dataset the same as a 458-target one.
    """
    names = list(datasets) if datasets is not None else [""] * len(per_run)
    contributing = [c for c in per_run if c is not None]
    if not contributing:
        return None
    pooled = {
        key: sum(int(c[key]) for c in contributing)
        for key in ("n_targets", "n_correct", "n_wrong", "n_fallback", "n_error")
    }
    n = pooled["n_targets"]
    pooled["accuracy"] = pooled["n_correct"] / n if n else np.nan
    pooled["fallback_rate"] = pooled["n_fallback"] / n if n else np.nan
    # Which datasets this *cell* rests on, which is not the row's list once a
    # method is missing from one of them.
    pooled["datasets"] = [name for name, c in zip(names, per_run) if c is not None]
    return pooled


def headline_long(
    accuracy: pd.DataFrame, plan: list[dict], *, top_n: int, methods: list[str]
) -> pd.DataFrame:
    """One row per (plan row, method): the table in long form.

    Long rather than already-pivoted because this is also the CSV twin, and the
    `is_best` flag has to travel with the number it marks — a reader
    reproducing the bolding from a pivoted CSV would have to re-derive the tie
    rule and could pick a different one. The counts travel too, so the pooled
    rows can be re-derived rather than trusted.
    """
    column = f"accuracy_top{top_n}"
    if column not in accuracy.columns:
        available = sorted(
            c for c in accuracy.columns if c.startswith("accuracy_top")
        )
        raise ValueError(
            f"{column} not in topn_accuracy.csv (have {available}); re-run "
            f"`classify` with {top_n} in --topn"
        )
    by_run = {run_id: g.set_index("method") for run_id, g in accuracy.groupby("run_id")}

    rows: list[dict] = []
    for entry in plan:
        run_ids = entry["run_ids"]
        sources = [by_run.get(r) for r in run_ids]
        n_row_targets = sum(
            int(src["n_targets"].iloc[0]) for src in sources if src is not None and len(src)
        )
        per_method = {
            m: pool_method_counts(
                [method_counts(src, m, top_n=top_n) for src in sources],
                entry["datasets"],
            )
            for m in methods
        }
        placeholder = provisional_for(
            entry["kind"], entry["scope"], top_n, len(run_ids)
        )
        values = np.array(
            [
                # A breakdown row prints the published rate verbatim; only the
                # pooled row, which has no published rate, is computed.
                (
                    placeholder.get(m, {}).get("accuracy", np.nan)
                    if placeholder is not None
                    else np.nan
                    if per_method[m] is None
                    else (
                        per_method[m]["accuracy"]
                        if entry["scope"] == AGGREGATE
                        else float(by_run[run_ids[0]].loc[m, column])
                    )
                )
                for m in methods
            ],
            dtype=float,
        )
        cell_n = np.array(
            [np.nan if per_method[m] is None else per_method[m]["n_targets"] for m in methods],
            dtype=float,
        )
        marks = best_in_row(values, cell_n)
        for method, value, is_best in zip(methods, values, marks):
            counts = per_method[method]
            fallback = (
                placeholder.get(method, {}).get("fallback_rate", np.nan)
                if placeholder is not None
                else counts["fallback_rate"]
                if counts is not None and entry["scope"] == AGGREGATE
                else (
                    float(by_run[run_ids[0]].loc[method, "fallback_rate"])
                    if counts is not None
                    else np.nan
                )
            )
            n_cell = counts["n_targets"] if counts is not None else 0
            rows.append(
                {
                    "row_index": entry["row_index"],
                    "kind": entry["kind"],
                    "scope": entry["scope"],
                    "dataset": entry["dataset"],
                    "row_label": entry["label"],
                    "run_id": entry["run_id"],
                    "run_ids": ";".join(run_ids),
                    "datasets": ";".join(
                        counts["datasets"] if counts else entry["datasets"]
                    ),
                    "n_runs": len(run_ids),
                    "n_expected": entry["n_expected"],
                    "n_row_targets": n_row_targets or np.nan,
                    "n_targets": n_cell or np.nan,
                    "n_correct": counts["n_correct"] if counts else np.nan,
                    "n_wrong": counts["n_wrong"] if counts else np.nan,
                    "n_fallback": counts["n_fallback"] if counts else np.nan,
                    "n_error": counts["n_error"] if counts else np.nan,
                    "method": method,
                    "method_label": short_label(method),
                    "top_n": top_n,
                    "accuracy": value,
                    "fallback_rate": fallback,
                    "accuracy_is_reconstructed": entry["scope"] == AGGREGATE,
                    "denominator_complete": bool(
                        counts is not None and n_cell == n_row_targets
                    ),
                    "is_best": bool(is_best),
                    # A provisional row has no run either, so it stays `pending`
                    # for every purpose except printing — that is what keeps the
                    # "no run is paired here" footnote honest while the number is
                    # on the page.
                    "provisional": placeholder is not None
                    and np.isfinite(value),
                    "pending": len(run_ids) == 0,
                    "partial": 0 < len(run_ids) < entry["n_expected"],
                }
            )
    long = pd.DataFrame(rows)
    return _mark_unanimity(long)


def _mark_unanimity(long: pd.DataFrame) -> pd.DataFrame:
    """Per pooled cell, how many of its own datasets that method also leads.

    A pooled winner that leads only one of three datasets is a target-weighting
    artifact, not a fleet-wide result, and at top-3 that is exactly what
    Octant-Hull is. "Leads" reuses this table's own `is_best` rather than a bare
    argmax, so the two marks cannot disagree about what winning means.
    """
    won: list[str] = []
    n_won: list[float] = []
    for _, row in long.iterrows():
        if row["scope"] != AGGREGATE or not row["datasets"]:
            won.append("")
            n_won.append(np.nan)
            continue
        members = set(str(row["datasets"]).split(";"))
        peers = long[
            (long["kind"] == row["kind"])
            & (long["scope"] == DATASET)
            & (long["method"] == row["method"])
            & (long["dataset"].isin(members))
        ]
        leads = sorted(peers.loc[peers["is_best"], "dataset"].astype(str))
        won.append(";".join(leads))
        n_won.append(float(len(leads)))
    out = long.copy()
    out["won_datasets"] = won
    out["n_datasets_won"] = n_won
    out["unanimous"] = [
        bool(np.isnan(n) or n >= r) for n, r in zip(n_won, long["n_runs"])
    ]
    return out


def pct(rate: float) -> str:
    """A rate as a percentage to one decimal, as `plot-outcome-bars` labels it.

    One decimal because it is the same precision as the `.3f` this printed
    before — the precision the verbatim-vs-reconstructed invariant above is
    stated at — and because the bars drawing these same numbers use it
    (`figure_outcome_bars` labels every segment `{share * 100:.1f}%`), so a
    reader can check a cell against its segment without converting either.

    Same precision is not the same digit on every input: a rate landing exactly
    on a half at four decimals (`0.3885`) can round either way once it is
    multiplied by 100, and 267 of the 10,001 four-decimal rates take the other
    branch than `.3f` would. None of as01/02/03's cells are among them at
    top-1 or top-3 — the three real half-cases are pinned in
    `test_a_breakdown_rows_accuracy_is_the_csv_value_verbatim_never_reconstructed`
    — so the switch of units moved no printed digit here. A future run's cell
    could differ from `table-accuracy`'s markdown in the last place; the CSVs,
    which carry the unrounded rate, are the pair that must agree.

    The CSV twin keeps fractions: it is the re-analysis artifact, and the
    manifest's `weighting` deltas are meant to be read as rates.
    """
    return f"{rate * 100:.1f}%"


def _cell(
    accuracy: float,
    fallback: float,
    *,
    is_best: bool,
    complete: bool = True,
    unanimous: bool = True,
    provisional: bool = False,
) -> str:
    """One printed cell: the rate, its marks, and the fallback rate if non-zero.

    The fallback rate rides in the cell rather than in a column block of its
    own because only Vanilla CBG ever fallbacks — a parallel six-column block
    would be five-sixths zeros, and the one number that matters would be the
    hardest to find in it.

    The two marks sit against the number and outside the bold, so a cell can be
    best *and* qualified: `‡` when the cell's denominator is smaller than its
    row's, `†` when a pooled winner does not lead every dataset it pools.
    """
    if not np.isfinite(accuracy):
        return PENDING
    text = pct(accuracy)
    if is_best:
        text = f"**{text}**"
    if provisional:
        text = f"{text}{PROVISIONAL_MARK}"
    if not complete:
        text = f"{text}{INCOMPLETE_MARK}"
    if is_best and not unanimous:
        text = f"{text}{NON_UNANIMOUS_MARK}"
    if np.isfinite(fallback) and fallback > 0:
        text = f"{text} (fb {pct(fallback)})"
    return text


def render_markdown(
    long: pd.DataFrame, *, top_n: int, grid: str, resolution: int, methods: list[str]
) -> str:
    """The table as GitHub markdown, ready to paste into §8.1.

    Emitted beside the CSV because the CSV is for re-analysis and this is for
    the paper; hand-transcribing between the two is where a digit changes.
    """
    headers = [short_label(m) for m in methods]
    role = "headline" if top_n == BODY_TOP_N else "appendix"
    lines = [
        f"# §8.1 top-{top_n} classification accuracy — {grid_slug(grid, resolution)}"
        f" ({role})",
        "",
        f"Top-{top_n} accuracy per dataset type and method, as a percentage to one",
        "decimal, with the per-AS breakdown beneath each type. A type's row pools its",
        "datasets' targets and scores once, so it is target-weighted. **Bold** marks",
        "every method within one standard error of that row's best, so a near-tie shows",
        "as a tie rather than as a winner. `(fb x%)` is the fallback rate where",
        "non-zero; fallbacks count as failures (§7.2), so they are already subtracted",
        f"from the accuracy beside them. `{INCOMPLETE_MARK}` marks a cell pooled over",
        f"fewer datasets than its row, `{NON_UNANIMOUS_MARK}` a pooled winner that does",
        f"not lead every dataset it pools, and `{PENDING}` a row that is reserved with",
        "no data yet — never a measured zero.",
        "",
        "| dataset | n | " + " | ".join(headers) + " |",
        "| --- | --: | " + " | ".join("--:" for _ in headers) + " |",
    ]

    incomplete: list[str] = []
    non_unanimous: list[str] = []

    # Grouped on the plan index, not on (dataset, kind): an aggregate row has no
    # dataset, and `groupby` drops null keys by default — which would delete
    # every pooled row from the table without raising, and put the literal text
    # `nan` in the footnote below.
    for _, group in long.groupby("row_index", sort=False):
        indexed = group.set_index("method")
        label = str(group["row_label"].iloc[0])
        scope = str(group["scope"].iloc[0])
        n = group["n_row_targets"].iloc[0]
        cells = []
        for m in methods:
            provisional = bool(indexed["provisional"].get(m, False))
            complete = bool(indexed["denominator_complete"].get(m, False)) or provisional
            best = bool(indexed["is_best"].get(m, False))
            unanimous = bool(indexed["unanimous"].get(m, True))
            cells.append(
                _cell(
                    indexed["accuracy"].get(m, np.nan),
                    indexed["fallback_rate"].get(m, np.nan),
                    is_best=best,
                    complete=complete,
                    unanimous=unanimous,
                    provisional=provisional,
                )
            )
            if np.isfinite(indexed["accuracy"].get(m, np.nan)) and not complete:
                incomplete.append(
                    f"{short_label(m)} on {label} is pooled over "
                    f"{str(indexed['datasets'].get(m, '')).replace(';', '+')} "
                    f"({int(indexed['n_targets'].get(m, 0))} targets), not the row's "
                    f"{int(n) if np.isfinite(n) else 0}."
                )
            if best and not unanimous:
                led = [d for d in str(indexed["won_datasets"].get(m, "")).split(";") if d]
                members = [d for d in str(indexed["datasets"].get(m, "")).split(";") if d]
                lost = [d for d in members if d not in led]
                winners = {
                    d: ", ".join(
                        sorted(
                            long.loc[
                                (long["dataset"] == d)
                                & (long["scope"] == DATASET)
                                & long["is_best"],
                                "method_label",
                            ].astype(str)
                        )
                    )
                    for d in lost
                }
                non_unanimous.append(
                    f"{short_label(m)} leads the pooled {label} population "
                    f"({len(led)} of {len(members)} datasets: "
                    f"{', '.join(led) if led else 'none'}). "
                    + " ".join(f"{d}'s leader is {w}." for d, w in winners.items())
                    + " The pooled lead is target-weighted."
                )
        if bool(group["provisional"].any()):
            # The coverage count is about paired runs, and a provisional row has
            # none; printing both would say "0 of 3" beside six numbers.
            label = f"{KIND_LABELS[str(group['kind'].iloc[0])]} — provisional{PROVISIONAL_MARK}"
        if scope == AGGREGATE:
            label = f"**{label}**"
        n_text = f"{int(n):,}" if np.isfinite(n) else PENDING
        lines.append(f"| {label} | {n_text} | " + " | ".join(cells) + " |")

    pending = sorted(
        long.loc[
            long["pending"] & (long["scope"] == DATASET) & (~long["provisional"]),
            "dataset",
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    if pending:
        lines += [
            "",
            f"**Pending:** the TRAFFIC-WEIGHTED rows for {', '.join(pending)} are "
            "reserved. No run carries traffic weights (`has_weight: false`) and the "
            "source CSVs have no weight column, so the traffic-weighted campaign is "
            "uncollected rather than unanalyzed. Pass `--weighted-run-id` once it "
            "lands.",
        ]
    if bool(long["provisional"].any()):
        lines += ["", f"{PROVISIONAL_MARK} **Provisional:** {PROVISIONAL_NOTE}"]
    if incomplete:
        lines += ["", f"{INCOMPLETE_MARK} " + " ".join(dict.fromkeys(incomplete))]
    if non_unanimous:
        lines += ["", f"{NON_UNANIMOUS_MARK} " + " ".join(dict.fromkeys(non_unanimous))]
    return "\n".join(lines) + "\n"


def ranking(long: pd.DataFrame) -> dict:
    """Each pooled row's methods in accuracy order, with the top-three spread.

    §8.1's story line is an ordinal claim — "Octant Hull, Octant Spline, SoI CBG
    (show spread)" — and the pooled row is the only row that supports it: as02
    and as03 rank differently from each other and from the pool. Recorded here
    rather than printed as a column, because a bare spread number hides that its
    third-place *identity* changes between rows.
    """
    out: dict[str, dict] = {}
    for _, group in long[long["scope"] == AGGREGATE].groupby("row_index", sort=False):
        finite = group[np.isfinite(group["accuracy"])]
        if finite.empty:
            continue
        order = finite.sort_values("accuracy", ascending=False)
        ranked = [
            {"method": str(r["method"]), "accuracy": round(float(r["accuracy"]), 4)}
            for _, r in order.iterrows()
        ]
        out[str(group["row_label"].iloc[0])] = {
            "ranked": ranked,
            "spread_rank1_to_rank3": (
                round(ranked[0]["accuracy"] - ranked[2]["accuracy"], 4)
                if len(ranked) >= 3
                else None
            ),
        }
    return out


def weighting_check(long: pd.DataFrame) -> dict:
    """Per pooled cell, the micro-average against the macro mean it is not.

    The same shape the band figures record, so the two artifacts' caveats are
    comparable. On as01/02/03 the largest gap is 0.0064, which makes the
    weighting a choice that has been measured rather than a distortion.
    """
    out: dict[str, dict] = {}
    for _, group in long[long["scope"] == AGGREGATE].groupby("row_index", sort=False):
        kind = str(group["kind"].iloc[0])
        label = str(group["row_label"].iloc[0])
        for _, row in group.iterrows():
            if not np.isfinite(row["accuracy"]):
                continue
            members = set(str(row["datasets"]).split(";"))
            peers = long[
                (long["kind"] == kind)
                & (long["scope"] == DATASET)
                & (long["method"] == row["method"])
                & (long["dataset"].isin(members))
            ]["accuracy"].dropna()
            micro = float(row["accuracy"])
            macro = float(peers.mean()) if len(peers) else np.nan
            out[f"{label} | {row['method']}"] = {
                "pooled_micro": round(micro, 4),
                "dataset_macro_mean": round(macro, 4) if np.isfinite(macro) else None,
                "delta": round(micro - macro, 4) if np.isfinite(macro) else None,
                "n_datasets": int(len(peers)),
            }
    return out


def _target_ids(
    run: RunPaths, analysis_root: Path | None, grid: str, resolution: int
) -> set[str]:
    """One run's target ids, from a `classify` parquet in the directory the
    numbers themselves come from.

    Not `target_labels.csv`: that is a `build-proximity` artifact, so depending
    on it would make this table fail on a run that has been classified but not
    proximity-analyzed, for a check that has nothing to do with proximity — and
    it is regenerated on a different schedule, so it could go stale relative to
    the accuracies it is guarding. The `*_seed_distances.parquet` files are
    written by `classify` into `cls_accuracy_dir`, which `accuracy_rows` already
    resolves, so the guard and the numbers cannot drift apart.
    """
    directory = run.cls_accuracy_dir(root=analysis_root, grid=grid, resolution=resolution)
    parquets = sorted(directory.glob("*_seed_distances.parquet"))
    if not parquets:
        raise MissingArtifactError(
            f"no *_seed_distances.parquet in {directory}; run `classify` on "
            f"{run.run_id} before pooling it into a dataset-type row"
        )
    return set(pd.read_parquet(parquets[0], columns=["target_id"])["target_id"])


def build(
    mesh_runs: dict[str, RunPaths],
    weighted_runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    top_ns: tuple[int, ...] = (BODY_TOP_N, APPENDIX_TOP_N),
    allow_mixed_setups: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, str], dict]:
    """The long tables and their markdown renders, one of each per top-N.

    `weighted_runs` is keyed by dataset, `mesh_runs` by `run_id`.
    """
    wanted = list(methods) if methods else list(PUBLISHED_METHODS)
    plan = row_plan(mesh_runs, weighted_runs)
    pooled = any(e["scope"] == AGGREGATE for e in plan)

    all_runs = {**mesh_runs, **{r.run_id: r for r in weighted_runs.values()}}

    # Both guards exist because of the pooled row and are skipped without it.
    # `table-accuracy` exempts itself from the setup guard on the grounds that
    # it "puts each run on its own row where no such averaging can happen"; an
    # aggregate row is exactly that averaging, so the exemption lapses here.
    # Mixing the families would pool a 134-VP fleet with a 53-VP one, and the
    # target-id guard would not catch it — those target sets are disjoint.
    if pooled:
        guard_one_setup(all_runs, allow_mixed=allow_mixed_setups)
        guard_disjoint_targets(
            {rid: _target_ids(r, analysis_root, grid, resolution) for rid, r in all_runs.items()},
            remedy=(
                "A dataset-type row is a rate over a population, so a target counted "
                "twice makes it a rate over a multiset. Run the command once per "
                "dataset, or select runs with disjoint target sets."
            ),
        )
    accuracy = accuracy_rows(
        all_runs,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=wanted,
        include_counts=True,
    )
    present = set(accuracy["method"])
    absent = [m for m in wanted if m not in present]
    printed = [m for m in wanted if m in present]
    if not printed:
        raise ValueError(
            f"none of {wanted} appear in the selected runs' topn_accuracy.csv"
        )

    longs = {n: headline_long(accuracy, plan, top_n=n, methods=printed) for n in top_ns}
    markdowns = {
        n: render_markdown(
            longs[n], top_n=n, grid=grid, resolution=resolution, methods=printed
        )
        for n in top_ns
    }

    manifest = {
        "mesh_runs": sorted(mesh_runs),
        "weighted_runs": {d: r.run_id for d, r in sorted(weighted_runs.items())},
        "grid": {"scheme": grid, "resolution": resolution},
        "methods": printed,
        "methods_requested_but_absent": absent,
        "top_ns": list(top_ns),
        "body_top_n": BODY_TOP_N,
        "appendix_top_n": APPENDIX_TOP_N,
        "rows": plan,
        # `run_id` is None on every aggregate row by construction, so counting
        # that would file the mesh aggregate as reserved.
        "n_reserved_rows": sum(1 for e in plan if not e["run_ids"]),
        "aggregation": (
            "each kind's leading row pools its datasets' targets and scores once "
            "(a target-count micro-average). Breakdown rows print accuracy_topN "
            "verbatim; only the pooled row recomputes, because only it has no "
            "published rate. See `weighting` for the macro mean it is not."
        ),
        "aggregate_suppressed": (
            None
            if any(e["scope"] == AGGREGATE for e in plan)
            else "one dataset — the micro-average over a single dataset is that dataset"
        ),
        "provisional_rows": (
            {
                "note": PROVISIONAL_NOTE,
                "top_ns": sorted(PROVISIONAL_WEIGHTED),
                "values": PROVISIONAL_WEIGHTED,
            }
            if any(longs[n]["provisional"].any() for n in top_ns)
            else None
        ),
        "weighting": {n: weighting_check(longs[n]) for n in top_ns},
        "ranking": {n: ranking(longs[n]) for n in top_ns},
        "guards": (
            "with a pooled row: cross.guard_one_setup (the aggregate voids "
            "table-accuracy's exemption) and cross.guard_disjoint_targets over each "
            "run's classify parquets. Both skipped when no aggregate is emitted."
            if any(e["scope"] == AGGREGATE for e in plan)
            else "none — a table of independent rows pools nothing"
        ),
        "tie_rule": (
            "a cell is marked best when accuracy >= row_max - se, where se is the "
            "binomial standard error of the row's own maximum, sqrt(p*(1-p)/n). One "
            "threshold per row, so the mark reads as 'indistinguishable from the "
            "winner here' and several methods can be marked."
        ),
        "reserved_rows_note": (
            "WEIGHTED rows are emitted with no data: has_weight is false on every "
            "run and the canonical source CSVs carry no weight column, so the "
            "traffic-weighted campaign is uncollected. Pass "
            "--weighted-run-id <dataset>=<run_id> and the row fills in unchanged."
        ),
        "fallback_policy": (
            "fallbacks count as failures in accuracy (§7.2); fallback_rate is "
            "printed in-cell only where non-zero, which today is Vanilla CBG alone"
        ),
        "source": (
            "topn_accuracy.csv per run, via accuracy_table.accuracy_rows — the same "
            "reader table-accuracy uses, so the two tables cannot disagree"
        ),
    }
    return longs, markdowns, manifest


def parse_weighted_pairs(specs: list[str]) -> dict[str, str]:
    """`["as01=as01-weighted-260728"]` -> `{"as01": "as01-weighted-260728"}`.

    The `=` is required rather than optional-with-a-fallback. A bare `run_id`
    would have to be reduced to its dataset by `short_dataset`, which cannot do
    it for any plausible weighted run name (module docstring), so accepting one
    would mean guessing — and a wrong guess files the row under the wrong
    dataset instead of failing.
    """
    pairs: dict[str, str] = {}
    for spec in specs:
        dataset, sep, run = spec.partition("=")
        if not sep or not dataset.strip() or not run.strip():
            raise typer.BadParameter(
                f"--weighted-run-id expects <dataset>=<run_id>, got {spec!r}. The "
                "dataset is the key the mesh row is printed under, e.g. "
                "as01=as01-weighted-260728."
            )
        dataset, run = dataset.strip(), run.strip()
        if dataset in pairs and pairs[dataset] != run:
            raise typer.BadParameter(
                f"dataset {dataset!r} is paired to two weighted runs "
                f"({pairs[dataset]!r} and {run!r}); it gets one WEIGHTED row."
            )
        pairs[dataset] = run
    return pairs


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("table-headline")
    def table_headline_cmd(
        run_id: list[str] = typer.Option(
            None,
            "--run-id",
            help="Mesh runs, one per dataset (repeatable). These fix the row order.",
        ),
        weighted_run_id: list[str] = typer.Option(
            None,
            "--weighted-run-id",
            help="Traffic-weighted twin of a mesh run, as <dataset>=<run_id> "
            "(repeatable), e.g. as01=as01-weighted-260728. None exist yet, so "
            "their rows print as reserved.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        method: list[str] = typer.Option(
            None,
            "--method",
            help="Override the six published variants (repeatable), in print order.",
        ),
        allow_mixed_setups: bool = typer.Option(
            False,
            "--allow-mixed-setups",
            help="Pool anchors_to_probes with probes_to_anchors anyway. Refused by "
            "default because the dataset-type row averages across the runs.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """§8.1's headline table: dataset types as rows, methods as columns.

        Writes headline_top{1,3}.<grid>.{csv,md} + a manifest into
        _cross/accuracy-table/<dataset-set>/. Needs `classify` on every run.
        """
        if not run_id:
            raise typer.BadParameter(
                "pass at least one --run-id. Like `table-accuracy`, this command has "
                "no --all-runs: which datasets belong in one table is the caller's "
                "call, and §7.3 declines the operator/public head-to-head."
            )
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=False)
        mesh_runs = {rid: resolve_run(rid, outputs_root) for rid in run_id}
        weighted_runs = {
            dataset: resolve_run(rid, outputs_root)
            for dataset, rid in parse_weighted_pairs(weighted_run_id or []).items()
        }

        for res in resolutions:
            longs, markdowns, manifest = build(
                mesh_runs,
                weighted_runs,
                analysis_root=analysis_root,
                grid=g.name,
                resolution=res,
                methods=list(method) if method else None,
                allow_mixed_setups=allow_mixed_setups,
            )
            # Mesh runs alone name the directory. `weighted_runs` is keyed by
            # *dataset*, not `run_id`, so merging the two dicts here leaked the
            # dataset key into the slug and forked output into a sibling
            # `as01+as01+as02+as03/`. And the weighted run's own id would fork it
            # too, since `short_dataset` cannot reduce it (see `row_plan`). The
            # dataset set is fully determined by the mesh runs — `row_plan`
            # refuses an unpaired weighted run — so pairing a weighted twin
            # enriches this table in place rather than starting a parallel one.
            out_dir = cross_dir(analysis_root, sorted(mesh_runs), kind=CROSS_KIND)
            slug = grid_slug(g.name, res)
            for n, long in longs.items():
                long.to_csv(out_dir / f"headline_top{n}.{slug}.csv", index=False)
                (out_dir / f"headline_top{n}.{slug}.md").write_text(markdowns[n])
            (out_dir / f"headline.{slug}.manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )

            body = longs[BODY_TOP_N]
            marked = body.loc[body["is_best"], "method_label"].value_counts()
            typer.echo(
                f"{len(manifest['rows'])} rows "
                f"({manifest['n_reserved_rows']} reserved) x "
                f"{len(manifest['methods'])} methods at {slug} · "
                f"top-{BODY_TOP_N} best-in-row: "
                f"{', '.join(f'{m} x{c}' for m, c in marked.items()) or 'none'} "
                f"-> {out_dir}"
            )
