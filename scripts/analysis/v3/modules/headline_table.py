"""§8.1's headline table: datasets down the rows, methods across the columns.

`table-accuracy` already emits the same numbers as one row per (run, method).
That shape is right for re-analysis and wrong for the paper: §8.1 compares
*methods within a dataset*, so the methods have to be adjacent columns a reader
can scan in one line. This module transposes it and adds the two things a paper
table needs and a data table does not — a best-in-row mark, and a row for every
dataset variant the section promises whether or not it has data yet.

Nothing is recomputed. `accuracy_rows` reads each run's `topn_accuracy.csv`, so
this table and `table-accuracy`'s cannot disagree.

## Rows are (dataset, kind), and the weighted kind is reserved

§8.1 compares each operator dataset's mesh campaign against its
traffic-weighted subset, so the row order is `AS01 MESH`, `AS01 WEIGHTED`,
`AS02 MESH`, … Every one of those weighted rows is **empty today**: no run in
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

## Best-in-row is marked with a tie rule, not with an argmax

Bolding the row maximum alone asserts a ranking the sample size does not
support. On as03 the gap from Octant-Hull (0.502) to Spotter (0.474) is 0.028
against a standard error of 0.023 on 458 targets; on as02 the baseline leads
SoI by 0.003. So the mark is "within one standard error of the row's best",
computed on the best cell's own rate, and the caption states it. Several
methods can therefore be marked in one row, which is the honest rendering of a
near-tie and is exactly what happens on as03 at top-3, where Shortest-Ping and
SoI both reach 0.926 and beat Octant-Hull.

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
from scripts.analysis.v3.modules.cross import cross_dir, short_dataset
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
    RunPaths,
    grid_slug,
    resolve_run,
)

#: Same family as `table-accuracy` — one dataset set, one directory.
CROSS_KIND = "accuracy-table"

MESH = "mesh"
WEIGHTED = "weighted"

#: Row order within a dataset. Mesh first because it is the superset: the
#: weighted campaign keeps only the flows and target locations carrying the top
#: 95% of traffic, so it reads as a filter applied to the row above it.
KINDS: tuple[str, ...] = (MESH, WEIGHTED)

#: Printed in every cell of a reserved row. An em dash rather than `0` or `nan`,
#: because "not measured" and "measured as zero" are the two readings this
#: table must never conflate.
PENDING = "—"

#: Body table vs appendix table. Top-1 is the operator's decision; top-3 is
#: where the ranking changes hands (the baseline wins as03 at top-3 and loses it
#: at top-1), which is a §8.1 claim that needs the numbers printed somewhere.
BODY_TOP_N = 1
APPENDIX_TOP_N = 3


def row_plan(
    mesh_runs: dict[str, RunPaths], weighted_runs: dict[str, RunPaths]
) -> list[dict]:
    """The (dataset, kind) rows to print, mesh runs fixing the dataset order.

    `mesh_runs` is keyed by `run_id` and its dataset comes from
    `short_dataset`; `weighted_runs` is keyed by **dataset** already, because
    the caller stated the pairing (see this module's docstring for why it cannot
    be inferred).

    A weighted run naming a dataset with no mesh run is an error rather than a
    new row: the table's premise is that the two are the same target population
    filtered, so a weighted row with nothing above it has no comparison to make.
    """
    order: list[str] = []
    mesh_by_dataset: dict[str, str] = {}
    for run_id in mesh_runs:
        dataset = short_dataset(run_id)
        if dataset not in mesh_by_dataset:
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

    plan: list[dict] = []
    for dataset in order:
        for kind in KINDS:
            run_id = (
                mesh_by_dataset[dataset]
                if kind == MESH
                else getattr(weighted_runs.get(dataset), "run_id", None)
            )
            plan.append({"dataset": dataset, "kind": kind, "run_id": run_id})
    return plan


def best_in_row(accuracy: np.ndarray, n_targets: int) -> np.ndarray:
    """Which cells are within one standard error of the row's best.

    The error is the *best* cell's binomial standard error, not each cell's, so
    one threshold applies across the row and the mark answers a single question:
    "is this method indistinguishable from the winner here?". A cell within that
    band is marked alongside the maximum, which is why several methods can be
    bold in one row.

    `n_targets <= 0` or an all-missing row marks nothing — there is no winner to
    be near.
    """
    acc = np.asarray(accuracy, dtype=float)
    finite = np.isfinite(acc)
    if not finite.any() or n_targets <= 0:
        return np.zeros(acc.shape, dtype=bool)
    top = float(np.nanmax(acc[finite]))
    se = math.sqrt(max(top * (1.0 - top), 0.0) / n_targets)
    return finite & (acc >= top - se)


def headline_long(
    accuracy: pd.DataFrame, plan: list[dict], *, top_n: int, methods: list[str]
) -> pd.DataFrame:
    """One row per (dataset, kind, method): the table in long form.

    Long rather than already-pivoted because this is also the CSV twin, and the
    `is_best` flag has to travel with the number it marks — a reader
    reproducing the bolding from a pivoted CSV would have to re-derive the tie
    rule and could pick a different one.
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
        run_id = entry["run_id"]
        source = by_run.get(run_id) if run_id else None
        n_targets = (
            int(source["n_targets"].iloc[0]) if source is not None and len(source) else 0
        )
        values = np.array(
            [
                float(source[column].get(m, np.nan))
                if source is not None and m in source.index
                else np.nan
                for m in methods
            ],
            dtype=float,
        )
        marks = best_in_row(values, n_targets)
        for method, value, is_best in zip(methods, values, marks):
            fallback = (
                float(source["fallback_rate"].get(method, np.nan))
                if source is not None and method in source.index
                else np.nan
            )
            rows.append(
                {
                    "dataset": entry["dataset"],
                    "kind": entry["kind"],
                    "run_id": run_id,
                    "n_targets": n_targets or np.nan,
                    "method": method,
                    "method_label": short_label(method),
                    "top_n": top_n,
                    "accuracy": value,
                    "fallback_rate": fallback,
                    "is_best": bool(is_best),
                    "pending": run_id is None,
                }
            )
    return pd.DataFrame(rows)


def _cell(accuracy: float, fallback: float, *, is_best: bool) -> str:
    """One printed cell: the rate, bolded if best-in-row, fallback if non-zero.

    The fallback rate rides in the cell rather than in a column block of its
    own because only Vanilla CBG ever fallbacks — a parallel six-column block
    would be five-sixths zeros, and the one number that matters would be the
    hardest to find in it.
    """
    if not np.isfinite(accuracy):
        return PENDING
    text = f"{accuracy:.3f}"
    if is_best:
        text = f"**{text}**"
    if np.isfinite(fallback) and fallback > 0:
        text = f"{text} (fb {fallback:.2f})"
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
        f"Top-{top_n} accuracy per dataset and method. **Bold** marks every method",
        "within one standard error of that row's best, so a near-tie shows as a tie",
        "rather than as a winner. `(fb x)` is the fallback rate where non-zero;",
        "fallbacks count as failures (§7.2), so they are already subtracted from the",
        "accuracy beside them. `—` rows are reserved and have no data yet.",
        "",
        "| dataset | n | " + " | ".join(headers) + " |",
        "| --- | --: | " + " | ".join("--:" for _ in headers) + " |",
    ]

    for (dataset, kind), group in long.groupby(["dataset", "kind"], sort=False):
        indexed = group.set_index("method")
        n = indexed["n_targets"].iloc[0]
        cells = [
            _cell(
                indexed["accuracy"].get(m, np.nan),
                indexed["fallback_rate"].get(m, np.nan),
                is_best=bool(indexed["is_best"].get(m, False)),
            )
            for m in methods
        ]
        label = f"{dataset.upper()} {kind.upper()}"
        n_text = f"{int(n)}" if np.isfinite(n) else PENDING
        lines.append(f"| {label} | {n_text} | " + " | ".join(cells) + " |")

    pending = long.loc[long["pending"], "dataset"].unique()
    if len(pending):
        lines += [
            "",
            f"**Pending:** the WEIGHTED rows for {', '.join(sorted(pending))} are "
            "reserved. No run carries traffic weights (`has_weight: false`) and the "
            "source CSVs have no weight column, so the traffic-weighted campaign is "
            "uncollected rather than unanalyzed. Pass `--weighted-run-id` once it "
            "lands.",
        ]
    return "\n".join(lines) + "\n"


def build(
    mesh_runs: dict[str, RunPaths],
    weighted_runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    top_ns: tuple[int, ...] = (BODY_TOP_N, APPENDIX_TOP_N),
) -> tuple[dict[int, pd.DataFrame], dict[int, str], dict]:
    """The long tables and their markdown renders, one of each per top-N.

    `weighted_runs` is keyed by dataset, `mesh_runs` by `run_id`.
    """
    wanted = list(methods) if methods else list(PUBLISHED_METHODS)
    plan = row_plan(mesh_runs, weighted_runs)

    all_runs = {**mesh_runs, **{r.run_id: r for r in weighted_runs.values()}}
    accuracy = accuracy_rows(
        all_runs,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=wanted,
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
        "n_reserved_rows": sum(1 for e in plan if e["run_id"] is None),
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
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """§8.1's headline table: datasets as rows, methods as columns.

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
            )
            out_dir = cross_dir(
                analysis_root, sorted({**mesh_runs, **weighted_runs}), kind=CROSS_KIND
            )
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
