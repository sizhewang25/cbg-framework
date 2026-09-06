"""Accuracy crossed with VP proximity: which stratum each method wins (§8.1).

`classify` says whether a method got a target right; `build-proximity` says
whether the target was answerable by proximity at all. This module multiplies
the two, which is what turns an aggregate accuracy number into a claim about
*where* a variant earns its result.

That distinction is the whole point of §8.1. A variant three points ahead
overall may be ahead only on targets a VP was already sitting on — in which case
it is being credited for the dataset's geometry — or ahead on `geometry_only`
targets, where nothing but its multilateration could have produced the answer.
The two look identical in `topn_accuracy.csv`.

## Two outputs, deliberately

* **`accuracy_by_flag.csv`** — one row per (method, flag, N): accuracy inside
  and outside the flag, both cell counts, the rate difference and phi. Each flag
  is a separate 2x2, so the four can disagree; the diamond is not a chain and
  this table does not pretend it is.
* **`accuracy_by_taxonomy.csv`** — one row per (method, term, N) over
  `geometry_only` / `selection_miss` / `selection_hit`, which *do* partition the
  targets. This is the table §8.2 reads.

## Cell counts ship with every rate

A rate over four targets and a rate over four hundred print the same width. Both
tables carry `n` beside every accuracy, and a flag with no variance on this run
is marked `zero_variance` rather than being reported as a difference of 0.0 —
`has_proximate_vp` really is constant on as01 and as03, and "no contrast to
separate on" is not "no effect".

## The tautological cell

`has_proximate_sping_vp` x `shortest_ping` x top-1 is Shortest-Ping's own
correctness by construction (see `proximity.py`). It is emitted with
`is_tautological=True` so a reader cannot mistake its perfect separation for a
finding. The same cell at top-3 is informative, because the flag stays top-1.

Command: `breakdown-accuracy`. Writes to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules.classify import DEFAULT_TOPN, SHORTEST_PING
from scripts.analysis.v3.modules.diagram.common.labels import label_for
from scripts.analysis.v3.modules.diagram.common.membership import (
    available_methods,
    build_membership,
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
    resolve_run,
)
from scripts.analysis.v3.modules.proximity import FLAGS, TAXONOMY, load_proximity

BY_FLAG_CSV = "accuracy_by_flag.csv"
BY_TAXONOMY_CSV = "accuracy_by_taxonomy.csv"
MANIFEST_JSON = "breakdown_manifest.json"

#: The one (flag, method, N) cell that is true by construction rather than by
#: measurement. Named here so both the CSV and any figure reading it agree.
TAUTOLOGICAL_CELL = ("has_proximate_sping_vp", SHORTEST_PING, 1)


def taxonomy_of(labels: pd.DataFrame) -> pd.Series:
    """Each target's §8.2 term — the diamond's argmin chain, cut twice.

    A partition, unlike the four flags: every target is in exactly one term, so
    this is the frame that can be read as "where the targets are".
    """
    return pd.Series(
        np.select(
            [
                ~labels["has_proximate_vp"].to_numpy(dtype=bool),
                ~labels["has_proximate_sping_vp"].to_numpy(dtype=bool),
            ],
            [TAXONOMY[0], TAXONOMY[1]],
            default=TAXONOMY[2],
        ),
        index=labels["target_id"].to_numpy(),
        name="term",
    )


def _phi(a: int, b: int, c: int, d: int) -> float:
    """Phi coefficient of a 2x2 — Pearson correlation between two booleans.

    Reported beside the rate difference because the two answer different
    questions and a stratum can move one without the other: a 40-point gap over
    six targets is a large difference and a weak correlation. Undefined (NaN)
    when either margin is empty, which is exactly the zero-variance case.
    """
    num = a * d - b * c
    den = np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    return float(num / den) if den > 0 else np.nan


def _rate(hits: np.ndarray) -> float:
    return round(float(hits.mean()), 4) if hits.size else np.nan


def breakdown_by_flag(
    membership: dict[int, pd.DataFrame], labels: pd.DataFrame
) -> pd.DataFrame:
    """Accuracy inside vs outside each flag, per method and N.

    `membership` maps N to the boolean target x method matrix `build_membership`
    produces — so the fallbacks-are-failures policy (§7.2) and the common-
    denominator check are inherited rather than re-implemented here.
    """
    lab = labels.set_index("target_id")
    rows: list[dict] = []
    for n, matrix in sorted(membership.items()):
        common = matrix.index.intersection(lab.index)
        matrix = matrix.loc[common]
        sub = lab.loc[common]
        for flag in FLAGS:
            mask = sub[flag].to_numpy(dtype=bool)
            for method in matrix.columns:
                correct = matrix[method].to_numpy(dtype=bool)
                a, b = int((mask & correct).sum()), int((mask & ~correct).sum())
                c, d = int((~mask & correct).sum()), int((~mask & ~correct).sum())
                inside, outside = _rate(correct[mask]), _rate(correct[~mask])
                rows.append(
                    {
                        "method": method,
                        "method_label": label_for(method),
                        "flag": flag,
                        "top_n": n,
                        "n_true": a + b,
                        "n_false": c + d,
                        "accuracy_when_true": inside,
                        "accuracy_when_false": outside,
                        "rate_difference": (
                            round(inside - outside, 4)
                            if np.isfinite(inside) and np.isfinite(outside)
                            else np.nan
                        ),
                        "phi": round(_phi(a, b, c, d), 4),
                        "zero_variance": bool(a + b == 0 or c + d == 0),
                        "is_tautological": (flag, method, n) == TAUTOLOGICAL_CELL,
                    }
                )
    return pd.DataFrame(rows)


def breakdown_by_taxonomy(
    membership: dict[int, pd.DataFrame], labels: pd.DataFrame
) -> pd.DataFrame:
    """Accuracy within each §8.2 term, per method and N.

    The terms partition the targets, so `n` sums to the target count for every
    (method, N) — which is the check that makes this table readable as a
    decomposition of the headline accuracy rather than as three unrelated rates.
    """
    lab = labels.set_index("target_id")
    term = taxonomy_of(labels)
    rows: list[dict] = []
    for n, matrix in sorted(membership.items()):
        common = matrix.index.intersection(lab.index)
        matrix, t = matrix.loc[common], term.loc[common]
        for name in TAXONOMY:
            mask = (t == name).to_numpy()
            for method in matrix.columns:
                correct = matrix[method].to_numpy(dtype=bool)[mask]
                rows.append(
                    {
                        "method": method,
                        "method_label": label_for(method),
                        "term": name,
                        "top_n": n,
                        "n_targets": int(mask.sum()),
                        "n_correct": int(correct.sum()),
                        "accuracy": _rate(correct),
                        "share_of_targets": round(float(mask.mean()), 4)
                        if mask.size
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_for_run(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int | None = None,
    methods: list[str] | None = None,
    ns: tuple[int, ...] = DEFAULT_TOPN,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """Both tables plus the manifest for one run at one quantization."""
    cls_dir = run.cls_accuracy_dir(root=analysis_root, grid=grid, resolution=resolution)
    prox_dir = run.proximity_dir(root=analysis_root, grid=grid, resolution=resolution)
    prox = load_proximity(prox_dir)
    chosen = methods or available_methods(cls_dir)
    membership = {n: build_membership(cls_dir, chosen, top_n=n) for n in ns}

    by_flag = breakdown_by_flag(membership, prox.labels)
    by_tax = breakdown_by_taxonomy(membership, prox.labels)
    manifest = {
        "run_id": run.run_id,
        "grid": prox.meta.get("grid", {}),
        "proximity": str(prox_dir),
        "classification": str(cls_dir),
        "methods": chosen,
        "topn_reported": list(ns),
        "n_targets": int(len(prox.labels)),
        "zero_variance_flags": prox.meta.get("zero_variance", []),
        "tautological_cell": {
            "flag": TAUTOLOGICAL_CELL[0],
            "method": TAUTOLOGICAL_CELL[1],
            "top_n": TAUTOLOGICAL_CELL[2],
            "note": (
                "true by construction, not by measurement: the baseline predicts "
                "its VP's coordinate and the flag asks whether that VP's nearest "
                "seed is the target's. A pipeline self-check, never a finding. "
                "The same cell at top-3 is informative, because the flag is top-1."
            ),
        },
        "flag_policy": (
            "the four flags are a diamond, not a chain, so each is its own 2x2 "
            "and they may disagree. accuracy_by_taxonomy.csv is the partition."
        ),
        "correctness_policy": (
            "membership comes from diagram.common.membership.build_membership, so "
            "fallbacks count as failures (§7.2) and every method is scored over "
            "the same target set"
        ),
    }
    return by_flag, by_tax, manifest, cls_dir


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("breakdown-accuracy")
    def breakdown_cmd(
        run_id: str = typer.Option(
            None, help="Run to break down. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Break down every run under --outputs-root."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        topn: str = typer.Option(
            ",".join(str(n) for n in DEFAULT_TOPN),
            help="Comma-separated Ns to report. Proximity flags stay top-1 "
                 "regardless: they describe the dataset, not the scoring.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Cross each method's correctness with the VP proximity strata.

        Writes accuracy_by_flag.csv + accuracy_by_taxonomy.csv into
        target-cls-accuracy/<grid>-<resolution>/. Needs `classify` and
        `build-proximity` to have run on the same quantization.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        ns = tuple(int(x) for x in topn.split(",") if x.strip())
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]

        for run in runs:
            for res in resolutions:
                by_flag, by_tax, manifest, out_dir = build_for_run(
                    run,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=res,
                    methods=list(method) if method else None,
                    ns=ns,
                )
                by_flag.to_csv(out_dir / BY_FLAG_CSV, index=False)
                by_tax.to_csv(out_dir / BY_TAXONOMY_CSV, index=False)
                (out_dir / MANIFEST_JSON).write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                head = by_tax[(by_tax.top_n == ns[0]) & (by_tax.term == TAXONOMY[0])]
                best = (
                    head.sort_values("accuracy", ascending=False).iloc[0]
                    if len(head) and head["n_targets"].iloc[0]
                    else None
                )
                lead = (
                    f"{TAXONOMY[0]} n={int(head['n_targets'].iloc[0])}, best "
                    f"{best['method']} {best['accuracy']:.3f}"
                    if best is not None
                    else f"{TAXONOMY[0]} empty"
                )
                typer.echo(
                    f"{run.run_id}: {len(manifest['methods'])} methods x "
                    f"{len(FLAGS)} flags x top{ns} · {lead} -> {out_dir}"
                )
