"""v5 CLI: the two-partition answer space and its classifier.

    python -m scripts.analysis.v5.cli build-answer-space --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v5.cli classify           --run-id as01-260728-260802-mesh

`classify` needs the answer space. Both write under `outputs/analysis/v5/`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.analysis.v5.modules import answer_space, classify
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    discover_runs,
    resolve_run,
)

app = typer.Typer(
    add_completion=False,
    help="Grid + cell answer space and two-label classification (v5).",
)

_NSIDE_HELP = (
    "Rung to build (repeatable). Must be a power of two. Defaults to the full "
    f"ladder {list(G.NSIDE_LADDER)}."
)


def _runs(run_id: str | None, all_runs: bool, outputs_root: Path):
    if all_runs == bool(run_id):
        raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
    return discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]


def _nsides(values: list[int] | None) -> tuple[int, ...]:
    if not values:
        return G.NSIDE_LADDER
    try:
        return tuple(sorted({G.validate_nside(v) for v in values}, reverse=True))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("build-answer-space")
def build_answer_space_cmd(
    run_id: str = typer.Option(None, help="Run to build for."),
    all_runs: bool = typer.Option(False, "--all-runs", help="Every run under the root."),
    nside: list[int] = typer.Option(None, "--nside", "-n", help=_NSIDE_HELP),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Place TGs in the grid partition and the landmass-bounded cell partition."""
    for run in _runs(run_id, all_runs, outputs_root):
        spaces = answer_space.build_for_run(
            run, nsides=_nsides(nside), analysis_root=analysis_root
        )
        rungs = ", ".join(
            f"nside={s.nside} grids={s.meta['n_grids']} seeds={s.n_seeds}" for s in spaces
        )
        m = spaces[0].meta
        typer.echo(f"{run.run_id}: {m['n_tgs']} TGs, {m['n_sites']} sites -> {rungs}")


@app.command("classify")
def classify_cmd(
    run_id: str = typer.Option(None, help="Run to score."),
    all_runs: bool = typer.Option(False, "--all-runs", help="Every run under the root."),
    nside: list[int] = typer.Option(None, "--nside", "-n", help=_NSIDE_HELP),
    method: list[str] = typer.Option(None, "--method", "-m", help="Score only these."),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
    strict: bool = typer.Option(
        True,
        help=(
            "Fail if ring0 accuracy is not monotone across the ladder. Exact "
            "nesting guarantees it, so a violation is a bug, not data."
        ),
    ),
) -> None:
    """Label every prediction with its ring and its cell, at every rung."""
    for run in _runs(run_id, all_runs, outputs_root):
        long = classify.score_for_run(
            run,
            nsides=_nsides(nside),
            methods=list(method) if method else None,
            analysis_root=analysis_root,
        )
        for _, r in long[long.nside == long.nside.max()].iterrows():
            typer.echo(
                f"{run.run_id} nside={int(r.nside)} {r.method}: "
                f"ring0={r.accuracy_ring0} ring1={r.accuracy_ring1} ring2={r.accuracy_ring2} "
                f"beyond={int(r.n_beyond)} | cell true={int(r.n_cell_true)} "
                f"wrong={int(r.n_cell_wrong)} outland={int(r.n_cell_outland)} "
                f"failed={int(r.n_failed)}"
            )
        bad = classify.monotonicity_violations(long)
        if len(bad):
            typer.echo(f"MONOTONICITY VIOLATIONS ({len(bad)}):", err=True)
            typer.echo(bad.to_string(index=False), err=True)
            if strict:
                raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
