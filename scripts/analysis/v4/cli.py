"""Typer CLI for the v4 analysis layer — HEALPix only.

Three commands, in dependency order:

    python -m scripts.analysis.v4.cli build-answer-space --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v4.cli build-bipartite    --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v4.cli classify           --run-id as01-260728-260802-mesh

`classify` needs the answer space; `build-bipartite` is independent of both and
can run in any order. Each writes one directory per rung of the ladder
(`healpix-128` .. `healpix-16`) plus a merged CSV one level above holding the
curve across rungs.

There is no `--grid`. v4 is HEALPix because the laddered metric it computes
cannot be expressed on H3 — see `modules/healpix.py`. `--nside` selects which
rungs to build, defaulting to the full ladder.
"""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.analysis.v4.modules import answer_space, bipartite, classify, healpix
from scripts.analysis.v4.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    discover_runs,
    resolve_run,
)

app = typer.Typer(
    add_completion=False,
    help="HEALPix answer-space and laddered classification accuracy (v4).",
)

_NSIDE_HELP = (
    "Rung to build (repeatable). Must be a power of two. Defaults to the full "
    f"ladder {list(healpix.NSIDE_LADDER)}."
)


def _runs(run_id: str | None, all_runs: bool, outputs_root: Path):
    if all_runs == bool(run_id):
        raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
    return discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]


def _nsides(values: list[int] | None) -> tuple[int, ...]:
    if not values:
        return healpix.NSIDE_LADDER
    try:
        return tuple(sorted({healpix.validate_nside(v) for v in values}, reverse=True))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("build-answer-space")
def build_answer_space_cmd(
    run_id: str = typer.Option(None, help="Run to build for."),
    all_runs: bool = typer.Option(False, "--all-runs", help="Every run under the root."),
    nside: list[int] = typer.Option(None, "--nside", "-n", help=_NSIDE_HELP),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v4 writes."),
) -> None:
    """Quantize targets into HEALPix classes, one seed per occupied cell."""
    for run in _runs(run_id, all_runs, outputs_root):
        spaces = answer_space.build_for_run(
            run, nsides=_nsides(nside), analysis_root=analysis_root
        )
        rungs = ", ".join(f"nside={s.nside} K={s.n_seeds}" for s in spaces)
        typer.echo(f"{run.run_id}: {spaces[0].meta['n_targets']} targets -> {rungs}")


@app.command("build-bipartite")
def build_bipartite_cmd(
    run_id: str = typer.Option(None, help="Run to build for."),
    all_runs: bool = typer.Option(False, "--all-runs", help="Every run under the root."),
    nside: list[int] = typer.Option(None, "--nside", "-n", help=_NSIDE_HELP),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v4 writes."),
) -> None:
    """Co-quantize targets and vantage points on the same grid, every rung."""
    for run in _runs(run_id, all_runs, outputs_root):
        qs = bipartite.build_for_run(
            run, nsides=_nsides(nside), analysis_root=analysis_root
        )
        rungs = ", ".join(
            f"nside={q.nside} tg={q.meta['n_target_cells']} vp={q.meta['n_vp_cells']}"
            for q in qs
        )
        typer.echo(f"{run.run_id}: {rungs}")


@app.command("classify")
def classify_cmd(
    run_id: str = typer.Option(None, help="Run to score."),
    all_runs: bool = typer.Option(False, "--all-runs", help="Every run under the root."),
    nside: list[int] = typer.Option(None, "--nside", "-n", help=_NSIDE_HELP),
    method: list[str] = typer.Option(None, "--method", "-m", help="Score only these."),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v4 writes."),
    strict: bool = typer.Option(
        True,
        help=(
            "Fail if accuracy is not monotone across the ladder. On by default: "
            "exact nesting guarantees it, so a violation means the grid or the "
            "metric is wrong, not the data."
        ),
    ),
) -> None:
    """Score every method at every rung, ring-graded."""
    for run in _runs(run_id, all_runs, outputs_root):
        long = classify.score_for_run(
            run,
            nsides=_nsides(nside),
            methods=list(method) if method else None,
            analysis_root=analysis_root,
        )
        bad = classify.monotonicity_violations(long)
        for _, r in long[long.nside == long.nside.max()].iterrows():
            typer.echo(
                f"{run.run_id} nside={int(r.nside)} {r.method}: "
                f"ring0={r.accuracy_ring0} ring1={r.accuracy_ring1} "
                f"ring2={r.accuracy_ring2} unplaced={int(r.n_unplaced)} "
                f"(retired nearest-seed would say {r.accuracy_nearest_seed_retired})"
            )
        if len(bad):
            typer.echo(f"MONOTONICITY VIOLATIONS ({len(bad)}):", err=True)
            typer.echo(bad.to_string(index=False), err=True)
            if strict:
                raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
