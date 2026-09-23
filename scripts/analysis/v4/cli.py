"""Typer CLI for the v4 analysis layer — HEALPix only.

Build commands first, then figures, in dependency order:

    python -m scripts.analysis.v4.cli build-answer-space --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v4.cli build-bipartite    --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v4.cli classify           --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v4.cli plot-answer-space  --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v4.cli plot-outcome-bars  --run-id as01-... --run-id as02-...
    python -m scripts.analysis.v4.cli plot-euler         --run-id as01-... --run-id as02-...

`classify` needs the answer space; `build-bipartite` is independent of both and
can run in any order. `plot-answer-space` needs only `build-bipartite` — one
occupied target cell is exactly one class, so those artifacts already carry the
answer space, and the VP side with it.

Each build command writes one directory per rung of the ladder (`healpix-128` ..
`healpix-16`) plus a merged CSV one level above holding the curve across rungs.

There is no `--grid`. v4 is HEALPix because the laddered metric it computes
cannot be expressed on H3 — see `modules/healpix.py`. `--nside` selects which
rungs to build, defaulting to the full ladder.

The two cross-dataset figure commands differ in what they vary.
`plot-outcome-bars` sweeps the **rungs**: one figure per nside, each showing
where every method's predictions landed. `plot-euler` fixes **one** rung
(nside 128 by default) and sweeps the **tolerance** instead — top-1 in the
cell, top-2 within a ring, top-3 within two. That is the axis that changes
*which targets are in which set*, and therefore the only one that changes the
overlaps a Euler diagram is drawn from; four rungs would be four fitted
layouts whose circles cannot be compared by eye anyway.
"""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.analysis.v4.modules import (
    answer_space,
    bipartite,
    classify,
    figure_euler,
    figure_outcome_bars,
    healpix,
    map_answer_space,
    mapping,
)
from scripts.analysis.v4.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
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


@app.command("plot-outcome-bars")
def plot_outcome_bars_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run, one per dataset (repeatable)."
    ),
    nside: list[int] = typer.Option(None, "--nside", "-n", help=_NSIDE_HELP),
    method: list[str] = typer.Option(None, "--method", "-m", help="Plot only these."),
    layout: list[str] = typer.Option(
        None,
        "--layout",
        help=(
            f"{figure_outcome_bars.COMPARE} (one panel per dataset) or "
            f"{figure_outcome_bars.POOLED} (every run's targets as one "
            f"count-weighted population). Repeatable; default: both."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v4 writes."),
) -> None:
    """Outcome composition bars, one figure per rung, across datasets.

    Cross-dataset by nature, so `--run-id` is repeatable and there is no
    `--all-runs`: the figure IS the comparison between named datasets, and
    sweeping every run on disk would silently mix populations that were never
    meant to share an axis.

    `--layout pooled` adds a second figure per rung, `outcome_bars.pooled.<slug>`,
    holding one panel over every input run's targets at once. It is a
    micro-average: the per-target rows are concatenated and re-scored, so a
    target counts the same whichever dataset it came from. Pooling a **single**
    `--run-id` is a no-op view of that dataset — a micro-average over one
    population is that population — and is written anyway rather than
    second-guessed, since the caller asked for it.

    No traffic-weighted arm is drawn. None exists, and v3 filled that half of
    its figure from a hard-coded dict — 99.3% bars that measured nothing.
    """
    if not run_id:
        raise typer.BadParameter(
            "pass at least one --run-id; this figure compares named datasets"
        )
    layouts = tuple(dict.fromkeys(layout or ())) or figure_outcome_bars.LAYOUTS
    unknown = [x for x in layouts if x not in figure_outcome_bars.LAYOUTS]
    if unknown:
        raise typer.BadParameter(
            f"unknown --layout {unknown}; pick from "
            f"{list(figure_outcome_bars.LAYOUTS)}"
        )
    runs = [resolve_run(r, outputs_root) for r in run_id]
    try:
        pngs = figure_outcome_bars.build_for_runs(
            runs,
            nsides=_nsides(nside),
            methods=list(method) if method else None,
            layouts=layouts,
            analysis_root=analysis_root,
        )
    except ValueError as exc:
        # Coverage and target-overlap refusals are the caller's to fix, so they
        # read as a usage error rather than a traceback.
        raise typer.BadParameter(str(exc)) from exc
    for png in pngs:
        typer.echo(f"wrote {png}")


@app.command("plot-euler")
def plot_euler_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run to pool (repeatable)."
    ),
    nside: int = typer.Option(
        figure_euler.DEFAULT_NSIDE,
        "--nside",
        "-n",
        help=(
            "The ONE rung to draw at. Not a sweep: an Euler diagram is a fitted "
            "layout per population, so four rungs would be four figures whose "
            "circles cannot be compared by eye. Vary --top-n instead."
        ),
    ),
    top_n: list[int] = typer.Option(
        None,
        "--top-n",
        help=(
            f"Tolerance rung (repeatable): 1 = in the cell (ring 0), 2 = within "
            f"1 ring, 3 = within 2 rings. Default: all of "
            f"{list(figure_euler.DEFAULT_TOP_NS)}."
        ),
    ),
    method: list[str] = typer.Option(None, "--method", "-m", help="Draw only these."),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v4 writes."),
) -> None:
    """Area-proportional Euler diagram of which targets the methods agree on.

    Cross-dataset by nature, so `--run-id` is repeatable and there is no
    `--all-runs`. The runs' targets are pooled into one population (they are
    disjoint, so this is a concat) under
    `_cross/cls-accuracy/<dataset-set>/`.

    **`--top-n` is a ring, not a seed rank.** `1` means the prediction is in
    the truth's own cell, `2` within one ring of it, `3` within two — the same
    containment ladder `classify` scores, cumulative, so each set can only
    grow. v3's `top_n` ranked class seeds by distance and is retired.

    A method that places nothing within the tolerance gets no circle (Spotter
    at `--top-n 1`, which is zero on all three meshes). That is lossless: an
    empty set belongs to no region, so every other region is unchanged. It
    stays in the denominator and in the count tables, and the figure's
    footnote names it.
    """
    if not run_id:
        raise typer.BadParameter(
            "pass at least one --run-id; this figure pools named datasets"
        )
    runs = [resolve_run(r, outputs_root) for r in run_id]
    try:
        sets = figure_euler.build_for_runs(
            runs,
            nside=nside,
            top_ns=tuple(top_n) if top_n else figure_euler.DEFAULT_TOP_NS,
            methods=list(method) if method else None,
            analysis_root=analysis_root,
        )
    except ValueError as exc:
        # Coverage, tolerance and target-overlap refusals are the caller's to
        # fix, so they read as a usage error rather than a traceback.
        raise typer.BadParameter(str(exc)) from exc
    for written in sets:
        if "png" in written:
            typer.echo(f"wrote {written['png']}")
        else:
            typer.echo(
                f"no figure for {written['manifest'].name}: fewer than two "
                f"methods placed anything at this tolerance"
            )


@app.command("plot-answer-space")
def plot_answer_space_cmd(
    run_id: str = typer.Option(None, help="Run to map."),
    all_runs: bool = typer.Option(False, "--all-runs", help="Every run under the root."),
    us_only: bool = typer.Option(
        True,
        "--us-only/--auto-extent",
        help=(
            "Frame the continental US (the default: these meshes live there), "
            "or derive the frame from the run's own targets and VPs."
        ),
    ),
    extent: tuple[float, float, float, float] = typer.Option(
        (None, None, None, None),
        "--extent",
        help="LON_MIN LON_MAX LAT_MIN LAT_MAX. Overrides --us-only/--auto-extent.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v4 writes."),
) -> None:
    """Map the answer space: the lattice, the target cells, and the VP cells.

    Needs `build-bipartite`, **not** `build-answer-space`. One occupied target
    cell is exactly one class, so the bipartite artifacts already carry the
    answer space — and the VP side with it, co-quantized on the same grid.

    There is no `--nside`. The figure draws the whole ladder in one 2x2, because
    resolution is the tolerance dial and how the cells merge as it turns is what
    the figure is for; a single rung is a different, smaller figure. Written
    beside `occupancy_by_resolution.healpix.csv`, whose numbers it draws.
    """
    if all_runs == (run_id is not None):
        raise typer.BadParameter("pass exactly one of --run-id or --all-runs")

    chosen = None
    if extent and all(v is not None for v in extent):
        chosen = tuple(float(v) for v in extent)

    for run in _runs(run_id, all_runs, outputs_root):
        frame = chosen
        if frame is None:
            frame = (
                mapping.US_MAINLAND_EXTENT
                if us_only
                else map_answer_space.auto_extent_for_run(
                    run, analysis_root=analysis_root
                )
            )
        try:
            png = map_answer_space.build_for_run(
                run, extent=frame, analysis_root=analysis_root
            )
        except MissingArtifactError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"wrote {png}")


if __name__ == "__main__":
    app()
