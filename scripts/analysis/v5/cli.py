"""v5 CLI: the two-partition answer space and its classifier.

    python -m scripts.analysis.v5.cli build-answer-space --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v5.cli classify           --run-id as01-260728-260802-mesh

    python -m scripts.analysis.v5.cli plot-answer-space  --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v5.cli plot-outcome-bars \
        --run-id as01-260728-260802-mesh \
        --run-id as02-260728-260802-mesh \
        --run-id as03-260728-260802-mesh
    python -m scripts.analysis.v5.cli plot-error-cdf --layout per-run --layout pooled \
        --run-id as01-260728-260802-mesh \
        --run-id as02-260728-260802-mesh \
        --run-id as03-260728-260802-mesh
    python -m scripts.analysis.v5.cli plot-vp-proximity -c p5 -c p25 -c all \
        --run-id as01-260728-260802-mesh \
        --run-id as02-260728-260802-mesh \
        --run-id as03-260728-260802-mesh

    python -m scripts.analysis.v5.cli report-cohort-overlap -c p5 -c p25 \
        --run-id as01-260728-260802-mesh \
        --run-id as02-260728-260802-mesh \
        --run-id as03-260728-260802-mesh

`classify` and `plot-answer-space` need the answer space; `plot-outcome-bars`
`plot-error-cdf`, `plot-vp-proximity` and `report-cohort-overlap` need `classify` on every run. Everything writes under `outputs/analysis/v5/`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.analysis.v5.modules import (
    answer_space,
    classify,
    cohort_overlap,
    figure_error_cdf,
    figure_outcome_bars,
    figure_vp_distance_cdf,
    figure_vp_proximity,
    map_answer_space,
    mapping,
)
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
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


@app.command("plot-answer-space")
def plot_answer_space_cmd(
    run_id: str = typer.Option(None, help="Run to map."),
    all_runs: bool = typer.Option(False, "--all-runs", help="Every run under the root."),
    us_only: bool = typer.Option(
        True,
        "--us-only/--auto-extent",
        help="Frame the continental US (default), or derive the frame from the run's sites.",
    ),
    extent: tuple[float, float, float, float] = typer.Option(
        (None, None, None, None),
        "--extent",
        help="LON_MIN LON_MAX LAT_MIN LAT_MAX. Overrides --us-only/--auto-extent.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Map both partitions: the grid lattice and TG grids, the cells, the landmass.

    No `--nside`: the figure draws the whole ladder in one 2x2, because grid_km
    turns both partitions at once and how they coarsen together is the figure.
    Written into `answer-space/`, beside `sweep.csv`.
    """
    chosen = None
    if extent and all(v is not None for v in extent):
        chosen = tuple(float(v) for v in extent)
    for run in _runs(run_id, all_runs, outputs_root):
        frame = chosen or (
            mapping.US_MAINLAND_EXTENT
            if us_only
            else map_answer_space.auto_extent_for_run(run, analysis_root=analysis_root)
        )
        try:
            png = map_answer_space.build_for_run(run, extent=frame, analysis_root=analysis_root)
        except MissingArtifactError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"wrote {png}")


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
            f"{figure_outcome_bars.POOLED} (every run's TGs as one population). "
            f"Repeatable; default: both."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Outcome bars, one figure per rung: cell label first (true / wrong /
    outland / no answer), each broken down by ring tier. Colour = ring tier,
    stripe = cell label.

    Cross-dataset by nature, so `--run-id` is repeatable and there is no
    `--all-runs`. Written to `_cross/classify/<datasets>[@<arm>]/`.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id; this figure compares named datasets")
    layouts = tuple(dict.fromkeys(layout or ())) or figure_outcome_bars.LAYOUTS
    unknown = [x for x in layouts if x not in figure_outcome_bars.LAYOUTS]
    if unknown:
        raise typer.BadParameter(
            f"unknown --layout {unknown}; pick from {list(figure_outcome_bars.LAYOUTS)}"
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
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for png in pngs:
        typer.echo(f"wrote {png}")


@app.command("plot-error-cdf")
def plot_error_cdf_cmd(
    run_id: list[str] = typer.Option(None, "--run-id", help="Run (repeatable)."),
    all_runs: bool = typer.Option(
        False, "--all-runs", help="Every run under --outputs-root (per-run only)."
    ),
    layout: list[str] = typer.Option(
        None,
        "--layout",
        help=(
            f"{figure_error_cdf.PER_RUN} (one figure per run) or "
            f"{figure_error_cdf.POOLED} (every run's TGs as one population, one "
            f"curve per method). Repeatable; default: {figure_error_cdf.PER_RUN}."
        ),
    ),
    method: list[str] = typer.Option(None, "--method", "-m", help="Plot only these."),
    nside: int = typer.Option(
        figure_error_cdf.SOURCE_NSIDE,
        "--nside",
        "-n",
        help=(
            "Which rung's *_tgs.parquet to read. It does not change the output -- "
            "pred_dist_to_tg_km is identical at every rung -- so this exists to "
            "let you prove that, not to select a variant."
        ),
    ),
    min_x_km: float = typer.Option(
        figure_error_cdf.X_MIN_KM,
        "--min-x-km",
        help="Lower bound of the log x axis (km). Distances below it are clamped "
        "up in the drawn curve only; the CSV is unclamped.",
    ),
    max_x_km: float = typer.Option(
        figure_error_cdf.DEFAULT_X_MAX_KM, "--max-x-km", help="Upper bound (km)."
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Error-distance CDF per method (pred_dist_to_tg_km), log x, unanswered
    rows excluded.

    The companion to `plot-outcome-bars`: those say where a prediction landed,
    this says how far off it was. Artifacts carry **no rung in their names**:
    `per-run` writes `error_cdf.{png,csv,manifest.json}` into `classify/`,
    beside the `healpix-<n>/` directories; `pooled` writes the `.pooled.`
    triple into `_cross/classify/<datasets>[@<arm>]/`, beside the outcome bars.
    Pooled percentiles are recomputed from the concatenated rows, never
    averaged. Needs `classify` on every run.
    """
    if all_runs and run_id:
        raise typer.BadParameter("pass --run-id or --all-runs, not both")
    layouts = tuple(dict.fromkeys(layout or ())) or (figure_error_cdf.PER_RUN,)
    unknown = [x for x in layouts if x not in figure_error_cdf.LAYOUTS]
    if unknown:
        raise typer.BadParameter(
            f"unknown --layout {unknown}; pick from {list(figure_error_cdf.LAYOUTS)}"
        )
    if all_runs and figure_error_cdf.POOLED in layouts:
        raise typer.BadParameter(
            f"--layout {figure_error_cdf.POOLED} needs explicit --run-id: which "
            "datasets form one population is the caller's call"
        )
    runs = (
        discover_runs(outputs_root)
        if all_runs
        else [resolve_run(r, outputs_root) for r in (run_id or [])]
    )
    if not runs:
        raise typer.BadParameter("pass at least one --run-id, or --all-runs")
    try:
        pngs = figure_error_cdf.build_for_runs(
            runs,
            layouts=layouts,
            nside=nside,
            methods=list(method) if method else None,
            analysis_root=analysis_root,
            min_x_km=min_x_km,
            max_x_km=max_x_km,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for png in pngs:
        typer.echo(f"wrote {png}")


@app.command("plot-vp-proximity")
def plot_vp_proximity_cmd(
    run_id: list[str] = typer.Option(None, "--run-id", help="Run (repeatable); pooled."),
    cohort: list[str] = typer.Option(
        None,
        "--cohort",
        "-c",
        help=(
            "Which TGs to describe: p5 / p25 / p95 (each method's own most "
            "accurately placed 5%, 25% or 95%) or all -- which, unlike p95, "
            "keeps the unanswered rows too. Repeatable; default p25."
        ),
    ),
    method: list[str] = typer.Option(None, "--method", "-m", help="Draw only these."),
    geo: bool = typer.Option(True, "--geo/--no-geo", help="Draw the geographically closest VP violin."),
    sping: bool = typer.Option(True, "--sping/--no-sping", help="Draw the smallest-RTT VP violin."),
    nside: int = typer.Option(
        figure_vp_proximity.SOURCE_NSIDE,
        "--nside",
        "-n",
        help=(
            "Which rung's *_tgs.parquet supplies pred_dist_to_tg_km and status. "
            "It does not change the answer, so this selects a file, not a variant."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """How close a VP was, for the TGs each method placed best (ported from v4).

    Two violins per method: the geographically closest VP, and the smallest-RTT
    VP whose coordinate S-P returns. The gap between them is RTT inflation.
    Writes `vp_proximity.<cohort>.{png,csv,manifest.json}` into
    `_cross/vp-proximity/<datasets>[@<arm>]/`. The CSV carries `max_km` -- the
    bound -- beside `distinct_values` and `max_tie_share`, which say how much
    of the drawn violin is smoothing over replica ties. Needs `classify` on
    every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id")
    try:
        pngs = figure_vp_proximity.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            cohorts=list(cohort) if cohort else None,
            methods=list(method) if method else None,
            geo=geo,
            sping=sping,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for png in pngs:
        typer.echo(f"wrote {png}")


@app.command("plot-vp-distance-cdf")
def plot_vp_distance_cdf_cmd(
    run_id: list[str] = typer.Option(None, "--run-id", help="Run (repeatable); pooled."),
    method: list[str] = typer.Option(
        None,
        "--method",
        "-m",
        help=(
            "Restrict which methods must be scored before a TG is included. "
            "It does not change the curves -- the two distances are TG "
            "properties, not method outputs."
        ),
    ),
    nside: int = typer.Option(
        figure_vp_distance_cdf.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's *_tgs.parquet is read. It does not change the answer.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """How far a TG is from its nearest VP, from its smallest-RTT VP, and the gap.

    One panel over the whole population, no method on the axis. The third
    curve is the one that matters: `d_geo <= d_sp` is a per-TG inequality, and
    two marginal CDFs show only stochastic dominance, which is weaker. The gap
    is computed per TG, so its support establishes the pointwise claim --
    `min_gap_km` is in the manifest because a log axis cannot draw a negative
    gap and would hide a violation rather than show it.

    The gap is exactly 0 for the TGs whose two VPs coincide (19.1% on
    as01-03), which `log` cannot place. The curve starts at the axis floor
    already carrying that share; read the left intercept as the share.

    Writes `vp_distance_cdf.{png,csv,manifest.json}` into
    `_cross/vp-distance-cdf/<datasets>[@<arm>]/`. Needs `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id")
    try:
        pngs = figure_vp_distance_cdf.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            methods=list(method) if method else None,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for png in pngs:
        typer.echo(f"wrote {png}")


@app.command("report-cohort-overlap")
def report_cohort_overlap_cmd(
    run_id: list[str] = typer.Option(None, "--run-id", help="Run (repeatable); pooled."),
    cohort: list[str] = typer.Option(
        None,
        "--cohort",
        "-c",
        help=(
            "Which TGs to describe: p5 / p25 / p95 (each method's own most "
            "accurately placed 5%, 25% or 95%) or all. Repeatable; default "
            "p5 and p25."
        ),
    ),
    method: list[str] = typer.Option(None, "--method", "-m", help="Report only these."),
    reference: str = typer.Option(
        cohort_overlap.DEFAULT_REFERENCE,
        "--reference",
        help=(
            "Whose cohort the other methods are measured on. Its own row is "
            "emitted, as the scale the error column is read against."
        ),
    ),
    margin_km: float = typer.Option(
        cohort_overlap.DEFAULT_MARGIN_KM,
        "--margin-km",
        help=(
            "How much closer the shortest-ping VP must be than a prediction to "
            "count as beating it. Required, not cosmetic: at 0 the S-P control "
            "scores ~100% against itself on floating-point noise."
        ),
    ),
    nside: int = typer.Option(
        figure_vp_proximity.SOURCE_NSIDE,
        "--nside",
        "-n",
        help=(
            "Which rung's *_tgs.parquet supplies pred_dist_to_tg_km and status. "
            "It does not change the answer, so this selects a file, not a variant."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Whose easy TGs are whose, and what the others did on them.

    The layer above `plot-vp-proximity`: cohort union and degree histogram, the
    pairwise overlap matrix, the shared / answered-not-best / refused split on
    the reference's cohort with error percentiles, the per-TG win rate against
    it, how often the smallest-RTT VP is the closest VP, and how often the
    baseline's own VP beats a method's prediction. Writes seven CSVs and a
    manifest per cohort into `_cross/cohort-overlap/<datasets>[@<arm>]/`.
    No figure -- see the module docstring for why. Needs `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id")
    try:
        paths = cohort_overlap.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            cohorts=list(cohort) if cohort else None,
            methods=list(method) if method else None,
            reference=reference,
            margin_km=margin_km,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for path in paths:
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
