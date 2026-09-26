"""v5 CLI: the two-partition answer space and its classifier.

    python -m scripts.analysis.v5.cli build-answer-space --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v5.cli classify           --run-id as01-260728-260802-mesh

    python -m scripts.analysis.v5.cli plot-answer-space  --run-id as01-260728-260802-mesh
    python -m scripts.analysis.v5.cli plot-mtl-map       --run-id as01-260728-260802-mesh -m vanilla_cbg
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

`classify`, `plot-answer-space` and `plot-mtl-map` need the answer space; `plot-outcome-bars`
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
    figure_vp_dist_gap,
    figure_vp_distance_cdf,
    figure_vp_proximity,
    map_answer_space,
    map_mtl,
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
    """Place TGs in the grid partition and the unbounded cell partition."""
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
) -> None:
    """Label every prediction with its ring and its cell."""
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
                f"beyond={int(r.n_beyond)} | cell correct={int(r.n_cell_correct)} "
                f"wrong={int(r.n_cell_wrong)} unanswered={int(r.n_cell_unanswered)}"
            )


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
    """Map both partitions: the grid lattice and TG grids, and the cells.

    The cells are unbounded, so they run to the edge of whatever frame is
    drawn -- that overreach is the point of the figure. Written into
    `answer-space/`, one level above the rung directories.
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
    mode: list[str] = typer.Option(
        None,
        "--mode",
        help=(
            f"{figure_outcome_bars.BOUNDED} (cell label broken down by ring "
            f"tier) or {figure_outcome_bars.UNBOUNDED} (the cell axis alone, "
            f"the plain nearest-seed verdict). Repeatable; default: "
            f"{figure_outcome_bars.BOUNDED}."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Outcome bars, one figure per rung: cell label first (correct / wrong /
    no answer). In `bounded` each label is broken down by ring tier, colour =
    ring tier and stripe = cell label; in `unbounded` the cell axis is drawn
    alone and colour = cell label.

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
    modes = tuple(dict.fromkeys(mode or ())) or (figure_outcome_bars.BOUNDED,)
    bad = [x for x in modes if x not in figure_outcome_bars.MODES]
    if bad:
        raise typer.BadParameter(
            f"unknown --mode {bad}; pick from {list(figure_outcome_bars.MODES)}"
        )
    runs = [resolve_run(r, outputs_root) for r in run_id]
    try:
        pngs = figure_outcome_bars.build_for_runs(
            runs,
            nsides=_nsides(nside),
            methods=list(method) if method else None,
            layouts=layouts,
            modes=modes,
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
        None,
        "--max-x-km",
        help=(
            f"Upper bound (km). Default: {figure_error_cdf.DEFAULT_X_MAX_KM:,.0f} under "
            f"--unanswered {figure_error_cdf.EXCLUDE}, {figure_error_cdf.SENTINEL_X_MAX_KM:,.0f} "
            f"under {figure_error_cdf.SENTINEL}, which has to clear the sentinel."
        ),
    ),
    unanswered: str = typer.Option(
        figure_error_cdf.EXCLUDE,
        "--unanswered",
        help=(
            f"{figure_error_cdf.EXCLUDE}: drop the rows a method did not answer, so "
            f"each curve rests on its own population (the default; the only one that "
            f"joins to accuracy.csv). {figure_error_cdf.SENTINEL}: park them at "
            f"--sentinel-km so every curve is drawn over the same denominator and the "
            f"height at the sentinel is the method's answer rate. Writes "
            f"`.sentinel.` filenames, so it does not overwrite the other."
        ),
    ),
    sentinel_km: float = typer.Option(
        figure_error_cdf.SENTINEL_KM,
        "--sentinel-km",
        help="Where --unanswered sentinel parks an unanswered TG (km).",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Error-distance CDF per method (pred_dist_to_tg_km), log x, unanswered
    rows excluded — or parked at a sentinel, with `--unanswered sentinel`.

    The companion to `plot-outcome-bars`: those say where a prediction landed,
    this says how far off it was. Artifacts carry **no rung in their names**:
    `per-run` writes `error_cdf.{png,csv,manifest.json}` into `classify/`,
    beside the `healpix-<n>/` directories; `pooled` writes the `.pooled.`
    triple into `_cross/classify/<datasets>[@<arm>]/`, beside the outcome bars.
    Pooled percentiles are recomputed from the concatenated rows, never
    averaged. Needs `classify` on every run.

    `--unanswered sentinel` adds the `.sentinel.` triple beside them: the same
    curves over the whole TG roster, unanswered rows at 10,000 km, so refusal
    rates are readable off the figure. Its percentiles are censored and do not
    join to `accuracy.csv`.
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
            unanswered=unanswered,
            sentinel_km=sentinel_km,
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


@app.command("plot-vp-dist-gap")
def plot_vp_dist_gap_cmd(
    run_id: list[str] = typer.Option(None, "--run-id", help="Run (repeatable); pooled."),
    reference: str = typer.Option(
        figure_vp_dist_gap.SHORTEST_PING,
        "--reference",
        "-r",
        help="Method whose best-case cohorts are drawn. Its error is d_sp when it is S-P.",
    ),
    cohort: list[str] = typer.Option(
        None,
        "--cohort",
        "-c",
        help="Percentile cohorts drawn beside the population (repeatable). Default p25, p5.",
    ),
    method: list[str] = typer.Option(
        None,
        "--method",
        "-m",
        help="Restrict which methods must be scored before a TG is included.",
    ),
    nside: int = typer.Option(
        figure_vp_dist_gap.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's *_tgs.parquet is read. It does not change the gap.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """The gap `d_sp - d_geo` over the population and a method's best cases.

    S-P's error *is* `d_sp`, and `d_sp = d_geo + gap` splits it into VP
    proximity and a term measuring how faithfully latency orders VPs by
    distance. This asks whether S-P's near-exact predictions are the TGs
    where that second term vanishes.

    **Quote `max_km`, not the zero share.** At p5 an all-zero gap is forced by
    arithmetic: the cohort bound (1.55 km on as01-03) sits below the smallest
    positive gap in the population (1.71 km), so no nonzero gap can fit.
    `forced_zero_gap` in the manifest carries that test per cohort. At p25 the
    zero share (67.5%) understates a cohort whose gap never exceeds 5.62 km
    against a bound of 25.3 km that would have admitted four times that.

    Writes `vp_dist_gap.{png,csv,manifest.json}` into
    `_cross/vp-dist-gap/<datasets>[@<arm>]/`. Needs `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id")
    try:
        pngs = figure_vp_dist_gap.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            methods=list(method) if method else None,
            reference=reference,
            cohorts=tuple(cohort) if cohort else figure_vp_dist_gap.DEFAULT_COHORTS,
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


@app.command("plot-mtl-map")
def plot_mtl_map_cmd(
    run_id: str = typer.Option(..., help="Run to render. One run per invocation."),
    method: list[str] = typer.Option(
        [],
        "--method",
        "-m",
        help=(
            "Method to render; repeatable. Defaults to every combo in the run "
            f"plus the {map_mtl.SHORTEST_PING!r} control."
        ),
    ),
    nside: int = typer.Option(
        G.DEFAULT_NSIDE,
        "--nside",
        "-n",
        help=(
            "The ONE rung to render at. Not a sweep: this is a case viewer, and "
            "two HTML files are two answers to a question asked about one TG."
        ),
    ),
    cell_extent: tuple[float, float, float, float] = typer.Option(
        (None, None, None, None),
        "--cell-extent",
        help=(
            "LON_MIN LON_MAX LAT_MIN LAT_MAX to build the serving cells against. "
            "NOT the view: the map is pannable and refits per TG. This only sets "
            "how far the unbounded cells are drawn before they are cut. Shrunk "
            "automatically if it leaves EPSG:5070's usable domain."
        ),
    ),
    us_only: bool = typer.Option(
        False,
        "--us-only",
        help=(
            "Cut the cells to the continental US instead of the default frame. "
            "Tighter, but it clips predictions that land north of it."
        ),
    ),
    no_regions: bool = typer.Option(
        False,
        "--no-regions",
        help=(
            "Skip the MTL feasible-region layer -- the only expensive part. The "
            "benchmark never stores the regions, so each is a full re-run of the "
            "planar intersection (~7 s/TG on the Octant family). Cached under "
            "mtl-map/regions/, which is rung-free, so the cost is paid once per "
            "(method, TG) no matter which --nside you render."
        ),
    ),
    workers: int = typer.Option(
        map_mtl.DEFAULT_WORKERS, "--workers", "-j", help="Processes for the replay."
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Interactive per-TG map of one method's MTL result and BOTH verdicts.

    Draws the TG's grid and its ring-1/ring-2 neighbours, the grid the
    prediction fell in, the serving cell of every seed with the TG's own and
    the prediction's highlighted, each VP's LTD constraint, and the MTL
    feasible region those constraints intersect to.

    The cells are unbounded -- the frame they are cut to is a rendering bound,
    and the page says so. Read `cell_label` beside the grid offset: neither is
    a verdict alone.
    """
    try:
        nside = G.validate_nside(nside)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if cell_extent and all(v is not None for v in cell_extent):
        extent = tuple(float(v) for v in cell_extent)
    elif us_only:
        extent = mapping.US_MAINLAND_EXTENT
    else:
        extent = map_mtl.CELL_FRAME

    try:
        run = resolve_run(run_id, outputs_root)
        methods = list(method) or [*run.combo_ids, map_mtl.SHORTEST_PING]
        rendered = map_mtl.build_for_run(
            run,
            methods=methods,
            nside=nside,
            extent=extent,
            analysis_root=analysis_root,
            regions=not no_regions,
            workers=max(1, int(workers)),
            progress=typer.echo,
        )
    except MissingArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc

    for name, path, payload in rendered:
        # The tally is the assertion that this map and `accuracy.csv` agree on
        # BOTH axes, so a mismatch is visible without opening the file.
        grid = {s: 0 for s in map_mtl.STATUSES}
        cell = {lab: 0 for lab in classify.CELL_LABELS}
        offsets = []
        for t in payload["tgs"]:
            grid[t["status"]] += 1
            cell[t["cell_label"]] += 1
            # ANSWERED rows only, and linear interpolation, because that is
            # what `summarize` does -- `df.loc[answered, GRID_OFFSET]` then
            # `.quantile()`. Pooling the fallbacks in moves vanilla_cbg's p50
            # from 2 to 1, and this line exists to be diffed against
            # `accuracy.csv`, so it has to be the same statistic.
            if t["status"] != "failed" and t["grid_offset"] >= 0:
                offsets.append(t["grid_offset"])

        def _q(frac: float) -> str:
            if not offsets:
                return "—"
            import numpy as _np

            return f"{float(_np.quantile(offsets, frac)):g}"

        typer.echo(
            f"{run.run_id}: nside={payload['nside']} ({payload['grid_km']} km) · "
            f"{name} · {len(payload['tgs'])} TGs · K={payload['n_seeds']} seeds"
        )
        typer.echo(
            "  grid: " + " / ".join(f"{s} {grid[s]}" for s in map_mtl.STATUSES)
            + f" | offset p50={_q(0.50)} p90={_q(0.90)} max={max(offsets) if offsets else '—'}"
        )
        typer.echo(
            "  cell: " + " / ".join(f"{lab} {cell[lab]}" for lab in classify.CELL_LABELS)
            + f" | {payload['cell_meta']['agreement']:.4f} agreement, "
            f"frame {tuple(round(v, 1) for v in extent)}"
        )
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
