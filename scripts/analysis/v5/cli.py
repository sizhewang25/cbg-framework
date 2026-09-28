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
    python -m scripts.analysis.v5.cli plot-outcome-map \
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
`plot-error-cdf`, `plot-vp-proximity` and `report-cohort-overlap` need `classify` on every run.
`plot-outcome-map` needs both. Everything writes under `outputs/analysis/v5/`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.analysis.v5.modules import (
    answer_space,
    classify,
    cohort_overlap,
    figure_contest_map,
    figure_error_cdf,
    figure_error_diff,
    figure_exclusive_error,
    figure_ltd_model,
    figure_outcome_bars,
    figure_outcome_map,
    figure_peripherality,
    figure_rtt_cdf,
    figure_stability,
    figure_vp_dist_gap,
    figure_vp_distance_cdf,
    figure_vp_proximity,
    map_answer_space,
    map_mtl,
    mapping,
    octant_finetuning,
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


@app.command("plot-outcome-map")
def plot_outcome_map_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run, one per dataset (repeatable); one column each."
    ),
    cohort: list[str] = typer.Option(
        None,
        "--cohort",
        "-c",
        help=(
            f"Which cell label to map: {figure_outcome_map.COHORTS[0]} or "
            f"{figure_outcome_map.COHORTS[1]}. Repeatable; default: both. "
            f"{classify.UNANSWERED!r} has no coordinate, so it cannot be drawn."
        ),
    ),
    method: list[str] = typer.Option(None, "--method", "-m", help="Draw only these rows."),
    extent: tuple[float, float, float, float] = typer.Option(
        figure_outcome_map.DEFAULT_EXTENT,
        "--extent",
        help=(
            "Shared frame, as lon_min lon_max lat_min lat_max. One frame for "
            "every panel, so the columns compare; predictions outside it are "
            "drawn as carets at the edge, never dropped."
        ),
    ),
    nside: int = typer.Option(
        figure_outcome_map.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's answer space and *_tgs.parquet to draw.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Where one cohort's predictions landed: a map per method x dataset.

    The companion to `plot-outcome-bars`: those count the outcomes, this one
    places them. Each panel draws the run's cells and seeds, every prediction
    of the cohort coloured by its grid offset with a line to its target, and a
    per-cell count keyed on `tg_seed_id` -- so on the `wrong` map the number
    reads "predictions that should have landed in this cell". `solved_mask` is
    applied, and the FALLBACK rows it removes are named in the panel title.

    Writes `outcome_map.<cohort>.{png,csv,manifest.json}` into
    `_cross/outcome-map/<datasets>[@<arm>]/`. Needs `build-answer-space` and
    `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id; this figure compares named datasets")
    try:
        pngs = figure_outcome_map.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            cohorts=list(cohort) if cohort else None,
            methods=list(method) if method else None,
            extent=tuple(extent),
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for png in pngs:
        typer.echo(f"wrote {png}")


@app.command("plot-contest-map")
def plot_contest_map_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run, one per dataset (repeatable); one column each."
    ),
    method_a: str = typer.Option(
        figure_contest_map.DEFAULT_METHOD_A,
        "--method-a",
        help="First method; its hue marks the sites it wins.",
    ),
    method_b: str = typer.Option(
        figure_contest_map.DEFAULT_METHOD_B,
        "--method-b",
        help="Second method; its hue marks the sites it wins.",
    ),
    extent: tuple[float, float, float, float] = typer.Option(
        figure_contest_map.DEFAULT_EXTENT,
        "--extent",
        help=(
            "Shared frame, as lon_min lon_max lat_min lat_max. Defaults to the "
            "outcome maps' frame so the two can be read side by side."
        ),
    ),
    ncols: int = typer.Option(
        None,
        "--ncols",
        help=(
            "Panels per row; default is one row. Use 1 to stack the meshes, "
            "which is what keeps the per-site labels legible at paper width."
        ),
    ),
    panel_width: float = typer.Option(
        figure_contest_map._PANEL_W_IN,
        "--panel-width",
        help=(
            "Width of one panel in inches. Set it to the printed width a panel "
            "will occupy; the site labels are a fixed point size, so this is "
            "what decides whether they can be separated at all."
        ),
    ),
    labels: bool = typer.Option(
        True,
        "--labels/--no-labels",
        help=(
            "Draw the per-site `a | b | total` counts. --no-labels leaves the "
            "markers and sends the counts to the CSV twin only, which is what "
            "makes a three-panel row printable at textwidth."
        ),
    ),
    nside: int = typer.Option(
        figure_contest_map.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's answer space and *_tgs.parquet to read.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Which of two methods wins each site on the cell axis: a map per dataset.

    One marker per site, shaped and coloured by whether the two methods tied,
    one of them won outright, or neither reached the site, with the correct
    counts behind that verdict printed underneath as `a | b | total`.
    Categories compare counts directly -- no majority rule -- and `solved_mask`
    is applied, so a FALLBACK row never reaches a count.

    The category counts and the exact McNemar test over the discordant sites
    are written to the manifest and the CSV twin, not drawn on the figure.

    Writes `contest_map.<A>-vs-<B>.{png,csv,manifest.json}` into
    `_cross/contest-map/<datasets>[@<arm>]/`. Needs `build-answer-space` and
    `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id; this figure compares named datasets")
    try:
        png = figure_contest_map.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            method_a=method_a,
            method_b=method_b,
            extent=tuple(extent),
            ncols=ncols,
            panel_width=panel_width,
            labels=labels,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"wrote {png}")


@app.command("plot-peripherality")
def plot_peripherality_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run (repeatable). Their sites are pooled."
    ),
    method_a: str = typer.Option(
        figure_peripherality.DEFAULT_METHOD_A, "--method-a", help="First method."
    ),
    method_b: str = typer.Option(
        figure_peripherality.DEFAULT_METHOD_B, "--method-b", help="Second method."
    ),
    nside: int = typer.Option(
        figure_peripherality.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's answer space and *_tgs.parquet to read.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """How far out the sites each method wins sit: two boxes on one axis.

    Distance is to the run's seed-cloud centroid in EPSG:5070, min-max
    normalised over every pooled site so the axis reads as position within the
    range of peripherality the answer space has. The raw kilometres are in the
    CSV twin, one row per site including the tied and neither ones the figure
    does not draw.

    Writes `peripherality.<A>-vs-<B>.{png,csv,manifest.json}` into
    `_cross/peripherality/<datasets>[@<arm>]/`. Needs `build-answer-space` and
    `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id; this figure pools named datasets")
    try:
        png = figure_peripherality.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            method_a=method_a,
            method_b=method_b,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"wrote {png}")


@app.command("plot-stability")
def plot_stability_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run (repeatable). Their sites are pooled."
    ),
    method_a: str = typer.Option(
        figure_stability.DEFAULT_METHOD_A, "--method-a", help="First method."
    ),
    method_b: str = typer.Option(
        figure_stability.DEFAULT_METHOD_B, "--method-b", help="Second method."
    ),
    figure: list[str] = typer.Option(
        None,
        "--figure",
        "-f",
        help=(
            f"Which to draw: {figure_stability.FIGURES[0]} or "
            f"{figure_stability.FIGURES[1]}. Repeatable; default both."
        ),
    ),
    nside: int = typer.Option(
        figure_stability.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's answer space and *_tgs.parquet to read.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """How consistently each method answers the replicas of one site.

    Two figures over the pooled sites, filed separately so a paper can place
    them apart: a violin of the share of a site's targets placed in the
    correct cell, and a boxplot of the standard deviation of the grid offset
    across them. Both use `solved_mask`, so a FALLBACK row's baseline
    coordinate never enters either.

    The spread figure is the one number in this family that does not depend on
    the cell metric at all. Its whiskers are p5 and p95 with no outliers, so
    read `spread_max` from the manifest before quoting a range.

    Writes `paired_ratio_of_success.<a>_vs_<b>.{png,csv,manifest.json}` and
    `paired_std_grid_offset.<a>_vs_<b>.*` into
    `_cross/stability/<datasets>[@<arm>]/`. Needs `build-answer-space` and
    `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id; this figure pools named datasets")
    try:
        pngs = figure_stability.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            method_a=method_a,
            method_b=method_b,
            figures=list(figure) if figure else None,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for png in pngs:
        typer.echo(f"wrote {png}")


@app.command("plot-error-diff")
def plot_error_diff_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run (repeatable). Their targets are pooled."
    ),
    method_a: str = typer.Option(
        figure_error_diff.DEFAULT_METHOD_A, "--method-a", help="First method; the minuend."
    ),
    method_b: str = typer.Option(
        figure_error_diff.DEFAULT_METHOD_B, "--method-b", help="Second method; the subtrahend."
    ),
    nside: int = typer.Option(
        figure_error_diff.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's answer space and *_tgs.parquet to read.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Where both methods are right, which lands nearer and by how much.

    A CDF of `offset_a - offset_b` over the targets both place in the correct
    cell, zero at the centre and symmetric log either side. Positive means the
    first method is further out. Restricting to the shared correct set is what
    makes the difference paired: comparing each method over its own would put
    them on different populations.

    Writes `paired_error_diff.<a>_vs_<b>.{png,csv,manifest.json}` into
    `_cross/error-diff/<datasets>[@<arm>]/`. Needs `build-answer-space` and
    `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id; this figure pools named datasets")
    try:
        png = figure_error_diff.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            method_a=method_a,
            method_b=method_b,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"wrote {png}")


@app.command("plot-exclusive-error")
def plot_exclusive_error_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Mesh run (repeatable). Their targets are pooled."
    ),
    method_a: str = typer.Option(
        figure_exclusive_error.DEFAULT_METHOD_A, "--method-a", help="First method."
    ),
    method_b: str = typer.Option(
        figure_exclusive_error.DEFAULT_METHOD_B, "--method-b", help="Second method."
    ),
    nside: int = typer.Option(
        figure_exclusive_error.SOURCE_NSIDE,
        "--nside",
        "-n",
        help="Which rung's answer space and *_tgs.parquet to read.",
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """How far out each method is on the targets only it got right.

    The complement of `plot-error-diff`: that compares the two where both
    placed the target correctly, this draws the two cohorts where exactly one
    did. A CDF per cohort of the offset of whichever method was right, on a
    symlog axis so an offset of zero is still drawn.

    Each curve is normalised to its own cohort, so the axis compares shapes;
    the cohort sizes and the tail shares -- of the cohort and of every
    evaluated target both -- are in the manifest.

    Writes `exclusive_error_cdf.<a>_vs_<b>.{png,csv,manifest.json}` into
    `_cross/exclusive-error/<datasets>[@<arm>]/`. Needs `build-answer-space`
    and `classify` on every run.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id; this figure pools named datasets")
    try:
        png = figure_exclusive_error.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            method_a=method_a,
            method_b=method_b,
            nside=nside,
            analysis_root=analysis_root,
        )
    except (ValueError, MissingArtifactError) as exc:
        raise typer.BadParameter(str(exc)) from exc
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


@app.command("plot-rtt-cdf")
def plot_rtt_cdf_cmd(
    run_id: list[str] = typer.Option(
        None, "--run-id", help="Run (repeatable); one curve each, never pooled."
    ),
    x_max: float = typer.Option(
        figure_rtt_cdf.DEFAULT_X_MAX_MS,
        "--x-max",
        help=(
            "Right edge of the linear x axis, in ms. Fixed rather than "
            "data-driven so two runs of this figure are comparable; the "
            "manifest records how much of each dataset lies beyond it."
        ),
    ),
    x_scale: str = typer.Option(
        figure_rtt_cdf.DEFAULT_X_SCALE,
        "--x-scale",
        help=(
            "linear (default) or symlog. Plain log is not offered: symlog keeps "
            "a linear window over the small RTTs and stays defined at 0, so a "
            "source that ever rounds a sub-millisecond RTT down degrades "
            "visibly rather than by dropping the row."
        ),
    ),
    linthresh: float = typer.Option(
        figure_rtt_cdf.DEFAULT_LINTHRESH_MS,
        "--linthresh",
        help=(
            "symlog only: where the axis hands over from linear to log, in ms. "
            "Set it above the bulk of the data and symlog draws a linear axis "
            "under a log label; the manifest records the share below it."
        ),
    ),
    x_step: float = typer.Option(
        None,
        "--x-step",
        help=(
            "Linear scale only: x tick spacing in ms. Unset leaves "
            "matplotlib's automatic locator. Ignored on symlog, where the "
            "decades are the ticks."
        ),
    ),
    medians: bool = typer.Option(
        True,
        "--medians/--no-medians",
        help="Dashed vertical at each dataset's p50, labelled with its RTT.",
    ),
    annotate_clipping: bool = typer.Option(
        True,
        "--annotate-clipping/--no-annotate-clipping",
        help=(
            "Note in the top right, in the curve's own colour, for each "
            "dataset running past --x-max, carrying its p99. Suppressing them "
            "lets a curve stop short of 1.0 unexplained; the manifest records "
            "it either way."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """The RTT distribution of each dataset, one CDF per dataset on one axis.

    Describes the *input*, not a method: one row per (vp, tg) edge at its
    minimum RTT, read from each run's canonical CSV. No answer space and no
    `classify` needed -- this is the only v5 figure that runs on a freshly
    materialized run.

    Nothing is pooled. Each dataset keeps its own denominator, so
    `guard_disjoint_tgs` does not apply and a TG shared by two meshes lands on
    both curves, which is what a reader comparing them wants.

    The x cut is real: a truncated axis looks identical whether the tail
    beyond it is empty or holds a third of the data. The manifest carries
    `share_within_xmax_pct` and `observed_max_ms` per dataset, and each
    clipped curve is named in the top right with its p99 -- a p99 above the
    axis says the truncation is structural rather than one outlier.

    Writes `rtt_cdf.{png,csv,manifest.json}` into
    `_cross/rtt-cdf/<datasets>[@<arm>]/`.
    """
    if not run_id:
        raise typer.BadParameter("pass at least one --run-id")
    try:
        pngs = figure_rtt_cdf.build_for_runs(
            [resolve_run(r, outputs_root) for r in run_id],
            x_max_ms=x_max,
            x_scale=x_scale,
            linthresh_ms=linthresh,
            x_step_ms=x_step,
            show_medians=medians,
            annotate_clipping=annotate_clipping,
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


@app.command("plot-ltd-model")
def plot_ltd_model_cmd(
    run_id: str = typer.Option(None, help="Run to render, one viewer per method."),
    all_runs: bool = typer.Option(
        False, "--all-runs", help="Render every run found under --outputs-root."
    ),
    method: list[str] = typer.Option(
        [],
        "--method",
        "-m",
        help=(
            "Method to render; repeatable. Defaults to every combo in the run. "
            "Unlike plot-mtl-map this appends no shortest_ping control -- the "
            "baseline has no LTD stage, and so no fit to draw."
        ),
    ),
    fold: list[str] = typer.Option(
        [], "--fold", help="Folds to include, e.g. `fold_4`; repeatable. Default: all."
    ),
    tg: list[str] = typer.Option(
        [],
        "--tg",
        help=(
            "Eval TG ids to pre-select in the page's overlay picker; repeatable. "
            "The overlay is off by default; this only decides what is already "
            "ticked on load."
        ),
    ),
    max_points_per_vp: int = typer.Option(
        figure_ltd_model.DEFAULT_MAX_POINTS_PER_VP,
        "--max-points-per-vp",
        help="Scatter cap per VP, deterministically strided. Bounds page weight.",
    ),
    inputs_root: Path = typer.Option(
        None,
        "--inputs-root",
        help=(
            "Root holding materialized benchmark inputs. Default: "
            "inputs/benchmark/v2/. Falls back to the dataset CSV plus its pinned "
            "stratification when a fold is not materialized there."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Interactive RTT-vs-distance viewer per method: scatter, fit band, 2/3 c.

    Writes ltd_model.<method>.html plus a manifest into ltd-model/, which carries
    no `healpix-<n>`: an LTD fit is upstream of both v5 partitions and no nside
    enters it.

    The one v5 command that needs no `build-answer-space`. It reads the fold
    checkpoints directly -- but it does need the fit scatter, which is not in the
    output tree at all, and is resolved from the materialized inputs or rebuilt
    from the dataset CSV's pinned stratification. `manifest.json` records which.
    """
    from scripts.benchmark.v2.inputs import DEFAULT_INPUTS_ROOT

    root_in = Path(inputs_root) if inputs_root is not None else DEFAULT_INPUTS_ROOT

    try:
        runs = _runs(run_id, all_runs, outputs_root)
    except MissingArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc

    for run in runs:
        try:
            written, skipped = figure_ltd_model.build_for_run(
                run,
                methods=list(method) or None,
                fold_ids=list(fold) or None,
                analysis_root=analysis_root,
                inputs_root=root_in,
                max_points_per_vp=max_points_per_vp,
                preselect_tgs=list(tg) or None,
            )
        except MissingArtifactError as exc:
            raise typer.BadParameter(str(exc)) from exc

        for name, path, payload in written:
            # Which route the scatter came from is the one thing a reader
            # cannot infer from the page, so it is echoed rather than left to
            # the manifest alone.
            routes = {p["route"] for p in payload["provenance"].values()}
            n_vps = sum(len(payload["folds"][f]["vps"]) for f in payload["fold_ids"])
            typer.echo(
                f"{run.run_id} {payload['method_label']} ({name}) · "
                f"{payload['ltd']} · {len(payload['fold_ids'])} folds · "
                f"{n_vps} VP panels · scatter from {'+'.join(sorted(routes))}"
            )
            typer.echo(f"wrote {path}")
        for name, why in skipped.items():
            typer.echo(f"{run.run_id} {name} · skipped: {why}")


def _octant_runs(run_id: list[str], all_runs: bool, outputs_root: Path):
    """Runs holding at least one scored arm of any registered Octant sweep.

    `--all-runs` filters rather than taking everything: a mesh or weighted run
    carries no sweep arms, and asking for one would raise on a run the caller
    never meant to include.

    The predicate is EXACT combo-id membership (`variants_present`), not the
    `startswith(f"{v}_") and split("_")[-1] in ARM_TAGS` heuristic it replaced.
    That heuristic pooled every sweep's tags into one namespace, so the moment
    a second sweep contributed a tag it matched production combos: the on-disk
    last segments include `hull`, `spl`, `top`, `geo` and a bare `octant_cbg`.
    """
    if all_runs and run_id:
        raise typer.BadParameter("pass --run-id or --all-runs, not both")
    if run_id:
        return [resolve_run(r, outputs_root) for r in run_id]
    runs = [r for r in discover_runs(outputs_root)
            if octant_finetuning.variants_present(r)]
    if not runs:
        raise typer.BadParameter(
            f"no run under {outputs_root} holds arms of any registered Octant "
            f"sweep; run ./cli.sh --configfile <a config whose combos include "
            f"sweep arms> first")
    return runs


def _octant_variants(variant: list[str], group) -> list[str]:
    """Variant prefixes to render for one group of runs.

    Defaults to what the group actually holds rather than to every registered
    variant: with two sweeps registered, a fixed default would print a skip
    line for the other sweep's variant on every invocation.
    """
    if variant:
        return list(dict.fromkeys(variant))
    present = [octant_finetuning.variants_present(r) for r in group]
    return [v for v in present[0] if all(v in p for p in present[1:])]


def _octant_check_variants(variant: list[str]) -> None:
    known = [v for v, _ in octant_finetuning.VARIANTS]
    if unknown := [v for v in (variant or ()) if v not in known]:
        raise typer.BadParameter(
            f"unknown --variant {unknown}; pick from {known}")


def _octant_layouts(layout, all_runs, runs):
    """Validate --layout and refuse the combinations that are not answerable."""
    layouts = tuple(dict.fromkeys(layout or ())) or (octant_finetuning.PER_RUN,)
    unknown = [x for x in layouts if x not in octant_finetuning.LAYOUTS]
    if unknown:
        raise typer.BadParameter(
            f"unknown --layout {unknown}; pick from "
            f"{list(octant_finetuning.LAYOUTS)}")
    if octant_finetuning.POOLED in layouts:
        if all_runs:
            raise typer.BadParameter(
                f"--layout {octant_finetuning.POOLED} needs explicit --run-id: "
                "which datasets form one population is the caller's call")
        if len(runs) < 2:
            raise typer.BadParameter(
                f"--layout {octant_finetuning.POOLED} needs two or more "
                "--run-id; pooling one run is per-run under another name")
    return layouts


@app.command("plot-octant-cdf")
def plot_octant_cdf_cmd(
    run_id: list[str] = typer.Option(None, "--run-id", help="Sweep run (repeatable)."),
    all_runs: bool = typer.Option(
        False,
        "--all-runs",
        help="Every run under --outputs-root holding arms of any registered "
             "Octant sweep.",
    ),
    variant: list[str] = typer.Option(
        None,
        "--variant",
        help=(
            "Variant prefix, repeatable. Default: whichever the run holds. "
            "Known: " + ", ".join(v for v, _ in octant_finetuning.VARIANTS)
        ),
    ),
    layout: list[str] = typer.Option(
        None,
        "--layout",
        help=(
            f"{octant_finetuning.PER_RUN} (one artifact set per run) or "
            f"{octant_finetuning.POOLED} (every run's targets as one "
            f"population, into _cross/). Repeatable; default: "
            f"{octant_finetuning.PER_RUN}."
        ),
    ),
    dpi: int = typer.Option(300, "--dpi"),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Error-distance CDF across the Octant weight scorers, log x.

    Writes `error_cdf.<variant>.{png,csv,manifest.json}` into
    `octant-finetuning/` -- rung-free, because no answer space enters a
    comparison of weight functions on distance to the raw TG coordinate.

    Every arm the sweep config declares is drawn, hue by decay family and dash
    by steepness rank within it. Only arms from this run appear: the shipped
    `exp(-rtt/50)` lives in the parent mesh run and is deliberately absent, so
    the figure shows what the weight function buys over not weighting at all.

    Being unpaired, it cannot show what an arm costs on the targets it makes
    worse -- the steep arms differ sharply there while their curves nearly
    coincide. Use `report-octant-finetuning` for that.
    """
    _octant_check_variants(variant)
    runs = _octant_runs(run_id, all_runs, outputs_root)
    layouts = _octant_layouts(layout, all_runs, runs)
    for lay in layouts:
        groups = [[r] for r in runs] if lay == octant_finetuning.PER_RUN else [runs]
        for group in groups:
            for v in _octant_variants(variant, group):
                try:
                    path = octant_finetuning.write_cdf(
                        group, v, analysis_root=analysis_root, dpi=dpi)
                except (MissingArtifactError, ValueError) as exc:
                    typer.echo(f"{'+'.join(r.run_id for r in group)} {v} "
                               f"[{lay}] · skipped: {exc}")
                    continue
                typer.echo(f"{'+'.join(r.run_id for r in group)} {v} "
                           f"[{lay}] · wrote {path}")


@app.command("report-octant-finetuning")
def report_octant_finetuning_cmd(
    run_id: list[str] = typer.Option(None, "--run-id", help="Sweep run (repeatable)."),
    all_runs: bool = typer.Option(
        False,
        "--all-runs",
        help="Every run under --outputs-root holding arms of any registered "
             "Octant sweep.",
    ),
    variant: list[str] = typer.Option(
        None,
        "--variant",
        help=(
            "Variant prefix, repeatable. Default: whichever the run holds. "
            "Known: " + ", ".join(v for v, _ in octant_finetuning.VARIANTS)
        ),
    ),
    layout: list[str] = typer.Option(
        None,
        "--layout",
        help=(
            f"{octant_finetuning.PER_RUN} (one artifact set per run) or "
            f"{octant_finetuning.POOLED} (every run's targets as one "
            f"population, into _cross/). Repeatable; default: "
            f"{octant_finetuning.PER_RUN}."
        ),
    ),
    baseline: list[str] = typer.Option(
        None,
        "--baseline",
        help=(
            "Sweep arm to pair against, repeatable. Default: each sweep's "
            "own control (" + ", ".join(
                f"{sw.baseline} for {sw.key}"
                for sw in octant_finetuning.SWEEPS) + "). Named in every "
            "filename, so two baselines cannot overwrite each other or be "
            "quoted for one another."
        ),
    ),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 writes."),
) -> None:
    """Paired per-target table for the Octant weight scorers.

    Writes `paired.<variant>.vs-<baseline>.{csv,tex,manifest.json}` into
    `octant-finetuning/`. This is the paired half of the pair: better / worse /
    tie shares against the unweighted control, and the magnitude of each
    conditional on a target moving that way. Ties are around half the roster --
    the arrangement's top face simply does not move -- so the statistic worth
    quoting is the win:loss ratio among targets that did.

    Regressions report n, median and max rather than a high percentile: those
    cohorts hold 4-34 targets, where a p99 is the maximum under another name.
    """
    _octant_check_variants(variant)
    # Hard-refuse only a tag that belongs to NO sweep, so `--baseline unw`
    # still works when the variant list spans both. A tag from the wrong sweep
    # needs no check here: it never lands in that sweep's `present`, so
    # `build_table` raises ValueError and the loop below reports a skip.
    all_arms = {a.tag for sw in octant_finetuning.SWEEPS for a in sw.arms}
    if unknown := [b for b in (baseline or ()) if b not in all_arms]:
        raise typer.BadParameter(
            f"unknown --baseline {unknown}; pick from "
            f"{ {sw.key: [a.tag for a in sw.arms] for sw in octant_finetuning.SWEEPS} }")
    runs = _octant_runs(run_id, all_runs, outputs_root)
    layouts = _octant_layouts(layout, all_runs, runs)
    for lay in layouts:
        groups = [[r] for r in runs] if lay == octant_finetuning.PER_RUN else [runs]
        for group in groups:
            label = "+".join(r.run_id for r in group)
            for v in _octant_variants(variant, group):
                # Default per variant, so one invocation does the right thing
                # for both sweeps at once.
                for b in (baseline or [octant_finetuning.sweep_for(v).baseline]):
                    try:
                        path = octant_finetuning.write_table(
                            group, v, baseline=b, analysis_root=analysis_root)
                    except (MissingArtifactError, ValueError) as exc:
                        typer.echo(f"{label} {v} vs {b} [{lay}] · skipped: {exc}")
                        continue
                    typer.echo(f"{label} {v} vs {b} [{lay}] · wrote {path}")


if __name__ == "__main__":
    app()
