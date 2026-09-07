"""Error distance against class error: two band figures, one renderer.

§2.4(a) claims accuracy and error distance are different metrics and that the
disagreement is itself a finding. The accuracy table reports one axis and the
error CDF the other; these are the only figures that put both on the same
point, so the claim becomes readable rather than rhetorical — and each method
turns out to have its own signature in the pair.

One band per class-error level, one thin line per target inside it, x = the
coordinate error on a log axis.

## Two y modes, because the two quantities are not the same

* **`cells`** — `seeds_crossed`, how many class boundaries lie between the true
  cell and the predicted one. "Two cells away" is a statement about the answer
  space's local density, which is the nearest-seed snapping story §8.1 asks
  about.
* **`rank`** — `tg_seed_rank`, how many seeds sit closer to the estimate than
  the true one. This is the quantity the reported metric is built on:
  `tg_seed_rank < N` *is* top-N.

SCHEMA.md warns these get confused and that they order the methods
differently — on as01, 42% of wrong rows are at rank 1 while 67% are one
boundary away, and by rank Octant-Spline leads SoI while by crossings SoI
leads. They therefore live in one module, so the distinction is documented
once and the two figures share an x axis, a renderer and a denominator; the CLI
exposes them as separate commands so each figure has its own name.

## Density is drawn, not binned and not jittered

Each point is a vertical line spanning its band, in the method's hue at low
alpha, so coincident values darken by overplotting. No bin width to choose and
no random offset to mislead — an x position on the figure is an x position in
the data.

The one limit: alpha accumulation saturates at roughly 1/alpha coincident lines
(about 7 at alpha 0.15). Shortest-Ping and SoI have exact repeated errors,
because many targets share a VP coordinate and the answer *is* that
coordinate, so their darkest stripes stop distinguishing 7 coincident points
from 30. The band's row-share label carries the count, so this costs
within-band shading detail and no reported number.

## The denominator is every target, so band 0 is the accuracy

Band 0 means the method named the true class: `seeds_crossed == 0` and
`tg_seed_rank == 0` are the same event (the crossing matrix is >= 1 off the
diagonal), and both are top-1 correctness. `topn_accuracy.csv` divides by
*every* target and counts fallbacks as failures, so this figure must too —
dividing by solved rows instead would make band 0 disagree with the accuracy
the paper reports. On as02 Vanilla, 123 band-0 rows over 412 targets is 0.298,
its top-1 accuracy exactly; over its 337 solved rows it would read 0.365 and
match nothing.

The bands therefore sum to `1 - fallback_rate` rather than to 1, and the
shortfall is labelled: Vanilla's four bands total 81.8% and the missing 18.2%
is where its fallbacks went.

Commands: `plot-error-vs-cells`, `plot-error-vs-rank`. Both write to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.answer_space import (
    load_answer_space,
    seed_crossing_matrix,
)
from scripts.analysis.v3.modules.confusion import load_scored
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.labels import (
    PUBLISHED_METHODS,
    short_label,
)
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
    _SURFACE,
    method_colors,
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
    grid_slug,
    resolve_run,
)

#: The error axis is shared with `plot-error-cdf` on purpose — same column, same
#: row filter, same bounds — so a reader can put the figures side by side and
#: read one x position across all of them. That module owns the error axis;
#: these borrow it rather than declaring bounds free to drift.
from scripts.analysis.v3.modules.figure_error_cdf import (  # noqa: E402
    DEFAULT_X_MAX_KM,
    ERROR_COLUMN,
    X_MIN_KM,
)


@dataclass(frozen=True)
class YMode:
    """One class-error axis: where its values come from and how they read."""

    key: str
    #: Column on the scored frame, or `None` when the values are walked from the
    #: answer space rather than read off a row.
    column: str | None
    #: Values at or above this fold into a top bucket. Set from the observed
    #: range: crossings reach 4 and rank reaches 7 at `h3-4`, and both tail off
    #: to a couple of percent, so a sparse row per value would be mostly white.
    max_level: int
    axis_label: str
    level0_label: str
    stem: str
    title: str

CELLS = YMode(
    key="cells",
    column=None,
    max_level=3,
    axis_label="Cells away from the true class",
    level0_label="0 · true class",
    stem="error_vs_cells",
    title="Coordinate error against class distance",
)

RANK = YMode(
    key="rank",
    column="tg_seed_rank",
    max_level=5,
    axis_label="Classes closer to the estimate than the true one",
    level0_label="0 · top-1 correct",
    stem="error_vs_rank",
    title="Coordinate error against class rank",
)

MODES: dict[str, YMode] = {CELLS.key: CELLS, RANK.key: RANK}

#: Half-height of a band, in level units. Leaves a 0.4 gap between bands, which
#: is what the median readout above each band needs to not touch the one above.
BAND_HALF = 0.30

#: Per-line alpha. Low enough that ~7 coincident targets read as solid and a
#: lone one is still visible. Past that the band saturates; see the module
#: docstring.
LINE_ALPHA = 0.15
LINE_WIDTH = 0.9


def load_points(
    run: RunPaths,
    mode: YMode,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """One row per (method, placed target), plus each method's row counts.

    `counts[method]["n_targets"]` is every scored row including fallbacks, and
    it is the denominator every share in this figure uses — see the module
    docstring for why solved rows would be the wrong one.
    """
    cls_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    chosen = list(methods) if methods else list(PUBLISHED_METHODS)

    crossings = pos = None
    if mode.column is None:
        space = load_answer_space(
            run.answer_space_dir(root=analysis_root, grid=grid, resolution=resolution)
        )
        crossings = seed_crossing_matrix(space.seeds)
        pos = {int(s): i for i, s in enumerate(space.seeds["seed_id"].to_numpy())}

    frames: list[pd.DataFrame] = []
    counts: dict[str, dict] = {}
    for method in chosen:
        df = load_scored(cls_dir, method)
        n_targets = int(len(df))
        solved = df.loc[io.solved_mask(df)]
        if mode.column is None:
            placed = solved.loc[solved["pred_seed_id"] >= 0]
            values = np.array(
                [
                    crossings[pos[int(t)], pos[int(p)]]
                    for t, p in zip(placed["tg_seed_id"], placed["pred_seed_id"])
                ],
                dtype=int,
            )
        else:
            placed = solved.loc[solved[mode.column] >= 0]
            values = placed[mode.column].to_numpy(dtype=int)
        counts[method] = {
            "n_targets": n_targets,
            "n_solved": int(len(solved)),
            "n_placed": int(len(placed)),
        }
        if placed.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "method_label": short_label(method),
                    "target_id": placed["target_id"].to_numpy(),
                    "error_km": placed[ERROR_COLUMN].to_numpy(dtype=float),
                    "class_error": values,
                    "level": np.clip(values, 0, mode.max_level),
                }
            )
        )
    if not frames:
        raise ValueError(
            f"no placed rows for {chosen} on {run.run_id} at "
            f"{grid_slug(grid, resolution)}; run `classify` first"
        )
    return pd.concat(frames, ignore_index=True), counts


def band_table(
    points: pd.DataFrame, counts: dict[str, dict], mode: YMode
) -> pd.DataFrame:
    """Per (method, band): count, share of **all** targets, error percentiles.

    `share` is over `n_targets`, so `share` at level 0 is the method's top-1
    accuracy and the bands sum to `1 - fallback_rate`. `cumulative_share` makes
    the rank mode's top-N reading direct: through level 2 it is `accuracy_top3`.
    """
    rows: list[dict] = []
    for method, g in points.groupby("method", sort=False):
        n_targets = counts.get(method, {}).get("n_targets", len(g)) or 1
        running = 0
        for level in range(mode.max_level + 1):
            gl = g.loc[g["level"] == level]
            running += len(gl)
            rows.append(
                {
                    "method": method,
                    "method_label": short_label(method),
                    "y_mode": mode.key,
                    "level": level,
                    "n": len(gl),
                    "n_targets": n_targets,
                    "share": round(len(gl) / n_targets, 4),
                    "cumulative_share": round(running / n_targets, 4),
                    "error_km_p25": round(float(gl["error_km"].quantile(0.25)), 3)
                    if len(gl)
                    else np.nan,
                    "error_km_p50": round(float(gl["error_km"].median()), 3)
                    if len(gl)
                    else np.nan,
                    "error_km_p75": round(float(gl["error_km"].quantile(0.75)), 3)
                    if len(gl)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def level_labels(mode: YMode, max_observed: int) -> list[str]:
    """Band tick labels, with a `+` on the top one only when it is a bucket."""
    labels = [mode.level0_label] + [str(i) for i in range(1, mode.max_level)]
    labels.append(
        f"{mode.max_level}+" if max_observed > mode.max_level else str(mode.max_level)
    )
    return labels


def _fmt_km(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


def plot_bands(
    points: pd.DataFrame,
    table: pd.DataFrame,
    out_path: Path,
    *,
    mode: YMode,
    title: str,
    subtitle: str,
    max_observed: int,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> Path:
    """Small multiples, one panel per method, shared log x and shared bands.

    One panel per method rather than six hues in one frame: the bands would
    overlap into one another and the panel already carries the method's
    identity, so colour never has to separate six series here and the
    palette's all-pairs margin is not called on.
    """
    from matplotlib.ticker import FuncFormatter

    methods = [m for m in PUBLISHED_METHODS if m in set(points["method"])]
    methods += [m for m in points["method"].unique() if m not in methods]
    colors = method_colors(methods)
    labels = level_labels(mode, max_observed)

    n_cols = 3
    n_rows = int(np.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.9 * n_cols, 0.62 * (mode.max_level + 1) * n_rows + 2.0 * n_rows + 1.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    by_method = {m: g for m, g in points.groupby("method", sort=False)}
    tb = table.set_index(["method", "level"])

    for idx, ax in enumerate(axes.flat):
        if idx >= len(methods):
            ax.set_visible(False)
            continue
        method = methods[idx]
        g = by_method[method]
        hue = colors[method]
        ax.set_facecolor(_SURFACE)
        shares: dict[int, float] = {}

        for level in range(mode.max_level + 1):
            gl = g.loc[g["level"] == level]
            if len(gl):
                # One line per target, no binning and no jitter: density comes
                # from lines landing on the same x, so a dark stripe is a real
                # cluster of targets rather than a bin that happens to be wide.
                ax.vlines(
                    np.clip(gl["error_km"].to_numpy(dtype=float), min_x_km, max_x_km),
                    level - BAND_HALF,
                    level + BAND_HALF,
                    color=hue,
                    alpha=LINE_ALPHA,
                    linewidth=LINE_WIDTH,
                    zorder=2,
                )
            if (method, level) not in tb.index:
                continue
            row = tb.loc[(method, level)]
            p50 = float(row["error_km_p50"])
            if np.isfinite(p50):
                ax.vlines(
                    p50,
                    level - BAND_HALF - 0.04,
                    level + BAND_HALF + 0.04,
                    color=_C_INK,
                    linewidth=1.5,
                    zorder=4,
                )
                ax.annotate(
                    _fmt_km(p50),
                    xy=(p50, level + BAND_HALF + 0.07),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=_C_INK,
                    zorder=5,
                )
            shares[level] = float(row["share"])

        n_targets = int(tb.loc[(method, 0), "n_targets"])
        drawn = int(table.loc[table["method"] == method, "n"].sum())
        missing = n_targets - drawn
        header = f"n = {n_targets}"
        if missing:
            header += f" · {missing / n_targets * 100:.1f}% fell back"
        ax.annotate(
            header,
            xy=(0.015, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7.5,
            color=_C_MUTED,
            family="monospace",
        )

        ax.set_title(short_label(method), fontsize=10.5, fontweight="bold", color=hue)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(min_x_km, max_x_km)
        ax.set_ylim(-0.7, mode.max_level + 0.7)
        ax.set_yticks(range(mode.max_level + 1))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.grid(True, axis="x", which="major", color=_C_GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_C_AXIS)
        ax.tick_params(colors=_C_MUTED, labelsize=8.5)

        # Shares hang off a twin y axis rather than off annotations at the panel
        # edge: placed text lands in the gap *between* panels, where it reads as
        # belonging to either, and inside the panel it collides with the long
        # tail on as01/as03 (Shortest-Ping's p95 is 3,871 km, ~93% along the log
        # axis). A right-hand tick is aligned with its band and cannot overlap.
        share_ax = ax.twinx()
        share_ax.set_ylim(ax.get_ylim())
        share_ax.set_yticks(range(mode.max_level + 1))
        share_ax.set_yticklabels(
            [f"{shares.get(lv, 0.0) * 100:.1f}%" for lv in range(mode.max_level + 1)],
            fontsize=7.5,
        )
        share_ax.tick_params(axis="y", length=0, colors=_C_INK_2)
        for tick in share_ax.get_yticklabels():
            tick.set_family("monospace")
        for side in ("top", "right", "left", "bottom"):
            share_ax.spines[side].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=0.995)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9.5, color=_C_INK_2)
    fig.text(
        0.5,
        0.075,
        "Error distance to the raw target (km)",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=_C_INK_2,
    )
    fig.supylabel(mode.axis_label, fontsize=10.5, color=_C_INK_2)
    fig.text(
        0.5,
        0.008,
        "One line per target, no binning: a dark stripe is targets sharing an "
        "error. Black rule and figure above each band are its median error.\n"
        "Percentages on the right are shares of all targets, so the top band is "
        "the method's top-1 accuracy and the bands sum to 100% minus fallbacks.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=_C_MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.115, 0.99, 0.945))
    # `tight_layout` spaces the columns for the widest tick label, which leaves
    # the share column sitting midway between two panels where it reads as
    # belonging to either. Tightened afterwards so each panel's shares hug its
    # own right spine.
    fig.subplots_adjust(wspace=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def build(
    run: RunPaths,
    *,
    mode: YMode,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict]:
    """Render and return `(png, points, table, manifest)`."""
    points, counts = load_points(
        run,
        mode,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    table = band_table(points, counts, mode)
    max_observed = int(points["class_error"].max())
    out_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    png = plot_bands(
        points,
        table,
        out_dir / f"{mode.stem}.png",
        mode=mode,
        title=mode.title,
        subtitle=f"{run.run_id} · {grid_slug(grid, resolution)}",
        max_observed=max_observed,
        max_x_km=max_x_km,
        min_x_km=min_x_km,
    )
    manifest = {
        "run_id": run.run_id,
        "grid": {"scheme": grid, "resolution": resolution},
        "y_mode": mode.key,
        "y_axis": mode.axis_label,
        "methods": [m for m in PUBLISHED_METHODS if m in set(points["method"])],
        "error_column": ERROR_COLUMN,
        "max_level_drawn": mode.max_level,
        "max_level_observed": max_observed,
        "top_level_is_a_bucket": max_observed > mode.max_level,
        "n_lines_drawn": int(len(points)),
        "counts": counts,
        "share_denominator": (
            "n_targets — every scored row, fallbacks included. Level 0 is "
            "therefore the method's top-1 accuracy exactly, matching "
            "topn_accuracy.csv, and the bands sum to 1 - fallback_rate rather "
            "than to 1. Dividing by solved rows would make level 0 disagree "
            "with the accuracy the paper reports (as02 Vanilla: 0.365 vs 0.298)."
        ),
        "level_0_meaning": (
            "the method named the true class. seeds_crossed == 0 and "
            "tg_seed_rank == 0 are the same event, since the crossing matrix is "
            ">= 1 off the diagonal, and both are top-1 correctness."
        ),
        "density_encoding": (
            f"one {LINE_WIDTH} pt line per target at alpha {LINE_ALPHA}, no "
            "binning and no jitter, so an x position on the figure is an x "
            "position in the data. Accumulation saturates near "
            f"{int(1 / LINE_ALPHA)} coincident lines; Shortest-Ping and SoI hit "
            "that because many targets share a VP coordinate and the answer is "
            "that coordinate. The band's share label carries the count."
        ),
        "row_policy": (
            "solved rows with a placeable class error are drawn (FALLBACK "
            "excluded per §7.2; Shortest-Ping's all-BASELINE rows counted as "
            "solved), but every share divides by n_targets"
        ),
    }
    return png, points, table, manifest


# ---- CLI --------------------------------------------------------------------


def _run_mode(
    mode: YMode,
    *,
    run_id: list[str] | None,
    all_runs: bool,
    grid: str,
    resolution: list[int],
    sweep: bool,
    method: list[str] | None,
    max_x_km: float,
    min_x_km: float,
    outputs_root: Path,
    analysis_root: Path,
) -> None:
    """Shared body of both commands — they differ only in their `YMode`."""
    if all_runs and run_id:
        raise typer.BadParameter("pass --run-id or --all-runs, not both")
    runs = (
        discover_runs(outputs_root)
        if all_runs
        else [resolve_run(rid, outputs_root) for rid in (run_id or [])]
    )
    if not runs:
        raise typer.BadParameter("pass at least one --run-id, or --all-runs")

    g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
    for run in runs:
        for res in resolutions:
            png, points, table, manifest = build(
                run,
                mode=mode,
                analysis_root=analysis_root,
                grid=g.name,
                resolution=res,
                methods=list(method) if method else None,
                max_x_km=max_x_km,
                min_x_km=min_x_km,
            )
            points.to_csv(png.parent / f"{mode.stem}_points.csv", index=False)
            table.to_csv(png.parent / f"{mode.stem}_bands.csv", index=False)
            (png.parent / f"{mode.stem}_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            top = table.loc[table["level"] == 0].set_index("method_label")["share"]
            best = top.idxmax()
            typer.echo(
                f"{run.run_id} {grid_slug(g.name, res)} {mode.key} · "
                f"{manifest['n_lines_drawn']} lines · "
                f"level-0 share best {best} {top[best]:.3f} -> {png}"
            )


def register(app: typer.Typer) -> None:
    #: Two commands rather than one with a `--y` switch: each figure has its own
    #: name in `--help` and its own config sub-block, and the two answer
    #: different questions (SCHEMA.md §"Fields that are not accurate for the
    #: obvious reading"). They share `_run_mode`, so the axes cannot diverge.
    @app.command("plot-error-vs-cells")
    def plot_error_vs_cells_cmd(
        run_id: list[str] = typer.Option(
            None, "--run-id", help="Runs to render (repeatable), one figure each."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Render every run found under --outputs-root."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        max_x_km: float = typer.Option(
            DEFAULT_X_MAX_KM, "--max-x-km", help="Upper bound of the log x axis (km)."
        ),
        min_x_km: float = typer.Option(
            X_MIN_KM, "--min-x-km", help="Lower bound of the log x axis (km)."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Coordinate error against cells away from the true class.

        Writes error_vs_cells.{png,_points.csv,_bands.csv,_manifest.json} into
        target-cls-accuracy/<grid>-<resolution>/. Needs `classify`.
        """
        _run_mode(
            CELLS,
            run_id=run_id,
            all_runs=all_runs,
            grid=grid,
            resolution=resolution,
            sweep=sweep,
            method=method,
            max_x_km=max_x_km,
            min_x_km=min_x_km,
            outputs_root=outputs_root,
            analysis_root=analysis_root,
        )

    @app.command("plot-error-vs-rank")
    def plot_error_vs_rank_cmd(
        run_id: list[str] = typer.Option(
            None, "--run-id", help="Runs to render (repeatable), one figure each."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Render every run found under --outputs-root."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        max_x_km: float = typer.Option(
            DEFAULT_X_MAX_KM, "--max-x-km", help="Upper bound of the log x axis (km)."
        ),
        min_x_km: float = typer.Option(
            X_MIN_KM, "--min-x-km", help="Lower bound of the log x axis (km)."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Coordinate error against how many classes outrank the true one.

        `tg_seed_rank < N` is top-N, so each band's cumulative share is that
        top-N accuracy: level 0 is top-1, through level 2 is top-3.

        Writes error_vs_rank.{png,_points.csv,_bands.csv,_manifest.json} into
        target-cls-accuracy/<grid>-<resolution>/. Needs `classify`.
        """
        _run_mode(
            RANK,
            run_id=run_id,
            all_runs=all_runs,
            grid=grid,
            resolution=resolution,
            sweep=sweep,
            method=method,
            max_x_km=max_x_km,
            min_x_km=min_x_km,
            outputs_root=outputs_root,
            analysis_root=analysis_root,
        )
