"""Error-distance CDF, one curve per method, log x.

The companion to the classification tables: §2.4(a) says accuracy and error
distance can disagree and that the disagreement is itself a finding, so the
paper needs both. This is the error half.

## The distance is to the raw target, never through the seed

`classify` writes two error columns per row and this figure reads
`error_to_target_km` — prediction to the ground-truth coordinate. The
alternative, `error_to_tg_seed_km`, routes through the class seed, and because
seeds sit at cell centres that would add the row's `cell_offset_km` (p50 16-20
km at `h3-4`) to *every* answer including the correct ones. The grid exists to
define the classes; it has no business inside a distance that has its own
ground truth. One useful consequence: re-quantizing the answer space changes
accuracy and must leave this figure bit-identical.

## Fallbacks are excluded, and the baseline is not a fallback

A `FALLBACK` row's coordinate is the shortest-ping VP's, so its error is the
baseline's error wearing a variant's name. Pooling it would pull every
variant's curve toward the baseline exactly where the variant failed — the
comparison RQ2 rests on. `io.solved_mask` is the shared predicate, so this
figure's per-method `n` equals `topn_accuracy.csv`'s `n_solved` by
construction.

That predicate also carries the case a hand-written filter gets wrong:
Shortest-Ping's rows are all `BASELINE`, never `SUCCESS`, so
`status == "SUCCESS"` would drop the baseline curve entirely.

## The baseline is grey and dashed

Every other figure in this paper paints Shortest-Ping with its own variant hue.
Here it is grey and dashed instead, because the CDF's job is to show the
variants *against* a reference rather than to compare seven peers, and a
recessive baseline is the convention the v2 plotter established. The dash
carries the distinction on its own, so the grey cannot be confused with the
`_C_OTHER` bucket a non-published method would land in.

Command: `plot-error-cdf`. Writes to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.confusion import load_scored
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.labels import (
    PREFERRED_ORDER,
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

FIGURE_PNG = "error_cdf.png"
CURVE_CSV = "error_cdf_percentiles.csv"
MANIFEST_JSON = "error_cdf_manifest.json"

#: The column this figure is about. Named as a constant so the "not through the
#: seed" decision is greppable rather than a string literal in a plot call.
ERROR_COLUMN = "error_to_target_km"

#: Reported per method beside each curve.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)

#: Reference verticals (km). Neutral ink, **not** the v2 plotter's
#: green/orange/red — those three hexes are Octant-Hull, Vanilla and Spotter in
#: this paper's palette, so coloured guides would read as series.
THRESHOLDS_KM: tuple[int, ...] = (100, 500, 1000)

#: Fixed x range, both bounds, so the three operator runs' curves are read on
#: one axis rather than three auto-fitted ones.
#:
#: The floor is 0.1 km rather than the v2 plotter's 1 km because sub-kilometre
#: errors are **not** rare here — 8 to 20 rows per method across as01/02/03,
#: up to 4.4% of a run's targets, with an observed minimum of 0.135 km. A 1 km
#: floor flattens the left tail of exactly the curves that earned it and pins
#: their p5 to the clamp. 0.1 km sits below every observed value on this data,
#: so nothing is clamped and the figure reports what was measured; the manifest
#: records the count that *would* be clamped so a future run cannot start
#: clipping silently.
X_MIN_KM = 0.1
DEFAULT_X_MAX_KM = 10_000.0


def load_errors(
    run: RunPaths,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Per method, the solved rows' error-to-target distances plus row counts.

    Returns `(errors, counts)`. `counts[method]` carries `n_total`, `n_solved`
    and `n_fallback`, so the figure can print the denominator beside a curve
    that was drawn over a subset of the targets.
    """
    cls_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    chosen = list(methods) if methods else list(PUBLISHED_METHODS)
    errors: dict[str, np.ndarray] = {}
    counts: dict[str, dict] = {}
    for method in chosen:
        df = load_scored(cls_dir, method)
        solved = io.solved_mask(df)
        values = df.loc[solved, ERROR_COLUMN].to_numpy(dtype=float)
        errors[method] = values[np.isfinite(values)]
        counts[method] = {
            "n_total": int(len(df)),
            "n_solved": int(solved.sum()),
            "n_fallback": int((df["status"] == "FALLBACK").sum()),
        }
    return errors, counts


def method_order(methods) -> list[str]:
    """Published order first, then anything else alphabetically.

    The baseline is drawn *last* despite sorting first, so its grey dashed line
    sits on top of the variants rather than under them — see `plot_error_cdf`.
    """
    known = [m for m in PREFERRED_ORDER if m in set(methods)]
    return known + sorted(set(methods) - set(known))


def percentile_table(
    errors: dict[str, np.ndarray], counts: dict[str, dict]
) -> pd.DataFrame:
    """One row per method: counts plus the reported percentiles.

    Interpolation is numpy's **default (linear)**, matching
    `classify.topn_summary`. That is a compatibility constraint rather than a
    preference: both files sit in one directory and both call the column
    `error_km_p50`, and `topn_accuracy.csv` is what the paper's accuracy table
    reads. The v2 plotter used `method="nearest"` so a quoted percentile named
    a real target and lined up with the MTL world-map viewer's bookmarks —
    a good reason there, and worth 0.7 to 1.3 km of disagreement with the
    paper's own error column here, which is not a trade worth making.
    """
    rows: list[dict] = []
    for method in method_order(errors):
        values = errors[method]
        row = {
            "method": method,
            "method_label": short_label(method),
            "is_baseline": method == SHORTEST_PING,
            **counts.get(method, {}),
            "n_plotted": int(len(values)),
        }
        for p in PERCENTILES:
            row[f"error_km_p{p}"] = (
                round(float(np.percentile(values, p)), 3)
                if len(values)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cdf(
    values: np.ndarray, min_x_km: float = X_MIN_KM
) -> tuple[np.ndarray, np.ndarray]:
    """`(x, y)` for an empirical CDF, x clamped up to the log floor.

    The clamp is a rendering concession — a log axis cannot show 0 — and it is
    applied *here only*. `percentile_table` reads the unclamped values, so a
    reported percentile is never the floor in disguise.
    """
    xs = np.sort(np.maximum(values, min_x_km))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def plot_error_cdf(
    errors: dict[str, np.ndarray],
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
    figsize: tuple[float, float] = (8.6, 6.4),
) -> Path:
    """One panel, one curve per method, log x.

    Colour is the variant, except the baseline which is grey and dashed. Six
    curves share one coordinate space, which is the case where the palette's
    all-pairs CVD margin only *warns* — so identity is carried twice over: the
    legend names every curve and the percentile table beside it is the table
    view the contrast rule asks for.
    """
    from matplotlib.ticker import FuncFormatter

    order = method_order(errors)
    colors = method_colors(order)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_SURFACE)

    # --- reference guides, under everything -------------------------------
    for km in THRESHOLDS_KM:
        if min_x_km < km < max_x_km:
            ax.axvline(km, color=_C_GRID, linestyle=":", linewidth=1.2, zorder=1)
            # Labels ride just under the top spine, rotated, rather than along
            # the baseline: on a log axis 500 and 1,000 km sit a third of a
            # decade apart, so horizontal labels there overlap each other and
            # whichever curve happens to be low at that x.
            ax.annotate(
                f"{km}",
                xy=(km, 0.995),
                xytext=(-3, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=_C_MUTED,
                ha="right",
                va="top",
                rotation=90,
                zorder=1,
            )
    ax.axhline(0.5, color=_C_GRID, linestyle="--", linewidth=1.2, zorder=1)

    # --- curves; baseline last so it reads on top -------------------------
    variants = [m for m in order if m != SHORTEST_PING]
    for method in variants + ([SHORTEST_PING] if SHORTEST_PING in order else []):
        values = errors[method]
        if len(values) == 0:
            continue
        xs, ys = _cdf(values, min_x_km)
        is_baseline = method == SHORTEST_PING
        ax.plot(
            xs,
            ys,
            color=_C_MUTED if is_baseline else colors[method],
            linestyle="--" if is_baseline else "-",
            linewidth=2.2 if is_baseline else 2.0,
            alpha=0.95,
            label=f"{short_label(method)} (baseline)"
            if is_baseline
            else short_label(method),
            zorder=3 if is_baseline else 2,
        )

    ax.set_xscale("log")
    # `%g` rather than `ScalarFormatter(set_scientific(False))`, which renders the
    # 0.1 km decade as a bare "0" — an axis that claims the curve starts at zero
    # when it starts at 100 m.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(min_x_km, max_x_km)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Error distance to the raw target (km)", fontsize=10.5, color=_C_INK_2)
    ax.set_ylabel("Fraction of solved targets", fontsize=10.5, color=_C_INK_2)
    ax.grid(True, which="both", color=_C_GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    ax.tick_params(colors=_C_MUTED, labelsize=9)

    legend = ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(_C_INK_2)

    ax.set_title(title, fontsize=13, fontweight="bold", color=_C_INK, pad=16)
    ax.annotate(
        subtitle,
        xy=(0.5, 1.005),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=_C_INK_2,
    )
    ax.annotate(
        "Fallbacks excluded (§7.2): their coordinate is the shortest-ping VP's, so "
        "their error is the baseline's.\nDistance is to the raw target, not through "
        "the class seed, so the grid's quantization stays out of it.",
        xy=(0.5, -0.13),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color=_C_MUTED,
    )

    # --- percentile table, below the legend -------------------------------
    fig.canvas.draw()
    leg_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    header = f"{'':<14}{'solved':>8}" + "".join(f"{'p' + str(p):>7}" for p in PERCENTILES)
    lines = [header]
    for _, row in table.sort_values("error_km_p50").iterrows():
        counts = f"{int(row['n_plotted'])}/{int(row['n_total'])}"
        cells = "".join(
            f"{row[f'error_km_p{p}']:>7.0f}"
            if np.isfinite(row[f"error_km_p{p}"])
            else f"{'—':>7}"
            for p in PERCENTILES
        )
        lines.append(f"{short_label(row['method'])[:14]:<14}{counts:>8}{cells}")
    ax.text(
        0.02,
        leg_bbox.ymin - 0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color=_C_INK_2,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor=_SURFACE, edgecolor=_C_GRID, alpha=0.95),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def build(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> tuple[Path, pd.DataFrame, dict]:
    """Render the figure and return `(png_path, percentile_table, manifest)`."""
    errors, counts = load_errors(
        run,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    if not errors:
        raise ValueError(
            f"no methods found for {run.run_id} at {grid_slug(grid, resolution)}; "
            "run `classify` first"
        )
    table = percentile_table(errors, counts)
    out_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    png = plot_error_cdf(
        errors,
        table,
        out_dir / FIGURE_PNG,
        title="Error distance to the raw target",
        subtitle=f"{run.run_id} · {grid_slug(grid, resolution)} · solved rows only",
        max_x_km=max_x_km,
        min_x_km=min_x_km,
    )
    clamped = {
        m: int((v < min_x_km).sum())
        for m, v in errors.items()
        if (v < min_x_km).any()
    }
    manifest = {
        "run_id": run.run_id,
        "grid": {"scheme": grid, "resolution": resolution},
        "methods": method_order(errors),
        "baseline": SHORTEST_PING,
        "error_column": ERROR_COLUMN,
        "percentiles": list(PERCENTILES),
        "x_axis": {
            "scale": "log",
            "min_km": min_x_km,
            "max_km": max_x_km,
            "fixed_note": (
                "both bounds are fixed rather than auto-fitted, so the operator "
                "runs' curves are read on one axis"
            ),
            "n_clamped_to_floor": clamped,
            "clamp_note": (
                f"a log axis cannot render 0, so the curve clamps errors below "
                f"{min_x_km} km up to the floor. n_clamped_to_floor is empty on "
                "as01/02/03, whose observed minimum is 0.135 km. The clamp is "
                "applied to the drawn curve only — error_cdf_percentiles.csv "
                "reads the unclamped values, so no reported percentile can be "
                "the floor in disguise."
            ),
        },
        "row_policy": (
            "solved rows only, via io.solved_mask — FALLBACK excluded (§7.2), and "
            "Shortest-Ping's all-BASELINE rows counted as solved so the baseline "
            "curve is drawn. Per-method n therefore equals topn_accuracy.csv's "
            "n_solved."
        ),
        "distance_policy": (
            "error_to_target_km: prediction to the ground-truth coordinate. NOT "
            "error_to_tg_seed_km, which routes through the cell-centre seed and "
            "would add cell_offset_km (p50 16-20 km at h3-4) to every answer, "
            "correct ones included."
        ),
        "baseline_encoding": (
            "grey dashed rather than its variant hue: the figure shows variants "
            "against a reference, and the dash keeps the grey distinct from the "
            "_C_OTHER bucket a non-published method would use"
        ),
        "counts": counts,
    }
    return png, table, manifest


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-error-cdf")
    def plot_error_cdf_cmd(
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
            None,
            "--method",
            help="Restrict to these method ids (repeatable). Default: the six "
            "published variants.",
        ),
        max_x_km: float = typer.Option(
            DEFAULT_X_MAX_KM, "--max-x-km", help="Upper bound of the log x axis (km)."
        ),
        min_x_km: float = typer.Option(
            X_MIN_KM,
            "--min-x-km",
            help="Lower bound of the log x axis (km). Errors below it are clamped "
            "up in the drawn curve only; the percentile CSV is unclamped.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Error-distance CDF per method, log x, fallbacks excluded.

        Writes error_cdf.png + error_cdf_percentiles.csv + a manifest into
        target-cls-accuracy/<grid>-<resolution>/. Needs `classify`.
        """
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
                png, table, manifest = build(
                    run,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=res,
                    methods=list(method) if method else None,
                    max_x_km=max_x_km,
                    min_x_km=min_x_km,
                )
                out_dir = png.parent
                table.to_csv(out_dir / CURVE_CSV, index=False)
                (out_dir / MANIFEST_JSON).write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                best = table.loc[table["error_km_p50"].idxmin()]
                typer.echo(
                    f"{run.run_id} {grid_slug(g.name, res)} · "
                    f"{len(manifest['methods'])} methods · "
                    f"best p50 {best['error_km_p50']:.0f} km "
                    f"({best['method_label']}) -> {png}"
                )
