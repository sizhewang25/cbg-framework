"""Error distance against class distance: where the two metrics disagree.

§2.4(a) claims accuracy and error distance can disagree, and that the
disagreement is itself a finding. The accuracy table and the error CDF each
report one axis; this figure is the only one that puts them on the same point,
so the claim becomes countable rather than rhetorical.

One point per (method, solved target): x is the coordinate error, y is how many
class boundaries lie between the true cell and the predicted one. `y == 0` is
the correct cell, `y == 1` one boundary out, and so on. Two off-diagonal
regions are the payload:

* **`y == 0` with a large x** — right class, bad coordinate. The answer space
  absorbed the error, which is exactly what a bounded metro-granular criterion
  is *for*. On as02 this is 16.3% of all points.
* **`y >= 1` with a small x** — wrong class despite a good coordinate. This is
  the nearest-seed snapping artifact §8.1 asks about: the estimate was close
  but fell on the wrong side of a boundary. 2.6% on as02, so it is real and
  much rarer than the first.

The two axes do correlate — median error rises monotonically with crossings on
every method (25 → 530 → 757 → 1,448 km for Shortest-Ping on as02) — which is
what makes the off-diagonal points worth naming rather than assuming.

## Every solved row, not just the wrong ones

`confusion_pairs.csv` cannot back this figure: it keeps only rows that were
*wrong* at the reported top-N, so the entire `y == 0` column is absent from it.
Crossings are therefore recomputed here for all solved rows, from the same
`answer_space.seed_crossing_matrix` that module uses — one walk, so a point at
`y == 1` here and a `seeds_crossed == 1` row there mean the same thing.

Row and distance policy are shared with the error CDF (`io.solved_mask`,
`error_to_target_km`) and so is the x range, so the two figures can be read
against each other without rescaling.

Command: `plot-error-vs-cells`. Writes to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`.
"""

from __future__ import annotations

import json
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
#: row filter, same bounds — so a reader can put the two figures side by side
#: and read one x position across both. That module owns the error axis; this
#: one borrows it rather than declaring a second set of bounds free to drift.
from scripts.analysis.v3.modules.figure_error_cdf import (  # noqa: E402
    DEFAULT_X_MAX_KM,
    ERROR_COLUMN,
    X_MIN_KM,
)

FIGURE_PNG = "error_vs_cells.png"
POINTS_CSV = "error_vs_cells_points.csv"
SUMMARY_CSV = "error_vs_cells_summary.csv"
MANIFEST_JSON = "error_vs_cells_manifest.json"

#: Crossing levels drawn. 4 is the most observed at `h3-4` on any operator run
#: and levels 3+ hold 2-5% of points, so the top level is a `3+` bucket rather
#: than its own sparse row.
MAX_LEVEL = 3

#: Vertical spread within a level, in level units. y is ordinal with four values
#: and there are ~400 points per method, so without jitter each level is one
#: opaque line and the density along x is unreadable.
JITTER = 0.3

#: Fixed, so re-running the command does not reshuffle the cloud and make two
#: renders of the same data look like different data.
JITTER_SEED = 0

#: The x reference: an error smaller than this was smaller than the typical
#: half-distance to the nearest other class, so it "should" have landed in the
#: right cell. Read per run from the answer space rather than hardcoded — it is
#: 150-172 km across as01/02/03 at `h3-4`.
MARGIN_QUANTILE = 0.5


def crossings_per_target(
    run: RunPaths,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, float, int]:
    """One row per (method, solved target): error distance and boundaries crossed.

    Returns `(points, margin_km, max_crossings)` — the reference scale and the
    unclipped maximum travel with the points, since both are figure furniture
    that has to come from the same answer space the crossings did.
    """
    space = load_answer_space(
        run.answer_space_dir(root=analysis_root, grid=grid, resolution=resolution)
    )
    crossings = seed_crossing_matrix(space.seeds)
    pos = {int(s): i for i, s in enumerate(space.seeds["seed_id"].to_numpy())}
    margin_km = float(space.seeds["margin_km"].quantile(MARGIN_QUANTILE))

    cls_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    chosen = list(methods) if methods else list(PUBLISHED_METHODS)

    frames: list[pd.DataFrame] = []
    for method in chosen:
        df = load_scored(cls_dir, method)
        df = df.loc[io.solved_mask(df)]
        if df.empty:
            continue
        # A row with no predicted seed cannot be placed on the y axis at all;
        # `load_scored` marks those with pred_seed_id < 0.
        df = df.loc[df["pred_seed_id"] >= 0]
        walked = np.array(
            [
                crossings[pos[int(t)], pos[int(p)]]
                for t, p in zip(df["tg_seed_id"], df["pred_seed_id"])
            ],
            dtype=int,
        )
        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "method_label": short_label(method),
                    "target_id": df["target_id"].to_numpy(),
                    "error_km": df[ERROR_COLUMN].to_numpy(dtype=float),
                    "seeds_crossed": walked,
                    "level": np.clip(walked, 0, MAX_LEVEL),
                }
            )
        )
    if not frames:
        raise ValueError(
            f"no solved rows for {chosen} on {run.run_id} at "
            f"{grid_slug(grid, resolution)}; run `classify` first"
        )
    points = pd.concat(frames, ignore_index=True)
    return points, margin_km, int(points["seeds_crossed"].max())


def summarise(points: pd.DataFrame, *, margin_km: float) -> pd.DataFrame:
    """Per (method, level): count, share and the level's error percentiles.

    Also the two disagreement counts per method, so the figure's annotation and
    any number quoted in the text come from one place.
    """
    rows: list[dict] = []
    for method, g in points.groupby("method", sort=False):
        n = len(g)
        far_but_right = int(((g["level"] == 0) & (g["error_km"] > margin_km)).sum())
        near_but_wrong = int(((g["level"] >= 1) & (g["error_km"] <= margin_km)).sum())
        for level, gl in g.groupby("level", sort=True):
            rows.append(
                {
                    "method": method,
                    "method_label": short_label(method),
                    "level": int(level),
                    "n": len(gl),
                    "share": round(len(gl) / n, 4) if n else np.nan,
                    "error_km_p25": round(float(gl["error_km"].quantile(0.25)), 3),
                    "error_km_p50": round(float(gl["error_km"].median()), 3),
                    "error_km_p75": round(float(gl["error_km"].quantile(0.75)), 3),
                    "n_right_cell_beyond_margin": far_but_right,
                    "n_wrong_cell_within_margin": near_but_wrong,
                    "margin_km": round(margin_km, 3),
                }
            )
    return pd.DataFrame(rows)


def _level_labels(max_crossings: int) -> list[str]:
    labels = ["0 · correct cell"] + [str(i) for i in range(1, MAX_LEVEL)]
    labels.append(f"{MAX_LEVEL}+" if max_crossings > MAX_LEVEL else str(MAX_LEVEL))
    return labels


def plot_error_vs_cells(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    margin_km: float,
    max_crossings: int,
    max_x_km: float = DEFAULT_X_MAX_KM,
    min_x_km: float = X_MIN_KM,
) -> Path:
    """Small multiples, one panel per method, shared log x and shared ordinal y.

    One panel per method rather than one panel with six colours: 2,400 points
    over four y levels in one frame is an ink blot, and the panel *is* the
    method's identity, so colour is redundant here and the palette's all-pairs
    CVD margin never has to carry anything.
    """
    from matplotlib.ticker import FuncFormatter

    methods = [m for m in PUBLISHED_METHODS if m in set(points["method"])]
    methods += [m for m in points["method"].unique() if m not in methods]
    colors = method_colors(methods)
    labels = _level_labels(max_crossings)

    n_cols = 3
    n_rows = int(np.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.7 * n_cols, 3.1 * n_rows + 1.1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    rng = np.random.default_rng(JITTER_SEED)
    by_method = {m: g for m, g in points.groupby("method", sort=False)}
    sm = summary.set_index(["method", "level"])

    for idx, ax in enumerate(axes.flat):
        if idx >= len(methods):
            ax.set_visible(False)
            continue
        method = methods[idx]
        g = by_method[method]
        ax.set_facecolor(_SURFACE)

        ax.axvline(margin_km, color=_C_AXIS, linestyle="--", linewidth=1.1, zorder=1)
        for level in range(MAX_LEVEL + 1):
            ax.axhline(level, color=_C_GRID, linewidth=0.7, zorder=0)

        y = g["level"].to_numpy(dtype=float) + rng.uniform(
            -JITTER, JITTER, size=len(g)
        )
        ax.scatter(
            np.maximum(g["error_km"].to_numpy(dtype=float), min_x_km),
            y,
            s=11,
            color=colors[method],
            alpha=0.42,
            linewidths=0,
            zorder=2,
        )
        # Per-level median: the monotone trend the off-diagonal points depart
        # from, drawn as a rule rather than a marker so it reads against the
        # cloud it summarises.
        for level in range(MAX_LEVEL + 1):
            if (method, level) not in sm.index:
                continue
            p50 = float(sm.loc[(method, level), "error_km_p50"])
            ax.plot(
                [p50, p50],
                [level - JITTER - 0.08, level + JITTER + 0.08],
                color=_C_INK,
                linewidth=1.6,
                alpha=0.75,
                zorder=3,
                solid_capstyle="butt",
            )

        row = summary[summary["method"] == method].iloc[0]
        ax.annotate(
            f"right cell, > margin: {int(row['n_right_cell_beyond_margin'])}\n"
            f"wrong cell, ≤ margin: {int(row['n_wrong_cell_within_margin'])}",
            # Upper-left, not lower-right: the y == 0 row is where the points
            # are, and the panel's top-left is empty in every method because a
            # high crossing count only ever comes with a high error.
            xy=(0.015, 0.96),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7.5,
            color=_C_INK_2,
            family="monospace",
            bbox=dict(
                boxstyle="round", facecolor=_SURFACE, edgecolor=_C_GRID, alpha=0.92
            ),
        )
        ax.set_title(
            short_label(method), fontsize=10.5, fontweight="bold", color=colors[method]
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(min_x_km, max_x_km)
        ax.set_ylim(-0.55, MAX_LEVEL + 0.55)
        ax.set_yticks(range(MAX_LEVEL + 1))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_C_AXIS)
        ax.tick_params(colors=_C_MUTED, labelsize=8.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", color=_C_INK, y=0.995)
    fig.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=9.5, color=_C_INK_2)
    # `fig.supxlabel` self-positions at the figure bottom and lands on top of
    # the footnote below; placed text keeps the two apart at any panel count.
    fig.text(
        0.5,
        0.082,
        "Error distance to the raw target (km)",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=_C_INK_2,
    )
    fig.supylabel("Class boundaries crossed", fontsize=10.5, color=_C_INK_2)
    fig.text(
        0.5,
        0.008,
        f"Dashed rule: {margin_km:.0f} km, the median half-distance to the nearest "
        "other class — an error inside it was small enough to have landed in the "
        "right cell.\nBlack ticks are each level's median error. Solved rows only; "
        "fallbacks excluded (§7.2). Vertical jitter is cosmetic.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=_C_MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.125, 1, 0.95))

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
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict]:
    """Render and return `(png, points, summary, manifest)`."""
    points, margin_km, max_crossings = crossings_per_target(
        run,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    summary = summarise(points, margin_km=margin_km)
    out_dir = run.cls_accuracy_dir(
        root=analysis_root, grid=grid, resolution=resolution
    )
    png = plot_error_vs_cells(
        points,
        summary,
        out_dir / FIGURE_PNG,
        title="Coordinate error against class error",
        subtitle=f"{run.run_id} · {grid_slug(grid, resolution)} · solved rows only",
        margin_km=margin_km,
        max_crossings=max_crossings,
        max_x_km=max_x_km,
        min_x_km=min_x_km,
    )
    n = len(points)
    manifest = {
        "run_id": run.run_id,
        "grid": {"scheme": grid, "resolution": resolution},
        "methods": [m for m in PUBLISHED_METHODS if m in set(points["method"])],
        "n_points": n,
        "error_column": ERROR_COLUMN,
        "margin_km": round(margin_km, 3),
        "margin_basis": (
            f"quantile {MARGIN_QUANTILE} of seeds.csv's margin_km — half the "
            "distance from each seed to its nearest other seed, i.e. the scale at "
            "which a coordinate error becomes a wrong label"
        ),
        "max_crossings_observed": max_crossings,
        "top_level_is_a_bucket": max_crossings > MAX_LEVEL,
        "disagreement": {
            "right_cell_beyond_margin": int(
                ((points["level"] == 0) & (points["error_km"] > margin_km)).sum()
            ),
            "wrong_cell_within_margin": int(
                ((points["level"] >= 1) & (points["error_km"] <= margin_km)).sum()
            ),
            "note": (
                "the two off-diagonal regions §2.4(a) predicts: the answer space "
                "absorbing a coordinate error, and nearest-seed snapping losing a "
                "good coordinate to the wrong side of a boundary"
            ),
        },
        "crossings_basis": (
            "answer_space.seed_crossing_matrix, the same walk confusion_pairs.csv "
            "uses, so a point at y == 1 here and seeds_crossed == 1 there mean the "
            "same thing. Recomputed rather than read from that file because it "
            "keeps only rows wrong at the reported top-N, which is the whole y == 0 "
            "column missing."
        ),
        "row_policy": (
            "solved rows only via io.solved_mask (FALLBACK excluded, §7.2; "
            "Shortest-Ping's BASELINE rows counted as solved), plus rows with a "
            "predicted seed — a row without one cannot be placed on the y axis"
        ),
        "jitter": {"amount": JITTER, "seed": JITTER_SEED, "note": "cosmetic"},
    }
    return png, points, summary, manifest


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
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
            None,
            "--method",
            help="Restrict to these method ids (repeatable). Default: the six "
            "published variants.",
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
        """Coordinate error vs class boundaries crossed, one panel per method.

        Writes error_vs_cells.png + points/summary CSVs + a manifest into
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
                png, points, summary, manifest = build(
                    run,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=res,
                    methods=list(method) if method else None,
                    max_x_km=max_x_km,
                    min_x_km=min_x_km,
                )
                points.to_csv(png.parent / POINTS_CSV, index=False)
                summary.to_csv(png.parent / SUMMARY_CSV, index=False)
                (png.parent / MANIFEST_JSON).write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                d = manifest["disagreement"]
                typer.echo(
                    f"{run.run_id} {grid_slug(g.name, res)} · "
                    f"{manifest['n_points']} points · margin {manifest['margin_km']:.0f} km · "
                    f"right-cell-but-far {d['right_cell_beyond_margin']}, "
                    f"wrong-cell-but-near {d['wrong_cell_within_margin']} -> {png}"
                )
