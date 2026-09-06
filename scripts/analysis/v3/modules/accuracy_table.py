"""The §8.1 headline: per-run, per-method accuracy, fallbacks and error.

Assembles what `classify` already scored into the table the paper prints, plus
the per-run context that makes the numbers readable — how many of each run's
targets were answerable by proximity at all.

## Runs are rows, never a pooled average

`as01`, `as02` and `as03` are the same VP fleet under different peering and
routing conditions, and that difference *is* the comparison §8.1's "clean
network topology matters" subsection rests on. Averaging them would delete the
finding to produce a tidier number. So this command writes one row per (run,
method) and offers no pooling switch.

The operator/public split is likewise the caller's: pass the runs you want in
one table and invoke it again for the others. Classifying a run as "operator" or
"public" here would be a second source of truth about something the run
inventory already knows, and the reason the two are not compared head-to-head is
substantive (§7.3 declines it) rather than schematic.

**`setup` gates nothing.** `plot-pareto` refuses to pool `anchors_to_probes`
with `probes_to_anchors` because a cost frontier over both would compare a
134-VP fleet against a 53-VP one. That guard is deliberately *not* inherited: on
as01-03 the setup name is a placeholder with no meaning, and this table puts
each run on its own row where no such averaging can happen. Mixed setups are
recorded in the manifest as a note instead of refused.

## Context columns are per run, not per method

`geometry_only` / `selection_miss` / `selection_hit` describe the dataset, so
they sit in their own `dataset_context.csv` rather than being repeated down every
method row where they would read as a property of the variant. The markdown
render prints them as a second table directly under the first, which is how §8.1
reads them: the accuracy column means something different when 37% of the
targets have no proximate VP than when none do.

Command: `table-accuracy`. Writes to
`outputs/analysis/v3/_cross/accuracy-table/<dataset-set>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules.classify import TOPN_CSV
from scripts.analysis.v3.modules.cross import cross_dir, short_dataset
from scripts.analysis.v3.modules.diagram.common.labels import PREFERRED_ORDER, label_for
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    grid_slug,
    resolve_run,
)
from scripts.analysis.v3.modules.proximity import TAXONOMY, load_proximity

CROSS_KIND = "accuracy-table"

#: Columns lifted from `topn_accuracy.csv`, in print order. Everything here is
#: already computed by `classify` under its own documented policies (fallbacks
#: are failures in accuracy; excluded from error percentiles) — recomputing any
#: of it would put a second definition behind the same column name.
_METRIC_COLUMNS = (
    "n_targets",
    "accuracy_top1",
    "accuracy_top3",
    "fallback_rate",
    "error_km_p50",
    "error_km_p90",
)


def _method_order(methods) -> list[str]:
    known = [m for m in PREFERRED_ORDER if m in set(methods)]
    return known + sorted(set(methods) - set(known))


def accuracy_rows(
    runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (run, method), read straight out of each run's `topn_accuracy.csv`."""
    frames = []
    for run_id, run in runs.items():
        path = (
            run.cls_accuracy_dir(root=analysis_root, grid=grid, resolution=resolution)
            / TOPN_CSV
        )
        if not path.exists():
            raise MissingArtifactError(
                f"{path} missing; run `classify` on {run_id} at {grid_slug(grid, resolution)}"
            )
        df = pd.read_csv(path)
        if methods:
            df = df[df["method"].isin(methods)]
        df.insert(0, "run_id", run_id)
        df.insert(1, "dataset", short_dataset(run_id))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    missing = [c for c in _METRIC_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"topn_accuracy.csv lacks {missing}; re-run `classify` with those Ns "
            f"in --topn (the table prints top-1 and top-3)"
        )
    out["method_label"] = out["method"].map(label_for)
    order = {m: i for i, m in enumerate(_method_order(out["method"].unique()))}
    out = out.sort_values(
        ["run_id", "method"], key=lambda s: s.map(order) if s.name == "method" else s
    ).reset_index(drop=True)
    return out[["run_id", "dataset", "method", "method_label", *_METRIC_COLUMNS]]


def dataset_context(
    runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
) -> pd.DataFrame:
    """Per-run proximity context: the three §8.2 shares plus VP and seed counts.

    Read from each run's `target-proximity/` meta rather than recomputed, so the
    shares in this table and the strata in `breakdown-accuracy` are the same
    numbers by construction.
    """
    rows = []
    for run_id, run in runs.items():
        meta = load_proximity(
            run.proximity_dir(root=analysis_root, grid=grid, resolution=resolution)
        ).meta
        tax = meta.get("section_8_2_taxonomy", {})
        n = int(meta.get("n_targets", 0)) or 1
        row = {
            "run_id": run_id,
            "dataset": short_dataset(run_id),
            "setup": run.setup,
            "n_vps": meta.get("n_vps"),
            "n_targets": meta.get("n_targets"),
            "n_seeds": meta.get("n_seeds"),
        }
        for term in TAXONOMY:
            row[f"{term}_n"] = tax.get(term)
            row[f"{term}_share"] = (
                round(tax[term] / n, 4) if tax.get(term) is not None else np.nan
            )
        row["zero_variance_flags"] = ";".join(meta.get("zero_variance", []))
        rows.append(row)
    return pd.DataFrame(rows)


def render_markdown(
    accuracy: pd.DataFrame, context: pd.DataFrame, *, grid: str, resolution: int
) -> str:
    """Both tables as GitHub markdown, ready to paste into §8.1.

    Emitted alongside the CSVs because the CSV is for re-analysis and this is for
    the paper, and hand-transcribing between the two is where a digit changes.
    """
    lines = [
        f"# §8.1 accuracy — {grid_slug(grid, resolution)}",
        "",
        "Accuracy counts fallbacks as failures (§7.2); error percentiles are over",
        "solved rows only and measure distance to the raw target, not to its seed.",
        "",
        "| dataset | method | n | top-1 | top-3 | fallback | err p50 (km) | err p90 (km) |",
        "| --- | --- | --: | --: | --: | --: | --: | --: |",
    ]
    for _, r in accuracy.iterrows():
        lines.append(
            f"| {r.dataset} | {r.method_label} | {int(r.n_targets)} | "
            f"{r.accuracy_top1:.3f} | {r.accuracy_top3:.3f} | {r.fallback_rate:.3f} | "
            f"{r.error_km_p50:.1f} | {r.error_km_p90:.1f} |"
        )
    lines += [
        "",
        "## Dataset context",
        "",
        "Per run, not per method. `geometry_only` targets have no measured VP whose",
        "nearest seed is theirs — a ceiling on Shortest-Ping, not on CBG.",
        "",
        "| dataset | setup | VPs | targets | seeds | geometry_only | selection_miss | selection_hit |",
        "| --- | --- | --: | --: | --: | --: | --: | --: |",
    ]
    for _, r in context.iterrows():
        lines.append(
            f"| {r.dataset} | {r.setup} | {r.n_vps} | {r.n_targets} | {r.n_seeds} | "
            + " | ".join(
                f"{int(r[f'{t}_n'])} ({r[f'{t}_share']:.1%})" for t in TAXONOMY
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build(
    runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict]:
    accuracy = accuracy_rows(
        runs,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        methods=methods,
    )
    context = dataset_context(
        runs, analysis_root=analysis_root, grid=grid, resolution=resolution
    )
    markdown = render_markdown(accuracy, context, grid=grid, resolution=resolution)
    setups = sorted({r.setup for r in runs.values()})
    manifest = {
        "runs": sorted(runs),
        "grid": {"scheme": grid, "resolution": resolution},
        "methods": _method_order(accuracy["method"].unique()),
        "setups": setups,
        "pooling": (
            "none, by design: as01/02/03 are one VP fleet under different peering "
            "and routing conditions, and that difference is the comparison §8.1 "
            "rests on. One row per (run, method); no average is offered."
        ),
        "setup_note": (
            "mixed setups in one table"
            if len(setups) > 1
            else "single setup across the selected runs"
        )
        + ". `setup` gates nothing here — on as01-03 it is a placeholder — and "
        "each run keeps its own row, so no cross-setup average can form.",
        "source": (
            "topn_accuracy.csv per run (classify) + target-proximity/meta.json "
            "(build-proximity); nothing is recomputed here"
        ),
    }
    return accuracy, context, markdown, manifest


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("table-accuracy")
    def table_accuracy_cmd(
        run_id: list[str] = typer.Option(
            None, "--run-id", help="Runs for this table (repeatable). One table "
                                   "per invocation; operator and public runs are "
                                   "kept apart by calling this twice."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """The §8.1 headline table: accuracy, fallbacks and error per run and method.

        Writes accuracy_table.<grid>.{csv,md} + dataset_context.<grid>.csv into
        _cross/accuracy-table/<dataset-set>/. Needs `classify` and
        `build-proximity` on every selected run.
        """
        if not run_id:
            raise typer.BadParameter(
                "pass at least one --run-id. This command has no --all-runs: the "
                "operator and public datasets belong in separate tables (§7.3), "
                "and which runs share one is the caller's call."
            )
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=False)
        runs = {rid: resolve_run(rid, outputs_root) for rid in run_id}

        for res in resolutions:
            accuracy, context, markdown, manifest = build(
                runs,
                analysis_root=analysis_root,
                grid=g.name,
                resolution=res,
                methods=list(method) if method else None,
            )
            out_dir = cross_dir(analysis_root, sorted(runs), kind=CROSS_KIND)
            slug = grid_slug(g.name, res)
            accuracy.to_csv(out_dir / f"accuracy_table.{slug}.csv", index=False)
            context.to_csv(out_dir / f"dataset_context.{slug}.csv", index=False)
            (out_dir / f"accuracy_table.{slug}.md").write_text(markdown)
            (out_dir / f"accuracy_table.{slug}.manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            best = accuracy.loc[accuracy["accuracy_top1"].idxmax()]
            typer.echo(
                f"{len(runs)} run(s) x {len(manifest['methods'])} methods at {slug} · "
                f"best top1 {best['accuracy_top1']:.3f} "
                f"({best['method']} on {best['dataset']}) -> {out_dir}"
            )
