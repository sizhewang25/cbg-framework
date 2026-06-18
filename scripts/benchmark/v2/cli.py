"""Typer CLI — three commands wired for Snakemake fan-out.

    python -m scripts.benchmark.v2.cli materialize-inputs --source vultr_csv --slice top1
    python -m scripts.benchmark.v2.cli run-combo \
        --source vultr_csv --slice top1 \
        --ltd speed_of_internet --mtl planar_circle --ctr geometric_centroid \
        --run-id smoke-001
    python -m scripts.benchmark.v2.cli summarize --run-id smoke-001

Each command performs one unit of work — Snakemake's rule graph orchestrates
the (source × slice × combo) cross product.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
import typer

from scripts.benchmark.v2 import schema as bench_schema
from scripts.benchmark.v2.inputs import (
    DEFAULT_INPUTS_ROOT,
    inputs_dir_for,
    materialize_inputs,
    outputs_combo_dir,
)
from scripts.benchmark.v2.runner import ComboSpec, run_one_combo
from scripts.benchmark.v2.sources import SOURCES
from scripts.benchmark.v2.sources.base import DataSource as _DataSource

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="v2 CBG benchmark CLI (LTD/MTL/CTR sweeps with per-stage instrumentation).",
)

DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parent / "outputs"


# ---- materialize-inputs ------------------------------------------------------

@app.command("materialize-inputs")
def cmd_materialize_inputs(
    source: str = typer.Option(..., help=f"Data source name. One of: {sorted(SOURCES)}"),
    slice: str = typer.Option(..., help="Slice identifier (source-specific, e.g. 'top1' / 'fold_0')."),
    run_id: str = typer.Option(
        ...,
        help=(
            "Run identifier — becomes a path component "
            "(<inputs_root>/<source>/<run_id>/<setup>/<slice>/). Different runs "
            "that share a source but differ in source_kwargs get parallel trees."
        ),
    ),
    setup: str = typer.Option(
        _DataSource.PROBES_TO_ANCHORS,
        help=(
            "Role assignment for the (probe, anchor) pair. "
            f"One of: {list(_DataSource.ALLOWED_SETUPS)}. "
            "'probes_to_anchors' = probes are VPs, anchors are targets (IMC 2023 primary). "
            "'anchors_to_probes' = anchors are VPs, probes are targets (pressure test)."
        ),
    ),
    source_kwargs: str = typer.Option(
        "{}",
        help=(
            "JSON dict forwarded to the source constructor as **kwargs. "
            "Source-specific knobs live in the yaml's `source_kwargs:` block "
            "(e.g. `{\"stratification_path\": \"datasets/ripe_atlas/stratifications/distgeo/k5_seed42_top20.json\"}` for ripe_atlas)."
        ),
    ),
    inputs_root: Path = typer.Option(
        DEFAULT_INPUTS_ROOT,
        help="Root directory for materialized inputs. Output goes to <root>/<source>/<run_id>/<setup>/<slice>/.",
    ),
    force: bool = typer.Option(False, help="Re-materialize even if manifest already exists."),
) -> None:
    """Pull data from a DataSource into vp_configs / fit_samples / eval_observations parquet."""
    if source not in SOURCES:
        typer.echo(f"Unknown source {source!r}. Available: {sorted(SOURCES)}", err=True)
        raise typer.Exit(code=2)

    source_cls = SOURCES[source]
    src = source_cls(slice=slice, setup=setup, **json.loads(source_kwargs))
    out_dir = inputs_dir_for(src, inputs_root, run_id=run_id)
    if (out_dir / "manifest.json").exists() and not force:
        typer.echo(f"Already materialized at {out_dir} (use --force to overwrite).")
        return

    written = materialize_inputs(src, root=inputs_root, run_id=run_id)
    typer.echo(f"Materialized {written}")


# ---- run-combo ---------------------------------------------------------------

@app.command("run-combo")
def cmd_run_combo(
    source: str = typer.Option(..., help="Source name (must already be materialized)."),
    slice: str = typer.Option(..., help="Slice id (matches materialize step)."),
    setup: str = typer.Option(
        _DataSource.PROBES_TO_ANCHORS,
        help=f"Role assignment (one of {list(_DataSource.ALLOWED_SETUPS)}). Must match the materialize step.",
    ),
    ltd: str = typer.Option(..., help="LTD model name (e.g. speed_of_internet)."),
    mtl: str = typer.Option(..., help="MTL method name (e.g. planar_circle)."),
    ctr: str = typer.Option(..., help="CTR method name (e.g. geometric_centroid)."),
    run_id: str = typer.Option(..., help="Run identifier — groups combos into one output tree."),
    combo_id: Optional[str] = typer.Option(
        None,
        help="Override combo id used in the output path. Defaults to '<ltd>__<mtl>__<ctr>'.",
    ),
    ltd_kwargs: str = typer.Option("{}", help="JSON dict forwarded to the LTD constructor."),
    mtl_kwargs: str = typer.Option("{}", help="JSON dict forwarded to the MTL constructor."),
    ctr_kwargs: str = typer.Option("{}", help="JSON dict forwarded to the CTR constructor."),
    source_kwargs: str = typer.Option(
        "{}",
        help=(
            "JSON dict forwarded to the source constructor as **kwargs (must match "
            "what materialize-inputs received so the inputs path resolves correctly)."
        ),
    ),
    seed: Optional[int] = typer.Option(
        None,
        help=(
            "Base RNG seed for deterministic re-runs. Per-target seeds are derived "
            "via numpy.random.SeedSequence([seed, target_index]) and applied to any "
            "stochastic stage (currently MonteCarloMedoidCTR). Recorded in targets.parquet."
        ),
    ),
    inputs_root: Path = typer.Option(DEFAULT_INPUTS_ROOT, help="Root containing <source>/<slice>/*.parquet."),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Root for run outputs."),
    enable_fallback: bool = typer.Option(True, help="Enable nearest-VP fallback on pipeline failure."),
) -> None:
    """Fit + geolocate one (LTD, MTL, CTR) combo over a materialized slice."""
    if source not in SOURCES:
        typer.echo(f"Unknown source {source!r}. Available: {sorted(SOURCES)}", err=True)
        raise typer.Exit(code=2)
    # Construct only to derive canonical inputs/outputs paths — no I/O happens
    # until iter_* is called, so this is cheap even for RipeAtlasSource.
    src = SOURCES[source](slice=slice, setup=setup, **json.loads(source_kwargs))
    inputs_dir = inputs_dir_for(src, inputs_root, run_id=run_id)
    if not (inputs_dir / "eval_observations.parquet").exists():
        typer.echo(
            f"No inputs at {inputs_dir}. Run 'materialize-inputs --source {source} "
            f"--run-id {run_id} --slice {slice} --setup {setup}' first.",
            err=True,
        )
        raise typer.Exit(code=2)

    cid = combo_id or f"{ltd}__{mtl}__{ctr}"
    spec = ComboSpec(
        combo_id=cid,
        ltd=ltd, mtl=mtl, ctr=ctr,
        ltd_kwargs=json.loads(ltd_kwargs),
        mtl_kwargs=json.loads(mtl_kwargs),
        ctr_kwargs=json.loads(ctr_kwargs),
        base_seed=seed,
    )
    out_dir = outputs_combo_dir(outputs_root, run_id, src, cid)
    run_one_combo(
        spec,
        inputs_dir=inputs_dir,
        out_dir=out_dir,
        run_id=run_id,
        source_name=source,
        slice_name=slice,
        setup_name=setup,
        enable_fallback=enable_fallback,
    )
    typer.echo(f"Combo done: {out_dir}")


# ---- summarize ---------------------------------------------------------------

@app.command("summarize")
def cmd_summarize(
    run_id: str = typer.Option(..., help="Run id to summarize."),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Root containing <run_id>/."),
) -> None:
    """Aggregate every combo's targets.parquet into summary.parquet (one row per combo)."""
    run_root = outputs_root / run_id
    if not run_root.exists():
        typer.echo(f"No such run dir: {run_root}", err=True)
        raise typer.Exit(code=2)

    rows: list[dict] = []
    # New layout: <run_id>/<source>/<setup>/<slice>/<combo>/run.json (4 levels
    # under the run dir). rglob covers any future re-organization too.
    for run_json in run_root.rglob("run.json"):
        rows.append(_summarize_combo(run_json))
    if not rows:
        typer.echo(f"No combos found under {run_root}", err=True)
        raise typer.Exit(code=1)

    import pyarrow as pa
    columns: dict[str, list] = {field.name: [] for field in bench_schema.SUMMARY_SCHEMA}
    for row in rows:
        for name in columns:
            columns[name].append(row.get(name))
    table = pa.table(columns, schema=bench_schema.SUMMARY_SCHEMA)
    out_path = run_root / "summary.parquet"
    pq.write_table(table, out_path)
    typer.echo(f"Wrote {out_path} ({table.num_rows} combos)")


_STAT_QUANTILES = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}


def _stat_block(series) -> dict[str, float | None]:
    """Compute the uniform 7-stat block (p5/p25/p50/p75/p95/mean/std) for a
    pandas Series. Returns None for every stat if the series is empty —
    keeps the parquet shape stable when a combo had no successful targets
    or a stage never ran."""
    s = series.dropna()
    if s.empty:
        return {stat: None for stat in bench_schema.SUMMARY_STATS}
    out: dict[str, float | None] = {
        stat: float(s.quantile(q)) for stat, q in _STAT_QUANTILES.items()
    }
    out["mean"] = float(s.mean())
    # pandas std with n=1 returns NaN; coerce to None for parquet hygiene.
    std = s.std()
    out["std"] = None if std != std else float(std)
    return out


def _summarize_combo(run_json: Path) -> dict:
    """Read run.json + targets.parquet for one combo and produce a SUMMARY_SCHEMA row."""
    meta = json.loads(run_json.read_text())
    combo_dir = run_json.parent
    table = pq.read_table(combo_dir / "targets.parquet")
    df = table.to_pandas()

    sc = meta.get("status_counts", {})

    row: dict = {
        "run_id": meta["run_id"],
        "source": meta["source"],
        # Fallback for run.jsons written before `setup` existed.
        "setup": meta.get("setup", "probes_to_anchors"),
        "slice": meta["slice"],
        "combo_id": meta["combo_id"],
        "ltd": meta["ltd"],
        "mtl": meta["mtl"],
        "ctr": meta["ctr"],
        "n_targets": int(meta["n_targets"]),
        "n_success": int(sc.get("SUCCESS", 0)),
        "n_fallback": int(sc.get("FALLBACK", 0)),
        "n_error": int(sc.get("ERROR", 0)),
        "fit_ms": float(meta.get("fit_ms", 0.0)),
        "fit_alloc_peak_bytes": int(meta.get("fit_alloc_peak_bytes", 0)),
        "fit_rss_peak_bytes": int(meta.get("fit_rss_peak_bytes", 0)),
        "run_baseline_rss_bytes": int(meta.get("run_baseline_rss_bytes", 0)),
        "run_peak_rss_bytes": int(meta["run_peak_rss_bytes"]),
    }
    # Aggregate every per-target metric over the SUCCESS+FALLBACK subset.
    # error_km is naturally NaN on ERROR rows, so dropna in _stat_block does
    # the right thing for that one even before filtering.
    completed = df[df["status"].isin(["SUCCESS", "FALLBACK"])]
    for metric in bench_schema.SUMMARY_METRICS:
        series = completed[metric] if metric in completed.columns else df[metric][:0]
        for stat, value in _stat_block(series).items():
            row[f"{metric}_{stat}"] = value
    return row


# ---- build-airports ----------------------------------------------------------

_OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


@app.command("build-airports")
def cmd_build_airports(
    src_csv: Optional[Path] = typer.Option(
        None, help="Local OurAirports airports.csv. If omitted, downloads the latest."
    ),
    out: Optional[Path] = typer.Option(
        None, help="Output slim parquet path (defaults to the committed-by-convention location)."
    ),
) -> None:
    """(Re)build the slim airport reference parquet from OurAirports.

    Like the other datasets/ reference files, the artifact is regenerated rather
    than committed. Filters to large airports with an IATA code, a
    municipality, and scheduled commercial service (~1,158 worldwide)."""
    import tempfile
    import urllib.request

    from scripts.benchmark.v2.airports import DEFAULT_AIRPORTS_PARQUET, build_slim_airports

    out_path = out or DEFAULT_AIRPORTS_PARQUET

    if src_csv is None:
        tmp = Path(tempfile.mkdtemp()) / "airports.csv"
        typer.echo(f"Downloading {_OURAIRPORTS_URL} ...")
        urllib.request.urlretrieve(_OURAIRPORTS_URL, tmp)
        src_csv = tmp

    slim = build_slim_airports(Path(src_csv), Path(out_path))
    typer.echo(f"Wrote {out_path} ({len(slim)} airports)")


# ---- airport-eval (postprocessing) -------------------------------------------

@app.command("airport-eval")
def cmd_airport_eval(
    run_id: str = typer.Option(..., help="Run id whose targets.parquet files to annotate."),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Root containing <run_id>/."),
    airports: Optional[Path] = typer.Option(
        None, help="Override path to the slim airport parquet (defaults to the committed one)."
    ),
    threshold_km: float = typer.Option(
        40.0, help="City-level threshold (km) for the forgiving airport_match_rate_within_<T>km."
    ),
) -> None:
    """Append closest-airport columns to every combo's targets.parquet (in place).

    Decoupled postprocessing: re-runnable any time and backfills existing runs
    without re-running CBG. Also writes airport_summary.parquet (match rate +
    pred_airport_km stats per combo).
    """
    from scripts.benchmark.v2.airport_eval import process_parquet, summarize_airport  # noqa: F401
    from scripts.benchmark.v2.airports import load_airport_index

    run_root = outputs_root / run_id
    if not run_root.exists():
        typer.echo(f"No such run dir: {run_root}", err=True)
        raise typer.Exit(code=2)

    index = load_airport_index(str(airports) if airports else None)

    import pandas as pd

    rows: list[dict] = []
    for run_json in run_root.rglob("run.json"):
        meta = json.loads(run_json.read_text())
        targets_path = run_json.parent / "targets.parquet"
        if not targets_path.exists():
            continue
        summary = process_parquet(targets_path, index, thresholds=(threshold_km,))
        rows.append(
            {
                "run_id": meta["run_id"],
                "source": meta["source"],
                "setup": meta.get("setup", "probes_to_anchors"),
                "slice": meta["slice"],
                "combo_id": meta["combo_id"],
                **summary,
            }
        )

    if not rows:
        typer.echo(f"No targets.parquet found under {run_root}", err=True)
        raise typer.Exit(code=1)

    out_path = run_root / "airport_summary.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    typer.echo(f"Annotated {len(rows)} combos; wrote {out_path}")


if __name__ == "__main__":
    app()
