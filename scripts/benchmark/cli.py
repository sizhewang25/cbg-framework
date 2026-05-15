"""Typer CLI for scaled Vultr-7 CBG benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from scripts.benchmark.dataset import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_ROOT,
    build_dataset_specs,
    materialize_dataset,
)
from scripts.benchmark.runner import (
    DEFAULT_COMBO_IDS,
    parse_combo_ids,
    run_benchmark_evaluation,
    write_summary_index,
)

app = typer.Typer(no_args_is_help=True)


@app.command("list-datasets")
def list_datasets(
    input_csv: Annotated[Path, typer.Option(help="Source Vultr US-only CSV.")] = DEFAULT_INPUT_CSV,
    max_top_k: Annotated[int, typer.Option(help="Maximum cumulative top-k ASN dataset.")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
) -> None:
    """List cumulative top-k and all-US benchmark dataset specs."""
    specs = build_dataset_specs(input_csv=input_csv, max_top_k=max_top_k)
    rows = [spec.to_dict() for spec in specs]
    if json_output:
        typer.echo(json.dumps(rows, indent=2))
        return

    typer.echo("dataset_id\tprobes\trows\tanchors\tselected_asns")
    for row in rows:
        selected = row["selected_asns"]
        asns = "all" if selected is None else ",".join(str(asn) for asn in selected)
        typer.echo(
            f"{row['dataset_id']}\t{row['n_probes']}\t{row['n_rows']}\t"
            f"{row['n_anchors']}\t{asns}"
        )


@app.command("materialize-dataset")
def materialize_dataset_command(
    dataset_id: Annotated[str, typer.Argument(help="Dataset id: top1..top10 or all_us.")],
    input_csv: Annotated[Path, typer.Option(help="Source Vultr US-only CSV.")] = DEFAULT_INPUT_CSV,
    output: Annotated[
        Optional[Path],
        typer.Option(help="Selected dataset CSV output path."),
    ] = None,
    manifest_output: Annotated[
        Optional[Path],
        typer.Option(help="Selected dataset JSON manifest path."),
    ] = None,
    max_top_k: Annotated[int, typer.Option(help="Maximum cumulative top-k ASN dataset.")] = 10,
) -> None:
    """Materialize one selected benchmark dataset."""
    output = output or DEFAULT_OUTPUT_ROOT / "datasets" / f"{dataset_id}.csv"
    manifest_output = manifest_output or output.with_suffix(".json")
    spec = materialize_dataset(
        dataset_id=dataset_id,
        input_csv=input_csv,
        output_csv=output,
        manifest_output=manifest_output,
        max_top_k=max_top_k,
    )
    typer.echo(json.dumps(spec.to_dict(), indent=2))


@app.command("run-evaluation")
def run_evaluation_command(
    dataset_id: Annotated[str, typer.Argument(help="Dataset id for output metadata.")],
    input_csv: Annotated[Path, typer.Option(help="Full source CSV or preselected CSV.")] = DEFAULT_INPUT_CSV,
    combo_ids: Annotated[
        str,
        typer.Option(help="Comma-separated combo ids."),
    ] = ",".join(DEFAULT_COMBO_IDS),
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Exact evaluation output directory. Defaults to "
                "outputs/vultr7/runs/<run_id>/<dataset_id>."
            ),
        ),
    ] = None,
    run_id: Annotated[
        Optional[str],
        typer.Option(
            help="Run id for default timestamped outputs. Defaults to a UTC timestamp.",
        ),
    ] = None,
    preselected: Annotated[
        bool,
        typer.Option(help="Treat input CSV as already filtered/materialized."),
    ] = False,
    with_maps: Annotated[
        bool,
        typer.Option("--with-maps", help="Generate percentile maps."),
    ] = False,
) -> None:
    """Run selected CBG combinations for one dataset scale."""
    parsed_combo_ids = parse_combo_ids(combo_ids)
    summary_path = run_benchmark_evaluation(
        dataset_id=dataset_id,
        input_csv=input_csv,
        combo_ids=parsed_combo_ids,
        output_dir=output_dir,
        preselected=preselected,
        generate_maps=with_maps,
        run_id=run_id,
    )
    typer.echo(str(summary_path))


@app.command("summarize")
def summarize_command(
    results_root: Annotated[
        Path,
        typer.Option(help="Root containing per-dataset evaluation outputs."),
    ] = DEFAULT_OUTPUT_ROOT,
    output_json: Annotated[
        Optional[Path],
        typer.Option(help="Summary index JSON path."),
    ] = None,
    output_csv: Annotated[
        Optional[Path],
        typer.Option(help="Summary index CSV path."),
    ] = None,
) -> None:
    """Collect per-dataset summaries into JSON and CSV indexes."""
    output_json = output_json or results_root / "scaled_benchmark_summary.json"
    output_csv = output_csv or results_root / "scaled_benchmark_summary.csv"
    write_summary_index(results_root, output_json, output_csv)
    typer.echo(str(output_json))


if __name__ == "__main__":
    app()
