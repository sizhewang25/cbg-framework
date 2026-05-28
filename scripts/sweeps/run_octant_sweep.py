"""Orchestrate the octant_weighted_cbg weight_threshold sweep across the 6 per-ASN corpora.

Pipeline per ASN:
    1. scripts/benchmark/v2/Snakefile     --configfile <stem>.yaml   (5 folds × 10 thresholds)
    2. scripts/analysis/Snakefile         --configfile <stem>.yaml   (merge folds → CDFs)
Then once across all ASN configs:
    3. scripts/visualization/benchmark/v2/mtl_world_map.smk          (world-map render)

Usage::

    python -m scripts.sweeps.run_octant_sweep                       # full run
    python -m scripts.sweeps.run_octant_sweep --asn as7018 --asn as3209
    python -m scripts.sweeps.run_octant_sweep -j 8
    python -m scripts.sweeps.run_octant_sweep --dry-run
    python -m scripts.sweeps.run_octant_sweep --skip-viz
    python -m scripts.sweeps.run_octant_sweep --skip-analysis
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]

# Slug → config stem shared by scripts/benchmark/v2/config and scripts/analysis/config.
CONFIGS = {
    "as7018":  "north_america_as7018_octant_sweep",
    "as7922":  "north_america_as7922_octant_sweep",
    "as3209":  "europe_as3209_octant_sweep",
    "as3215":  "europe_as3215_octant_sweep",
    "as16509": "global_as16509_octant_sweep",
    "as31898": "global_as31898_octant_sweep",
}
DEFAULT_ASNS = list(CONFIGS.keys())


def _run(cmd: list[str], *, dry_run: bool) -> None:
    """Stream-execute one snakemake invocation; bail on non-zero exit."""
    full = cmd + (["-n"] if dry_run else [])
    typer.echo(f"$ {' '.join(full)}")
    result = subprocess.run(full, cwd=REPO_ROOT)
    if result.returncode != 0:
        typer.echo(f"Command failed with exit code {result.returncode}", err=True)
        raise typer.Exit(code=result.returncode)


def main(
    asn: List[str] = typer.Option(
        DEFAULT_ASNS,
        "--asn",
        help=(
            f"ASN slug to run (repeatable). One of: {sorted(CONFIGS)}. "
            "Defaults to all six."
        ),
    ),
    jobs: int = typer.Option(4, "--jobs", "-j", help="Snakemake -j (parallel jobs per stage)."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Pass -n to every snakemake call."),
    skip_analysis: bool = typer.Option(False, "--skip-analysis", help="Skip the analysis-merge stage."),
    skip_viz: bool = typer.Option(False, "--skip-viz", help="Skip the final world-map render."),
) -> None:
    """Run bench → analysis → viz for the octant weight_threshold sweep."""
    unknown = [a for a in asn if a not in CONFIGS]
    if unknown:
        typer.echo(f"Unknown ASN(s): {unknown}. Available: {sorted(CONFIGS)}", err=True)
        raise typer.Exit(code=2)

    for slug in asn:
        stem = CONFIGS[slug]
        typer.echo(f"\n=== [{slug}] benchmark sweep ===")
        _run(
            [
                "snakemake",
                "-s", "scripts/benchmark/v2/Snakefile",
                "--configfile", f"scripts/benchmark/v2/config/{stem}.yaml",
                "-j", str(jobs),
            ],
            dry_run=dry_run,
        )

        if skip_analysis:
            typer.echo(f"=== [{slug}] skipping analysis (--skip-analysis) ===")
            continue

        typer.echo(f"\n=== [{slug}] analysis merge ===")
        _run(
            [
                "snakemake",
                "-s", "scripts/analysis/Snakefile",
                "--configfile", f"scripts/analysis/config/{stem}.yaml",
                "-j", str(jobs),
            ],
            dry_run=dry_run,
        )

    if skip_viz:
        typer.echo("\n=== skipping world-map render (--skip-viz) ===")
        return

    typer.echo("\n=== world-map render (auto-discovers every *_as*.yaml under scripts/benchmark/v2/config) ===")
    _run(
        [
            "snakemake",
            "-s", "scripts/visualization/benchmark/v2/mtl_world_map.smk",
            "-j", str(jobs),
        ],
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(main)
