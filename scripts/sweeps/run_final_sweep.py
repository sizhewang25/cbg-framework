"""Orchestrate the final 16-combo benchmark across the 6 per-ASN corpora.

Pipeline per ASN:
    1. scripts/benchmark/v2/Snakefile     --configfile <stem>.yaml   (5 folds × 16 combos)
    2. scripts/analysis/Snakefile         --configfile <stem>.yaml   (merge folds → CDFs +
                                                                      per_target_table.parquet)
Then once across all ASN configs:
    3. scripts/visualization/benchmark/v2/mtl_world_map.smk          (world-map render)

Note on viz coverage: mtl_world_map.smk auto-discovers every *_as*.yaml under
scripts/benchmark/v2/config and renders combos matching NAMED_COMBOS
(vanilla_cbg, million_scale_cbg, octant_cbg, spotter_cbg) or SWEEP_COMBO_RE
(octant_cbg_t*, octant_cbg_top). The new *_final.yaml siblings introduce
combos like *_geo, *_top_geo, *_hull, *_c100, *_c80 that none of those rules
match — those combos are skipped silently by the viz step. Extend
SWEEP_COMBO_RE / NAMED_COMBOS in mtl_world_map.smk to include them if you
need maps for the new variants.

Usage::

    python -m scripts.sweeps.run_final_sweep                        # full run (bench + analysis + viz)
    python -m scripts.sweeps.run_final_sweep --asn as7018 --asn as3209
    python -m scripts.sweeps.run_final_sweep -j 12
    python -m scripts.sweeps.run_final_sweep --dry-run
    python -m scripts.sweeps.run_final_sweep --skip-viz
    python -m scripts.sweeps.run_final_sweep --no-analysis          # skip analysis stage
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]

# Slug → config stem. Same stem is shared between the bench config and the
# sibling analysis config under scripts/analysis/config/.
CONFIGS = {
    "as7018":  "north_america_as7018_final",
    "as7922":  "north_america_as7922_final",
    "as3209":  "europe_as3209_final",
    "as3215":  "europe_as3215_final",
    "as16509": "global_as16509_final",
    "as31898": "global_as31898_final",
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
    jobs: int = typer.Option(12, "--jobs", "-j", help="Snakemake -j (parallel jobs per stage)."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Pass -n to every snakemake call."),
    with_analysis: bool = typer.Option(
        True,
        "--with-analysis/--no-analysis",
        help=(
            "Run the analysis-merge stage after bench (CDFs + per_target_table.parquet). "
            "Enabled by default; pass --no-analysis to skip."
        ),
    ),
    skip_viz: bool = typer.Option(False, "--skip-viz", help="Skip the final world-map render."),
) -> None:
    """Run bench → analysis (CDFs + per_target_table) → viz for the final 16-combo benchmark."""
    unknown = [a for a in asn if a not in CONFIGS]
    if unknown:
        typer.echo(f"Unknown ASN(s): {unknown}. Available: {sorted(CONFIGS)}", err=True)
        raise typer.Exit(code=2)

    for slug in asn:
        stem = CONFIGS[slug]
        typer.echo(f"\n=== [{slug}] benchmark ===")
        _run(
            [
                "snakemake",
                "-s", "scripts/benchmark/v2/Snakefile",
                "--configfile", f"scripts/benchmark/v2/config/{stem}.yaml",
                "-j", str(jobs),
            ],
            dry_run=dry_run,
        )

        if not with_analysis:
            continue

        analysis_cfg = REPO_ROOT / "scripts" / "analysis" / "config" / f"{stem}.yaml"
        if not analysis_cfg.exists():
            typer.echo(
                f"=== [{slug}] skipping analysis — {analysis_cfg.relative_to(REPO_ROOT)} not found ===",
                err=True,
            )
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
