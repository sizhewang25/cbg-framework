# Scaled Vultr-7 CBG Benchmark

This workflow scales probe counts while keeping the anchor set fixed to the
seven US Vultr anchors already present in `datasets/cbg_test/vultr_pings_us_only.csv`.

Default scale ladder:

- `top1` through `top10`: cumulative top-k US probe ASNs ranked by unique probe count.
- `all_us`: all US Vultr probes in the source CSV.

Default pipeline combinations (combo IDs from `scripts/libs/core/combinations.py`):

- `Vanilla CBG, V2, V3, Million-scale CBG, M2, M3, Octant, O2, O3, O4`

The benchmark workflow uses Typer and Snakemake and expects the Poetry
environment to satisfy the project Python requirement (`>=3.11,<4.0`).

## CLI

```bash
poetry run python -m scripts.benchmark.cli list-datasets
poetry run python -m scripts.benchmark.cli materialize-dataset top1
poetry run python -m scripts.benchmark.cli run-evaluation top1 \
  --combo-ids "Vanilla CBG,V2,V3,Million-scale CBG,M2,M3,Octant,O2,O3,O4"
```

By default, each `run-evaluation` invocation writes to a fresh timestamped
directory:

```text
scripts/benchmark/v1/outputs/vultr7/runs/<run_id>/<dataset_id>/
```

Use `--run-id <id>` to group or resume deterministic outputs. Use
`--output-dir <path>` only when you intentionally want an exact output path;
that option bypasses the timestamped default.

## Single-shot driver

To evaluate all combinations on the legacy `million_scale.load_data()` source
(rather than the scaled top-k sweep), run:

```bash
poetry run python -m scripts.benchmark.run_evaluation
```

Outputs land in `scripts/benchmark/v1/outputs/single_shot/`.

## Snakemake

Run a dry-run first:

```bash
poetry run snakemake -n -s scripts/benchmark/v1/Snakefile \
  --configfile scripts/benchmark/v1/config/vultr7_smoke.yaml
```

Run the full configured workflow:

```bash
poetry run snakemake -s scripts/benchmark/v1/Snakefile \
  --configfile scripts/benchmark/v1/config/vultr7_smoke.yaml
```

Outputs are written under
`scripts/benchmark/v1/outputs/vultr7/runs/<run_id>/`. If `run_id` is not
set in the config or via `--config run_id=<id>`, the Snakefile creates a fresh
UTC timestamp run id for the invocation.

## Checkpoints

Long runs write partial checkpoints after each completed pipeline setting:

- `checkpoints/<combo_id>_probe_results.csv`
- `checkpoints/<combo_id>_checkpoint.json`
- `checkpoints/progress.json`
- refreshed accumulated `evaluation_summary.json`
- refreshed accumulated `benchmark_phase_raw.csv`
- refreshed accumulated `benchmark_phase_summary.json`

Final plots are still generated only after all requested settings complete.
