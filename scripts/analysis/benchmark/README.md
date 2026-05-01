# Scaled Vultr-7 CBG Benchmark

This workflow scales probe counts while keeping the anchor set fixed to the
seven US Vultr anchors already present in `datasets/cbg_test/vultr_pings_us_only.csv`.

Default scale ladder:

- `top1` through `top10`: cumulative top-k US probe ASNs ranked by unique probe count.
- `all_us`: all US Vultr probes in the source CSV.

Default pipeline combinations:

- `S1,S2,L1,L2,B1,B2,B3,B4`

The benchmark workflow uses Typer and Snakemake and expects the Poetry
environment to satisfy the project Python requirement (`>=3.11,<4.0`).

## CLI

```bash
poetry run python -m scripts.analysis.benchmark.cli list-datasets
poetry run python -m scripts.analysis.benchmark.cli materialize-dataset top1
poetry run python -m scripts.analysis.benchmark.cli run-evaluation top1 --combo-ids S1,S2,L1,L2,B1,B2,B3,B4
```

## Snakemake

Run a dry-run first:

```bash
poetry run snakemake -n -s scripts/analysis/benchmark/Snakefile \
  --configfile scripts/analysis/benchmark/config/vultr7_smoke.yaml
```

Run the full configured workflow:

```bash
poetry run snakemake -s scripts/analysis/benchmark/Snakefile \
  --configfile scripts/analysis/benchmark/config/vultr7_smoke.yaml
```

Outputs are written under `scripts/analysis/benchmark/outputs/vultr7/`.
