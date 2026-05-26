# v2 CBG Benchmark CLI

End-to-end benchmark for the `scripts.framework.v2` CBG pipeline (LTD → MTL → CTR).
Captures every stage's intermediate result, timing, and peak memory for
post-hoc forensic analysis.

## Components


| File                           | Role                                                                                                                                                                                                                             |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [cli.py](cli.py)               | Typer commands: `materialize-inputs`, `run-combo`, `summarize`                                                                                                                                                                   |
| [Snakefile](Snakefile)         | Parameterizes the (source × slice × combo) grid                                                                                                                                                                                  |
| [sources/](sources/)           | DataSource adapters — [generic_csv.py](sources/generic_csv.py), [vultr_csv.py](sources/vultr_csv.py), [ripe_atlas.py](sources/ripe_atlas.py). See [sources/README.md](sources/README.md) for the contract + how to add your own. |
| [inputs.py](inputs.py)         | Materializes a DataSource into three parquets                                                                                                                                                                                    |
| [runner.py](runner.py)         | Per-combo fit + geolocate loop with instrumentation                                                                                                                                                                              |
| [checkpoint.py](checkpoint.py) | Picks LTD checkpoint snapshot (or `.stateless` marker)                                                                                                                                                                           |
| [instrument.py](instrument.py) | Per-stage timing + tracemalloc peak collector                                                                                                                                                                                    |
| [schema.py](schema.py)         | PyArrow schemas — single source of truth for all parquets                                                                                                                                                                        |
| [config/](config/)             | Snakemake configs (smoke, full, ...)                                                                                                                                                                                             |


## Install

```bash
# From the repo root. Python 3.11 or 3.12 (3.13 lacks wheels for several
# pinned scientific deps — pyenv install 3.12.x if needed).
poetry install
```

This pulls every runtime dep declared in [pyproject.toml](../../../pyproject.toml)
(numpy / pandas / pyarrow / shapely / snakemake / typer / matplotlib / …) plus
dev tools (pytest). Verified clean from a fresh checkout — see the test suite
below.

## Quick start

```bash
# 0. Easiest path for your own data — point generic_csv at a CSV that has the
#    required columns (vp_id, vp_lat, vp_lon, target_id, target_lat,
#    target_lon, rtt_ms). See sources/README.md.
poetry run python -m scripts.benchmark.v2.cli materialize-inputs \
    --source generic_csv --slice all

# 1. Smoke run via Snakemake (recommended)
poetry run snakemake -s scripts/benchmark/v2/Snakefile \
    --configfile scripts/benchmark/v2/config/smoke.yaml -j 4

# 2. Or invoke the three steps manually
poetry run python -m scripts.benchmark.v2.cli materialize-inputs \
    --source vultr_csv --slice top1

poetry run python -m scripts.benchmark.v2.cli run-combo \
    --source vultr_csv --slice top1 --run-id smoke-001 \
    --ltd speed_of_internet --mtl planar_circle --ctr geometric_centroid

poetry run python -m scripts.benchmark.v2.cli summarize --run-id smoke-001
```

## Data flow

```
DataSource ──→ inputs/<source>/<slice>/{vp_configs,fit_samples,eval_observations}.parquet
                                    │
                                    ▼
                          run-combo (fit + 1× geolocate per target)
                                    │
                                    ▼
       outputs/<run_id>/<source>/<slice>/<combo_id>/
           ├── run.json           # combo metadata + fit + run-level RSS (bytes)
           ├── fit_checkpoint.pkl # pickled LTD (or `.stateless` marker)
           └── targets.parquet    # one row per target with full forensics
                                    │
                                    ▼
                              summarize  →  outputs/<run_id>/summary.parquet
```

## Sources

- **vultr_csv** — wraps `datasets/cbg_test/vultr_pings_us_only.csv`.
Slices: `all_us`, `top1`..`top10` (cumulative probe-ASN top-k, deterministic
ranking). Smoke runs here are fast — 7 anchor targets, 266+ VPs.
- **ripe_atlas** — IMC 2023 probes → anchors (the "primary eval" dataset).
Slices: `all_anchors`, `n<K>`. Requires ClickHouse running with the
`ping_10k_to_anchors` table populated.

Both adapters yield the same shape (VPs, FitSamples, EvalTargets), so a combo
spec can be moved between sources by changing one flag.

### Setup (role assignment)

Every CLI command (and the Snakefile config) takes a `--setup` axis that picks
which side of the (probe, anchor) pair is treated as the vantage point:

- `probes_to_anchors` *(default)* — probes are VPs, anchors are targets. Matches
IMC 2023's primary eval direction. `all_us` slice gives 1422 VPs × 7 targets;
`all_anchors` slice gives ~10K VPs × 723 hard-GT targets.
- `anchors_to_probes` — anchors are VPs, probes are targets (the pressure test
from the memory entry). `all_us` slice flips to 7 VPs × 1422 targets;
`all_anchors` flips to 723 VPs × ~12K hard-GT targets.

The setup is part of the inputs/outputs path so the two configurations never
collide:

```
inputs/<source>/<setup>/<slice>/{vp_configs,fit_samples,eval_observations}.parquet
outputs/<run_id>/<source>/<setup>/<slice>/<combo_id>/{run.json, targets.parquet, fit_checkpoint.pkl}
```

## Stage instrumentation

`CBGModel.geolocate(obs, instrument=...)` accepts a callable that returns a
context manager wrapping each stage call. The runner uses
`TimingMemoryInstrument` from [instrument.py](instrument.py), which records
per-stage `(duration_ns, peak_bytes)` via `time.perf_counter_ns` and
`tracemalloc`.

**Caveat — read before staring at the numbers.** For fast stages (~10–100 µs),
the per-call `tracemalloc.start`/`stop` overhead is on the same order as the
stage runtime. The recorded per-target-per-stage numbers are kept because the
spec asks for them, but the trustworthy reading is the aggregated p50/p95/max
across the full target sweep (computed by `summarize` into `summary.parquet`).
For run-level memory, look at `run_peak_rss_bytes` (psutil RSS, not contaminated
by tracemalloc).

## Stats collected (per target)

Per the spec, every row of `targets.parquet` carries:

- **VP / RTT inputs**: `target_id`, `n_obs`, and nested `ltd_predictions` (per-VP `vp_id`, `success`, `error`, `upper_km`, `lower_km`).
- **CBG combo**: stamped on `run.json` (one per combo dir).
- **LTD checkpoint**: pickle in `fit_checkpoint.pkl`, or `.stateless` marker.
- **LTDResult / MTLResult / CTRResult**: nested `ltd_predictions`; `mtl_`* and `ctr_*` columns.
- **CBGResult coord + status**: `pred_lat/lon`, `status`, `error`, `error_km`.
- **Runtime per stage**: `ltd_ms`, `mtl_ms`, `ctr_ms`.
- **Peak memory per stage**: `ltd_peak_bytes`, `mtl_peak_bytes`, `ctr_peak_bytes`.

Run-level: `fit_ms`, `fit_peak_bytes`, `run_peak_rss_bytes` live in `run.json`
and roll into `summary.parquet`.

## Reproducibility

Pass `--seed N` to `run-combo` (or set `seed: N` on a Snakemake combo entry)
to make stochastic stages deterministic. The runner derives a per-target seed
via `numpy.random.SeedSequence([seed, target_index])` and resets the CTR's
internal RNG before each `geolocate` call. The realized per-target seed is
saved in the `seed` column of `targets.parquet`, and the base seed appears as
`base_seed` in `run.json` — replaying any (combo, target) cell reproduces the
exact prediction byte-for-byte.

Without `--seed`, the `seed` column stays NULL and stochastic combos
(currently only `monte_carlo_medoid` CTR) use their built-in random seed.

## Per-target durability

`targets.parquet` is written via a streaming `pq.ParquetWriter`: each target
becomes its own row group, flushed to disk before the next target's
`geolocate` runs. A crash 600 targets into a 723-target sweep leaves 600
completed row groups on disk — Parquet's footer is still only written at
clean close, so the partial file isn't directly readable by `pq.read_table`,
but the row-group data is recoverable.

`fit_checkpoint.pkl` is written before the target loop even starts, so the
fitted LTD is always durable independent of the per-target loop.

## Tests

Tests are stdlib `unittest` — no extra dep needed. Run from the repo root:

```bash
poetry run python -m unittest discover -s scripts/framework/v2/tests -t .
poetry run python -m unittest discover -s scripts/benchmark/v2/tests -t .
```

The `-t .` flag pins the top-level dir to the repo root so the `from scripts.…`
imports inside the tests resolve. 22 + 24 = 46 tests pass on a fresh
`poetry install` against this commit.

## Synthetic data for a stand-alone smoke

If you don't have the Vultr CSV or RIPE Atlas ClickHouse data, generate a
small synthetic CSV that matches `GenericCSVSource`'s schema:

```bash
poetry run python -m scripts.benchmark.v2.sources._make_smoke_csv /tmp/smoke.csv
# wrote /tmp/smoke.csv (750 rows, 25 VPs × 30 targets)
```

Then edit `DEFAULT_CSV` in [sources/generic_csv.py](sources/generic_csv.py) to
point at `/tmp/smoke.csv` and run the smoke benchmark as in [Quick start](#quick-start).