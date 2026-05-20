# v2 CBG Benchmark CLI

End-to-end benchmark for the `scripts.framework.v2` CBG pipeline (LTD → MTL → CTR).
Captures every stage's intermediate result, timing, and peak memory for
post-hoc forensic analysis.

## Components

| File | Role |
|---|---|
| [cli.py](cli.py) | Typer commands: `materialize-inputs`, `run-combo`, `summarize` |
| [Snakefile](Snakefile) | Parameterizes the (source × slice × combo) grid |
| [sources/](sources/) | DataSource adapters — [vultr_csv.py](sources/vultr_csv.py), [ripe_atlas.py](sources/ripe_atlas.py) |
| [inputs.py](inputs.py) | Materializes a DataSource into three parquets |
| [runner.py](runner.py) | Per-combo fit + geolocate loop with instrumentation |
| [checkpoint.py](checkpoint.py) | Picks LTD checkpoint snapshot (or `.stateless` marker) |
| [instrument.py](instrument.py) | Per-stage timing + tracemalloc peak collector |
| [schema.py](schema.py) | PyArrow schemas — single source of truth for all parquets |
| [config/](config/) | Snakemake configs (smoke, full, ...) |

## Quick start

```bash
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
- **LTDResult / MTLResult / CTRResult**: nested `ltd_predictions`; `mtl_*` and `ctr_*` columns.
- **CBGResult coord + status**: `pred_lat/lon`, `status`, `error`, `error_km`.
- **Runtime per stage**: `ltd_ms`, `mtl_ms`, `ctr_ms`.
- **Peak memory per stage**: `ltd_peak_bytes`, `mtl_peak_bytes`, `ctr_peak_bytes`.

Run-level: `fit_ms`, `fit_peak_bytes`, `run_peak_rss_bytes` live in `run.json`
and roll into `summary.parquet`.

## Tests

```bash
poetry run python -m unittest discover -s scripts/benchmark/v2/tests -t .
```
