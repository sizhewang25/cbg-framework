# CBG Phase Benchmark — Plan

## Goals

- Measure per-probe, per-combination phase runtime in milliseconds.
- Measure per-phase Python allocation peak via `tracemalloc`.
- Measure per-phase process RSS before/after via `psutil`.
- Preserve existing geolocation behavior and result semantics.
- Keep instrumentation local to `scripts/analysis/cbg_evaluation`.

## Phase Boundaries

- `distance_estimation`: `pipe.distance.estimate(...)`
- `filtering`: `pipe.filtering.filter(...)`
- `multilateration`: `pipe.multilateration.multilaterate(...)`
- `centroid`: `pipe.centroid.select(...)`
- `pipeline_overhead`: fallback and glue not covered by wrapped component calls
- `total_geolocate`: full `pipe.geolocate_with_metadata(...)` call

## Setup Metrics

- `load_data_ms`
- `fitting_model_ms`: combined LP and Octant model fitting time
- `total_setup_ms`

## Outputs

- `outputs/benchmark_phase_raw.csv`
- `outputs/benchmark_phase_summary.json`
- `evaluation_summary.json` references the benchmark files and setup metrics.

## Non-Goals

- Do not modify core framework phase implementations.
- Do not measure plotting runtime as a CBG phase.
- Do not use seconds in benchmark output fields.
