# CBG Phase Benchmark — Report

**Status**: In Progress
**Created**: 2026-05-01

## Design Notes

The benchmark should be evaluation-local. It will wrap pipeline component
methods at runtime so the framework classes remain reusable without benchmark
dependencies or timing code.

`pipeline_overhead` is intentionally reported instead of exact fallback timing.
With wrapper-based instrumentation, fallback lives inside
`CBGPipeline.geolocate_with_metadata()` and is not a separate component method.
The overhead bucket captures fallback plus Python glue code.

## Implementation Notes

- Added an evaluation-local benchmark recorder instead of modifying framework
  phase implementations.
- Component methods are wrapped at runtime for each combination:
  `distance.estimate`, `filtering.filter`, `multilateration.multilaterate`,
  and `centroid.select`.
- `total_geolocate` records total wall time and RSS snapshots. Component rows
  record `tracemalloc` peaks; total rows intentionally skip `tracemalloc`
  because component measurements are nested and reset phase peaks.
- `pipeline_overhead` is derived as total geolocation time minus measured
  component phase time. It covers fallback plus Python glue code.
- Setup timing uses `load_data_ms`, `fitting_model_ms`, and `total_setup_ms`.

## Evaluation Run

Full evaluation completed with benchmark instrumentation enabled.

- Raw rows: `scripts/analysis/cbg_evaluation/outputs/benchmark_phase_raw.csv`
- Summary: `scripts/analysis/cbg_evaluation/outputs/benchmark_phase_summary.json`
- Raw benchmark rows: 15,960
- Combination count: 10
- Total runtime with instrumentation: 194.2s
- Setup benchmark: `load_data_ms=29.701`, `fitting_model_ms=39.823`,
  `total_setup_ms=84.405`

Example median phase timings from the benchmark summary:

| Combo | Distance | Filtering | Multilateration | Centroid | Total |
|-------|---------:|----------:|----------------:|---------:|------:|
| S1 | 0.013 ms | 0.196 ms | 0.025 ms | 0.004 ms | 0.368 ms |
| B1 | 0.368 ms | 0.002 ms | 17.612 ms | 80.116 ms | 99.851 ms |
| B5 | 0.372 ms | 0.002 ms | 80.120 ms | 56.185 ms | 142.560 ms |
