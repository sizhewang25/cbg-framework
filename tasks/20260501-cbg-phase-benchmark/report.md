# CBG Phase Benchmark — Report

**Status**: Completed
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
- Setting-level timing now uses `load_data_ms`, `fit_lp_model_ms`,
  `fit_octant_model_ms`, derived `fitting_model_ms`, and `setting_total_ms`.

## Evaluation Run

Full evaluation completed with benchmark instrumentation enabled.

- Raw rows: `scripts/analysis/cbg_evaluation/outputs/benchmark_phase_raw.csv`
- Summary: `scripts/analysis/cbg_evaluation/outputs/benchmark_phase_summary.json`
- Raw benchmark rows: 16,022
- Combination count: 10
- Total runtime with instrumentation: 201.3s
- Benchmark scope: per-setting end-to-end.
- Cache behavior is recorded through `model_cache_lookup` rows with
  `cache_hit`, `model_family`, and `cache_key`.

Example median phase timings from the benchmark summary:

| Combo | Distance | Filtering | Multilateration | Centroid | Total |
|-------|---------:|----------:|----------------:|---------:|------:|
| S1 | 0.013 ms | 0.203 ms | 0.031 ms | 0.004 ms | 0.385 ms |
| B1 | 0.401 ms | 0.003 ms | 18.116 ms | 84.999 ms | 105.534 ms |
| B5 | 0.408 ms | 0.003 ms | 83.435 ms | 58.108 ms | 146.707 ms |
