# CBG End-to-End Setting Evaluation — Plan

## Goals

- Treat one pipeline combination as the benchmark unit.
- Include data loading and data preparation inside each setting run.
- Cache RTT-distance models only when data fingerprint, method, and method
  parameters are unchanged.
- Preserve per-probe phase benchmarking for framework components.
- Keep plotting and summary generation compatible with per-setting artifacts.

## Benchmark Scope

- `setting_total`: full setting execution.
- `load_data`: load source CBG evaluation data.
- `prepare_data`: build anchor coordinates and per-probe measurement dicts.
- `data_fingerprint`: hash loaded data for model-cache keys.
- `model_cache_lookup`: check whether a fitted RTT-distance model is reusable.
- `fit_lp_model`: fit low-envelope RTT-distance models on cache miss.
- `fit_octant_model`: fit bounded-spline RTT-distance models on cache miss.
- `pipeline_build`: instantiate and fit the configured framework pipeline.
- Existing per-probe phases remain unchanged.

## Cache Key

The cache key is `(model_family, data_fingerprint, model_params)`.

This means model reuse is allowed across settings that differ only in
filtering, multilateration, centroid, or weighted-annulus threshold, but not
across different data contents or model-fitting parameters.

