# CBG End-to-End Setting Evaluation — Report

## Implementation Notes

- The evaluation unit is now one `PipelineSpec`.
- Each setting loads data, prepares probe/anchor inputs, fingerprints the data,
  resolves its required RTT-distance model through a cache, builds the pipeline,
  and evaluates probes.
- Speed-of-internet settings still record `model_cache_lookup`, but no model
  fitting phase is expected.
- LP and Octant model fitting are separate benchmark phases:
  `fit_lp_model` and `fit_octant_model`.
- The combined `fitting_model_ms` field remains in per-setting summary data for
  quick comparison.

## Verification

- Focused unit tests passed: `16 tests OK`.
- Full evaluation completed successfully in 201.3 seconds.
- Raw benchmark rows: 16,022.
- `evaluation_summary.json` now reports `benchmark_scope:
  per_setting_end_to_end`.
- Cache behavior was observed in raw benchmark rows:
  `L1` low-envelope miss, `L2` low-envelope hit, `B1` bounded-spline miss,
  later B-series bounded-spline hits.

## Example Setting Timings

- `L1`: `fit_lp_model_ms=30.374`, `setting_total_ms=532.502`.
- `L2`: `fit_lp_model_ms=0.0`, cache hit, `setting_total_ms=953.893`.
- `B1`: `fit_octant_model_ms=88.018`, `setting_total_ms=32821.941`.
- `B2`: `fit_octant_model_ms=0.0`, cache hit, `setting_total_ms=24026.272`.
