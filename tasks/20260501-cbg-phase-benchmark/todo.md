# CBG Phase Benchmark — Todo

- [x] Decide memory metrics: report both `tracemalloc` and RSS.
- [x] Decide time unit: milliseconds only.
- [x] Split model-fitting setup buckets into `fit_lp_model_ms` and `fit_octant_model_ms`.
- [x] Keep combined `fitting_model_ms` as a derived convenience field.
- [x] Add benchmark recorder/helper module.
- [x] Wire benchmark runner into evaluation.
- [x] Write raw CSV and aggregate JSON outputs.
- [x] Add benchmark references to `evaluation_summary.json`.
- [x] Add tests for benchmark aggregation and pipeline parity.
- [x] Run focused tests.
- [x] Rerun full evaluation and inspect benchmark outputs.
- [x] Stage/write outputs after each setting in scaled benchmark runs so partial
      results survive if a long run fails before final summary plotting.
- [x] Add timestamped run output directories so repeated benchmark invocations
      do not overwrite previous results by default.
- [x] Refine per-phase memory reporting: keep raw per-phase `tracemalloc`
      current/peak and RSS before/after/delta, and add explicit derived
      high-water fields only if we can label them as incremental process
      high-water contribution rather than true phase-local memory usage.
- [ ] Investigate pipeline optimization for scaled runs, including per-setting
      or per-probe parallelization and algorithmic improvements to expensive
      annulus/Monte Carlo/geometric-centroid paths.
