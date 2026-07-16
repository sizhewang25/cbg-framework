# eval_bench_results.py — Lessons

## 2026-07-15

- `targets.parquet`'s `ltd_predictions` column has no VP coordinates or
  latency — only `eval_observations.parquet` (per fold) has those. Any
  post-hoc recomputation over `ltd_predictions` needs a join back to the
  matching fold's inputs, not just the run outputs.
- MTL/CTR stages are stateless past LTD's `fit()` — `mtl.multilaterate()` and
  `ctr.select_centroid()` take no fitted state, so leave-one-out recomputation
  can reuse the exact registered classes from `run.json`'s combo config
  instead of reimplementing any geometry. This made LOO tractable within
  scope; a from-scratch geometric reimplementation would not have been.
- `spherical_circle`'s `MTLResult.intersection` is an *unordered* list of
  pairwise circle-crossing points (`circle_intersections` returns them in
  combination order, not polygon order) — must angle-sort around the centroid
  before treating it as a polygon boundary for area/point-in-region.
- Don't reinvent the vertex-ordering logic for that case: `GeometricCentroidCTR`
  in `ctr/geometric_centroid.py` already has private `_dedupe_vertices` /
  `_local_project` helpers built for exactly this (dedup + circular-mean
  longitude handling so points near ±180° don't break). Importing and reusing
  them means the polygon this script measures area/inclusion over is
  guaranteed to be the same one the run's own CTR reasoned about — a
  from-scratch reimplementation risked a subtly different ordering that would
  silently disagree with the original run.
- Validate a reconstruction pipeline like this by recomputing the *full*
  (non-leave-one-out) result too and diffing against what the run actually
  stored (`recompute_matches`). Running against real data immediately proved
  the join was exact (100% match on both tested combos) instead of leaving
  that as an assumption.
- A brittleness signal needs two channels, not one: `loo_any_flip` (hard
  MTL/CTR failure) missed a real case where dropping a participant left a
  trivially "successful" but wildly wrong region (single remaining disk,
  error +856 km). Tracking `loo_max_error_delta_km` alongside the flip flag
  caught it.
