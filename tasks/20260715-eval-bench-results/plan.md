# eval_bench_results.py — Plan

## Background

`scripts/benchmark/v2/eval_source.py` scores a dataset's CBG-friendliness
*before* any CBG run (topology + RTT quality over available VPs). We now want
the output-side twin: given a completed benchmark run's `targets.parquet`
(per-target CBG outcome + nested `ltd_predictions` / `mtl_participants`),
compute metrics that explain *why* a target succeeded or failed, extending
the ad hoc `scripts/analysis/partvp/*.py` feature work into a proper sibling
of `eval_source.py` inside `scripts/benchmark/v2/`.

Discussed and agreed metric set (see conversation): participant-VP counts and
distance/RTT stats, MTL region area + component count, truth-inclusion against
the actual final region (with outer-vs-inner exclusion direction), includer
vs excluder VP stats, the closest-vs-shortest-ping bridge to the precheck,
cell-gap/answer-space context, and — the one requiring real pipeline
recomputation — per-target **leave-one-participant-out (LOO) brittleness**:
does dropping any single participating VP flip MTL success to failure or
blow up the error, formalizing the "SphericalCircleMTL brittle to a single
bad disk" finding ([[finding_spherical_circle_brittle]] in memory).

## Context

- `targets.parquet` schema: `scripts/benchmark/v2/schema.py` — `ltd_predictions`
  (per-VP: vp_id, success, error, upper_km, lower_km — **no vp coords/latency**)
  and `mtl_participants` (per-participant: vp_id, rtt_ms, echoed_lower/upper_km,
  vp_lat/vp_lon — the VPs that decided the region).
- `ltd_predictions` lacks vp coords/latency; must join against the matching
  fold's `eval_observations.parquet` (target_id, vp_id → vp_lat, vp_lon,
  latency_ms) to reconstruct full `LTDResult` objects for recomputation.
- Stages are pure/stateless past `fit()`: `MTLMethod.multilaterate(list[LTDResult])
  -> MTLResult` and `CTRMethod.select_centroid(MTLResult) -> CTRResult` need no
  refit, so LOO can reuse the *actual* combo's MTL/CTR classes (via
  `MTL_REGISTRY`/`CTR_REGISTRY`, constructed from `run.json`'s `mtl`/`ctr` +
  kwargs) rather than reimplementing geometry. This is the key design decision
  that makes LOO both cheap and faithful.
- `mtl_intersection_kind` is persisted but the geometry/area is NOT — area
  must be recomputed by re-running `mtl.multilaterate()` on the reconstructed
  LTDResults (validated against the stored `participating_vp_ids`/`n_mtl_participants`
  as a consistency check).
- Area/point-in-region for both geometry families (Shapely Polygon/MultiPolygon
  from planar MTLs, `list[Coord]` vertex lists from `spherical_circle`) reuse the
  same local-degree-to-km scale factor the codebase's own `_circle_to_planar_polygon`
  uses (km_per_deg_lat=111.0, km_per_deg_lon=111·cos(lat)) — consistent with how
  these regions were actually built, not a separate/conflicting geodesic model.
  Vertex lists are ordered by angle around their centroid before polygonizing
  (circle_intersections returns pairwise crossing points, not polygon order).
- Test config: `scripts/benchmark/v2/config/as7018_us_test01.yaml` (53 RIPE
  AS7018 probes × 78 US anchors, 5 folds, 16 combos across vanilla/million_scale/
  octant/spotter families — mixes `planar_circle` and `planar_annulus_weighted`,
  good MTL-family coverage). Already fully run at
  `scripts/benchmark/v2/outputs/as7018_us_test01/` — no re-run needed for testing.
- CLI convention: decoupled postprocessing scripts (`airport-eval`, `geo-eval`,
  `cluster-score`, `eval-source`) are wired as `typer` subcommands in
  `scripts/benchmark/v2/cli.py`, not standalone `argparse` mains. Add
  `eval-bench-results` there to match.

## Goals

1. `scripts/benchmark/v2/eval_bench_results.py` — core module with per-target
   metric functions + an orchestration entry point over a run's combos/folds.
2. Wire a `cli.py` `eval-bench-results` command (run_id-based, mirrors
   `airport-eval`/`geo-eval`'s discovery-and-annotate pattern).
3. All 7 originally-proposed metric groups implemented, plus the two gaps
   flagged in discussion: MTL area/n_components, truth-inclusion with
   outer/inner exclusion direction, includer/excluder VP stats, and LOO
   brittleness.
4. Verified end-to-end against `as7018_us_test01` (existing run, no
   re-benchmark needed) — at least one combo per MTL family
   (`vanilla_cbg`=planar_circle, `octant_cbg`=planar_annulus_weighted).

## Approach

- One row per (combo_id, target_id), pooled across folds (K-fold disjoint by
  construction, same convention as `per_target_table.py`).
- Reconstruct `LTDResult` list per target by joining `ltd_predictions` with
  `eval_observations.parquet` on `(target_id, vp_id)`.
- Recompute the full MTL result via the registry-instantiated method (sanity
  check against stored `n_mtl_participants`), derive area/n_components/
  truth-inclusion/exclusion-direction from it.
- Includer/excluder stats computed directly from `mtl_participants`' echoed
  bands vs. true VP→target distance (extends the existing "blocker" idea from
  `characterize_failures.py` into per-group stats instead of one fraction).
- Bridge fields (closest VP / shortest-ping VP / agreement) computed the same
  way as `per_target_table.py`, joined in.
- Cell-gap / nearest-other-centroid pulled from the run's existing
  `cluster-eval` output (`<source>/<setup>/clusters/`) via the same
  `cluster_ground_truth` construction `extract_features.py` uses — no
  reclustering.
- LOO: for each target, for each participant VP, drop it from the
  reconstructed LTDResult list, re-run `mtl.multilaterate()` +
  `ctr.select_centroid()` (seeded from the target's stored `seed` for
  stochastic CTRs), record status flip + error delta. Aggregate to
  `loo_any_flip`, `loo_max_error_delta_km`, `loo_critical_vp_id`.
- Outputs: `<run_dir>/bench_eval/<combo_id>_bench_per_target.csv` (detailed,
  per combo) + `<run_dir>/bench_eval_summary.parquet` (one row per combo,
  percentile/share rollup, matching the `airport_summary.parquet` /
  `geo_summary.parquet` convention).

## Caveats

- Area is a planar approximation consistent with how the region was built,
  not a true geodesic area — degrades away from the equator / at large radii
  (same documented caveat as `_circle_to_planar_polygon`).
- `spherical_circle`'s vertex list is reordered by centroid angle before
  polygonizing for area/point-in-region; valid for the roughly-convex regions
  CBG produces here, not a general simple-polygon guarantee.
- LOO reruns MTL+CTR only (not LTD refit) — correct because LTD is fit once
  per combo over training samples, independent of any single target's
  participant set; per-VP LTD predictions are cached and reused as-is.
- `highest_weight_only=True` MTL variants and stochastic CTRs (Monte Carlo
  medoid) need the original per-target `seed` reused for LOO CTR calls to stay
  comparable to the original run's outcome.
