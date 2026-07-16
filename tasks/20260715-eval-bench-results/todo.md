# eval_bench_results.py — Todo

## Phase 0: Setup & Discovery
- [x] Read `eval_source.py` in full to match its docstring/CLI conventions
- [x] Read `targets.parquet` schema (`schema.py`), `runner.py`'s row builder
- [x] Read MTL/CTR/LTD base classes + one concrete class per family
      (planar_circle, planar_annulus_weighted, spherical_circle,
      monte_carlo_medoid) to confirm recomputation is feasible without refit
- [x] Read `_v2_io.py` (discover_combos/group_combos_by_id/load_targets) and
      `cli.py` (postprocessing command conventions)
- [x] Confirm `as7018_us_test01` has a completed run to test against (no
      re-benchmark needed)

## Phase 1: Core metric functions
- [x] `_reconstruct_ltd_results`: join `ltd_predictions` + `eval_observations`
      into `list[LTDResult]` per target
- [x] `_area_km2` / `_n_components` / `_point_in_region`: handle Shapely
      Polygon/MultiPolygon and `list[Coord]` vertex lists (reuses
      `geometric_centroid.py`'s own dedupe + local-projection ordering)
- [x] Participant stats: n_part, dist/RTT min/mean/median, inflation
- [x] Includer/excluder VP stats from `mtl_participants` echoed bands vs. true
      distance, with outer-vs-inner exclusion direction
- [x] Closest-VP / shortest-ping-VP bridge fields (reuse the
      `per_target_table.py` computation, joined in)
- [x] Cell-gap / nearest-other-centroid via the run's existing `cluster-eval`
      output

## Phase 2: Leave-one-out brittleness
- [x] Per-target, per-participant: drop one VP, recompute MTL (registry
      instance from run.json's mtl/mtl_kwargs) + CTR (ctr/ctr_kwargs, seeded
      from stored `seed`), record status flip + error delta
- [x] Aggregate to `loo_any_flip`, `loo_max_error_delta_km`,
      `loo_critical_vp_id`, `loo_n_tested`, `loo_trivial` (n_part<=1 flag)

## Phase 3: Orchestration + CLI
- [x] `compute_bench_metrics(run_dir, inputs_root, ...)` looping combos/folds,
      pooling per target_id
- [x] Wire `eval-bench-results` command in `cli.py` (run_id-based, mirrors
      `airport-eval`/`geo-eval`)
- [x] Write per-combo `<combo_id>_bench_per_target.csv` +
      `summary.parquet`

## Phase 4: Verification
- [x] Run against `as7018_us_test01` for `vanilla_cbg` (planar_circle) and
      `octant_cbg` (planar_annulus_weighted, stochastic CTR) — both completed
      successfully via `cli.py eval-bench-results`
- [x] Spot-check: `recompute_matches_share == 1.0` for both combos — the
      LTDResult reconstruction + fresh MTL recompute exactly match the
      original run's stored success/participant set
- [x] Spot-check: `vanilla_cbg` shows 1/64 eligible targets with
      `loo_any_flip=True` (target `37.10.126.72`, dropping VP
      `70.236.176.160` flips a 5-participant region); a separate 2-participant
      target shows no flip but a 856 km `loo_max_error_delta_km` — brittleness
      shows up as both hard flips and silent error blow-ups, as intended
- [x] Isolated unit check of the untested `spherical_circle` vertex-list path
      (not exercised by `as7018_us_test01`'s combos): a synthetic 0.1°x0.1°
      square near the equator area-converts to ~123 km^2 as expected, and
      point-in-region / degenerate (<3 vertices) cases behave correctly
- [x] Update report.md / lesson.md with findings

## Phase 5: Make LOO optional (perf follow-up)
- [x] Add `compute_loo` param to `_leave_one_out` / `compute_bench_metrics` /
      `eval_bench_results`, plus a `loo_computed` output column distinguishing
      "skipped by request" from "ran, found nothing"
- [x] Wire `--skip-loo` in `cli.py`'s `eval-bench-results` command
- [x] Verify: `--skip-loo` on `vanilla_cbg`+`octant_cbg` finishes in 9.4s
      (vs. >120s with LOO on) and every non-`loo_*` column is byte-identical
      to the full run
