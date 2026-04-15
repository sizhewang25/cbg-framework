# CBG Combination Evaluation — Todo

## Phase 0: Discovery & Design
- [x] Audit `circle_preprocessing()` to document exact filtering logic and edge cases — 2026-04-15
- [x] Audit `circle_intersections()` to document the spherical intersection + vertex-filtering pipeline — 2026-04-15
- [x] Audit Shapely polygon intersection path (`_circle_to_shapely_polygon()` + `reduce(intersection)`) — 2026-04-15
- [x] Audit Octant `compute_feasible_region_weighted()` to understand fused filtering+multilateration — 2026-04-15
- [x] Define the canonical data contract for each phase boundary (Phase 1→2→3→4) — 2026-04-15
  - `CircleConstraint` dataclass bridges all phases; `MultilatResult` carries vertices or Shapely region
- [x] Decide whether Path D (arithmetic_mean on Shapely geometry) is worth including — 2026-04-15
  - Yes: `arithmetic_mean` extracts boundary coords from Shapely geometry via `_extract_vertex_coords()`
- [x] Decide whether `bounded_spline + circle_removal` (A3/B3/D3) should use outer radius only or full annulus — 2026-04-15
  - Uses `radius_km` (outer) for legacy tuple conversion; `inner_radius_km` preserved in dataclass

## Phase 1: Decoupling & Wrappers
- [x] Create modular framework under `scripts/framework/` with HF-style architecture — 2026-04-15
  - Base class + registry + per-variant files pattern (18 files, 1,362 lines)
  - `CBGPipeline` with `from_config()` factory replaces standalone wrapper functions
- [x] Implement Phase 1 variants: `speed_of_internet`, `low_envelope`, `bounded_spline` — 2026-04-15
- [x] Implement Phase 2 variants: `redundant_circle`, `none` (passthrough) — 2026-04-15
- [x] Implement Phase 3 variants: `spherical`, `shapely`, `weighted_grid` — 2026-04-15
- [x] Implement Phase 4 variants: `arithmetic_mean`, `geometric_centroid` — 2026-04-15
- [x] Implement the combination runner via `CBGPipeline.geolocate()` + `geolocate_batch()` — 2026-04-15
- [x] Add fallback logic (closest-VP by min RTT) in `CBGPipeline.geolocate()` — 2026-04-15
- [ ] Verify: A1 (speed_of_internet + redundant_circle + spherical + arithmetic_mean) matches `run_million_scale_cbg()` output exactly
- [ ] Verify: A2 (low_envelope + redundant_circle + spherical + arithmetic_mean) matches `run_vanilla_cbg()` output exactly

## Phase 2: Evaluation Harness (`scripts/analysis/cbg_evaluation/`)
- [x] Create `combinations.py` — PipelineSpec registry (9 combos: A1-A3, B1-B3, C1-C3) + 6 diff pairs — 2026-04-15
- [x] Create `evaluate.py` — load_and_prepare(), build_pipeline(), evaluate_combination(), evaluate_all() — 2026-04-15
- [x] Create `plot_error_cdf.py` — N-series Error CDF (generalize evaluate_million_scale pattern) — 2026-04-15
- [x] Create `plot_error_diff_cdf.py` — pairwise error delta CDF (NEW plot type) — 2026-04-15
- [x] Create `plot_rtt_error_scatter.py` — 3×3 subplot grid with binned median trend — 2026-04-15
- [x] Create `plot_percentile_maps.py` — Cartopy maps at p5/p25/p50/p75/p95 (reuses plot_circles_on_map) — 2026-04-15
- [x] Create `run_evaluation.py` — CLI entry point wiring all above — 2026-04-15
- [x] Output JSON results with per-combination statistics — 2026-04-15
  - 8 source files, 1,029 lines total. Commit `855f45c`. End-to-end run: 9.4s, 266 probes.

## Phase 3: Validation & Polish
- [ ] Review all plots for readability (colors, labels, legends, axis ranges)
- [ ] Sanity-check: combinations with same Phase 1+2+3 but different Phase 4 should isolate centroid-method impact
- [ ] Sanity-check: A1 should match existing Million-Scale CBG results from `evaluate_million_scale.py`
- [ ] Sanity-check: B1 should match existing geometric centroid results from `centroid_comparison.py`
- [ ] Document key findings in report.md
