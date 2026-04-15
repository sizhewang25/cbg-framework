# CBG Combination Evaluation — Todo

## Phase 0: Discovery & Design
- [ ] Audit `circle_preprocessing()` to document exact filtering logic and edge cases
- [ ] Audit `circle_intersections()` to document the spherical intersection + vertex-filtering pipeline
- [ ] Audit Shapely polygon intersection path (`_circle_to_shapely_polygon()` + `reduce(intersection)`)
- [ ] Audit Octant `compute_feasible_region_weighted()` to understand fused filtering+multilateration
- [ ] Define the canonical data contract for each phase boundary (Phase 1→2→3→4)
- [ ] Decide whether Path D (arithmetic_mean on Shapely geometry) is worth including
- [ ] Decide whether `bounded_spline + circle_removal` (A3/B3/D3) should use outer radius only or full annulus

## Phase 1: Decoupling & Wrappers
- [ ] Create `scripts/analysis/million_scale/cbg_pipeline.py` with the 4-phase framework
- [ ] Implement Phase 1 wrappers: `rtt_to_circles_speed_of_internet()`, `rtt_to_circles_low_envelope()`, `rtt_to_circles_bounded_spline()`
- [ ] Implement Phase 2 wrappers: `filter_redundant_circles()`, `filter_none()` (passthrough)
- [ ] Implement Phase 3 wrappers: `multilaterate_spherical()`, `multilaterate_shapely()`, `multilaterate_weighted_grid()`
- [ ] Implement Phase 4 wrappers: `centroid_arithmetic_mean()`, `centroid_geometric_weighted()`
- [ ] Implement the combination runner: takes (p1, p2, p3, p4) config and returns per-probe results
- [ ] Add fallback logic (closest-VP) consistent across all combinations
- [ ] Verify: A1 (speed_of_internet + circle_removal + spherical + arithmetic_mean) matches `run_million_scale_cbg()` output exactly
- [ ] Verify: A2 (low_envelope + circle_removal + spherical + arithmetic_mean) matches `run_vanilla_cbg()` output exactly

## Phase 2: Evaluation Script
- [ ] Create `scripts/analysis/million_scale/combination_evaluation.py` — main evaluation harness
- [ ] Enumerate all valid combinations and run each through the pipeline
- [ ] Implement Error CDF plot (all combination lines on one figure, with legend and stats box)
- [ ] Implement RTT-Error scatterplot (per-combination scatter of min_rtt vs error_km)
- [ ] Implement Error-Diff CDF (pairwise delta between selected combination pairs)
- [ ] Implement percentile map plots (p5, p25, p50, p75, p95) for each combination
- [ ] Output JSON results with per-combination statistics

## Phase 3: Validation & Polish
- [ ] Review all plots for readability (colors, labels, legends, axis ranges)
- [ ] Sanity-check: combinations with same Phase 1+2+3 but different Phase 4 should isolate centroid-method impact
- [ ] Sanity-check: A1 should match existing Million-Scale CBG results from `evaluate_million_scale.py`
- [ ] Sanity-check: B1 should match existing geometric centroid results from `centroid_comparison.py`
- [ ] Document key findings in report.md
