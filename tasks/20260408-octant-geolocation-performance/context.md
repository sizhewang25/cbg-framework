# Technical Context

## Problem Definition

**Goal**: Speed up Octant geolocation in the million-scale evaluation pipeline.

**Constraint**: Preserve the current geolocation behavior closely enough that
the Octant error CDF remains usable and accuracy does not regress materially.

## Latest Dataset Size

From the latest million-scale evaluation run:

```python
total_measurements = 9866
as_measurements = 1854
unique_anchors = 7
unique_probes = 266
avg_constraints_per_probe = 6.94
```

## Latest Runtime Breakdown

From:

- [comparison_results.json](../../scripts/analysis/million_scale/outputs/comparison/comparison_results.json)

### End-to-End Runtime

```python
total_runtime_sec = 130.275
run_octant_cbg_sec = 121.892
```

### Top-Level Phases

```python
run_octant_cbg_sec = 121.892
plot_percentile_maps_vanilla_sec = 5.084
plot_percentile_maps_ms_sec = 1.766
plot_scatter_sec = 0.756
run_vanilla_cbg_sec = 0.237
run_million_scale_cbg_sec = 0.130
```

### Octant Per-Probe Breakdown

```python
probe_total_sec = 121.888            # 458.23 ms/probe
estimate_location_sec = 121.748      # 457.70 ms/probe
geometric_median_sec = 94.214        # 354.19 ms/probe
weighted_region_sec = 15.890         # 59.74 ms/probe
sample_points_sec = 11.636           # 43.75 ms/probe
collect_rtts_sec = 0.040
form_constraints_sec = 0.032
unweighted_region_sec = 0.002
```

## Main Insight

The feasible region formation is not the dominant bottleneck.

The current hotspot ranking is:

1. `geometric_median_sec`
2. `weighted_region_sec`
3. `sample_points_sec`

So if the goal is pure speed, optimizing geometric median selection should
likely come before major changes to weighted region formation.

## Current Region Selection Outcomes

```python
weighted = 263 probes
centroid_fallback = 3 probes
fallback_count = 3
```

This means almost all probes use the weighted region path, so improvements to
that path have broad impact.

## Accuracy Reference

Latest Octant accuracy after Sobol + `geom-median`:

```python
median_km = 361.6
mean_km = 457.8
p75_km = 684.7
p90_km = 955.9
```

These numbers are the baseline to compare against for future optimization work.

## Candidate Optimization Areas

1. **Geometric median**
   - Reduce sample count adaptively
   - Replace repeated full-sample solves with coarser-to-finer selection
   - Investigate whether Euclidean median on lat/lon can be approximated more cheaply

2. **Weighted feasible region**
   - Reduce grid resolution adaptively from region size / anchor count
   - Pre-prune obviously dominated landmarks
   - Vectorize or cache more of the annulus membership work

3. **Sampling**
   - Reuse candidate batches more efficiently
   - Improve acceptance rate for narrow feasible regions

## Code References

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
  - `compute_feasible_region_weighted()`
  - `sample_points_in_region()`
  - `geometric_median_approx()`
  - `estimate_location()`

- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)
  - `run_octant_cbg()`
  - `print_speed_benchmarks()`

## Acceptance Criteria For Future Work

Any optimization task should verify:

1. Octant tests still pass
2. `error_cdf_comparison.png` still includes Octant
3. `run_octant_cbg_sec` improves meaningfully relative to `121.892s`
4. Octant median / p90 error do not regress materially from the current baseline
