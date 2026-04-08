# Technical Context

## Problem Definition

Recent Octant geolocation changes appear to have worsened final accuracy.

We need to identify the impact of each change rather than treating the current
implementation as one opaque bundle.

The right next step is an attribution study:

1. Start from the last-week baseline commit
2. Reintroduce changes one by one
3. Re-run evaluation after each step
4. Record the effect on accuracy and runtime

## Baseline Commit

Use this commit as the reference point:

```text
a9bb3d6  2026-04-01  feat: add visualization for monte carlo
```

This is the latest relevant Octant geolocation commit from last week before the
April 8 implementation changes.

## Changes Since The Baseline

These are the committed changes that matter for Octant geolocation behavior.

### 1. Zero-radius robustness in Shapely

Commit:

```text
cf94226  Keep Octant distance model honest at zero
```

Behavior change:

- `_circle_to_shapely()` now clamps circle radii to a tiny epsilon before
  polygonization
- Intended to avoid degenerate zero-radius Shapely geometries during region
  operations

Relevant code:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)

### 2. Keep high-RTT constraints instead of filtering them out

Commit:

```text
0944c7f  feat: remove max_rtt filter for form_constraints and align test coverage with the visualization
```

Behavior changes:

- `form_constraints()` no longer drops RTT measurements above `200 ms`
- `fit_octant_models(..., target_coverage=0.80)` is now aligned with the
  visualization path instead of the old `0.90` target

Potential effect:

- More weak or noisy landmarks now participate in geolocation
- The shared `delta` band may be narrower than before

Relevant code:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)

### 3. Switch evaluator from unweighted region shortcut to weighted-region flow

Commit:

```text
ff80232  Align Octant geolocation with weighted region flow
```

Behavior changes:

- The million-scale evaluator stopped using a local hard unweighted region path
- It now calls `estimate_location(method='weighted', ...)`
- Weighted region uses thresholded grid accumulation before falling back to
  unweighted logic

Potential effect:

- This is the biggest algorithmic change in region formation
- The result may differ even when the same annular constraints are used

Relevant code:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)

### 4. Replace IID random sampling with Sobol QMC sampling

Commit:

```text
0c0e532  Use Sobol sampling and geom-median for Octant
```

Behavior change:

- `sample_points_in_region()` now uses scrambled Sobol low-discrepancy sampling
  with rejection inside the feasible region

Potential effect:

- Sampling coverage is more even
- But the accepted sample distribution may differ from the old Monte Carlo path
  enough to move the final estimate

Relevant code:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)

### 5. Replace sampled-point argmin with `geom-median`

Commit:

```text
0c0e532  Use Sobol sampling and geom-median for Octant
```

Behavior change:

- `geometric_median_approx()` no longer returns one of the sampled points with
  minimum total pairwise haversine distance
- It now uses `geom_median.numpy.compute_geometric_median(...)`

Potential effect:

- This changes the center-selection rule directly
- It can improve smoothness and speed, but may also move estimates in a way that
  hurts the final error distribution

Relevant code:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [pyproject.toml](../../pyproject.toml)
- [poetry.lock](../../poetry.lock)

## Current Observation

The latest rerun suggests that the newer implementation is worse at the end of
the pipeline than the earlier post-change baseline we observed.

Recent observed Octant median errors:

```python
previous_post_change_median_km = 361.6
latest_rerun_median_km = 397.7
```

The latest rerun also reported:

```python
mean_km = 466.7
p75_km = 659.9
p90_km = 944.2
p95_km = 1059.2
```

So the working conclusion is:

- newer Octant logic is not obviously a strict improvement
- accuracy needs to be decomposed by change
- speedups are not sufficient justification if the CDF regresses

## Important Debugging Principle

Do not compare only:

- last week vs now

Also compare:

- baseline + one change
- baseline + two changes
- weighted-region change alone vs sampling/centroid change alone

Otherwise we will not know whether the regression comes from:

- region formation
- delta calibration target
- inclusion of high-RTT constraints
- Sobol sampling
- `geom-median`
- interaction between those pieces

## Suggested Experiment Matrix

Keep the dataset fixed and vary only one dimension at a time.

### Region Path

1. Baseline unweighted region path
2. Weighted region path with all else held constant

### Constraint Inputs

1. High-RTT filter enabled
2. High-RTT filter disabled

### Delta Coverage Target

1. `target_coverage=0.90`
2. `target_coverage=0.80`

### Point Sampling

1. IID uniform rejection sampling
2. Sobol rejection sampling

### Point Selection

1. Old sampled-point sum-of-distances argmin
2. `geom-median`

## Metrics To Record

For each experiment, record at least:

```python
median_km
mean_km
p75_km
p90_km
p95_km
within_50_km
within_100_km
within_250_km
within_500_km
within_1000_km
run_octant_cbg_sec
geometric_median_sec
weighted_region_sec
sample_points_sec
```

Also record:

- region-selection counts by method
- number of centroid fallbacks
- shared `delta`
- cutoff variant used

## Current Scope Boundary

This task is based on committed changes relative to the April 1 baseline.

There are also additional Octant-related edits currently present in the
worktree. Those should be treated carefully and only folded into this study if
they are intentionally part of the attribution plan.
