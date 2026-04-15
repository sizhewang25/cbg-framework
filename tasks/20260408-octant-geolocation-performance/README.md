# Octant Geolocation Performance

This task tracks optimization work for the current Octant geolocation pipeline.

The main goal is to reduce the runtime of the Octant path in million-scale
evaluation while preserving geolocation quality and the current output artifacts
(CDFs, maps, JSON stats).

Current implementation references:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)
- [algorithm.md](../octant-geolocation/algorithm.md)

The current Octant flow is:

1. Form annular constraints from RTTs
2. Build a weighted feasible region on a grid
3. Sample points from the region
4. Compute a geometric median of sampled points
5. Fall back to centroid logic when the region is empty or tiny

The latest benchmark shows the main bottleneck is still the point-selection
stage, especially geometric median computation.
