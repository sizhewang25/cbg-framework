# Octant Geolocation: Current Algorithm & Implementation

This task captures how the current Octant geolocation stage works in this repo.
It documents the implementation that turns fitted Octant RTT models into a
single location estimate for a target.

The geolocation stage is implemented in:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)

The current pipeline is:

1. Build annular constraints from landmark RTTs
2. Prefer a weighted feasible region
3. Fall back to an unweighted feasible region if needed
4. Select a point inside the region via Monte Carlo geometric median
5. Fall back to a weighted landmark centroid if no region survives

This is broadly aligned with the Octant paper, but there are a few practical
approximations in the current code:

- Weighted region formation is grid-based, not exact Bezier/sub-region boolean algebra
- The million-scale evaluator uses one shared `delta` across anchors
- The evaluator uses a fixed weighted-region threshold and grid resolution
- The current pipeline does not iteratively tighten `delta` until a target area is reached

See [algorithm.md](algorithm.md) for the full phase-by-phase description.
