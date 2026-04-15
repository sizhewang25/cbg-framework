# Octant Geolocation - Current State Report

## Summary

Documented the current Octant geolocation implementation used by the shared
Octant module and the million-scale evaluator.

The current implementation:

- builds annular constraints from fitted Octant RTT models
- uses weighted region selection first
- falls back to unweighted region formation if needed
- selects a representative point by Monte Carlo geometric median
- falls back to a weighted landmark centroid when no feasible region exists

## Current Behavior

The main runtime entry points are:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)

The current evaluator settings are:

- target coverage: `0.80`
- no max-RTT filtering during constraint formation
- weighted-region threshold: `0.5`
- weighted grid resolution: `0.25` degrees
- Monte Carlo samples: `5000`

## Notes

- This task documents the implementation as it exists now, not the full
  original Octant paper.
- The main remaining paper gap is exact weighted sub-region computation and
  iterative target-area tightening.
