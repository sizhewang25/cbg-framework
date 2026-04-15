# CBG Combination Evaluation — Report

**Status**: In Progress
**Created**: 2026-04-15
**Last Updated**: 2026-04-15

## Summary

Systematic evaluation of CBG geolocation pipeline combinations across four phases:
- 3 distance models × 2 filters × 3 multilateration methods × 2 centroid methods = 10 valid combinations (constrained by compatibility)

## Findings

### 2026-04-15 — Framework Implementation Complete

Implemented HF-style modular framework under `scripts/framework/` (commit `faedb9f`).

**Architecture**: Base class + registry + per-variant files + `CBGPipeline` with `from_config()` factory.

**Components registered** (verified via import):
- Distance: `speed_of_internet`, `low_envelope`, `bounded_spline`
- Filtering: `redundant_circle`, `none`
- Multilateration: `spherical`, `shapely`, `weighted_grid`
- Centroid: `arithmetic_mean`, `geometric_centroid`

**Key design decisions**:
- `CircleConstraint` dataclass with `to_legacy_tuple()` bridges new types to `helpers.py` functions
- `MultilatResult` carries either `vertices` (spherical) or `region` (Shapely geometry)
- Both centroid methods handle both input types for maximum composability
- `weighted_grid` requires `bounded_spline` — validated in `from_config()`
- Deferred imports (try/except) for optional heavy dependencies (Octant, LP models)

**Files**: 18 source files + 4 task files, 1,362 insertions total.

## Conclusions

*Final assessment when task completes.*
