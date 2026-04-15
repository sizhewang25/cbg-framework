# CBG Combination Evaluation — Lessons

## 2026-04-15

### Framework Design
- **4 phases, not 3**: User corrected initial 3-phase design. Multilateration (region formation) and centroid selection (point estimate) are distinct phases with different implementations — must not be merged.
- **HF pattern works well**: Base class + registry decorator + per-variant file pattern scales cleanly. `from_config()` with string names makes pipeline composition ergonomic.

### Legacy Compatibility
- **`circle_preprocessing` is idempotent**: Called both in Phase 2 filter wrapper and internally by `circle_intersections()`. Safe because pre-computed `d`/`r` values pass through unchanged.
- **`check_circle_inclusion` removes the LARGER circle**: When `d_1 > (d + d_2)`, returns `(c_1, c_2)` = `(remove, keep)`. The larger containing circle is discarded, keeping the tighter bound. Non-obvious from variable names alone.
- **Matching back after `circle_preprocessing`**: Uses `(vp_lat, vp_lon, rtt_ms)` composite key to map filtered tuples back to `CircleConstraint` objects. Works because co-located VPs with identical RTT is astronomically unlikely.

### Evaluation Results
- **Shapely multilateration outperforms spherical**: For the same distance model, Shapely polygon intersection (Paths B, C) consistently beats spherical circle intersection (Path A). The degree-space polygon approximation captures the feasible region better than the vertex-only approach.
- **Arithmetic centroid on Shapely > geometric centroid**: C1 (shapely+arith, 333 km) slightly beats B1 (shapely+geom, 395 km). The area-weighted Shapely centroid doesn't always help — extracting boundary vertices and averaging can be more robust for skewed intersection polygons.
- **LP lower envelope underperforms**: Despite per-anchor calibration, LP bestline (A2, B2, C2) consistently ranks below both the theoretical 2/3c model and the Octant spline. The LP fit may be too conservative (tight radius = frequent empty intersections → fallback to closest VP).

### Scope Decisions
- **Removed weighted_grid (C1) from evaluation**: User requested removal — Octant's fused Phase 2+3 needs separate validation before including in systematic comparison. Renamed D-path to C-path.
- **bounded_spline + redundant_circle uses outer radius only**: When annular constraints pass through legacy `circle_preprocessing()`, only `radius_km` (outer) is used. This is correct for the filtering step — the inner radius is only meaningful for weighted grid multilateration.
