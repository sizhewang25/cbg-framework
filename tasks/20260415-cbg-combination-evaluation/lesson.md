# CBG Combination Evaluation — Lessons

## 2026-04-15

### Framework Design
- **4 phases, not 3**: User corrected initial 3-phase design. Multilateration (region formation) and centroid selection (point estimate) are distinct phases with different implementations — must not be merged.
- **HF pattern works well**: Base class + registry decorator + per-variant file pattern scales cleanly. `from_config()` with string names makes pipeline composition ergonomic.

### Legacy Compatibility
- **`circle_preprocessing` is idempotent**: Called both in Phase 2 filter wrapper and internally by `circle_intersections()`. Safe because pre-computed `d`/`r` values pass through unchanged.
- **`check_circle_inclusion` removes the LARGER circle**: When `d_1 > (d + d_2)`, returns `(c_1, c_2)` = `(remove, keep)`. The larger containing circle is discarded, keeping the tighter bound. Non-obvious from variable names alone.
- **Matching back after `circle_preprocessing`**: Uses `(vp_lat, vp_lon, rtt_ms)` composite key to map filtered tuples back to `CircleConstraint` objects. Works because co-located VPs with identical RTT is astronomically unlikely.
