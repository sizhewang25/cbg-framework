# Octant Spline Fitting Algorithm

End-to-end description of `OctantRTTModel.fit()` as implemented in
[octant_model.py](../../scripts/analysis/octant/octant_model.py).

---

## Phase 0 — Pre-filter (Speed-of-Internet)

Before any fitting, discard physically impossible measurements:

```
Remove (rtt_i, d_i)  if  rtt_i < d_i × THEORETICAL_SLOPE
```

`THEORETICAL_SLOPE = 2 / (200,000 km/s) ≈ 0.01 ms/km` is the minimum possible RTT per km
for a signal traveling at 2/3 the speed of light in a straight line. Points below this threshold
imply faster-than-physical propagation — artifacts of GPS error or clock drift in landmark
reported coordinates.

---

## Phase 1 — Convex Hull (Hard Bounds)

### Monotone Chain Algorithm

Given calibration points `{(rtt_i, d_i)}` sorted by RTT ascending:

**Lower hull** (r_L — minimum-distance boundary):
```
stack = []
for p in points left→right:
    while |stack| ≥ 2 and cross(stack[-2], stack[-1], p) ≤ 0: pop
    push p
```

**Upper hull** (R_L — maximum-distance boundary):
```
stack = []
for p in points right→left:
    while |stack| ≥ 2 and cross(stack[-2], stack[-1], p) ≤ 0: pop
    push p
reverse result  →  sorted by RTT ascending
```

`cross(O, A, B) = (A.x−O.x)(B.y−O.y) − (A.y−O.y)(B.x−O.x)` — positive = left turn (kept), ≤ 0 = right turn or collinear (discarded).

Each chain is a piecewise linear boundary. For any measured RTT `r`:

```
r_L(r)  ≤  true distance  ≤  R_L(r)
```

### Bilateral Density Cutoff (bin_size = 5 ms, min_points = 5)

Scan RTT bins upward from `min_rtt` to `max_rtt`:

- `low_cutoff_rtt` = left edge of the **first** bin with ≥ `min_points` points
- `cutoff_rtt` = right edge of the **last** bin with ≥ `min_points` points

Upward scanning ensures isolated high-RTT clusters do not inflate the cutoff.

### Hull Evaluation Outside the Reliable Region

| Region | Upper hull R_L | Lower hull r_L |
|---|---|---|
| `rtt < low_cutoff_rtt` | `rtt / THEORETICAL_SLOPE` (2/3c line) | `0` (no lower constraint) |
| `low_cutoff_rtt ≤ rtt ≤ cutoff_rtt` | Piecewise linear interpolation between hull vertices | Same |
| `rtt > cutoff_rtt` | `R_L(cutoff_rtt) + (rtt − cutoff_rtt) / THEORETICAL_SLOPE` | `r_L(cutoff_rtt)` (flat) |

---

## Phase 2 — Initial Spline Fit

Restrict data to the candidate reliable region:

```
spline data = {(rtt_i, d_i) : low_cutoff_rtt ≤ rtt_i ≤ cutoff_rtt}
```

**n_knots selection:**

```
upper_count = hull upper vertices with RTT in [low_cutoff_rtt, cutoff_rtt]
lower_count = hull lower vertices with RTT in [low_cutoff_rtt, cutoff_rtt]
n_knots = max(3, max(upper_count, lower_count))
```

Rationale: the hull already identified where the RTT–distance relationship changes slope. The spline uses the same resolution — matching the structural complexity of the hull.

**Fitting (`scipy.interpolate.make_lsq_spline`, k=1):**

Minimizes `Σ (d_i − f(rtt_i))²` subject to `f` being piecewise linear with `n_knots` breakpoints. The full knot vector with boundary multiplicity 2:

```
t_full = [rtt_min, rtt_min, t_1, ..., t_{n-1}, rtt_max, rtt_max]
```

where `t_1, ..., t_{n-1}` are interior knots placed uniformly in `(rtt_min, rtt_max)`.

**Monotonicity enforcement (post-fit):**

```
for i in 1..n:
    knot_dists[i] = max(knot_dists[i], knot_dists[i-1])
```

Ensures distance is non-decreasing with RTT.

---

## Phase 3 — Cutoff Calibration via Spline-Hull Intersections

After the initial spline fit, density-based cutoffs are refined by checking where the spline
first exits the hull bounds. This makes the cutoffs geometrically grounded rather than purely
density-driven.

Evaluate spline and both hulls on a 500-point grid within `[low_cutoff_rtt, cutoff_rtt]`:

```
grid = linspace(low_cutoff_rtt, cutoff_rtt, 500)
spline_vals = interp(grid, knot_rtts, knot_dists)
upper_vals  = R_L(grid)
lower_vals  = r_L(grid)
```

**High cutoff refinement** (spline vs upper hull):
- Find the **last** grid index where `spline_vals[i] ≤ upper_vals[i]`
- If earlier than the density cutoff → replace `cutoff_rtt` with this intersection RTT

**Low cutoff refinement** (spline vs lower hull):
- Find the **first** grid index where `spline_vals[i] ≥ lower_vals[i]`
- If later than the density low_cutoff → replace `low_cutoff_rtt` with this intersection RTT

---

## Phase 4 — Spline Refit on Calibrated Region

With the refined `[low_cutoff_rtt, cutoff_rtt]` from Phase 3, repeat Phase 2:
- Recompute `reliable_mask` and `n_knots`
- Refit the spline via `make_lsq_spline`
- Re-enforce monotonicity

The final `spline_rtt_knots` / `spline_dist_knots` always correspond to the final calibrated cutoffs.

---

## Spline Evaluation (Prediction Time)

Given a query RTT `r`:

| Region | Spline output |
|---|---|
| `r < low_cutoff_rtt` | `r / THEORETICAL_SLOPE`  (2/3c line — no reliable data below) |
| `low_cutoff_rtt ≤ r ≤ cutoff_rtt` | `interp(r, knot_rtts, knot_dists)` |
| `r > cutoff_rtt` | `spline(cutoff_rtt) + (r − cutoff_rtt) / THEORETICAL_SLOPE`  (2/3c extension) |

For **delta-band prediction** (`use_polynomial=True`):

```
(predicted / δ,  predicted × δ)
```

where δ is found via binary search to achieve a target coverage fraction α (e.g. 90%) over
the calibration scatter: the smallest δ such that at least fraction α of points `d_i` satisfy
`d_i ∈ [f(rtt_i)/δ, f(rtt_i)×δ]`.

---

## Constants

| Symbol | Value | Meaning |
|---|---|---|
| `THEORETICAL_SLOPE` | ~0.01 ms/km | Minimum RTT per km at 2/3c |
| `bin_size_ms` | 5 ms | RTT bin width for cutoff detection |
| `cutoff_min_points` | 5 | Minimum points per bin to count as dense |
| `grid_size` | 500 | Points for spline-hull intersection search |

---

## Related Files

- [octant_model.py](../../scripts/analysis/octant/octant_model.py) — implementation
- [test_octant_model.py](../../scripts/analysis/octant/test_octant_model.py) — unit tests (15 tests)
- [octant_spline_visualization.py](../../scripts/analysis/octant/octant_spline_visualization.py) — visualization
- [README.md](README.md) — mathematical background
