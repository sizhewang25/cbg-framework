# Octant RTT-Distance Model: Math & Implementation

## 1. Why RTT → Distance Is Non-Trivial

A naive geolocation formula converts RTT to distance as:

```
d = (RTT / 2) × c_internet
```

where `c_internet ≈ 2/3 × c` (speed of light in fiber, ~200,000 km/s). This gives the **theoretical minimum** RTT for a given distance — a lower bound.

In practice, RTT always exceeds this minimum due to:
- Routing detours (packets do not travel in straight lines)
- Queuing delays at intermediate routers
- Processing delays at endpoints
- Variation in ISP topology

So the actual relationship is a **scatter cloud**, not a line. Octant models this scatter explicitly.

---

## 2. The Convex Hull: Deriving Hard Bounds

### Setup

Given a landmark (anchor) `L` with known location, we collect calibration measurements from many other landmarks `P_i` with known positions. For each pair we observe:
- `d_i` = great-circle distance (km) between `L` and `P_i`
- `rtt_i` = measured RTT (ms) from `L` to `P_i`

Plot these as points `(rtt_i, d_i)` in 2D space.

### Convex Hull

The **convex hull** is the smallest convex polygon enclosing all calibration points. Its boundary consists of two chains:

```
Upper chain (R_L):  the maximum-distance boundary for each RTT value
Lower chain (r_L):  the minimum-distance boundary for each RTT value
```

These give **hard guarantees**: for any measured RTT `r` from landmark `L` to an unknown target `T`, the true distance `d(L, T)` must satisfy:

```
r_L(r)  ≤  d(L, T)  ≤  R_L(r)
```

This produces an **annular constraint** (a ring, not a circle) around `L`. Intersecting rings from multiple landmarks localizes `T` much more tightly than intersecting circles.

### Computing the Chains: Monotone Chain Algorithm

Given calibration points `{(rtt_i, d_i)}` sorted by RTT ascending:

**Lower hull** (minimum-distance boundary):
```
Initialize stack = []
For each point p (left to right):
    While |stack| ≥ 2 and cross(stack[-2], stack[-1], p) ≤ 0:
        pop stack
    push p
```

**Upper hull** (maximum-distance boundary):
```
Initialize stack = []
For each point p (right to left):
    While |stack| ≥ 2 and cross(stack[-2], stack[-1], p) ≤ 0:
        pop stack
    push p
Reverse result (so sorted by RTT ascending)
```

The cross product `cross(O, A, B) = (A.x - O.x)(B.y - O.y) - (A.y - O.y)(B.x - O.x)`:
- `> 0`: left turn (counterclockwise) — keep
- `≤ 0`: right turn or collinear — discard (not on convex hull)

The key insight: the lower hull always makes **left turns** going left-to-right (staying below the interior). The upper hull always makes **left turns** going right-to-left (staying above the interior).

Result: a piecewise linear upper boundary and a piecewise linear lower boundary, each defined by a small set of hull vertices.

### Evaluating the Hull: Piecewise Linear Interpolation

Given hull vertices `{(rtt_0, d_0), ..., (rtt_k, d_k)}` and a query RTT `r`:

```
Find i such that rtt_i ≤ r < rtt_{i+1}
t = (r - rtt_i) / (rtt_{i+1} - rtt_i)       # interpolation parameter
d = d_i + t × (d_{i+1} - d_i)               # linear interpolation
```

Beyond the last vertex (extrapolation), extend using the slope of the last segment.

---

## 3. Reliability Cutoff

At high RTT values, calibration data becomes **sparse** — few landmarks are far enough away to produce high RTT measurements. Hull vertices computed from sparse data are unreliable: a single outlier point can distort the boundary.

### Cutoff Detection

Scan RTT bins of width `bin_size_ms` from high to low. The **cutoff** `ρ` is the highest RTT bin containing at least `cutoff_min_points` calibration points.

```
For rtt from max down to min, step = bin_size_ms:
    count = |{i : rtt - bin_size ≤ rtt_i < rtt}|
    If count ≥ cutoff_min_points:
        ρ = rtt
        break
```

### Conservative Extension Beyond Cutoff

For `r ≥ ρ`, the hull is not trusted. Instead, extend from the hull value at `ρ` using the **theoretical speed-of-light slope**:

```
R_L(r) = R_L(ρ) + (r - ρ) / THEORETICAL_SLOPE    # upper: grows with speed of light
r_L(r) = r_L(ρ)                                   # lower: stays flat (conservative)
```

where `THEORETICAL_SLOPE = 2 / (c_internet)` ≈ 0.01 ms/km (the minimum possible RTT per km, for a perfectly straight route at 2/3c).

---

## 4. Piecewise Linear Spline: Iterative Refinement

The hull gives **hard outer bounds** but may be too conservative for practical use — the annular region can be very wide. The spline enables **tighter probabilistic bounds**.

### What the Spline Is

The spline is a piecewise linear function `f(rtt)` that minimizes the squared error to the calibration scatter:

```
min_f  Σ_i (d_i - f(rtt_i))²
```

subject to `f` being piecewise linear with `n_knots` breakpoints. This gives the **central tendency** of the RTT-to-distance mapping — the typical distance for a given RTT.

### Fitting with `make_lsq_spline`

`scipy.interpolate.make_lsq_spline(x, y, t, k=1)` solves this least-squares problem:
- `k=1`: degree-1 B-spline (piecewise linear)
- `t`: the full knot vector, with boundary multiplicity `k+1=2`:
  ```
  t_full = [rtt_min, rtt_min, t_1, t_2, ..., t_{n-1}, rtt_max, rtt_max]
  ```
  where `t_1, ..., t_{n-1}` are `n_knots` interior knots placed uniformly in `(rtt_min, rtt_max)`

### Restricting to the Reliable Region

The spline is only fit on data with `rtt ≤ ρ` (below the cutoff). The unreliable sparse region is excluded — the hull's conservative slope handles it.

### Choosing n_knots

The number of breakpoints is derived from the upper hull vertex count within the reliable region:

```
n_knots = max(3, count of hull_upper vertices with rtt ≤ ρ)
```

Rationale: the upper hull already identified where the RTT-distance relationship changes slope. The spline uses the same resolution, so it captures the same structural features with a least-squares fit rather than a hard envelope.

### Monotonicity Enforcement

Distance should be non-decreasing with RTT (higher RTT → farther away). The spline fit doesn't guarantee this. After fitting, enforce:

```
for i in 1..n:
    knot_dists[i] = max(knot_dists[i], knot_dists[i-1])
```

### Delta Bounds

Given the spline `f(rtt)`, the **iterative refinement bounds** for a multiplier `δ ≥ 1` are:

```
[f(rtt) / δ,  f(rtt) × δ]
```

This is a multiplicative band around the central estimate. `δ = 1` → zero width (just the spline). Larger `δ` → wider band covering more of the calibration scatter.

---

## 5. Delta Search: Achieving Target Coverage

Given a target coverage fraction `α` (e.g. 0.90), find the smallest `δ` such that at least fraction `α` of calibration points fall within `[f(rtt_i)/δ, f(rtt_i)×δ]`.

### Binary Search

```
Initialize delta_max by doubling from 1.0 until coverage(delta_max) ≥ α

Binary search in [1.0, delta_max]:
    delta_mid = (delta_low + delta_high) / 2
    coverage = fraction of points i where d_i ∈ [f(rtt_i)/delta_mid, f(rtt_i)*delta_mid]
    If |coverage - α| ≤ tolerance: return delta_mid
    If coverage < α: delta_low = delta_mid   (need wider band)
    Else:            delta_high = delta_mid  (can tighten)
```

### Speed-of-Internet Pre-Filter

Before any fitting, discard physically impossible measurements:

```
Remove point (rtt_i, d_i) if rtt_i < d_i × THEORETICAL_SLOPE
```

These points imply the signal traveled faster than 2/3 the speed of light — they are measurement artifacts (e.g. GPS error in the landmark's reported location, or clock issues).

---

## 6. Putting It Together: Prediction

Given a measured RTT `r` from a fitted anchor model:

**Hull mode** (hard bounds, no calibration needed):
```
min_dist = r_L(r)     # lower hull at r
max_dist = R_L(r)     # upper hull at r
```

**Spline mode** (probabilistic band, requires delta):
```
center   = f(r)              # spline at r
min_dist = center / δ
max_dist = center × δ
```

where `δ` is found via delta search on the calibration data with target coverage `α`.

---

## 7. Summary of Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| `c` | 300,000 km/s | Speed of light in vacuum |
| `c_internet` | 200,000 km/s | Speed in fiber (2/3 c) |
| `THEORETICAL_SLOPE` | ~0.01 ms/km | Minimum RTT per km (2 × 1/c_internet) |
| `ρ` | data-driven | Cutoff RTT above which data is sparse |
| `δ` | data-driven | Spline multiplicative band (≥ 1.0) |

---

## Related Files

- [octant_model.py](../../scripts/analysis/octant/octant_model.py) — implementation
- [test_octant_model.py](../../scripts/analysis/octant/test_octant_model.py) — unit tests
- [octant_spline_visualization.py](../../scripts/analysis/octant/octant_spline_visualization.py) — visualization script
- [Octant Paper](../../scripts/analysis/cbg_feasibility/references/Wong%20et%20al.%20-%20Octant%20A%20Comprehensive%20Framework%20for%20the%20Geolocalization%20of%20Internet%20Hosts.pdf)
