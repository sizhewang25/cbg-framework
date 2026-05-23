# octant — spline fitting deep dive

The "best-guess curve" inside the hull band is a piecewise-linear LSQ spline. Three layers hand off to each other: `fit_rtt_distance_spline` does the math, `OctantRTTModel.fit` enforces the data-shape guards, and `predict_distance` evaluates the result at query time.

This document zooms into the spline subsystem of [octant_model.py](octant_model.py). For the broader module purpose (annular RTT-distance constraints), see the file's top docstring.

---

## 1. What kind of spline this is

`scipy.interpolate.make_lsq_spline(rtts, distances, t=t_full, k=1)` — order **k=1**, which means **piecewise linear**. A k=1 B-spline is just a polyline whose joints sit at the knots. The "spline" word is doing real work here only in the sense that scipy gives us LSQ machinery and B-spline basis functions; the output curve is the same shape you'd get from connecting dots with straight lines.

The LSQ problem being solved:

    min_c  Σ_i (y_i − Σ_j c_j · B_j(x_i))^2

- `x_i`, `y_i` are the training `(rtts, distances)` pairs.
- `B_j` are the k=1 B-spline basis functions — "tent" functions that peak at their center knot and decay linearly to zero at the adjacent knots on each side.
- `c_j` are the unknown coefficients scipy solves for.

For k=1, each `B_j` is non-zero over exactly two adjacent inter-knot intervals. That's the entire structure of the basis — no smoothness constraints beyond continuity at the knots, no derivatives matched, nothing exotic.

---

## 2. The knot vector

```python
interior = np.linspace(rtts[0], rtts[-1], n_knots + 2)[1:-1]
t_full = np.r_[(rtts[0],) * 2, interior, (rtts[-1],) * 2]
```

scipy needs the knot vector in **canonical B-spline form**: boundary knots repeated `k+1 = 2` times, interior knots strictly between them.

```
t_full = [rtts[0], rtts[0], t_1, t_2, ..., t_{n_knots}, rtts[-1], rtts[-1]]
          ──── repeat ────                                ──── repeat ────
```

Total length: `n_knots + 4`. Number of basis functions: `len(t) − k − 1 = n_knots + 2`.

The doubled boundary knots aren't a numerical hack — they're the B-spline convention for "the spline should hit the boundary value, not approach zero there." Without doubling, the leftmost/rightmost basis functions would taper to zero at the edges and the spline couldn't fit data at the extremes.

The `np.linspace(...)[1:-1]` gives **uniformly-spaced** interior knots. Alternatives:
- Quantile-based knots (each inter-knot interval has roughly the same number of data points)
- Adaptive placement based on residuals

Uniform is the simplest and works well when RTT data is roughly evenly distributed across the range. For clustered RTT data (the integration-test scenario with 5 distinct RTTs), uniform knot placement can land between data clusters and starve some basis functions of support — which is why we need the n_knots clamp upstream.

---

## 3. `fit_rtt_distance_spline` line by line

**Input filter.** Drop non-finite and non-positive rows.

**Minimum-points guard.** Needs `n_knots + 3` valid points for the LSQ system to be non-underdetermined. With `n_knots + 2` basis functions and fewer points, the design matrix has more columns than rows and the solution is non-unique. Raises `SplineFitError` — caught by `OctantRTTModel.fit` as a soft fail.

**Sort by RTT.** `make_lsq_spline` requires sorted x-values. LSQ is order-invariant in principle, but scipy enforces this as an API contract.

**The fit itself.** Wrapped in `try/except Exception` because scipy raises a variety of error types for malformed input (duplicate knots, knot order violations, Schoenberg-Whitney failures). All of them get rewrapped as `SplineFitError` so callers only need to catch one class.

**Evaluate at the uniform grid:**

```python
knot_rtts = np.linspace(rtts[0], rtts[-1], n_knots + 2)
knot_dists = spline(knot_rtts)
```

This is **not** the same as `t_full`. `t_full` is scipy's internal knot vector (with doubled boundaries) used for basis function definition. `knot_rtts` is the **output grid** we evaluate the fitted spline on — `n_knots + 2` uniform points from `rtts[0]` to `rtts[-1]`, inclusive. The returned `(knot_rtts, knot_dists)` pair is what gets stored on the model and what `np.interp` consumes at predict time.

This is a deliberate trick: at fit time we use scipy's LSQ + B-spline machinery, but at predict time the result is just a polyline, evaluable by `np.interp` (linear interpolation between adjacent knots). That makes the stored model JSON-friendly (no scipy spline object to serialize) and predict-time fast (np.interp is C, no Python overhead).

**Non-finite guard:**

```python
if not np.all(np.isfinite(knot_dists)):
    raise SplineFitError(...)
```

This is the cleanup pass after the n_knots clamp wasn't enough. Even with the upstream clamp, certain pathological data distributions can drive the LSQ system into ill-conditioning that produces NaN or Inf coefficients silently. Without this guard, NaN propagates into `np.interp` at predict time, into the δ-search coverage probe, and the symptom is "no δ ever achieves target coverage." Costly to debug — much cheaper to raise here.

**Monotonicity enforcement:**

```python
for i in range(1, len(knot_dists)):
    if knot_dists[i] < knot_dists[i - 1]:
        knot_dists[i] = knot_dists[i - 1]
```

Physical constraint: distance shouldn't decrease as RTT increases. The LSQ fit doesn't enforce this — it just minimizes squared residuals. For typical "noisy but trending up" data, the fit is already monotonic. But for sparse data or noisy regions, the spline can dip locally. This simple pass levels out dips by holding the previous value.

This is **not** isotonic regression (which would minimize squared error subject to a monotonicity constraint). It's a one-pass post-hoc fix that doesn't preserve the LSQ objective. The trade-off: cheap and predictable, but the post-fix curve is no longer the LSQ-optimal monotonic fit. For Octant's use case — where the spline is the "best guess" inside a hull band that will clip wild values anyway — this approximation is fine.

**R² computation:**

```python
predicted = spline(rtts)   # raw spline, pre-monotonicity
ss_res = np.sum((distances - predicted) ** 2)
ss_tot = np.sum((distances - np.mean(distances)) ** 2)
r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
```

Standard coefficient of determination, evaluated against the **raw** spline (the `spline` callable, not the monotonized `knot_dists`). This is honest — it reports the LSQ fit quality, not what was stored after monotonicity clipping. Reported only in `fit_message` for human inspection; the predict path doesn't use it.

---

## 4. The `n_knots` clamp

```python
n_distinct = int(np.unique(spline_rtts).size)
n_knots_used = max(3, min(spline_n_knots, max(3, n_distinct - 2)))
```

This is the load-bearing data-shape guard. Why `n_distinct − 2`?

Each k=1 basis function `B_j` has support over **two adjacent inter-knot intervals**. For the LSQ system to be non-singular, each `B_j` needs at least one data point in its support. With `n_distinct` distinct RTT values, the data has at most `n_distinct − 1` "natural breakpoints" where a piecewise-linear fit can change slope. If you ask for more knots than the data naturally supports, some basis functions get zero or one data point in their support → rank-deficient design matrix → NaN coefficients.

Empirically, `n_distinct − 2` is the safe cap. The `max(3, ...)` keeps it from going below 3 even on tiny data (the spline needs at least three knots to be non-trivial — two endpoints plus at least one interior).

The triple-nested `max(3, min(spline_n_knots, max(3, n_distinct − 2)))` reads as:

1. Inner `max(3, n_distinct − 2)`: the stability ceiling, never below 3.
2. `min(spline_n_knots, ceiling)`: respect the user's request, but cap at stability.
3. Outer `max(3, ...)`: floor at 3 regardless.

For the integration test data (5 distinct RTTs, user asks `spline_n_knots=4`), this resolves to `max(3, min(4, max(3, 3))) = max(3, min(4, 3)) = max(3, 3) = 3`. The fit succeeds with 3 interior knots — which is what the legacy module accidentally computed via a different heuristic (hull-vertex count). The simplification makes that workaround explicit.

---

## 5. Spline-fit decision tree in `OctantRTTModel.fit`

```python
if fit_spline:
    if self.cutoff_rtt > 0:
        mask = valid_rtts <= self.cutoff_rtt
    else:
        mask = np.ones_like(valid_rtts, dtype=bool)
    spline_rtts = valid_rtts[mask]
    spline_dists = valid_dists[mask]
    ...
    try:
        knot_rtts, knot_dists, spline_meta = fit_rtt_distance_spline(...)
        ...
    except SplineFitError as e:
        self.fit_message = f"hull OK, spline failed: {e}"
```

Four notable choices:

**Sub-baseline rows filtered before anything else.** The validity mask at the top of `fit` drops rows where `rtt < baseline_slope · distance` — physically impossible at 2/3·c, almost certainly a mislabeled coordinate or a measurement artifact. Both the hull and the spline then see the same baseline-clean data. This filter is unconditional; there's no flag to disable it.

**Data masked to `≤ cutoff_rtt`.** The spline only fits the "trusted region" — points above `cutoff_rtt` are sparse-data outliers that would distort the spline. Above the cutoff, prediction switches to baseline-slope extension.

**Hull is required, spline is optional.** If the spline fit fails (insufficient data, ill-conditioned, etc.), `fitted = True` still — but `spline_rtt_knots = None`. The model is "fitted" in the sense that hull-only predictions work; it's just degraded. Callers (like `BoundedSplineLTD`) handle this by skipping δ search for that VP and letting predict fall back to hull bounds.

**The exception is caught, not propagated.** `SplineFitError` from inside `fit_rtt_distance_spline` is caught locally and stored in `fit_message`. Caller code never sees it. This is the "best-effort fit" contract.

---

## 6. How the spline is consumed at predict time

### `predict_distance(rtt)` — the "best-guess" point estimate

```python
knot_rtts = np.array(self.spline_rtt_knots)
knot_dists = np.array(self.spline_dist_knots)
if rtt > self.cutoff_rtt and self.cutoff_rtt > 0:
    cutoff_val = float(np.interp(self.cutoff_rtt, knot_rtts, knot_dists))
    predicted = cutoff_val + (rtt - self.cutoff_rtt) / self.baseline_slope
else:
    predicted = float(np.interp(rtt, knot_rtts, knot_dists))

if self.hull_upper_rtts and self.hull_lower_rtts:
    predicted = max(self._inner(rtt), min(predicted, self._outer(rtt)))

return max(predicted, 0.0)
```

Two regimes:

- **Above cutoff:** the spline can't be trusted there (no data supports it). Pin to `spline(cutoff_rtt)` and extend with baseline slope `1/baseline_slope ≈ 100 km/ms`. This is conservative — we don't know what the data would say, so we use the speed-of-light bound.
- **Inside the trusted region:** plain `np.interp` between knots. The piecewise-linear evaluation is exact (mathematically equivalent to evaluating the scipy spline object, but cheaper and JSON-portable).

After the spline lookup, the result is **clipped to the hull band** via `max(inner, min(predicted, outer))`. The spline can wander outside the hull at certain RTTs (the LSQ fit doesn't know about the hull); clipping enforces that the "best guess" is at least geometrically possible.

### `predict_distance_bounds(rtt, delta)` — the annulus output

```python
if delta is not None:
    predicted = self.predict_distance(rtt)
    inner = max(predicted / delta, self._inner(rtt))
    outer = min(predicted * delta, self._outer(rtt))
    return (max(0.0, inner), outer)
```

The δ widening forms a **multiplicative band around the spline**: `(spline/δ, spline·δ)`. `predicted` has already been hull-clipped, so the band starts from a value inside the hull. Then `max` and `min` re-clip the widened endpoints against the hull — important when δ is large and `spline·δ` exceeds the hull upper.

Subtleties:

- δ ≥ 1 always (by the search constraint), so `predicted/δ ≤ predicted ≤ predicted·δ`. The band is always at least as wide as the point estimate.
- δ = 1 means the "band" is degenerate (just the spline value). The wrapper handles this by leaving `_deltas` empty when δ search fails, which makes `delta=None` at predict time and routes to hull-only bounds.
- The hull clip on both sides means a large δ doesn't blow up predictions — at most you get hull-only bounds.

---

## 7. What's deliberately not here

- **Higher-order splines (k≥2).** Octant's paper uses piecewise-linear; cubic splines would need more knots, careful boundary handling for monotonicity, and don't add much given the noise floor in real RTT measurements.
- **Adaptive / quantile-based knot placement.** Uniform is robust to most data shapes (with the n_knots clamp as a safety net). Quantile-based placement is one avenue if you wanted to handle very non-uniform RTT distributions, but it adds complexity without a clear win for this dataset.
- **Cross-validation for n_knots.** The user picks `spline_n_knots`; the data-distinct clamp caps it. No CV loop. For 4–20 knots over a few hundred RTTs, the optimum is broad and hard to over- or under-fit.
- **Regularization on coefficients.** With few knots, the LSQ system is small enough that L2 (ridge) regularization would mostly slow things down without changing answers. The monotonicity post-fix handles the one pathological case (LSQ dipping below physical reality) cheaply.
- **Confidence intervals around the spline.** δ-search is doing the analog: instead of analytic CIs, find the multiplicative widening factor that covers a target fraction of training data. It's coarser but distribution-free and matches what Octant publishes.

---

## Summary

The whole spline subsystem is ~90 lines, fits in your head, and the load-bearing decisions (k=1, uniform knots, `n_distinct − 2` clamp, non-finite guard, post-fit monotonicity) each have a one-line justification you can defend. The only piece that took on-the-ground debugging to find was the `n_distinct − 2` clamp — the legacy module hit the same NaN-spline brittleness and worked around it accidentally via hull-vertex counting; the simplification makes that workaround explicit.
