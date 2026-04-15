# RTT Filtering Research Findings

## Problem Statement

We need to filter RTT data to capture the **lower bound envelope** of RTT measurements at different distances for CBG (Constraint-Based Geolocation). The lower bound represents the minimum achievable RTT at each distance, which is used to convert observed RTT to maximum possible distance.

### Key Assumptions Evaluated

1. **"VP to probes of an ASN have similar network routes"**
   - **Verdict**: Partially valid with caveats
   - Works well for per-anchor calibration
   - Large ASNs (e.g., Comcast spanning entire US) may need regional grouping
   - Assumption is reasonable for the current per-anchor model approach

2. **"Linear fit is appropriate for RTT-distance"**
   - **Verdict**: Yes, linear is correct
   - RTT-distance relationship is fundamentally linear (physics: d = c × t)
   - Octant paper's Bezier curves are for geographic region representation, NOT RTT-distance modeling
   - Non-linear only helps for piecewise models (different slopes at different distance ranges)

---

## Methods Compared

### 1. Current Approach: LP + 5-Stage Filter

**Implementation**: `rtt_model.py` → `fit_bestline_lp()` + `filter_rtt_data()`

**5 Stages**:
1. Remove invalid values (zero, negative, inf)
2. Per-bin mean±σ filtering (removes both low and high outliers)
3. Per-bin 5th percentile (keeps only lowest RTTs per bin)
4. Global bin-min filter (removes entire anomalous bins)
5. Speed-of-light baseline filter (physical sanity check)

**Pros**:
- Explicit handling of known data quality issues
- Strict lower bound ("below ALL points" constraint)
- Matches original CBG paper methodology

**Cons**:
- 6+ tuning parameters
- Requires binning (introduces artifacts)
- May over-filter, losing valid low RTTs

### 2. Quantile Regression

**Implementation**: `statsmodels.regression.quantile_regression.QuantReg`

**Theory**: Instead of minimizing squared errors (OLS), minimize asymmetrically weighted absolute errors:

```
Loss for τ = 0.05:
  - Under-prediction (y > ŷ): penalty = 0.05 × |error|
  - Over-prediction (y < ŷ):  penalty = 0.95 × |error|
```

This asymmetric penalty naturally pushes the fit toward the 5th percentile.

**Pros**:
- No binning required
- Only 1 parameter (τ = quantile level)
- Statistically rigorous with confidence intervals
- Inherently robust to outliers
- Directly estimates conditional quantile

**Cons**:
- "Soft" lower bound (5% of points below by design)
- Requires statsmodels dependency
- Less explicit control over outlier handling

---

## Empirical Results

### Synthetic Data (500 points, true slope=0.012, intercept=8.0)

| Method | Slope | Intercept | % Below | Slope Error |
|--------|-------|-----------|---------|-------------|
| True lower bound | 0.0120 | 8.0 | 5.8% | - |
| **Quantile τ=0.05** | 0.0126 | 7.2 | **5.2%** | **5.3%** |
| OLS (mean) | 0.0169 | 12.8 | 67.6% | 41.2% |

### Real Data (Anchor 66.42.119.57, AS7922, n=266)

| Method | Slope (ms/km) | Intercept (ms) | % Below Line |
|--------|---------------|----------------|--------------|
| **Quantile τ=0.05** | 0.0162 | 8.8 | **5.3%** |
| Quantile τ=0.10 | 0.0160 | 10.0 | 10.2% |
| LP + 5-stage filter | 0.0131 | 12.3 | 8.3% |
| OLS (mean) | 0.0162 | 15.6 | 62.8% |

### Key Observations

1. **Quantile regression τ=0.05 achieves exactly ~5% of points below** — the target
2. **LP + filter has 8.3% below** — filtering removes some valid low RTTs
3. **QR slope (0.0162) vs LP slope (0.0131)** — QR better matches visual lower envelope
4. **QR requires no parameter tuning** beyond choosing τ

---

## Quantile Regression Theory

### The Check Function (Pinball Loss)

For quantile τ ∈ (0, 1):

```
ρ_τ(u) = u × (τ - I(u < 0))

       = { τ × u        if u ≥ 0  (under-prediction)
         { (τ - 1) × u  if u < 0  (over-prediction)
```

For τ = 0.05: over-prediction is penalized **19× more** than under-prediction, pushing the line down to the 5th percentile.

### Mathematical Formulation

```
minimize Σ ρ_τ(y_i - (β₀ + β₁x_i))

= minimize Σ [ τ × max(0, y_i - ŷ_i) + (1-τ) × max(0, ŷ_i - y_i) ]
```

### Implementation

```python
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# Fit 5th percentile directly
X = sm.add_constant(distances)
result = QuantReg(rtts, X).fit(q=0.05)

slope = result.params[1]
intercept = result.params[0]
conf_int = result.conf_int()  # Built-in confidence intervals!
```

### Key References

- **Koenker & Bassett (1978)**: "Regression Quantiles" - Econometrica
  https://people.eecs.berkeley.edu/~jordan/sail/readings/koenker-bassett.pdf

- **Koenker & Hallock (2001)**: "Quantile Regression" - Journal of Economic Perspectives
  https://www.aeaweb.org/articles?id=10.1257/jep.15.4.143

- **scikit-learn documentation**: Quantile Regression example
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html

---

## Recommendations

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| Clean data, want statistical rigor | **Quantile Regression** |
| Known mislabeled coordinates in data | LP + explicit filtering |
| Need confidence intervals | **Quantile Regression** |
| Reproducing CBG paper methodology | LP (paper's method) |
| Want to avoid parameter tuning | **Quantile Regression** (just set τ) |
| Strict "below all points" requirement | LP + filtering |

### Proposed Hybrid Approach

1. **Basic filtering** (Stages 1 & 4 only):
   - Remove invalid values (zero, negative, inf)
   - Remove physically impossible RTTs (below speed-of-light)

2. **Quantile regression** on filtered data:
   - τ = 0.05 for 5th percentile lower bound
   - No binning, no σ-based filtering needed

3. **Validation**:
   - Check that ~5% of points are below the line
   - Compare slope to theoretical (~0.01 ms/km)
   - Visual inspection of fit quality

### Next Steps

1. [ ] Add `fit_bestline_quantile()` function to `rtt_model.py`
2. [ ] Add `method='quantile'` option to `RTTDistanceModel.fit()`
3. [ ] Compare CBG geolocation accuracy between LP and QR methods
4. [ ] Document recommended default method

---

## Visualization

See generated plots:
- `outputs/quantile_vs_lp_real_data.png` - Comparison on real AS7922 data
- `outputs/quantile_check_function.png` - The asymmetric loss function

---

## Discussion Notes

### Why the "Minimum Filter" (Red Line) Was Off

In the original figure showing the red dashed line far from the data trend:
- LP constraint is "below ALL points"
- Even one anomalously low RTT (mislabeled coordinate) drags the entire line down
- This is why multi-stage filtering was added
- Quantile regression avoids this by design (5% below is acceptable)

### Heteroscedasticity

RTT variance increases with distance because:
- Longer paths have more routing variability
- More opportunities for congestion
- More diverse network conditions

Quantile regression handles this naturally — it estimates the conditional quantile at each distance, adapting to local variance.

### Physical Constraints

Both methods should respect the speed-of-light constraint:
- Minimum slope ≈ 0.01 ms/km (2/3 speed of light)
- For LP: explicit constraint in optimization
- For QR: add as post-hoc validation or pre-filter impossible points
