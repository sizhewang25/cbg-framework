# Technical Context

## Problem Definition

**Goal**: Convert RTT measurements to maximum possible distances for CBG multilateration.

**Challenge**: RTT data is noisy — we need to find the "lower bound" that represents the minimum achievable RTT at each distance.

## Key Insight

The RTT-distance relationship is fundamentally:
```
RTT = slope × distance + intercept + noise
```

Where:
- `slope` ≈ 0.01-0.02 ms/km (related to propagation speed, typically 2/3 of light speed)
- `intercept` = baseline processing/queuing delay (typically 5-15 ms)
- `noise` = mostly positive (congestion adds delay, rarely subtracts)

## Data Characteristics

1. **Asymmetric noise**: RTT can be inflated (congestion) but rarely faster than physics allows
2. **Heteroscedastic**: Variance increases with distance
3. **Occasional anomalies**: Mislabeled coordinates create impossibly-low RTTs
4. **Physical bound**: RTT ≥ 2 × distance / (speed of light)

## Code References

### Current Implementation
- `scripts/analysis/cbg_feasibility/rtt_model.py:58` - `filter_rtt_data()` - 5-stage filtering
- `scripts/analysis/cbg_feasibility/rtt_model.py:274` - `fit_bestline_lp()` - LP optimization
- `scripts/analysis/cbg_feasibility/rtt_model.py:505` - `fit_bestline()` - Binned percentile method
- `scripts/analysis/cbg_feasibility/rtt_model.py:692` - `RTTDistanceModel` class

### Quantile Regression Demo
- `scripts/analysis/cbg_feasibility/quantile_regression_demo.py` - Theory and demonstration

### Data
- `scripts/analysis/cbg_feasibility/data/vultr_pings_us_only.csv` - Test dataset

## Filter Configuration Parameters

Current LP + 5-stage filter parameters:
```python
filter_rtt_data(
    distances, rtts,
    baseline_slope=0.01,       # ms/km (2/3 speed of light)
    bin_size_km=100.0,         # Distance bin size
    n_std=1.0,                 # Per-bin σ threshold
    global_n_std=1.0,          # Global bin-min σ threshold
    bin_percentile=0.05,       # Per-bin percentile (5%)
    enable_bin_filter=True,
    enable_percentile_filter=True,
    enable_global_filter=True,
    enable_baseline_filter=True
)
```

Quantile regression parameters:
```python
QuantReg(rtts, X).fit(q=0.05)  # Just τ!
```

## Evaluation Metrics

1. **% points below line**: Should be ~5% for τ=0.05
2. **Slope accuracy**: Compare to theoretical 0.01 ms/km
3. **Visual fit quality**: Does line follow lower envelope?
4. **CBG geolocation error**: Ultimate measure of model quality
