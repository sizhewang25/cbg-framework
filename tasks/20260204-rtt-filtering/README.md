# Task: RTT Filtering for CBG Lower Bound Estimation

## Objective

Research and implement optimal RTT data filtering methods to capture the lower bound envelope for Constraint-Based Geolocation (CBG) RTT-distance modeling.

## Background

In CBG, we need to convert RTT measurements to maximum possible distances. This requires fitting a "bestline" that represents the lower bound of the RTT-distance relationship. The challenge is that RTT data is noisy with:
- Positive outliers (congestion, queuing delays)
- Occasional negative outliers (mislabeled coordinates)
- Heteroscedastic variance (increases with distance)

## Current Status

- [x] Research statistical approaches
- [x] Implement 5-stage LP filtering (current approach)
- [x] Research and demo quantile regression
- [ ] Integrate quantile regression into rtt_model.py
- [ ] Comparative evaluation on full dataset
- [ ] Document final recommendations

## Key Files

- `scripts/analysis/cbg_feasibility/rtt_model.py` - Current LP + 5-stage filter implementation
- `scripts/analysis/cbg_feasibility/quantile_regression_demo.py` - QR demonstration
- `tasks/rtt-filtering/findings.md` - Research findings and recommendations
