# Task: CBG Feasibility Exploration

## Background

We want to explore using **Constraint-Based Geolocation (CBG)** for a specific use case: **MNOs (Mobile Network Operators) using their mobile cores as vantage points to geolocate hypergiant IP addresses** from other ASNs.

The existing codebase implements a simplified CBG from the IMC 2023 replication paper, but it differs from the original CBG paper in critical ways - particularly the lack of **bestline calibration**.

## Goals

1. Understand how the current CBG implementation works
2. Identify gaps between current implementation and original CBG
3. Implement proper CBG with bestline calibration
4. Test feasibility using Vultr anchors as VPs and probes (grouped by ASN) as targets
5. Evaluate accuracy compared to fixed 2/3 speed threshold

## Requirements

### Data Setup
- Use Vultr anchors (AS20473) as **Vantage Points (VPs)**
- Use other probes grouped by ASN as **Targets**
- Reverse the current VP/target perspective

### Algorithm Implementation
- Implement **bestline calibration** using lower envelope of (RTT, distance) scatter
- Create **per-VP calibration parameters** (slope, intercept per anchor)
- Replace fixed `d = 100 × RTT` with calibrated `d = slope × RTT + intercept`

### Evaluation
- Compare accuracy: calibrated vs. fixed speed threshold
- Analyze by target ASN
- Visualize results with maps

## Success Criteria

- [ ] Bestline calibration implemented and validated
- [ ] Per-anchor calibration parameters computed
- [ ] CBG working with reversed VP/target roles
- [ ] Accuracy comparison completed
- [ ] Results documented with visualizations

## Related Files

### Data
- [datasets/cbg_test/vultr_pings_us_only.csv](../../datasets/cbg_test/vultr_pings_us_only.csv) - Main dataset

### Analysis Scripts
- [analysis/cbg_tutorial.ipynb](../../analysis/cbg_tutorial.ipynb) - CBG step-by-step tutorial
- [scripts/processing/analyze_vultr_pings_us_only.py](../../scripts/processing/analyze_vultr_pings_us_only.py) - ASN analysis + maps

### Core CBG Implementation
- [scripts/utils/helpers.py](../../scripts/utils/helpers.py) - `select_best_guess_centroid()`, `rtt_to_km()`, `circle_intersections()`
- [scripts/analysis/analysis.py](../../scripts/analysis/analysis.py) - `compute_geolocation_features_per_ip()`

### Documentation
- [notes/2026-01-28-cbg-analysis.md](../../notes/2026-01-28-cbg-analysis.md) - Initial analysis notes
- [CBG_ALGORITHM_VISUAL.md](../../CBG_ALGORITHM_VISUAL.md) - Visual explanation of CBG
