# Progress Log

## 2026-01-28 14:00 - Initial Setup & Gap Analysis

### What was done
1. Created CBG tutorial notebook (`analysis/cbg_tutorial.ipynb`) with step-by-step explanation
2. Analyzed current CBG implementation in codebase
3. Identified critical gaps between implementation and original CBG paper
4. Set up note-taking and task-focus skills for tracking

### Key Findings

**Gap 1: VP/Target Role Reversal Needed**
- Current: Probes (src_ip) → Anchors (dst_ip)
- Needed: Anchors (Vultr) → Probes (grouped by ASN)

**Gap 2: No Bestline Calibration (CRITICAL)**
- Current: Fixed `d = 100 × RTT` everywhere
- Original CBG: Calibrated from lower envelope of (RTT, distance) scatter
- Formula should be: `d = slope × RTT + intercept` per VP

**Gap 3: No Per-VP Calibration**
- Current: Global `speed_threshold = 2/3`
- Original: Each VP has own calibration parameters

### Data Available
- `datasets/cbg_test/vultr_pings_us_only.csv`: 9,866 measurements
- 1,423 unique probes, 7 Vultr anchors
- Probe locations known → can use for calibration

### Scripts Located
- `scripts/processing/analyze_vultr_pings_us_only.py` - ASN analysis + HTML maps
- Output maps in `scripts/processing/outputs/us_vultr_pings_AS*.html`

---

## 2026-01-28 16:00 - Implementation Plan Finalized

### What was done
1. Explored `scripts/analysis/cbg_feasibility/` folder structure
2. Found reference implementation in `references/scripts/get_rtt_dist_model_for_up.py`
3. Analyzed dataset composition and anchor locations
4. Created detailed implementation plan with design decisions

### Dataset Summary
- **7 Vultr anchors**: Seattle, Chicago, Atlanta, San Jose, LA, Dallas, Miami
- **335 unique ASNs**, 1,423 probes, 9,866 measurements
- **Top ASN**: 7922 (Comcast) with 266 probes → starting point for testing

### Design Decisions Made
1. **Bestline method**: 5th percentile quantile regression per distance bin (50km bins)
2. **Minimum threshold**: ≥3 distinct distance bins for valid model
3. **Starting ASN**: AS7922 (Comcast)
4. **Output formats**: Matplotlib PNG (scatter) + Folium HTML (maps)

### Implementation Plan (4 files)
1. `rtt_model.py` - Core module (haversine, fit_bestline, RTTDistanceModel class)
2. `test_rtt_model.py` - Unit tests
3. `fit_models.py` - Fit all anchor-ASN models, generate scatter plots
4. `visualize_cbg.py` - CBG multilateration maps

### Reference Code Found
- `references/scripts/get_rtt_dist_model_for_up.py` - Uses `cbg_mvp.bestline.calculate_bestline()`
- `references/scripts/geometry_utils.py` - Shapely-based circle intersection

---

## 2026-01-28 18:30 - Implementation Complete

### What was done
1. Created `rtt_model.py` with core functions (29 unit tests, all passing)
2. Created `fit_models.py` - fitted models for all 7 Vultr anchors with AS7922
3. Created `visualize_cbg.py` - CBG multilateration with Shapely circle intersection
4. Generated scatter plots and interactive maps

### Model Fitting Results (AS7922 Comcast, 266 probes)

| Anchor IP | Location | Slope (ms/km) | Intercept (ms) | R² | Bins |
|-----------|----------|---------------|----------------|-----|------|
| 45.77.211.82 | Seattle | 0.0156 | 11.6 | 0.73 | 48 |
| 66.42.119.57 | Chicago | 0.0171 | 12.5 | 0.85 | 44 |
| 144.202.18.114 | Atlanta | 0.0152 | 17.8 | 0.74 | 40 |
| 149.28.210.233 | San Jose | 0.0161 | 14.2 | 0.89 | 52 |
| 149.248.18.65 | LA | 0.0137 | 19.8 | 0.79 | 53 |
| 207.148.2.169 | Dallas | 0.0087 | 26.9 | 0.35 | 42 |
| 207.246.74.246 | Miami | 0.0138 | 18.1 | 0.76 | 48 |

**Summary Statistics:**
- Mean slope: 0.0143 ms/km (theoretical: 0.01 ms/km, ~43% slower)
- Mean intercept: 17.3 ms (processing/queuing delays)
- Mean R²: 0.73

### CBG Multilateration Results (10 test probes)

| Metric | Value |
|--------|-------|
| Mean error | 784 km |
| Median error | 788 km |
| Min error | 202 km |
| Max error | 1362 km |
| ≤250 km accuracy | 10% |
| ≤500 km accuracy | 30% |
| ≤1000 km accuracy | 70% |

### Key Observations

1. **Slopes are ~43% higher than theoretical (0.01 ms/km)**
   - Network inefficiency (routing detours, backbone latency)
   - This is expected for residential ISP traffic

2. **High intercepts (12-27 ms)**
   - Processing delays at endpoints
   - Dallas anchor has worst fit (R²=0.35, intercept=27ms)

3. **Accuracy limited by anchor diversity**
   - Only 7 anchors across continental US
   - Circles don't always intersect well
   - Edge probes (West Coast) harder to geolocate

### Files Created

```
scripts/analysis/cbg_feasibility/
├── rtt_model.py              # Core RTT modeling module
├── test_rtt_model.py         # 29 unit tests
├── fit_models.py             # Model fitting script
├── visualize_cbg.py          # CBG visualization script
└── outputs/vultr-7922-rtt-models/
    ├── *.pkl                 # 7 anchor models
    ├── scatter_*.png         # 7 RTT-distance plots
    ├── cbg_test_*.html       # 10 interactive maps
    ├── summary.json          # Model parameters
    └── cbg_results.json      # CBG test results
```

### Commits
- `c9f982d` - feat(cbg): add RTT-distance modeling module with unit tests
- `c60169b` - feat(cbg): add model fitting script and AS7922 results
- `e5092b1` - feat(cbg): add CBG multilateration visualization with Shapely

---
