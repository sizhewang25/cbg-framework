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
