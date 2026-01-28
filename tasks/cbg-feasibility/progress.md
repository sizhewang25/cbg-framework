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
