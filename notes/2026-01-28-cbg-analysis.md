# CBG Analysis for MNO Geolocation Use Case

## 2026-01-28 - Initial Data Assessment & Gap Analysis

### Project Goal
Mimic the scenario where **MNOs (Mobile Network Operators)** use their mobile cores as vantage points to geolocate **hypergiant IP addresses** from other ASNs.

### Data Available

**Dataset**: `datasets/cbg_test/vultr_pings_us_only.csv` (~1.4MB)

| Metric | Value |
|--------|-------|
| Total measurements | 9,866 |
| Unique probes (potential targets) | 1,423 |
| Unique probe ASNs | ~500+ |
| Vultr anchors (potential VPs) | 7 |
| Anchor ASN | AS20473 (Vultr) |

**Key columns**: `src_ip`, `dst_ip`, `min_rtt`, `probe_latitude/longitude`, `anchor_latitude/longitude`, `probe_asn`

### Analysis Scripts Available

| Script | Purpose |
|--------|---------|
| [scripts/processing/analyze_vultr_pings.py](../scripts/processing/analyze_vultr_pings.py) | Query ClickHouse for Vultr pings |
| [scripts/processing/analyze_vultr_pings_us_only.py](../scripts/processing/analyze_vultr_pings_us_only.py) | ASN analysis + interactive HTML maps |
| [analysis/cbg_tutorial.ipynb](../analysis/cbg_tutorial.ipynb) | Step-by-step CBG tutorial |

### [GAP] Current Implementation vs. Original CBG

**Gap 1: VP/Target Role Reversal**
- Current: Probes → Anchors (probes as VPs, anchors as targets)
- Needed: Anchors → Probes (Vultr anchors as VPs, probes grouped by ASN as targets)

**Gap 2: No Bestline Calibration** ⚠️ CRITICAL
- Current: Fixed `d = 100 × RTT` (using `speed_threshold = 2/3`)
- Original CBG: Calibrated bestline from **lower envelope** of (RTT, distance) scatter
  ```
  d_max = slope × RTT + intercept
  ```
  Where slope/intercept are fitted per-VP from known-location measurements

**Gap 3: No Per-VP Calibration**
- Current: Global constant for all VPs
- Original CBG: Each VP has its own calibrated distance function

### [ALGORITHM] Original CBG Calibration Method

```
For each VP (Vultr anchor):
  1. Collect (RTT, true_distance) pairs to other known-location nodes
  2. Plot scatter: X = RTT, Y = true_distance
  3. Fit LOWER ENVELOPE (e.g., 5th percentile quantile regression)
  4. Result: d_max = slope × RTT + intercept for this VP
  5. Use calibrated function when geolocating unknown targets
```

**Rationale**: RTT can be inflated (routing detours, queuing) but can NEVER be faster than physics. Lower envelope captures the "best case" network speed.

### [DATA] Available for Calibration

To calibrate, we need measurements between **pairs of known-location nodes**:
- Vultr anchor ↔ Vultr anchor (if available)
- Vultr anchor ↔ known-location probes

Current dataset has: Probes → Vultr anchors
- Probe locations are known (from RIPE Atlas)
- Can use this to calibrate each Vultr anchor's bestline!

### Next Steps

- [ ] Reverse data perspective: group by target probe ASN
- [ ] Implement bestline calibration using lower envelope fitting
- [ ] Create per-anchor calibration parameters
- [ ] Test CBG with calibrated distance function
- [ ] Compare accuracy: fixed 2/3 vs. calibrated bestline

---
