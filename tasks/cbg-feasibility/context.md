# Technical Context

## Key Findings

### Current Implementation (Million-Scale Paper)
```python
# Fixed formula everywhere:
d_max = (2/3) × 300 × RTT / 2 = 100 × RTT (km)

# In helpers.py:279
intersections, circles = circle_intersections(circles, speed_threshold=2/3)

# In analysis.py:59
max_theoretical_distance = (SPEED_OF_INTERNET * min_rtt_probe / 1000) / 2
# where SPEED_OF_INTERNET = 200,000 km/s
```

### Original CBG Method (What We Need)
```python
# Per-VP calibration from known-location measurements:
For each VP:
    1. Collect (RTT, true_distance) pairs to known nodes
    2. Fit lower envelope (5th percentile quantile regression)
    3. Result: d_max = slope[vp] × RTT + intercept[vp]
```

### Why Lower Envelope?
- RTT can be **inflated** (routing detours, queuing, processing)
- RTT can **NEVER** be faster than physics allows
- Lower envelope = "best case" = speed of network for that VP

## Code References

### Core CBG Functions
- `scripts/utils/helpers.py:244` - `select_best_guess_centroid()` - Main CBG entry point
- `scripts/utils/helpers.py:23` - `rtt_to_km()` - RTT to distance conversion (needs modification)
- `scripts/utils/helpers.py:107` - `circle_intersections()` - Geometric intersection
- `scripts/utils/helpers.py:58` - `circle_preprocessing()` - Remove redundant circles
- `scripts/utils/helpers.py:169` - `polygon_centroid()` - Calculate centroid

### Analysis Pipeline
- `scripts/analysis/analysis.py:277` - `compute_rtts_per_dst_src()` - Fetch RTT from ClickHouse
- `scripts/analysis/analysis.py:151` - `compute_geolocation_features_per_ip()` - Main analysis loop
- `scripts/analysis/analysis.py:363` - `compute_error()` - Calculate geolocation error

### Data Processing
- `scripts/processing/analyze_vultr_pings_us_only.py` - ASN grouping + map generation

## Dataset Schema

**vultr_pings_us_only.csv**:
| Column | Description |
|--------|-------------|
| `src_ip` | Probe IP (current VP, future target) |
| `dst_ip` | Anchor IP (current target, future VP) |
| `min_rtt` | Minimum RTT in ms |
| `probe_latitude/longitude` | Probe location (known) |
| `anchor_latitude/longitude` | Anchor location (known) |
| `probe_asn` | Probe's ASN (for grouping targets) |

## Decisions Made

1. **Use Vultr anchors as VPs**: They have known locations and represent datacenter infrastructure
2. **Group probes by ASN as targets**: Simulates geolocating hypergiant IPs
3. **Implement quantile regression**: For lower envelope fitting (5th percentile)

## Open Questions

1. How many measurements per anchor are needed for reliable calibration?
2. Should calibration be per-anchor or per-region?
3. What percentile to use for lower envelope? (5th? 10th?)
4. How to handle anchors with few measurements?

## Constants

```python
SPEED_OF_LIGHT = 300_000  # km/s
SPEED_OF_INTERNET = 200_000  # km/s (2/3 of c) - current fixed value
EARTH_RADIUS = 6371  # km
```
