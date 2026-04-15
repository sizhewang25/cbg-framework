# CBG Feasibility Technical Report

## Algorithm Design and Rationale

**Date**: 2026-01-28
**Task**: Constraint-Based Geolocation (CBG) Feasibility Exploration
**Target Dataset**: Vultr Anchors → RIPE Atlas Probes (AS7922 Comcast)

---

## 1. Overview

This report documents the algorithm design for implementing proper CBG with bestline calibration, addressing the gap between the existing million-scale implementation (fixed 2/3 speed threshold) and the original CBG method (per-VP calibrated parameters).

### Problem Statement

The existing codebase uses a fixed formula:
```
d_max = (2/3) × c × RTT / 2 = 100 × RTT (km)
```

This assumes uniform network characteristics globally. The original CBG paper instead calibrates each vantage point (VP) individually using measured RTT-distance relationships.

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CBG Pipeline                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. RTT-Distance Model Fitting (per anchor)                     │
│     └─> Binned 5th percentile → Linear regression → bestline    │
│                                                                  │
│  2. Distance Prediction (per RTT measurement)                   │
│     └─> d_max = (RTT - intercept) / slope                       │
│                                                                  │
│  3. Multilateration (circle intersection)                       │
│     └─> Shapely polygon intersection → centroid                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Haversine Distance Calculation

### Algorithm

```python
def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance on Earth's surface."""
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)² + cos(lat1) × cos(lat2) × sin(dlon/2)²
    c = 2 × arcsin(√a)

    return R_earth × c  # R_earth = 6371 km
```

### Rationale

- **Why Haversine?** Standard geodesic approximation for distances < 20,000 km
- **Alternatives considered**: Vincenty formula (more accurate but slower), Euclidean (invalid for geographic coordinates)
- **Accuracy**: < 0.5% error for continental US distances

---

## 3. Bestline Fitting Algorithm

### Design Choice: Binned 5th Percentile

The original CBG paper uses quantile regression to find the "lower envelope" of the RTT-distance scatter. We implement a simplified but equivalent approach:

```python
def fit_bestline(distances, rtts, bin_size_km=50, percentile=0.05):
    # Step 1: Bin data by distance
    bins = create_distance_bins(distances, bin_size_km)

    # Step 2: Compute 5th percentile RTT per bin
    bin_centers = []
    bin_rtts = []
    for bin in bins:
        if len(bin.rtts) >= 1:
            bin_centers.append(bin.center)
            bin_rtts.append(np.percentile(bin.rtts, 5))

    # Step 3: Linear regression through percentile points
    if len(bin_centers) >= 3:
        slope, intercept = np.polyfit(bin_centers, bin_rtts, 1)
        return {'slope': slope, 'intercept': intercept, 'success': True}
    else:
        return {'success': False}
```

### Rationale

1. **Why 5th percentile (not minimum)?**
   - Minimum is too sensitive to outliers and measurement noise
   - 5th percentile captures the "typical best case" for that distance bin
   - Robust to occasional anomalously low RTTs

2. **Why binning (not raw quantile regression)?**
   - Simpler implementation without specialized libraries
   - Handles heterogeneous point density (many probes at some distances, few at others)
   - Equivalent results when bins are well-populated

3. **Why 50 km bin size?**
   - Balances granularity vs statistical significance
   - At 50 km, even US coast-to-coast (~4000 km) gives ~80 potential bins
   - Matches the resolution of RIPE probe geolocation (~10-50 km accuracy)

4. **Why require ≥3 bins?**
   - 2 points always fit a perfect line (meaningless)
   - 3 points give minimum constraint for validating linearity
   - Ensures geographic spread, not just measurement count

### Mathematical Model

The bestline represents the physical lower bound on RTT:

```
RTT_min(d) = slope × d + intercept

Where:
- slope ≈ 2 / (f × c)  [ms/km, f = fraction of speed of light]
- intercept = processing delays at endpoints [ms]
```

Theoretical slope at 2/3 speed of light:
```
slope_theoretical = 2 / (2/3 × 300 km/ms) = 0.01 ms/km
```

---

## 4. RTT to Distance Conversion

### Algorithm

```python
def rtt_to_distance(rtt, slope, intercept) -> float:
    """
    Convert RTT to maximum possible distance.

    Inverts the bestline: RTT = slope × d + intercept
    Therefore: d = (RTT - intercept) / slope
    """
    if slope <= 0:
        return 0.0

    distance = (rtt - intercept) / slope
    return max(0.0, distance)  # Cannot be negative
```

### Rationale

1. **Why subtract intercept first?**
   - Intercept represents fixed delays (processing, queuing) unrelated to distance
   - Must be removed before distance calculation
   - Typical values: 12-27 ms in our dataset

2. **Why is this a maximum distance?**
   - RTT can be inflated (routing detours, congestion) but never below physical limit
   - Bestline represents the "best case" network speed for this VP
   - Actual target could be closer, but not farther

3. **Edge case: RTT < intercept**
   - Returns 0 distance (target must be very close)
   - Occurs when RTT is dominated by processing delays

---

## 5. Circle Intersection (Multilateration)

### Algorithm

```python
def find_circles_intersection(circles):
    """
    Find estimated location as centroid of circle intersection region.

    circles: List of (lat, lon, radius_km) tuples
    """
    # Step 1: Convert circles to polygon approximations
    polygons = [create_circle_polygon(c.lat, c.lon, c.radius) for c in circles]

    # Step 2: Iterative intersection
    intersection = polygons[0]
    for poly in polygons[1:]:
        intersection = intersection.intersection(poly)
        if intersection.is_empty:
            return fallback_weighted_centroid(circles)

    # Step 3: Return centroid of intersection region
    return intersection.centroid
```

### Rationale

1. **Why Shapely polygons (not analytical circle intersection)?**
   - Multiple circle intersection is computationally complex analytically
   - Shapely handles arbitrary n-circle intersection via polygon approximation
   - 64-point polygon approximation sufficient for km-scale accuracy

2. **Why centroid (not other estimators)?**
   - Centroid is the geometric "center of mass" of the feasible region
   - Minimizes expected squared error if target uniformly distributed in region
   - Simple, deterministic, no tuning parameters

3. **Fallback: Weighted centroid**
   - Used when circles don't fully intersect (common with only 7 anchors)
   - Weights each anchor position by 1/radius (closer constraints have more weight)
   - Provides reasonable estimate even with partial constraints

### Coordinate System Considerations

```python
def create_circle_polygon(lat, lon, radius_km, n_points=64):
    """
    Create polygon approximating circle on Earth's surface.

    Uses local tangent plane approximation:
    - 1° latitude ≈ 111 km (constant)
    - 1° longitude ≈ 111 × cos(lat) km (varies with latitude)
    """
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * cos(radians(lat))

    points = []
    for i in range(n_points):
        angle = 2π × i / n_points
        dx_km = radius × cos(angle)
        dy_km = radius × sin(angle)

        point_lat = lat + dy_km / km_per_deg_lat
        point_lon = lon + dx_km / km_per_deg_lon
        points.append((point_lon, point_lat))  # Shapely uses (x, y) = (lon, lat)

    return Polygon(points)
```

---

## 6. Model Persistence

### Design

```python
@dataclass
class RTTDistanceModel:
    anchor_ip: str
    anchor_lat: float
    anchor_lon: float
    slope: float          # ms/km
    intercept: float      # ms
    r_squared: float      # fit quality
    n_bins: int           # number of distance bins used
    bin_centers: List     # for visualization
    bin_rtts: List        # for visualization

    def save(self, path): pickle.dump(self, path)
    def load(cls, path): return pickle.load(path)
    def to_dict(self): ...  # For JSON export
```

### Rationale

1. **Why dataclass?**
   - Clean, self-documenting structure
   - Automatic `__init__`, `__repr__`
   - Easy serialization

2. **Why pickle (not JSON)?**
   - Preserves Python objects exactly
   - Faster for repeated load/save
   - JSON export available via `to_dict()` for interoperability

3. **Why store bin_centers/bin_rtts?**
   - Enables visualization of fitting process
   - Debugging: verify bestline matches data
   - No need to recompute for plotting

---

## 7. Validation Criteria

### Model Fitting Validation

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| n_bins ≥ 3 | Required | Minimum for meaningful fit |
| R² > 0.5 | Preferred | Indicates reasonable linear relationship |
| slope > 0 | Required | Distance must increase with RTT |
| slope < 0.03 | Expected | Would imply < 1/3 speed of light (unrealistic) |
| intercept > 0 | Expected | Processing delays always positive |
| intercept < 50 | Expected | > 50 ms suggests data quality issues |

### CBG Accuracy Validation

| Metric | Description |
|--------|-------------|
| Mean error | Average geolocation error in km |
| Median error | Robust central tendency (less affected by outliers) |
| ≤X km accuracy | Percentage of targets within X km of true location |

---

## 8. Results Summary

### Model Fitting (AS7922 Comcast)

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean slope | 0.0143 ms/km | 43% slower than theoretical (2/3 c) |
| Mean intercept | 17.3 ms | Typical endpoint processing delays |
| Mean R² | 0.73 | Good linear relationship |

### CBG Accuracy (10 test probes)

| Metric | Value |
|--------|-------|
| Mean error | 784 km |
| Median error | 788 km |
| ≤250 km | 10% |
| ≤500 km | 30% |
| ≤1000 km | 70% |

---

## 9. Limitations and Future Work

### Current Limitations

1. **Only 7 anchors**: Limited constraint diversity, especially for edge locations
2. **Continental US only**: Models may not generalize to other regions
3. **Single ASN tested**: AS7922 (Comcast) may have unique network characteristics
4. **Polygon approximation**: 64-point circles may introduce small errors for very large radii

### Potential Improvements

1. **More anchors**: Add Vultr locations in more cities
2. **Per-ASN models**: Different ISPs may have different network characteristics
3. **Time-of-day calibration**: Network performance varies throughout the day
4. **Confidence regions**: Report uncertainty, not just point estimate
5. **Machine learning**: Replace linear bestline with non-linear models for complex networks

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `rtt_model.py` | Core algorithms: haversine, fit_bestline, RTTDistanceModel |
| `test_rtt_model.py` | Unit tests (29 tests, 100% pass rate) |
| `fit_models.py` | Batch model fitting with scatter plot generation |
| `visualize_cbg.py` | Interactive multilateration maps |

---

## Appendix A: Constants

```python
EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_S = 300_000.0
SPEED_OF_LIGHT_KM_MS = 300.0  # km per millisecond
THEORETICAL_SLOPE = 2 / (300 * 2/3)  # ≈ 0.01 ms/km at 2/3 c
```

## Appendix B: Example Model Output

```json
{
  "anchor_ip": "45.77.211.82",
  "anchor_lat": 47.6095,
  "anchor_lon": -122.3415,
  "slope": 0.015600,
  "intercept": 11.57,
  "r_squared": 0.7294,
  "n_bins": 48,
  "n_measurements": 266,
  "fitted": true
}
```
