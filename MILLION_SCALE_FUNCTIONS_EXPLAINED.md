# Million-Scale Geolocation Functions Explained

This document provides detailed explanations of the key functions used in the million-scale IP geolocation algorithm, based on the IMC 2012 paper "Towards geolocation of millions of IP addresses".

---

## Table of Contents

1. [Overview](#overview)
2. [Function 1: compute_rtts_per_dst_src()](#1-compute_rtts_per_dst_src)
3. [Function 2: compute_closest_rtt_probes()](#2-compute_closest_rtt_probes)
4. [Function 3: circle_preprocessing()](#3-circle_preprocessing)
5. [Function 4: select_best_guess_centroid()](#4-select_best_guess_centroid)
6. [Function 5: compute_geolocation_features_per_ip()](#5-compute_geolocation_features_per_ip)
7. [Data Flow Diagram](#data-flow-diagram)
8. [Mathematical Background](#mathematical-background)

---

## Overview

The million-scale geolocation approach uses **Constraint-Based Geolocation (CBG)**, which:
1. Measures RTT (Round-Trip Time) from multiple vantage points (VPs) to a target IP
2. Converts RTT to distance using the speed of light in fiber
3. Creates circles centered at each VP with radius = calculated distance
4. Finds the intersection of all circles
5. Estimates the target's location as the centroid of the intersection region

**Key Insight:** The target must be within all circles simultaneously, so the intersection narrows down the possible location.

---

## 1. compute_rtts_per_dst_src()

**Location:** `scripts/analysis/analysis.py:277-306`

### Purpose
Queries ClickHouse database to extract minimum RTT measurements between source VPs and destination targets.

### Function Signature
```python
def compute_rtts_per_dst_src(
    table,           # ClickHouse table name (e.g., "probes_to_prefix_pings")
    filter,          # SQL WHERE clause filter
    threshold,       # RTT threshold for filtering
    is_per_prefix    # True: group by prefix, False: by IP
)
```

### What It Does

1. **Connects to ClickHouse** using credentials from environment
2. **Constructs SQL query** to get min RTT per source-destination pair
   - If `is_per_prefix=False`: `get_min_rtt_per_src_dst_query()`
   - If `is_per_prefix=True`: `get_min_rtt_per_src_dst_prefix_query()`
3. **Executes query iteratively** (memory-efficient for large datasets)
4. **Builds nested dictionary**:
   ```python
   {
       'dst_ip_1': {
           'src_ip_1': [min_rtt_1],
           'src_ip_2': [min_rtt_2],
           ...
       },
       'dst_ip_2': {
           ...
       }
   }
   ```
5. **Returns** the RTT data structure

### Example Output
```python
{
    '213.225.160.239': {
        '192.0.2.1': [45.2],     # min RTT from VP 192.0.2.1
        '198.51.100.5': [89.7],   # min RTT from VP 198.51.100.5
        '203.0.113.10': [123.4]   # min RTT from VP 203.0.113.10
    }
}
```

### Why It's Important
- **First step** in the geolocation pipeline
- Provides the **raw RTT data** needed for all subsequent calculations
- Handles **millions of measurements** efficiently using ClickHouse
- Filters out measurements with no response (`rcvd > 0`)

---

## 2. compute_closest_rtt_probes()

**Location:** `scripts/analysis/analysis.py:35-70`

### Purpose
Implements the **VP Selection Algorithm**: Selects the optimal subset of vantage points for geolocating each target based on RTT and physical distance constraints.

### Function Signature
```python
def compute_closest_rtt_probes(
    rtts_per_dst_prefix,      # RTT data from compute_rtts_per_dst_src()
    vp_coordinates_per_ip,    # VP coordinates {ip: (lat, lon)}
    vp_distance_matrix,       # Precomputed distances {dst: {vp: distance}}
    is_prefix,                # True if working with prefixes
    n_shortest=10             # Number of VPs to select (default 10)
)
```

### Algorithm Steps

#### Step 1: Sort VPs by RTT
For each target, sort all VPs by their minimum RTT (ascending):
```python
sorted_probes = sorted(src_min_rtt.items(), key=lambda x: x[1][0])
```

#### Step 2: Select N Shortest
Take the top N VPs with lowest RTT:
```python
n_shortest_probes = dict(sorted_probes[:n_shortest])  # Default: 10 VPs
```

#### Step 3: Speed-of-Light Validation (if not prefix)
For each selected VP, check if the physical distance is physically possible:

```python
# Calculate maximum possible distance based on RTT
max_theoretical_distance = (SPEED_OF_INTERNET * min_rtt_probe / 1000) / 2

# Check if actual distance violates physics
if vp_distance_matrix[dst][probe] > max_theoretical_distance:
    # Impossible! Either:
    # - Wrong geolocation data
    # - Routing anomaly
    # - Anycast
    continue  # Reject this VP
```

**Formula Breakdown:**
- `SPEED_OF_INTERNET = 200,000 km/s` (2/3 speed of light in fiber)
- `min_rtt_probe / 1000` = convert ms to seconds
- `× SPEED_OF_INTERNET` = total distance traveled (round trip)
- `/ 2` = one-way distance

#### Step 4: Return Valid VPs
```python
vps_per_prefix[dst] = n_shortest_probes_checked
```

### Example

**Input:**
- Target: `213.225.160.239` (France)
- 1000 VPs have measurements

**VP Ranking by RTT:**
```
VP              RTT     Physical Distance  Valid?
192.0.2.1       12ms    150 km            ✓ (200,000 × 0.012 / 2 = 1,200 km max)
198.51.100.5    45ms    500 km            ✓
203.0.113.10    89ms    2,000 km          ✓
203.0.113.20    15ms    5,000 km          ✗ (Impossible! Max = 1,500 km)
...
```

**Output:** Top 10 valid VPs

### Why This Matters
- **Geographic diversity**: Closest VPs provide tighter constraints
- **Accuracy**: Removing physically impossible VPs prevents location errors
- **Efficiency**: Only use the most informative VPs (10 instead of 1000)
- **Detects anomalies**: Identifies wrongly-geolocated probes or anycast

---

## 3. circle_preprocessing()

**Location:** `scripts/utils/helpers.py:58-88`

### Purpose
Removes redundant circles using the **circle inclusion rule**: If one circle is entirely contained within another, remove the larger circle (it provides no additional constraint).

### Function Signature
```python
def circle_preprocessing(circles, speed_threshold=None)
```

### Algorithm Steps

#### Step 1: Add Radius Information
Convert RTT to distance and angular radius:
```python
for c in circles:
    lat, lon, rtt, d, r = c
    if d is None:
        d = rtt_to_km(rtt, speed_threshold)  # RTT → km
    if r is None:
        r = d / 6371  # km → radians (Earth radius = 6371 km)
```

#### Step 2: Check Pairwise Circle Inclusion
For every pair of circles, check if one contains the other:

```python
def check_circle_inclusion(c_1, c_2):
    lat_1, lon_1, rtt_1, d_1, r_1 = c_1
    lat_2, lon_2, rtt_2, d_2, r_2 = c_2

    # Distance between circle centers
    d = haversine((lat_1, lon_1), (lat_2, lon_2))

    # Circle 1 contains Circle 2
    if d_1 > (d + d_2):
        return c_1, c_2  # Remove c_1 (larger), keep c_2

    # Circle 2 contains Circle 1
    elif d_2 > (d + d_1):
        return c_2, c_1  # Remove c_2 (larger), keep c_1

    return None, None  # No inclusion
```

#### Step 3: Remove Included Circles
```python
for i in range(len(circles)):
    for j in range(i + 1, len(circles)):
        remove, keep = check_circle_inclusion(c_i, c_j)
        if remove:
            circles_to_ignore.add(remove)

circles_to_keep = set(circles) - circles_to_ignore
```

### Visual Example

**Before Preprocessing:**
```
VP1: (Paris, 48.8°N, 2.3°E), RTT=10ms → radius=1000km
VP2: (Lyon, 45.7°N, 4.8°E), RTT=5ms  → radius=500km
```

If VP2's circle is entirely inside VP1's circle:
- **Remove VP1's circle** (less informative)
- **Keep VP2's circle** (tighter constraint)

**Result:** Better precision with fewer circles

### Why This Matters
- **Reduces computational complexity**: Fewer circles = faster intersection calculation
- **Improves accuracy**: Removes redundant/less precise constraints
- **Handles nested VPs**: When VPs are geographically close, only keep the closest one

---

## 4. select_best_guess_centroid()

**Location:** `scripts/utils/helpers.py:244-290`

### Purpose
The **core geolocation function**: Computes the estimated location by finding the centroid of the circle intersection region.

### Function Signature
```python
def select_best_guess_centroid(
    target_ip,                  # Target IP to geolocate
    vp_coordinates_per_ip,      # {ip: (lat, lon)} for all VPs
    rtt_per_vp_to_target        # {vp_ip: [rtts]} from VPs to target
)
```

### Algorithm Steps

#### Step 1: Create Circles from RTT Measurements
```python
probe_circles = {}
for vp_ip, rtts in rtt_per_vp_to_target.items():
    if target_ip == vp_ip:
        continue  # Skip self-measurement

    if vp_ip not in vp_coordinates_per_ip:
        continue  # Skip VPs with unknown location

    lat, lon = vp_coordinates_per_ip[vp_ip]
    min_rtt = min(rtts)

    if min_rtt > 100:
        continue  # Filter inflated RTTs (likely routing anomalies)

    probe_circles[vp_ip] = (lat, lon, min_rtt, None, None)
```

**Circle representation:** `(lat, lon, rtt, distance, radius)`
- `distance` and `radius` are computed later

#### Step 2: Find Circle Intersections
```python
intersections, circles = circle_intersections(circles, speed_threshold=2/3)
```

This performs **spherical geometry** to find all intersection points:
1. Converts lat/lon to 3D Cartesian coordinates
2. For each pair of circles, computes 2 intersection points
3. Filters points that lie within ALL circles
4. Returns valid intersection points

**Mathematical Details:** See [circle_intersections() in helpers.py:107-166]

#### Step 3: Compute Centroid

**Case 1: Multiple Intersection Points (>2)**
```python
if len(intersections) > 2:
    centroid = polygon_centroid(intersections)
```
Centroid = average of all intersection points:
```python
def polygon_centroid(points):
    x = sum(point[0] for point in points) / len(points)  # avg latitude
    y = sum(point[1] for point in points) / len(points)  # avg longitude
    return (x, y)
```

**Case 2: Two Intersection Points**
```python
elif len(intersections) == 2:
    centroid = get_middle_intersection(intersections)
```
Midpoint of the segment between the two points (spherical geometry).

**Case 3: Zero or One Circle**
```python
else:
    closest_vp, _ = min(min_rtt_per_vp_ip.items(), key=lambda x: x[1])
    centroid = vp_coordinates_per_ip[closest_vp]
```
Fallback: Use location of closest VP

#### Step 4: Return Result
```python
return centroid, circles
```

### Visual Example

**Scenario:** Geolocating target in Paris

**Input:**
- VP1 (London): RTT=12ms → Circle with radius 1200km centered at London
- VP2 (Frankfurt): RTT=15ms → Circle with radius 1500km centered at Frankfurt
- VP3 (Amsterdam): RTT=10ms → Circle with radius 1000km centered at Amsterdam

**Step 1:** Draw 3 circles on map
**Step 2:** Find intersection region (shaded area where all 3 circles overlap)
**Step 3:** Calculate intersection points: 6 points (2 per circle pair)
**Step 4:** Filter to points inside ALL circles: 4 points remain
**Step 5:** Centroid = average of 4 points ≈ Paris coordinates

**Output:** `(48.85°N, 2.35°E)` (near Paris)

### Why This Matters
- **The heart of CBG**: This is where actual geolocation happens
- **Geometric precision**: Uses spherical geometry for Earth's curvature
- **Robust**: Handles edge cases (0, 1, 2, or many intersections)
- **Simple**: Centroid is an intuitive location estimate

---

## 5. compute_geolocation_features_per_ip()

**Location:** `scripts/analysis/analysis.py:151-222`

### Purpose
**Top-level orchestration function**: Coordinates the entire geolocation pipeline for all targets with multiple configurations (thresholds, VP counts, etc.).

### Function Signature
```python
def compute_geolocation_features_per_ip(
    rtt_per_srcs_dst,         # RTT data for all targets
    vp_coordinates_per_ip,    # VP coordinates
    threshold_distances,      # [0, 40, 100, 500, 1000] km
    vps_per_target,           # Selected VPs per target (from compute_closest_rtt_probes)
    distance_operator,        # ">" or "<=" for filtering VPs
    max_vps,                  # Maximum VPs to use per target
    is_use_prefix,            # Prefix-based or IP-based
    vp_distance_matrix,       # Precomputed distances
    is_multiprocess=True      # Use parallel processing
)
```

### Algorithm Flow

#### Step 1: Prepare Arguments for Each Target
```python
args = []
for dst, rtt_per_src in sorted(rtt_per_srcs_dst.items()):
    if dst not in vp_coordinates_per_ip:
        continue  # Skip targets with unknown location (for ground truth comparison)

    args.append((
        dst,
        rtt_per_src,
        vps_per_target,
        vp_coordinates_per_ip,
        vp_distance_matrix[dst],
        threshold_distances,
        distance_operator,
        max_vps,
        is_use_prefix,
    ))
```

#### Step 2: Process in Parallel (if enabled)
```python
if is_multiprocess:
    with Pool(24) as p:  # 24 worker processes
        features_all_process = p.starmap(
            compute_geolocation_features_per_ip_impl,
            args
        )
```

**Benefit:** Process hundreds of targets simultaneously on multi-core CPU

#### Step 3: Per-Target Processing (compute_geolocation_features_per_ip_impl)

For each target, test multiple threshold configurations:

```python
for threshold_distance in threshold_distances:  # [0, 40, 100, 500, 1000]
    # Filter VPs based on distance threshold
    if distance_operator == ">":
        # Only use VPs farther than threshold (test global coverage)
        vp_filter = {vp for vp in vps
                     if vp_distance_matrix[dst][vp] > threshold_distance}
    elif distance_operator == "<=":
        # Only use VPs closer than threshold (test local accuracy)
        vp_filter = {vp for vp in vps
                     if vp_distance_matrix[dst][vp] <= threshold_distance}

    # Limit to max_vps (randomly sample if too many)
    if len(vp_filter) > max_vps:
        vp_filter = random.sample(vp_filter, max_vps)

    # Compute geolocation error for this configuration
    error, circles = compute_error(dst, vp_filter, rtt_per_src)

    features[threshold_distance].append((dst, error, len(circles)))
```

#### Step 4: Aggregate Results
```python
features = {}
for features_process in features_all_process:
    for threshold, dst_error_distances in features_process.items():
        features.setdefault(threshold, []).extend(dst_error_distances)
```

**Output Structure:**
```python
{
    0: [(dst1, error1, n_circles1), (dst2, error2, n_circles2), ...],
    40: [...],
    100: [...],
    500: [...],
    1000: [...]
}
```

### Example Workflow

**Input:**
- 500 targets to geolocate
- 10,000 available VPs
- 10 VPs selected per target (from compute_closest_rtt_probes)
- Test 5 distance thresholds: [0, 40, 100, 500, 1000] km

**Processing:**
```
For each target:
    Threshold 0km (all VPs):
        → Use all 10 VPs
        → Compute geolocation
        → Error = 45 km

    Threshold 40km (VPs > 40km away):
        → Use 8 VPs (2 too close)
        → Compute geolocation
        → Error = 52 km

    Threshold 100km (VPs > 100km away):
        → Use 6 VPs (4 too close)
        → Compute geolocation
        → Error = 78 km

    ... and so on
```

**Purpose:** Analyze how VP selection affects accuracy

### Why This Matters
- **Comprehensive evaluation**: Tests multiple configurations simultaneously
- **Scalable**: Multiprocessing handles millions of targets efficiently
- **Research-oriented**: Generates data for plotting accuracy vs. threshold
- **Flexible**: Supports both prefix-based and IP-based geolocation

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. compute_rtts_per_dst_src()                                   │
│    Input: ClickHouse database                                   │
│    Output: {dst: {src: [min_rtt]}}                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. compute_closest_rtt_probes()                                 │
│    Input: RTT data, VP coordinates, distance matrix             │
│    Process: Sort by RTT, validate speed-of-light constraint     │
│    Output: {dst: [selected_vps]}                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. compute_geolocation_features_per_ip()                        │
│    Orchestrates geolocation for all targets                     │
│    ├─→ For each target & threshold:                             │
│    │   ├─→ Filter VPs by distance threshold                     │
│    │   └─→ Call compute_error()                                 │
│    │       └─→ Call select_best_guess_centroid()                │
│    │           ├─→ Create circles from RTT                       │
│    │           ├─→ Call circle_preprocessing() (remove redundant)│
│    │           ├─→ Call circle_intersections() (find overlap)   │
│    │           └─→ Call polygon_centroid() (estimate location)  │
│    └─→ Aggregate errors across configurations                   │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output: Geolocation accuracy metrics                            │
│ {threshold: [(dst, error, n_circles), ...]}                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Background

### RTT to Distance Conversion

**Formula:**
```
distance (km) = (RTT / 2) × speed_of_internet
```

**Speed Model (adaptive):**
```python
if RTT >= 80ms:
    speed = 4/9 × speed_of_light  # Long distance, better routing
elif 5ms <= RTT < 80ms:
    speed = 3/9 × speed_of_light  # Medium distance
elif RTT < 5ms:
    speed = 1/6 × speed_of_light  # Short distance, more hops
```

**Why adaptive?** Routing efficiency varies with distance.

### Circle Inclusion Test

**Condition:** Circle 1 includes Circle 2 if:
```
d(center1, center2) + radius2 < radius1
```

Where `d()` is haversine distance.

### Haversine Formula (Great Circle Distance)

**Purpose:** Calculate distance between two points on a sphere.

**Formula:**
```
a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
c = 2 × atan2(√a, √(1−a))
distance = R × c
```

Where `R = 6371 km` (Earth's radius).

**Implementation:** `helpers.py:182-196`

### Circle Intersection (Spherical Geometry)

**Problem:** Find intersection points of two circles on a sphere.

**Approach:**
1. Convert lat/lon to 3D Cartesian coordinates (x, y, z)
2. Use vector math to find intersection line
3. Solve for two intersection points
4. Convert back to lat/lon

**Implementation:** `helpers.py:107-166`

**Reference:** https://gis.stackexchange.com/questions/48937/calculating-intersection-of-two-circles

---

## Key Constants

**From `default.py`:**
```python
SPEED_OF_LIGHT = 300,000 km/s
SPEED_OF_INTERNET = SPEED_OF_LIGHT × 2/3 = 200,000 km/s
THRESHOLD_DISTANCES = [0, 40, 100, 500, 1000] km  # For accuracy evaluation
```

---

## Common Pitfalls & Edge Cases

### 1. No Circle Intersections
**Cause:** VPs too far apart, RTTs inconsistent
**Solution:** Fallback to closest VP location

### 2. Inflated RTTs (>100ms)
**Cause:** Routing anomalies, congestion, anycast
**Solution:** Filter out RTTs >100ms in select_best_guess_centroid()

### 3. Wrongly-Geolocated VPs
**Cause:** Outdated geolocation databases
**Solution:** Speed-of-light validation in compute_closest_rtt_probes()

### 4. Anycast
**Cause:** Same IP serves from multiple locations
**Detection:** Distance violations in VP selection
**Solution:** Detected but not handled (requires different approach)

---

## Performance Optimizations

1. **Multiprocessing:** Process 24 targets simultaneously
2. **ClickHouse:** Columnar database for fast RTT queries
3. **Circle preprocessing:** Remove redundant circles early
4. **RTT filtering:** Skip inflated RTTs (>100ms)
5. **VP limiting:** Max 1000 VPs per target (random sampling if needed)

---

## Summary

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `compute_rtts_per_dst_src` | ClickHouse DB | RTT dict | Extract measurement data |
| `compute_closest_rtt_probes` | RTT data | Selected VPs | VP selection algorithm |
| `circle_preprocessing` | Circles | Filtered circles | Remove redundant constraints |
| `select_best_guess_centroid` | RTT, VPs | (lat, lon) | **Core geolocation** |
| `compute_geolocation_features_per_ip` | All above | Error metrics | Orchestrate & evaluate |

**Core Algorithm:** CBG (Constraint-Based Geolocation) using circle intersections from RTT measurements.

**Accuracy:** Typically 100-500 km median error for targets with 10+ VPs (from original paper).

---

## Next Steps

To use these functions in practice:

1. **Load data:**
   ```python
   rtts = compute_rtts_per_dst_src("probes_to_prefix_pings", filter="", threshold=0)
   ```

2. **Select VPs:**
   ```python
   vps = compute_closest_rtt_probes(rtts, vp_coords, distance_matrix, is_prefix=False)
   ```

3. **Geolocate:**
   ```python
   features = compute_geolocation_features_per_ip(
       rtts, vp_coords, [0, 40, 100, 500, 1000], vps, ">", 1000, False, distance_matrix
   )
   ```

4. **Analyze results:**
   ```python
   for threshold, results in features.items():
       errors = [r[1] for r in results if r[1] is not None]
       print(f"Threshold {threshold}km: Median error = {np.median(errors):.1f} km")
   ```

See `analysis/million_scale.ipynb` for complete implementation.
