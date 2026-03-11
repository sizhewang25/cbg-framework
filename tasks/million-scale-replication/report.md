# Task: Million-Scale CBG Replication & Comparison

## Background

We applied the **Million-Scale CBG method** (from the IMC 2012 paper replication in this codebase) to our `vultr_pings_us_only.csv` dataset, and compared it side-by-side with the **Calibrated (Vanilla) CBG method** (LP bestline fitting from `filter_demonstration.py`).

## Goal

Evaluate and compare two CBG multilateration approaches on the same dataset (AS7922, 266 probes, 7 anchors) to quantify the accuracy improvement from per-anchor calibration.

## Output Script

**File:** `scripts/analysis/million_scale/evaluate_million_scale.py`

Produces:
- Per-anchor RTT-distance scatter plots with both model lines
- Comparative Error CDF plot
- Statistics table (console)
- `comparison_results.json`

**Output directory:** `scripts/analysis/million_scale/outputs/comparison/`

## Implementation Comparison

### Two CBG Variants

| Aspect | Million-Scale CBG | Calibrated (Vanilla) CBG |
|---|---|---|
| **RTT-to-Distance Model** | Fixed theoretical: `d = rtt * c * (2/3) / 2` | Per-anchor LP bestline: `d = (rtt - intercept) / slope` |
| **Speed Assumption** | Universal 2/3 speed of light | Calibrated per anchor via LP fitting |
| **Model Fitting** | None — hardcoded constant | `fit_bestline_lp()` with 5-stage filtering pipeline |
| **RTT Filtering (at fitting)** | N/A (no fitting step) | Stage 1: remove zero/negative/inf RTTs; Stage 2: baseline filter (speed-of-light constraint: `rtt >= THEORETICAL_SLOPE * distance`); Stages 3-5: optional bin-sigma, percentile, global filters |
| **RTT Filtering (at evaluation)** | Drops `min_rtt > 100` ms; `isinstance(min_rtt, float)` guard silently skips integer RTTs | Drops anchors where `predict_distance(rtt) <= 0` (i.e., `rtt < intercept`); requires `model.fitted` |
| **Circle Geometry** | Spherical (3D Cartesian vector math on unit sphere) | Planar (Shapely polygons, 64-point circle approximation) |
| **Circle Intersection** | `circle_intersections()` from `helpers.py` — exact spherical pairwise | `find_circles_intersection()` from `filter_demonstration.py` — Shapely `.intersection()` |
| **Circle Preprocessing** | `circle_preprocessing()` removes fully-contained circles | None — intersects all circles directly |
| **Single Circle Handling** | 4 evenly-spaced points on circumference via `get_points_on_circle()` | N/A — requires `len(circles) >= 2`, otherwise returns failure |
| **Centroid Computation** | `polygon_centroid()` on spherical intersection points | Shapely `.centroid` on resulting polygon |
| **No-Intersection Fallback** | Closest VP by min RTT (`min_rtt_per_vp_ip`) | `estimate_location_fallback()` — inverse-radius weighted average of all anchor positions |
| **Earth Radius** | Inconsistent: 6367 km (haversine), 6371 km (preprocessing), 6378.137 km (point generation) | Consistent via `haversine_distance()` from `rtt_model.py` |
| **Input Construction** | `{anchor_ip: [min_rtt]}` dict -> `select_best_guess_centroid()` | DataFrame rows -> `evaluate_cbg_probe()` with fitted `RTTDistanceModel` objects |
| **Source Files** | `scripts/utils/helpers.py` | `scripts/analysis/cbg_feasibility/rtt_model.py` + `filter_demonstration.py` |

### Key Differences Explained

1. **RTT-to-Distance conversion**: Million-Scale uses a single global speed assumption (2/3 c = 200,000 km/s). Calibrated CBG fits a per-anchor lower-envelope line that accounts for each anchor's actual routing conditions (inflation, queuing, detour paths).

2. **Filtering**: Million-Scale has a crude `min_rtt > 100` ms hard cutoff plus the `isinstance(float)` guard. Calibrated CBG filters during model fitting (invalid values, speed-of-light violations, optional statistical outlier removal) and during evaluation (negative predicted distance means RTT is below the intercept, indicating the measurement is faster than the model expects — likely a very close target).

3. **Geometry**: Million-Scale works in true spherical geometry (3D vectors, exact intersection points). Calibrated CBG approximates circles as 64-sided polygons and uses Shapely's planar intersection. For US-scale distances, the planar approximation is adequate.

4. **Fallback strategy**: When circles don't intersect, Million-Scale picks the closest anchor by RTT. Calibrated CBG computes an inverse-radius weighted average across all anchors, which tends to give a more stable (though not necessarily more accurate) estimate.

## Results (AS7922, Vultr US)

| Metric | Million-Scale CBG | Calibrated CBG |
|---|---|---|
| N (probes) | 266 | 266 |
| Median error (km) | 686.7 | 402.0 |
| Mean error (km) | ~750 | ~520 |
| 75th percentile (km) | ~1100 | ~750 |
| 90th percentile (km) | ~1400 | ~1000 |

**Key takeaway**: Calibrated CBG outperforms Million-Scale by ~285 km median error. The per-anchor LP fitting learns actual RTT-distance relationships (accounting for routing inflation, queuing delay), producing tighter distance constraints than the universal 2/3c assumption.

## Files Created

| File | Purpose |
|---|---|
| `scripts/analysis/million_scale/evaluate_million_scale.py` | Main comparison script |
| `scripts/analysis/million_scale/outputs/comparison/error_cdf_comparison.png` | Comparative CDF plot |
| `scripts/analysis/million_scale/outputs/comparison/scatter_*.png` | Per-anchor RTT-distance scatter (7 files) |
| `scripts/analysis/million_scale/outputs/comparison/comparison_results.json` | Machine-readable results |

## How to Run

```bash
cd /Users/wang.sizh/workspace/atnt/geoloc-imc-2023
python scripts/analysis/million_scale/evaluate_million_scale.py
```

## Functions Reused

| Function | Source | Used By |
|---|---|---|
| `select_best_guess_centroid()` | `helpers.py:244` | Million-Scale CBG |
| `haversine()` | `helpers.py:182` | Million-Scale error computation |
| `evaluate_cbg_probe()` | `filter_demonstration.py:429` | Calibrated CBG |
| `find_circles_intersection()` | `filter_demonstration.py:389` | Calibrated CBG (internal) |
| `estimate_location_fallback()` | `filter_demonstration.py:408` | Calibrated CBG (internal) |
| `fit_bestline_lp()` | `rtt_model.py:274` | LP model fitting |
| `haversine_distance()` | `rtt_model.py` | Calibrated CBG error + distance computation |
| `THEORETICAL_SLOPE` | `rtt_model.py:31` | Scatter plot theoretical line |
| `RTTDistanceModel` | `rtt_model.py:695` | Per-anchor model container |

## Arc-Based Intersection Visualization

**Commit context:** Replaced ConvexHull-based intersection rendering with Shapely polygon intersection.

### Problem

The original `plot_circles_on_map()` drew intersection regions using `scipy.spatial.ConvexHull` of the pairwise intersection points returned by `circle_intersections()`. This was inaccurate:
- ConvexHull connects vertices with straight lines; the true boundary follows circle arcs
- ConvexHull forces convexity, losing concave (lens-shaped) regions
- The result overestimates the intersection area visually

### Solution

Each circle is approximated as a 100-point polygon in lat/lon space. We build `shapely.geometry.Polygon` objects and compute their geometric intersection via `functools.reduce(.intersection())`. This naturally traces the correct curvilinear boundary.

### Changes Made (`evaluate_million_scale.py`)

1. **Imports**: Replaced `from scipy.spatial import ConvexHull` with `from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon` and `from functools import reduce`
2. **Added `_circle_to_shapely_polygon()` helper** (line 53): Converts `(lat, lon, radius_km)` to a Shapely polygon
3. **Replaced `compute_intersection_area()`** (line 63): Changed signature from `(points)` to `(circles_data)` — builds Shapely polygons, computes true intersection area via `.area`
4. **Replaced ConvexHull block in `plot_circles_on_map()`** (line 574): Shapely intersection rendering with `MultiPolygon` and `geom_type` handling
5. **Updated call sites**: `run_million_scale_cbg()` (line 170) and `run_vanilla_cbg()` (line 311) now pass `circles_data` built from `circles_out` (the preprocessed/filtered circles)

### Important: Using `circles_out` Not `circles`

`circle_intersections()` internally calls `circle_preprocessing()` (`helpers.py:58-88`) which removes circles that fully contain other circles. The area computation must use `circles_out` (the filtered set) to be consistent with the centroid computation. Using the original unfiltered `circles` would include redundant large circles — which doesn't change the geometric intersection (a circle containing another is redundant in an intersection), but can cause `compute_intersection_area` to succeed for cases where preprocessing reduced circles to <2 (returning area=0 instead).

Verified: Shapely intersection of all circles equals intersection of filtered circles (areas match to <1e-6 deg²).

### Intersection Area CDF

After the fix, the area CDF shows:
- **Vanilla CBG (LP)**: Median 3.8 million km², N=232 (out of 266 probes)
- **Million-Scale CBG (2/3c)**: Median 23.5 million km², N=123 (out of 266 probes)

Million-Scale has far fewer valid intersections (123 vs 232) because its larger radii (median 5000 km) cause more circles to be removed during preprocessing, often leaving <2 circles. When intersections do exist, they are much larger due to the oversized radii.

## Centroid-Outside-Intersection Issue

### Observation

In several map plots (e.g., Vanilla CBG at MS P5 for probe `73.42.215.135`), the estimated location (red X) falls **outside** the yellow intersection region. This is not a bug — it is a known limitation of the arithmetic-mean centroid used by `polygon_centroid()` in `helpers.py:169-179`.

### Root Cause

`polygon_centroid()` computes the arithmetic mean of the intersection boundary vertices:

```python
def polygon_centroid(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return x, y
```

This is the centroid of a **finite set of points**, not the geometric centroid of the filled polygon area. For concave or arc-bounded regions, the arithmetic mean of boundary vertices can land outside the region when vertices are unevenly distributed.

### Worked Example: Probe `73.42.215.135` (Vanilla CBG)

**Setup**: 7 anchors, LP model produces 7 circles. After `circle_preprocessing()`, 4 circles remain:

| VP Location | RTT (ms) | LP Radius (km) |
|---|---|---|
| (47.6, -122.3) Seattle | 7.5 | 674 |
| (42.0, -88.0) Chicago | 55.7 | 3,349 |
| (33.8, -84.4) Atlanta | 65.3 | 3,938 |
| (32.8, -96.8) Dallas | 46.6 | 2,884 |

The 3 removed circles (San Jose r=1930km, LA r=2547km, Miami r=5101km) were fully contained by other circles.

**Pairwise intersection yields 6 boundary vertices** (points where circle arcs cross, filtered to those inside all 4 circles):

| Vertex | Lat | Lon |
|---|---|---|
| P1 | 43.43 | -128.62 |
| P2 | 53.05 | -126.54 |
| P3 | 53.30 | -119.03 |
| P4 | 42.68 | -127.36 |
| P5 | 50.59 | -130.41 |
| P6 | 44.17 | -129.53 |

**Arithmetic mean**:
- lat = (43.43 + 53.05 + 53.30 + 42.68 + 50.59 + 44.17) / 6 = 287.22 / 6 = **47.87**
- lon = (-128.62 + -126.54 + -119.03 + -127.36 + -130.41 + -129.53) / 6 = -761.49 / 6 = **-126.91**

**Result**: Centroid = **(47.87, -126.91)** — this is in the Pacific Ocean, ~400 km offshore from Washington state.

**Why it's outside**: 5 of the 6 vertices have longitude west of -126° (in the ocean), while only P3 (53.30, -119.03) is inland. The mean gets pulled toward the ocean-side cluster. The actual intersection polygon extends from roughly -119° to -130° lon, centered around -124° — but the vertex distribution is skewed westward because the Seattle circle (r=674km, the smallest) creates a tight arc on the east side with few intersection points, while the larger circles produce spread-out intersections on the west side.

**Shapely geometric centroid**: (46.27, -118.78) — inside the polygon, 310 km error vs 344 km for the arithmetic mean.

**True location**: (47.68, -122.31) — Seattle area, inside the intersection region.

### Implications

This is a property of the original Million-Scale paper's algorithm (`helpers.py:169-179`), not a bug in our evaluation code. The Shapely `.centroid` (area-weighted geometric centroid) would always fall inside a connected polygon, but changing the centroid method would alter the replication fidelity.

## Related Tasks

- `tasks/cbg-feasibility/` — Original CBG feasibility exploration (LP bestline development)
- `tasks/rtt-filtering/` — RTT filtering pipeline design
- `tasks/octant-rtt-model/` — Octant dual-bound model
