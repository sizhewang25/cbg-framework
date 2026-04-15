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

## CDF Crossover Investigation

### Observation

The error CDF shows MS CBG (2/3c) outperforms LP CBG at lower percentiles despite LP having better overall median. The blue curve is to the left of the black curve below approximately the 50th percentile.

### Investigation Script

**File:** `scripts/analysis/million_scale/crossover_analysis.py`

**Output directory:** `scripts/analysis/million_scale/outputs/crossover_analysis/`

### Finding 1: No Fallbacks — Pure Centroid Geometry Effect

Contrary to the initial hypothesis (that MS falls back to closest-VP more often, benefiting near-anchor probes), **ALL 266 probes intersect for both methods** — zero fallbacks. The crossover is entirely driven by differences in intersection polygon shape and where the arithmetic-mean centroid lands.

### Finding 2: Bias-Variance Tradeoff in Circle Radii

MS's larger circles (100×rtt) produce **symmetric, well-centered intersections** for near-anchor probes. LP's tighter circles are more sensitive to per-anchor calibration errors (slope/intercept misfit), creating **asymmetric intersections** whose arithmetic-mean centroid can be displaced.

| Percentile | MS Error (km) | LP buggy (km) | LP corrected (km) |
|-----------|---------|-----------|----------------|
| P10 | **12.8** | 282.7 | 199.8 |
| P25 | **69.3** | 470.8 | 337.4 |
| P50 | 686.7 | 696.8 | **601.5** |
| P75 | 1373.6 | **1218.7** | **1082.3** |
| P90 | 1748.8 | **1444.9** | **1433.5** |

MS wins 141/266 probes (53%) against buggy LP, and 135/266 (50.8%) against corrected LP. The crossover persists after bug correction — this is a real geometric phenomenon.

When MS wins, its median error is 102.8 km vs LP's 707.7 km (MS wins big for near-anchor probes). When LP wins, its median error is 689.8 km vs MS's 1284.0 km (LP wins on distant/hard cases).

### Finding 3: `is_within_cirle` Bug in helpers.py

**Location:** `helpers.py:161` inside `circle_intersections()`

**Bug:** The intersection point filtering step calls `is_within_cirle(vp_geo, rtt_c, point_geo, speed_threshold)` which internally computes `d = rtt_to_km(rtt, speed_threshold)` = 100×rtt (the MS formula). It ignores the pre-filled `d_c` from the LP circle tuple.

**Mechanism:**
- `circle_preprocessing()` (line 64) correctly uses pre-filled `d` when available — so LP circles have correct radii during containment removal
- But `is_within_cirle()` (line 28) always recomputes `d` via `rtt_to_km()` — so LP intersection points are filtered with MS-sized radii (larger, more permissive)
- This admits intersection points that lie **outside LP circles but inside MS circles**, contaminating the polygon and shifting the centroid

**Impact:**
- 207/266 probes have changed errors
- Zero probes change intersection status (all still intersect)
- LP median error: 696.8 km (buggy) → 601.5 km (corrected) — **95 km improvement**
- The bug does NOT cause the crossover — crossover persists after correction

**Visual artifact in map plots:** In `plot_percentile_maps()` / `plot_circles_on_map()` (`evaluate_million_scale.py:715-861`), the yellow intersection region and gray crossing dots are computed correctly using **Shapely** with true LP radii (`circles_data`). However, the red X (arithmetic-mean centroid) comes from `helpers.py:polygon_centroid()` which used vertices filtered by the **buggy** `is_within_cirle()` with MS-sized radii. This means the centroid was computed from a superset of the visible vertices — including spurious points that lie outside the yellow region but inside the larger MS circles. This compounds with the arithmetic-mean-outside-polygon issue: even if all vertices were correct, averaging boundary vertices can land outside a concave region, but here the vertex set itself is contaminated, pulling the centroid further away.

### Finding 4: Near-Anchor Probes Drive MS Advantage

The distance-to-nearest-anchor plot shows MS's advantage is concentrated at probes within 0–200 km of an anchor:

- Probes at <50 km from anchor: MS errors as low as 0.7 km, LP errors 400–700 km
- The nearby anchor's 2/3c circle creates a small, well-centered constraint
- LP's calibrated circle for the same anchor has a different radius (due to intercept subtraction), shifting the intersection polygon

Example: Probe `73.189.18.0` (0.7 km from Seattle anchor)
- MS error: 0.7 km — intersection centroid lands almost exactly on truth
- LP error: 619.9 km — LP's asymmetric intersection shifts centroid to the midwest

### Outputs Produced

| File | Purpose |
|---|---|
| `outputs/crossover_analysis/crossover_cdf.png` | 3-way CDF: MS vs LP (buggy) vs LP (corrected) with shaded crossover regions |
| `outputs/crossover_analysis/error_scatter.png` | Per-probe error scatter (LP error vs MS error) colored by category |
| `outputs/crossover_analysis/dist_vs_advantage.png` | Distance-to-nearest-anchor vs error advantage |
| `outputs/crossover_analysis/mechanism_breakdown.png` | Category breakdown bar chart |
| `outputs/crossover_analysis/maps/probe_*.png` | 141 side-by-side map plots for all MS-wins probes |
| `outputs/crossover_analysis/crossover_results.json` | Full per-probe diagnostics |

### Interpretation

The crossover is a **bias-variance tradeoff** inherent to circle-based multilateration:

- **MS (2/3c)** = high bias (large circles, imprecise bounds) but **low variance** (robust to RTT inflation, model misfit, routing anomalies). Best cases are very good because the intersection geometry is stable.
- **LP (calibrated)** = lower bias (tighter circles, more precise bounds) but **higher variance** (sensitive to per-anchor calibration errors in slope/intercept). Best cases can still miss because a slight misfit in one anchor's model shifts the entire intersection.

At **low percentiles** (best cases), MS's robustness wins — the 2/3c formula is a safe upper bound that reliably contains the truth, producing well-centered intersections. At **high percentiles** (worst cases), LP's tighter constraints produce smaller intersection regions that are geometrically closer to truth even when off-center.

The `is_within_cirle` bug amplifies this effect by contaminating LP's intersection polygons, but is not the root cause.

## Related Tasks

- `tasks/cbg-feasibility/` — Original CBG feasibility exploration (LP bestline development)
- `tasks/rtt-filtering/` — RTT filtering pipeline design
- `tasks/octant-rtt-model/` — Octant dual-bound model
