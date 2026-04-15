# CBG Combination Evaluation — Plan

## Background

The CBG geolocation pipeline can be decomposed into four independent phases:

1. **RTT-to-Distance Estimation** — converts measured RTT into a distance bound (radius or annulus)
2. **Filtering** — removes erroneous or redundant constraints before multilateration
3. **Multilateration** — intersects constraints to form a feasible region
4. **Centroid Selection** — collapses the feasible region into a single (lat, lon) estimate

Each phase currently has multiple implementations that are tightly coupled to a specific method (Million-Scale, Vanilla, Octant). We need to decouple them and systematically evaluate all valid combinations.

## Context

### Phase 1: RTT-to-Distance Estimation

| Method | Source | Description |
|--------|--------|-------------|
| `fit_speed_of_internet` | Million-Scale CBG | Theoretical 2/3c model: `radius = 100 × RTT` |
| `fit_low_envelope` | Vanilla CBG | Per-anchor LP bestline inversion via `RTTDistanceModel.fit(method='lp')` |
| `fit_bounded_spline` | Octant | Per-anchor bounded spline + shared delta band via `OctantRTTModel` — produces annuli (inner + outer radius) |

### Phase 2: Filtering

| Method | Source | Description |
|--------|--------|-------------|
| `redundant_circle_removal` | Million-Scale CBG | `circle_preprocessing()` in `helpers.py:59` — pairwise check, removes larger circle when one fully contains another, keeping tightest bound |
| `intersection_weighted_filtering` | Octant | `compute_feasible_region_weighted()` in `octant_geolocation.py:299` — grid-based weight accumulation over annuli, threshold on cumulative weight. Note: this fuses filtering + multilateration into one step |

### Phase 3: Multilateration (Region Formation)

| Method | Source | Description |
|--------|--------|-------------|
| `spherical_intersection` | Million-Scale CBG | `circle_intersections()` in `helpers.py:108` — exact spherical geometry: pairwise great-circle crossing points, filtered to points inside all circles. Returns vertex list |
| `shapely_intersection` | centroid_comparison.py | `_circle_to_shapely_polygon()` + `reduce(intersection)` — approximates circles as 100-point polygons in degree space, computes Shapely polygon intersection. Returns Shapely geometry |
| `unweighted_region` | Octant | `compute_feasible_region_unweighted()` in `octant_geolocation.py:253` — intersect all outer disks, subtract all inner disks. Returns Shapely geometry |
| `weighted_region` | Octant | `compute_feasible_region_weighted()` in `octant_geolocation.py:299` — grid-based weight accumulation (fused with Phase 2 filtering). Returns Shapely geometry |

Note: `weighted_region` is an integrated Phase 2+3 — it does both filtering and region formation in one step.

### Phase 4: Centroid Selection

| Method | Source | Description |
|--------|--------|-------------|
| `arithmetic_mean` | Million-Scale CBG | `polygon_centroid()` in `helpers.py:170` — simple average of intersection vertex coordinates. Operates on vertex list from `spherical_intersection` |
| `geometric_centroid` | centroid_comparison.py | Shapely `.centroid` — area-weighted center of mass of intersection polygon. Operates on Shapely geometry |

Note: Octant's Monte Carlo geometric median is excluded for now (pending validation).

### Phase compatibility constraints

- `arithmetic_mean` requires vertex list → only works after `spherical_intersection`
- `geometric_centroid` requires Shapely geometry → works after `shapely_intersection`, `unweighted_region`, or `weighted_region`
- `spherical_intersection` requires circles (not annuli) → after `redundant_circle_removal` or directly from Phase 1
- `weighted_region` requires annular constraints → only works with `fit_bounded_spline` (Octant)
- `intersection_weighted_filtering` is fused with `weighted_region` (they are the same step)

### Key Source Files

- `scripts/utils/helpers.py` — `circle_preprocessing()`, `circle_intersections()`, `polygon_centroid()`, `get_middle_intersection()`
- `scripts/analysis/million_scale/evaluate_million_scale.py` — `run_million_scale_cbg()`, `run_vanilla_cbg()`, `fit_lp_models()`, `load_data()`
- `scripts/analysis/million_scale/centroid_comparison.py` — `compute_shapely_centroid()`
- `scripts/analysis/octant/octant_geolocation.py` — `form_constraints()`, `compute_feasible_region_weighted()`, `compute_feasible_region_unweighted()`, `_annulus_to_shapely()`
- `scripts/analysis/octant/octant_evaluation.py` — `fit_octant_models()`, `run_octant_cbg()`
- `scripts/analysis/octant/octant_model.py` — `OctantRTTModel`, `find_delta_for_coverage()`
- `scripts/analysis/cbg_feasibility/rtt_model.py` — `RTTDistanceModel`, `fit_bestline_lp()`

### Data

- Input: `datasets/cbg_test/vultr_pings_us_only.csv` (AS7922 subset)
- Existing outputs: `scripts/analysis/million_scale/outputs/comparison/`, `outputs/centroid_comparison/`

## Goals

1. **Decouple** the four phases into atomic, composable functions with clear input/output contracts
2. **Enumerate** all valid phase combinations respecting compatibility constraints
3. **Evaluate** each combination using four visualization types:
   - Error CDF (all lines on one plot)
   - RTT-Error scatterplot (per-probe error distribution)
   - Error-Diff CDF (pairwise improvement between combinations)
   - Percentile maps (p5, p25, p50, p75, p95 for debugging)
4. **Preserve** the original authors' logic — wrap, don't rewrite

## Approach

### Step 1: Define the function contracts

```
Phase 1 output: List[CircleConstraint]
    CircleConstraint = (vp_lat, vp_lon, radius_km, rtt_ms, vp_ip)
    — for Octant: also inner_radius_km (annulus)
    — for MS/Vanilla: inner_radius_km = 0 (full disk)

Phase 2 input:  Phase 1 output
Phase 2 output: filtered list of CircleConstraint (same format, fewer items)

Phase 3 input:  Phase 2 output
Phase 3 output: one of:
    — vertex_list: List[(lat, lon)] from spherical intersection
    — shapely_region: Shapely Polygon/MultiPolygon from polygon intersection
    — weighted_region: Shapely geometry from grid-based weighted filtering

Phase 4 input:  Phase 3 output
Phase 4 output: (estimated_lat, estimated_lon) or None
```

### Step 2: Wrap existing functions

Phase 1 wrappers:
- `rtt_to_circles_speed_of_internet(probe_data, anchor_coords)` → circles
- `rtt_to_circles_low_envelope(probe_data, lp_models)` → circles
- `rtt_to_circles_bounded_spline(probe_data, octant_models, delta)` → annuli

Phase 2 wrappers:
- `filter_redundant_circles(circles)` → wraps `circle_preprocessing()`
- `filter_none(circles)` → passthrough (no filtering)

Phase 3 wrappers:
- `multilaterate_spherical(circles)` → wraps `circle_intersections()` → vertex list
- `multilaterate_shapely(circles)` → Shapely polygon intersection → Shapely geometry
- `multilaterate_weighted_grid(constraints)` → wraps `compute_feasible_region_weighted()` → Shapely geometry (fused Phase 2+3)

Phase 4 wrappers:
- `centroid_arithmetic_mean(vertex_list | shapely_region)` → wraps `polygon_centroid()` / `get_middle_intersection()`
- `centroid_geometric_weighted(shapely_region)` → wraps Shapely `.centroid`

### Step 3: Enumerate valid combinations

Compatibility rules:
- `arithmetic_mean` needs vertex list → requires `spherical` multilateration
- `geometric_centroid` needs Shapely geometry → requires `shapely` or `weighted_grid` multilateration
- `weighted_grid` requires annuli → requires `bounded_spline` Phase 1
- `weighted_grid` is fused Phase 2+3 → Phase 2 is implicitly `intersection_weighted_filtering`

**Path A: spherical multilateration → arithmetic_mean**

| # | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---------|---------|---------|---------|
| A1 | speed_of_internet | circle_removal | spherical | arithmetic_mean |
| A2 | low_envelope | circle_removal | spherical | arithmetic_mean |
| A3 | bounded_spline | circle_removal | spherical | arithmetic_mean |

**Path B: shapely multilateration → geometric_centroid**

| # | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---------|---------|---------|---------|
| B1 | speed_of_internet | circle_removal | shapely | geometric_centroid |
| B2 | low_envelope | circle_removal | shapely | geometric_centroid |
| B3 | bounded_spline | circle_removal | shapely | geometric_centroid |

**Path C: weighted grid (fused Phase 2+3) → geometric_centroid**

| # | Phase 1 | Phase 2+3 | Phase 4 |
|---|---------|-----------|---------|
| C1 | bounded_spline | weighted_grid | geometric_centroid |

**Path D: shapely multilateration → arithmetic_mean (cross-compatibility)**

| # | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---------|---------|---------|---------|
| D1 | speed_of_internet | circle_removal | shapely | arithmetic_mean |
| D2 | low_envelope | circle_removal | shapely | arithmetic_mean |
| D3 | bounded_spline | circle_removal | shapely | arithmetic_mean |

Note: Path D uses `arithmetic_mean` on Shapely geometry by extracting vertices from the polygon boundary. This tests whether the multilateration method matters when centroid method is held constant.

**Total: 10 combinations** (or 7 core + 3 cross-compatibility)

### Current ID mapping (after reordering)

| Path | Multilateration | Centroid | Combos |
|------|----------------|----------|--------|
| A | spherical | arithmetic_mean | A1 (SoI), A2 (LP), A3 (Spline) |
| B | spherical | geometric_centroid | B1, B2, B3 |
| C | shapely | arithmetic_mean | C1, C2, C3 |
| D | shapely | geometric_centroid | D1, D2, D3 |

**12 combinations total** across 3 distance models × 4 multilateration+centroid paths.

### Missing Octant-specific components (Phase 5 integration)

Two Octant features are NOT yet in the framework:

**1. Unweighted annulus intersection (multilateration)**
- Source: `octant_geolocation.py:253` → `compute_feasible_region_unweighted()`
- Logic: `intersection(all outer disks) - union(all inner disks)`
- Different from existing `shapely` multilateration which uses `radius_km` (outer) only and ignores `inner_radius_km`
- When `inner_radius_km > 0` (Octant spline), this subtracts the inner disks from the region — a fundamentally different feasible region shape
- Only meaningful with `bounded_spline` distance (which produces annuli)
- Framework target: new `"unweighted_annulus"` multilateration variant

**2. Monte Carlo geometric median (centroid)**
- Source: `octant_geolocation.py:402-477` → `sample_points_in_region()` + `geometric_median_approx()`
- Logic: Sobol QMC rejection sampling inside the Shapely region → `geom_median.numpy.compute_geometric_median()`
- Different from both `arithmetic_mean` (vertex average) and `geometric_centroid` (Shapely `.centroid` area-weighted center of mass)
- Geometric median minimizes sum of Euclidean distances to all sampled points — more robust to outlier boundary shapes
- Depends on `geom-median` package (already installed)
- Framework target: new `"monte_carlo_median"` centroid variant

### Integration plan

**Step 1: Add `unweighted_annulus` multilateration** (`scripts/framework/multilateration/unweighted_annulus.py`)
- Wraps `compute_feasible_region_unweighted()` from `octant_geolocation.py`
- Converts `CircleConstraint` → `AnnularConstraint` (same pattern as `weighted_grid.py`)
- Returns `MultilatResult(region=shapely_geometry)`

**Step 2: Add `monte_carlo_median` centroid** (`scripts/framework/centroid/monte_carlo_median.py`)
- Wraps `sample_points_in_region()` + `geometric_median_approx()` from `octant_geolocation.py`
- Input: `MultilatResult` with `.region` (Shapely geometry) — NOT compatible with vertex-only results
- Parameters: `n_samples=5000`, `rng` seed for reproducibility
- Returns `(lat, lon)` or `None`

**Step 3: Add new combinations to `combinations.py`**
- New paths E and F (unweighted_annulus with arith/geom/median centroids)
- New centroid column for monte_carlo_median on existing Shapely paths
- Exact combo list TBD after Step 1-2 are working

**Step 4: Validate**
- Compare `unweighted_annulus + geometric_centroid` against `octant_geolocation.py` standalone
- Compare `monte_carlo_median` against existing centroids on same regions

### Step 4: Build the evaluation harness

Single script that:
1. Loads data, fits all models (reuse existing functions)
2. Runs all combinations
3. Generates all 4 plot types
4. Outputs JSON results

## Caveats

1. **Preserve original logic**: The Million-Scale `circle_intersections()` pipeline is the original authors' code. Wrap it, don't rewrite it. The spherical intersection math must remain unchanged.
2. **Tight coupling**: Current `run_million_scale_cbg()` and `run_vanilla_cbg()` mix all 4 phases inline. Decoupling requires careful extraction to avoid behavior changes.
3. **Naming clarity**: Some existing function names are misleading (e.g., `circle_preprocessing` is really redundant-circle removal). Use wrapper functions with clear names.
4. **Octant Monte Carlo excluded**: `geometric_median_approx()` via `geom_median` is excluded until validated. The geometric centroid (Shapely `.centroid`) is the alternative.
5. **Weighted grid + non-Octant**: `weighted_grid` requires annular constraints (inner + outer radius). Only valid with `bounded_spline`. For MS/Vanilla (disk constraints with inner=0), the grid degenerates to unweighted intersection — not meaningful to test.
6. **Fallback behavior**: When multilateration fails (no intersection), current code falls back to closest-VP by min RTT. This fallback must be preserved consistently across all combinations.
7. **Arithmetic mean on Shapely geometry (Path D)**: To apply `arithmetic_mean` after Shapely multilateration, we extract boundary vertices from the Shapely polygon. This is conceptually different from the spherical intersection vertices — the vertex count and distribution differ. Flag this clearly in results.
