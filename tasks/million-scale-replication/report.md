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

## Related Tasks

- `tasks/cbg-feasibility/` — Original CBG feasibility exploration (LP bestline development)
- `tasks/rtt-filtering/` — RTT filtering pipeline design
- `tasks/octant-rtt-model/` — Octant dual-bound model
