# EXCLUSIVE_REGION Verification via Shapely Polygon-Disk Intersection

**Date:** 2026-06-24  
**Method:** Reconstruct the actual MTL intersection polygon R from `mtl_participants`, then test `R.intersects(D_truth)` using Shapely — NOT the per-VP individual band heuristic.

---

## Classification Definitions

| Category | Condition |
|----------|-----------|
| **EMPTY_REGION** | `status == 'FALLBACK'` (MTL intersection was empty; benchmark fell back to VP centroid) |
| **EXCLUSIVE_REGION** | `status == 'SUCCESS'`, R reconstructed, and `R.intersects(D_truth_disk) == False` |
| **INCL_success** | `status == 'SUCCESS'`, R ∩ D_truth ≠ ∅, and `error_km ≤ 50 km` (prediction inside truth cluster) |
| **INCL_misclass** | `status == 'SUCCESS'`, R ∩ D_truth ≠ ∅, but `error_km > 50 km` (region overlaps truth but centroid resolves wrong) |

**D_truth** = Shapely polygon approximation of a 50 km disk around the target's cluster centroid, using the planar approximation: `r_lat = 50/111°`, `r_lon = 50/(111·cos(lat))°`, 64-vertex polygon.

---

## Data Sources

| Setup | vanilla_cbg | million_scale_cbg | octant_cbg | spotter_cbg |
|-------|-------------|-------------------|------------|-------------|
| `north_america_as7018_final_us` | `outputs/` (planar_circle) | `outputs/` (planar_circle) | `outputs_partvp/` (planar_annulus_weighted) | `outputs_partvp/` (planar_annulus_weighted) |
| `europe_as3209_final_de` | `outputs/` (planar_circle) | `outputs/` (planar_circle) | `outputs/` (planar_annulus_weighted) | `outputs/` (planar_annulus_weighted) |

**Rationale:**
- `outputs/` has the official benchmark config (planar_circle for vanilla/million_scale) and stores `mtl_participants`.
- For `north_america_as7018_final_us`, octant/spotter in `outputs/` lack `mtl_participants` (26 columns vs 28); `outputs_partvp/` has them.
- `outputs_partvp/` vanilla_cbg/million_scale_cbg use the **old** `spherical_circle` config — those are NOT used here; `outputs/` is used instead.

---

## Reconstruction Methods

### vanilla_cbg / million_scale_cbg (planar_circle MTL)

The `mtl_intersection_kind` column shows `'polygon'` (correct config) or `'vertex_list'` (old spherical_circle config, not used).

Reconstruction: for each VP in `mtl_participants`, create a 64-vertex polygon in degree-space using `_circle_to_planar_polygon(vp_lat, vp_lon, echoed_upper_km, n_pts=64)`, then take the Shapely intersection of all VP disks. `enable_circle_filter=True` pre-removes redundant outer disks before intersection.

### octant_cbg / spotter_cbg (planar_annulus_weighted MTL)

The `mtl_intersection_kind` shows `'polygon'` or `'multipolygon'`.

Reconstruction: re-run `PlanarAnnulusWeightedMTL` with the stored config kwargs, feeding `LTDResult` objects built from `mtl_participants` fields (`vp_lat`, `vp_lon`, `rtt_ms`, `echoed_lower_km`, `echoed_upper_km`). This exactly reproduces the benchmark's joint weighted feasible region.

---

## Results

| Setup | combo | data_src | mtl | EMPTY | EXCLUSIVE | INCL-success | INCL-misclass | total_success | method |
|-------|-------|----------|-----|------:|----------:|-------------:|--------------:|--------------:|--------|
| north_america_as7018_final_us | vanilla_cbg | outputs | planar_circle | 14 | **10** | 18 | 54 | 82 | planar_circle → Shapely Polygon |
| north_america_as7018_final_us | million_scale_cbg | outputs | planar_circle | 0 | **1** | 35 | 60 | 96 | planar_circle → Shapely Polygon |
| north_america_as7018_final_us | octant_cbg | outputs_partvp | planar_annulus_weighted | 0 | **48** | 22 | 26 | 96 | PlanarAnnulusWeightedMTL rerun |
| north_america_as7018_final_us | spotter_cbg | outputs_partvp | planar_annulus_weighted | 0 | **95** | 0 | 1 | 96 | PlanarAnnulusWeightedMTL rerun |
| europe_as3209_final_de | vanilla_cbg | outputs | planar_circle | 36 | **13** | 12 | 35 | 60 | planar_circle → Shapely Polygon |
| europe_as3209_final_de | million_scale_cbg | outputs | planar_circle | 0 | **0** | 39 | 57 | 96 | planar_circle → Shapely Polygon |
| europe_as3209_final_de | octant_cbg | outputs | planar_annulus_weighted | 0 | **50** | 31 | 15 | 96 | PlanarAnnulusWeightedMTL rerun |
| europe_as3209_final_de | spotter_cbg | outputs | planar_annulus_weighted | 0 | **85** | 8 | 3 | 96 | PlanarAnnulusWeightedMTL rerun |

Each setup runs 5 folds × ~19 targets ≈ 96 total targets (exact counts vary by fold size).

---

## Key Findings

### EXCLUSIVE > 0 Confirmed for All Combos Except million_scale EU

The prior per-VP band heuristic (which returned EXCLUSIVE=0) was wrong. The correct joint Shapely test finds meaningful EXCLUSIVE rates:

- **vanilla_cbg**: 10–13 EXCLUSIVE / 82–96 SUCCESS (12–17%). The planar_circle disk intersection places the feasible region outside the 50 km truth zone when VPs are distant and RTT-inflated.
- **million_scale_cbg**: Nearly zero EXCLUSIVE (1 NA, 0 EU) because the speed_of_internet LTD creates very wide disks that almost always encompass the truth — the constraint is weak.
- **octant_cbg**: ~50% EXCLUSIVE in both setups (48/96 NA, 50/96 EU). The annular constraints (lower_km > 0) exclude the true location when RTT inflation pushes inner bounds outward.
- **spotter_cbg**: Near-total EXCLUSIVE — 95/96 in NA, 85/96 in EU. The normal_dist LTD generates very large inner bounds, and the joint weighted intersection almost never overlaps the truth disk.

### Failure Mode Illustrated (spotter NA, target 104.131.160.184)

- **Target**: New York City (lat=40.71, lon=-74.01)
- **Cluster centroid**: (40.71, -74.09) — same metro
- **MTL region R** (MultiPolygon): bounds `(-88.0, 36.0, -85.7, 39.4)` — Tennessee/Mississippi area
- **D_truth bounds**: `(-74.7, 40.3, -73.5, 41.2)` — New York
- `R.intersects(D_truth)` = **False** — EXCLUSIVE_REGION

Each individual VP's annulus individually reaches New York (dist_to_target < upper_km and lower_km < dist for individual VPs). But the **joint intersection** of 44 VP annuli finds a different overlap region ~1,300 km from the truth. This is the joint-intersection failure mode the per-VP heuristic cannot detect.

### MTL Method Comparison for vanilla_cbg (NA setup)

Testing vanilla_cbg with two reconstruction methods on outputs_partvp data:

| Reconstruction | EMPTY | EXCLUSIVE | INCLUSIVE |
|---------------|------:|----------:|----------:|
| `spherical_circle` (old partvp config) | 14 | 48 | 34 |
| `planar_circle` (correct outputs config) | 14 | 10 | 72 |

The spherical_circle method (pairwise great-circle crossing points) produces fewer and sparser vertices; when only 2 points are found (not enough for a polygon), the region is treated as a point set — inflating EXCLUSIVE. The planar_circle Shapely polygon intersection is more robust, reducing EXCLUSIVE by ~4×.

---

## Notes

- `north_america_as7018_final_us / spotter_cbg / fold_3` targets.parquet was transiently corrupted mid-analysis (a background benchmark re-run overwrote it). The final count of EXCLUSIVE=95 was verified after the re-run completed and fold_3 became readable again — the fold-by-fold breakdown confirms: fold_0=15, fold_1=16, fold_2=27, fold_3=14, fold_4=23.
- `INCL_success` uses `error_km ≤ 50 km` as the "correct cluster" proxy. This approximates the actual cluster-label equality check but is slightly different (a prediction at 51 km from truth could be in the right cluster if the cluster radius > 50 km). A stricter cluster-label equality check would give slightly different INCL_success/INCL_misclass split.
- The 50 km D_truth disk uses a flat-earth planar approximation in degree space — accurate to ~1% at mid-latitudes, sufficient for this 50 km radius.
