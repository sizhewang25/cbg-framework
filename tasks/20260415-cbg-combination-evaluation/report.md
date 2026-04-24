# CBG Combination Evaluation — Report

**Status**: In Progress
**Created**: 2026-04-15
**Last Updated**: 2026-04-24

## Summary

Systematic evaluation of CBG geolocation pipeline combinations across four phases:
- 3 distance models × 4 multilateration/centroid paths = 12 combinations (A1-D3)
- Next: integrate Octant-specific multilateration (unweighted annulus) and centroid (Monte Carlo geometric median)

## Findings

### 2026-04-15 — Framework Implementation Complete

Implemented HF-style modular framework under `scripts/framework/` (commit `faedb9f`).

**Architecture**: Base class + registry + per-variant files + `CBGPipeline` with `from_config()` factory.

**Components registered** (verified via import):
- Distance: `speed_of_internet`, `low_envelope`, `bounded_spline`
- Filtering: `redundant_circle`, `none`
- Multilateration: `spherical`, `shapely`, `weighted_grid`
- Centroid: `arithmetic_mean`, `geometric_centroid`

**Key design decisions**:
- `CircleConstraint` dataclass with `to_legacy_tuple()` bridges new types to `helpers.py` functions
- `MultilatResult` carries either `vertices` (spherical) or `region` (Shapely geometry)
- Both centroid methods handle both input types for maximum composability
- `weighted_grid` requires `bounded_spline` — validated in `from_config()`
- Deferred imports (try/except) for optional heavy dependencies (Octant, LP models)

**Files**: 18 source files + 4 task files, 1,362 insertions total.

### 2026-04-15 — Evaluation Harness Complete

Implemented evaluation harness under `scripts/analysis/cbg_evaluation/` (commit `855f45c`).

**Scope change**: Removed C1 (weighted_grid) per user request. Renamed D-path to C-path (shapely+arith). Now 9 combinations across 3 paths.

**Results** (266 probes, AS7922, 7 anchors):

| Rank | Combo | Label | Median Error (km) | Within 100km | Within 500km |
|------|-------|-------|-------------------|-------------|-------------|
| 1 | C1 | SoI + Shapely + Arith | 333 | 30.8% | 59.0% |
| 2 | A3 | Spline + Spherical + Arith | 337 | 28.2% | 57.1% |
| 3 | B1 | SoI + Shapely + Geom | 395 | 31.6% | 56.0% |
| 4 | C3 | Spline + Shapely + Arith | 447 | 19.5% | 53.0% |
| 5 | B3 | Spline + Shapely + Geom | 465 | 21.4% | 54.9% |
| 6 | B2 | LP + Shapely + Geom | 494 | 7.9% | 50.4% |
| 7 | C2 | LP + Shapely + Arith | 541 | 4.9% | 45.5% |
| 8 | A2 | LP + Spherical + Arith | 602 | 4.5% | 40.2% |
| 9 | A1 | SoI + Spherical + Arith | 687 | 28.6% | 45.1% |

**Key observations**:
- Shapely multilateration + arithmetic centroid (Path C) consistently outperforms spherical path (Path A) for same distance model
- Bounded spline (A3) is the best performer on the spherical path (337 km vs A1's 687 km)
- LP lower envelope (A2, B2, C2) consistently ranks lower than SoI and Spline variants
- SoI has highest within-100km rate despite high median — bimodal: very accurate or very wrong

**Outputs**: error_cdf_all.png, error_diff_cdf.png, rtt_error_scatter.png, 15 percentile maps, evaluation_summary.json

### 2026-04-24 — Multilateration Technique Comparison: Spherical vs Shapely vs Unweighted Annulus

Confirmed via code inspection (`scripts/framework/multilateration/`):

| Aspect | Spherical | Shapely | Unweighted Annulus |
|--------|-----------|---------|-------------------|
| Geometry | Exact great-circle on sphere | Planar degree-space approximation | Planar degree-space (Octant) |
| Output | Vertex point set | Shapely Polygon/MultiPolygon | Shapely Polygon/MultiPolygon |
| Inner radius (annulus) | Ignored — full disks only | Ignored — full disks only | Used — true annular subtraction |
| Algorithm | Pairwise crossing points → filter inside all circles | `reduce(intersection)` of 100-pt polygons | `∩(outer disks) − ∪(inner disks)` |
| Origin | IMC 2012 CBG | Octant-style polygon intersection | Octant (NSDI 2007) |

**Spherical** (`helpers.py:circle_intersections`): converts VP coordinates to 3D Cartesian, computes exact pairwise great-circle arc crossing points, keeps only points inside all circles. Returns a vertex list — no polygon area.

**Shapely**: approximates each RTT circle as a 100-point polygon in degree space (with latitude compression correction: `km_per_deg_lon = 111 × cos(lat)`), then sequentially intersects all polygons. Returns an explicit Shapely area geometry.

**Unweighted Annulus**: same polygon intersection as Shapely but uses the full annular constraint — intersects outer disks then *subtracts* inner disks. Only meaningful with `bounded_spline` distance model which produces non-zero `inner_radius_km`. With disk-only constraints it degenerates to Shapely.

**Key implication**: Shapely and Spherical both discard `inner_radius_km`, so pairing them with `bounded_spline` wastes the annular constraint. Only `unweighted_annulus` (and `weighted_grid`) exploit the full Octant spline output.

### 2026-04-20 — MC Median vs Other Centroid Methods

Compared three single-point estimation methods using their best-performing pipeline configuration (266 probes, AS7922, 7 anchors):

| Method | Best Config | Median Error | Within 100km | Within 500km | Within 1000km | Runtime (266 probes) |
|--------|------------|:------------:|:------------:|:------------:|:-------------:|:--------------------:|
| **MC Median** | G3 (Spline + Annulus) | **312.4 km** | 20.7% | **77.4%** | **94.0%** | ~27s |
| **Geometric Centroid** | F3 (Spline + Annulus) | 328.0 km | 21.4% | 74.4% | 94.0% | ~0.21s |
| **Arithmetic Mean** | A3 (Spline + Spherical) | 336.8 km | 28.2% | 57.1% | 86.8% | ~0.18s |

**Key takeaways**:
- MC Median achieves the best median error (312 km, ~5% better than geometric centroid, ~7% better than arithmetic mean) but is ~130x slower due to Sobol quasi-random sampling (1000 points) + geometric median optimization per target
- Geometric Centroid (area-weighted via Shapely) provides the best accuracy/speed trade-off: nearly identical to MC Median at the 1000 km threshold (both 94%) for ~1/130th the compute cost
- Arithmetic Mean is fastest but degrades significantly at the 500 km threshold (57% vs 74-77%) — likely because it does not account for region shape/area
- MC Median is only worth the cost in offline/batch settings where the ~5% median error improvement justifies the runtime penalty
- All three methods benefit most from the Spline distance model + Annulus multilateration; the centroid method choice is secondary to the upstream pipeline

## Conclusions

*Final assessment when task completes.*
