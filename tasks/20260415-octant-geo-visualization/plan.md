# Octant Geolocation Visualization — Plan

## Background

We need a visualization script that shows the full Octant geolocation pipeline on real measurement data: annular constraints from RTT-distance models, the weighted feasible region used for Monte Carlo sampling, and the single-point geometric median estimate — all rendered on a Cartopy US map.

## Context

- **Data source**: `datasets/cbg_test/vultr_pings_us_only.csv` — Vultr US pings, 266 probes (targets), 7 anchors (landmarks) for ASN 7922
- **Octant pipeline**: RTT measurements → per-anchor OctantRTTModel (spline + hull bounds + delta) → AnnularConstraint per landmark → weighted feasible region → MC sampling → geometric median
- **Key algorithm detail**: The weighted feasible region (`compute_feasible_region_weighted`) is NOT the strict geometric intersection of all annuli. It uses a grid-based approach where each grid cell accumulates weights from constraints whose annulus contains it. Cells with weight >= 50% of max possible weight are included. This means the region can extend beyond the strict intersection — far-away landmarks with low weight (high RTT → low `exp(-rtt/tau)`) don't veto grid cells
- **Script location**: `scripts/analysis/octant/visualize_octant_geolocation.py`

## Goals

- Randomly select N targets from real data, run the full Octant pipeline, and produce per-target map figures + a grid summary
- Each map shows: colored annulus rings per landmark, weighted feasible region (yellow/orange), geometric median (blue diamond), true location (red star)
- Maps use Cartopy LambertConformal projection with full US extent for consistent context

## Approach

1. Load data and fit Octant models (reuse `fit_octant_models` from `octant_evaluation.py`)
2. For each target, run geolocation retaining all intermediates (constraints, region, sampled points)
3. Plot on Cartopy GeoAxes with map features (land, ocean, state borders, coastlines)
4. Output individual PDFs + grid summary

## Caveats

- The weighted feasible region intentionally extends beyond strict annulus intersection — this is by design, not a bug
- Grid resolution (0.25°) causes jagged/pixelated region boundaries
- MC scatter plotting is temporarily commented out for visual clarity; can be re-enabled
