# Octant Geolocation Visualization — Report

**Status**: In Progress
**Created**: 2026-04-15
**Last Updated**: 2026-04-15

## Summary

Created `scripts/analysis/octant/visualize_octant_geolocation.py` that visualizes the full Octant geolocation pipeline on real Vultr measurement data. The script loads data, fits per-anchor Octant RTT-distance models, randomly selects targets, runs geolocation, and renders results on Cartopy US maps.

## Findings

### Sample run (ASN 7922, seed=42, 3 targets)

| Probe IP | Error (km) | Method | Constraints | Area (km2) |
|---|---|---|---|---|
| 73.222.219.183 | 261 | weighted | 7 | 539,206 |
| 24.62.170.218 | 883 | weighted | 7 | 2,422,451 |
| 76.118.144.201 | 955 | weighted | 7 | 3,027,545 |

Median error: 883 km, Mean error: 700 km.

### Key observation: Weighted region vs strict intersection

The weighted feasible region does NOT strictly fall within all annuli intersections. This is by design:

- `compute_feasible_region_weighted()` lays a 0.25° grid over the bounding box
- For each grid cell, accumulates `exp(-rtt/tau)` weights from constraints whose annulus contains the point
- Keeps cells where accumulated weight >= 50% of max possible weight
- Result: far-away landmarks (high RTT → low weight) don't veto grid cells, so the region extends beyond strict intersection
- This is the actual region used for MC sampling and geometric median estimation

This differs from `compute_feasible_region_unweighted()` which computes the strict geometric intersection: `intersection(all outer disks) - union(all inner disks)`.

## Conclusions

<To be filled on task completion.>
