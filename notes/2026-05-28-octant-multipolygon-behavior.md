# MultiPolygon MTL results — how the pipeline picks a component

**Date:** 2026-05-28
**Trigger:** Inspection of `north_america_as7018 · octant_cbg · fold_0` while diagnosing why the user's `23.130.139.89` visualization showed only one feasible region despite two visually overlapping ring clusters.
**Related memory:** [finding_octant_na_collapse_as7018](../../.claude/projects/-home-nuwinslab-workspace-atnt-cbg-framework/memory/finding_octant_na_collapse_as7018.md)

---

## TL;DR

When [PlanarAnnulusMTL](../scripts/framework/v2/mtl/planar_annulus.py) (and its weighted sibling) returns a `MultiPolygon`, no stage in the pipeline picks a component explicitly. The whole geometry propagates through. The CTR stage — for `monte_carlo_medoid` — Sobol-rejects samples inside the union; component sample share is proportional to component area; the haversine-min-sum medoid therefore lands in the **largest-area component**. There is no preference for the component closest to a VP, no weighting by which VPs each component satisfies, no tie-break against the truth.

---

## Pipeline reminders

1. **MTL solve** ([planar_annulus.py:46-62](../scripts/framework/v2/mtl/planar_annulus.py#L46-L62) → [octant_geolocation.py:255-298](../scripts/libs/octant/octant_geolocation.py#L255-L298))
   `region = (⋂ outer_disks) − (⋃ inner_disks)`. The `.difference(...)` is what produces multi-component results: inner disks bite chunks out of an otherwise connected outer-intersection.

2. **Wrap** ([_annulus_common.py:41-51](../scripts/framework/v2/mtl/_annulus_common.py#L41-L51))
   Accepts both `Polygon` and `MultiPolygon`; stuffs the whole geometry into `MTLResult.intersection`. No component dropped.

3. **CTR — Monte Carlo medoid** ([monte_carlo_medoid.py:44-49](../scripts/framework/v2/ctr/monte_carlo_medoid.py#L44-L49) → [geometry.py:195-239](../scripts/framework/geometry.py#L195-L239))
   `sample_points_in_region` Sobol-samples in the bbox of the union and keeps `region.contains(p)` hits. Sample share per component ∝ component area. [`sampled_medoid`](../scripts/framework/geometry.py#L130-L150) picks the haversine-min-sum sample → necessarily inside the most-sampled component.

Result: bigger blob wins. There is no mechanism to score components by VP coverage, satisfied-constraint count, or proximity to the shortest-ping VP.

---

## Empirical evidence — fold_0 / `north_america_as7018` / `octant_cbg`

`mtl_intersection_kind` from `targets.parquet`:

| kind | count |
|---|---:|
| polygon | 122 |
| none (EMPTY_REGION) | 14 |
| multipolygon | 7 |

The 7 MultiPolygon cases:

| target IP | truth (lat, lon) | err km | comps | comp areas (deg²) | pred in idx | truth in idx |
|---|---|---:|---:|---|:---:|:---:|
| `103.77.105.3` (Jakarta) | (−6.21, 106.82) | 3223 | 2 | **16139.9**, 39.6 | 0 (largest) | none |
| `138.255.248.5` (Dom. Rep.) | (19.39, −70.52) | 1314 | 2 | 6.8, **9.8** | 1 (largest) | none |
| `153.92.43.250` (London) | (51.24, −0.17) | 2137 | 3 | **3226.9**, 9.5, 0.01 | 0 (largest) | none |
| `178.250.7.79` (Paris) | (48.86, 2.39) | 2658 | 2 | **3571.8**, 0.04 | 0 (largest) | none |
| `185.119.168.5` (Toulouse) | (43.62, 1.42) | 1958 | 2 | **3210.6**, 2.1 | 0 (largest) | none |
| `192.102.254.220` (Vancouver) | (49.28, −123.11) | 472 | 2 | 19.5, **27.2** | 1 (largest) | none |
| `193.138.218.50` (Malmö) | (55.60, 13.04) | 2114 | 2 | **3233.6**, 10.4 | 0 (largest) | none |

Two observations are perfectly consistent across all 7:

1. **The prediction always sits in the largest-area component**, not the first one. For `138.255.248.5` and `192.102.254.220` the largest is `comp[1]`; the medoid follows area, not index.
2. **The truth is in no component.** Inner-disk false-exclusion is the underlying constraint failure — same mechanism as the AS7018 NA collapse note. The MultiPolygon shape is downstream of that failure: outer disks intersect to a wide mid-Atlantic crescent, inner disks chop it into a main blob plus tiny secondary slivers (often on the far side of the antimeridian).

---

## Caveats / things this does *not* claim

- This is only the `octant_cbg` (unweighted annular) wrapper. `octant_weighted_cbg` solves a different problem (face decomposition over annulus arrangement) and its multi-component behavior should be re-examined separately.
- Only `monte_carlo_medoid` CTR was traced here. Other CTR methods ([geometric_centroid](../scripts/framework/v2/ctr/geometric_centroid.py), [geometric_median](../scripts/framework/v2/ctr/geometric_median.py), [boundary_vertex_mean](../scripts/framework/v2/ctr/boundary_vertex_mean.py)) reduce a MultiPolygon by Shapely's own semantics (area-weighted centroid for `.centroid`, etc.), which can land in the gap between components — a separate failure mode not characterized here.
- "Largest area" is computed in planar degree² for these notes (haversine area returns NaN for the wraparound blobs); for non-wraparound comparisons the ranking is unchanged.

---

## How to reproduce

```python
import json
from pathlib import Path
from shapely.geometry import Polygon, Point

base = Path('scripts/visualization/benchmark/v2/outputs/north_america_as7018/static/octant_cbg')
# Region JSONs are emitted by mtl_world_map.py and use lat-lon rings.
d = json.loads((base / 'fold_0__103.77.105.3.json').read_text())
for i, r in enumerate(d['rings']):
    outer = [(lon, lat) for lat, lon in r['outer']]
    p = Polygon(outer)
    print(i, p.area, p.representative_point().wkt, p.contains(Point(106.82, -6.21)))
```

Targets.parquet has `mtl_intersection_kind` per row — filter `.str.lower() == 'multipolygon'` to enumerate cases.
