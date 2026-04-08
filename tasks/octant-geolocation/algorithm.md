# Octant Geolocation Algorithm

End-to-end description of the current geolocation stage implemented in:

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py)
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py)

This document describes the code as it exists now, not the full original
Octant paper system.

---

## Inputs

The geolocation stage assumes the RTT calibration layer already exists.

Required inputs:

- `models`: `{landmark_ip: fitted OctantRTTModel}`
- `landmark_coords`: `{landmark_ip: (lat, lon)}`
- `rtt_measurements`: `{landmark_ip: min_rtt_ms}` for one target
- optional shared `delta` for spline-band bounds

At the million-scale evaluation layer, the `delta` is computed once after model
fitting with target coverage `0.80`.

---

## Phase 1 - Constraint Formation

Each landmark RTT becomes an annular constraint:

```
inner_km, outer_km = model.predict_distance_bounds(rtt_ms, delta=delta)
weight = exp(-rtt_ms / 50.0)
```

The resulting constraint is:

```
AnnularConstraint(
    landmark_lat,
    landmark_lon,
    landmark_ip,
    rtt_ms,
    inner_radius_km,
    outer_radius_km,
    weight,
)
```

### Bound Semantics

For one landmark `L`, the target must satisfy:

```
inner_radius_km <= d(target, L) <= outer_radius_km
```

So each landmark defines an annulus rather than a single circle.

### Where the Bounds Come From

`OctantRTTModel.predict_distance_bounds(rtt, delta)` behaves as follows:

- Without `delta`, it returns convex-hull bounds `(r_L(rtt), R_L(rtt))`
- With `delta`, it returns a multiplicative spline band:
  ```
  predicted = spline(rtt)
  inner = predicted / delta
  outer = predicted * delta
  ```
- The spline band is clamped so it never goes below the lower hull or above the
  upper hull
- For RTTs beyond the model's high cutoff, the spline is treated as unreliable
  and the model falls back directly to hull bounds

### Filtering Rules

Current `form_constraints(...)` behavior:

- keep only landmarks that have a fitted model
- keep only landmarks with known coordinates
- skip degenerate constraints where `outer_radius_km <= inner_radius_km`
- do not apply a max-RTT filter anymore

Constraints are sorted by weight descending, so lower RTT landmarks come first.

---

## Phase 2 - Weighted Feasible Region

The primary geolocation path is `estimate_location(method='weighted')`.

Instead of intersecting all annuli as hard constraints immediately, the code
builds a weighted region on a latitude/longitude grid.

### Step 2.1 - Bounding Box

For each landmark, compute the outer-circle bounding box in degrees:

```
r_lat = outer_radius_km / 111.0
r_lon = outer_radius_km / (111.0 * cos(lat))
```

Then intersect the bounding boxes of all outer disks to get a tighter search
window.

If the bounding boxes already do not overlap, the weighted region is empty.

### Step 2.2 - Grid Sampling

Create a regular grid:

```
lat_grid = arange(min_lat, max_lat, grid_resolution_deg)
lon_grid = arange(min_lon, max_lon, grid_resolution_deg)
```

Each grid point is tested against every annulus.

### Step 2.3 - Weight Accumulation

For each grid point `p`, compute:

```
score(p) = sum(weight_j for all constraints j whose annulus contains p)
```

where annulus membership is:

```
inner_j <= haversine(p, landmark_j) <= outer_j
```

### Step 2.4 - Thresholding

Let:

```
max_weight = sum(weight_j for all constraints j)
threshold = weight_threshold * max_weight
```

Keep grid cells whose accumulated score satisfies:

```
score(p) >= threshold
```

The surviving cells are converted to little square polygons and unioned into a
single Shapely geometry.

This is the current repo's approximation to Octant's weighted sub-region logic.

### Weighted Fallback

If the weighted region is empty, retry once with half the threshold:

```
weight_threshold / 2.0
```

If that succeeds, the returned method label is:

```
weighted_low_threshold
```

---

## Phase 3 - Unweighted Region Fallback

If the weighted region is still empty, the code falls back to the hard
geometric construction:

```
intersection(all outer disks) - union(all inner disks)
```

This is implemented with Shapely polygon approximations of circles:

- outer disks are intersected
- inner disks are unioned
- the union of inner disks is subtracted from the outer intersection

The result may be:

- a single polygon
- a multipolygon
- empty

If this path succeeds after weighted mode failed, the returned method label is:

```
unweighted
```

---

## Phase 4 - Point Selection Inside the Region

If a non-empty feasible region exists, the algorithm produces a single point
estimate from that region.

### Step 4.1 - Monte Carlo Sampling

Uniformly sample random points inside the region via rejection sampling over the
region bounding box:

```
while collected < n_samples:
    draw random lon/lat in region bounds
    keep point if region.contains(point)
```

The output is an array of sampled `(lat, lon)` points.

### Step 4.2 - Approximate Geometric Median

If at least two sample points were collected, compute the point whose total
distance to all other sampled points is minimal:

```
argmin_i sum_j haversine(point_i, point_j)
```

This is an approximate geometric median of the feasible region.

### Small-Region Fallback

If the feasible region is too small to sample at least two points, use the
Shapely centroid of the region instead.

---

## Phase 5 - Final Fallback

If no weighted or unweighted region survives, return an inverse-RTT weighted
centroid of the landmarks:

```
lat = sum(lat_j * weight_j) / sum(weight_j)
lon = sum(lon_j * weight_j) / sum(weight_j)
```

If all weights were zero, it falls back to an equally weighted landmark mean.

The returned method label in this case is:

```
centroid_fallback
```

---

## Shared Estimator API

`estimate_location(...)` returns:

```
{
    'lat': ...,
    'lon': ...,
    'region_area_km2': ...,
    'n_constraints': ...,
    'method': ...,
    'fallback': ...,
    'n_samples': ...   # only when a region-based estimate was used
}
```

The fallback chain is:

1. weighted region at threshold
2. weighted region at threshold / 2
3. unweighted region
4. inverse-RTT weighted landmark centroid

---

## Million-Scale Evaluator Settings

The million-scale Octant evaluation currently does:

```
estimate_location(
    constraints,
    method='weighted',
    n_samples=5000,
    weight_threshold=0.5,
    grid_resolution_deg=0.25,
    n_pts=128,
)
```

Additional evaluator behavior:

- `fit_octant_models(..., target_coverage=0.80)`
- collect all RTTs from anchors with fitted Octant models
- form constraints without a max-RTT filter
- store mean outer radius as `avg_radius_km`
- store region area as `intersection_area_km2`
- record `geolocation_method` and whether fallback occurred

---

## Differences From The Paper

The current implementation is close in spirit, but not identical to the full
paper:

1. Weighted region construction is grid-based rather than exact weighted
   sub-region boolean decomposition.
2. The evaluator uses one shared `delta` found from aggregated calibration data.
3. The current pipeline does not iterate `delta` per target until the region
   reaches a target area.
4. Geographic priors such as oceans, ZIP codes, and population maps are not
   integrated into the current geolocation stage.

---

## Related Files

- [octant_geolocation.py](../../scripts/analysis/octant/octant_geolocation.py) - shared geolocation pipeline
- [evaluate_million_scale.py](../../scripts/analysis/million_scale/evaluate_million_scale.py) - million-scale Octant evaluation
- [test_octant_geolocation.py](../../scripts/analysis/octant/test_octant_geolocation.py) - geolocation tests
- [README.md](README.md) - task overview
