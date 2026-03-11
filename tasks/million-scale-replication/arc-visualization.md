# Task: Arc-Based Intersection Region Visualization

## Context
The intersection region in `plot_circles_on_map()` is currently drawn using `ConvexHull` of the intersection points returned by `circle_intersections()`. This is inaccurate because:
- ConvexHull connects vertices with **straight lines**, but the true boundary follows **circle arcs**
- ConvexHull forces **convexity**, losing concave (lens-shaped) regions
- The result **overestimates** the intersection area visually

## Approach: Shapely Polygon Intersection
Each circle is already approximated as a 100-point polygon in lat/lon space for drawing. Build `shapely.geometry.Polygon` from the same points, compute their geometric intersection via `functools.reduce(.intersection())`, and plot the resulting polygon. This naturally traces the correct curvilinear boundary.

Shapely is already available (dependency of geopandas in pyproject.toml).

## File to Modify
`scripts/analysis/million_scale/evaluate_million_scale.py`

## Changes

### 1. Add imports (top of file)
- `from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon`
- `from functools import reduce`
- Remove `from scipy.spatial import ConvexHull` (no longer needed after all changes)

### 2. Add helper `_circle_to_shapely_polygon()` (near `compute_intersection_area`)
```python
def _circle_to_shapely_polygon(clat, clon, radius_km, n_pts=100):
    angles = np.linspace(0, 2 * np.pi, n_pts)
    r_deg_lat = radius_km / 111.0
    r_deg_lon = radius_km / (111.0 * math.cos(math.radians(clat)))
    lons = clon + r_deg_lon * np.cos(angles)
    lats = clat + r_deg_lat * np.sin(angles)
    return ShapelyPolygon(zip(lons, lats))  # Shapely uses (x,y) = (lon,lat)
```

### 3. Replace ConvexHull block in `plot_circles_on_map()` (lines 502-514)
Replace with:
- Build Shapely polygons from `circles_data` using the helper
- `reduce(lambda a, b: a.intersection(b), shapely_circles)` for the true intersection
- Handle `MultiPolygon` results by iterating `.geoms`
- Filter by `geom_type == 'Polygon'` to skip degenerate line/point results
- Plot `poly.exterior.xy` (returns lon, lat) with same yellow fill + orange outline

```python
if len(circles_data) >= 2:
    try:
        shapely_circles = [
            _circle_to_shapely_polygon(clat, clon, radius_km)
            for clat, clon, radius_km in circles_data
        ]
        shapely_circles = [p for p in shapely_circles if p.is_valid and not p.is_empty]
        if shapely_circles:
            intersection_poly = reduce(lambda a, b: a.intersection(b), shapely_circles)
            if not intersection_poly.is_empty:
                polys = (list(intersection_poly.geoms)
                         if isinstance(intersection_poly, MultiPolygon)
                         else [intersection_poly])
                for k, poly in enumerate(polys):
                    if poly.is_empty or poly.geom_type != 'Polygon':
                        continue
                    xs, ys = poly.exterior.xy  # xs=lons, ys=lats
                    label = 'Intersection region' if k == 0 else None
                    ax.fill(list(xs), list(ys), color='yellow', alpha=0.4,
                            transform=ccrs.PlateCarree(), zorder=3, label=label)
                    ax.plot(list(xs), list(ys), color='orange', linewidth=2,
                            transform=ccrs.PlateCarree(), zorder=4)
    except Exception:
        pass
```

### 4. Update `compute_intersection_area()` to use Shapely
Change signature from `compute_intersection_area(points)` to `compute_intersection_area(circles_data)`:
- Build Shapely polygons, compute intersection, use `.area` for deg^2
- Same deg^2 to km^2 conversion: `area_deg2 * 111.0 * (111.0 * cos(mid_lat))`
- Use `intersection_poly.bounds` to get mid-latitude

### 5. Update call sites for `compute_intersection_area`
- `run_million_scale_cbg()`: build `circles_data_for_area = [(lat, lon, 100.0*rtt) for lat, lon, rtt, _, _ in circles]`, pass instead of `intersections`
- `run_vanilla_cbg()`: build `circles_data_for_area = [(lat, lon, d) for lat, lon, _, d, _ in circles]`, pass instead of `intersections`

## Edge Cases
- **Empty intersection**: `reduce` returns empty geometry; handled by `.is_empty` check
- **Thin slivers**: Valid Shapely polygons, render correctly
- **Single circle**: `len(circles_data) >= 2` guard skips (no intersection to show)
- **GeometryCollection**: `geom_type` check filters to only Polygon types

## Verification
```bash
python scripts/analysis/million_scale/evaluate_million_scale.py
```
- Inspect `outputs/comparison/map_van_p*.png` and `map_ms_p*.png` — intersection regions should follow circle arcs instead of straight convex hull edges
- Area CDF values should change slightly (more accurate, slightly smaller than ConvexHull overestimate)
- Error CDF and centroid computation should be **unchanged** (not affected by this change)

## Rollback
Committed before this change at `0c1045c`. Revert with `git revert 0c1045c` if needed (though this task creates a new commit on top).
