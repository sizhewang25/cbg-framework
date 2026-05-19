# Km-to-degree circle conversion in planar MTL

How `_circle_to_shapely` (and the equivalent `_circle_to_planar_polygon` in
`scripts/framework/v2/mtl/planar_circle.py`) turns a geographic disk of
radius `radius_km` centered at `(center_lat, center_lon)` into a Shapely
polygon in **degree space**.

Reference:
- [scripts/libs/octant/octant_geolocation.py:152-179](../scripts/libs/octant/octant_geolocation.py#L152-L179)
- [scripts/framework/v2/mtl/planar_circle.py:28-45](../scripts/framework/v2/mtl/planar_circle.py#L28-L45)

## The geometry

A circle of radius `r` centered at `(x₀, y₀)` is parameterized by angle
θ ∈ [0, 2π):

```
x = x₀ + r·cos(θ)
y = y₀ + r·sin(θ)
```

In the code, `x = lon` and `y = lat` are in **degrees**, while
`radius_km` is in **kilometers**. So we need a km → degree conversion
*per axis*, because a degree of longitude is shorter than a degree of
latitude away from the equator.

## The km/degree conversion

```python
km_per_deg_lat = 111.0                              # ≈ constant everywhere
km_per_deg_lon = 111.0 * cos(center_lat_radians)    # shrinks toward the poles
```

Why the cosine? Lines of longitude converge at the poles. At latitude
φ, the circumference of the parallel is `2π · R · cos(φ)`, so a degree
of longitude there spans `111 · cos(φ)` km:

| Latitude | km per degree of lon |
|---|---|
| 0° (equator) | 111.0 |
| 30° | 96.1 |
| 45° | 78.5 |
| 60° | 55.5 |
| 90° (pole) | 0 |

Then convert the km radius into a per-axis degree radius:

```python
r_lat = radius_km / 111.0          # degrees of latitude spanned by r
r_lon = radius_km / (111·cos φ)    # degrees of longitude spanned by r
```

The code also clamps `km_per_deg_lon` to at least `1.0` to avoid a
divide-by-zero at the poles — a defensive choice, not a physical one.

## Result: an ellipse in degree space

```python
lons = center_lon + r_lon · cos(angles)
lats = center_lat + r_lat · sin(angles)
```

In **km space** this traces a true circle of radius `radius_km`. In
**degree space** (what Shapely sees) it traces an **ellipse** — wider
in longitude than latitude when `cos φ < 1` — because the lon axis is
squashed to compensate for the fact that one degree of lon is
"smaller" than one degree of lat.

This matters because Shapely operates on degree-space coordinates with
no notion of map projection: intersection, area, contains-tests all
treat (lon, lat) as a flat Cartesian plane. The ellipse is the correct
shape *in that flat plane* so that the polygon in km space remains a
proper circle.

## Why it's an approximation

This treats a small patch of Earth as flat with locally constant
`km_per_deg_*` values. It breaks down when:

- **Radius is large.** The `cos(φ)` term changes across the disk, but
  we evaluate it once at the center.
- **Center is at high latitude.** `cos(φ)` changes fast near the poles,
  so the constant-cos approximation is poorest exactly where it matters
  most.
- **Disk crosses the antimeridian.** Longitude wraps from +180 to −180;
  this conversion does not handle the discontinuity.
- **Disk crosses a pole.** The single-φ approximation cannot represent
  a region that contains the pole at all.

For continental-scale geolocation in temperate latitudes this is fine;
for global problems use the spherical variant
(`scripts/framework/v2/mtl/spherical_circle.py`), which works directly
in great-circle geometry.

## Why not `Point(lon, lat).buffer(r)`?

Shapely's `Point.buffer(r)` would treat `r` as a degree-space radius
(same units as the coordinates) and produce a circle in degree space —
which is an ellipse in km space, with the wrong axis stretched. The
manual parametric construction here puts the correction in the right
place: degrees stretched on the lon axis so the result is a circle in
km space.

## Vertex count

Both `_circle_to_shapely` and `_circle_to_planar_polygon` default to
roughly Shapely's own circle approximation: `quad_segs=16` in
`buffer()` produces 64 vertices. The octant version uses `n_pts=100`
(legacy); the v2 version was changed to 64 to match Shapely's
convention.

The dominant error source is the flat-Earth `km_per_deg_*` model, not
vertex count — bumping `n_pts` past ~64 reduces polygon-discretization
error but cannot fix the projection error.
