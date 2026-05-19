# Local projection examples — `_local_project`

The vertex-path centroid in [scripts/framework/v2/ctr/geometric_centroid.py](../scripts/framework/v2/ctr/geometric_centroid.py)
projects (lat, lon) tuples into a local planar frame before angle-sorting and
area-weighting. The map is

    x = Δlon · cos(lat₀)
    y = Δlat

where `lat₀` is the arithmetic-mean latitude and `lon₀` is the *circular*-mean
longitude of the input vertices. `lon_scale = |cos(lat₀)|` is the equirectangular
correction — same factor used by `km_per_deg_lon` elsewhere in the codebase.

Below are three worked cases showing why each piece is needed.

---

## 1. Equator — identity projection

Input vertices `(lat, lon)`:

    (0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)

Frame:

- `lat₀ = 0.0`
- `lon₀ = 0.0`
- `lon_scale = cos(0°) = 1.0`

Local `(x, y)`:

    (-1.000, 0.0), ( 1.000, 0.0), ( 0.000, 1.0), ( 0.000, -1.0)

At the equator 1° lon ≈ 1° lat ≈ 111 km on the ground, so the projection is
the identity. The diamond stays a diamond.

---

## 2. Mid-latitude (Paris, ~49°N) — longitude shrink

Input:

    (49.0, 2.0), (49.0, 4.0), (50.0, 3.0), (48.0, 3.0)

Frame:

- `lat₀ = 49.0`
- `lon₀ = 3.0`
- `lon_scale = cos(49°) ≈ 0.6561`

Local `(x, y)`:

    (-0.656, 0.0), ( 0.656, 0.0), ( 0.000, 1.0), ( 0.000, -1.0)

The diamond becomes taller than wide, which is geometrically correct: at 49°N,
1° lon is only ~73 km while 1° lat is still ~111 km. After this scaling, equal
ground distance in x and y means equal weight in the centroid area integral —
without it, a polygon near the poles would have its longitude span over-counted.

---

## 3. Antimeridian crossing — circular mean saves the day

Input:

    (0.0, 179.0), (0.0, -179.0), (1.0, 180.0), (-1.0, 180.0)

These four points form a tight diamond *across* the date line, but a naive
arithmetic mean of the longitudes gives `(179 + -179 + 180 + 180)/4 = 90°` —
the opposite side of the globe.

Circular mean fixes it:

    sin_sum = sin(179°) + sin(-179°) + sin(180°) + sin(180°)
    cos_sum = cos(179°) + cos(-179°) + cos(180°) + cos(180°)
    lon₀    = atan2(sin_sum, cos_sum) ≈ 180.0°  (not 90°)

Then `_longitude_delta` computes the *shortest signed* angle:

- `_longitude_delta(179, 180)  = -1`
- `_longitude_delta(-179, 180) = +1`

Local `(x, y)` with `lat₀ = 0`, `lon_scale = 1`:

    (-1.0, 0.0), ( 1.0, 0.0), ( 0.000, 1.0), ( 0.000, -1.0)

Same clean diamond as case 1. Without circular mean + signed delta this would
expand into a 360°-wide phantom polygon and the centroid would land halfway
around the planet.

---

## Inverse projection

After centroiding in local space, the inverse is

    lat = lat₀ + y
    lon = normalize(lon₀ + x / lon_scale)

E.g. for the Paris case, suppose the local centroid is `(x=0.05, y=0.02)`:

- `lat = 49.0 + 0.02 = 49.02°`
- `lon = 3.0 + 0.05 / 0.6561 ≈ 3.076°`

`_normalize_longitude` only changes the answer when the result has drifted
across ±180° — for cases 1 and 2 it's a no-op.

---

## What the helpers guarantee

- `_circular_mean_longitude` — robust mean that respects the `[-180, 180]`
  wrap; reduces to the arithmetic mean when the inputs avoid the seam.
- `_longitude_delta(a, b)` — returns the shortest signed angle `a − b` in
  `(-180, 180]`, so subtraction never produces a 359° "long way around".
- `_normalize_longitude` — wraps a single longitude back into `(-180, 180]`,
  with a small special case that returns `+180` instead of `-180` for positive
  inputs landing on the boundary.

Together they make the projection well-behaved for polygons anywhere on the
globe **except** within `1e-12` of the poles, where `lon_scale → 0` and the
caller bails out with `Error.DEGENERATE_REGION`.
