"""The landmass: what bounds the cells, and what `outland` means.

The **landmass** is the US mainland -- the lower 48 -- **buffered by one
`grid_km`** at each rung. A prediction outside it is `outland`; inside it, it
falls in exactly one cell (the Voronoi cell of its nearest seed).

This bound is what makes the cell partition admissible. Unbounded, a Voronoi
partition over K seeds labels every point on Earth: v3 credited Seattle with a
prediction in the Canadian Arctic 2,360 km away (`tg-e1a1545`, as01) because
Seattle was marginally the nearest seed. Under the landmass that prediction is
outland at every rung.

## Territory, not coastline

The polygon is Natural Earth's **country** outline, whose border runs through
the Great Lakes, so the US side of each lake is inland. Leaving US territory is
the only way to be outland. Alaska and Hawaii are not part of the mainland and
are outland.

## Why buffer, and by `grid_km`

A prediction a few km offshore of a coastal site is not an operational
failure, and whether it lands on land depends on coastline resolution as much
as on the method. Buffering by the rung's own `grid_km` makes `outland` mean
"more than one grid outside the territory" -- the same tolerance the rest of
the answer space is built at. The whole polygon is buffered, land borders
included: at nside 16 (407 km) Toronto and Vancouver are inland, which is
consistent -- at that grid size they cannot be told apart from the US side.

The 110m source is ~10 km coarse at the coast, inside the smallest buffer
(50.9 km).

## Projection

The buffer is taken in **EPSG:5070** (CONUS Albers equal-area), whose distance
distortion over the lower 48 is ~1-2%. Buffering in degrees would be wrong by
a factor of cos(latitude) east-west.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely

#: The vendored, unbuffered lower-48 polygon. Rebuild with
#: `python -m scripts.analysis.v5.data.make_us_mainland`.
LANDMASS_GEOJSON = Path(__file__).resolve().parents[1] / "data" / "us_mainland.geojson"

LANDMASS_NAME = "US mainland (lower 48)"
LANDMASS_SOURCE = "Natural Earth ne_110m_admin_0_countries, ADM0_A3=USA"
PROJECTED_CRS = "EPSG:5070"


@dataclass(frozen=True)
class Landmass:
    """The buffered landmass in projected metres, prepared for point tests."""

    buffer_km: float
    geometry: shapely.Geometry

    def contains(self, lat_deg, lon_deg) -> np.ndarray:
        """Boolean per point: inside the buffered landmass. NaN -> False."""
        lat = np.asarray(lat_deg, dtype=float).ravel()
        lon = np.asarray(lon_deg, dtype=float).ravel()
        out = np.zeros(lat.shape, dtype=bool)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if ok.any():
            # Lists, not arrays: pyproj takes a length-1 array for a scalar and
            # trips NumPy's array-to-scalar deprecation.
            x, y = _to_projected().transform(lon[ok].tolist(), lat[ok].tolist())
            out[ok] = shapely.contains_xy(self.geometry, np.asarray(x), np.asarray(y))
        return out

    def project(self, lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
        """`(x, y)` metres in `PROJECTED_CRS`, the plane the landmass lives in."""
        lat = np.asarray(lat_deg, dtype=float).ravel()
        lon = np.asarray(lon_deg, dtype=float).ravel()
        x, y = _to_projected().transform(lon.tolist(), lat.tolist())
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def to_lonlat(self, geom: shapely.Geometry) -> shapely.Geometry:
        """A projected geometry back in `(lon, lat)` degrees, for drawing."""
        from shapely.ops import transform

        return transform(_to_geographic().transform, geom)

    def describe(self) -> dict:
        return {
            "name": LANDMASS_NAME,
            "source": LANDMASS_SOURCE,
            "file": LANDMASS_GEOJSON.name,
            "projected_crs": PROJECTED_CRS,
            "buffer_km": round(self.buffer_km, 3),
            "buffer_note": "whole polygon buffered by grid_km; outland = beyond it",
        }


@lru_cache(maxsize=1)
def _to_projected():
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)


@lru_cache(maxsize=1)
def _to_geographic():
    from pyproj import Transformer

    return Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)


@lru_cache(maxsize=1)
def _mainland_projected() -> shapely.Geometry:
    from shapely.geometry import shape
    from shapely.ops import transform

    features = json.loads(LANDMASS_GEOJSON.read_text())["features"]
    if len(features) != 1:
        raise ValueError(f"{LANDMASS_GEOJSON} must hold one feature, has {len(features)}")
    return transform(_to_projected().transform, shape(features[0]["geometry"]))


@lru_cache(maxsize=16)
def load_landmass(buffer_km: float) -> Landmass:
    """The mainland buffered by `buffer_km`. Cached: four rungs, four buffers."""
    if not np.isfinite(buffer_km) or buffer_km < 0:
        raise ValueError(f"buffer_km must be a non-negative number, got {buffer_km}")
    geom = _mainland_projected().buffer(float(buffer_km) * 1000.0)
    shapely.prepare(geom)
    return Landmass(buffer_km=float(buffer_km), geometry=geom)
