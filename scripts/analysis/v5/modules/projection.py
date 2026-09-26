"""The plane the cells are drawn in: EPSG:5070, and nothing else.

A Voronoi diagram is planar geometry, so drawing one needs a projection even
though the rule it draws is spherical. `classify` scores by great-circle
nearest seed and never touches this module; these functions exist so
`cells.py` can build polygons and hand them back in `(lon, lat)` for the map.

## Why EPSG:5070

CONUS Albers equal-area. Its distance distortion over the lower 48 is ~1-2%,
which is why a planar Voronoi here is not exactly the great-circle partition:
a boundary far from its seeds shifts by several km.
`cells.agreement_with_nearest_seed` measures that gap on a point sample and
the map's manifest publishes it, so the figure states how faithfully it draws
the rule rather than asserting it.

This module is the polygon-free remainder of what used to be `landmass.py`.
The landmass itself is gone: the cell partition is now unbounded, so there is
no polygon to buffer, test against, or clip to -- only a plane to work in.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import shapely

#: CONUS Albers equal-area. Fixed, not a parameter: `cells` builds its
#: diagram here and `geodesy` does every distance on the sphere, so nothing
#: downstream chooses a projection.
PROJECTED_CRS = "EPSG:5070"


def project(lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
    """`(x, y)` metres in `PROJECTED_CRS`."""
    lat = np.asarray(lat_deg, dtype=float).ravel()
    lon = np.asarray(lon_deg, dtype=float).ravel()
    # Lists, not arrays: pyproj takes a length-1 array for a scalar and trips
    # NumPy's array-to-scalar deprecation.
    x, y = _to_projected().transform(lon.tolist(), lat.tolist())
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def to_lonlat(geom: shapely.Geometry) -> shapely.Geometry:
    """A projected geometry back in `(lon, lat)` degrees, for drawing."""
    from shapely.ops import transform

    return transform(_to_geographic().transform, geom)


def describe() -> dict:
    """Static facts about the plane, for a `meta.json` block."""
    return {
        "projected_crs": PROJECTED_CRS,
        "note": (
            "the plane cell polygons are built in; scoring is great-circle "
            "nearest seed and uses no projection"
        ),
    }


@lru_cache(maxsize=1)
def _to_projected():
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)


@lru_cache(maxsize=1)
def _to_geographic():
    from pyproj import Transformer

    return Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)
