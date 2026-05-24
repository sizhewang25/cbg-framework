"""Country-polygon loading + landmark-to-nearest-border distance.

Used by `scripts/vp_selection/iclab_verifier.py` to evaluate whether a
landmark's RTT to a target implies a propagation speed faster than the
calibrated limit, given the target's *claimed country*. The "distance" the
verifier checks is the great-circle distance from the landmark to the nearest
point on the claimed country's border (zero if the landmark sits inside).

Implementation: per-call azimuthal-equidistant projection centred on the
landmark — `polygon.distance(point)` in that CRS returns the true geodesic
distance in km, since the AEQD projection preserves distances from its
projection centre.

The Natural Earth low-resolution country dataset (110 m) is the expected
input for `load_country_polygons`. Tests use synthetic polygons so they
don't depend on the shapefile being present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)


def load_country_polygons(
    shapefile_path: Path,
    iso_kind: str = "auto",
) -> tuple[dict[str, BaseGeometry], str]:
    """Load country polygons from a Natural Earth (or compatible) shapefile.

    Returns `(polygons, iso_kind)`:
      * `polygons`: `{iso_code: polygon}` — keyed by whichever ISO code the
        shapefile has (ISO_A2 or ISO_A3).
      * `iso_kind`: `"ISO_A2"` or `"ISO_A3"`, so callers know what kind of
        code to look up by.

    `iso_kind="auto"` prefers ISO_A2 if present, else ISO_A3. Pass an explicit
    value to force one. Rows whose ISO code is "-99" (Natural Earth sentinel
    for disputed/unknown) or empty are dropped.
    """
    import geopandas as gpd
    gdf = gpd.read_file(shapefile_path)
    iso_col = _resolve_iso_column(gdf.columns, iso_kind)
    resolved_kind = iso_col.upper().replace("ISO_", "ISO_")
    if resolved_kind not in ("ISO_A2", "ISO_A3"):
        # Normalize: 'iso_a2' / 'iso_a3' → 'ISO_A2' / 'ISO_A3'
        resolved_kind = "ISO_" + iso_col.upper().split("_")[-1]
    polygons: dict[str, BaseGeometry] = {}
    for _, row in gdf.iterrows():
        iso = row[iso_col]
        if not isinstance(iso, str) or iso in ("-99", ""):
            continue
        polygons[iso] = row.geometry
    return polygons, resolved_kind


def _resolve_iso_column(columns, iso_kind: str) -> str:
    """Find the ISO_A2 or ISO_A3 column (case-insensitive), per `iso_kind`."""
    cols_by_upper = {c.upper(): c for c in columns}
    if iso_kind == "auto":
        if "ISO_A2" in cols_by_upper:
            return cols_by_upper["ISO_A2"]
        if "ISO_A3" in cols_by_upper:
            return cols_by_upper["ISO_A3"]
        raise ValueError(
            f"no ISO_A2/ISO_A3 column found; got: {list(columns)}"
        )
    requested = iso_kind.upper()
    if requested not in ("ISO_A2", "ISO_A3"):
        raise ValueError(f"iso_kind must be 'auto', 'ISO_A2' or 'ISO_A3'")
    if requested not in cols_by_upper:
        raise ValueError(
            f"shapefile has no {requested} column; got: {list(columns)}"
        )
    return cols_by_upper[requested]


def nearest_border_distance_km(
    landmark: tuple[float, float],
    country_iso2: str,
    polygons: dict[str, BaseGeometry],
) -> float:
    """Great-circle distance (km) from landmark `(lat, lon)` to the nearest
    point on `country_iso2`'s border.

    Returns 0.0 if the landmark sits inside (or on) the polygon. Raises
    `KeyError` if the country isn't present in `polygons`.
    """
    if country_iso2 not in polygons:
        raise KeyError(country_iso2)
    polygon = polygons[country_iso2]
    lat, lon = landmark
    pt = Point(lon, lat)  # shapely is (x, y) = (lon, lat)
    if polygon.contains(pt) or polygon.touches(pt):
        return 0.0
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=km")
    project = Transformer.from_crs(_WGS84, aeqd, always_xy=True).transform
    polygon_aeqd = shapely_transform(project, polygon)
    pt_aeqd = shapely_transform(project, pt)  # at origin
    return polygon_aeqd.distance(pt_aeqd)


def precompute_landmark_country_distances(
    landmarks: dict[str, tuple[float, float]],
    country_iso2s: Iterable[str],
    polygons: dict[str, BaseGeometry],
) -> dict[tuple[str, str], float]:
    """Precompute the (landmark_id, country_iso2) → km lookup table.

    Skips countries not present in `polygons` (silently — the caller may pass
    a superset of country codes from anchor metadata).
    """
    countries = [cc for cc in country_iso2s if cc in polygons]
    skipped = sorted(set(country_iso2s) - set(countries))
    if skipped:
        logger.info("skipping %d country codes not in polygons: %s",
                    len(skipped), skipped)
    table: dict[tuple[str, str], float] = {}
    for lm_id, coord in landmarks.items():
        for cc in countries:
            table[(lm_id, cc)] = nearest_border_distance_km(coord, cc, polygons)
    return table
