"""Area-weighted geometric centroid via Shapely (v2 port).

For Shapely region inputs, returns the area-weighted centroid directly. For
unordered vertex inputs from spherical multilateration, deduplicates, orders
around the local-projection center, builds a local planar polygon, and
returns its area centroid.
"""

from __future__ import annotations

from math import atan2, cos, degrees, radians, sin
from typing import Optional, Tuple

from shapely.geometry.base import BaseGeometry

from scripts.framework.geometry import get_middle_intersection
from scripts.framework.v2.ctr.base import CTRMethod, CTRResult
from scripts.framework.v2.mtl.base import MTLResult
from scripts.framework.v2.registry import register_ctr
from scripts.framework.v2.types import Coord, Error

_MIN_LON_SCALE = 1e-12


@register_ctr("geometric_centroid")
class GeometricCentroidCTR(CTRMethod):
    """Area-weighted centroid via Shapely.

    Accepts Shapely region inputs from planar multilateration. Also accepts
    unordered vanilla-CBG crossing vertices from spherical multilateration:
    deduplicates them, orders them around their center, builds a local planar
    polygon, and returns its area-weighted centroid.
    """

    def __init__(
        self,
        dedupe_tolerance_deg: float = 1e-9,
        area_epsilon: float = 1e-12,
    ) -> None:
        self.dedupe_tolerance_deg = dedupe_tolerance_deg
        self.area_epsilon = area_epsilon

    def _select_centroid(self, mtl: MTLResult) -> CTRResult:
        if not mtl.success:
            return CTRResult(success=False, error=Error.EMPTY_REGION)

        intersection = mtl.intersection

        if isinstance(intersection, BaseGeometry):
            if intersection.is_empty:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            c = intersection.centroid
            if c.is_empty:
                return CTRResult(success=False, error=Error.NUMERICAL_FAILURE)
            return CTRResult(success=True, tg_coord=Coord(c.y, c.x))

        if isinstance(intersection, list):
            if len(intersection) == 0:
                return CTRResult(success=False, error=Error.EMPTY_REGION)
            verts = [(c.lat, c.lon) for c in intersection]
            point, error = _centroid_from_vertices(
                verts, self.dedupe_tolerance_deg, self.area_epsilon,
            )
            if point is None:
                return CTRResult(success=False, error=error)
            return CTRResult(success=True, tg_coord=Coord(point[0], point[1]))

        return CTRResult(success=False, error=Error.EMPTY_REGION)


def _centroid_from_vertices(
    vertices: list[Tuple[float, float]],
    dedupe_tolerance_deg: float,
    area_epsilon: float,
) -> Tuple[Optional[Tuple[float, float]], Optional[Error]]:
    unique = _dedupe_vertices(vertices, dedupe_tolerance_deg)

    if len(unique) == 0:
        return None, Error.EMPTY_REGION
    if len(unique) == 1:
        return unique[0], None
    if len(unique) == 2:
        return get_middle_intersection(unique), None

    local_points, lat0, lon0, lon_scale = _local_project(unique)
    if lon_scale < _MIN_LON_SCALE:
        return None, Error.DEGENERATE_REGION

    ordered = sorted(local_points, key=lambda p: atan2(p[1], p[0]))

    from shapely.geometry import Polygon

    polygon = Polygon(ordered)
    if polygon.is_empty or not polygon.is_valid or abs(polygon.area) <= area_epsilon:
        return None, Error.DEGENERATE_REGION

    centroid = polygon.centroid
    if centroid.is_empty:
        return None, Error.NUMERICAL_FAILURE

    lat = lat0 + centroid.y
    lon = _normalize_longitude(lon0 + centroid.x / lon_scale)
    return (lat, lon), None


def _dedupe_vertices(
    vertices: list[Tuple[float, float]],
    tolerance_deg: float,
) -> list[Tuple[float, float]]:
    unique: list[Tuple[float, float]] = []
    for lat, lon in vertices:
        point = (float(lat), float(lon))
        if not any(
            abs(point[0] - other[0]) <= tolerance_deg
            and abs(_longitude_delta(point[1], other[1])) <= tolerance_deg
            for other in unique
        ):
            unique.append(point)
    return unique


def _local_project(
    vertices: list[Tuple[float, float]],
) -> tuple[list[Tuple[float, float]], float, float, float]:
    """
    Given an unordered convex point set, 
    sorting by polar angle around the centroid produces a valid Counter-clockwise polygon.
    Assumption: vanilla-CBG circle crossings form a convex-ish ring.
    """
    lat0 = sum(lat for lat, _ in vertices) / len(vertices)
    lon0 = _circular_mean_longitude([lon for _, lon in vertices])
    lon_scale = abs(cos(radians(lat0)))

    local_points = []
    for lat, lon in vertices:
        x = _longitude_delta(lon, lon0) * max(lon_scale, _MIN_LON_SCALE)
        y = lat - lat0
        local_points.append((x, y))

    return local_points, lat0, lon0, lon_scale


def _circular_mean_longitude(lons: list[float]) -> float:
    """
    Averages unit vectors on the longitude circle so points near ±180° don't get a nonsense mean of 0°
    """
    sin_sum = sum(sin(radians(lon)) for lon in lons)
    cos_sum = sum(cos(radians(lon)) for lon in lons)
    if abs(sin_sum) < 1e-15 and abs(cos_sum) < 1e-15:
        return _normalize_longitude(sum(lons) / len(lons))
    return _normalize_longitude(degrees(atan2(sin_sum, cos_sum)))


def _longitude_delta(lon: float, origin_lon: float) -> float:
    return (lon - origin_lon + 180.0) % 360.0 - 180.0


def _normalize_longitude(lon: float) -> float:
    normalized = (lon + 180.0) % 360.0 - 180.0
    if normalized == -180.0 and lon > 0:
        return 180.0
    return normalized
