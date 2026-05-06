"""Phase 4 variant: Geometric Centroid (area-weighted).

Uses Shapely's .centroid -- the area-weighted center of mass of the
intersection polygon. For vanilla CBG vertex lists, orders the crossing
vertices into a polygon first, then computes the same area centroid.

Reference: centroid_comparison.py :: compute_shapely_centroid()
"""

from __future__ import annotations

from math import atan2, cos, degrees, radians, sin
from typing import Optional, Tuple

from scripts.framework.centroid import BaseCentroid
from scripts.framework.geometry import get_middle_intersection
from scripts.framework.registry import register_centroid
from scripts.framework.types import MultilatResult

_MIN_LON_SCALE = 1e-12


@register_centroid("geometric_centroid")
class GeometricCentroid(BaseCentroid):
    """Area-weighted centroid via Shapely.
    https://shapely.readthedocs.io/en/2.1.1/reference/shapely.centroid.html

    Accepts Shapely region inputs from planar multilateration. Also accepts
    unordered vanilla-CBG crossing vertices from spherical multilateration:
    deduplicates them, orders them around their center, builds a local planar
    polygon, and returns its area-weighted centroid.
    """

    name = "geometric_centroid"

    def __init__(
        self,
        dedupe_tolerance_deg: float = 1e-9,
        area_epsilon: float = 1e-12,
    ):
        self.dedupe_tolerance_deg = dedupe_tolerance_deg
        self.area_epsilon = area_epsilon

    def select(self, result: MultilatResult) -> Optional[Tuple[float, float]]:
        if not result.success:
            return None

        # Shapely region path (from planar multilateration)
        if result.region is not None:
            if result.region.is_empty:
                return None
            c = result.region.centroid
            if c.is_empty:
                return None
            return (c.y, c.x)  # Shapely: y=lat, x=lon

        # Vertex path (from spherical_circle vanilla CBG multilateration)
        if result.vertices is not None:
            return _centroid_from_vertices(
                result.vertices,
                self.dedupe_tolerance_deg,
                self.area_epsilon,
            )

        return None


def _centroid_from_vertices(
    vertices: list,
    dedupe_tolerance_deg: float,
    area_epsilon: float,
) -> Optional[Tuple[float, float]]:
    unique = _dedupe_vertices(vertices, dedupe_tolerance_deg)

    if len(unique) == 0:
        return None
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return get_middle_intersection(unique)

    local_points, lat0, lon0, lon_scale = _local_project(unique)
    if lon_scale < _MIN_LON_SCALE:
        return None

    ordered = sorted(local_points, key=lambda p: atan2(p[1], p[0]))

    from shapely.geometry import Polygon

    polygon = Polygon(ordered)
    if polygon.is_empty or not polygon.is_valid or abs(polygon.area) <= area_epsilon:
        return None

    centroid = polygon.centroid
    if centroid.is_empty:
        return None

    lat = lat0 + centroid.y
    lon = _normalize_longitude(lon0 + centroid.x / lon_scale)
    return lat, lon


def _dedupe_vertices(
    vertices: list,
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
    """Project (lat, lon) vertices to a local degree-based plane."""
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
