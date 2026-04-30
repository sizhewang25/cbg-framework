"""Framework-owned geometry helpers for CBG.

These functions intentionally duplicate the legacy helpers used by the
Million-Scale code path so the modular framework can fix behavior without
changing historical analysis scripts.
"""

from __future__ import annotations

import itertools
from math import cos, pi
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from scripts.framework.types import EARTH_RADIUS_KM

CircleTuple = Tuple[float, float, float, float, float]


def internet_speed(rtt: float, speed_threshold: Optional[float]) -> float:
    """Return the propagation-speed fraction used for an RTT measurement."""
    if speed_threshold is not None:
        return speed_threshold
    if rtt >= 80:
        return 4 / 9
    if 5 <= rtt < 80:
        return 3 / 9
    return 1 / 6


def rtt_to_km(
    rtt: float,
    speed_threshold: Optional[float] = None,
    c: float = 300,
) -> float:
    """Convert RTT in milliseconds to a one-way distance bound in km."""
    return internet_speed(rtt, speed_threshold) * rtt * c / 2


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [*a, *b])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    x = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(x))


def geo_to_cartesian(lat: float, lon: float) -> tuple[float, float, float]:
    """Convert latitude/longitude in degrees to unit-sphere Cartesian coords."""
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180
    x = np.cos(lon_rad) * np.cos(lat_rad)
    y = np.sin(lon_rad) * np.cos(lat_rad)
    z = np.sin(lat_rad)
    return x, y, z


def _normalize_circles(
    circles: Iterable[Sequence[float]],
    speed_threshold: Optional[float] = None,
) -> list[CircleTuple]:
    """Fill missing distance/radian radius fields in legacy circle tuples."""
    normalized = []
    for c in circles:
        lat, lon, rtt, d, r = c
        if d is None:
            d = rtt_to_km(rtt, speed_threshold)
        if r is None:
            r = d / EARTH_RADIUS_KM
        normalized.append((lat, lon, rtt, d, r))
    return normalized


def check_circle_inclusion(
    c1: CircleTuple,
    c2: CircleTuple,
) -> tuple[Optional[CircleTuple], Optional[CircleTuple]]:
    """Return (remove, keep) if one disk fully contains the other."""
    lat1, lon1, _, d1, _ = c1
    lat2, lon2, _, d2, _ = c2
    center_distance = haversine((lat1, lon1), (lat2, lon2))
    if d1 > center_distance + d2:
        return c1, c2
    if d2 > center_distance + d1:
        return c2, c1
    return None, None


def circle_preprocessing(
    circles: Iterable[Sequence[float]],
    speed_threshold: Optional[float] = None,
) -> list[CircleTuple]:
    """Remove disk constraints that fully contain another disk constraint.

    This is the redundant-circle rule from the legacy Million-Scale helper,
    returned in input order instead of as a set for deterministic framework
    behavior. It is disk-only and intentionally ignores annular inner radii.
    """
    normalized = _normalize_circles(circles, speed_threshold)
    ignored: set[CircleTuple] = set()

    for i, c1 in enumerate(normalized):
        if c1 in ignored:
            continue
        for c2 in normalized[i + 1 :]:
            if c2 in ignored:
                continue
            remove, _ = check_circle_inclusion(c1, c2)
            if remove is not None:
                ignored.add(remove)

    return [c for c in normalized if c not in ignored]


def get_points_on_circle(
    lat_c: float,
    lon_c: float,
    radius_km: float,
    nb_points: int = 4,
) -> list[tuple[float, float]]:
    """Return evenly spaced approximate points on a circle."""
    points = []
    for k in range(nb_points):
        angle = pi * 2 * k / nb_points
        dx = radius_km * 1000 * cos(angle)
        dy = radius_km * 1000 * np.sin(angle)
        lat = lat_c + (180 / pi) * (dy / 6378137)
        lon = lon_c + (180 / pi) * (dx / 6378137) / cos(lat_c * pi / 180)
        points.append((lat, lon))
    return points


def circle_intersections(
    circles: Iterable[Sequence[float]],
    speed_threshold: Optional[float] = None,
    preprocess: bool = False,
) -> tuple[list[tuple[float, float]], list[CircleTuple]]:
    """Compute spherical circle intersections and filter points by all disks.

    This is the corrected framework copy of the legacy helper. Corrections:
    - redundant-circle preprocessing is optional instead of implicit
    - point filtering uses the precomputed distance `d` rather than recomputing
      radius from RTT
    - coincident/antipodal center degeneracies are guarded
    """
    normalized = _normalize_circles(circles, speed_threshold)
    used_circles = (
        circle_preprocessing(normalized, speed_threshold)
        if preprocess
        else normalized
    )

    if not used_circles:
        return [], used_circles

    if len(used_circles) == 1:
        lat, lon, _, d, _ = used_circles[0]
        return get_points_on_circle(lat, lon, d), used_circles

    intersect_points = []
    for c1, c2 in itertools.combinations(used_circles, 2):
        lat1, lon1, _, _, r1 = c1
        lat2, lon2, _, _, r2 = c2

        x1 = np.array(list(geo_to_cartesian(lat1, lon1)))
        x2 = np.array(list(geo_to_cartesian(lat2, lon2)))

        q = np.dot(x1, x2)
        denom = 1 - q**2
        if abs(denom) < 1e-12:
            continue

        a = (np.cos(r1) - np.cos(r2) * q) / denom
        b = (np.cos(r2) - np.cos(r1) * q) / denom

        x0 = a * x1 + b * x2
        n = np.cross(x1, x2)
        nn = np.dot(n, n)
        if nn < 1e-12:
            continue

        val = (1 - np.dot(x0, x0)) / nn
        if val <= 0:
            continue

        t = np.sqrt(val)
        for sign in (1, -1):
            point = x0 + sign * t * n
            i_lon = np.arctan2(point[1], point[0]) * (180 / np.pi)
            i_lat = np.arctan(
                point[2] / np.sqrt(point[0] ** 2 + point[1] ** 2)
            ) / (np.pi / 180)
            intersect_points.append((i_lat, i_lon))

    filtered_points = []
    for point_geo in intersect_points:
        inside_all = True
        for lat_c, lon_c, _, d_c, _ in used_circles:
            if d_c < haversine((lat_c, lon_c), point_geo):
                inside_all = False
                break
        if inside_all:
            filtered_points.append(point_geo)

    return filtered_points, used_circles
