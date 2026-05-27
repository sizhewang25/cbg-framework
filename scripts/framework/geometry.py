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

EARTH_RADIUS_KM = 6371.0

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


def arithmetic_mean_centroid(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Arithmetic mean of a finite set of (lat, lon) points.

    Implements the finite-set centroid from
    https://en.wikipedia.org/wiki/Centroid#Of_a_finite_set_of_points: the
    coordinate-wise mean. Not an area-weighted polygon centroid. Matches the
    legacy Million-Scale helper `scripts.utils.helpers.polygon_centroid`.
    """
    lat_sum = 0.0
    lon_sum = 0.0
    for lat, lon in points:
        lat_sum += lat
        lon_sum += lon
    return lat_sum / len(points), lon_sum / len(points)


def get_middle_intersection(
    intersections: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Return the geodetic midpoint for exactly two intersection points."""
    (lat1, lon1) = intersections[0]
    (lat2, lon2) = intersections[1]

    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    bx = np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
    by = np.cos(lat2_rad) * np.sin(lon2_rad - lon1_rad)
    lat_mid = np.arctan2(
        np.sin(lat1_rad) + np.sin(lat2_rad),
        np.sqrt((np.cos(lat1_rad) + bx) ** 2 + by**2),
    )
    lon_mid = lon1_rad + np.arctan2(by, np.cos(lat1_rad) + bx)

    return float(np.degrees(lat_mid)), float(np.degrees(lon_mid))


def _as_point_array(points: Sequence[tuple[float, float]]) -> np.ndarray:
    """Normalize a point sequence to an (n, 2) float array of (lat, lon)."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be an array-like of (lat, lon) pairs")
    if len(arr) == 0:
        raise ValueError("points must not be empty")
    return arr


def _haversine_matrix_chunk(chunk: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances from chunk rows to all point rows."""
    lat1 = np.radians(chunk[:, 0])[:, None]
    lon1 = np.radians(chunk[:, 1])[:, None]
    lat2 = np.radians(points[:, 0])[None, :]
    lon2 = np.radians(points[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def sampled_medoid(
    points: Sequence[tuple[float, float]],
    chunk_size: int = 512,
) -> tuple[float, float]:
    """Return the sampled point with minimum total distance to all samples.

    This matches Octant's Monte Carlo point-selection semantics: the selected
    estimate is one of the sampled feasible points, so it remains inside the
    sampled region by construction.
    """
    arr = _as_point_array(points)
    if len(arr) == 1:
        return float(arr[0, 0]), float(arr[0, 1])

    totals = np.zeros(len(arr), dtype=float)
    for start in range(0, len(arr), chunk_size):
        end = min(start + chunk_size, len(arr))
        totals[start:end] = _haversine_matrix_chunk(arr[start:end], arr).sum(axis=1)

    idx = int(np.argmin(totals))
    return float(arr[idx, 0]), float(arr[idx, 1])


def nearest_sample_point(
    points: Sequence[tuple[float, float]],
    target: tuple[float, float],
) -> tuple[float, float]:
    """Return the sampled point nearest to target using great-circle distance."""
    arr = _as_point_array(points)
    target_arr = np.asarray([target], dtype=float)
    distances = _haversine_matrix_chunk(target_arr, arr).reshape(-1)
    idx = int(np.argmin(distances))
    return float(arr[idx, 0]), float(arr[idx, 1])


def continuous_geometric_median(
    points: Sequence[tuple[float, float]],
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[float, float]:
    """Approximate the continuous geometric median with Weiszfeld iterations."""
    arr = _as_point_array(points)
    if len(arr) == 1:
        return float(arr[0, 0]), float(arr[0, 1])

    estimate = np.mean(arr, axis=0)
    for _ in range(max_iterations):
        deltas = arr - estimate
        distances = np.linalg.norm(deltas, axis=1)

        exact = np.where(distances < tolerance)[0]
        if len(exact) > 0:
            point = arr[int(exact[0])]
            return float(point[0]), float(point[1])

        weights = 1.0 / distances
        next_estimate = np.sum(arr * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(next_estimate - estimate) < tolerance:
            estimate = next_estimate
            break
        estimate = next_estimate

    return float(estimate[0]), float(estimate[1])


def sample_points_in_region(
    region,
    n_samples: int = 5000,
    rng: Optional[np.random.Generator] = None,
    max_attempts_factor: int = 20,
) -> np.ndarray:
    """Sample (lat, lon) points inside a Shapely region using Sobol QMC."""
    from scipy.stats import qmc
    from shapely.geometry import Point

    if n_samples <= 0:
        return np.empty((0, 2))

    if rng is None:
        rng = np.random.default_rng()

    # Sobol gives better coverage uniformity with fewer points, compared to uniform sampling
    sobol_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    try:
        sampler = qmc.Sobol(d=2, scramble=True, rng=sobol_seed)
    except TypeError:
        sampler = qmc.Sobol(d=2, scramble=True, seed=sobol_seed)

    minx, miny, maxx, maxy = region.bounds  # Shapely: lon/lat bounds
    collected = []
    max_attempts = n_samples * max_attempts_factor
    attempts = 0

    while len(collected) < n_samples and attempts < max_attempts:
        remaining_needed = n_samples - len(collected)
        remaining_budget = max_attempts - attempts
        batch_size = min(max(remaining_needed * 4, 1), remaining_budget)
        sobol_points = sampler.random(batch_size)
        attempts += batch_size

        rand_lons = minx + (maxx - minx) * sobol_points[:, 0]
        rand_lats = miny + (maxy - miny) * sobol_points[:, 1]

        for lon, lat in zip(rand_lons, rand_lats):
            if region.contains(Point(lon, lat)):
                collected.append((lat, lon))
                if len(collected) >= n_samples:
                    break

    return np.array(collected) if collected else np.empty((0, 2))


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

    # Tolerance for the "inside every disk" filter (km). Every candidate is
    # one of the great-circle crossings of two disks' boundaries, so by
    # construction it sits *exactly* on those two boundaries; floating-point
    # noise from `haversine` then puts it ~1e-7 km outside, and a strict-`<`
    # check would reject it. EPS_KM = 1 mm is far below any meaningful disk
    # size and only absorbs that float-edge noise.
    EPS_KM = 1e-6
    filtered_points = []
    for point_geo in intersect_points:
        inside_all = True
        for lat_c, lon_c, _, d_c, _ in used_circles:
            if d_c + EPS_KM < haversine((lat_c, lon_c), point_geo):
                inside_all = False
                break
        if inside_all:
            filtered_points.append(point_geo)

    #  The feasible region's vertices are exactly the crossings that satisfy every other disk
    # list[(lat, lon)] of feasible-region vertices. 
    # Empty if no pairwise crossing lies inside all caps (i.e. caps don't have a common intersection).
    # the same circle tuples after normalization/preprocessing, so callers know which disks actually contributed.
    return filtered_points, used_circles
