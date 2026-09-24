"""Great-circle distances and the spherical centroid.

Chord form (unit-vector dot products) rather than haversine, so one matrix
multiply covers every pair; equivalent to within floating point. Copied from
v4's `answer_space`, where it lived beside code it had nothing to do with.
"""

from __future__ import annotations

import numpy as np

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM


def unit_vectors(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=float).ravel())
    lon = np.radians(np.asarray(lon_deg, dtype=float).ravel())
    cos_lat = np.cos(lat)
    return np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])


def pairwise_km(lat_a, lon_a, lat_b=None, lon_b=None) -> np.ndarray:
    """Distance matrix in km; `a` against itself when `b` is omitted."""
    a = unit_vectors(lat_a, lon_a)
    b = a if lat_b is None else unit_vectors(lat_b, lon_b)
    return EARTH_RADIUS_KM * np.arccos(np.clip(a @ b.T, -1.0, 1.0))


def elementwise_km(lat_a, lon_a, lat_b, lon_b) -> np.ndarray:
    """Paired distance: `out[i] = d(a[i], b[i])`."""
    a = unit_vectors(lat_a, lon_a)
    b = unit_vectors(lat_b, lon_b)
    return EARTH_RADIUS_KM * np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))


def spherical_centroid(lat_deg, lon_deg) -> tuple[float, float]:
    """Normalised mean of the unit vectors, as `(lat, lon)` degrees.

    Not the mean of the coordinates: that is wrong across the antimeridian and
    biased poleward at continental spans. Undefined for antipodal inputs, which
    no set of sites within one grid diameter can be.
    """
    v = unit_vectors(lat_deg, lon_deg).mean(axis=0)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("centroid undefined: points cancel on the sphere")
    x, y, z = v / norm
    return float(np.degrees(np.arcsin(z))), float(np.degrees(np.arctan2(y, x)))
