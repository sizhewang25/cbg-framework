from __future__ import annotations

import numpy as np

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM


def resolve_effective_top_n(requested_top_n: int, n_centroids: int) -> int:
    """Validate top-N and cap it by the number of centroids."""
    if requested_top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {requested_top_n}")
    if n_centroids < 1:
        raise ValueError("n_centroids must be >= 1")
    return min(int(requested_top_n), int(n_centroids))


def build_truth_neighbor_index(
    centroid_lat: np.ndarray,
    centroid_lon: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """Return centroid-neighbor ids for each centroid (self first).

    Output shape: (n_centroids, effective_top_n).
    """
    lat = np.asarray(centroid_lat, dtype=float)
    lon = np.asarray(centroid_lon, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError("centroid_lat and centroid_lon must have same shape")
    n_centroids = lat.shape[0]
    k = resolve_effective_top_n(top_n, n_centroids)

    # Pairwise great-circle distances between all centroid pairs.
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    central = 2.0 * np.arcsin(np.sqrt(a))
    dist_km = EARTH_RADIUS_KM * central

    # Rank nearest centroids for each row; self will be first (distance 0).
    idx = np.argsort(dist_km, axis=1)
    return idx[:, :k].astype(int)


def compute_topk_match_from_truth_neighbors(
    pred_centroid_idx: np.ndarray,
    truth_centroid_idx: np.ndarray,
    truth_neighbors: np.ndarray,
    top_n: int,
    *,
    column_prefix: str = "match_top",
) -> dict[str, np.ndarray]:
    """Compute cumulative top-K match booleans.

    Semantics:
    - Build S_K(y) from truth centroid id y as: {y} + (K-1) nearest centroid
      neighbors to y (derived from truth_neighbors row for y).
    - Row is correct at K iff predicted centroid is in S_K(y).

    Invalid predicted/truth centroid ids are treated as False for all K.
    """
    pred = np.asarray(pred_centroid_idx, dtype=int)
    truth = np.asarray(truth_centroid_idx, dtype=int)
    if pred.shape != truth.shape:
        raise ValueError("pred_centroid_idx and truth_centroid_idx must have same shape")
    if truth_neighbors.ndim != 2 or truth_neighbors.shape[0] == 0:
        raise ValueError("truth_neighbors must be a non-empty 2D array")

    max_k = resolve_effective_top_n(top_n, truth_neighbors.shape[1])
    n = pred.shape[0]
    out = {
        f"{column_prefix}{k}": np.zeros(n, dtype=bool)
        for k in range(1, max_k + 1)
    }
    if n == 0:
        return out

    n_centroids = truth_neighbors.shape[0]
    valid = (
        (pred >= 0)
        & (pred < n_centroids)
        & (truth >= 0)
        & (truth < n_centroids)
    )
    if not valid.any():
        return out

    valid_pred = pred[valid]
    valid_truth = truth[valid]
    neighbor_rows = truth_neighbors[valid_truth]  # shape (n_valid, max_k)

    for k in range(1, max_k + 1):
        in_set = (neighbor_rows[:, :k] == valid_pred[:, None]).any(axis=1)
        col = out[f"{column_prefix}{k}"]
        col[valid] = in_set

    return out
