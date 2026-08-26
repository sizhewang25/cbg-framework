from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import QhullError, Voronoi

from scripts.libs.cbg.rtt_model import haversine_distance


def voronoi_adjacency_distances(lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    """Compute adjacent-neighbor distances from a Voronoi ridge graph.

    Returns one row per seed with:
      - adjacent_distances_km: full list of distances to adjacent cells
      - mean_dist_km: arithmetic mean over adjacent distances
      - median_dist_km: median over adjacent distances
    """
    n = len(lat)
    if n == 0:
        return pd.DataFrame(columns=["adjacent_distances_km", "mean_dist_km", "median_dist_km"])
    if n == 1:
        return pd.DataFrame(
            {
                "adjacent_distances_km": [[]],
                "mean_dist_km": [np.nan],
                "median_dist_km": [np.nan],
            }
        )
    if n == 2:
        dist = haversine_distance(lat[0], lon[0], lat[1], lon[1])
        d = float(dist)
        return pd.DataFrame(
            {
                "adjacent_distances_km": [[d], [d]],
                "mean_dist_km": [d, d],
                "median_dist_km": [d, d],
            }
        )

    points = np.column_stack([lon, lat]).astype(float)
    try:
        vor = Voronoi(points)
    except QhullError:
        # Degenerate point sets can fail Voronoi construction.
        return pd.DataFrame(
            {
                "adjacent_distances_km": [[] for _ in range(n)],
                "mean_dist_km": np.full(n, np.nan),
                "median_dist_km": np.full(n, np.nan),
            }
        )

    neighbors: list[set[int]] = [set() for _ in range(n)]
    for i, j in vor.ridge_points:
        neighbors[int(i)].add(int(j))
        neighbors[int(j)].add(int(i))

    adjacent_distances: list[list[float]] = []
    mean_adjacent = np.full(n, np.nan)
    median_adjacent = np.full(n, np.nan)
    for idx, adj in enumerate(neighbors):
        if not adj:
            adjacent_distances.append([])
            continue
        adj_idx = np.fromiter(sorted(adj), dtype=int)
        adj_dist = haversine_distance(
            lat[idx],
            lon[idx],
            lat[adj_idx],
            lon[adj_idx],
        )
        adj_dist_arr = np.asarray(adj_dist, dtype=float)
        adjacent_distances.append([float(x) for x in adj_dist_arr.tolist()])
        mean_adjacent[idx] = float(np.mean(adj_dist_arr))
        median_adjacent[idx] = float(np.median(adj_dist_arr))

    return pd.DataFrame(
        {
            "adjacent_distances_km": adjacent_distances,
            "mean_dist_km": mean_adjacent,
            "median_dist_km": median_adjacent,
        }
    )


def adjacency_concentration_from_representative(
    cluster_ids: np.ndarray,
    n_members: pd.Series,
    representative_dist_km: pd.Series,
) -> dict[str, Any]:
    """Compute concentration from a representative per-cluster distance.

    Uses per-cluster representative distances a_i (e.g., median adjacent
    distance), then computes:
      a_cluster = mean(a_i)
      a_wgt_cluster = sum(n_i * a_i) / sum(n_i)
            r_concentration = a_wgt_cluster / a_cluster
    """
    conc = pd.DataFrame(
        {
            "cluster_id": cluster_ids.astype(int),
            "a_i": representative_dist_km.to_numpy(dtype=float),
        }
    )

    n_members_map = n_members.copy()
    if "cluster_id" in n_members_map.index.names:
        n_members_map.index = n_members_map.index.astype(int)
    conc["n_i"] = conc["cluster_id"].map(n_members_map)

    a_vals = conc["a_i"].to_numpy(dtype=float)
    n_vals = conc["n_i"].to_numpy(dtype=float)
    valid = np.isfinite(a_vals) & np.isfinite(n_vals) & (n_vals > 0)

    if not np.any(valid):
        return {
            "a_cluster_mean_km": None,
            "a_cluster_std_km": None,
            "a_wgt_cluster_mean_km": None,
            "r_concentration": None,
            "tg_concentration_index": None,
            "valid_n_clusters": 0,
            "total_targets_used": 0,
        }

    a_valid = a_vals[valid]
    n_valid = n_vals[valid]
    total_targets = float(np.sum(n_valid))

    a_cluster = float(np.mean(a_valid))
    a_cluster_std = float(np.std(a_valid))
    a_wgt_cluster = float(np.sum(n_valid * a_valid) / total_targets)

    r_concentration = None
    if np.isfinite(a_cluster) and a_cluster > 0:
        r_concentration = float(a_wgt_cluster / a_cluster)

    target_concentration_ratio = None
    if r_concentration is not None:
        target_concentration_ratio = float(1.0 - r_concentration)

    return {
        "a_cluster_mean_km": round(a_cluster, 3),
        "a_cluster_std_km": round(a_cluster_std, 3),
        "a_wgt_cluster_mean_km": round(a_wgt_cluster, 3),
        "r_concentration": None if r_concentration is None else round(r_concentration, 6),
        "tg_concentration_index": (
            None if target_concentration_ratio is None else round(target_concentration_ratio, 6)
        ),
        "valid_n_clusters": int(np.sum(valid)),
        "total_targets_used": int(round(total_targets)),
    }
