"""Lean dataset easiness evaluator (v2).

This module keeps the same input and artifact flow as eval_source but emits only
metrics defined in notes/2026-07-20-eval-source-classification-easiness-design.md:

- vertex_props
- edge_props

Everything else from eval_source (legacy precheck summaries) is intentionally
excluded to avoid duplicated logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from scripts.benchmark.v2.eval_source import (
    DEFAULT_CLUSTER_RADIUS_KM,
    DEFAULT_TOP_N_NEIGHBORS,
    apply_eval_target_filters,
    anycast_metrics,
    build_pairs,
    cluster_targets,
    load_canonical_csv,
    per_target_metrics,
    proximity_summary,
    proximity_metrics,
    rtt_quality_summary,
    write_mesh,
)
from scripts.benchmark.v2.modules.adjacency_metrics import (
    adjacency_concentration_from_representative,
    voronoi_adjacency_distances,
)
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM, haversine_distance

_PCTS = (5, 25, 50, 75, 95)


def _stat_block(values: pd.Series) -> dict[str, Any]:
    """Summarize a numeric distribution with std and percentiles."""
    v = values.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, _PCTS)
    return {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "mean": round(float(v.mean()), 3),
        "std": round(float(v.std()), 3),
        "percentiles": {f"p{p}": round(float(x), 3) for p, x in zip(_PCTS, q)},
    }


def _weighted_stat_block(values: pd.Series, weights: pd.Series) -> dict[str, Any]:
    """Summarize a numeric distribution using non-negative sample weights."""
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]
    if v.size == 0:
        return {"n": 0}

    order = np.argsort(v)
    vs = v[order]
    ws = w[order]
    wsum = float(ws.sum())
    cdf = np.cumsum(ws) / wsum

    weighted_mean = float(np.sum(ws * vs) / wsum)
    weighted_var = float(np.sum(ws * (vs - weighted_mean) ** 2) / wsum)

    pct_vals = []
    for p in _PCTS:
        q = p / 100.0
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(idx, len(vs) - 1)
        pct_vals.append(vs[idx])

    return {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "mean": round(weighted_mean, 3),
        "std": round(float(np.sqrt(weighted_var)), 3),
        "percentiles": {f"p{p}": round(float(x), 3) for p, x in zip(_PCTS, pct_vals)},
    }


def _knn_mean_gap_distribution(lat: np.ndarray, lon: np.ndarray, *, k: int = 3) -> dict[str, Any]:
    """Average distance to k nearest neighbors for each node."""
    n = len(lat)
    if n < 2:
        return {"k": int(k), "distribution": {"n": 0}}

    coords = np.radians(np.column_stack([lat, lon]))
    tree = BallTree(coords, metric="haversine")
    used_k = int(min(k, n - 1))
    ndist, _ = tree.query(coords, k=used_k + 1)
    # Column 0 is self-distance, exclude it then mean over k neighbors.
    mean_gap_km = np.mean(ndist[:, 1:] * EARTH_RADIUS_KM, axis=1)
    return {"k": used_k, "distribution": _stat_block(pd.Series(mean_gap_km))}


def _classification_easiness_v2(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    per_target: pd.DataFrame,
    clusters: pd.DataFrame,
    cluster_radius_km: float,
) -> dict[str, Any]:
    """Build only the v2 easiness metrics contract."""
    target_cluster_map = per_target[["target_id", "cluster_id"]]
    pair_with_cluster = pairs.merge(target_cluster_map, on="target_id", how="left")

    # Count targets per city over unique targets; fallback to unique target
    # coordinates when city labels are unavailable.
    if "target_norm_city" in df.columns:
        target_city_map = df[["target_id", "target_norm_city"]].drop_duplicates("target_id")
        city_counts = (
            target_city_map.dropna(subset=["target_norm_city"])
            .groupby("target_norm_city", sort=True)["target_id"]
            .nunique()
        )
    elif "target_city" in df.columns:
        target_city_map = df[["target_id", "target_city"]].drop_duplicates("target_id")
        city_counts = (
            target_city_map.dropna(subset=["target_city"])
            .groupby("target_city", sort=True)["target_id"]
            .nunique()
        )
    else:
        target_city_map = (
            df[["target_id", "target_lat", "target_lon"]]
            .drop_duplicates("target_id")
            .copy()
        )
        target_city_map["target_city_fallback"] = (
            target_city_map["target_lat"].round(6).astype(str)
            + "|"
            + target_city_map["target_lon"].round(6).astype(str)
        )
        city_counts = target_city_map.groupby("target_city_fallback", sort=True)["target_id"].nunique()

    cluster_degree = pair_with_cluster.groupby("cluster_id", sort=True)["vp_id"].nunique()
    vp_degree = pair_with_cluster.groupby("vp_id", sort=True)["cluster_id"].nunique()

    target_asn_count = None
    if "target_asn" in df.columns:
        target_asn_count = int(df["target_asn"].dropna().astype(str).nunique())

    # VPs: kNN gap on unique VP coordinates.
    vp_unique = pairs[["vp_id", "vp_lat", "vp_lon"]].drop_duplicates("vp_id")
    vp_knn = _knn_mean_gap_distribution(
        vp_unique["vp_lat"].to_numpy(dtype=float),
        vp_unique["vp_lon"].to_numpy(dtype=float),
        k=1,
    )

    # Target clusters: kNN gap on cluster centroids.
    cluster_unique = clusters[["cluster_id", "centroid_lat", "centroid_lon"]].drop_duplicates("cluster_id")
    cluster_knn = _knn_mean_gap_distribution(
        cluster_unique["centroid_lat"].to_numpy(dtype=float),
        cluster_unique["centroid_lon"].to_numpy(dtype=float),
        k=1,
    )
    tg_cluster_voronoi_cell_distance = voronoi_adjacency_distances(
        cluster_unique["centroid_lat"].to_numpy(dtype=float),
        cluster_unique["centroid_lon"].to_numpy(dtype=float),
    )
    tg_cluster_voronoi_concentration = adjacency_concentration_from_representative(
        cluster_unique["cluster_id"].to_numpy(dtype=int),
        clusters.set_index("cluster_id")["n_members"],
        tg_cluster_voronoi_cell_distance["median_dist_km"],
    )
    tg_cluster_voronoi_cell_distance_mean = _stat_block(
        tg_cluster_voronoi_cell_distance["mean_dist_km"]
    )
    tg_cluster_voronoi_cell_distance_median = _stat_block(
        tg_cluster_voronoi_cell_distance["median_dist_km"]
    )

    # VP-target-cluster edge set (unique cluster-vp links).
    cluster_centroids = clusters[["cluster_id", "centroid_lat", "centroid_lon"]]
    cluster_vp = (
        pair_with_cluster[["cluster_id", "vp_id", "vp_lat", "vp_lon"]]
        .drop_duplicates(["cluster_id", "vp_id"])
        .merge(cluster_centroids, on="cluster_id", how="left")
    )
    if len(cluster_vp):
        cluster_vp["vp_to_cluster_centroid_km"] = haversine_distance(
            cluster_vp["vp_lat"].to_numpy(dtype=float),
            cluster_vp["vp_lon"].to_numpy(dtype=float),
            cluster_vp["centroid_lat"].to_numpy(dtype=float),
            cluster_vp["centroid_lon"].to_numpy(dtype=float),
        )

    # Aggregate traffic using unique (vp_id, target_lat_lon) contributions per
    # cluster, then map to unique (cluster_id, vp_id) edges.
    if {"target_lat", "target_lon"}.issubset(pair_with_cluster.columns):
        loc_key = (
            pair_with_cluster["target_lat"].round(6).astype(str)
            + "|"
            + pair_with_cluster["target_lon"].round(6).astype(str)
        )
    else:
        loc_key = pair_with_cluster["target_id"].astype(str)

    pair_with_cluster = pair_with_cluster.assign(_loc_key=loc_key)
    city_level = (
        pair_with_cluster.groupby(["cluster_id", "vp_id", "_loc_key"], sort=False)["weight"]
        .max()
        .reset_index()
    )
    edge_weights = (
        city_level.groupby(["cluster_id", "vp_id"], sort=False)["weight"]
        .sum()
        .rename("edge_traffic_weight")
        .reset_index()
    )
    cluster_vp = cluster_vp.merge(edge_weights, on=["cluster_id", "vp_id"], how="left")
    tw_dist = (
        _weighted_stat_block(cluster_vp["vp_to_cluster_centroid_km"], cluster_vp["edge_traffic_weight"])
        if len(cluster_vp)
        else {"n": 0}
    )

    return {
        "vertex_props": {
            "targets": {
                "n_unique": int(df["target_id"].nunique()),
                "targets_per_city_distribution": _stat_block(city_counts),
                "n_unique_target_asns": target_asn_count,
            },
            "target_clusters": {
                "n_unique": int(clusters["cluster_id"].nunique()),
                "n_singletons": int(clusters["is_singleton"].sum()),
                "cluster_radius": round(float(cluster_radius_km), 3),
                "targets_per_cluster_distribution": _stat_block(clusters["n_members"]),
                "tg_cluster_voronoi_cell_distance_distribution": {
                    "n": int(len(tg_cluster_voronoi_cell_distance)),
                    "per_cluster": [
                        {
                            "cluster_id": int(cluster_id),
                            "adjacent_distances_km": distances,
                            "mean_dist_km": None if pd.isna(mean_dist) else round(float(mean_dist), 3),
                            "median_dist_km": None if pd.isna(median_dist) else round(float(median_dist), 3),
                        }
                        for cluster_id, distances, mean_dist, median_dist in zip(
                            cluster_unique["cluster_id"].to_numpy(dtype=int),
                            tg_cluster_voronoi_cell_distance["adjacent_distances_km"].tolist(),
                            tg_cluster_voronoi_cell_distance["mean_dist_km"].to_numpy(dtype=float),
                            tg_cluster_voronoi_cell_distance["median_dist_km"].to_numpy(dtype=float),
                        )
                    ],
                    "mean_dist": tg_cluster_voronoi_cell_distance_mean,
                    "median_dist": tg_cluster_voronoi_cell_distance_median,
                    "adjacency_concentration": tg_cluster_voronoi_concentration,
                },
                "tg_cluster_degree_wrt_vp_distribution": _stat_block(cluster_degree),
                "knn_gap_km_distribution": cluster_knn,
            },
            "vps": {
                "n_unique": int(df["vp_id"].nunique()),
                "vp_degree_wrt_tg_cluster_distribution": _stat_block(vp_degree),
                "knn_gap_km_distribution": vp_knn,
            },
        },
        "edge_props": {
            "vp_target_clusters": {
                "n_unique_edges": int(len(cluster_vp)),
                "geography_edge_distance_km_distribution": (
                    _stat_block(cluster_vp["vp_to_cluster_centroid_km"])
                    if len(cluster_vp)
                    else {"n": 0}
                ),
                "traffic_weighted_edge_distance_km_distribution": tw_dist,
                "proximity": proximity_summary(per_target),
                "min_rtt_inflation_distribution": _stat_block(per_target["min_inflation"]),
                "rtt_quality": rtt_quality_summary(pairs, per_target),
            },
        },
    }


def eval_source_v2(
    csv_path: Path,
    out_dir: Path,
    cluster_radius_km: float = DEFAULT_CLUSTER_RADIUS_KM,
    *,
    top_n_neighbors: int = DEFAULT_TOP_N_NEIGHBORS,
    spearman_min_pairs: int = 8,
    min_obs: int | None = None,
    eval_pair_weight_min: float | None = None,
    eval_kept_traffic_fraction: float | None = None,
) -> dict[str, Any]:
    """Score one canonical CSV and emit only the v2 easiness summary."""
    df = load_canonical_csv(csv_path)

    # Prefer explicit city labels when available; otherwise fall back to
    # unique coordinate locations as city-level proxies.
    if "vp_norm_city" in df.columns:
        n_vp_city = int(df["vp_norm_city"].dropna().astype(str).nunique())
    elif "vp_city" in df.columns:
        n_vp_city = int(df["vp_city"].dropna().astype(str).nunique())
    else:
        n_vp_city = int(df[["vp_lat", "vp_lon"]].drop_duplicates().shape[0])

    if "target_norm_city" in df.columns:
        n_tg_city = int(df["target_norm_city"].dropna().astype(str).nunique())
    elif "target_city" in df.columns:
        n_tg_city = int(df["target_city"].dropna().astype(str).nunique())
    else:
        n_tg_city = int(df[["target_lat", "target_lon"]].drop_duplicates().shape[0])

    filters_applied = (
        min_obs is not None
        or eval_pair_weight_min is not None
        or eval_kept_traffic_fraction is not None
    )
    resolved_eval_pair_weight_min = eval_pair_weight_min
    if filters_applied:
        df, resolved_eval_pair_weight_min = apply_eval_target_filters(
            df,
            min_obs=min_obs,
            eval_pair_weight_min=eval_pair_weight_min,
            eval_kept_traffic_fraction=eval_kept_traffic_fraction,
        )

    pairs = build_pairs(df)
    per_target = per_target_metrics(pairs, spearman_min_pairs=spearman_min_pairs)

    clusters_dir = out_dir / f"{csv_path.stem}_clusters"
    per_target, clusters, _ = cluster_targets(
        per_target,
        radius_km=cluster_radius_km,
        top_n_neighbors=top_n_neighbors,
        write_dir=clusters_dir,
    )

    # Adds closest_vp_to_centroid_km needed by vp_proximity_per_cluster_share.
    per_target = proximity_metrics(pairs, per_target)
    per_target = per_target.merge(anycast_metrics(pairs), on="target_id")

    stats: dict[str, Any] = {
        "csv": str(csv_path),
        "n_pairs": int(len(pairs)),
        "n_vp_city": n_vp_city,
        "n_tg_city": n_tg_city,
        **_classification_easiness_v2(df, pairs, per_target, clusters, cluster_radius_km),
    }

    if filters_applied:
        stats["eval_filters"] = {
            "min_obs": min_obs,
            "eval_pair_weight_min": resolved_eval_pair_weight_min,
            "eval_kept_traffic_fraction": eval_kept_traffic_fraction,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    per_target_path = out_dir / f"{csv_path.stem}_eval_per_target.csv"
    clusters_path = out_dir / f"{csv_path.stem}_eval_clusters.csv"
    stats_path = out_dir / f"{csv_path.stem}_eval_stats.json"

    per_target.to_csv(per_target_path, index=False)
    clusters.to_csv(clusters_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    stats["per_target_csv"] = str(per_target_path)
    stats["clusters_csv"] = str(clusters_path)
    stats["stats_json"] = str(stats_path)
    stats["clusters_dir"] = str(clusters_dir)

    vp_mesh = write_mesh(
        pairs,
        "vp_id",
        "vp_lat",
        "vp_lon",
        out_dir / f"{csv_path.stem}_vp_mesh_km.csv",
    )
    cl_mesh = write_mesh(
        clusters,
        "cluster_id",
        "centroid_lat",
        "centroid_lon",
        out_dir / f"{csv_path.stem}_cluster_mesh_km.csv",
    )
    stats["vp_mesh_csv"] = str(vp_mesh)
    stats["cluster_mesh_csv"] = str(cl_mesh)
    return stats
