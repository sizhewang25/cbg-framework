"""Cluster ground-truth coordinates into the finite answer space for CBG.

The CBG accuracy validation is reframed as **classification over a finite answer
space**: many hosts encode an IATA-style locality in their rDNS PTR, so a
coordinate's *region* — not its exact lat/lon — is the meaningful unit. We build
that answer space directly from all available ground-truth coordinates (the
anchors / targets) by grouping them into **coherent regions**, each fitting
inside a circle of radius ``R`` (default 50 km). Each region's spherical
centroid becomes one answer-space point.

A prediction is then scored as *exactly correct* (judged downstream, not here)
when the CBG multilateration region intersects the cluster's region, or a single
point estimate falls within ``R`` of the cluster centroid. Coordinates with no
nearby neighbour fall out as **singleton clusters** (radius 0) — the "no nearby
ground truth" case, where the nearest ground truth is taken as the prediction
with the error distance reported as a confidence.

Clustering method — complete-linkage agglomerative, capped on centroid radius
-----------------------------------------------------------------------------
k-means is the wrong tool for a radius-bounded partition: its objective is
within-cluster variance for a *fixed k*, with no notion of a radius, so a cap
can only be bolted on by searching k (non-monotone, over-splits dense areas, and
forces singletons by brute force). Instead we use **agglomerative clustering
with complete linkage** over a precomputed **haversine** distance matrix —
deterministic, no ``k``, and isolated points drop out as singletons naturally.

The quantity that must be bounded is the one the scoring rule uses: each
region's **centroid radius** (max member→centroid distance ≤ ``R``), so the
centroid is a faithful representative within ``R`` of every member. Complete
linkage natively bounds the *diameter*, not the centroid radius, and the two
diverge — a region of diameter ``2R`` can have a (mean) centroid radius well
above ``R`` (observed ~72 km at ``R``=50). So we do it in two deterministic
stages:

  1. complete-linkage at ``distance_threshold = 2R`` builds the largest
     candidate regions that *could* satisfy the cap (centroid radius ≤ ``R`` ⇒
     diameter ≤ ``2R``);
  2. any candidate whose centroid radius still exceeds ``R`` is recursively
     bisected (2-way complete linkage) until every region complies —
     guaranteed to terminate, since singletons have radius 0.

This keeps regions maximal while hard-guaranteeing centroid radius ≤ ``R``.
Centroids are spherical means (normalized mean unit vector back to lat/lon,
antimeridian-safe); all distances are haversine.

Inputs come from a `dump_csv_targets` output (or any CSV/JSON carrying
``target_id``/``target_lat``/``target_lon``).

CLI::

    python -m scripts.benchmark.v2.sources.cluster_ground_truth \\
        --targets datasets/ripe_atlas/asn_corpora/targets.csv --radius-km 50

Outputs (default ``datasets/<targets-stem>/clusters/``):
    clusters.csv      one row per region (the answer space): cluster_id,
                      centroid_lat, centroid_lon, n_members, radius_km,
                      diameter_km, is_singleton
    assignments.csv   one row per input coordinate: target_id, target_lat,
                      target_lon, cluster_id, dist_to_centroid_km
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

logger = logging.getLogger(__name__)

_DEFAULT_RADIUS_KM = 50.0


def _to_unit_vectors(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """(lat, lon) degrees → Nx3 unit vectors on the sphere."""
    la, lo = np.radians(lats), np.radians(lons)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _spherical_centroid(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    """Great-circle centroid: mean unit vector, renormalized, back to lat/lon."""
    v = _to_unit_vectors(lats, lons).mean(axis=0)
    norm = np.linalg.norm(v)
    if norm == 0:  # antipodal cancellation — fall back to the first point
        return float(lats[0]), float(lons[0])
    v /= norm
    return float(np.degrees(np.arcsin(v[2]))), float(np.degrees(np.arctan2(v[1], v[0])))


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance (km) between aligned coordinate arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Full NxN great-circle distance matrix (km) for precomputed clustering."""
    la = np.radians(lats)[:, None]
    lo = np.radians(lons)[:, None]
    dlat = la - la.T
    dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class ClusterResult:
    """The answer space and per-point assignment for one radius cap."""
    labels: np.ndarray          # per-input cluster id (0..n_clusters-1)
    centroid_lat: np.ndarray    # per-cluster
    centroid_lon: np.ndarray    # per-cluster
    member_counts: np.ndarray   # per-cluster
    radius_km: np.ndarray       # per-cluster max member→centroid distance
    diameter_km: np.ndarray     # per-cluster max pairwise distance (linkage bound)
    dist_km: np.ndarray         # per-input distance to its own centroid
    radius_target_km: float     # the requested cap R
    n_clusters: int


def _label_geometry(
    lats: np.ndarray, lons: np.ndarray, labels: np.ndarray, dist_matrix: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster spherical centroids, counts, centroid-radius and diameter, plus
    the per-input distance to its assigned centroid. Labels are remapped to a
    dense ``0..k-1`` range."""
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    dense = np.array([remap[v] for v in labels])
    k = len(uniq)

    c_lat = np.empty(k)
    c_lon = np.empty(k)
    counts = np.empty(k, dtype=int)
    radius = np.empty(k)
    diameter = np.empty(k)
    dist = np.empty(len(labels))
    for c in range(k):
        m = dense == c
        idx = np.where(m)[0]
        clat, clon = _spherical_centroid(lats[m], lons[m])
        d = _haversine_km(lats[m], lons[m], clat, clon)
        c_lat[c], c_lon[c] = clat, clon
        counts[c] = int(m.sum())
        radius[c] = float(d.max())
        dist[m] = d
        if len(idx) > 1 and dist_matrix is not None:
            diameter[c] = float(dist_matrix[np.ix_(idx, idx)].max())
        else:
            diameter[c] = 0.0
    return dense, c_lat, c_lon, counts, radius, diameter, dist


def _centroid_radius(lats: np.ndarray, lons: np.ndarray, idx: np.ndarray) -> float:
    """Max member→spherical-centroid distance (km) for the points at ``idx``."""
    clat, clon = _spherical_centroid(lats[idx], lons[idx])
    return float(_haversine_km(lats[idx], lons[idx], clat, clon).max())


def _split_to_radius(
    idx: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    dist_matrix: np.ndarray, radius_km: float,
) -> list[np.ndarray]:
    """Recursively bisect a candidate region (2-way complete linkage) until every
    sub-region's centroid radius ≤ ``radius_km``. Terminates at singletons."""
    if len(idx) <= 1 or _centroid_radius(lats, lons, idx) <= radius_km:
        return [idx]
    from sklearn.cluster import AgglomerativeClustering

    sub = dist_matrix[np.ix_(idx, idx)]
    parts = AgglomerativeClustering(
        n_clusters=2, metric="precomputed", linkage="complete",
    ).fit_predict(sub)
    out: list[np.ndarray] = []
    for c in (0, 1):
        out.extend(_split_to_radius(idx[parts == c], lats, lons, dist_matrix, radius_km))
    return out


def cluster_ground_truth(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    radius_km: float = _DEFAULT_RADIUS_KM,
) -> ClusterResult:
    """Group coordinates so every region's centroid radius ≤ ``radius_km``.

    Deterministic: complete-linkage agglomerative (diameter ≤ ``2R``) builds the
    largest candidate regions, then `_split_to_radius` bisects any region that
    exceeds the centroid-radius cap. Isolated points become singletons."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    n = len(lats)
    if n == 0:
        raise ValueError("no coordinates to cluster")

    if n == 1:
        labels = np.array([0])
        dist_matrix = None
    else:
        from sklearn.cluster import AgglomerativeClustering

        dist_matrix = _haversine_matrix(lats, lons)
        base = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=2.0 * radius_km,
        ).fit_predict(dist_matrix)

        labels = np.empty(n, dtype=int)
        next_label = 0
        for c in np.unique(base):
            idx = np.where(base == c)[0]
            for group in _split_to_radius(idx, lats, lons, dist_matrix, radius_km):
                labels[group] = next_label
                next_label += 1

    dense, c_lat, c_lon, counts, radius, diameter, dist = _label_geometry(
        lats, lons, labels, dist_matrix
    )
    return ClusterResult(
        labels=dense,
        centroid_lat=c_lat,
        centroid_lon=c_lon,
        member_counts=counts,
        radius_km=radius,
        diameter_km=diameter,
        dist_km=dist,
        radius_target_km=float(radius_km),
        n_clusters=len(c_lat),
    )


def _load_targets(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        df = pd.read_csv(path)
    missing = [c for c in ("target_lat", "target_lon") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns {missing}; expected target coords")
    df = df.dropna(subset=["target_lat", "target_lon"]).copy()
    df["target_lat"] = df["target_lat"].astype(float)
    df["target_lon"] = df["target_lon"].astype(float)
    if "target_id" not in df.columns:
        df["target_id"] = [f"t{i}" for i in range(len(df))]
    return df.reset_index(drop=True)


def _write_outputs(df: pd.DataFrame, res: ClusterResult, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters = pd.DataFrame({
        "cluster_id": np.arange(res.n_clusters),
        "centroid_lat": res.centroid_lat,
        "centroid_lon": res.centroid_lon,
        "n_members": res.member_counts,
        "radius_km": res.radius_km.round(3),
        "diameter_km": res.diameter_km.round(3),
        "is_singleton": res.member_counts == 1,
    })
    assignments = pd.DataFrame({
        "target_id": df["target_id"].to_numpy(),
        "target_lat": df["target_lat"].to_numpy(),
        "target_lon": df["target_lon"].to_numpy(),
        "cluster_id": res.labels,
        "dist_to_centroid_km": res.dist_km.round(3),
    })
    clusters_path = out_dir / "clusters.csv"
    assignments_path = out_dir / "assignments.csv"
    clusters.to_csv(clusters_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    return clusters_path, assignments_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--targets", type=Path,
        default=Path("datasets/ripe_atlas/asn_corpora/targets.csv"),
        help="Ground-truth file (csv/json) with target_id/target_lat/target_lon.",
    )
    parser.add_argument("--radius-km", type=float, default=_DEFAULT_RADIUS_KM,
                        help="Region radius R; complete-linkage caps diameter at 2R. Default 50.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir. Defaults to datasets/<targets-stem>/clusters/.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.targets.exists():
        raise SystemExit(f"targets file not found at {args.targets}")

    df = _load_targets(args.targets)
    res = cluster_ground_truth(
        df["target_lat"].to_numpy(), df["target_lon"].to_numpy(),
        radius_km=args.radius_km,
    )

    out_dir = args.out_dir or (args.targets.parent / "clusters")
    clusters_path, assignments_path = _write_outputs(df, res, out_dir)

    n = len(df)
    singletons = int((res.member_counts == 1).sum())
    logger.info("clustered %d coords → %d regions (complete-linkage, centroid radius ≤ %.0f km)",
                n, res.n_clusters, res.radius_target_km)
    logger.info("  reduction:    %d → %d  (%.1fx)", n, res.n_clusters, n / res.n_clusters)
    logger.info("  singletons:   %d  (%.1f%% of regions, %.1f%% of coords)",
                singletons, 100 * singletons / res.n_clusters, 100 * singletons / n)
    logger.info("  members/region: max %d, median %.0f",
                int(res.member_counts.max()), float(np.median(res.member_counts)))
    logger.info("  centroid radius_km: max %.1f, p95 %.1f, median %.1f",
                float(res.radius_km.max()), float(np.percentile(res.radius_km, 95)),
                float(np.median(res.radius_km)))
    logger.info("  diameter_km:  max %.1f, p95 %.1f",
                float(res.diameter_km.max()), float(np.percentile(res.diameter_km, 95)))
    logger.info("wrote %s", clusters_path)
    logger.info("wrote %s", assignments_path)


if __name__ == "__main__":
    main()
