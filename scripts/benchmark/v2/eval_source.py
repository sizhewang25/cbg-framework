"""Score a canonical CSV's CBG-friendliness from geographic topology + RTT quality.

Consumes a canonical-schema CSV (one row per `(vp, target, rtt_ms)` observation
— see sources/README.md) and scores each target over its *available* VPs only:
a VP counts for a target iff the CSV holds >= 1 RTT observation for the pair,
so sparse meshes are scored on what a benchmark run would actually see.

Multiple observations of the same (vp, target) pair are collapsed to the
min-RTT one before aggregation (same convention as inspect_source's flow map).

Per-target metrics, grouped by axis:

  availability  n_avail_vps          distinct VPs with >= 1 observation
  geography     closest_vp_km        min geodesic VP distance — the floor any
                                     measurement could achieve with ideal routing
                shortest_ping_vp_km  geodesic distance of the min-RTT VP — where
                                     the shortest-ping baseline would snap; equals
                                     closest_vp_km only when the fastest VP is
                                     also the geographically closest one
  RTT           min_rtt_ms           min RTT over available VPs
                best_radius_km       min_rtt_ms / THEORETICAL_SLOPE — the
                                     tightest CBG constraint radius the data
                                     supports (rtt/2 x 2/3c = rtt x 100 km)
                min_inflation        min over pairs of rtt / (slope x gc_km),
                                     routing efficiency decoupled from proximity
                                     (NaN when every pair is colocated)
  combined      rtt_weighted_dist_km inverse-RTT weighted mean VP distance,
                                     sum(gc/rtt) / sum(1/rtt) — the fleet's
                                     effective geographic distance as CBG
                                     experiences it: low-RTT VPs dominate,
                                     high-RTT VPs fade out. Bounded by
                                     [min gc, max gc]; parameter-free.

The dataset-level summary is a percentile block per metric plus a
"resolvability" table: the share of targets whose closest_vp_km /
best_radius_km / rtt_weighted_dist_km falls within each km threshold
(defaults match THRESHOLD_DISTANCES minus the exact-match 0).

The summary also sizes the dataset's *answer space*: targets are clustered
with `cluster_ground_truth` (complete-linkage, centroid radius capped at the
coherence radius R, default 50 km — the same construction as cluster-eval).
`target_clustering` reports the number of unique R-coherent regions the
targets occupy, plus two VP-proximity checks at two strictness levels, each
for both the closest VP and the min-RTT (shortest-ping) VP:

  strict   *_within_radius_share      VP within R km of the target — the CDF
                                      of closest_vp_km / shortest_ping_vp_km
                                      at the coherence radius.
  loose    *_in_same_cluster_share    VP Voronoi-assigned to the target's own
                                      cluster (its nearest answer-space
                                      centroid is the target's centroid) —
                                      proximity with Voronoi split tolerance,
                                      the same judgment cluster-score's
                                      shortest-ping baseline uses. A VP can
                                      clear this while sitting beyond R (deep
                                      inside a large cell) and, near a cluster
                                      boundary, can fail it while within R.

Each target's `cluster_id`, the identities/coords of its closest and min-RTT
VPs, and the two boolean same-cluster columns are carried in the per-target CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from scripts.benchmark.v2.sources.cluster_ground_truth import cluster_ground_truth
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE, haversine_distance

_REQUIRED = (
    "vp_id", "vp_lat", "vp_lon",
    "target_id", "target_lat", "target_lon",
    "rtt_ms",
)

DEFAULT_THRESHOLDS_KM: tuple[float, ...] = (40.0, 100.0, 500.0, 1000.0)

# Answer-space coherence radius R — matches cluster-eval's default cap.
DEFAULT_CLUSTER_RADIUS_KM = 50.0

PER_TARGET_METRICS = (
    "n_avail_vps",
    "closest_vp_km",
    "shortest_ping_vp_km",
    "min_rtt_ms",
    "best_radius_km",
    "min_inflation",
    "rtt_weighted_dist_km",
)

# Resolvability is reported on every distance-valued axis: the geography
# floor, the data-supported constraint radius, and the combined fleet view.
RESOLVABILITY_METRICS = ("closest_vp_km", "best_radius_km", "rtt_weighted_dist_km")

_PCTS = (5, 25, 50, 75, 95)


def load_canonical_csv(csv_path: Path) -> pd.DataFrame:
    """Load the required canonical columns, case-insensitively, dropping rows
    with missing values or non-positive RTTs (mirrors GenericCSVSource)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is not a canonical CSV — missing columns: {missing}"
        )
    df = df[list(_REQUIRED)].copy()
    for col in ("vp_id", "target_id"):
        df[col] = df[col].astype(str)
    for col in ("vp_lat", "vp_lon", "target_lat", "target_lon", "rtt_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(_REQUIRED))
    df = df[df["rtt_ms"] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"{csv_path}: no usable rows after dropping NaNs and non-positive RTTs"
        )
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (vp, target) pair — the min-RTT observation — with the
    derived per-pair geometry columns (gc_km, radius_km, inflation)."""
    pairs = (
        df.sort_values("rtt_ms", kind="stable")
        .drop_duplicates(["vp_id", "target_id"], keep="first")
        .reset_index(drop=True)
    )
    pairs["gc_km"] = haversine_distance(
        pairs["vp_lat"].to_numpy(), pairs["vp_lon"].to_numpy(),
        pairs["target_lat"].to_numpy(), pairs["target_lon"].to_numpy(),
    )
    pairs["radius_km"] = pairs["rtt_ms"] / THEORETICAL_SLOPE
    # Routing inflation vs the 2/3c physical floor; undefined for colocated
    # endpoints (same guard as partvp extract_features).
    ideal_ms = THEORETICAL_SLOPE * pairs["gc_km"]
    pairs["inflation"] = np.where(
        ideal_ms > 1e-9, pairs["rtt_ms"] / ideal_ms, np.nan
    )
    return pairs


def per_target_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    """Reduce the pair frame to one row per target (see module docstring)."""
    inv_rtt = 1.0 / pairs["rtt_ms"]
    work = pairs.assign(_w=inv_rtt, _wd=pairs["gc_km"] * inv_rtt)
    g = work.groupby("target_id", sort=True)
    # Identity + coords of the geographically closest VP and of the min-RTT
    # VP (the shortest-ping baseline's snap point) — the two need not agree.
    closest = pairs.loc[
        pairs.groupby("target_id")["gc_km"].idxmin()
    ].set_index("target_id")
    shortest_ping = pairs.loc[
        pairs.groupby("target_id")["rtt_ms"].idxmin()
    ].set_index("target_id")
    out = pd.DataFrame({
        "target_lat": g["target_lat"].first(),
        "target_lon": g["target_lon"].first(),
        "n_avail_vps": g["vp_id"].nunique(),
        "closest_vp_km": closest["gc_km"],
        "closest_vp_id": closest["vp_id"],
        "closest_vp_lat": closest["vp_lat"],
        "closest_vp_lon": closest["vp_lon"],
        "shortest_ping_vp_km": shortest_ping["gc_km"],
        "shortest_ping_vp_id": shortest_ping["vp_id"],
        "shortest_ping_vp_lat": shortest_ping["vp_lat"],
        "shortest_ping_vp_lon": shortest_ping["vp_lon"],
        "min_rtt_ms": g["rtt_ms"].min(),
        "best_radius_km": g["radius_km"].min(),
        "min_inflation": g["inflation"].min(),
        "rtt_weighted_dist_km": g["_wd"].sum() / g["_w"].sum(),
    })
    return out.reset_index()


def _stat_block(values: pd.Series) -> dict[str, Any]:
    v = values.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, _PCTS)
    return {
        "n": int(v.size),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
        "mean": round(float(v.mean()), 3),
        "percentiles": {f"p{p}": round(float(x), 3) for p, x in zip(_PCTS, q)},
    }


def summarize(
    per_target: pd.DataFrame, thresholds: Sequence[float]
) -> dict[str, Any]:
    resolvability = {
        metric: {
            f"within_{t:g}km": round(float((per_target[metric] <= t).mean()), 4)
            for t in thresholds
        }
        for metric in RESOLVABILITY_METRICS
    }
    return {
        "n_targets": int(len(per_target)),
        "metrics": {m: _stat_block(per_target[m]) for m in PER_TARGET_METRICS},
        "resolvability": resolvability,
    }


def cluster_targets(
    per_target: pd.DataFrame, radius_km: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster the targets into R-coherent regions (answer-space sizing).

    Returns the per-target frame with `cluster_id` and the two boolean
    same-cluster columns appended, plus the `target_clustering` summary block:
    the number of unique regions and, for both the closest VP and the min-RTT
    VP, the strict within-R share (CDF at the coherence radius) and the loose
    same-cluster share (VP's nearest answer-space centroid is the target's —
    Voronoi split tolerance)."""
    res = cluster_ground_truth(
        per_target["target_lat"].to_numpy(),
        per_target["target_lon"].to_numpy(),
        radius_km=radius_km,
    )
    labels = res.labels.astype(int)

    tree = BallTree(
        np.radians(np.column_stack([res.centroid_lat, res.centroid_lon])),
        metric="haversine",
    )

    def _nearest_cell(lat: pd.Series, lon: pd.Series) -> np.ndarray:
        coords = np.radians(np.column_stack([lat.to_numpy(dtype=float),
                                             lon.to_numpy(dtype=float)]))
        _, idx = tree.query(coords, k=1)
        return idx[:, 0]

    per_target = per_target.assign(
        cluster_id=labels,
        closest_vp_in_same_cluster=(
            _nearest_cell(per_target["closest_vp_lat"],
                          per_target["closest_vp_lon"]) == labels
        ),
        shortest_ping_vp_in_same_cluster=(
            _nearest_cell(per_target["shortest_ping_vp_lat"],
                          per_target["shortest_ping_vp_lon"]) == labels
        ),
    )
    block = {
        "radius_km": float(radius_km),
        "n_clusters": int(res.n_clusters),
        "targets_per_cluster": round(len(per_target) / res.n_clusters, 3),
        "closest_vp_within_radius_share": round(
            float((per_target["closest_vp_km"] <= radius_km).mean()), 4
        ),
        "shortest_ping_vp_within_radius_share": round(
            float((per_target["shortest_ping_vp_km"] <= radius_km).mean()), 4
        ),
        "closest_vp_in_same_cluster_share": round(
            float(per_target["closest_vp_in_same_cluster"].mean()), 4
        ),
        "shortest_ping_vp_in_same_cluster_share": round(
            float(per_target["shortest_ping_vp_in_same_cluster"].mean()), 4
        ),
    }
    return per_target, block


def eval_source(
    csv_path: Path,
    out_dir: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS_KM,
    cluster_radius_km: float = DEFAULT_CLUSTER_RADIUS_KM,
) -> dict[str, Any]:
    """Score one canonical CSV; write per-target CSV + stats JSON into out_dir
    (named after the CSV's stem) and return the stats dict with output paths."""
    df = load_canonical_csv(csv_path)
    pairs = build_pairs(df)
    per_target = per_target_metrics(pairs)
    per_target, clustering = cluster_targets(per_target, cluster_radius_km)

    stats: dict[str, Any] = {
        "csv": str(csv_path),
        "n_obs": int(len(df)),
        "n_pairs": int(len(pairs)),
        "n_vps": int(pairs["vp_id"].nunique()),
        **summarize(per_target, thresholds),
        "target_clustering": clustering,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    per_target_path = out_dir / f"{csv_path.stem}_eval_per_target.csv"
    stats_path = out_dir / f"{csv_path.stem}_eval_stats.json"
    per_target.to_csv(per_target_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    stats["per_target_csv"] = str(per_target_path)
    stats["stats_json"] = str(stats_path)
    return stats
