"""Score a canonical CSV's CBG-friendliness from geographic topology + RTT quality.

Consumes a canonical-schema CSV (one row per `(vp, target, rtt_ms)` observation
— see sources/README.md) and scores each target over its *available* VPs only:
a VP counts for a target iff the CSV holds >= 1 RTT observation for the pair,
so sparse meshes are scored on what a benchmark run would actually see.

This is the dataset *precheck*: it runs on the full CSV (never per-fold, never
on benchmark outputs) and answers "how well could shortest-ping / CBG work if
the whole dataset were the evaluation set", with every property made explicit.
"Whole dataset" defaults to every row, but when the benchmark's own
`source_kwargs.min_obs` or top-level `eval_pair_weight_min` /
`eval_kept_traffic_fraction` yaml keys are set, the *actual* eval set a
GenericCSVSource/GenericPresplitSource run would see is already a filtered
subset — pass the matching `min_obs`/`eval_pair_weight_min`/
`eval_kept_traffic_fraction` kwargs (see `apply_eval_target_filters`) so the
precheck scores that subset instead of silently diverging from it. The
resolved filters are recorded in the stats JSON's `eval_filters` block.

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
                min_inflation        min over pairs of rtt / (slope x gc_km),
                                     routing efficiency decoupled from proximity
                                     (NaN when every pair is colocated)
  combined      rtt_weighted_dist_km inverse-RTT weighted mean VP distance,
                                     sum(gc/rtt) / sum(1/rtt) — the fleet's
                                     effective geographic distance as CBG
                                     experiences it: low-RTT VPs dominate,
                                     high-RTT VPs fade out. Bounded by
                                     [min gc, max gc]; parameter-free.

Answer-space topology (coherence radius R, `cluster_ground_truth` — the same
construction as cluster-eval; R=0 degenerates to one singleton per target):

  cell_gap_km                       distance from the target's truth centroid
                                    to the nearest *other* centroid (inf when
                                    the answer space has a single cluster)
  target_distinguishable_vp_dist_km cell_gap_km / 2 — the direction-free bound:
                                    a VP within it is guaranteed to favor the
                                    truth centroid over any competitor
  closest_vp_to_centroid_km /       VP proximity measured to the truth cluster
  shortest_ping_vp_to_centroid_km   centroid (fleet-geometry semantics)

Discriminative set D = {VPs with vp_to_centroid < cell_gap/2} drives the
proximity label ladder (mutually exclusive; the HAS_* labels split the
has-proximity population by whether the baseline uses it; Voronoi assignment
never moves a label — it is a diagnostic within NO_PROXIMITY):

  NO_PROXIMITY            no VP meets the cell_gap/2 requirement (D empty) —
                          no proximity method can resolve the target; only
                          multilateration can help;
                          `shortest_ping_vp_in_same_cluster` splits it into
                          "baseline lands correctly by directional luck" vs
                          "baseline provably snaps to a non-truth cluster"
  HAS_NOT_USED_PROXIMITY  proximate VPs exist but the shortest-ping VP is not
                          one of them — winnable by better VP selection or
                          by multilateration
  HAS_USED_PROXIMITY      the shortest-ping VP is one of the proximate VPs —
                          guaranteed correct by the bound

  n_discriminative_vps / has_vp_proximity / shortest_ping_vp_is_discriminative
  best_discriminative_rtt_rank      min normalized RTT rank over D — how
                                    buried the discriminative set is in the
                                    target's RTT ordering (NaN when D empty)

RTT quality / regime diagnostics:

  rtt_dist_spearman                 Spearman rho(gc_km, rtt_ms) over the
                                    target's pairs — global distance-RTT
                                    coherence (NaN below spearman_min_pairs
                                    or under zero variance)
  closest_vp_rtt_rank               normalized RTT rank of the geographically
                                    closest VP (0 = it is also the fastest)
  closest_is_shortest_ping /        the closest-vs-fastest disagreement flag
  closest_to_shortest_ping_km       and the VP-to-VP separation behind it —
                                    a large separation means RTT inflation
                                    (congestion) or indirect routing at the
                                    close VP
  soi_violation_share               share of the target's pairs with
                                    rtt < slope x gc (faster than 2/3c ⇒ bad
                                    ground truth on an endpoint, or anycast)
  vp_pair_disk_overlap_km                 min over low-RTT VP pairs (rtt <= min_rtt
                                    + anycast_delta_ms) of r_i + r_j - d(vp_i,
                                    vp_j); negative ⇒ two disjoint constraint
                                    disks — physically impossible for unicast
                                    and a predictor of empty-intersection MTL
                                    failures (NaN with < 2 low-RTT VPs)
  n_disjoint_sites                  iGreedy-style greedy count of mutually
                                    disjoint low-RTT disks; >= 2 flags
                                    anycast_suspect via vp_pair_disk_overlap < 0
                                    (the exact pair test; the greedy count
                                    approximates the number of visible sites)

The dataset-level summary keeps the percentile block per metric and adds
`target_clustering` (answer-space sizing + the strict within-R / loose
same-cluster VP shares), `proximity` (label shares;
`cbg_opportunity_share` = NO_PROXIMITY + HAS_NOT_USED_PROXIMITY — the
population where the shortest-ping baseline carries no correctness
guarantee, so CBG multilateration or better VP selection could improve —
with the fraction of it the baseline converts anyway by Voronoi
assignment), and `rtt_quality` (dataset-level SOI share, anycast-suspect
share, closest-is-fastest agreement share).

Artifacts (named after the CSV stem):
  <stem>_eval_per_target.csv   per-target metrics + labels
  <stem>_eval_clusters.csv     per-cluster: centroid, members, radius,
                               cell_gap_km, top-N neighbor ids + distances
  <stem>_vp_mesh_km.csv /      great-circle distance matrices (id-indexed;
  <stem>_cluster_mesh_km.csv   skipped above mesh_max_n endpoints). The
                               answer-space mesh is over cluster centroids,
                               not raw targets — targets can be million-scale
                               while centroids stay bounded.
  <stem>_eval_stats.json       the dataset summary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import BallTree

from scripts.benchmark.v2.sources.cluster_ground_truth import (
    _write_outputs,
    cluster_ground_truth,
)
from scripts.benchmark.v2.sources.generic_csv import _raw_str
from scripts.libs.cbg.rtt_model import (
    EARTH_RADIUS_KM,
    THEORETICAL_SLOPE,
    haversine_distance,
)

_REQUIRED = (
    "vp_id", "vp_lat", "vp_lon",
    "target_id", "target_lat", "target_lon",
    "rtt_ms",
)

# Kept alongside the required columns, when present, so the eval-side filters
# below (min_obs / eval_pair_weight_min / eval_kept_traffic_fraction) can be
# applied identically to how GenericCSVSource/GenericPresplitSource do it at
# materialize time. `_raw_str` opts target_city out of pandas' NA-sentinel
# coercion, same reason as generic_csv.py's `_OPTIONAL_STR`.
_OPTIONAL_FOR_FILTERS = ("weight", "target_city")

# Answer-space coherence radius R — matches cluster-eval's default cap.
DEFAULT_CLUSTER_RADIUS_KM = 50.0

# Neighbor centroids recorded per cluster (self excluded).
DEFAULT_TOP_N_NEIGHBORS = 5

# Below this many pairs the Spearman coherence is not computed.
DEFAULT_SPEARMAN_MIN_PAIRS = 8

# Low-RTT VP set for the anycast / infeasibility test: rtt <= min_rtt + delta.
DEFAULT_ANYCAST_DELTA_MS = 10.0

# Distance-mesh artifacts are skipped past this many endpoints (O(n^2) cells).
DEFAULT_MESH_MAX_N = 2000

PROXIMITY_LABELS = (
    "NO_PROXIMITY",
    "HAS_NOT_USED_PROXIMITY",
    "HAS_USED_PROXIMITY",
)

PER_TARGET_METRICS = (
    "n_avail_vps",
    "closest_vp_km",
    "shortest_ping_vp_km",
    "min_rtt_ms",
    "min_inflation",
    "rtt_weighted_dist_km",
    # answer-space topology / proximity
    "cell_gap_km",
    "closest_vp_to_centroid_km",
    "shortest_ping_vp_to_centroid_km",
    "n_discriminative_vps",
    "best_discriminative_rtt_rank",
    # RTT quality / regimes
    "rtt_dist_spearman",
    "closest_vp_rtt_rank",
    "closest_to_shortest_ping_km",
    "soi_violation_share",
    "vp_pair_disk_overlap_km",
    "n_disjoint_sites",
)

_PCTS = (5, 25, 50, 75, 95)


def load_canonical_csv(csv_path: Path) -> pd.DataFrame:
    """Load the required canonical columns, case-insensitively, dropping rows
    with missing values or non-positive RTTs (mirrors GenericCSVSource).

    Also keeps `weight` (normalized to a numeric >=0 column, defaulting to
    1.0 when absent — same two-default convention as generic_csv.py) and
    `target_city`, when either is present in the CSV, so
    `apply_eval_target_filters` below can reproduce the materialize-time
    eval-side filters."""
    converters = {c: _raw_str for c in ("target_city", "TARGET_CITY")}
    df = pd.read_csv(csv_path, converters=converters)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is not a canonical CSV — missing columns: {missing}"
        )
    keep = list(_REQUIRED) + [c for c in _OPTIONAL_FOR_FILTERS if c in df.columns]
    df = df[keep].copy()
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
    if "weight" not in df.columns:
        df["weight"] = 1.0
    else:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        if (df["weight"] < 0).any():
            raise ValueError(f"{csv_path}: weight must be >= 0")
    return df


def apply_eval_target_filters(
    df: pd.DataFrame,
    *,
    min_obs: int | None = None,
    eval_pair_weight_min: float | None = None,
    eval_kept_traffic_fraction: float | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """Restrict `df` to the rows a real benchmark run's eval_observations.parquet
    would actually contain, mirroring GenericCSVSource/GenericPresplitSource's
    materialize-time eval-side filters (see sources/generic_csv.py's
    `_apply_min_obs_filter` / `_apply_eval_weight_filter` /
    `_derive_eval_weight_min_from_fraction`). Without this, the precheck
    silently scores every row in the CSV even when `source_kwargs.min_obs` or
    the top-level `eval_pair_weight_min` / `eval_kept_traffic_fraction` yaml
    keys shrink what the benchmark actually evaluates.

    `min_obs` drops targets with fewer than that many rows (raw CSV row
    count, matching the source classes' per-target-id row count). Then, at
    most one of `eval_pair_weight_min` / `eval_kept_traffic_fraction` narrows
    to eval-surviving obs: a target survives iff >= 1 of its rows has
    `weight >= threshold`, and only rows clearing the threshold are kept for
    surviving targets — exactly what `iter_eval_targets` would emit.

    Returns (filtered_df, resolved_eval_pair_weight_min) — the second value
    is the threshold actually used (derived from `eval_kept_traffic_fraction`
    when that's what was passed), so callers can record it for transparency.
    """
    if eval_pair_weight_min is not None and eval_kept_traffic_fraction is not None:
        raise ValueError(
            "pass only one of eval_pair_weight_min or eval_kept_traffic_fraction"
        )
    if eval_pair_weight_min is not None and eval_pair_weight_min < 0:
        raise ValueError(f"eval_pair_weight_min must be >= 0, got {eval_pair_weight_min}")
    if eval_kept_traffic_fraction is not None and not (0 < eval_kept_traffic_fraction <= 1):
        raise ValueError(
            f"eval_kept_traffic_fraction must be in (0, 1], got {eval_kept_traffic_fraction}"
        )

    if min_obs is not None:
        counts = df.groupby("target_id")["target_id"].transform("count")
        before = df["target_id"].nunique()
        df = df[counts >= min_obs].reset_index(drop=True)
        after = df["target_id"].nunique()
        print(f"min_obs={min_obs}: {before} -> {after} targets")
        if df.empty:
            raise ValueError(f"min_obs={min_obs} left zero targets")

    if eval_kept_traffic_fraction is not None:
        eval_pair_weight_min = _derive_eval_pair_weight_min(
            df, eval_kept_traffic_fraction
        )

    if eval_pair_weight_min is not None:
        thr = eval_pair_weight_min
        before = df["target_id"].nunique()
        surviving_targets = set(df.loc[df["weight"] >= thr, "target_id"].astype(str))
        df = df[
            df["target_id"].astype(str).isin(surviving_targets)
            & (df["weight"] >= thr)
        ].reset_index(drop=True)
        after = df["target_id"].nunique()
        print(
            f"eval_pair_weight_min={thr}: {before} -> {after} targets "
            f"({len(df)} surviving obs)"
        )
        if df.empty:
            raise ValueError(f"eval_pair_weight_min={thr} left zero eval obs")

    return df, eval_pair_weight_min


def _derive_eval_pair_weight_min(df: pd.DataFrame, frac: float) -> float:
    """Same derivation as generic_csv.py's `_derive_eval_weight_min_from_fraction`:
    dedupe at `(vp_id, target_city)` by per-pair max(weight), then descending
    cumulative sum to the requested kept traffic fraction."""
    if "target_city" not in df.columns:
        raise ValueError("eval_kept_traffic_fraction requires a target_city column")
    city = df["target_city"].astype(str).str.strip()
    blank = ~df["target_city"].notna() | (city == "")
    if blank.any():
        raise ValueError(
            "eval_kept_traffic_fraction requires non-blank target_city on every row"
        )
    per_pair = df.groupby(["vp_id", "target_city"], as_index=False).agg(
        weight=("weight", "max")
    )
    weights = per_pair["weight"].to_numpy(dtype=float)
    total = float(weights.sum())
    if total <= 0:
        print(
            f"eval_kept_traffic_fraction={frac}: all pair weights are zero; "
            "derived eval_pair_weight_min=0.0"
        )
        return 0.0
    weights_sorted = np.sort(weights)[::-1]
    target = frac * total
    cum = np.cumsum(weights_sorted)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(weights_sorted) - 1)
    threshold = float(weights_sorted[idx])
    print(
        f"eval_kept_traffic_fraction={frac}: derived eval_pair_weight_min="
        f"{threshold:.12g} from {len(per_pair)} (vp_id, target_city) pairs"
    )
    return threshold


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (vp, target) pair — the min-RTT observation — with the
    derived per-pair columns (gc_km, radius_km, inflation, rtt_rank_norm)."""
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
    # Normalized RTT rank of each pair within its target: 0 = the target's
    # fastest VP, 1 = its slowest (0 for single-VP targets; ties share the
    # lower rank so a tied-fastest VP still ranks 0).
    grp = pairs.groupby("target_id")["rtt_ms"]
    n = grp.transform("size").to_numpy(dtype=float)
    rank = grp.rank(method="min").to_numpy(dtype=float) - 1.0
    pairs["rtt_rank_norm"] = np.where(n > 1, rank / np.maximum(n - 1, 1), 0.0)
    return pairs


def per_target_metrics(
    pairs: pd.DataFrame,
    spearman_min_pairs: int = DEFAULT_SPEARMAN_MIN_PAIRS,
) -> pd.DataFrame:
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

    def _spearman(sub: pd.DataFrame) -> float:
        if (
            len(sub) < spearman_min_pairs
            or sub["gc_km"].nunique() < 2
            or sub["rtt_ms"].nunique() < 2
        ):
            return float("nan")
        return float(spearmanr(sub["gc_km"], sub["rtt_ms"]).statistic)

    spear = pairs.groupby("target_id", sort=True)[["gc_km", "rtt_ms"]].apply(_spearman)

    # SOI violations: inflation < 1 means faster than the 2/3c floor.
    # Colocated pairs (NaN inflation) are excluded from the denominator.
    soi = (pairs["inflation"] < 1.0).groupby(pairs["target_id"]).sum()
    n_inflation = g["inflation"].count()

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
        "min_inflation": g["inflation"].min(),
        "rtt_weighted_dist_km": g["_wd"].sum() / g["_w"].sum(),
        "rtt_dist_spearman": spear,
        "closest_vp_rtt_rank": closest["rtt_rank_norm"],
        "closest_is_shortest_ping": closest["vp_id"] == shortest_ping["vp_id"],
        "closest_to_shortest_ping_km": haversine_distance(
            closest["vp_lat"].to_numpy(), closest["vp_lon"].to_numpy(),
            shortest_ping["vp_lat"].to_numpy(), shortest_ping["vp_lon"].to_numpy(),
        ),
        "soi_violation_share": soi / n_inflation,
    })
    return out.reset_index()


def anycast_metrics(
    pairs: pd.DataFrame,
    delta_ms: float = DEFAULT_ANYCAST_DELTA_MS,
) -> pd.DataFrame:
    """Per-target disk-disjointness test over the low-RTT VP set.

    Low-RTT set = pairs with rtt <= min_rtt + delta_ms. `vp_pair_disk_overlap_km`
    is the min over its VP pairs of (r_i + r_j) - d(vp_i, vp_j): negative
    means two constraint disks are disjoint — physically impossible for a
    unicast target and a predictor of empty-intersection MTL failures.
    `n_disjoint_sites` is the iGreedy-style greedy count of mutually
    disjoint disks (smallest radius first); `anycast_suspect` uses the exact
    pair test (vp_pair_disk_overlap_km < 0)."""
    rows: list[tuple[str, float, int, bool]] = []
    for target_id, sub in pairs.groupby("target_id", sort=True):
        low = sub[sub["rtt_ms"] <= sub["rtt_ms"].min() + delta_ms]
        if len(low) < 2:
            rows.append((target_id, float("nan"), 1, False))
            continue
        lat = low["vp_lat"].to_numpy()
        lon = low["vp_lon"].to_numpy()
        r = low["radius_km"].to_numpy()
        d = haversine_distance(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        slack = r[:, None] + r[None, :] - d
        iu = np.triu_indices(len(low), k=1)
        min_slack = float(slack[iu].min())
        selected: list[int] = []
        for i in np.argsort(r, kind="stable"):
            if all(d[i, j] > r[i] + r[j] for j in selected):
                selected.append(int(i))
        rows.append((target_id, min_slack, len(selected), min_slack < 0))
    return pd.DataFrame(
        rows,
        columns=["target_id", "vp_pair_disk_overlap_km", "n_disjoint_sites", "anycast_suspect"],
    )


def cluster_targets(
    per_target: pd.DataFrame,
    radius_km: float,
    top_n_neighbors: int = DEFAULT_TOP_N_NEIGHBORS,
    write_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Cluster the targets into R-coherent regions (answer-space sizing).

    Returns (per_target, clusters, block):
      per_target — the input frame with cluster_id, truth centroid coords,
        cell_gap_km, target_distinguishable_vp_dist_km (= gap/2) and the two
        boolean same-cluster columns appended;
      clusters — one row per answer-space region: centroid, n_members,
        radius_km, is_singleton, cell_gap_km, top-N neighbor ids + distances;
      block — the `target_clustering` summary (region count, singletons, and
        the strict within-R / loose same-cluster VP-proximity shares).

    A single-cluster answer space gets cell_gap_km = inf: with no competing
    centroid every VP is discriminative and classification is trivial.

    ``write_dir``, if given, also writes the canonical
    `cluster_ground_truth` triplet (clusters.csv/assignments.csv/meta.json)
    there — lets `scripts.visualization.cluster.plot_ground_truth_clusters`
    render the answer-space map layer straight from this clustering, no
    recomputation."""
    res = cluster_ground_truth(
        per_target["target_lat"].to_numpy(),
        per_target["target_lon"].to_numpy(),
        radius_km=radius_km,
    )
    if write_dir is not None:
        _write_outputs(per_target, res, write_dir)
    labels = res.labels.astype(int)
    n_clusters = int(res.n_clusters)

    centroids_rad = np.radians(np.column_stack([res.centroid_lat, res.centroid_lon]))
    tree = BallTree(centroids_rad, metric="haversine")

    # Cell gap + top-N neighbors from one self-query (self sits in column 0).
    k = min(top_n_neighbors + 1, n_clusters)
    ndist, nidx = tree.query(centroids_rad, k=k)
    ndist_km = ndist * EARTH_RADIUS_KM
    cell_gap = ndist_km[:, 1] if n_clusters > 1 else np.full(1, np.inf)

    clusters = pd.DataFrame({
        "cluster_id": np.arange(n_clusters),
        "centroid_lat": res.centroid_lat,
        "centroid_lon": res.centroid_lon,
        "n_members": res.member_counts,
        "radius_km": res.radius_km.round(3),
        "is_singleton": res.member_counts == 1,
        "cell_gap_km": cell_gap.round(3),
    })
    for i in range(1, k):
        clusters[f"neighbor{i}_cluster_id"] = nidx[:, i]
        clusters[f"neighbor{i}_km"] = ndist_km[:, i].round(3)

    def _nearest_cell(lat: pd.Series, lon: pd.Series) -> np.ndarray:
        coords = np.radians(np.column_stack([lat.to_numpy(dtype=float),
                                             lon.to_numpy(dtype=float)]))
        _, idx = tree.query(coords, k=1)
        return idx[:, 0]

    per_target = per_target.assign(
        cluster_id=labels,
        truth_centroid_lat=res.centroid_lat[labels],
        truth_centroid_lon=res.centroid_lon[labels],
        cell_gap_km=cell_gap[labels],
        target_distinguishable_vp_dist_km=cell_gap[labels] / 2.0,
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
        "n_clusters": n_clusters,
        "n_singletons": int((res.member_counts == 1).sum()),
        "targets_per_cluster": round(len(per_target) / n_clusters, 3),
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
    return per_target, clusters, block


def proximity_metrics(
    pairs: pd.DataFrame,
    per_target: pd.DataFrame,
) -> pd.DataFrame:
    """Discriminative-set metrics + the proximity label ladder.

    D per target = VPs whose distance to the truth cluster centroid is below
    cell_gap/2 (the direction-free bound). Labels are purely D-based:
    NO_PROXIMITY (D empty) -> HAS_NOT_USED_PROXIMITY (shortest-ping VP not
    in D) -> HAS_USED_PROXIMITY. Requires the centroid columns from
    `cluster_targets`."""
    merged = pairs.merge(
        per_target[["target_id", "truth_centroid_lat", "truth_centroid_lon",
                    "target_distinguishable_vp_dist_km"]],
        on="target_id", how="inner",
    )
    merged["vp_to_centroid_km"] = haversine_distance(
        merged["vp_lat"].to_numpy(), merged["vp_lon"].to_numpy(),
        merged["truth_centroid_lat"].to_numpy(),
        merged["truth_centroid_lon"].to_numpy(),
    )
    merged["is_discriminative"] = (
        merged["vp_to_centroid_km"] < merged["target_distinguishable_vp_dist_km"]
    )

    g = merged.groupby("target_id", sort=True)
    shortest_ping = merged.loc[g["rtt_ms"].idxmin()].set_index("target_id")
    n_disc = g["is_discriminative"].sum().astype(int)
    best_rank = (
        merged[merged["is_discriminative"]]
        .groupby("target_id")["rtt_rank_norm"].min()
    )

    out = per_target.set_index("target_id")
    out["closest_vp_to_centroid_km"] = g["vp_to_centroid_km"].min()
    out["shortest_ping_vp_to_centroid_km"] = shortest_ping["vp_to_centroid_km"]
    out["n_discriminative_vps"] = n_disc
    out["has_vp_proximity"] = n_disc > 0
    out["shortest_ping_vp_is_discriminative"] = shortest_ping["is_discriminative"]
    out["best_discriminative_rtt_rank"] = best_rank  # NaN when D is empty
    out["proximity_label"] = np.select(
        [
            ~out["has_vp_proximity"],
            ~out["shortest_ping_vp_is_discriminative"],
        ],
        ["NO_PROXIMITY", "HAS_NOT_USED_PROXIMITY"],
        default="HAS_USED_PROXIMITY",
    )
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


def summarize(per_target: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_targets": int(len(per_target)),
        "metrics": {m: _stat_block(per_target[m]) for m in PER_TARGET_METRICS},
    }


def proximity_summary(per_target: pd.DataFrame) -> dict[str, Any]:
    """Label shares + the CBG-opportunity headline (see module docstring).

    Opportunity = NO_PROXIMITY + HAS_NOT_USED_PROXIMITY: everywhere the
    shortest-ping baseline carries no correctness guarantee — either no
    discriminative VP exists (only multilateration can help) or one exists
    but the baseline does not land on it (better VP selection or
    multilateration can help). Only HAS_USED_PROXIMITY is excluded: there
    the baseline is guaranteed correct by the cell_gap/2 bound. The lucky
    share reports what fraction of the opportunity population the baseline
    nevertheless converts via Voronoi assignment.

    The two HAS_* labels partition the has-proximity population by whether
    the baseline uses it; their sum is the share with |D| > 0 — the
    geometric ceiling for any proximity method on this dataset."""
    labels = per_target["proximity_label"]
    shares = {
        f"{lab.lower()}_share": round(float((labels == lab).mean()), 4)
        for lab in PROXIMITY_LABELS
    }
    opp_rows = per_target[labels != "HAS_USED_PROXIMITY"]
    lucky = (
        round(float(opp_rows["shortest_ping_vp_in_same_cluster"].mean()), 4)
        if len(opp_rows) else None
    )
    return {
        **shares,
        "cbg_opportunity_share": round(
            shares["no_proximity_share"] + shares["has_not_used_proximity_share"], 4
        ),
        "opportunity_baseline_lucky_share": lucky,
    }


def rtt_quality_summary(pairs: pd.DataFrame, per_target: pd.DataFrame) -> dict[str, Any]:
    """Dataset-level RTT hygiene: SOI share over pairs, anycast + agreement shares."""
    inflation = pairs["inflation"].dropna()
    return {
        "pair_soi_violation_share": (
            round(float((inflation < 1.0).mean()), 4) if len(inflation) else None
        ),
        "anycast_suspect_share": round(float(per_target["anycast_suspect"].mean()), 4),
        "closest_is_shortest_ping_share": round(
            float(per_target["closest_is_shortest_ping"].mean()), 4
        ),
    }


def write_mesh(
    frame: pd.DataFrame,
    id_col: str,
    lat_col: str,
    lon_col: str,
    out_path: Path,
    max_n: int = DEFAULT_MESH_MAX_N,
) -> Path | None:
    """Write the id-indexed great-circle distance matrix over the frame's
    unique endpoints; None (with a notice) past max_n endpoints."""
    uniq = frame.drop_duplicates(id_col)
    n = len(uniq)
    if n > max_n:
        print(f"{n} unique {id_col}s exceeds the {max_n} mesh cap — "
              f"skipping {out_path.name}")
        return None
    lat = uniq[lat_col].to_numpy(dtype=float)
    lon = uniq[lon_col].to_numpy(dtype=float)
    d = haversine_distance(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    ids = uniq[id_col].to_numpy()
    pd.DataFrame(d.round(3), index=ids, columns=ids).to_csv(
        out_path, index_label=id_col
    )
    return out_path


def eval_source(
    csv_path: Path,
    out_dir: Path,
    cluster_radius_km: float = DEFAULT_CLUSTER_RADIUS_KM,
    *,
    top_n_neighbors: int = DEFAULT_TOP_N_NEIGHBORS,
    anycast_delta_ms: float = DEFAULT_ANYCAST_DELTA_MS,
    spearman_min_pairs: int = DEFAULT_SPEARMAN_MIN_PAIRS,
    mesh_max_n: int = DEFAULT_MESH_MAX_N,
    min_obs: int | None = None,
    eval_pair_weight_min: float | None = None,
    eval_kept_traffic_fraction: float | None = None,
) -> dict[str, Any]:
    """Score one canonical CSV; write the per-target CSV, per-cluster CSV,
    distance meshes and stats JSON into out_dir (named after the CSV's stem)
    and return the stats dict with output paths.

    `min_obs` / `eval_pair_weight_min` / `eval_kept_traffic_fraction`, when
    given, restrict scoring to the same eval-side subset a real benchmark
    run's `source_kwargs` (or top-level yaml keys, for the two weight knobs)
    would produce — see `apply_eval_target_filters`. Pass none of them to
    score the whole CSV as-is (the original, filter-free precheck)."""
    df = load_canonical_csv(csv_path)
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
    per_target, clusters, clustering = cluster_targets(
        per_target, cluster_radius_km, top_n_neighbors=top_n_neighbors,
        write_dir=clusters_dir,
    )
    per_target = proximity_metrics(pairs, per_target)
    per_target = per_target.merge(anycast_metrics(pairs, anycast_delta_ms), on="target_id")

    stats: dict[str, Any] = {
        "csv": str(csv_path),
        "n_obs": int(len(df)),
        "n_pairs": int(len(pairs)),
        "n_vps": int(pairs["vp_id"].nunique()),
        **summarize(per_target),
        "target_clustering": clustering,
        "proximity": proximity_summary(per_target),
        "rtt_quality": rtt_quality_summary(pairs, per_target),
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
        pairs, "vp_id", "vp_lat", "vp_lon",
        out_dir / f"{csv_path.stem}_vp_mesh_km.csv", max_n=mesh_max_n,
    )
    cl_mesh = write_mesh(
        clusters, "cluster_id", "centroid_lat", "centroid_lon",
        out_dir / f"{csv_path.stem}_cluster_mesh_km.csv", max_n=mesh_max_n,
    )
    if vp_mesh is not None:
        stats["vp_mesh_csv"] = str(vp_mesh)
    if cl_mesh is not None:
        stats["cluster_mesh_csv"] = str(cl_mesh)
    return stats
