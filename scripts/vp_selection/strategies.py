"""VP-selection strategies — lifted from Cho et al. 2024 (`upstream_py/analyze_air.py`).

Two strategy families, distinguished by output shape:

  * `select_vps(...)` — **sequence strategies**. Returns the full N-element
    ordered selection. Callers slice `[:K]` for the K-VP subset.

      dist_geo  — DIST-GEO. Greedy Prim maximizing Σ pair-geodesic.
      h1_as     — Hybrid 1 (AS). dist_geo + cluster-balance preference.
      h1_city   — Hybrid 1 (city). Same as h1_as on city.
      h2_as     — Hybrid 2 (AS). 100-random-seed + h1_as continuation.

  * `sample_vps(...)` — **sampling strategies**. Returns an independently-
    drawn K-element subset per call (Cho-strict per-K sampling).

      random       — uniform random K-subset.
      cluster_as   — stratified random by ASN (even base + round-robin
                     remainder, deficit absorbed by larger clusters).
      cluster_city — same on city.

All stochastic calls thread `seed` through a local `random.Random(seed)`
instance — no global state. `dist_rtt` is intentionally out of scope.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional


SequenceStrategy = Literal["dist_geo", "h1_as", "h1_city", "h2_as"]
SamplingStrategy = Literal["random", "cluster_as", "cluster_city"]
_SEQUENCE_STRATEGIES = ("dist_geo", "h1_as", "h1_city", "h2_as")
_SAMPLING_STRATEGIES = ("random", "cluster_as", "cluster_city")
_SEED_SIZE_H2 = 100  # Cho's choice for the random-init prefix of H2.


@dataclass(frozen=True)
class VpMeta:
    """Minimal VP metadata the selection strategies need.

    `asn`/`city`/`country` may be None — h1_as / h1_city use them only when
    set, treating None as a distinct cluster value (consistent with dict-key
    semantics)."""
    lat: float
    lon: float
    asn: Optional[int] = None
    city: Optional[str] = None
    country: Optional[str] = None


def select_vps(
    pool: dict[str, VpMeta],
    distances: dict[tuple[str, str], float],
    strategy: SequenceStrategy,
    seed: int = 0,
) -> list[str]:
    """Order the pool by the chosen sequence strategy.

    Returns the full N-element ordered selection sequence; first K elements
    are the K-VP subset. For sampling strategies (`random`, `cluster_*`),
    use `sample_vps` instead.
    """
    if strategy not in _SEQUENCE_STRATEGIES:
        raise ValueError(
            f"unknown sequence strategy {strategy!r}; "
            f"expected one of {_SEQUENCE_STRATEGIES} "
            f"(sampling strategies {_SAMPLING_STRATEGIES} use sample_vps)"
        )
    if not pool:
        return []

    nodes = sorted(pool)
    rng = random.Random(seed)

    if len(nodes) == 1:
        return nodes

    adj = _build_adjacency(distances)

    if strategy == "h2_as":
        return _select_h2_as(pool, adj, nodes, rng)

    # max_edge start for dist_geo / h1_as / h1_city
    start = _max_edge_start(distances, rng)
    selected = {start}
    weights = {n: d for n, d in adj.get(start, {}).items() if n not in selected}

    cluster_attr = _cluster_attr_for(strategy)
    if cluster_attr is None:
        cluster_set: Optional[set] = None
        n_clusters = 0
    else:
        cluster_set = {getattr(pool[start], cluster_attr)}
        n_clusters = len({getattr(m, cluster_attr) for m in pool.values()})

    continuation = _greedy_prim_continue(
        adj=adj,
        weights=weights,
        selected=selected,
        all_nodes=set(nodes),
        pool=pool,
        cluster_attr=cluster_attr,
        cluster_set=cluster_set,
        n_clusters=n_clusters,
    )
    return [start] + continuation


# ---- helpers --------------------------------------------------------------


def _build_adjacency(
    distances: dict[tuple[str, str], float],
) -> dict[str, dict[str, float]]:
    """Convert canonical (a, b) → d pairs to an adjacency-list view."""
    adj: dict[str, dict[str, float]] = {}
    for (a, b), d in distances.items():
        adj.setdefault(a, {})[b] = d
        adj.setdefault(b, {})[a] = d
    return adj


def _max_edge_start(
    distances: dict[tuple[str, str], float],
    rng: random.Random,
) -> str:
    """Pick a random endpoint of the heaviest edge (upstream `max_edge` mode)."""
    max_pair = max(distances.items(), key=lambda kv: kv[1])[0]
    return rng.choice(max_pair)


def _cluster_attr_for(strategy: Strategy) -> Optional[str]:
    if strategy == "h1_as":
        return "asn"
    if strategy == "h1_city":
        return "city"
    return None


def _select_h2_as(
    pool: dict[str, VpMeta],
    adj: dict[str, dict[str, float]],
    nodes: list[str],
    rng: random.Random,
) -> list[str]:
    """H2-AS: 100-random seed + h1_as continuation. Cluster_set is initialized
    from the asns of the random seeds (matches upstream lines 277–279)."""
    seed_size = min(_SEED_SIZE_H2, len(nodes))
    seed_list = rng.sample(nodes, seed_size)
    if seed_size == len(nodes):
        return seed_list  # nothing left to greedy

    selected = set(seed_list)
    weights: dict[str, float] = {}
    for node in seed_list:
        for neighbor, d in adj.get(node, {}).items():
            if neighbor in selected:
                continue
            weights[neighbor] = weights.get(neighbor, 0.0) + d

    cluster_set = {pool[v].asn for v in seed_list}
    n_clusters = len({m.asn for m in pool.values()})
    continuation = _greedy_prim_continue(
        adj=adj,
        weights=weights,
        selected=selected,
        all_nodes=set(nodes),
        pool=pool,
        cluster_attr="asn",
        cluster_set=cluster_set,
        n_clusters=n_clusters,
    )
    return seed_list + continuation


def _greedy_prim_continue(
    adj: dict[str, dict[str, float]],
    weights: dict[str, float],
    selected: set,
    all_nodes: set,
    pool: dict[str, VpMeta],
    cluster_attr: Optional[str],
    cluster_set: Optional[set],
    n_clusters: int,
) -> list[str]:
    """Continue the greedy Prim selection until every node is selected.

    `weights[v]` accumulates Σ dist(v, s) over s in `selected`. At each step,
    pick argmax(weights). With `cluster_attr`, prefer a node whose cluster
    isn't yet in `cluster_set` until all clusters are covered.

    Mirrors upstream `_select_prim` (analyze_air.py lines 253–321), with the
    single fix that for non-random100 starts the cluster_set is seeded with
    the start node's cluster (handled by the caller, not here)."""
    new_order: list[str] = []
    cluster_mode = cluster_attr is not None and cluster_set is not None

    while len(selected) < len(all_nodes):
        if not weights:
            # Disconnected component — pick an arbitrary unselected node
            # deterministically to keep going. Shouldn't happen on a complete
            # pair-distance graph, but defend against it.
            remaining = sorted(all_nodes - selected)
            picked = remaining[0]
            selected.add(picked)
            new_order.append(picked)
            for neighbor, d in adj.get(picked, {}).items():
                if neighbor in selected:
                    continue
                weights[neighbor] = weights.get(neighbor, 0.0) + d
            continue

        if cluster_mode and len(cluster_set) < n_clusters:
            sorted_by_weight = sorted(weights.items(), key=lambda kv: -kv[1])
            picked = sorted_by_weight[0][0]  # fallback
            for node, _w in sorted_by_weight:
                node_cluster = getattr(pool[node], cluster_attr)
                if node_cluster not in cluster_set:
                    picked = node
                    cluster_set.add(node_cluster)
                    break
        else:
            picked = max(weights, key=weights.get)

        del weights[picked]
        selected.add(picked)
        new_order.append(picked)

        for neighbor, d in adj.get(picked, {}).items():
            if neighbor in selected:
                continue
            weights[neighbor] = weights.get(neighbor, 0.0) + d

    return new_order


# ---- sampling strategies --------------------------------------------------


def sample_vps(
    pool: dict[str, VpMeta],
    strategy: SamplingStrategy,
    K: int,
    seed: int = 0,
) -> list[str]:
    """Draw an independent K-element subset by the chosen sampling strategy.

    - `random`: uniform random K-subset.
    - `cluster_as` / `cluster_city`: stratified random by ASN / city. Even
      `base = K // M` from each cluster, plus `remainder = K % M` distributed
      round-robin. If a cluster has fewer members than `base`, its deficit is
      absorbed by the round-robin pass over other clusters.

    Distances are NOT used (none of the sampling strategies need them).
    """
    if strategy not in _SAMPLING_STRATEGIES:
        raise ValueError(
            f"unknown sampling strategy {strategy!r}; "
            f"expected one of {_SAMPLING_STRATEGIES} "
            f"(sequence strategies {_SEQUENCE_STRATEGIES} use select_vps)"
        )
    if not pool or K <= 0:
        return []
    if K >= len(pool):
        return sorted(pool)

    rng = random.Random(seed)
    if strategy == "random":
        return rng.sample(sorted(pool), K)

    cluster_attr = "asn" if strategy == "cluster_as" else "city"
    return _stratified_random(pool, cluster_attr, K, rng)


def _stratified_random(
    pool: dict[str, VpMeta],
    cluster_attr: str,
    K: int,
    rng: random.Random,
) -> list[str]:
    """Stratified random with deficit redistribution.

    Steps:
      1. Group landmarks by `cluster_attr`; shuffle members within each cluster
         (seed-deterministic).
      2. Iterate clusters in shuffled order; first pass takes `base` members
         from each (or all members if cluster smaller than `base`). Deficit
         accumulates if a cluster runs out.
      3. Second pass: round-robin one-extra from each cluster with remaining
         capacity until we've drawn K elements total. This absorbs both the
         `remainder = K % M` and any deficit from step 2.
    """
    clusters_by_attr: dict = defaultdict(list)
    for vp_id, meta in pool.items():
        clusters_by_attr[getattr(meta, cluster_attr)].append(vp_id)

    cluster_keys = sorted(clusters_by_attr.keys(), key=lambda k: (k is None, str(k)))
    rng.shuffle(cluster_keys)

    members: dict = {}
    for key in cluster_keys:
        bucket = sorted(clusters_by_attr[key])
        rng.shuffle(bucket)
        members[key] = bucket

    M = len(cluster_keys)
    base = K // M
    selected: list[str] = []

    # First pass: take `base` from each cluster (or all if smaller).
    drawn: dict = {key: 0 for key in cluster_keys}
    for key in cluster_keys:
        take = min(base, len(members[key]))
        selected.extend(members[key][:take])
        drawn[key] = take

    # Second pass: round-robin distribute the remainder + any deficit.
    while len(selected) < K:
        progress = False
        for key in cluster_keys:
            if len(selected) >= K:
                break
            if drawn[key] < len(members[key]):
                selected.append(members[key][drawn[key]])
                drawn[key] += 1
                progress = True
        if not progress:
            break  # all clusters exhausted; pool too small for K (shouldn't happen given the early-return)

    return selected
