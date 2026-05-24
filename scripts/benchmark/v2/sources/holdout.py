"""Anchor holdout policy for the v2 benchmark — Sechidis-style K-fold split.

Solves the fit/eval leakage problem in fitted LTDs: today every anchor that
appears as an eval target also appears in the LTD training corpus, so the
curve is fit on the exact (RTT, distance) point it's later asked to predict.

The split is anchor-level. For fold *i*, anchors in fold *i* are evaluated;
all other anchors are fit-training material. The same policy is applied
uniformly across LTD variants so leaderboard comparisons reflect technique
differences rather than asymmetric eval conditions.

Algorithm (Sechidis et al. 2011, "On the Stratification of Multi-Label Data"):
greedy iterative multi-label stratification, deterministic given a seed.
Handles singleton labels (ASNs with one anchor) by sending them to one fold
each — unavoidable with K folds and N=1 samples per label.

Spatial pre-clustering (Roberts et al. 2017, "Cross-validation strategies for
data with temporal, spatial, hierarchical, or phylogenetic structure") makes
the unit of stratification a k-means cluster of nearby anchors rather than an
individual anchor — so anchors in the same metro never straddle the train/test
boundary, eliminating the spatial-autocorrelation leakage path.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnchorInfo(NamedTuple):
    """Minimal anchor record the splitter needs.

    `asn` and `country` may be None for malformed entries; they're treated as
    their own ('asn_none', 'country_none') labels for stratification purposes
    so missing-metadata anchors don't all pile into a single fold.
    """

    ip: str
    lat: float
    lon: float
    country: Optional[str]
    asn: Optional[int]


@dataclass(frozen=True)
class HoldoutPolicy:
    """Cross-validation holdout configuration for anchor-level splits.

    `kind` is reserved for future strategies (BLOO, environmental-only, etc.);
    v1 only implements `sechidis_kfold`. The `slice_suffix_fmt` controls how
    the policy serializes into the source's `slice_id()`, which in turn
    becomes the materialize/runner output directory — so two folds get
    parallel parquet trees with no orchestration changes upstream.
    """

    kind: str = "sechidis_kfold"
    k: int = 5
    fold_index: int = 0
    seed: int = 42
    labels: tuple[str, ...] = ("country", "asn_bucket")
    asn_bucket_top_n: int = 20
    spatial_clusters: Optional[int] = 30
    slice_suffix_fmt: str = "fold{fold_index}of{k}_seed{seed}"

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"HoldoutPolicy.k must be >=2 (got {self.k})")
        if not 0 <= self.fold_index < self.k:
            raise ValueError(
                f"HoldoutPolicy.fold_index must be in [0, k), got {self.fold_index} for k={self.k}"
            )
        if self.kind != "sechidis_kfold":
            raise ValueError(
                f"unknown HoldoutPolicy.kind {self.kind!r}; only 'sechidis_kfold' is implemented"
            )
        allowed_labels = {"country", "asn_bucket"}
        unknown = set(self.labels) - allowed_labels
        if unknown:
            raise ValueError(f"unknown labels {unknown!r}; allowed: {sorted(allowed_labels)}")
        if self.asn_bucket_top_n < 0:
            raise ValueError(f"asn_bucket_top_n must be >=0 (got {self.asn_bucket_top_n})")
        if self.spatial_clusters is not None and self.spatial_clusters < 2:
            raise ValueError(
                f"spatial_clusters must be None or >=2 (got {self.spatial_clusters})"
            )

    def slice_suffix(self) -> str:
        return self.slice_suffix_fmt.format(
            fold_index=self.fold_index, k=self.k, seed=self.seed,
        )


# ---- Public algorithm entry point -------------------------------------------


def compute_fold_assignments(
    anchors: list[AnchorInfo],
    policy: HoldoutPolicy,
) -> dict[str, int]:
    """Assign each anchor to a fold in [0, policy.k).

    Returns `{anchor_ip: fold_index}`. Deterministic given `policy.seed`.

    The stratification unit is either an individual anchor or, if
    `policy.spatial_clusters` is set, the k-means cluster the anchor belongs
    to (atomic — co-clustered anchors always land in the same fold). Labels
    used for balancing are derived per `policy.labels`.

    Edge cases:
      - 0 anchors: returns {}.
      - fewer anchors than folds: still produces a valid mapping; some folds
        end up empty.
      - spatial_clusters > #anchors: clamps down to #anchors with a log warning.
    """
    if not anchors:
        return {}

    # Sort early — deterministic order for tiebreaks and reproducibility.
    anchors = sorted(anchors, key=lambda a: a.ip)

    asn_bucket = _bucket_asns(anchors, policy.asn_bucket_top_n)

    if policy.spatial_clusters is not None:
        cluster_by_ip = _kmeans_spatial_clusters(
            anchors, policy.spatial_clusters, policy.seed,
        )
        units = _build_cluster_units(anchors, cluster_by_ip, asn_bucket, policy.labels)
    else:
        units = _build_anchor_units(anchors, asn_bucket, policy.labels)

    fold_by_unit = _sechidis_assign(units, policy.k, policy.seed)

    out: dict[str, int] = {}
    for unit in units:
        fold = fold_by_unit[unit["id"]]
        for ip in unit["ips"]:
            out[ip] = fold
    return out


# ---- ASN bucketing -----------------------------------------------------------


def _bucket_asns(anchors: list[AnchorInfo], top_n: int) -> dict[str, str]:
    """Return `{anchor_ip: asn_bucket_label}`.

    Top-N ASNs by anchor count each keep their own bucket (`"AS{asn}"`); the
    rest collapse into `"other_AS"`. ASN None → `"asn_none"`. Tiebreak on
    counts is ASN ascending so the ranking is deterministic.
    """
    asn_counts: Counter[Optional[int]] = Counter(a.asn for a in anchors)
    # Sort by (-count, asn) so ties break on ASN value, with None pushed last.
    ranked = sorted(
        asn_counts.items(),
        key=lambda kv: (-kv[1], kv[0] if kv[0] is not None else float("inf")),
    )
    top_asns = {asn for asn, _ in ranked[:top_n] if asn is not None}
    out: dict[str, str] = {}
    for a in anchors:
        if a.asn is None:
            out[a.ip] = "asn_none"
        elif a.asn in top_asns:
            out[a.ip] = f"AS{a.asn}"
        else:
            out[a.ip] = "other_AS"
    return out


# ---- Spatial clustering ------------------------------------------------------


def _kmeans_spatial_clusters(
    anchors: list[AnchorInfo],
    n_clusters: int,
    seed: int,
) -> dict[str, int]:
    """Cluster anchors by (lat, lon) using k-means on 3D unit vectors.

    The 3D-unit-vector projection lets euclidean k-means produce
    great-circle-ish clusters and gracefully handles antimeridian wrap-around
    (which planar lat/lon clustering would split incorrectly).
    """
    from scipy.cluster.vq import kmeans2

    n = len(anchors)
    if n_clusters > n:
        logger.warning(
            "spatial_clusters=%d > #anchors=%d; clamping to %d (one anchor per cluster).",
            n_clusters, n, n,
        )
        n_clusters = n

    # Convert (lat, lon) in degrees → 3D unit vector.
    lats = np.array([math.radians(a.lat) for a in anchors])
    lons = np.array([math.radians(a.lon) for a in anchors])
    x = np.cos(lats) * np.cos(lons)
    y = np.cos(lats) * np.sin(lons)
    z = np.sin(lats)
    coords = np.column_stack([x, y, z])

    _centroids, labels = kmeans2(
        coords, n_clusters, seed=seed, minit="++", missing="warn",
    )
    return {a.ip: int(label) for a, label in zip(anchors, labels)}


# ---- Unit construction -------------------------------------------------------


def _build_anchor_units(
    anchors: list[AnchorInfo],
    asn_bucket: dict[str, str],
    label_names: tuple[str, ...],
) -> list[dict]:
    """One unit per anchor (no spatial clustering)."""
    units = []
    for a in anchors:
        labels = _labels_for_anchor(a, asn_bucket[a.ip], label_names)
        units.append({"id": a.ip, "ips": [a.ip], "labels": labels})
    return units


def _build_cluster_units(
    anchors: list[AnchorInfo],
    cluster_by_ip: dict[str, int],
    asn_bucket: dict[str, str],
    label_names: tuple[str, ...],
) -> list[dict]:
    """One unit per k-means cluster; unit's label-set = union of its anchors."""
    cluster_anchors: dict[int, list[AnchorInfo]] = defaultdict(list)
    for a in anchors:
        cluster_anchors[cluster_by_ip[a.ip]].append(a)

    units = []
    for cluster_id in sorted(cluster_anchors):
        members = sorted(cluster_anchors[cluster_id], key=lambda a: a.ip)
        all_labels: set[str] = set()
        for a in members:
            all_labels.update(_labels_for_anchor(a, asn_bucket[a.ip], label_names))
        units.append({
            "id": f"cluster_{cluster_id}",
            "ips": [a.ip for a in members],
            "labels": frozenset(all_labels),
        })
    return units


def _labels_for_anchor(
    anchor: AnchorInfo,
    asn_bucket_label: str,
    label_names: tuple[str, ...],
) -> frozenset[str]:
    """Build the label set for one anchor. Namespaced per axis so a country
    code can never collide with an ASN bucket label."""
    out: set[str] = set()
    for name in label_names:
        if name == "country":
            out.add(f"country:{anchor.country or 'none'}")
        elif name == "asn_bucket":
            out.add(f"asn:{asn_bucket_label}")
    return frozenset(out)


# ---- Sechidis iterative stratification --------------------------------------


def _sechidis_assign(
    units: list[dict],
    k: int,
    seed: int,
) -> dict[str, int]:
    """Iterative multi-label stratification (Sechidis et al. 2011, Szymański
    variant).

    Process units one at a time, in order of their rarest label (rarest unit
    first), so scarce labels get placed while folds have maximum flexibility.
    Each unit lands in the fold that maximizes the **sum** of remaining-need
    across that unit's labels — using the sum (rather than max over labels)
    means a unit carrying multiple under-served labels prefers folds that are
    starved on multiple axes, not just the worst one. Empirically this gives
    better joint balance across country + ASN_bucket than scoring by max.

    Tiebreak chain: highest summed-need → fold with smallest current count →
    deterministic by unit_id → seeded random.
    """
    rng = random.Random(seed)

    # Frequency per label across all units. Drives rarest-first unit ordering.
    label_total: Counter[str] = Counter()
    for unit in units:
        for label in unit["labels"]:
            label_total[label] += 1

    # Initial demand per (fold, label): ceil(|units with L| / K). Decrements
    # as units are placed.
    remaining_need: dict[tuple[int, str], int] = {}
    for label, total in label_total.items():
        share = math.ceil(total / k)
        for f in range(k):
            remaining_need[(f, label)] = share

    fold_sizes: dict[int, int] = {f: 0 for f in range(k)}
    fold_by_unit: dict[str, int] = {}

    def unit_rarity(unit: dict) -> int:
        """Rarest-label frequency this unit carries. Lower = scarcer = placed first."""
        if not unit["labels"]:
            return 10**9  # no labels → place last
        return min(label_total[lbl] for lbl in unit["labels"])

    # Sort units rarest-first, deterministic by id on ties.
    units_in_order = sorted(units, key=lambda u: (unit_rarity(u), u["id"]))

    for unit in units_in_order:
        chosen = _pick_fold(unit, remaining_need, fold_sizes, k, rng)
        fold_by_unit[unit["id"]] = chosen
        fold_sizes[chosen] += 1
        for lbl in unit["labels"]:
            if (chosen, lbl) in remaining_need:
                remaining_need[(chosen, lbl)] -= 1

    return fold_by_unit


def _pick_fold(
    unit: dict,
    remaining_need: dict[tuple[int, str], int],
    fold_sizes: dict[int, int],
    k: int,
    rng: random.Random,
) -> int:
    """Score each fold by the **sum** of remaining_need over the unit's
    labels; pick the highest. Tiebreaks: smallest current fold size,
    deterministic on unit_id, then seeded random."""
    labels = unit["labels"]

    def fold_score(f: int) -> int:
        if not labels:
            return 0
        return sum(remaining_need.get((f, lbl), 0) for lbl in labels)

    scores = {f: fold_score(f) for f in range(k)}
    best_score = max(scores.values())
    candidates = [f for f in range(k) if scores[f] == best_score]
    if len(candidates) == 1:
        return candidates[0]

    # Tiebreak by smallest current fold size — keeps fold sizes from drifting.
    min_size = min(fold_sizes[f] for f in candidates)
    candidates = [f for f in candidates if fold_sizes[f] == min_size]
    if len(candidates) == 1:
        return candidates[0]

    # Final tiebreak: seeded random (still deterministic given seed).
    return rng.choice(candidates)
