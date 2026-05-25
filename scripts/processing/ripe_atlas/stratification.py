"""Anchor stratification algorithms for the v2 benchmark — K-fold splits.

Solves the fit/eval leakage problem in fitted LTDs: today every anchor that
appears as an eval target also appears in the LTD training corpus, so the
curve is fit on the exact (RTT, distance) point it's later asked to predict.

The split is anchor-level (per `report.md` lock-in). For fold *i*, anchors in
fold *i* are evaluated; all other anchors are fit-training material. The same
stratification is applied uniformly across LTD variants so leaderboard
comparisons reflect technique differences rather than asymmetric eval
conditions.

Three classes live here:

- `SechidisStratification` — iterative multi-label stratification (Sechidis
  et al. 2011) balancing country + ASN-bucket. Optional spatial k-means
  pre-clustering (Roberts et al. 2017) makes the unit of stratification a
  metro rather than an individual anchor.
- `DistGeoStratification` — per-ASN-bucket greedy-Prim ordering (reuses
  `select_vps(strategy="dist_geo")` from scripts.vp_selection.strategies) +
  balanced round-robin into K folds. Each fold gets ~1/K of each ASN bucket
  with explicit intra-fold spatial spread.
- `LoadedStratification` — reads a precomputed stratification JSON
  (the artifact written by `stratify.py`). What `RipeAtlasSource` actually
  consumes in production. The other two are used by `stratify.py` to
  produce that JSON.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
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
class SechidisStratification:
    """Sechidis-style iterative multi-label stratification.

    Used by `stratify.py` to produce a stratification JSON; consumed by
    `RipeAtlasSource` indirectly via `LoadedStratification`.

    `kind` is reserved for future variants (BLOO, environmental-only, etc.);
    v1 only implements `sechidis_kfold`.
    """

    kind: str = "sechidis_kfold"
    k: int = 5
    fold_index: int = 0
    seed: int = 42
    labels: tuple[str, ...] = ("country", "asn_bucket")
    asn_bucket_top_n: int = 20
    spatial_clusters: Optional[int] = 30

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"SechidisStratification.k must be >=2 (got {self.k})")
        if not 0 <= self.fold_index < self.k:
            raise ValueError(
                f"SechidisStratification.fold_index must be in [0, k), got "
                f"{self.fold_index} for k={self.k}"
            )
        if self.kind != "sechidis_kfold":
            raise ValueError(
                f"unknown SechidisStratification.kind {self.kind!r}; "
                f"only 'sechidis_kfold' is implemented"
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

    def compute_fold_assignments(
        self, anchors: list[AnchorInfo],
    ) -> dict[str, int]:
        """Polymorphic dispatch entry — same name as DistGeoStratification's
        so `stratify.py` can call `algo.compute_fold_assignments(...)`
        without knowing which subclass is in hand."""
        return compute_fold_assignments(anchors, self)


@dataclass(frozen=True)
class DistGeoStratification:
    """K-fold anchor split via per-ASN-bucket dist_geo ordering + round-robin.

    Algorithm (see `compute_dist_geo_fold_assignments`):
      1. Bucket anchors by ASN (top-N ASNs each keep their own bucket; rest
         collapse to "other_AS"; ASN None → "asn_none").
      2. For each bucket, run greedy-Prim distance ordering (the `dist_geo`
         strategy from `scripts/vp_selection/strategies.py`) on the
         bucket members' pairwise haversine distances.
      3. Assign each bucket's ordered anchors to folds via balanced
         round-robin (i mod k by default; shift to the smallest fold when
         singleton placements from earlier buckets unbalance counts).

    Each fold gets ~1/K of each ASN bucket with spatially-spread anchors
    within each bucket → per-fold ASN balance + intra-fold spatial diversity.
    Sibling to `SechidisStratification`; not a drop-in replacement — the two
    answer slightly different questions (balance-by-label vs
    spread-by-distance) and we intend to run both and compare.
    """

    kind: str = "dist_geo_kfold"
    k: int = 5
    fold_index: int = 0
    seed: int = 42
    asn_bucket_top_n: int = 20

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"DistGeoStratification.k must be >=2 (got {self.k})")
        if not 0 <= self.fold_index < self.k:
            raise ValueError(
                f"DistGeoStratification.fold_index must be in [0, k), got "
                f"{self.fold_index} for k={self.k}"
            )
        if self.kind != "dist_geo_kfold":
            raise ValueError(
                f"unknown DistGeoStratification.kind {self.kind!r}; "
                f"only 'dist_geo_kfold' is implemented"
            )
        if self.asn_bucket_top_n < 0:
            raise ValueError(
                f"asn_bucket_top_n must be >=0 (got {self.asn_bucket_top_n})"
            )

    def compute_fold_assignments(
        self, anchors: list[AnchorInfo],
    ) -> dict[str, int]:
        return compute_dist_geo_fold_assignments(anchors, self)


@dataclass(frozen=True)
class LoadedStratification:
    """Consume a precomputed stratification JSON (from `stratify.py`).

    Decouples the split decision from the source's active corpus:
    `stratify.py` runs once over the canonical anchor set to produce a
    reviewable JSON artifact; then `RipeAtlasSource` is constructed with this
    class to read those assignments instead of recomputing.

    Mismatch handling: `compute_fold_assignments(anchors)` intersects the
    loaded assignments with the active anchor list, logs counts on both
    sides, and raises if the target fold or its complement ends up empty.

    Anchors in the active corpus but not in the stratification naturally drop
    out of both train and test (they are absent from the returned mapping
    and `RipeAtlasSource`'s iterators skip anything not in either set).
    """

    path: Path
    fold_index: int = 0

    def __post_init__(self) -> None:
        # Normalize to Path so callers can pass strings.
        object.__setattr__(self, "path", Path(self.path))
        if not self.path.exists():
            raise FileNotFoundError(f"stratification file not found: {self.path}")
        data = self._load()
        if "policy" not in data or "fold_assignments" not in data:
            raise ValueError(
                f"stratification file {self.path} missing required keys "
                f"(expected 'policy' and 'fold_assignments')"
            )
        k = int(data["policy"]["k"])
        if not 0 <= self.fold_index < k:
            raise ValueError(
                f"LoadedStratification.fold_index must be in [0, {k}), got {self.fold_index}"
            )

    @property
    def k(self) -> int:
        return int(self._load()["policy"]["k"])

    def _load(self) -> dict:
        with self.path.open() as fh:
            return json.load(fh)

    def compute_fold_assignments(
        self, anchors: list[AnchorInfo],
    ) -> dict[str, int]:
        data = self._load()
        loaded = {ip: int(f) for ip, f in data["fold_assignments"].items()}
        active = {a.ip for a in anchors}
        stratification_ips = set(loaded)

        in_both = active & stratification_ips
        only_active = active - stratification_ips
        only_stratification = stratification_ips - active

        if only_active:
            logger.warning(
                "LoadedStratification: %d active anchor(s) missing from stratification %s — dropped from both fit and eval",
                len(only_active), self.path,
            )
        if only_stratification:
            logger.warning(
                "LoadedStratification: %d stratification anchor(s) absent from active corpus — ignored",
                len(only_stratification),
            )

        result = {ip: loaded[ip] for ip in in_both}

        n_test = sum(1 for f in result.values() if f == self.fold_index)
        n_train = sum(1 for f in result.values() if f != self.fold_index)
        if n_test == 0:
            raise ValueError(
                f"LoadedStratification: fold_index={self.fold_index} eval set is empty "
                f"after intersecting stratification ({self.path}) with active corpus"
            )
        if n_train == 0:
            raise ValueError(
                f"LoadedStratification: fold_index={self.fold_index} fit set is empty "
                f"after intersecting stratification ({self.path}) with active corpus"
            )
        return result


# ---- Public algorithm entry point -------------------------------------------


def compute_fold_assignments(
    anchors: list[AnchorInfo],
    policy: "SechidisStratification",
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


# ---- DistGeo K-fold (per-ASN-bucket greedy Prim + balanced round-robin) -----


def compute_dist_geo_fold_assignments(
    anchors: list[AnchorInfo],
    policy: "DistGeoStratification",
) -> dict[str, int]:
    """Assign each anchor to a fold in [0, policy.k) via dist_geo + ASN bucket.

    Returns `{anchor_ip: fold_index}`. Deterministic given `policy.seed`.

    Algorithm:
      1. Bucket anchors by ASN (top-N + "other_AS"; None → "asn_none").
      2. For each bucket (in deterministic key order):
         - Singleton bucket (≤1 anchor): place in the currently-smallest fold.
         - General case: compute pairwise haversine distances, run
           `select_vps(strategy="dist_geo")` → ordered list, then assign each
           anchor to the smallest fold (tiebreak: prefer position `i mod k`,
           then lowest fold_index).
      3. Result: each fold has ~1/K of each ASN bucket with intra-bucket
         spatial spread.

    Edge cases:
      - 0 anchors: returns {}.
      - fewer anchors than folds: still produces a valid mapping; some folds
        end up empty.
    """
    if not anchors:
        return {}

    # Lazy imports — avoid hard deps at module-load.
    from scripts.utils.helpers import haversine
    from scripts.vp_selection.strategies import VpMeta, select_vps

    anchors = sorted(anchors, key=lambda a: a.ip)
    asn_bucket = _bucket_asns(anchors, policy.asn_bucket_top_n)

    by_bucket: dict[str, list[AnchorInfo]] = defaultdict(list)
    for a in anchors:
        by_bucket[asn_bucket[a.ip]].append(a)

    fold_sizes: dict[int, int] = {f: 0 for f in range(policy.k)}
    out: dict[str, int] = {}

    for bucket_name in sorted(by_bucket):
        bucket_anchors = sorted(by_bucket[bucket_name], key=lambda a: a.ip)

        if len(bucket_anchors) <= 1:
            for a in bucket_anchors:
                fold = _smallest_fold(fold_sizes)
                out[a.ip] = fold
                fold_sizes[fold] += 1
            continue

        # Pairwise haversine; canonical (a, b) keying suffices — strategies.py
        # builds a symmetric adjacency from it.
        distances: dict[tuple[str, str], float] = {}
        for i, a in enumerate(bucket_anchors):
            for b in bucket_anchors[i + 1:]:
                distances[(a.ip, b.ip)] = float(
                    haversine((a.lat, a.lon), (b.lat, b.lon))
                )

        pool = {
            a.ip: VpMeta(lat=a.lat, lon=a.lon, asn=a.asn, country=a.country)
            for a in bucket_anchors
        }
        ordered = select_vps(pool, distances, strategy="dist_geo", seed=policy.seed)

        # Balanced round-robin: preferred fold is (i % k); shift to a smaller
        # fold if the preferred one is already 2+ ahead globally.
        for i, ip in enumerate(ordered):
            preferred = i % policy.k
            min_size = min(fold_sizes.values())
            if fold_sizes[preferred] <= min_size + 1:
                fold = preferred
            else:
                fold = _smallest_fold(fold_sizes)
            out[ip] = fold
            fold_sizes[fold] += 1

    return out


def _smallest_fold(fold_sizes: dict[int, int]) -> int:
    """Return the fold with the smallest current size; tiebreak by lowest
    fold_index. Used by DistGeo K-fold for balanced placement across
    buckets."""
    return min(fold_sizes, key=lambda f: (fold_sizes[f], f))
