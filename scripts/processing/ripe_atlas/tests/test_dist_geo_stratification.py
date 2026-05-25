"""Tests for the DistGeoStratification anchor splitter.

Algorithmic invariants: determinism, disjointness, ASN balance, intra-fold
spatial spread, edge cases, policy validation. Integration with
RipeAtlasSource is exercised in test_sources.py (parametrized).
"""

from __future__ import annotations

import math
import unittest

from scripts.processing.ripe_atlas.stratification import (
    AnchorInfo,
    DistGeoStratification,
    _bucket_asns,
    compute_dist_geo_fold_assignments,
)


def _synth_anchors(n: int = 50, n_countries: int = 6, n_asns: int = 10) -> list[AnchorInfo]:
    """Same synthetic helper as test_holdout.py — `n` anchors dealt round-robin
    into countries and ASNs, lat/lon on a coarse grid. Keeps tests independent."""
    countries = [f"C{i}" for i in range(n_countries)]
    asns = list(range(1000, 1000 + n_asns))
    out: list[AnchorInfo] = []
    for i in range(n):
        lat = -60 + (i % 12) * 10
        lon = -150 + ((i // 12) % 30) * 10
        out.append(AnchorInfo(
            ip=f"10.0.{i // 256}.{i % 256}",
            lat=float(lat),
            lon=float(lon),
            country=countries[i % n_countries],
            asn=asns[i % n_asns],
        ))
    return out


def _haversine_km(a: AnchorInfo, b: AnchorInfo) -> float:
    """Minimal pure-python haversine, only used for spatial-spread assertions
    so the test doesn't depend on scripts.utils.helpers being importable."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6367.0 * 2 * math.asin(math.sqrt(h))


class TestDistGeoDeterminism(unittest.TestCase):
    def test_same_seed_same_assignments(self) -> None:
        anchors = _synth_anchors(60)
        policy = DistGeoStratification(k=5, fold_index=0, seed=7)
        a = compute_dist_geo_fold_assignments(anchors, policy)
        b = compute_dist_geo_fold_assignments(anchors, policy)
        self.assertEqual(a, b)

    def test_anchor_input_order_does_not_matter(self) -> None:
        """The algorithm sorts anchors by IP early; permuted input must
        produce the same fold map."""
        anchors = _synth_anchors(40)
        policy = DistGeoStratification(k=4, fold_index=0, seed=42)
        a = compute_dist_geo_fold_assignments(anchors, policy)
        b = compute_dist_geo_fold_assignments(list(reversed(anchors)), policy)
        self.assertEqual(a, b)

    def test_seed_is_honored_without_crash(self) -> None:
        """Note: unlike Sechidis (where the seed feeds many tiebreak draws),
        dist_geo's seed enters only via `select_vps._max_edge_start` — the
        random pick of one endpoint of the max-distance edge. On symmetric
        corpora both endpoints produce equivalent orderings, so the seed may
        have no observable effect. We just check the seed parameter is
        accepted and produces a valid mapping; we do *not* assert that
        different seeds produce different outputs."""
        anchors = _synth_anchors(60)
        for seed in (1, 7, 42, 999):
            p = DistGeoStratification(k=5, fold_index=0, seed=seed)
            m = compute_dist_geo_fold_assignments(anchors, p)
            self.assertEqual(set(m), {a.ip for a in anchors})
            self.assertTrue(all(0 <= f < 5 for f in m.values()))


class TestDistGeoFoldDisjointness(unittest.TestCase):
    def test_every_anchor_in_exactly_one_fold(self) -> None:
        anchors = _synth_anchors(73)
        k = 5
        policy = DistGeoStratification(k=k, fold_index=0, seed=42)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)
        self.assertEqual(set(fold_by_ip), {a.ip for a in anchors})
        self.assertTrue(all(0 <= f < k for f in fold_by_ip.values()))

    def test_fold_sizes_are_balanced(self) -> None:
        """Balanced round-robin should keep |max-min| ≤ 1 globally even when
        ASN buckets are uneven."""
        anchors = _synth_anchors(100)
        k = 5
        policy = DistGeoStratification(k=k, fold_index=0, seed=42)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)
        counts = [sum(1 for f in fold_by_ip.values() if f == i) for i in range(k)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_empty_input(self) -> None:
        policy = DistGeoStratification(k=5, fold_index=0, seed=42)
        self.assertEqual(compute_dist_geo_fold_assignments([], policy), {})


class TestDistGeoAsnBalance(unittest.TestCase):
    def test_top_n_buckets_balanced_across_folds(self) -> None:
        """Each top-N ASN bucket's anchors should be split into ⌈count/K⌉ or
        ⌊count/K⌋ per fold."""
        # 5 ASNs × 10 anchors each = 50 anchors. With K=5 → expect 2 per (fold, ASN).
        anchors = []
        for asn in range(1000, 1005):
            for j in range(10):
                anchors.append(AnchorInfo(
                    ip=f"10.0.{asn - 1000}.{j}",
                    lat=-60.0 + (j * 12),
                    lon=-150.0 + (asn % 360),
                    country="US",
                    asn=asn,
                ))
        policy = DistGeoStratification(k=5, fold_index=0, seed=42, asn_bucket_top_n=5)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)

        # Count (asn, fold) cells.
        by_asn: dict[int, list[int]] = {}
        for a in anchors:
            by_asn.setdefault(a.asn or 0, []).append(fold_by_ip[a.ip])
        for asn, folds in by_asn.items():
            counts = [folds.count(i) for i in range(5)]
            self.assertLessEqual(
                max(counts) - min(counts), 1,
                f"asn={asn} counts={counts}",
            )

    def test_other_AS_bucket_balanced(self) -> None:
        """When top_n=2, the rest collapse into 'other_AS' and the larger
        pooled bucket should still spread evenly across folds."""
        # ASN 1: 4 anchors (top-N). ASN 2: 4 anchors (top-N). ASNs 3..7: 1 each → "other_AS".
        anchors = []
        ip_counter = 0
        for asn, count in [(1, 4), (2, 4), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]:
            for j in range(count):
                anchors.append(AnchorInfo(
                    ip=f"10.0.0.{ip_counter}",
                    lat=-60.0 + ip_counter * 5,
                    lon=-150.0 + ip_counter * 10,
                    country="US",
                    asn=asn,
                ))
                ip_counter += 1

        policy = DistGeoStratification(k=5, fold_index=0, seed=42, asn_bucket_top_n=2)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)

        # 13 anchors / 5 folds → 2-3 per fold globally; balance constraint ≤ 1.
        counts = [sum(1 for f in fold_by_ip.values() if f == i) for i in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)


class TestDistGeoSpatialSpread(unittest.TestCase):
    def test_intra_fold_spread_at_least_half_of_corpus_spread(self) -> None:
        """Sanity guard against degenerate clustering — each fold's anchors
        should be spread, not piled in one region. Use a 5-cluster
        synthetic corpus: 5 tight metros × 8 anchors each. Round-robin via
        dist_geo should ensure each fold sees multiple metros.

        Spec: per-fold mean pairwise distance ≥ 0.5 × corpus mean pairwise
        distance. (Empirically dist_geo lands around 0.8–1.0× on this
        corpus; 0.5× is a wide safety margin.)"""
        cluster_centers = [
            (40.0, -100.0),  # North America
            (50.0, 10.0),    # Central Europe
            (35.0, 135.0),   # East Asia
            (-15.0, -50.0),  # South America
            (-30.0, 130.0),  # Australia
        ]
        anchors = []
        ip = 0
        for c_lat, c_lon in cluster_centers:
            for j in range(8):
                anchors.append(AnchorInfo(
                    ip=f"10.0.{ip // 256}.{ip % 256}",
                    lat=c_lat + j * 0.1, lon=c_lon + j * 0.1,
                    country="C",
                    asn=1000 + (ip % 5),  # 5 ASNs cycled
                ))
                ip += 1

        policy = DistGeoStratification(k=5, fold_index=0, seed=42, asn_bucket_top_n=10)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)

        by_ip = {a.ip: a for a in anchors}

        def mean_pairwise(ips: list[str]) -> float:
            dists = []
            for i, ip1 in enumerate(ips):
                for ip2 in ips[i + 1:]:
                    dists.append(_haversine_km(by_ip[ip1], by_ip[ip2]))
            return sum(dists) / len(dists) if dists else 0.0

        corpus_mean = mean_pairwise([a.ip for a in anchors])
        for fold in range(5):
            fold_ips = [ip for ip, f in fold_by_ip.items() if f == fold]
            fold_mean = mean_pairwise(fold_ips)
            self.assertGreaterEqual(
                fold_mean, 0.5 * corpus_mean,
                f"fold {fold} mean pairwise {fold_mean:.0f}km < 0.5× corpus mean "
                f"{corpus_mean:.0f}km — folds look degenerate-clustered",
            )


class TestDistGeoEdgeCases(unittest.TestCase):
    def test_singleton_bucket_lands_in_a_fold(self) -> None:
        """Bucket with only 1 anchor: skips dist_geo, goes straight to
        smallest-fold placement."""
        anchors = _synth_anchors(40)
        anchors.append(AnchorInfo(ip="9.9.9.9", lat=0.0, lon=0.0, country="ZZ", asn=99999))
        policy = DistGeoStratification(k=5, fold_index=0, seed=42)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)
        self.assertIn("9.9.9.9", fold_by_ip)
        self.assertIn(fold_by_ip["9.9.9.9"], range(5))

    def test_fewer_anchors_than_folds(self) -> None:
        """3 anchors, K=5: must still produce valid assignment; some folds empty."""
        anchors = [
            AnchorInfo(ip="1.1.1.1", lat=0.0, lon=0.0, country="US", asn=1),
            AnchorInfo(ip="2.2.2.2", lat=10.0, lon=10.0, country="DE", asn=2),
            AnchorInfo(ip="3.3.3.3", lat=-10.0, lon=-10.0, country="JP", asn=3),
        ]
        policy = DistGeoStratification(k=5, fold_index=0, seed=42)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)
        self.assertEqual(len(fold_by_ip), 3)
        # Each anchor in a singleton bucket → all hit _smallest_fold,
        # tiebreak by fold_index ascending, so they land in folds 0, 1, 2.
        self.assertEqual(set(fold_by_ip.values()), {0, 1, 2})

    def test_none_asn_treated_as_own_bucket(self) -> None:
        """Anchors with asn=None get bucketed as 'asn_none' — they share a
        bucket and are dist_geo-ordered together."""
        anchors = []
        for i in range(6):
            anchors.append(AnchorInfo(
                ip=f"10.0.0.{i}",
                lat=float(i * 15),
                lon=float(i * 30),
                country="US",
                asn=None,
            ))
        policy = DistGeoStratification(k=3, fold_index=0, seed=42, asn_bucket_top_n=10)
        fold_by_ip = compute_dist_geo_fold_assignments(anchors, policy)
        counts = [sum(1 for f in fold_by_ip.values() if f == i) for i in range(3)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_bucketing_consistent_with_holdout_policy(self) -> None:
        """_bucket_asns reuse — same bucket labels as SechidisStratification uses."""
        anchors = _synth_anchors(40)
        bucketed = _bucket_asns(anchors, top_n=5)
        # 40 anchors, 10 ASNs, round-robin → 4 anchors per ASN; top 5 → AS1000..AS1004; rest → other_AS
        labels = set(bucketed.values())
        self.assertTrue({"AS1000", "AS1001", "AS1002", "AS1003", "AS1004", "other_AS"}.issubset(labels))


class TestDistGeoStratificationValidation(unittest.TestCase):
    def test_k_must_be_at_least_two(self) -> None:
        with self.assertRaises(ValueError):
            DistGeoStratification(k=1)

    def test_fold_index_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            DistGeoStratification(k=5, fold_index=5)
        with self.assertRaises(ValueError):
            DistGeoStratification(k=5, fold_index=-1)

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DistGeoStratification(kind="random_kfold")

    def test_negative_asn_top_n_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DistGeoStratification(asn_bucket_top_n=-1)


if __name__ == "__main__":
    unittest.main()
