"""Tests for the Sechidis-style anchor holdout splitter.

Algorithmic invariants live here (determinism, disjointness, label balance,
spatial cluster atomicity, ASN bucketing). Integration with RipeAtlasSource
is exercised in test_sources.py.
"""

from __future__ import annotations

import unittest

from scripts.processing.ripe_atlas.holdout import (
    AnchorInfo,
    HoldoutPolicy,
    _bucket_asns,
    compute_fold_assignments,
)


def _synth_anchors(n: int = 50, n_countries: int = 6, n_asns: int = 10) -> list[AnchorInfo]:
    """Synthetic anchor corpus that covers most balance edge cases.

    `n` anchors are dealt round-robin into countries and ASNs so each label
    has a deterministic, easily-checked count. Lat/lon are spread across a
    grid (no clustering); spatial blocking tests build their own coords.
    """
    countries = [f"C{i}" for i in range(n_countries)]
    asns = list(range(1000, 1000 + n_asns))
    out: list[AnchorInfo] = []
    for i in range(n):
        # Spread lats over -60..60 and lons over -150..150 in a grid.
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


class TestSechidisDeterminism(unittest.TestCase):
    def test_same_seed_same_assignments(self) -> None:
        anchors = _synth_anchors(60)
        policy = HoldoutPolicy(k=5, fold_index=0, seed=7, spatial_clusters=None)
        a = compute_fold_assignments(anchors, policy)
        b = compute_fold_assignments(anchors, policy)
        self.assertEqual(a, b)

    def test_different_seed_can_differ(self) -> None:
        anchors = _synth_anchors(60)
        p1 = HoldoutPolicy(k=5, fold_index=0, seed=1, spatial_clusters=None)
        p2 = HoldoutPolicy(k=5, fold_index=0, seed=999, spatial_clusters=None)
        a = compute_fold_assignments(anchors, p1)
        b = compute_fold_assignments(anchors, p2)
        # Not strictly required to differ, but with N=60 and uniform labels
        # the seed should yield a different assignment most of the time.
        self.assertNotEqual(a, b)

    def test_anchor_input_order_does_not_matter(self) -> None:
        """The algorithm sorts anchors by IP early; permuted input must
        produce the same fold map."""
        anchors = _synth_anchors(40)
        policy = HoldoutPolicy(k=4, fold_index=0, seed=42, spatial_clusters=None)
        a = compute_fold_assignments(anchors, policy)
        b = compute_fold_assignments(list(reversed(anchors)), policy)
        self.assertEqual(a, b)


class TestFoldDisjointness(unittest.TestCase):
    def test_every_anchor_in_exactly_one_fold(self) -> None:
        anchors = _synth_anchors(73)
        k = 5
        # Compute once with fold_index=0 — assignment is the same regardless
        # of fold_index (that's just which fold becomes "test" downstream).
        policy = HoldoutPolicy(k=k, fold_index=0, seed=42, spatial_clusters=None)
        fold_by_ip = compute_fold_assignments(anchors, policy)
        self.assertEqual(set(fold_by_ip), {a.ip for a in anchors})
        self.assertTrue(all(0 <= f < k for f in fold_by_ip.values()))

    def test_fold_sizes_are_balanced(self) -> None:
        anchors = _synth_anchors(100)
        k = 5
        policy = HoldoutPolicy(k=k, fold_index=0, seed=42, spatial_clusters=None)
        fold_by_ip = compute_fold_assignments(anchors, policy)
        counts = [sum(1 for f in fold_by_ip.values() if f == i) for i in range(k)]
        # 100 anchors across 5 folds → 20 each in the ideal case; the greedy
        # algorithm should land within ±2 even with multi-label balancing.
        self.assertLessEqual(max(counts) - min(counts), 2)

    def test_empty_input(self) -> None:
        policy = HoldoutPolicy(k=5, fold_index=0, seed=42, spatial_clusters=None)
        self.assertEqual(compute_fold_assignments([], policy), {})


class TestLabelBalance(unittest.TestCase):
    def test_country_distribution_per_fold(self) -> None:
        """Each country's anchors should be spread across folds, not piled
        into one. With 60 anchors / 6 countries / 5 folds → 10 anchors per
        country, ~2 per (country, fold)."""
        anchors = _synth_anchors(60, n_countries=6, n_asns=10)
        policy = HoldoutPolicy(k=5, fold_index=0, seed=42, spatial_clusters=None)
        fold_by_ip = compute_fold_assignments(anchors, policy)

        # Count (country, fold) cells.
        by_country: dict[str, list[int]] = {}
        for a in anchors:
            by_country.setdefault(a.country or "none", []).append(fold_by_ip[a.ip])
        for country, folds in by_country.items():
            counts = [folds.count(i) for i in range(5)]
            # 10 anchors / 5 folds = 2 expected → max-min ≤ 1 in the ideal case.
            self.assertLessEqual(
                max(counts) - min(counts), 2,
                f"country={country} counts={counts}",
            )

    def test_singleton_label_goes_to_one_fold(self) -> None:
        """An ASN with exactly one anchor → that anchor lives in one fold;
        the other K-1 folds don't see this ASN at all (unavoidable)."""
        anchors = _synth_anchors(40)
        # Add a singleton-ASN anchor.
        singleton = AnchorInfo(ip="9.9.9.9", lat=0.0, lon=0.0, country="ZZ", asn=99999)
        anchors.append(singleton)

        policy = HoldoutPolicy(k=5, fold_index=0, seed=42, spatial_clusters=None)
        fold_by_ip = compute_fold_assignments(anchors, policy)
        self.assertIn("9.9.9.9", fold_by_ip)


class TestAsnBucketing(unittest.TestCase):
    def test_top_n_buckets_plus_other(self) -> None:
        # Build anchors where ASN 1 has 5 anchors, ASN 2 has 3, ASN 3 has 2,
        # and ASNs 4..10 each have 1 anchor.
        anchors = []
        ip_counter = 0
        for asn, count in [(1, 5), (2, 3), (3, 2)] + [(i, 1) for i in range(4, 11)]:
            for _ in range(count):
                anchors.append(AnchorInfo(
                    ip=f"10.0.0.{ip_counter}", lat=0.0, lon=0.0,
                    country="US", asn=asn,
                ))
                ip_counter += 1
        bucketed = _bucket_asns(anchors, top_n=2)
        # Top 2 ASNs (1 and 2) keep their own bucket; the rest → "other_AS".
        labels = set(bucketed.values())
        self.assertEqual(labels, {"AS1", "AS2", "other_AS"})

    def test_none_asn_gets_own_bucket(self) -> None:
        anchors = [
            AnchorInfo(ip="1.1.1.1", lat=0.0, lon=0.0, country="US", asn=None),
            AnchorInfo(ip="2.2.2.2", lat=0.0, lon=0.0, country="US", asn=42),
        ]
        bucketed = _bucket_asns(anchors, top_n=10)
        self.assertEqual(bucketed["1.1.1.1"], "asn_none")
        self.assertEqual(bucketed["2.2.2.2"], "AS42")


class TestSpatialClustering(unittest.TestCase):
    def test_clustered_anchors_share_a_fold(self) -> None:
        """Build two tight geographic clusters (Europe-ish + South-America-ish)
        plus a small third cluster (Asia). With spatial_clusters=3, anchors
        in the same physical cluster should end up in the same fold."""
        anchors = []
        # Cluster A: tight around (50, 10) — central Europe.
        for i in range(8):
            anchors.append(AnchorInfo(
                ip=f"10.0.0.{i}", lat=50.0 + i * 0.1, lon=10.0 + i * 0.1,
                country="DE", asn=100,
            ))
        # Cluster B: tight around (-15, -50) — South America.
        for i in range(8):
            anchors.append(AnchorInfo(
                ip=f"10.0.1.{i}", lat=-15.0 + i * 0.1, lon=-50.0 + i * 0.1,
                country="BR", asn=200,
            ))
        # Cluster C: tight around (35, 135) — East Asia.
        for i in range(6):
            anchors.append(AnchorInfo(
                ip=f"10.0.2.{i}", lat=35.0 + i * 0.1, lon=135.0 + i * 0.1,
                country="JP", asn=300,
            ))

        policy = HoldoutPolicy(
            k=3, fold_index=0, seed=42,
            spatial_clusters=3,
        )
        fold_by_ip = compute_fold_assignments(anchors, policy)

        # Each tight cluster should land entirely in one fold.
        folds_a = {fold_by_ip[f"10.0.0.{i}"] for i in range(8)}
        folds_b = {fold_by_ip[f"10.0.1.{i}"] for i in range(8)}
        folds_c = {fold_by_ip[f"10.0.2.{i}"] for i in range(6)}
        self.assertEqual(len(folds_a), 1, f"cluster A spans folds {folds_a}")
        self.assertEqual(len(folds_b), 1, f"cluster B spans folds {folds_b}")
        self.assertEqual(len(folds_c), 1, f"cluster C spans folds {folds_c}")

    def test_spatial_clusters_greater_than_n_clamps(self) -> None:
        """spatial_clusters > #anchors should not crash; it clamps to N."""
        anchors = _synth_anchors(5)
        policy = HoldoutPolicy(k=2, fold_index=0, seed=42, spatial_clusters=100)
        fold_by_ip = compute_fold_assignments(anchors, policy)
        self.assertEqual(len(fold_by_ip), 5)


class TestHoldoutPolicyValidation(unittest.TestCase):
    def test_k_must_be_at_least_two(self) -> None:
        with self.assertRaises(ValueError):
            HoldoutPolicy(k=1)

    def test_fold_index_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            HoldoutPolicy(k=5, fold_index=5)
        with self.assertRaises(ValueError):
            HoldoutPolicy(k=5, fold_index=-1)

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HoldoutPolicy(kind="bloo")

    def test_unknown_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HoldoutPolicy(labels=("country", "continent"))

    def test_slice_suffix_format(self) -> None:
        p = HoldoutPolicy(k=5, fold_index=2, seed=99)
        self.assertEqual(p.slice_suffix(), "fold2of5_seed99")


if __name__ == "__main__":
    unittest.main()
