"""Tests for scripts.vp_selection.agreement.

Focuses on the pure compute helpers so we can test without ClickHouse,
parquet files, or a real shapefile.
"""

from __future__ import annotations

import unittest

from scripts.vp_selection.agreement import (
    agreement_curve,
    compute_sampling_rows,
    compute_sequence_rows,
    detection_curve,
    find_first_violator,
    sub_verdict_at_k,
)


class TestSubVerdictAtK(unittest.TestCase):

    def test_none_violator_always_accepts(self):
        for k in (1, 10, 100):
            self.assertEqual(sub_verdict_at_k(None, k), "accept")

    def test_below_first_violator_accepts(self):
        self.assertEqual(sub_verdict_at_k(5, 1), "accept")
        self.assertEqual(sub_verdict_at_k(5, 4), "accept")

    def test_at_or_above_first_violator_rejects(self):
        self.assertEqual(sub_verdict_at_k(5, 5), "reject")
        self.assertEqual(sub_verdict_at_k(5, 10), "reject")


class TestFindFirstViolator(unittest.TestCase):

    def test_no_violator_returns_none(self):
        self.assertIsNone(find_first_violator(
            selection=["a", "b"],
            tgt_rtts={"a": 50.0, "b": 50.0},
            claim="US",
            distances={("a", "US"): 100.0, ("b", "US"): 100.0},
            speed_limit_km_per_ms=168.62,
        ))

    def test_first_violator_position(self):
        # 'b' at position 2 violates: distance=200, rtt=1 → speed = 400 km/ms
        result = find_first_violator(
            selection=["a", "b", "c"],
            tgt_rtts={"a": 50.0, "b": 1.0, "c": 50.0},
            claim="US",
            distances={
                ("a", "US"): 100.0,
                ("b", "US"): 200.0,
                ("c", "US"): 100.0,
            },
            speed_limit_km_per_ms=168.62,
        )
        self.assertEqual(result, 2)

    def test_skips_zero_and_negative_rtts(self):
        result = find_first_violator(
            selection=["bad", "good"],
            tgt_rtts={"bad": 0.0, "good": 50.0},
            claim="US",
            distances={("bad", "US"): 1000.0, ("good", "US"): 100.0},
            speed_limit_km_per_ms=168.62,
        )
        self.assertIsNone(result)


class TestComputeSequenceRows(unittest.TestCase):

    def test_emits_one_row_per_strategy_seed_target_claim(self):
        rows = compute_sequence_rows(
            selection_sequences={
                ("h1_as", 0): ["a", "b"],
                ("h1_as", 1): ["b", "a"],
            },
            full_landmark_rtts_by_target={
                "T1": {"a": 50.0, "b": 50.0},
                "T2": {"a": 50.0, "b": 50.0},
            },
            targets_and_claims=[("T1", "US", True), ("T2", "FR", False)],
            distances={
                ("a", "US"): 100.0, ("b", "US"): 100.0,
                ("a", "FR"): 100.0, ("b", "FR"): 100.0,
            },
            speed_limit_km_per_ms=168.62,
            full_verdicts={("T1", "US"): "accept", ("T2", "FR"): "accept"},
        )
        # 2 strategy-seed × 2 (target, claim) = 4 rows
        self.assertEqual(len(rows), 4)
        for r in rows:
            self.assertIn(r["strategy"], ("h1_as",))
            self.assertIn(r["seed"], (0, 1))
            self.assertEqual(r["k"], None)
            self.assertEqual(r["selection_length"], 2)

    def test_first_violator_recorded(self):
        rows = compute_sequence_rows(
            selection_sequences={("h1_as", 0): ["a", "b"]},
            full_landmark_rtts_by_target={"T": {"a": 50.0, "b": 1.0}},
            targets_and_claims=[("T", "FR", False)],
            distances={("a", "FR"): 100.0, ("b", "FR"): 200.0},
            speed_limit_km_per_ms=168.62,
            full_verdicts={("T", "FR"): "reject"},
        )
        self.assertEqual(rows[0]["first_violator_k"], 2)


class TestComputeSamplingRows(unittest.TestCase):

    def test_emits_one_row_per_subset_target_claim(self):
        rows = compute_sampling_rows(
            selection_subsets={
                ("random", 0, 5): ["a", "b"],
                ("random", 0, 10): ["a", "b", "c"],
            },
            full_landmark_rtts_by_target={
                "T1": {"a": 50.0, "b": 50.0, "c": 50.0},
            },
            targets_and_claims=[("T1", "US", True)],
            distances={
                ("a", "US"): 100.0, ("b", "US"): 100.0, ("c", "US"): 100.0,
            },
            speed_limit_km_per_ms=168.62,
            full_verdicts={("T1", "US"): "accept"},
        )
        # 2 (strategy, seed, k) × 1 target = 2 rows
        self.assertEqual(len(rows), 2)
        ks = {r["k"] for r in rows}
        self.assertEqual(ks, {5, 10})
        for r in rows:
            self.assertEqual(r["first_violator_k"], None)
            self.assertIn(r["sub_verdict"], ("accept", "reject"))

    def test_subset_filter_excludes_outside_landmarks(self):
        """'b' violates the claim but the subset only contains 'a' → no violation."""
        rows = compute_sampling_rows(
            selection_subsets={("random", 0, 1): ["a"]},
            full_landmark_rtts_by_target={"T": {"a": 50.0, "b": 1.0}},
            targets_and_claims=[("T", "FR", False)],
            distances={("a", "FR"): 100.0, ("b", "FR"): 200.0},
            speed_limit_km_per_ms=168.62,
            full_verdicts={("T", "FR"): "reject"},
        )
        # subset has only 'a' which doesn't violate → sub = accept
        self.assertEqual(rows[0]["sub_verdict"], "accept")
        self.assertEqual(rows[0]["full_verdict"], "reject")


class TestAgreementCurve(unittest.TestCase):

    def test_sequence_row_derives_per_K_verdict(self):
        rows = [
            # sequence row: first_violator at K=5; for K<5 accept, K>=5 reject
            {"strategy": "s", "seed": 0, "target_id": "T1",
             "claimed_country": "FR", "is_real": False,
             "full_verdict": "reject", "first_violator_k": 5,
             "selection_length": 100, "k": None, "sub_verdict": None},
        ]
        curve = agreement_curve(rows, k_grid=[1, 5, 10])
        # K=1: sub=accept, full=reject → disagree → rate 0
        self.assertEqual(curve[("s", 1, False)]["rate"], 0.0)
        # K=5: sub=reject, full=reject → agree → rate 1
        self.assertEqual(curve[("s", 5, False)]["rate"], 1.0)
        # K=10: sub=reject, full=reject → agree → rate 1
        self.assertEqual(curve[("s", 10, False)]["rate"], 1.0)

    def test_sampling_row_contributes_only_at_its_K(self):
        rows = [
            # sampling row at K=5, agrees
            {"strategy": "s", "seed": 0, "target_id": "T1",
             "claimed_country": "FR", "is_real": False,
             "full_verdict": "reject", "first_violator_k": None,
             "selection_length": None, "k": 5, "sub_verdict": "reject"},
        ]
        curve = agreement_curve(rows, k_grid=[1, 5, 10])
        # Only K=5 has any data
        self.assertIn(("s", 5, False), curve)
        self.assertNotIn(("s", 1, False), curve)
        self.assertEqual(curve[("s", 5, False)]["rate"], 1.0)


class TestDetectionCurve(unittest.TestCase):

    def test_tpr_fpr_per_strategy_k(self):
        rows = [
            # Fake row, sampling K=5, sub=reject → TP
            {"strategy": "s", "seed": 0, "target_id": "T_fake",
             "claimed_country": "FR", "is_real": False,
             "full_verdict": "reject", "first_violator_k": None,
             "selection_length": None, "k": 5, "sub_verdict": "reject"},
            # Real row, sampling K=5, sub=accept → not FP
            {"strategy": "s", "seed": 0, "target_id": "T_real",
             "claimed_country": "US", "is_real": True,
             "full_verdict": "accept", "first_violator_k": None,
             "selection_length": None, "k": 5, "sub_verdict": "accept"},
        ]
        det = detection_curve(rows, k_grid=[5])
        self.assertEqual(det[("s", 5)]["tpr"], 1.0)
        self.assertEqual(det[("s", 5)]["fpr"], 0.0)
        self.assertEqual(det[("s", 5)]["n_fake"], 1)
        self.assertEqual(det[("s", 5)]["n_real"], 1)


if __name__ == "__main__":
    unittest.main()
