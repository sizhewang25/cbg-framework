"""Tests for scripts.vp_selection.iclab_verifier.

TDD-first. The verifier matches Niaki 2020 Appendix B / Cho 2024 §III: given a
target's claimed country and a set of landmarks with measured RTTs, ACCEPT if
no landmark's implied one-way propagation speed exceeds the calibrated limit;
REJECT as soon as one does.
"""

from __future__ import annotations

import unittest

from scripts.vp_selection.iclab_verifier import iclab_verify

S = 168.62  # km/ms — calibrated speed limit from Step 1 (matches our agreement defaults)


class TestIclabVerify(unittest.TestCase):

    def test_empty_landmarks_accepts(self):
        """No measurements → trivially accept (no evidence against the claim)."""
        self.assertEqual(
            iclab_verify(
                landmark_rtts={},
                claimed_country="US",
                distances={},
                speed_limit_km_per_ms=S,
            ),
            "accept",
        )

    def test_single_slow_landmark_accepts(self):
        """RTT=20 ms, distance to border=100 km → implied one-way = 2*100/20 = 10 km/ms,
        well below S = 168.62 → ACCEPT."""
        verdict = iclab_verify(
            landmark_rtts={"lm1": 20.0},
            claimed_country="US",
            distances={("lm1", "US"): 100.0},
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "accept")

    def test_single_fast_landmark_rejects(self):
        """RTT=1 ms, distance to border=200 km → implied one-way = 2*200/1 = 400 km/ms,
        above S → REJECT."""
        verdict = iclab_verify(
            landmark_rtts={"lm1": 1.0},
            claimed_country="US",
            distances={("lm1", "US"): 200.0},
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "reject")

    def test_any_one_violation_rejects(self):
        """One violating landmark is enough — even if others accept."""
        verdict = iclab_verify(
            landmark_rtts={"slow": 50.0, "fast": 0.5},
            claimed_country="US",
            distances={
                ("slow", "US"): 100.0,   # implied 4 km/ms — fine
                ("fast", "US"): 200.0,   # implied 800 km/ms — violation
            },
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "reject")

    def test_threshold_exactly_at_S_accepts(self):
        """The check is strict `>`. At implied_v == S exactly, the verifier
        accepts. (Matches Cho's quote: 'greater than a calibrated speed limit'.)"""
        # Pick numbers s.t. implied_v = 2 * d / rtt == S
        # 2 * 84.31 / 1.0 = 168.62
        verdict = iclab_verify(
            landmark_rtts={"lm": 1.0},
            claimed_country="US",
            distances={("lm", "US"): S / 2.0},  # gives implied_v = S exactly
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "accept")

    def test_threshold_just_above_S_rejects(self):
        """A hair over S rejects."""
        d_just_over = (S / 2.0) + 1e-6  # implied_v slightly above S
        verdict = iclab_verify(
            landmark_rtts={"lm": 1.0},
            claimed_country="US",
            distances={("lm", "US"): d_just_over},
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "reject")

    def test_zero_rtt_landmarks_are_skipped(self):
        """rtt=0 would divide by zero. Verifier must skip these landmarks
        rather than crash."""
        verdict = iclab_verify(
            landmark_rtts={"zero": 0.0, "good": 20.0},
            claimed_country="US",
            distances={("zero", "US"): 1000.0, ("good", "US"): 100.0},
            speed_limit_km_per_ms=S,
        )
        # Should not crash; verdict reflects only the 'good' landmark
        self.assertEqual(verdict, "accept")

    def test_negative_rtt_landmarks_are_skipped(self):
        verdict = iclab_verify(
            landmark_rtts={"bad": -5.0, "good": 20.0},
            claimed_country="US",
            distances={("bad", "US"): 1000.0, ("good", "US"): 100.0},
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "accept")

    def test_missing_distance_lookup_skipped(self):
        """If the precomputed distance for a (landmark, country) pair is
        missing, skip that landmark (verifier degrades gracefully rather than
        rejecting on uncertainty)."""
        verdict = iclab_verify(
            landmark_rtts={"lm": 1.0},
            claimed_country="US",
            distances={},  # no entry for ("lm", "US")
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "accept")

    def test_landmark_inside_country_accepts(self):
        """Distance 0 means implied_v = 0 / owtt = 0, definitely below S."""
        verdict = iclab_verify(
            landmark_rtts={"lm": 0.001},  # any positive rtt
            claimed_country="US",
            distances={("lm", "US"): 0.0},
            speed_limit_km_per_ms=S,
        )
        self.assertEqual(verdict, "accept")


if __name__ == "__main__":
    unittest.main()
