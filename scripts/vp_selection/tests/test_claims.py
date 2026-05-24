"""Tests for scripts.vp_selection.claims.

TDD-first: written before claims.py exists. Drives the `assign_claims` API +
its determinism / rate / no-self-claim invariants.
"""

from __future__ import annotations

import unittest

from scripts.vp_selection.claims import assign_claims


def _targets(n: int) -> dict[str, str]:
    """Build n synthetic targets with a small set of countries to choose from."""
    countries = ["US", "FR", "DE", "JP", "BR"]
    return {f"T{i:04d}": countries[i % len(countries)] for i in range(n)}


class TestAssignClaims(unittest.TestCase):

    def test_empty_targets_returns_empty(self):
        result = assign_claims(
            target_countries={},
            all_countries=["US", "FR"],
            fake_fraction=0.3,
            seed=0,
        )
        self.assertEqual(result, [])

    def test_returns_one_row_per_target(self):
        targets = _targets(100)
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.3,
            seed=0,
        )
        self.assertEqual(len(result), 100)
        self.assertEqual({r["target_id"] for r in result}, set(targets))

    def test_row_schema(self):
        targets = _targets(10)
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.5,
            seed=0,
        )
        expected_keys = {"target_id", "real_country", "claimed_country", "is_real"}
        for r in result:
            self.assertEqual(set(r.keys()), expected_keys)
            self.assertIsInstance(r["is_real"], bool)

    def test_determinism(self):
        targets = _targets(50)
        a = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.3,
            seed=42,
        )
        b = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.3,
            seed=42,
        )
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        targets = _targets(50)
        a = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.3,
            seed=0,
        )
        b = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.3,
            seed=1,
        )
        # Some difference somewhere — fake assignment is stochastic
        self.assertNotEqual(a, b)

    def test_fake_fraction_zero_means_all_real(self):
        targets = _targets(100)
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE"],
            fake_fraction=0.0,
            seed=0,
        )
        for r in result:
            self.assertTrue(r["is_real"])
            self.assertEqual(r["claimed_country"], r["real_country"])

    def test_fake_fraction_one_means_all_fake(self):
        targets = _targets(100)
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=1.0,
            seed=0,
        )
        for r in result:
            self.assertFalse(r["is_real"])
            self.assertNotEqual(r["claimed_country"], r["real_country"])

    def test_fake_fraction_approximate_rate_within_5pct(self):
        """At N=2000 with fake_fraction=0.3, expect ~600 fakes ± 5%."""
        targets = _targets(2000)
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.3,
            seed=0,
        )
        n_fake = sum(1 for r in result if not r["is_real"])
        # 0.3 ± 0.05 of 2000 → 500-700
        self.assertGreaterEqual(n_fake, 500)
        self.assertLessEqual(n_fake, 700)

    def test_fake_country_never_equals_real(self):
        targets = _targets(300)
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.5,
            seed=0,
        )
        for r in result:
            if not r["is_real"]:
                self.assertNotEqual(r["claimed_country"], r["real_country"])

    def test_single_country_universe_forces_real_claims(self):
        """If `all_countries` has only one entry and every target is in it,
        we can't pick a different country — fall back to keeping the real
        claim rather than crashing."""
        targets = {"T1": "US", "T2": "US"}
        result = assign_claims(
            target_countries=targets,
            all_countries=["US"],
            fake_fraction=1.0,
            seed=0,
        )
        for r in result:
            self.assertTrue(r["is_real"])

    def test_target_with_unknown_country_skipped(self):
        """A target whose real_country isn't in `all_countries` would have
        the entire universe as alternatives. We should still produce a row
        for it — that's a normal scenario when the universe is the union
        of polygon-known country codes."""
        targets = {"T1": "ZZ"}  # ZZ not in known countries
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR"],
            fake_fraction=1.0,
            seed=0,
        )
        self.assertEqual(len(result), 1)
        # Could be either US or FR; just not ZZ
        self.assertIn(result[0]["claimed_country"], ["US", "FR"])
        self.assertFalse(result[0]["is_real"])

    def test_real_claim_keeps_real_country(self):
        targets = _targets(20)
        result = assign_claims(
            target_countries=targets,
            all_countries=["US", "FR", "DE", "JP", "BR"],
            fake_fraction=0.0,
            seed=0,
        )
        for r in result:
            self.assertEqual(r["real_country"], r["claimed_country"])


if __name__ == "__main__":
    unittest.main()
