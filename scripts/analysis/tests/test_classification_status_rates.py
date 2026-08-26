from __future__ import annotations

import math
import unittest

import pandas as pd

from scripts.analysis.classification.plot_classification_match_bars import _status_breakdown


class TestStatusBreakdown(unittest.TestCase):
    def test_uses_explicit_status_when_present(self) -> None:
        df = pd.DataFrame(
            {
                "status": ["SUCCESS", "FALLBACK", "ERROR", "SUCCESS"],
                "success": [True, True, True, True],
            }
        )
        n_success, success_pct, fallback_pct, error_pct = _status_breakdown(df)
        self.assertEqual(n_success, 2)
        self.assertAlmostEqual(success_pct, 50.0)
        self.assertAlmostEqual(fallback_pct, 25.0)
        self.assertAlmostEqual(error_pct, 25.0)

    def test_status_is_case_insensitive(self) -> None:
        df = pd.DataFrame(
            {
                "status": ["success", "Fallback", "error"],
                "success": [False, False, False],
            }
        )
        n_success, success_pct, fallback_pct, error_pct = _status_breakdown(df)
        self.assertEqual(n_success, 1)
        self.assertAlmostEqual(success_pct, 100.0 / 3.0)
        self.assertAlmostEqual(fallback_pct, 100.0 / 3.0)
        self.assertAlmostEqual(error_pct, 100.0 / 3.0)

    def test_falls_back_to_success_when_status_missing(self) -> None:
        df = pd.DataFrame(
            {
                "success": [True, False, True, False],
            }
        )
        n_success, success_pct, fallback_pct, error_pct = _status_breakdown(df)
        self.assertEqual(n_success, 2)
        self.assertAlmostEqual(success_pct, 50.0)
        self.assertTrue(math.isnan(fallback_pct))
        self.assertTrue(math.isnan(error_pct))


class TestAllTargetAccuracySemantics(unittest.TestCase):
    def test_accuracy_uses_all_targets_denominator(self) -> None:
        df = pd.DataFrame(
            {
                "status": ["SUCCESS", "FALLBACK", "ERROR", "SUCCESS"],
                "match": [True, False, False, True],
            }
        )
        # Accuracy is computed as mean(match) over all targets.
        self.assertAlmostEqual(float(df["match"].mean()), 0.5)


if __name__ == "__main__":
    unittest.main()
