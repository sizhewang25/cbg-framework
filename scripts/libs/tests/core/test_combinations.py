"""Tests for CBG evaluation combination registry semantics."""

from __future__ import annotations

import unittest

from scripts.libs.core.combinations import (
    COMBINATIONS,
    DIFF_PAIRS,
    SPECS_BY_ID,
)
from scripts.libs.core.evaluate import build_pipeline


class TestCombinations(unittest.TestCase):
    def test_combo_ids_are_unique(self):
        combo_ids = [spec.combo_id for spec in COMBINATIONS]

        self.assertEqual(len(combo_ids), len(set(combo_ids)))

    def test_diff_pairs_reference_registered_combinations(self):
        for id_a, id_b in DIFF_PAIRS:
            self.assertIn(id_a, SPECS_BY_ID)
            self.assertIn(id_b, SPECS_BY_ID)

    def test_filtering_ablation_pairs_share_color_only(self):
        for id_a, id_b in [("S1", "S2"), ("L1", "L2")]:
            spec_a = SPECS_BY_ID[id_a]
            spec_b = SPECS_BY_ID[id_b]

            self.assertEqual(spec_a.distance, spec_b.distance)
            self.assertEqual(spec_a.multilateration, spec_b.multilateration)
            self.assertEqual(spec_a.centroid, spec_b.centroid)
            self.assertNotEqual(spec_a.filtering, spec_b.filtering)
            self.assertEqual(spec_a.color, spec_b.color)
            self.assertNotEqual(spec_a.linestyle, spec_b.linestyle)

    def test_weighted_annulus_pairs_share_color_only(self):
        for id_a, id_b in [("B1", "B2"), ("B3", "B4"), ("B5", "B2"), ("B6", "B4")]:
            spec_a = SPECS_BY_ID[id_a]
            spec_b = SPECS_BY_ID[id_b]

            self.assertEqual(spec_a.distance, spec_b.distance)
            self.assertEqual(spec_a.filtering, spec_b.filtering)
            self.assertEqual(spec_a.centroid, spec_b.centroid)
            self.assertNotEqual(spec_a.multilateration, spec_b.multilateration)
            self.assertEqual(spec_a.color, spec_b.color)
            self.assertNotEqual(spec_a.linestyle, spec_b.linestyle)

    def test_weighted_threshold_pairs_share_color_only(self):
        for id_a, id_b in [("B1", "B5"), ("B3", "B6")]:
            spec_a = SPECS_BY_ID[id_a]
            spec_b = SPECS_BY_ID[id_b]

            self.assertEqual(spec_a.distance, spec_b.distance)
            self.assertEqual(spec_a.filtering, spec_b.filtering)
            self.assertEqual(spec_a.multilateration, spec_b.multilateration)
            self.assertEqual(spec_a.centroid, spec_b.centroid)
            self.assertEqual(spec_a.color, spec_b.color)
            self.assertNotEqual(spec_a.linestyle, spec_b.linestyle)
            self.assertEqual(
                spec_a.multilateration_kwargs,
                {"weight_threshold": 0.9},
            )
            self.assertEqual(
                spec_b.multilateration_kwargs,
                {"weight_threshold": 0.5},
            )

    def test_distinct_centroid_paths_have_distinct_colors(self):
        self.assertNotEqual(SPECS_BY_ID["B1"].color, SPECS_BY_ID["B3"].color)
        self.assertNotEqual(SPECS_BY_ID["B2"].color, SPECS_BY_ID["B4"].color)
        self.assertNotEqual(SPECS_BY_ID["B5"].color, SPECS_BY_ID["B6"].color)

    def test_weighted_annulus_thresholds_reach_pipeline(self):
        for combo_id, expected_threshold in [
            ("B1", 0.9),
            ("B3", 0.9),
            ("B5", 0.5),
            ("B6", 0.5),
        ]:
            pipe = build_pipeline(
                SPECS_BY_ID[combo_id],
                lp_models={},
                octant_models={},
                octant_delta=None,
            )

            self.assertEqual(
                pipe.multilateration.weight_threshold,
                expected_threshold,
            )


if __name__ == "__main__":
    unittest.main()
