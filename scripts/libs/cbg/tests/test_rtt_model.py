"""Unit tests for the simplified LP best-line model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.libs.cbg.rtt_model import (
    RTTDistanceModel,
    THEORETICAL_SLOPE,
    haversine_distance,
)


def _make_linear_samples(slope: float = 0.02, intercept: float = 5.0):
    distances = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=float)
    rtts = slope * distances + intercept
    return distances, rtts


class TestFilterBaseline(unittest.TestCase):
    def test_keeps_rows_on_or_above_speed_of_light_line(self):
        distances, rtts = _make_linear_samples(slope=0.02, intercept=5.0)
        d_out, r_out = RTTDistanceModel.filter_baseline(distances, rtts)
        np.testing.assert_array_equal(d_out, distances)
        np.testing.assert_array_equal(r_out, rtts)

    def test_drops_subluminal_rows(self):
        # 1000 km at 1 ms is well below the 2/3·c floor (~10 ms).
        distances = np.array([100.0, 200.0, 1000.0, 400.0])
        rtts = np.array([10.0, 14.0, 1.0, 22.0])
        d_out, r_out = RTTDistanceModel.filter_baseline(distances, rtts)
        self.assertEqual(len(d_out), 3)
        self.assertNotIn(1000.0, d_out.tolist())
        self.assertNotIn(1.0, r_out.tolist())

    def test_respects_custom_baseline_slope(self):
        distances = np.array([100.0, 200.0, 300.0])
        rtts = np.array([2.0, 4.0, 6.0])
        # baseline_slope=0.01 → keep all (rtt ≥ 0.01·d).
        d_in, _ = RTTDistanceModel.filter_baseline(distances, rtts, baseline_slope=0.01)
        self.assertEqual(len(d_in), 3)
        # baseline_slope=0.03 → drop all (rtt < 0.03·d).
        d_out, _ = RTTDistanceModel.filter_baseline(distances, rtts, baseline_slope=0.03)
        self.assertEqual(len(d_out), 0)


class TestFitBestlineLP(unittest.TestCase):
    """The static LP solver — assumes pre-filtered inputs, does no filtering."""

    def test_recovers_linear_envelope(self):
        distances, rtts = _make_linear_samples(slope=0.02, intercept=5.0)
        result = RTTDistanceModel.fit_bestline_lp(distances, rtts)
        self.assertTrue(result["success"], msg=result["message"])
        self.assertAlmostEqual(result["slope"], 0.02, places=6)
        self.assertAlmostEqual(result["intercept"], 5.0, places=4)
        self.assertEqual(result["violations"], 0)
        self.assertEqual(result["n_points"], 8)

    def test_returns_failure_when_too_few_points(self):
        result = RTTDistanceModel.fit_bestline_lp(
            np.array([100.0, 200.0]), np.array([10.0, 14.0])
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["n_points"], 2)
        self.assertIsNone(result["slope"])
        self.assertIsNone(result["intercept"])

    def test_respects_baseline_slope_floor(self):
        # Steep data; forcing baseline_slope=0.05 should pin slope to at least that.
        distances = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        rtts = np.array([10.0, 14.0, 18.0, 22.0, 26.0])
        result = RTTDistanceModel.fit_bestline_lp(
            distances, rtts, baseline_slope=0.05
        )
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["slope"], 0.05 - 1e-9)


class TestRTTDistanceModelFit(unittest.TestCase):
    """Public `fit()` orchestrates filter → LP → field updates."""

    def _fitted(self) -> RTTDistanceModel:
        m = RTTDistanceModel(anchor_ip="anchor-a", anchor_lat=40.0, anchor_lon=-74.0)
        d, r = _make_linear_samples(slope=0.02, intercept=5.0)
        self.assertTrue(m.fit(d, r), msg=m.fit_message)
        return m

    def test_fit_marks_fitted_and_stores_params(self):
        m = self._fitted()
        self.assertTrue(m.fitted)
        self.assertAlmostEqual(m.slope, 0.02, places=6)
        self.assertAlmostEqual(m.intercept, 5.0, places=4)
        self.assertEqual(m.n_measurements, 8)

    def test_fit_drops_invalid_rows_before_lp(self):
        # 3 invalid rows + 5 valid → LP should still succeed on the survivors.
        distances = np.array([100.0, 200.0, 300.0, 0.0, -50.0, np.nan, 400.0, 500.0])
        rtts = np.array([10.0, 14.0, 18.0, 5.0, 8.0, 12.0, 22.0, 26.0])
        m = RTTDistanceModel(anchor_ip="x", anchor_lat=0.0, anchor_lon=0.0)
        ok = m.fit(distances, rtts, enable_baseline_filter=False)
        self.assertTrue(ok, msg=m.fit_message)
        self.assertEqual(m.n_measurements, 8)  # raw input count is preserved

    def test_baseline_filter_rescues_fit_from_subluminal_outlier(self):
        # Any sub-baseline row violates the LP's `m >= baseline_slope` hard
        # constraint and makes the problem infeasible. The default baseline
        # filter is what keeps fit() viable in the presence of such rows.
        distances, rtts = _make_linear_samples(slope=0.02, intercept=5.0)
        distances = np.append(distances, 1000.0)
        rtts = np.append(rtts, 1.0)

        with_filter = RTTDistanceModel(anchor_ip="a", anchor_lat=0.0, anchor_lon=0.0)
        self.assertTrue(with_filter.fit(distances, rtts, enable_baseline_filter=True))
        self.assertAlmostEqual(with_filter.intercept, 5.0, places=4)

        no_filter = RTTDistanceModel(anchor_ip="b", anchor_lat=0.0, anchor_lon=0.0)
        self.assertFalse(no_filter.fit(distances, rtts, enable_baseline_filter=False))

    def test_fit_returns_false_on_mismatched_lengths(self):
        m = RTTDistanceModel(anchor_ip="x", anchor_lat=0.0, anchor_lon=0.0)
        ok = m.fit(np.array([100.0, 200.0]), np.array([10.0]))
        self.assertFalse(ok)
        self.assertFalse(m.fitted)
        self.assertIn("same length", m.fit_message)


class TestPredictDistance(unittest.TestCase):
    def _fitted(self) -> RTTDistanceModel:
        m = RTTDistanceModel(anchor_ip="anchor-a", anchor_lat=40.0, anchor_lon=-74.0)
        d, r = _make_linear_samples(slope=0.02, intercept=5.0)
        self.assertTrue(m.fit(d, r), msg=m.fit_message)
        return m

    def test_inverts_envelope(self):
        m = self._fitted()
        self.assertAlmostEqual(m.predict_distance(25.0), 1000.0, places=3)

    def test_returns_none_when_unfitted(self):
        m = RTTDistanceModel(anchor_ip="x", anchor_lat=0.0, anchor_lon=0.0)
        self.assertIsNone(m.predict_distance(20.0))

    def test_clamps_at_zero(self):
        # rtt below intercept → negative raw distance → clamped to 0.
        m = self._fitted()
        self.assertEqual(m.predict_distance(1.0), 0.0)


class TestSerializationJSON(unittest.TestCase):
    def _fitted(self) -> RTTDistanceModel:
        m = RTTDistanceModel(anchor_ip="anchor-a", anchor_lat=40.0, anchor_lon=-74.0)
        d, r = _make_linear_samples(slope=0.02, intercept=5.0)
        self.assertTrue(m.fit(d, r), msg=m.fit_message)
        return m

    def test_save_writes_human_readable_json(self):
        m = self._fitted()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "model.json"
            m.save(path)
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text())
        self.assertEqual(payload["anchor_ip"], "anchor-a")
        self.assertAlmostEqual(payload["slope"], 0.02, places=6)
        self.assertAlmostEqual(payload["intercept"], 5.0, places=4)
        self.assertTrue(payload["fitted"])

    def test_save_load_roundtrip(self):
        m = self._fitted()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            m.save(path)
            loaded = RTTDistanceModel.load(path)
        self.assertTrue(loaded.fitted)
        self.assertAlmostEqual(loaded.slope, m.slope)
        self.assertAlmostEqual(loaded.intercept, m.intercept)
        self.assertAlmostEqual(loaded.predict_distance(25.0), 1000.0, places=3)

    def test_unfitted_model_roundtrips_with_none_params(self):
        m = RTTDistanceModel(anchor_ip="x", anchor_lat=0.0, anchor_lon=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.json"
            m.save(path)
            loaded = RTTDistanceModel.load(path)
        self.assertFalse(loaded.fitted)
        self.assertIsNone(loaded.slope)
        self.assertIsNone(loaded.intercept)

    def test_to_dict_carries_only_kept_fields(self):
        m = self._fitted()
        self.assertEqual(
            set(m.to_dict()),
            {
                "anchor_ip",
                "anchor_lat",
                "anchor_lon",
                "slope",
                "intercept",
                "n_measurements",
                "fitted",
                "fit_message",
            },
        )

    def test_legacy_fields_are_absent(self):
        """Confirm the simplification actually removed the inert fields."""
        m = RTTDistanceModel(anchor_ip="x", anchor_lat=0.0, anchor_lon=0.0)
        for field in (
            "r_squared",
            "n_bins",
            "bin_size_km",
            "percentile",
            "bin_centers",
            "bin_rtts",
            "fit_method",
        ):
            self.assertFalse(hasattr(m, field), msg=field)
        self.assertFalse(hasattr(m, "predict_rtt"))


class TestHaversineAndConstants(unittest.TestCase):
    def test_haversine_zero_distance(self):
        self.assertAlmostEqual(haversine_distance(40.0, -74.0, 40.0, -74.0), 0.0)

    def test_haversine_one_degree_latitude(self):
        d = haversine_distance(40.0, -74.0, 41.0, -74.0)
        self.assertAlmostEqual(d, 111.195, places=2)

    def test_theoretical_slope_at_two_thirds_c(self):
        # rtt = 2·d / (2/3·c) = d · (2 / 200 km/ms) = 0.01 ms/km
        self.assertAlmostEqual(THEORETICAL_SLOPE, 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
