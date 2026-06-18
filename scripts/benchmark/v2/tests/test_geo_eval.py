"""Geographic (continent/country) postprocessing over a targets.parquet.

`annotate_targets_geo` reverse-geocodes each target's ground-truth coordinates
into a continent + ISO country code without touching runner state.
`summarize_geo` reduces the annotated frame to per-(level, value) eval rows
(overall, per-continent, per-country) carrying success counts and error_km
stats over SUCCESS/FALLBACK. `process_parquet` round-trips a file in place and
is idempotent.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark.v2.geo_eval import (
    GEO_COLUMNS,
    annotate_targets_geo,
    process_parquet,
    summarize_geo,
)

# (lat, lon) for unambiguous, well-separated metros so the kdtree lookup is
# stable regardless of GeoNames revision.
_NYC = (40.7128, -74.0060)    # US  / North America
_PARIS = (48.8566, 2.3522)    # FR  / Europe
_TOKYO = (35.6762, 139.6503)  # JP  / Asia


def _targets() -> pd.DataFrame:
    # Two US targets (one SUCCESS, one ERROR), one FR SUCCESS, one JP FALLBACK.
    return pd.DataFrame(
        {
            "target_id": ["A", "B", "C", "D"],
            "target_lat": [_NYC[0], _NYC[0], _PARIS[0], _TOKYO[0]],
            "target_lon": [_NYC[1], _NYC[1], _PARIS[1], _TOKYO[1]],
            "pred_lat": [40.7, None, 48.9, 35.7],
            "pred_lon": [-74.0, None, 2.4, 139.7],
            "status": ["SUCCESS", "ERROR", "SUCCESS", "FALLBACK"],
            "error_km": [5.0, None, 8.0, 12.0],
        }
    )


class TestAnnotateTargetsGeo(unittest.TestCase):
    def test_adds_all_geo_columns(self) -> None:
        out = annotate_targets_geo(_targets())
        for col in GEO_COLUMNS:
            self.assertIn(col, out.columns)

    def test_country_and_continent_from_coordinates(self) -> None:
        out = annotate_targets_geo(_targets()).set_index("target_id")
        self.assertEqual(out.loc["A", "target_country"], "US")
        self.assertEqual(out.loc["A", "target_continent"], "North America")
        self.assertEqual(out.loc["C", "target_country"], "FR")
        self.assertEqual(out.loc["C", "target_continent"], "Europe")
        self.assertEqual(out.loc["D", "target_country"], "JP")
        self.assertEqual(out.loc["D", "target_continent"], "Asia")

    def test_populated_regardless_of_prediction_status(self) -> None:
        # B is an ERROR row (no prediction) — the geo label still resolves
        # because it is derived from the ground-truth target coords.
        out = annotate_targets_geo(_targets()).set_index("target_id")
        self.assertEqual(out.loc["B", "target_country"], "US")
        self.assertEqual(out.loc["B", "target_continent"], "North America")

    def test_idempotent(self) -> None:
        once = annotate_targets_geo(_targets())
        twice = annotate_targets_geo(once)
        pd.testing.assert_frame_equal(once, twice)


class TestSummarizeGeo(unittest.TestCase):
    def test_overall_row_present(self) -> None:
        rows = summarize_geo(annotate_targets_geo(_targets()))
        overall = [r for r in rows if r["group_level"] == "all"]
        self.assertEqual(len(overall), 1)
        self.assertEqual(overall[0]["n_targets"], 4)
        self.assertEqual(overall[0]["n_success"], 2)
        self.assertEqual(overall[0]["n_fallback"], 1)
        self.assertEqual(overall[0]["n_error"], 1)

    def test_continent_breakdown_counts(self) -> None:
        rows = summarize_geo(annotate_targets_geo(_targets()))
        by_cont = {r["group_value"]: r for r in rows if r["group_level"] == "continent"}
        self.assertEqual(by_cont["North America"]["n_targets"], 2)
        self.assertEqual(by_cont["North America"]["n_success"], 1)
        self.assertEqual(by_cont["North America"]["n_error"], 1)
        self.assertEqual(by_cont["Europe"]["n_targets"], 1)
        self.assertEqual(by_cont["Asia"]["n_targets"], 1)

    def test_country_breakdown_present(self) -> None:
        rows = summarize_geo(annotate_targets_geo(_targets()))
        ccs = {r["group_value"] for r in rows if r["group_level"] == "country"}
        self.assertEqual(ccs, {"US", "FR", "JP"})

    def test_error_stats_over_scored_rows(self) -> None:
        rows = summarize_geo(annotate_targets_geo(_targets()))
        # North America scored rows: only A (B is ERROR) → p50 == 5.0.
        na = next(
            r for r in rows
            if r["group_level"] == "continent" and r["group_value"] == "North America"
        )
        self.assertAlmostEqual(na["error_km_p50"], 5.0)

    def test_empty_frame_is_safe(self) -> None:
        empty = annotate_targets_geo(_targets()).iloc[0:0]
        rows = summarize_geo(empty)
        overall = [r for r in rows if r["group_level"] == "all"]
        self.assertEqual(overall[0]["n_targets"], 0)


class TestProcessParquet(unittest.TestCase):
    def test_roundtrip_adds_columns_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "targets.parquet"
            _targets().to_parquet(path)
            process_parquet(path)
            back = pd.read_parquet(path)
            for col in GEO_COLUMNS:
                self.assertIn(col, back.columns)

    def test_returns_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "targets.parquet"
            _targets().to_parquet(path)
            rows = process_parquet(path)
            self.assertTrue(any(r["group_level"] == "all" for r in rows))
            self.assertTrue(any(r["group_level"] == "continent" for r in rows))


if __name__ == "__main__":
    unittest.main()
