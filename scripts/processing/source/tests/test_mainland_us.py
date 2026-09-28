"""Invariants of the mainland-US row filter.

Three things carry the filter's meaning and are easy to regress:

  * **both** endpoints must be inside the window -- a mainland VP measuring a
    Honolulu target is not a mainland row, and neither is the reverse;
  * a country code, when the CSV carries one, overrides the box -- southern
    Ontario and northern Mexico sit inside the extent but are not the US; and
  * headers are matched case-insensitively, so the upper-case raw schema and
    the lower-case canonical one behave identically.
"""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.processing.source.mainland_us import (
    US_MAINLAND_EXTENT,
    filter_non_mainland_us_targets_and_vps,
)

SF = (37.77, -122.42)
NYC = (40.71, -74.01)
ANCHORAGE = (61.22, -149.90)
HONOLULU = (21.31, -157.86)
TORONTO = (43.65, -79.38)  # inside the box, outside the US


def _row(vp, target, vp_country="US", target_country="US"):
    return {
        "vp_lat": vp[0], "vp_lon": vp[1], "vp_country": vp_country,
        "target_lat": target[0], "target_lon": target[1],
        "target_country": target_country,
    }


class TestMainlandFilter(unittest.TestCase):
    def test_keeps_only_rows_with_both_endpoints_inside(self):
        df = pd.DataFrame([
            _row(SF, NYC),
            _row(ANCHORAGE, NYC),
            _row(SF, HONOLULU),
            _row(ANCHORAGE, HONOLULU),
        ])
        kept = filter_non_mainland_us_targets_and_vps(df)
        self.assertEqual(kept.index.tolist(), [0])

    def test_country_code_overrides_the_box(self):
        df = pd.DataFrame([_row(SF, NYC), _row(SF, TORONTO, target_country="CA")])
        kept = filter_non_mainland_us_targets_and_vps(df)
        self.assertEqual(kept.index.tolist(), [0])

    def test_missing_country_columns_fall_back_to_coordinates(self):
        df = pd.DataFrame([_row(SF, NYC), _row(SF, HONOLULU)]).drop(
            columns=["vp_country", "target_country"]
        )
        kept = filter_non_mainland_us_targets_and_vps(df)
        self.assertEqual(kept.index.tolist(), [0])

    def test_blank_country_is_not_treated_as_non_us(self):
        df = pd.DataFrame([_row(SF, NYC, vp_country="", target_country=None)])
        kept = filter_non_mainland_us_targets_and_vps(df)
        self.assertEqual(len(kept), 1)

    def test_header_case_does_not_matter(self):
        df = pd.DataFrame([_row(SF, NYC), _row(SF, HONOLULU)])
        upper = df.rename(columns=str.upper)
        self.assertEqual(
            filter_non_mainland_us_targets_and_vps(upper).index.tolist(),
            filter_non_mainland_us_targets_and_vps(df).index.tolist(),
        )

    def test_non_numeric_coordinates_are_dropped_not_raised(self):
        df = pd.DataFrame([_row(SF, NYC), _row(SF, NYC)])
        df.loc[1, "vp_lat"] = "not-a-number"
        kept = filter_non_mainland_us_targets_and_vps(df)
        self.assertEqual(kept.index.tolist(), [0])

    def test_missing_coordinate_column_raises(self):
        df = pd.DataFrame([_row(SF, NYC)]).drop(columns=["vp_lat"])
        with self.assertRaises(ValueError):
            filter_non_mainland_us_targets_and_vps(df)

    def test_extent_matches_the_repo_frame(self):
        # The one continental-US window, shared with the analysis/plot modules.
        self.assertEqual(US_MAINLAND_EXTENT, (-125.0, -66.0, 24.0, 50.0))


if __name__ == "__main__":
    unittest.main()
