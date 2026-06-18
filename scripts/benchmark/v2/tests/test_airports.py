"""AirportIndex + OurAirports filtering.

Covers the airport reference layer for the closest-airport eval metric:
- `filter_airports` keeps only large/medium airports that carry both an IATA
  code and a municipality (the operator-facing metro granularity).
- `AirportIndex` returns the nearest IATA code and its great-circle distance
  (km) via a haversine BallTree, vectorized over many query points, and is
  NaN-safe for missing coordinates.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from scripts.benchmark.v2.airports import AirportIndex, filter_airports

# A few real airports (lat, lon) used as a tiny synthetic reference set.
_JFK = (40.6413, -73.7781)
_LAX = (33.9416, -118.4085)
_LHR = (51.4700, -0.4543)


def _index() -> AirportIndex:
    df = pd.DataFrame(
        {
            "iata_code": ["JFK", "LAX", "LHR"],
            "latitude_deg": [_JFK[0], _LAX[0], _LHR[0]],
            "longitude_deg": [_JFK[1], _LAX[1], _LHR[1]],
            "municipality": ["New York", "Los Angeles", "London"],
        }
    )
    return AirportIndex(df)


class TestFilterAirports(unittest.TestCase):
    def _raw(self) -> pd.DataFrame:
        # Mimics the OurAirports CSV columns we depend on.
        return pd.DataFrame(
            {
                "type": [
                    "large_airport",   # keep
                    "medium_airport",  # keep
                    "small_airport",   # drop: wrong type
                    "heliport",        # drop: wrong type
                    "large_airport",   # drop: no IATA
                    "medium_airport",  # drop: no municipality
                    "medium_airport",  # drop: no scheduled service (e.g. PAO/NUQ)
                ],
                "iata_code": ["JFK", "AUS", "XXX", "HHH", None, "ZZZ", "PAO"],
                "municipality": ["New York", "Austin", "Nowhere", "Pad", "Bigcity", None, "Palo Alto"],
                "scheduled_service": ["yes", "yes", "yes", "yes", "yes", "yes", "no"],
                "latitude_deg": [40.6, 30.2, 1.0, 2.0, 3.0, 4.0, 37.46],
                "longitude_deg": [-73.8, -97.7, 1.0, 2.0, 3.0, 4.0, -122.11],
                "name": ["a", "b", "c", "d", "e", "f", "Palo Alto Airport"],
                "iso_country": ["US", "US", "US", "US", "US", "US", "US"],
            }
        )

    def test_keeps_only_large_medium_with_iata_municipality_and_scheduled(self) -> None:
        out = filter_airports(self._raw())
        self.assertEqual(sorted(out["iata_code"]), ["AUS", "JFK"])

    def test_unscheduled_airport_dropped(self) -> None:
        # PAO (Palo Alto) is medium + IATA + municipality but has no scheduled
        # service — exactly the GA-field artifact the filter must exclude.
        out = filter_airports(self._raw())
        self.assertNotIn("PAO", set(out["iata_code"]))

    def test_blank_strings_treated_as_missing(self) -> None:
        raw = self._raw()
        raw.loc[0, "iata_code"] = "   "  # JFK now blank → dropped
        raw.loc[1, "municipality"] = ""   # AUS now blank → dropped
        out = filter_airports(raw)
        self.assertEqual(len(out), 0)


class TestAirportIndexQuery(unittest.TestCase):
    def test_exact_airport_location_returns_itself_at_zero_km(self) -> None:
        idx = _index()
        iata, km = idx.query_many([_JFK[0]], [_JFK[1]])
        self.assertEqual(iata[0], "JFK")
        self.assertAlmostEqual(km[0], 0.0, places=3)

    def test_nearby_point_resolves_to_closest_airport(self) -> None:
        idx = _index()
        # A point in Manhattan is nearest to JFK among {JFK, LAX, LHR}.
        iata, _ = idx.query_many([40.7580], [-73.9855])
        self.assertEqual(iata[0], "JFK")

    def test_distance_matches_independent_haversine(self) -> None:
        idx = _index()
        # Manhattan point; compare returned km to a hand-rolled haversine to JFK.
        qlat, qlon = 40.7580, -73.9855
        _, km = idx.query_many([qlat], [qlon])
        expected = _haversine(qlat, qlon, *_JFK)
        self.assertAlmostEqual(km[0], expected, places=2)

    def test_vectorized_query_returns_per_point_results(self) -> None:
        idx = _index()
        iata, km = idx.query_many(
            [_JFK[0], _LAX[0], _LHR[0]],
            [_JFK[1], _LAX[1], _LHR[1]],
        )
        self.assertEqual(list(iata), ["JFK", "LAX", "LHR"])
        self.assertTrue(np.allclose(km, 0.0, atol=1e-3))

    def test_nan_coordinates_yield_null_iata_and_nan_km(self) -> None:
        idx = _index()
        iata, km = idx.query_many([np.nan, _LAX[0]], [np.nan, _LAX[1]])
        self.assertIsNone(iata[0])
        self.assertTrue(math.isnan(km[0]))
        # The valid second point is unaffected by the NaN sibling.
        self.assertEqual(iata[1], "LAX")
        self.assertAlmostEqual(km[1], 0.0, places=3)

    def test_query_full_returns_matched_airport_coordinates(self) -> None:
        idx = _index()
        iata, km, ap_lat, ap_lon = idx.query_full([40.7580, np.nan], [-73.9855, np.nan])
        # Matched airport is JFK; returned coords are JFK's, not the query's.
        self.assertEqual(iata[0], "JFK")
        self.assertAlmostEqual(ap_lat[0], _JFK[0], places=4)
        self.assertAlmostEqual(ap_lon[0], _JFK[1], places=4)
        # NaN query → null iata and NaN coords.
        self.assertIsNone(iata[1])
        self.assertTrue(math.isnan(ap_lat[1]))
        self.assertTrue(math.isnan(ap_lon[1]))


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


if __name__ == "__main__":
    unittest.main()
