"""Pipeline failure path: fallback to nearest VP, or surface ERROR.

When MTL or CTR fails, CBGModel's owned fallback returns the coord of
the lowest-latency VP and stamps status=FALLBACK. With enable_fallback=
False, the same failure surfaces as status=ERROR (no coord). With no
observations at all, fallback is disabled implicitly.

The failure trigger here is geometric: SpeedOfInternetLTD at RTT=0.5 ms
yields a 50 km disk per VP, while the four VPs sit ~500 km apart — no
overlap, EMPTY_REGION.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2 import (
    CBGModel,
    Error,
    GeoStatus,
    GeometricCentroidCTR,
    PlanarCircleMTL,
    SpeedOfInternetLTD,
)
from scripts.framework.v2.tests.helpers import VPS, observations
from scripts.framework.v2.types import Coord, Latency, VpId


def _disjoint_obs() -> list[tuple[VpId, Coord, Latency]]:
    """50 km disks centered ~500 km apart — empty intersection."""
    return observations(0.5)


class TestFallbackPath(unittest.TestCase):
    def test_empty_intersection_falls_back_to_nearest_vp(self) -> None:
        """MTL fails (EMPTY_REGION) → coord is the lowest-latency VP's location."""
        model = CBGModel(
            SpeedOfInternetLTD(), PlanarCircleMTL(), GeometricCentroidCTR(),
        )
        # vp-w gets the strictly smallest RTT, so it's the unique nearest VP.
        obs = [
            (VpId("vp-n"), VPS[0][1], Latency(0.5)),
            (VpId("vp-s"), VPS[1][1], Latency(0.5)),
            (VpId("vp-e"), VPS[2][1], Latency(0.5)),
            (VpId("vp-w"), VPS[3][1], Latency(0.4)),
        ]

        result = model.geolocate(obs)

        self.assertEqual(result.status, GeoStatus.FALLBACK)
        self.assertEqual(result.error, Error.EMPTY_REGION)
        self.assertEqual(result.coord, VPS[3][1])
        # LTD ran for every observation; MTL ran and failed; CTR didn't run.
        self.assertEqual(len(result.ltd_results), 4)
        self.assertTrue(all(r.success for r in result.ltd_results))
        self.assertIsNotNone(result.mtl_result)
        self.assertFalse(result.mtl_result.success)
        self.assertIsNone(result.ctr_result)

    def test_disabled_fallback_surfaces_error_without_coord(self) -> None:
        """enable_fallback=False: same failure yields ERROR + None coord."""
        model = CBGModel(
            SpeedOfInternetLTD(), PlanarCircleMTL(), GeometricCentroidCTR(),
            enable_fallback=False,
        )

        result = model.geolocate(_disjoint_obs())

        self.assertEqual(result.status, GeoStatus.ERROR)
        self.assertEqual(result.error, Error.EMPTY_REGION)
        self.assertIsNone(result.coord)
        self.assertIsNotNone(result.mtl_result)
        self.assertFalse(result.mtl_result.success)

    def test_empty_observations_returns_error_even_with_fallback_enabled(self) -> None:
        """Fallback needs at least one observation to point at — none → ERROR."""
        model = CBGModel(
            SpeedOfInternetLTD(), PlanarCircleMTL(), GeometricCentroidCTR(),
            enable_fallback=True,
        )

        result = model.geolocate([])

        self.assertEqual(result.status, GeoStatus.ERROR)
        self.assertIsNone(result.coord)
        self.assertEqual(result.ltd_results, ())
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)


if __name__ == "__main__":
    unittest.main()
