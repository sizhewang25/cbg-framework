"""Cross-product integration tests: every permitted LTD × MTL × CTR composition.

The framework permits three groups of pairings (CTR is orthogonal):
  1. CircleLTDModel  + CircleMTLMethod   — native Circle
  2. AnnulusLTDModel + AnnulusMTLMethod  — native Annulus
  3. AnnulusLTDModel + CircleMTLMethod   — degraded; inner bound discarded

That's 2×2×4 = 16 native Circle, 2×2×4 = 16 native Annulus, and 2×2×4 = 16
degraded combinations — 48 total. CircleLTD + AnnulusMTL is the only
cross-family pair the framework still rejects (covered in
test_family_validation.py).

Each combination is exercised against a single shared scenario (see
`helpers.py`): four ~500 km VPs around TARGET (0, 0). For each combo we
assert:
  - status == SUCCESS (the pipeline completed without falling back),
  - coord is finite and within 350 km of TARGET (loose; vertex-based CTRs
    on a SphericalCircleMTL output snap to a region corner),
  - every per-stage result is populated and stamped with its method name.
"""

from __future__ import annotations

import math
import unittest
import warnings
from dataclasses import dataclass
from typing import Callable, Type

from scripts.framework.v2 import (
    BoundaryVertexMeanCTR,
    BoundedSplineLTD,
    CBGModel,
    CTRMethod,
    GeometricCentroidCTR,
    GeometricMedianCTR,
    GeoStatus,
    LowEnvelopeLTD,
    LTDModel,
    MonteCarloMedoidCTR,
    MTLMethod,
    NormalDistLTD,
    PlanarAnnulusMTL,
    PlanarAnnulusWeightedMTL,
    PlanarCircleMTL,
    SpeedOfInternetLTD,
    SphericalCircleMTL,
)
from scripts.framework.v2.tests.helpers import (
    TARGET,
    bounded_spline_pipeline,
    low_envelope_pipeline,
    normal_dist_pipeline,
    speed_of_internet_pipeline,
)
from scripts.libs.cbg.rtt_model import haversine_distance


@dataclass(frozen=True)
class _LTDSpec:
    name: str
    factory: Callable[[], tuple]
    ltd_cls: Type[LTDModel]


_CIRCLE_LTDS = [
    _LTDSpec("speed_of_internet", speed_of_internet_pipeline, SpeedOfInternetLTD),
    _LTDSpec("low_envelope", low_envelope_pipeline, LowEnvelopeLTD),
]

_ANNULUS_LTDS = [
    _LTDSpec("normal_dist", normal_dist_pipeline, NormalDistLTD),
    _LTDSpec("bounded_spline", bounded_spline_pipeline, BoundedSplineLTD),
]

_CIRCLE_MTLS: list[tuple[str, Type[MTLMethod]]] = [
    ("planar_circle", PlanarCircleMTL),
    ("spherical_circle", SphericalCircleMTL),
]

_ANNULUS_MTLS: list[tuple[str, Type[MTLMethod]]] = [
    ("planar_annulus", PlanarAnnulusMTL),
    ("planar_annulus_weighted", PlanarAnnulusWeightedMTL),
]

_CTRS: list[tuple[str, Type[CTRMethod]]] = [
    ("geometric_centroid", GeometricCentroidCTR),
    ("geometric_median", GeometricMedianCTR),
    ("monte_carlo_medoid", MonteCarloMedoidCTR),
    ("boundary_vertex_mean", BoundaryVertexMeanCTR),
]

# Generous bound — vertex-based CTRs (geometric_median / monte_carlo_medoid)
# on top of SphericalCircleMTL snap to one of the 4 region corners, which
# sit ~250–260 km off TARGET in our jittered setup. 350 km still proves the
# coord landed inside (or on the boundary of) the feasible region without
# locking us into corner-specific numerics.
_MAX_ERROR_KM = 350.0


class TestPipelineCombinations(unittest.TestCase):
    """End-to-end SUCCESS check for every valid (LTD, MTL, CTR) triple."""

    def setUp(self) -> None:
        # MonteCarloMedoidCTR's Sobol sampler complains for non-power-of-2
        # sample counts. We don't tune n_samples here — the default is fine
        # for asserting SUCCESS — so silence the noise for these tests.
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol' points require n to be a power of 2.",
        )

    def _assert_success(self, combo: str, ltd_spec, mtl_cls, ctr_cls) -> None:
        ltd, obs = ltd_spec.factory()
        model = CBGModel(ltd, mtl_cls(), ctr_cls())

        result = model.geolocate(obs)

        self.assertEqual(
            result.status, GeoStatus.SUCCESS,
            f"{combo}: expected SUCCESS, got {result.status} (error={result.error})",
        )
        self.assertIsNone(result.error, f"{combo}: SUCCESS path must have error=None")
        self.assertIsNotNone(result.coord, f"{combo}: SUCCESS path must return a coord")
        self.assertTrue(math.isfinite(result.coord.lat))
        self.assertTrue(math.isfinite(result.coord.lon))

        dist = haversine_distance(
            TARGET.lat, TARGET.lon, result.coord.lat, result.coord.lon,
        )
        self.assertLess(
            dist, _MAX_ERROR_KM,
            f"{combo}: coord {result.coord} is {dist:.1f} km from TARGET (> {_MAX_ERROR_KM})",
        )

        # Every stage's result must be populated and method-stamped.
        self.assertEqual(len(result.ltd_results), len(obs))
        for r in result.ltd_results:
            self.assertTrue(r.success, f"{combo}: per-VP LTD failure for {r.vp_id}")
            self.assertEqual(r.method, ltd_spec.ltd_cls.__name__)
            self.assertEqual(r.latency, obs[0][2])  # all four obs share one RTT

        self.assertIsNotNone(result.mtl_result)
        self.assertTrue(result.mtl_result.success)
        self.assertEqual(result.mtl_result.method, mtl_cls.__name__)

        self.assertIsNotNone(result.ctr_result)
        self.assertTrue(result.ctr_result.success)
        self.assertEqual(result.ctr_result.method, ctr_cls.__name__)
        self.assertEqual(result.ctr_result.tg_coord, result.coord)

    def test_every_circle_combo_succeeds(self) -> None:
        for ltd_spec in _CIRCLE_LTDS:
            for mtl_name, mtl_cls in _CIRCLE_MTLS:
                for ctr_name, ctr_cls in _CTRS:
                    combo = f"{ltd_spec.name} | {mtl_name} | {ctr_name}"
                    with self.subTest(combo=combo):
                        self._assert_success(combo, ltd_spec, mtl_cls, ctr_cls)

    def test_every_annulus_combo_succeeds(self) -> None:
        for ltd_spec in _ANNULUS_LTDS:
            for mtl_name, mtl_cls in _ANNULUS_MTLS:
                for ctr_name, ctr_cls in _CTRS:
                    combo = f"{ltd_spec.name} | {mtl_name} | {ctr_name}"
                    with self.subTest(combo=combo):
                        self._assert_success(combo, ltd_spec, mtl_cls, ctr_cls)

    def test_annulus_ltd_through_circle_mtl_degrades_cleanly(self) -> None:
        """Permitted cross-family: AnnulusLTDModel + CircleMTLMethod.

        Circle MTLs read only `tg_distance.upper_km`. An annular constraint
        [400, 600] km therefore reaches the MTL as a disk of radius 600 km —
        the inner-bound information is lost. Each VP sits 480–520 km from
        TARGET, so a 600 km disk per VP still contains TARGET and the
        pipeline runs end-to-end.
        """
        for ltd_spec in _ANNULUS_LTDS:
            for mtl_name, mtl_cls in _CIRCLE_MTLS:
                for ctr_name, ctr_cls in _CTRS:
                    combo = f"{ltd_spec.name} | {mtl_name} | {ctr_name} (degraded)"
                    with self.subTest(combo=combo):
                        self._assert_success(combo, ltd_spec, mtl_cls, ctr_cls)


if __name__ == "__main__":
    unittest.main()
