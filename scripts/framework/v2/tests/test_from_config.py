"""Registry-driven composition: CBGModel.from_config picks classes by name.

Mirrors test_pipeline_combinations on a smaller scale — one Circle and
one Annulus combo built via `from_config` — plus the negative paths
(unknown LTD/MTL/CTR name → KeyError).

The from_config path does NOT accept pre-built stage state, so models
that need fitted submodels (LowEnvelopeLTD, NormalDistLTD, BoundedSpline)
would either need real fit data or a stateless LTD. SpeedOfInternetLTD
is stateless, so we use it for the happy-path Circle case. The Annulus
case uses a fresh NormalDistLTD and re-fits inside the test from real
samples.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2 import (
    CBGModel,
    GeometricCentroidCTR,
    GeoStatus,
    PlanarCircleMTL,
    SpeedOfInternetLTD,
)
from scripts.framework.v2.ltd.base import FitSample
from scripts.framework.v2.tests.helpers import (
    TARGET,
    VPS,
    bounded_spline_pipeline,
    observations,
    speed_of_internet_pipeline,
)
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.libs.cbg.rtt_model import haversine_distance


class TestFromConfig(unittest.TestCase):
    def test_circle_combo_via_from_config_succeeds(self) -> None:
        """Registry lookup yields a working Circle pipeline."""
        model = CBGModel.from_config(
            ltd="speed_of_internet",
            mtl="planar_circle",
            ctr="geometric_centroid",
        )
        _, obs = speed_of_internet_pipeline()  # reuse the 700 km-disk RTT

        result = model.geolocate(obs)

        self.assertEqual(result.status, GeoStatus.SUCCESS)
        self.assertEqual(
            result.ltd_results[0].method, "SpeedOfInternetLTD",
        )
        self.assertEqual(result.mtl_result.method, "PlanarCircleMTL")
        self.assertEqual(result.ctr_result.method, "GeometricCentroidCTR")

    def test_annulus_combo_via_from_config_with_fit(self) -> None:
        """Annulus combo via from_config: build BoundedSplineLTD from real samples,
        copy its private state onto the from_config-built LTD, then geolocate.

        This exercises the registry-side construction and the full pipeline
        without trying to push a fitted submodel through `from_config` kwargs.
        """
        primed_ltd, obs = bounded_spline_pipeline()

        model = CBGModel.from_config(
            ltd="bounded_spline",
            mtl="planar_annulus",
            ctr="geometric_centroid",
        )
        # Copy fitted state into the from_config-built LTD. This is the
        # only practical way to share calibration across both LTDs without
        # re-running OctantRTTModel.fit four times here.
        model.ltd._submodels = primed_ltd._submodels
        model.ltd._deltas = primed_ltd._deltas

        result = model.geolocate(obs)

        self.assertEqual(result.status, GeoStatus.SUCCESS)
        self.assertEqual(result.ltd_results[0].method, "BoundedSplineLTD")
        self.assertEqual(result.mtl_result.method, "PlanarAnnulusMTL")
        self.assertEqual(result.ctr_result.method, "GeometricCentroidCTR")
        dist = haversine_distance(
            TARGET.lat, TARGET.lon, result.coord.lat, result.coord.lon,
        )
        self.assertLess(dist, 250.0)

    def test_unknown_ltd_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError) as exc:
            CBGModel.from_config("not_a_model", "planar_circle", "geometric_centroid")
        self.assertIn("not_a_model", str(exc.exception))

    def test_unknown_mtl_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            CBGModel.from_config("speed_of_internet", "not_a_mtl", "geometric_centroid")

    def test_unknown_ctr_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            CBGModel.from_config("speed_of_internet", "planar_circle", "not_a_ctr")

    def test_kwargs_forwarded_to_stage_constructors(self) -> None:
        """ltd_kwargs / mtl_kwargs / ctr_kwargs reach the underlying classes."""
        model = CBGModel.from_config(
            ltd="speed_of_internet",
            mtl="planar_circle",
            ctr="monte_carlo_medoid",
            ltd_kwargs={"speed_ratio": 0.5},
            mtl_kwargs={"n_pts": 32},
            ctr_kwargs={"n_samples": 512, "seed": 7},
        )
        self.assertEqual(model.ltd.speed_ratio, 0.5)
        self.assertEqual(model.mtl.n_pts, 32)
        self.assertEqual(model.ctr.n_samples, 512)


class TestRealFitIntegration(unittest.TestCase):
    """One end-to-end test that actually exercises `model.fit(samples)`."""

    def test_speed_of_internet_fit_then_geolocate(self) -> None:
        """SpeedOfInternetLTD.fit is a no-op; the pipeline still runs end-to-end."""
        model = CBGModel(
            SpeedOfInternetLTD(), PlanarCircleMTL(), GeometricCentroidCTR(),
        )
        # One filler FitSample — enough to prove fit returns success even
        # though the wrapper is stateless.
        samples = [
            FitSample(
                vp_id=VpId("vp-n"),
                vp_coord=VPS[0][1],
                probe_coord=Coord(0.0, 0.0),
                latency=Latency(5.0),
            )
        ]
        fit_result = model.fit(samples)
        self.assertTrue(fit_result.success)
        self.assertEqual(fit_result.method, "SpeedOfInternetLTD")

        result = model.geolocate(observations(7.0))
        self.assertEqual(result.status, GeoStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
