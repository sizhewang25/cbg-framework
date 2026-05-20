"""Contract test for CBGModel.geolocate's `instrument` hook.

The hook lets benchmarks wrap each stage with timing/memory profilers without
the framework depending on either. The stage-name contract (`"ltd"`, `"mtl"`,
`"ctr"`) is load-bearing for downstream tooling.

Three properties must hold:
  1. On a SUCCESS path, the hook fires exactly ("ltd", "mtl", "ctr") in order.
  2. When MTL fails (empty intersection), CTR never runs — sequence is
     ("ltd", "mtl") only.
  3. With instrument=None (default), no callback is invoked and the result
     matches the un-instrumented call (backward compatibility).
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager

from scripts.framework.v2 import (
    CBGModel,
    GeometricCentroidCTR,
    GeoStatus,
    PlanarCircleMTL,
    SpeedOfInternetLTD,
)
from scripts.framework.v2.tests.helpers import speed_of_internet_pipeline
from scripts.framework.v2.types import Coord, Latency, VpId


class _RecordingInstrument:
    """Context-manager hook that records the order of stages entered/exited."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []  # (stage, "enter"|"exit")

    @contextmanager
    def __call__(self, stage: str):
        self.events.append((stage, "enter"))
        try:
            yield
        finally:
            self.events.append((stage, "exit"))

    @property
    def stages(self) -> list[str]:
        return [s for s, kind in self.events if kind == "enter"]


class TestInstrumentHook(unittest.TestCase):
    def test_success_path_fires_all_three_stages_in_order(self) -> None:
        ltd, obs = speed_of_internet_pipeline()
        model = CBGModel(ltd, PlanarCircleMTL(), GeometricCentroidCTR())
        instr = _RecordingInstrument()

        result = model.geolocate(obs, instrument=instr)

        self.assertEqual(result.status, GeoStatus.SUCCESS)
        self.assertEqual(instr.stages, ["ltd", "mtl", "ctr"])
        # Strict balance: every enter must be paired with an exit, in stack order.
        self.assertEqual(
            instr.events,
            [
                ("ltd", "enter"), ("ltd", "exit"),
                ("mtl", "enter"), ("mtl", "exit"),
                ("ctr", "enter"), ("ctr", "exit"),
            ],
        )

    def test_mtl_failure_skips_ctr_stage(self) -> None:
        """Disjoint disks → MTL returns NO_INTERSECTION; CTR must not run."""
        # SpeedOfInternetLTD with tiny RTT → 50 km disks per VP at TARGET ±500 km
        # gives no intersection between any pair.
        ltd = SpeedOfInternetLTD()
        too_small_obs = [
            (VpId("vp-n"), Coord(lat=4.5, lon=0.0), Latency(0.5)),
            (VpId("vp-s"), Coord(lat=-4.5, lon=0.0), Latency(0.5)),
            (VpId("vp-e"), Coord(lat=0.0, lon=4.5), Latency(0.5)),
            (VpId("vp-w"), Coord(lat=0.0, lon=-4.5), Latency(0.5)),
        ]
        # Disable fallback so a FALLBACK status doesn't mask the contract we care
        # about — we only want to verify which stages the hook sees.
        model = CBGModel(
            ltd, PlanarCircleMTL(), GeometricCentroidCTR(), enable_fallback=False,
        )
        instr = _RecordingInstrument()

        result = model.geolocate(too_small_obs, instrument=instr)

        self.assertEqual(result.status, GeoStatus.ERROR)
        self.assertEqual(instr.stages, ["ltd", "mtl"])  # no "ctr"

    def test_default_no_instrument_is_silent_and_unchanged(self) -> None:
        """instrument=None must be byte-identical to the un-kwarg call."""
        ltd, obs = speed_of_internet_pipeline()
        model = CBGModel(ltd, PlanarCircleMTL(), GeometricCentroidCTR())

        baseline = model.geolocate(obs)
        with_none = model.geolocate(obs, instrument=None)

        self.assertEqual(baseline.status, with_none.status)
        self.assertEqual(baseline.coord, with_none.coord)
        self.assertEqual(
            [r.method for r in baseline.ltd_results],
            [r.method for r in with_none.ltd_results],
        )


if __name__ == "__main__":
    unittest.main()
