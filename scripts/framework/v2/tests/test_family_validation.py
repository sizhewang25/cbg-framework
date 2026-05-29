"""Composition-time guard: every (LTD family × MTL family) pair is legal.

Historical context: CBGModel used to reject Circle LTD + Annulus MTL with
`IncompatibleStagesError` to prevent silent degradation. That wall was
removed once CircleLTDModel became a subclass of AnnulusLTDModel — a disk
is just an annulus with inner radius 0, so the pairing is semantically
well-defined and downstream code already handles it. These tests pin the
new contract: all four (LTD family × MTL family) combinations construct
and run.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2 import (
    AnnulusLTDModel,
    BoundaryVertexMeanCTR,
    BoundedSplineLTD,
    CBGModel,
    CircleLTDModel,
    GeometricCentroidCTR,
    LowEnvelopeLTD,
    NormalDistLTD,
    PlanarAnnulusMTL,
    PlanarAnnulusWeightedMTL,
    PlanarCircleMTL,
    SpeedOfInternetLTD,
    SphericalCircleMTL,
)


class TestFamilyComposition(unittest.TestCase):
    def test_circle_ltd_is_a_subclass_of_annulus_ltd(self) -> None:
        """The hierarchy itself encodes 'a disk is an annulus with inner=0'."""
        self.assertTrue(issubclass(CircleLTDModel, AnnulusLTDModel))
        for ltd_cls in (SpeedOfInternetLTD, LowEnvelopeLTD):
            with self.subTest(ltd=ltd_cls.__name__):
                self.assertTrue(issubclass(ltd_cls, AnnulusLTDModel))

    def test_circle_ltd_with_annulus_mtl_is_allowed(self) -> None:
        """Previously rejected pairing; now legal — inner radius is 0."""
        cross_pairs = [
            (SpeedOfInternetLTD(), PlanarAnnulusMTL()),
            (SpeedOfInternetLTD(), PlanarAnnulusWeightedMTL()),
            (LowEnvelopeLTD(), PlanarAnnulusMTL()),
            (LowEnvelopeLTD(), PlanarAnnulusWeightedMTL()),
        ]
        for ltd, mtl in cross_pairs:
            with self.subTest(ltd=type(ltd).__name__, mtl=type(mtl).__name__):
                model = CBGModel(ltd, mtl, GeometricCentroidCTR())
                self.assertIs(model.ltd, ltd)
                self.assertIs(model.mtl, mtl)

    def test_annulus_ltd_with_circle_mtl_is_allowed_degraded(self) -> None:
        """Circle MTLs only read `tg_distance.upper_km`, so an annular
        constraint works — the inner bound is dropped.
        """
        permitted_pairs = [
            (NormalDistLTD(), PlanarCircleMTL()),
            (NormalDistLTD(), SphericalCircleMTL()),
            (BoundedSplineLTD(), PlanarCircleMTL()),
            (BoundedSplineLTD(), SphericalCircleMTL()),
        ]
        for ltd, mtl in permitted_pairs:
            with self.subTest(ltd=type(ltd).__name__, mtl=type(mtl).__name__):
                model = CBGModel(ltd, mtl, GeometricCentroidCTR())
                self.assertIs(model.ltd, ltd)
                self.assertIs(model.mtl, mtl)

    def test_matching_families_construct_successfully(self) -> None:
        """Native pairings keep working — sanity check that the cross-family
        unblock didn't break the same-family cases."""
        ok_pairs = [
            (SpeedOfInternetLTD(), PlanarCircleMTL()),
            (LowEnvelopeLTD(), SphericalCircleMTL()),
            (NormalDistLTD(), PlanarAnnulusMTL()),
            (BoundedSplineLTD(), PlanarAnnulusWeightedMTL()),
        ]
        for ltd, mtl in ok_pairs:
            with self.subTest(ltd=type(ltd).__name__, mtl=type(mtl).__name__):
                model = CBGModel(ltd, mtl, BoundaryVertexMeanCTR())
                self.assertIs(model.ltd, ltd)
                self.assertIs(model.mtl, mtl)


if __name__ == "__main__":
    unittest.main()
