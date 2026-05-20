"""Composition-time guard: only the unsafe cross-family pair raises.

CBGModel's family check is asymmetric:
  * AnnulusMTLMethod + CircleLTDModel  → IncompatibleStagesError
        Annulus MTLs are built around the inner bound; pairing them with
        a Circle LTD (which always emits lower_km=0) would silently strip
        their distinguishing information.
  * AnnulusLTDModel + CircleMTLMethod  → allowed, degraded
        Circle MTLs only consume upper_km, so the annular constraint
        degrades cleanly to a disk. The inner bound is dropped; the
        pipeline still runs.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2 import (
    BoundaryVertexMeanCTR,
    BoundedSplineLTD,
    CBGModel,
    GeometricCentroidCTR,
    IncompatibleStagesError,
    LowEnvelopeLTD,
    NormalDistLTD,
    PlanarAnnulusMTL,
    PlanarAnnulusWeightedMTL,
    PlanarCircleMTL,
    SpeedOfInternetLTD,
    SphericalCircleMTL,
)


class TestFamilyValidation(unittest.TestCase):
    def test_circle_ltd_with_annulus_mtl_raises(self) -> None:
        """CircleLTDModel produces disks; an AnnulusMTLMethod must reject it."""
        cross_pairs = [
            (SpeedOfInternetLTD(), PlanarAnnulusMTL()),
            (SpeedOfInternetLTD(), PlanarAnnulusWeightedMTL()),
            (LowEnvelopeLTD(), PlanarAnnulusMTL()),
            (LowEnvelopeLTD(), PlanarAnnulusWeightedMTL()),
        ]
        for ltd, mtl in cross_pairs:
            with self.subTest(ltd=type(ltd).__name__, mtl=type(mtl).__name__):
                with self.assertRaises(IncompatibleStagesError):
                    CBGModel(ltd, mtl, GeometricCentroidCTR())

    def test_annulus_ltd_with_circle_mtl_is_allowed_degraded(self) -> None:
        """AnnulusLTDModel + CircleMTLMethod is the deliberate degraded path.

        Circle MTLs only read `tg_distance.upper_km`, so an annular constraint
        works — the inner bound is dropped. CBGModel allows it.
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
        """Matched pairs build without raising — the negative test above means nothing if these fail."""
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

    def test_incompatible_stages_error_is_typeerror(self) -> None:
        """Catching TypeError must continue to catch IncompatibleStagesError."""
        with self.assertRaises(TypeError):
            CBGModel(SpeedOfInternetLTD(), PlanarAnnulusMTL(), GeometricCentroidCTR())

    def test_incompatible_stages_error_message_names_both_classes(self) -> None:
        """Error message must identify both offending stages for debuggability."""
        with self.assertRaises(IncompatibleStagesError) as exc:
            CBGModel(SpeedOfInternetLTD(), PlanarAnnulusMTL(), GeometricCentroidCTR())
        message = str(exc.exception)
        self.assertIn("PlanarAnnulusMTL", message)
        self.assertIn("SpeedOfInternetLTD", message)


if __name__ == "__main__":
    unittest.main()
