"""Tests for the four density-aware CTRs.

`test_mle_beats_the_grid_argmax` is the one that matters for the paper: it pins
that `DensityMLECTR` removes the grid's quantisation error rather than merely
reproducing the seed, which is what makes it usable as the exact reference for
scoring approximations of Spotter.
"""

from __future__ import annotations

import unittest

from scripts.framework.geometry import haversine
from scripts.framework.v2.ctr.density_point import (
    DensityArgmaxCTR,
    DensityMeanCTR,
    DensityMLECTR,
    DensityRegionCenterCTR,
)
from scripts.framework.v2.mtl.base import DensityField, MTLResult
from scripts.framework.v2.mtl.gaussian_density import GaussianDensityMTL
from scripts.framework.v2.mtl.tests.test_gaussian_density import (
    TRUTH,
    error_km,
    exact_constraints,
)
from scripts.framework.v2.registry import CTR_REGISTRY
from scripts.framework.v2.types import Coord, Error

ALL_CTRS = (
    DensityArgmaxCTR,
    DensityMeanCTR,
    DensityRegionCenterCTR,
    DensityMLECTR,
)


def solved(**mtl_kwargs) -> MTLResult:
    return GaussianDensityMTL(**mtl_kwargs).multilaterate(exact_constraints())


class TestRegistration(unittest.TestCase):
    def test_all_four_are_registered(self):
        for name, cls in (
            ("density_argmax", DensityArgmaxCTR),
            ("density_mean", DensityMeanCTR),
            ("density_region_center", DensityRegionCenterCTR),
            ("density_mle", DensityMLECTR),
        ):
            with self.subTest(name=name):
                self.assertIs(CTR_REGISTRY[name], cls)

    def test_method_is_stamped(self):
        mtl = solved()
        for cls in ALL_CTRS:
            with self.subTest(ctr=cls.__name__):
                self.assertEqual(cls().select_centroid(mtl).method, cls.__name__)


class TestRefusesWithoutDensity(unittest.TestCase):
    """A geometric MTLResult must not be silently read as a density one."""

    def test_no_density_field_is_insufficient_data(self):
        geometric = MTLResult(success=True, intersection=[Coord(0.0, 0.0)])
        for cls in ALL_CTRS:
            with self.subTest(ctr=cls.__name__):
                result = cls().select_centroid(geometric)
                self.assertFalse(result.success)
                self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_empty_field_is_insufficient_data(self):
        empty = MTLResult(
            success=True,
            density=DensityField(cells=(), log_density=(), constraints=()),
        )
        for cls in ALL_CTRS:
            with self.subTest(ctr=cls.__name__):
                self.assertFalse(cls().select_centroid(empty).success)


class TestRecovery(unittest.TestCase):
    def test_every_estimator_lands_near_the_truth(self):
        mtl = solved()
        for cls in ALL_CTRS:
            with self.subTest(ctr=cls.__name__):
                result = cls().select_centroid(mtl)
                self.assertTrue(result.success)
                self.assertLess(error_km(result.tg_coord), 60.0)

    def test_mle_beats_the_grid_argmax(self):
        """The whole point of the continuous refine: sub-cell accuracy."""
        mtl = solved(resolution=3)  # a deliberately coarse grid
        grid = DensityArgmaxCTR().select_centroid(mtl)
        mle = DensityMLECTR().select_centroid(mtl)
        self.assertTrue(grid.success and mle.success)
        self.assertLess(error_km(mle.tg_coord), error_km(grid.tg_coord))
        # With exact mu the least-squares optimum is the truth itself.
        self.assertLess(error_km(mle.tg_coord), 1.0)

    def test_argmax_is_invariant_to_credible_mass(self):
        """Truncation moves the mean and the region centre, never the mode."""
        a = DensityArgmaxCTR().select_centroid(solved(credible_mass=0.5))
        b = DensityArgmaxCTR().select_centroid(solved(credible_mass=0.99))
        self.assertAlmostEqual(a.tg_coord.lat, b.tg_coord.lat, places=9)
        self.assertAlmostEqual(a.tg_coord.lon, b.tg_coord.lon, places=9)


class TestSphericalAveraging(unittest.TestCase):
    def test_mean_does_not_break_across_the_antimeridian(self):
        """Arithmetic lon averaging would put this near 0 instead of 180."""
        field = DensityField(
            cells=(Coord(0.0, 179.0), Coord(0.0, -179.0)),
            log_density=(0.0, 0.0),
            constraints=(),
        )
        result = DensityMeanCTR().select_centroid(MTLResult(success=True, density=field))
        self.assertTrue(result.success)
        self.assertLess(
            haversine((result.tg_coord.lat, result.tg_coord.lon), (0.0, 180.0)), 1.0
        )

    def test_antipodal_pair_has_no_mean_direction(self):
        field = DensityField(
            cells=(Coord(0.0, 0.0), Coord(0.0, 180.0)),
            log_density=(0.0, 0.0),
            constraints=(),
        )
        result = DensityMeanCTR().select_centroid(MTLResult(success=True, density=field))
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.DEGENERATE_REGION)


class TestMLERobustness(unittest.TestCase):
    def test_mle_without_constraints_reports_insufficient_data(self):
        """The grid alone cannot support a continuous refine."""
        field = DensityField(
            cells=(Coord(39.0, -105.0),), log_density=(0.0,), constraints=()
        )
        result = DensityMLECTR().select_centroid(MTLResult(success=True, density=field))
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_mle_output_stays_in_range(self):
        mtl = solved()
        coord = DensityMLECTR().select_centroid(mtl).tg_coord
        self.assertGreaterEqual(coord.lat, -90.0)
        self.assertLessEqual(coord.lat, 90.0)
        self.assertGreaterEqual(coord.lon, -180.0)
        self.assertLessEqual(coord.lon, 180.0)

    def test_rejects_bad_max_nfev(self):
        with self.assertRaises(ValueError):
            DensityMLECTR(max_nfev=0)


class TestRegionCenter(unittest.TestCase):
    def test_region_center_needs_the_region(self):
        """Density present but intersection absent is an empty region."""
        field = DensityField(
            cells=(Coord(39.0, -105.0),), log_density=(0.0,), constraints=()
        )
        result = DensityRegionCenterCTR().select_centroid(
            MTLResult(success=True, density=field, intersection=None)
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.EMPTY_REGION)

    def test_region_center_ignores_density_inside_the_region(self):
        """Otherwise it is density_mean under another name."""
        field = DensityField(
            cells=(Coord(0.0, 0.0), Coord(0.0, 2.0)),
            log_density=(0.0, -20.0),
            constraints=(),
        )
        mtl = MTLResult(
            success=True, density=field, intersection=[Coord(0.0, 0.0), Coord(0.0, 2.0)]
        )
        centre = DensityRegionCenterCTR().select_centroid(mtl).tg_coord
        weighted = DensityMeanCTR().select_centroid(mtl).tg_coord
        self.assertAlmostEqual(centre.lon, 1.0, places=6)
        self.assertLess(weighted.lon, 0.5)


if __name__ == "__main__":
    unittest.main()
