"""Tests for DensityArgmaxCTR, the only density-aware CTR.

`test_refuses_a_geometric_mtl_result` is the load-bearing one: a density CTR
handed a geometric region must refuse rather than quietly average the
`intersection` coords, or the combo id stops describing what produced the
number.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2.ctr.density_point import DensityArgmaxCTR
from scripts.framework.v2.mtl.base import DensityField, MTLResult
from scripts.framework.v2.mtl.gaussian_density import GaussianDensityMTL
from scripts.framework.v2.mtl.tests.test_gaussian_density import (
    error_km,
    exact_constraints,
)
from scripts.framework.v2.registry import CTR_REGISTRY
from scripts.framework.v2.types import Coord, Error


def solved(**mtl_kwargs) -> MTLResult:
    return GaussianDensityMTL(**mtl_kwargs).multilaterate(exact_constraints())


class TestRegistration(unittest.TestCase):
    def test_registered_under_density_argmax(self):
        self.assertIs(CTR_REGISTRY["density_argmax"], DensityArgmaxCTR)

    def test_the_retired_ctrs_are_gone_from_the_registry(self):
        """They were measured, then removed. A stale combo id naming one of
        them must fail at composition rather than silently resolve."""
        for retired in ("density_mean", "density_region_center", "density_mle"):
            with self.subTest(ctr=retired):
                self.assertNotIn(retired, CTR_REGISTRY)

    def test_method_is_stamped(self):
        self.assertEqual(
            DensityArgmaxCTR().select_centroid(solved()).method, "DensityArgmaxCTR"
        )


class TestRefusesWithoutDensity(unittest.TestCase):
    def test_refuses_a_geometric_mtl_result(self):
        geometric = MTLResult(success=True, intersection=[Coord(0.0, 0.0)])
        result = DensityArgmaxCTR().select_centroid(geometric)
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_empty_field_is_insufficient_data(self):
        empty = MTLResult(
            success=True, density=DensityField(cells=(), log_density=())
        )
        self.assertFalse(DensityArgmaxCTR().select_centroid(empty).success)


class TestRecovery(unittest.TestCase):
    def test_lands_near_the_truth(self):
        result = DensityArgmaxCTR().select_centroid(solved())
        self.assertTrue(result.success)
        self.assertLess(error_km(result.tg_coord), 60.0)

    def test_picks_the_maximum_not_the_first_cell(self):
        field = DensityField(
            cells=(Coord(0.0, 0.0), Coord(10.0, 10.0), Coord(20.0, 20.0)),
            log_density=(-5.0, -1.0, -9.0),
        )
        coord = DensityArgmaxCTR().select_centroid(
            MTLResult(success=True, density=field)
        ).tg_coord
        self.assertEqual((coord.lat, coord.lon), (10.0, 10.0))

    def test_invariant_to_credible_mass(self):
        """Truncation moves a mean or a region centre; it cannot move the mode.

        This is the property that made argmax the one CTR kept under
        coarse-to-fine pruning.
        """
        a = DensityArgmaxCTR().select_centroid(solved(credible_mass=0.5))
        b = DensityArgmaxCTR().select_centroid(solved(credible_mass=0.99))
        self.assertAlmostEqual(a.tg_coord.lat, b.tg_coord.lat, places=9)
        self.assertAlmostEqual(a.tg_coord.lon, b.tg_coord.lon, places=9)


if __name__ == "__main__":
    unittest.main()
