"""Tests for GaussianDensityMTL.

The load-bearing ones are `test_one_hostile_constraint_cannot_empty_the_field`
and `test_inner_hole_is_low_density_not_excluded`: those two properties are the
entire reason this class exists rather than another AnnulusMTLMethod, and both
are things the annular path gets wrong for Spotter (see
notes/2026-09-18-spotter-mtl-fidelity-gap.md).
"""

from __future__ import annotations

import unittest

from scripts.framework.geometry import haversine
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import DensityMTLMethod
from scripts.framework.v2.mtl.gaussian_density import GaussianDensityMTL
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId

TRUTH = Coord(39.74, -104.99)  # Denver
VPS = (
    Coord(47.61, -122.33),  # Seattle
    Coord(34.05, -118.24),  # Los Angeles
    Coord(41.88, -87.63),   # Chicago
    Coord(29.76, -95.37),   # Houston
    Coord(40.71, -74.01),   # New York
    Coord(33.75, -84.39),   # Atlanta
)


def density_result(vp: Coord, mu_km: float, sigma_km: float, vp_id: str) -> LTDResult:
    """An LTDResult carrying a distribution, as NormalDistLTD emits."""
    return LTDResult(
        success=True,
        vp_id=VpId(vp_id),
        vp_coord=vp,
        latency=Latency(max(1.0, mu_km / 100.0)),
        tg_distance=Distance(
            upper_km=mu_km + sigma_km,
            lower_km=max(0.0, mu_km - sigma_km),
            mu_km=mu_km,
            sigma_km=sigma_km,
        ),
    )


def exact_constraints(sigma_frac: float = 0.05) -> list[LTDResult]:
    """One constraint per VP with mu set to the true distance."""
    out = []
    for i, vp in enumerate(VPS):
        d = haversine((vp.lat, vp.lon), (TRUTH.lat, TRUTH.lon))
        out.append(density_result(vp, d, max(1.0, sigma_frac * d), f"vp{i}"))
    return out


def error_km(coord: Coord) -> float:
    return haversine((coord.lat, coord.lon), (TRUTH.lat, TRUTH.lon))


class TestRegistrationAndFamily(unittest.TestCase):
    def test_registered_under_gaussian_density(self):
        self.assertIs(MTL_REGISTRY["gaussian_density"], GaussianDensityMTL)

    def test_is_its_own_family_not_an_annulus_method(self):
        self.assertTrue(issubclass(GaussianDensityMTL, DensityMTLMethod))

    def test_method_is_stamped(self):
        result = GaussianDensityMTL().multilaterate(exact_constraints())
        self.assertEqual(result.method, "GaussianDensityMTL")


class TestConstructorValidation(unittest.TestCase):
    def test_rejects_out_of_range_settings(self):
        for kwargs in (
            {"resolution": 16},
            {"resolution": -1},
            {"coarse_resolution": 16},
            {"top_k": 0},
            {"neighbor_ring": -1},
            {"credible_mass": 0.0},
            {"credible_mass": 1.5},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    GaussianDensityMTL(**kwargs)


class TestInsufficientInput(unittest.TestCase):
    def test_empty_input_is_insufficient_data(self):
        result = GaussianDensityMTL().multilaterate([])
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)

    def test_bounds_only_ltd_is_insufficient_data_not_a_crash(self):
        """A composition error (geometric LTD + density MTL) must be reported."""
        bounds_only = [
            LTDResult(
                success=True,
                vp_id=VpId(f"vp{i}"),
                vp_coord=vp,
                latency=Latency(10.0),
                tg_distance=Distance(upper_km=1000.0, lower_km=500.0),
            )
            for i, vp in enumerate(VPS)
        ]
        result = GaussianDensityMTL().multilaterate(bounds_only)
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)
        self.assertIsNone(result.density)

    def test_two_constraints_is_insufficient_constraints(self):
        """Distinguished from INSUFFICIENT_DATA: VPs answered, just too few."""
        result = GaussianDensityMTL().multilaterate(exact_constraints()[:2])
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_CONSTRAINTS)


class TestRecovery(unittest.TestCase):
    def test_mode_lands_within_one_cell_of_the_truth(self):
        result = GaussianDensityMTL(resolution=4).multilaterate(exact_constraints())
        self.assertTrue(result.success)
        best = max(
            range(len(result.density.log_density)),
            key=lambda i: result.density.log_density[i],
        )
        # An H3-4 cell is ~20 km edge, so the grid argmax cannot do better than
        # its own quantisation; 50 km is that bound with room for the pentagon
        # cells' distortion.
        self.assertLess(error_km(result.density.cells[best]), 50.0)

    def test_coarse_to_fine_agrees_with_the_global_pass(self):
        """The pruning is an approximation; on a unimodal field it must be free."""
        pruned = GaussianDensityMTL(resolution=3, coarse_resolution=1)
        globally = GaussianDensityMTL(resolution=3, coarse_resolution=3)
        cons = exact_constraints()

        def argmax_coord(mtl):
            r = mtl.multilaterate(cons)
            i = max(range(len(r.density.log_density)), key=lambda j: r.density.log_density[j])
            return r.density.cells[i]

        a, b = argmax_coord(pruned), argmax_coord(globally)
        self.assertAlmostEqual(a.lat, b.lat, places=6)
        self.assertAlmostEqual(a.lon, b.lon, places=6)

    def test_participants_records_every_distribution_carrying_vp(self):
        result = GaussianDensityMTL().multilaterate(exact_constraints())
        self.assertEqual(len(result.participating_vp_ids), len(VPS))


class TestNoVetoProperties(unittest.TestCase):
    """The two properties that make this Spotter rather than Octant-with-sigma."""

    def test_one_hostile_constraint_cannot_empty_the_field(self):
        """The annular path empties on a single under-predicting disk."""
        cons = exact_constraints()
        # Claim the target is 50 km from New York, which every other constraint
        # contradicts. Under an AND this is fatal; here it is one low score.
        cons.append(density_result(VPS[4], 50.0, 10.0, "hostile"))
        result = GaussianDensityMTL().multilaterate(cons)
        self.assertTrue(result.success)
        self.assertGreater(len(result.intersection), 0)

    def test_a_hostile_constraint_still_moves_the_estimate(self):
        """Not vetoing is not the same as ignoring — the product must shift."""
        base = GaussianDensityMTL().multilaterate(exact_constraints())
        cons = exact_constraints()
        cons.append(density_result(VPS[4], 50.0, 10.0, "hostile"))
        with_hostile = GaussianDensityMTL().multilaterate(cons)

        def argmax(r):
            i = max(range(len(r.density.log_density)), key=lambda j: r.density.log_density[j])
            return r.density.cells[i]

        moved = haversine(
            (argmax(base).lat, argmax(base).lon),
            (argmax(with_hostile).lat, argmax(with_hostile).lon),
        )
        self.assertGreater(moved, 0.0)

    def test_inner_hole_is_low_density_not_excluded(self):
        """A position at s << mu scores badly but remains in the field."""
        cons = exact_constraints()
        result = GaussianDensityMTL(resolution=3, coarse_resolution=3).multilaterate(cons)
        # Every VP's own location sits deep inside its own annulus hole. Under
        # the annular path those points are subtracted away entirely; here they
        # must still be present, merely improbable.
        vp = VPS[0]
        nearest = min(
            range(len(result.density.cells)),
            key=lambda i: haversine(
                (result.density.cells[i].lat, result.density.cells[i].lon),
                (vp.lat, vp.lon),
            ),
        )
        best = max(
            range(len(result.density.log_density)),
            key=lambda i: result.density.log_density[i],
        )
        self.assertLess(
            result.density.log_density[nearest], result.density.log_density[best]
        )
        self.assertTrue(
            all(v == v for v in result.density.log_density), "no NaN in the field"
        )


class TestCredibleRegion(unittest.TestCase):
    def test_region_is_never_empty(self):
        result = GaussianDensityMTL(credible_mass=1e-9).multilaterate(
            exact_constraints()
        )
        self.assertGreaterEqual(len(result.intersection), 1)

    def test_larger_credible_mass_is_never_a_smaller_region(self):
        cons = exact_constraints(sigma_frac=0.25)
        small = GaussianDensityMTL(credible_mass=0.5).multilaterate(cons)
        large = GaussianDensityMTL(credible_mass=0.99).multilaterate(cons)
        self.assertLessEqual(len(small.intersection), len(large.intersection))

    def test_wider_sigma_widens_the_region(self):
        tight = GaussianDensityMTL().multilaterate(exact_constraints(sigma_frac=0.02))
        loose = GaussianDensityMTL().multilaterate(exact_constraints(sigma_frac=0.30))
        self.assertLess(len(tight.intersection), len(loose.intersection))

    def test_intersection_is_a_vertex_list_for_the_benchmark_writer(self):
        """`_intersection_kind` maps list -> "vertex_list"; anything else is "unknown"."""
        result = GaussianDensityMTL().multilaterate(exact_constraints())
        self.assertIsInstance(result.intersection, list)
        self.assertTrue(all(isinstance(c, Coord) for c in result.intersection))


if __name__ == "__main__":
    unittest.main()
