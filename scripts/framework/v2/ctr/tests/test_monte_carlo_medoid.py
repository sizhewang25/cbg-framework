"""Tests for the v2 Monte Carlo sampled-medoid centroid."""

from __future__ import annotations

import unittest

from shapely.geometry import Point

from scripts.framework.v2.ctr.monte_carlo_medoid import MonteCarloMedoidCTR
from scripts.framework.v2.ctr.tests.helpers import (
    assert_coord_almost_equal,
    failed_vertices,
    rectangle_region,
    successful_region,
    successful_vertices,
    unequal_area_multipolygon,
)
from scripts.framework.v2.types import Error


class TestMonteCarloMedoidCTR(unittest.TestCase):
    def test_vertex_list_selects_sampled_medoid(self):
        ctr = MonteCarloMedoidCTR()

        result = ctr.select_centroid(
            successful_vertices([(0.0, 0.0), (0.0, 1.0), (0.0, 10.0)])
        )

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (0.0, 1.0))

    def test_single_vertex_returns_that_vertex(self):
        ctr = MonteCarloMedoidCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertTrue(result.success)
        assert_coord_almost_equal(self, result.tg_coord, (3.0, -7.0))

    def test_region_sampling_returns_feasible_point(self):
        region = rectangle_region()
        ctr = MonteCarloMedoidCTR(n_samples=8, seed=7)

        result = ctr.select_centroid(successful_region(region))

        self.assertTrue(result.success)
        self.assertTrue(region.contains(Point(result.tg_coord.lon, result.tg_coord.lat)))

    def test_multipolygon_region_sampling_returns_feasible_point(self):
        region = unequal_area_multipolygon()
        ctr = MonteCarloMedoidCTR(n_samples=32, seed=7)

        result = ctr.select_centroid(successful_region(region))

        self.assertTrue(result.success)
        self.assertTrue(region.contains(Point(result.tg_coord.lon, result.tg_coord.lat)))

    def test_zero_region_samples_falls_back_to_representative_point(self):
        region = rectangle_region()
        ctr = MonteCarloMedoidCTR(n_samples=0, seed=7)

        result = ctr.select_centroid(successful_region(region))

        self.assertTrue(result.success)
        self.assertTrue(region.contains(Point(result.tg_coord.lon, result.tg_coord.lat)))

    def test_failed_or_empty_results_carry_error(self):
        ctr = MonteCarloMedoidCTR()

        for mtl in (failed_vertices([(1.0, 2.0)]), successful_vertices([])):
            result = ctr.select_centroid(mtl)
            self.assertFalse(result.success)
            self.assertIsNone(result.tg_coord)
            self.assertEqual(result.error, Error.EMPTY_REGION)

    def test_method_field_is_stamped(self):
        ctr = MonteCarloMedoidCTR()

        result = ctr.select_centroid(successful_vertices([(3.0, -7.0)]))

        self.assertEqual(result.method, "MonteCarloMedoidCTR")


if __name__ == "__main__":
    unittest.main()


class TestReproducibility(unittest.TestCase):
    """The Sobol sampler is scrambled from `self.rng`, so the seed decides
    whether two runs of the same benchmark produce the same error_km.

    Before `DEFAULT_SEED` the default was None -- OS entropy -- and the sampled
    medoid moved ~2 km per call on a 1-degree region. Small against an H3-4
    cell, which is why the accuracy tables never showed it, but every
    `error_cdf_percentiles.csv` is written to metre precision.
    """

    def _medoid(self, **kwargs):
        ctr = MonteCarloMedoidCTR(n_samples=256, **kwargs)
        c = ctr.select_centroid(successful_region(rectangle_region())).tg_coord
        return (c.lat, c.lon)

    def test_the_default_is_reproducible(self):
        first = self._medoid()
        for _ in range(3):
            self.assertEqual(self._medoid(), first)

    def test_the_default_seed_is_pinned_to_the_other_sampling_ctrs(self):
        """`GeometricMedianCTR` has defaulted to 42 all along. Two sampling
        CTRs disagreeing on their default seed is a difference nobody intends
        and nobody would look for."""
        from scripts.framework.v2.ctr.geometric_median import GeometricMedianCTR

        import inspect

        other = inspect.signature(GeometricMedianCTR.__init__).parameters["seed"]
        self.assertEqual(MonteCarloMedoidCTR.DEFAULT_SEED, other.default)

    def test_an_explicit_seed_still_wins(self):
        a = self._medoid(seed=7)
        self.assertEqual(a, self._medoid(seed=7))
        self.assertNotEqual(a, self._medoid(seed=8), "seed had no effect")

    def test_seed_none_opts_back_into_entropy(self):
        """Kept reachable for measuring the sampler's own variance. Asserted as
        "not all identical" over several draws rather than "differs once",
        because two OS-seeded draws can legitimately collide on a coarse grid.
        """
        draws = {self._medoid(seed=None) for _ in range(6)}
        self.assertGreater(len(draws), 1)

    def test_the_runner_still_overrides_per_target(self):
        """runner.py replaces `.rng` per target; that must keep working and
        must take precedence over the constructor default, or a seeded run
        would silently collapse to one generator for every target."""
        import numpy as np

        def with_rng(seed):
            ctr = MonteCarloMedoidCTR(n_samples=256)
            ctr.rng = np.random.default_rng(seed)
            c = ctr.select_centroid(successful_region(rectangle_region())).tg_coord
            return (c.lat, c.lon)

        self.assertEqual(with_rng(99), with_rng(99))
        self.assertNotEqual(
            with_rng(99), self._medoid(), "the injected rng was ignored"
        )
