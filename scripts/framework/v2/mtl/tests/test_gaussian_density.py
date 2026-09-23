"""Tests for GaussianDensityMTL.

The load-bearing ones are `test_one_hostile_constraint_cannot_empty_the_field`
and `test_inner_hole_is_low_density_not_excluded`: those two properties are the
entire reason this class exists rather than another AnnulusMTLMethod, and both
are things the annular path gets wrong for Spotter (see
notes/2026-09-18-spotter-mtl-fidelity-gap.md).
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.framework.geometry import haversine
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.mtl.base import DensityMTLMethod
from scripts.framework.v2.mtl import gaussian_density as gd
from scripts.framework.v2.mtl.gaussian_density import GaussianDensityMTL
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.healpix import grid as HP

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


def mtl(**kwargs) -> GaussianDensityMTL:
    """A model with the grid named, so the 20-odd call sites below need not.

    `grid` is a required keyword with no default -- deliberately, so a stored
    `mtl_kwargs` payload from the H3 era fails loudly rather than replaying
    `resolution: 4` as a 192-cell nside. That makes every construction a
    two-argument call, and repeating the literal twenty times would bury the
    one thing worth reading. `TestConstructorValidation` still constructs the
    class directly, because the tripwire is what it is testing.
    """
    return GaussianDensityMTL(grid="healpix", **kwargs)


#: A cheap global pass for tests that do not care about the resolution. 768
#: cells, and coarse enough that a full pass costs nothing -- but not nside 1 or
#: 2, where a ring-1 disk is 58% and 19% of the globe and the pruning under test
#: stops being pruning.
SMALL = 8

def error_km(coord: Coord) -> float:
    return haversine((coord.lat, coord.lon), (TRUTH.lat, TRUTH.lon))


class TestRegistrationAndFamily(unittest.TestCase):
    def test_registered_under_gaussian_density(self):
        self.assertIs(MTL_REGISTRY["gaussian_density"], GaussianDensityMTL)

    def test_is_its_own_family_not_an_annulus_method(self):
        self.assertTrue(issubclass(GaussianDensityMTL, DensityMTLMethod))

    def test_method_is_stamped(self):
        result = mtl().multilaterate(exact_constraints())
        self.assertEqual(result.method, "GaussianDensityMTL")


class TestConstructorValidation(unittest.TestCase):
    """Constructed directly rather than through `mtl()` -- the argument
    handling is what these test."""

    def test_rejects_out_of_range_settings(self):
        for kwargs in (
            {"resolution": 3},          # not a power of two
            {"resolution": 100},        # nor this
            {"resolution": 0},
            {"resolution": -1},
            {"coarse_resolution": 12},
            {"top_k": 0},
            {"neighbor_ring": -1},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    GaussianDensityMTL(grid="healpix", **kwargs)

    def test_an_nside_past_the_budget_is_refused(self):
        """The global grid is built eagerly, before any output exists, so an
        over-large nside is an OOM with nothing to attribute it to. 1024 is a
        plausible typo for 128 and `validate_nside` accepts every power of two."""
        with self.assertRaises(ValueError) as ctx:
            GaussianDensityMTL(grid="healpix", resolution=1024)
        msg = str(ctx.exception)
        self.assertIn("12,582,912 cells", msg)
        self.assertIn("MB", msg)

    def test_the_grid_must_be_named(self):
        """A stored H3-era payload has `resolution: 4` and no `grid`. Since 4 is
        a legal nside, a default would replay it as a 192-cell globe instead of
        failing -- so `grid` is required, and the failure is a TypeError."""
        with self.assertRaises(TypeError):
            GaussianDensityMTL()
        with self.assertRaises(TypeError):
            GaussianDensityMTL(resolution=4, coarse_resolution=2, top_k=8)

    def test_h3_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            GaussianDensityMTL(grid="h3", resolution=4)
        self.assertIn("aperture-7", str(ctx.exception))

    def test_an_nside_as_a_float_is_accepted(self):
        """A YAML round-trip can hand this a `128.0`."""
        self.assertEqual(GaussianDensityMTL(grid="healpix", resolution=128.0).resolution, 128)


class TestInsufficientInput(unittest.TestCase):
    def test_empty_input_is_insufficient_data(self):
        result = mtl().multilaterate([])
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
        result = mtl().multilaterate(bounds_only)
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_DATA)
        self.assertIsNone(result.density)

    def test_two_constraints_is_insufficient_constraints(self):
        """Distinguished from INSUFFICIENT_DATA: VPs answered, just too few."""
        result = mtl().multilaterate(exact_constraints()[:2])
        self.assertFalse(result.success)
        self.assertEqual(result.error, Error.INSUFFICIENT_CONSTRAINTS)


class TestRecovery(unittest.TestCase):
    def test_mode_lands_within_one_cell_of_the_truth(self):
        result = mtl().multilaterate(exact_constraints())
        self.assertTrue(result.success)
        best = max(
            range(len(result.density.log_density)),
            key=lambda i: result.density.log_density[i],
        )
        # The grid argmax cannot beat its own quantisation. An nside-128 cell is
        # 50.9 km across, so the ceiling is the half-diagonal, sqrt(2)/2 of that
        # -- about 36 km. 50 km leaves ~40% headroom. (HEALPix has no pentagons;
        # it has 24 seven-neighbour corner cells, which is a different thing and
        # does not distort area at all.)
        self.assertLess(error_km(result.density.cells[best]), 50.0)

    def test_the_quantisation_ceiling_is_the_cell_half_diagonal(self):
        """Pins the number the tolerance above is derived from, so a change of
        resolution cannot silently invalidate it."""
        half_diagonal = HP.nominal_cell_km(128) * (2 ** 0.5) / 2
        self.assertAlmostEqual(half_diagonal, 36.0, delta=1.0)
        self.assertLess(half_diagonal, 50.0)

    def test_coarse_to_fine_agrees_with_the_global_pass(self):
        """The pruning is an approximation; on a unimodal field it must be free."""
        pruned = mtl(resolution=SMALL * 4, coarse_resolution=SMALL)
        globally = mtl(resolution=SMALL * 4, coarse_resolution=SMALL * 4)
        cons = exact_constraints()

        def argmax_coord(mtl):
            r = mtl.multilaterate(cons)
            i = max(range(len(r.density.log_density)), key=lambda j: r.density.log_density[j])
            return r.density.cells[i]

        a, b = argmax_coord(pruned), argmax_coord(globally)
        self.assertAlmostEqual(a.lat, b.lat, places=6)
        self.assertAlmostEqual(a.lon, b.lon, places=6)

    def test_participants_records_every_distribution_carrying_vp(self):
        result = mtl().multilaterate(exact_constraints())
        self.assertEqual(len(result.participating_vp_ids), len(VPS))


class TestNoVetoProperties(unittest.TestCase):
    """The two properties that make this Spotter rather than Octant-with-sigma."""

    def test_one_hostile_constraint_cannot_empty_the_field(self):
        """The annular path empties on a single under-predicting disk."""
        cons = exact_constraints()
        # Claim the target is 50 km from New York, which every other constraint
        # contradicts. Under an AND this is fatal; here it is one low score.
        cons.append(density_result(VPS[4], 50.0, 10.0, "hostile"))
        result = mtl().multilaterate(cons)
        self.assertTrue(result.success)
        self.assertGreater(len(result.intersection), 0)

    def test_a_hostile_constraint_still_moves_the_estimate(self):
        """Not vetoing is not the same as ignoring — the product must shift."""
        base = mtl().multilaterate(exact_constraints())
        cons = exact_constraints()
        cons.append(density_result(VPS[4], 50.0, 10.0, "hostile"))
        with_hostile = mtl().multilaterate(cons)

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
        result = mtl(resolution=SMALL, coarse_resolution=SMALL).multilaterate(cons)
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


class TestReportedRegion(unittest.TestCase):
    """`intersection` is the retained field, not a credible sub-region.

    `credible_mass` used to select the smallest set of cells reaching a mass
    threshold -- §III-B's "union of the most probable cells according to a
    required confidence level". It was removed because it could not mean that:
    the mass is normalised over the retained cells rather than the globe, so
    pruning turned "95%" into a statement about a neighbourhood. Nothing read
    it either; `density_argmax` takes the maximum of `log_density` directly.
    """

    def test_the_region_is_exactly_the_field(self):
        result = mtl().multilaterate(exact_constraints())
        self.assertEqual(len(result.intersection), len(result.density.cells))
        self.assertEqual(list(result.intersection), list(result.density.cells))

    def test_credible_mass_is_gone_rather_than_ignored(self):
        """Silently accepting it would leave configs asserting a threshold with
        no effect, which is the failure mode `_reject_binning_kwargs` exists to
        prevent on the LTD side."""
        with self.assertRaises(TypeError):
            GaussianDensityMTL(grid="healpix", credible_mass=0.95)

    def test_the_region_is_never_empty_on_success(self):
        result = mtl().multilaterate(exact_constraints())
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(result.intersection), 1)

    def test_the_footprint_area_is_exact(self):
        """The one thing equal-area cells buy here: a cell count converts to an
        area with no qualification, so the pruning's extent is statable rather
        than hedged."""
        result = mtl().multilaterate(exact_constraints())
        n = len(result.density.cells)
        area = n * HP.pixel_area_km2(128)
        self.assertEqual(result.density.resolution, 128)
        self.assertGreater(area, 0.0)
        self.assertAlmostEqual(area / n, 2594.3, delta=0.1)

    def test_wider_sigma_does_not_change_the_retained_count(self):
        """The retained set is decided by `top_k`/`neighbor_ring` and the
        descent depth, not by the posterior -- which is precisely why a mass
        threshold over it was meaningless."""
        tight = mtl().multilaterate(exact_constraints(sigma_frac=0.02))
        loose = mtl().multilaterate(exact_constraints(sigma_frac=0.30))
        self.assertEqual(len(tight.intersection), len(loose.intersection))

    def test_intersection_is_a_vertex_list_for_the_benchmark_writer(self):
        """`_intersection_kind` maps list -> "vertex_list"; anything else is "unknown"."""
        result = mtl().multilaterate(exact_constraints())
        self.assertIsInstance(result.intersection, list)
        self.assertTrue(all(isinstance(c, Coord) for c in result.intersection))

    def test_the_field_reports_the_grid_it_used(self):
        result = mtl().multilaterate(exact_constraints())
        self.assertEqual(result.density.grid, "healpix")


class TestGridIsBuiltOutsideInstrumentation(unittest.TestCase):
    """The coarse grid must be built in `__init__`, not on first use.

    `runner.py` constructs the model before `measure_block("fit")` and before
    the per-target loop, so construction is measured by nothing. A lazy build
    instead lands inside the first target's `cm("mtl")` block. Measured when it
    was lazy: first target 113 ms against a 74 ms median -- the *maximum* in
    most folds -- and `mtl_alloc_peak_bytes` 1.15 MB against 0.53 MB.
    `analysis/v3/modules/cost.py` reduces memory across stages with `max` and
    already records a 22 MB warmup artifact setting a combo's whole MTL memory
    figure; this would have been the next one.
    """

    def test_the_grid_exists_before_any_multilateration(self):
        m = mtl(resolution=SMALL, coarse_resolution=SMALL)
        cells, lats, lons = m._coarse_grid()
        self.assertEqual(len(cells), HP.npix(SMALL))
        self.assertEqual(len(lats), len(cells))
        self.assertEqual(len(lons), len(cells))

    def test_the_grid_is_cached_the_moment_the_model_exists(self):
        """Replaces a timing assertion that has stopped discriminating.

        That test compared the first `multilaterate` against later ones on the
        premise that a lazy build was ~15x. `pix2ang` is vectorised, so the
        build is now ~27 ms even for the full nside-128 globe, against ~1,085 ms
        for one global `_log_density` pass -- a lazy build would be a 2.5%
        first-call penalty, well under any bound worth asserting. The property
        itself still matters, so assert it directly instead of through a clock.
        """
        gd._GLOBAL_GRID_CACHE.pop(("healpix", SMALL), None)
        m = mtl(resolution=SMALL, coarse_resolution=SMALL)
        self.assertIn(("healpix", SMALL), gd._GLOBAL_GRID_CACHE)
        self.assertIs(m._coarse_grid()[1], gd._GLOBAL_GRID_CACHE[("healpix", SMALL)][1])

    def test_the_shared_grid_is_read_only(self):
        """It is shared across instances, so an in-place write would corrupt
        every model in the process, not one result. `cells` is covered too:
        under H3 it was a list of opaque strings nobody indexed into, but it is
        now an int64 array that `_evaluate` slices every descent."""
        cells, lats, lons = mtl(
            resolution=SMALL, coarse_resolution=SMALL
        )._coarse_grid()
        for arr in (cells, lats, lons):
            with self.assertRaises(ValueError):
                arr[0] = 0

    def test_instances_at_one_resolution_share_the_grid(self):
        a = mtl(resolution=SMALL, coarse_resolution=SMALL)._coarse_grid()
        b = mtl(resolution=SMALL, coarse_resolution=SMALL)._coarse_grid()
        self.assertIs(a[1], b[1], "second instance rebuilt the grid")

    def test_the_cache_key_names_the_scheme(self):
        """The grid's identity is (scheme, nside, order). A bare nside works
        only because the order is a module constant elsewhere; naming the
        scheme keeps that from being an invisible assumption."""
        mtl(resolution=SMALL, coarse_resolution=SMALL)
        self.assertIn(("healpix", SMALL), gd._GLOBAL_GRID_CACHE)


class TestDescentNesting(unittest.TestCase):
    """Every HEALPix id is a legal id at *some* nside, so a descent that forgets
    to advance its nside produces silently displaced centres rather than an
    exception -- H3's opaque strings raised instead. These pin the nesting the
    descent relies on."""

    def test_each_descended_cell_degrades_into_the_level_above(self):
        carried = np.array([5, 97, 300], dtype=np.int64)
        kids = HP.children(carried).ravel()
        np.testing.assert_array_equal(
            HP.degrade(kids, SMALL * 2, SMALL), np.repeat(carried, 4)
        )

    def test_a_descended_cell_round_trips_at_its_own_nside(self):
        """Catches the lat/lon swap: `pix2ang` returns (lat, lon) while
        `cell_rings` returns (lon, lat), and `_haversine_km_to_many` consumes
        either without complaint."""
        kids = HP.children(np.array([5, 97, 300], dtype=np.int64)).ravel()
        centres = HP.pix2ang(kids, SMALL * 2)
        np.testing.assert_array_equal(
            HP.ang2pix(centres[:, 0], centres[:, 1], SMALL * 2), kids
        )

    def test_the_field_is_reported_at_the_requested_resolution(self):
        """The descent must land exactly on `resolution`, not one level short
        or long -- the failure a mis-advanced nside would produce."""
        result = mtl(resolution=SMALL * 4, coarse_resolution=SMALL).multilaterate(
            exact_constraints()
        )
        lats = [c.lat for c in result.density.cells]
        lons = [c.lon for c in result.density.cells]
        pix = HP.ang2pix(lats, lons, SMALL * 4)
        np.testing.assert_array_equal(
            HP.pix2ang(pix, SMALL * 4)[:, 0], np.asarray(lats)
        )

    def test_a_single_global_pass_skips_the_descent(self):
        result = mtl(resolution=SMALL, coarse_resolution=SMALL).multilaterate(
            exact_constraints()
        )
        self.assertEqual(len(result.density.cells), HP.npix(SMALL))

    def test_a_coarse_resolution_above_the_fine_one_is_a_global_pass(self):
        """Documented behaviour: it is how a caller asks for no pruning."""
        result = mtl(resolution=SMALL, coarse_resolution=SMALL * 4).multilaterate(
            exact_constraints()
        )
        self.assertEqual(len(result.density.cells), HP.npix(SMALL))

    def test_neighbor_ring_zero_still_descends(self):
        result = mtl(
            resolution=SMALL * 2, coarse_resolution=SMALL, top_k=4, neighbor_ring=0
        ).multilaterate(exact_constraints())
        self.assertTrue(result.success)
        self.assertEqual(len(result.density.cells), 4 * 4)


if __name__ == "__main__":
    unittest.main()
