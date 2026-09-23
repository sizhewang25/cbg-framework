"""Tests for v4's HEALPix primitives.

The load-bearing ones are `TestLadderIsMonotone` and
`TestDegradeEqualsRebinning`: those two properties are the entire reason v4
exists as a separate package, and they are the two H3 cannot satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.v4.modules import healpix as H

#: A scattered sample, avoiding the poles where every grid is awkward.
_RNG = np.random.default_rng(20260923)
_LAT = _RNG.uniform(-85.0, 85.0, 4000)
_LON = _RNG.uniform(-180.0, 180.0, 4000)

DENVER = (39.7392, -104.9903)
SEATTLE = (47.4490, -122.3090)
#: Spotter's prediction for tg-e1a1545 — the Canadian Arctic. Scored *correct*
#: under v3's nearest-seed rule at 2,360 km from the Seattle truth.
ARCTIC = (63.7369, -97.4685)


class TestValidation:
    def test_nside_must_be_a_power_of_two(self):
        for bad in (0, -1, 3, 100, 130):
            with pytest.raises(ValueError, match="power of two"):
                H.validate_nside(bad)

    def test_powers_of_two_pass(self):
        for good in (1, 2, 16, 128, 1024):
            assert H.validate_nside(good) == good

    def test_degrade_refuses_to_refine(self):
        with pytest.raises(ValueError, match="must not exceed"):
            H.degrade([0], 16, 128)


class TestGeometry:
    def test_cell_count_and_area_close_the_sphere(self):
        for nside in H.NSIDE_LADDER:
            assert H.npix(nside) == 12 * nside**2
            total = H.pixel_area_km2(nside) * H.npix(nside)
            assert total == pytest.approx(4 * np.pi * 6371.0**2, rel=1e-9)

    def test_nside_128_is_the_metro_scale(self):
        """The working resolution, quoted throughout the module docs."""
        assert H.npix(128) == 196_608
        assert H.pixel_area_km2(128) == pytest.approx(2594.3, abs=0.1)
        assert H.nominal_cell_km(128) == pytest.approx(50.9, abs=0.1)

    def test_pix2ang_round_trips(self):
        """A swapped (lat, lon) pair cannot survive this — it lands in a
        different cell, or is rejected for a latitude past +/-90."""
        pix = H.ang2pix(_LAT, _LON, 128)
        centres = H.pix2ang(pix, 128)
        assert np.array_equal(H.ang2pix(centres[:, 0], centres[:, 1], 128), pix)

    def test_pix2ang_normalises_longitude(self):
        """astropy returns [0, 360); a map picking its projection centre from
        `lon.mean()` would render in the wrong hemisphere."""
        centres = H.pix2ang(H.ang2pix(_LAT, _LON, 128), 128)
        assert centres[:, 1].min() >= -180.0
        assert centres[:, 1].max() < 180.0

    def test_empty_input_is_shaped_not_raised(self):
        assert H.pix2ang([], 128).shape == (0, 2)
        assert H.neighbours([], 128).shape == (0, 8)
        assert H.cell_rings([], 128) == []


class TestDegradeEqualsRebinning:
    """The bit shift must equal geometric containment, or the ladder is a lie.

    This is what buys the whole ladder for one `ang2pix` pass, and it is exactly
    what H3 cannot offer: `cell_to_parent` there is exact on the index but is
    not a geometric container, and the two disagreed on 552 of 5,906 targets at
    res 2 on this repo's own data.
    """

    @pytest.mark.parametrize("nside_to", [64, 32, 16, 8, 4, 2, 1])
    def test_shift_matches_ang2pix(self, nside_to):
        pix = H.ang2pix(_LAT, _LON, 128)
        assert np.array_equal(
            H.degrade(pix, 128, nside_to), H.ang2pix(_LAT, _LON, nside_to)
        )

    def test_degrade_is_identity_at_the_same_nside(self):
        pix = H.ang2pix(_LAT, _LON, 128)
        assert np.array_equal(H.degrade(pix, 128, 128), pix)


class TestLadderIsMonotone:
    """Two points that share a cell must share every coarser cell.

    Under H3 this failed 705 times on the real predictions, and visibly in
    aggregate: `vanilla_cbg` on as01 scored 0.594 at res 1 but 0.659 at res 2,
    higher than its own parent. Zero tolerance here.
    """

    def test_sharing_a_fine_cell_implies_sharing_every_coarser_one(self):
        a_lat, a_lon = _LAT[:2000], _LON[:2000]
        b_lat = a_lat + _RNG.normal(0, 0.4, a_lat.size)
        b_lon = a_lon + _RNG.normal(0, 0.4, a_lon.size)
        agree = {
            n: H.ang2pix(a_lat, a_lon, n) == H.ang2pix(b_lat, b_lon, n)
            for n in H.NSIDE_LADDER
        }
        fine_to_coarse = sorted(H.NSIDE_LADDER, reverse=True)
        violations = 0
        for finer, coarser in zip(fine_to_coarse, fine_to_coarse[1:]):
            violations += int(np.sum(agree[finer] & ~agree[coarser]))
        assert violations == 0

    def test_occupancy_never_grows_as_cells_grow(self):
        counts = H.occupied_cells_by_nside(_LAT, _LON)
        for finer, coarser in zip(H.NSIDE_LADDER, H.NSIDE_LADDER[1:]):
            assert counts[coarser] <= counts[finer]

    def test_ladder_is_downward_only(self):
        """A rung finer than the answer space was built at would score against
        cells no class was ever defined on."""
        assert H.ladder_for(32) == (32, 16)
        assert H.ladder_for(128) == (128, 64, 32, 16)
        assert H.ladder_for(16) == (16,)


class TestNeighbours:
    def test_eight_neighbours_and_the_24_corner_cells(self):
        """24 cells per nside sit at a base-tessellation corner and have 7.
        Real, and the reason `-1` has to be filtered rather than trusted."""
        for nside, share in ((16, 0.0078), (128, 0.00012)):
            nb = H.neighbours(np.arange(H.npix(nside)), nside)
            assert nb.shape == (H.npix(nside), 8)
            short = int((nb < 0).any(axis=1).sum())
            assert short == 24, f"nside={nside} had {short}"
            assert short / H.npix(nside) == pytest.approx(share, abs=5e-4)

    def test_emits_no_warning_despite_the_corner_cells(self):
        """astropy warns on the -1s; that is noise, not information, and a
        warning per target would drown a run's real output."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            H.neighbours(np.arange(H.npix(16)), 16)

    def test_adjacency_is_symmetric(self):
        pix = H.ang2pix(_LAT[:200], _LON[:200], 64)
        for cell, row in zip(pix, H.neighbours(pix, 64)):
            for other in row[row >= 0]:
                back = H.neighbours([other], 64)[0]
                assert cell in back, f"{cell} -> {other} was not mutual"


class TestRingDistance:
    def test_same_cell_is_zero(self):
        pix = H.ang2pix(_LAT[:100], _LON[:100], 128)
        assert np.all(H.ring_distance(pix, pix, 128) == 0)

    def test_every_immediate_neighbour_is_ring_one(self):
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)
        nb = H.neighbours(a, 128)[0]
        nb = nb[nb >= 0]
        assert nb.size == 8
        assert np.all(H.ring_distance(np.repeat(a, nb.size), nb, 128) == 1)

    def test_second_ring_is_ring_two(self):
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)
        first = set(int(x) for x in H.neighbours(a, 128)[0] if x >= 0) | {int(a[0])}
        second = set()
        for c in first:
            second |= {int(x) for x in H.neighbours([c], 128)[0] if x >= 0}
        second -= first
        assert second, "no second ring found"
        cells = np.fromiter(second, dtype=np.int64)
        assert np.all(H.ring_distance(np.repeat(a, cells.size), cells, 128) == 2)

    def test_beyond_max_ring_is_minus_one_not_a_big_number(self):
        """Deliberately not a distance. Once the prediction is outside the
        neighbourhood the metric asks about, how far is `error_km`'s job."""
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)
        b = H.ang2pix([SEATTLE[0]], [SEATTLE[1]], 128)
        assert H.ring_distance(a, b, 128)[0] == -1

    def test_the_arctic_regression_case(self):
        """tg-e1a1545: truth Seattle, Spotter predicted the Canadian Arctic
        2,360 km away, and v3's nearest-seed rule scored it CORRECT because the
        nearest of 18 US seeds happened to be Seattle. Ring distance is local,
        so no arrangement of far-away cells can make this adjacent."""
        truth = H.ang2pix([SEATTLE[0]], [SEATTLE[1]], 128)
        pred = H.ang2pix([ARCTIC[0]], [ARCTIC[1]], 128)
        for nside in H.NSIDE_LADDER:
            t = H.degrade(truth, 128, nside)
            p = H.degrade(pred, 128, nside)
            assert H.ring_distance(t, p, nside)[0] == -1, f"placed at nside={nside}"

    def test_max_ring_zero_only_reports_exact_matches(self):
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)
        nb = H.neighbours(a, 128)[0][:1]
        assert H.ring_distance(a, a, 128, max_ring=0)[0] == 0
        assert H.ring_distance(a, nb, 128, max_ring=0)[0] == -1

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            H.ring_distance([1, 2], [1], 128)

    def test_ring_grows_with_coarser_cells(self):
        """Two fixed points get closer in cell steps as cells grow -- the
        property that makes the ladder a tolerance dial."""
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)
        b = H.ang2pix([DENVER[0] + 0.9], [DENVER[1]], 128)
        rings = []
        for nside in H.NSIDE_LADDER:
            r = H.ring_distance(
                H.degrade(a, 128, nside), H.degrade(b, 128, nside), nside
            )[0]
            rings.append(r)
        # -1 sorts as "unplaced", so compare only the rungs that placed it.
        placed = [r for r in rings if r >= 0]
        assert placed == sorted(placed, reverse=True) or len(set(placed)) <= 1


class TestCellRings:
    def test_ring_is_contiguous_across_the_antimeridian(self):
        """A straddling cell comes back with mixed-sign longitudes; drawing that
        as one polygon smears it across the map."""
        pix = H.ang2pix([0.0], [179.99], 64)
        ring = H.cell_rings(pix, 64)[0]
        assert ring.shape[1] == 2
        assert np.ptp(ring[:, 0]) < 180.0, "ring wrapped instead of staying contiguous"

    def test_one_ring_per_cell(self):
        pix = H.ang2pix(_LAT[:5], _LON[:5], 64)
        rings = H.cell_rings(pix, 64)
        assert len(rings) == 5
        assert all(r.ndim == 2 and r.shape[1] == 2 for r in rings)
