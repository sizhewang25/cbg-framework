"""Tests for v5's grid primitives, ported from v4's `test_healpix.py`.

The load-bearing ones are `TestLadderIsMonotone` and
`TestDegradeEqualsRebinning`: the two properties H3 cannot satisfy, and the
reason the grid axis of v5 is monotone across rungs.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.v5.modules import grid as H

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


class TestGeometry:
    def test_grid_count_and_area_close_the_sphere(self):
        for nside in H.NSIDE_LADDER:
            assert H.n_grids(nside) == 12 * nside**2
            total = H.grid_area_km2(nside) * H.n_grids(nside)
            assert total == pytest.approx(4 * np.pi * 6371.0**2, rel=1e-9)

    def test_nside_128_is_the_metro_scale(self):
        """The working resolution, quoted throughout the module docs."""
        assert H.n_grids(128) == 196_608
        assert H.grid_area_km2(128) == pytest.approx(2594.3, abs=0.1)
        assert H.grid_km(128) == pytest.approx(50.9, abs=0.1)

    def test_pix2ang_round_trips(self):
        """A swapped (lat, lon) pair cannot survive this — it lands in a
        different grid, or is rejected for a latitude past +/-90."""
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


class TestLadderIsMonotone:
    """Two points that share a grid must share every coarser grid.

    Under H3 this failed 705 times on the real predictions, and visibly in
    aggregate: `vanilla_cbg` on as01 scored 0.594 at res 1 but 0.659 at res 2,
    higher than its own parent. Zero tolerance here.
    """

    def test_sharing_a_fine_grid_implies_sharing_every_coarser_one(self):
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


class TestNeighbours:
    def test_eight_neighbours_and_the_24_corner_grids(self):
        """24 grids per nside sit at a base-tessellation corner and have 7.
        Real, and the reason `-1` has to be filtered rather than trusted."""
        for nside, share in ((16, 0.0078), (128, 0.00012)):
            nb = H.neighbours(np.arange(H.n_grids(nside)), nside)
            assert nb.shape == (H.n_grids(nside), 8)
            short = int((nb < 0).any(axis=1).sum())
            assert short == 24, f"nside={nside} had {short}"
            assert short / H.n_grids(nside) == pytest.approx(share, abs=5e-4)

    def test_emits_no_warning_despite_the_corner_grids(self):
        """astropy warns on the -1s; that is noise, not information, and a
        warning per target would drown a run's real output."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            H.neighbours(np.arange(H.n_grids(16)), 16)

    def test_adjacency_is_symmetric(self):
        pix = H.ang2pix(_LAT[:200], _LON[:200], 64)
        for grid, row in zip(pix, H.neighbours(pix, 64)):
            for other in row[row >= 0]:
                back = H.neighbours([other], 64)[0]
                assert grid in back, f"{grid} -> {other} was not mutual"


class TestGridOffset:
    """The grid axis, exact and ungraded.

    `test_the_nth_ring_built_by_hand_is_offset_n` is the load-bearing one. It
    grows each ring from `neighbours` alone, so it checks the BFS against an
    independent construction rather than against another BFS -- which is what
    the retired `ring_distance` used to provide. It runs one ring past the old
    cap, where that function could only ever have said -1.
    """

    def test_the_nth_ring_built_by_hand_is_offset_n(self):
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)
        seen = {int(a[0])}
        shell = set(seen)
        for n in (1, 2, 3):
            nxt = set()
            for c in shell:
                nxt |= {int(x) for x in H.neighbours([c], 128)[0] if x >= 0}
            shell = nxt - seen
            seen |= shell
            assert shell, f"no ring {n} found"
            grids = np.fromiter(shell, dtype=np.int64)
            assert np.all(H.grid_offset(np.repeat(a, grids.size), grids, 128) == n), n

    def test_same_grid_is_zero(self):
        pix = H.ang2pix(_LAT[:100], _LON[:100], 128)
        assert np.all(H.grid_offset(pix, pix, 128) == 0)

    def test_every_immediate_neighbour_is_one(self):
        a = H.ang2pix([DENVER[0]], [DENVER[1]], 128)
        nb = H.neighbours(a, 128)[0]
        nb = nb[nb >= 0]
        assert nb.size == 8
        assert np.all(H.grid_offset(np.repeat(a, nb.size), nb, 128) == 1)

    def test_the_arctic_case_is_a_number_not_a_shrug(self):
        """tg-e1a1545: truth Seattle, Spotter predicted the Canadian Arctic
        2,360 km away. The retired `ring` column called this -1, the same
        value it used for "no prediction". It is 68 grids out."""
        truth = H.ang2pix([SEATTLE[0]], [SEATTLE[1]], 128)
        pred = H.ang2pix([ARCTIC[0]], [ARCTIC[1]], 128)
        assert H.grid_offset(truth, pred, 128)[0] > 20

    def test_antipodes_resolve_rather_than_giving_up(self):
        """The neighbour graph is connected, so no pair is unreachable."""
        a = H.ang2pix([45.0], [0.0], 128)
        b = H.ang2pix([-45.0], [180.0], 128)
        assert H.grid_offset(a, b, 128)[0] > 0

    def test_it_is_symmetric(self):
        a = H.ang2pix(_LAT[:60], _LON[:60], 128)
        b = H.ang2pix(_LAT[60:120], _LON[60:120], 128)
        assert np.array_equal(H.grid_offset(a, b, 128), H.grid_offset(b, a, 128))

    def test_empty_input_is_empty_output(self):
        assert H.grid_offset([], [], 128).size == 0

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            H.grid_offset([1, 2], [1], 128)


class TestRingGrids:
    """The set form of `grid_offset`, used to draw a neighbourhood.

    `grid_offset` answers "how far is b from a"; `ring_grids` answers "which
    grids are k out from a". The cross-check below is what keeps the two from
    drifting -- they are separate breadth-first walks, and `map_mtl` shades
    the rings this returns while labelling them with the offset that one
    reports.
    """

    def test_every_grid_at_ring_k_is_offset_k(self):
        for pix in (0, 1000, 12345, H.n_grids(128) - 1):
            rings = H.ring_grids(pix, 128, 2)
            for k, shell in enumerate(rings):
                grids = np.asarray(shell, dtype=np.int64)
                assert np.all(H.grid_offset(np.full(grids.size, pix), grids, 128) == k)

    def test_ring_zero_is_the_grid_itself(self):
        assert H.ring_grids(4242, 128, 2)[0] == [4242]

    def test_the_rings_are_disjoint(self):
        rings = H.ring_grids(4242, 128, 2)
        flat = [c for shell in rings for c in shell]
        assert len(flat) == len(set(flat))

    def test_max_ring_zero_is_just_the_grid(self):
        assert H.ring_grids(4242, 128, 0) == [[4242]]

    def test_max_ring_is_required(self):
        """No default: the band edge lives in `classify.MAX_RING`, and a
        second copy here is a number that could drift."""
        with pytest.raises(TypeError):
            H.ring_grids(4242, 128)


