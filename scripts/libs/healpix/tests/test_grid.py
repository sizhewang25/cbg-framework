"""Tests for the shared HEALPix primitives.

The load-bearing ones are `TestChildrenNest` and `TestDegradeEqualsRebinning`:
together they say that `pix << 2 | k` and `pix >> 2` are not merely index
arithmetic but *geometric* containment. Every caller of this module relies on
that -- the density MTL's coarse-to-fine descent for correctness, v4's accuracy
ladder for monotonicity -- and it is the property H3 cannot provide.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.libs.healpix import grid as HP

REPO_ROOT = Path(__file__).resolve().parents[4]

_RNG = np.random.default_rng(20260923)
#: Avoids the poles, where every tessellation is awkward, but keeps the full
#: longitude range so antimeridian handling is exercised.
_LAT = _RNG.uniform(-85.0, 85.0, 3000)
_LON = _RNG.uniform(-180.0, 180.0, 3000)

#: The rungs any caller actually uses, plus 2 and 4 because those are where the
#: corner cells stop being a rare edge case.
LADDER = (2, 4, 8, 16, 32, 64, 128)


class TestValidateNside:
    @pytest.mark.parametrize("n", [1, 2, 4, 16, 128, 1024])
    def test_accepts_powers_of_two(self, n):
        assert HP.validate_nside(n) == n

    @pytest.mark.parametrize("n", [0, -1, -8, 3, 5, 100, 130])
    def test_rejects_everything_else(self, n):
        with pytest.raises(ValueError, match="power of two"):
            HP.validate_nside(n)

    def test_a_float_nside_is_cast_not_rejected(self):
        """A YAML round-trip or a pandas column can hand this a `128.0`.

        The cast has to precede the bit test: `128.0 & 127.0` raises a
        `TypeError` about bitwise operands, which tells the caller nothing.
        """
        assert HP.validate_nside(128.0) == 128

    def test_a_non_integral_float_is_still_rejected(self):
        with pytest.raises(ValueError, match="power of two"):
            HP.validate_nside(12.5)


class TestClosedForms:
    @pytest.mark.parametrize("n", LADDER)
    def test_npix_is_twelve_nside_squared(self, n):
        assert HP.npix(n) == 12 * n * n

    @pytest.mark.parametrize("n", LADDER)
    def test_the_cells_tile_the_sphere_exactly(self, n):
        total = HP.pixel_area_km2(n) * HP.npix(n)
        assert total == pytest.approx(4 * np.pi * HP.EARTH_RADIUS_KM**2)

    def test_the_working_resolution_is_the_documented_size(self):
        assert HP.npix(128) == 196_608
        assert HP.pixel_area_km2(128) == pytest.approx(2594.3, abs=0.1)
        assert HP.nominal_cell_km(128) == pytest.approx(50.9, abs=0.1)

    @pytest.mark.parametrize("n", LADDER)
    def test_the_pitch_is_the_area_root(self, n):
        assert HP.nominal_cell_km(n) ** 2 == pytest.approx(HP.pixel_area_km2(n))

    def test_coarsening_one_rung_quadruples_the_area(self):
        for n in (4, 8, 16, 32, 64):
            assert HP.pixel_area_km2(n // 2) == pytest.approx(
                4.0 * HP.pixel_area_km2(n)
            )


class TestRoundTrip:
    @pytest.mark.parametrize("n", LADDER)
    def test_a_cell_centre_re_bins_to_its_own_cell(self, n):
        pix = np.arange(HP.npix(n))
        centres = HP.pix2ang(pix, n)
        np.testing.assert_array_equal(
            HP.ang2pix(centres[:, 0], centres[:, 1], n), pix
        )

    @pytest.mark.parametrize("n", LADDER)
    def test_scattered_points_land_in_range(self, n):
        pix = HP.ang2pix(_LAT, _LON, n)
        assert pix.dtype == np.int64
        assert pix.min() >= 0
        assert pix.max() < HP.npix(n)

    def test_longitude_comes_back_signed(self):
        """`healpix_to_lonlat` wraps to [0, 360); a Chicago cell must not
        arrive as 271.77, or any map centring on `lon.mean()` flips hemisphere."""
        centres = HP.pix2ang(np.arange(HP.npix(16)), 16)
        assert centres[:, 1].min() >= -180.0
        assert centres[:, 1].max() < 180.0
        assert centres[:, 1].min() < 0.0

    def test_pix2ang_is_lat_then_lon(self):
        """Pinned because the swap is silent -- haversine returns a plausible
        number either way -- and `cell_rings` uses the opposite order."""
        chicago = HP.ang2pix(41.88, -87.63, 128)
        lat, lon = HP.pix2ang(chicago, 128)[0]
        assert lat == pytest.approx(41.88, abs=1.0)
        assert lon == pytest.approx(-87.63, abs=1.0)

    def test_empty_input_gives_an_empty_frame(self):
        out = HP.pix2ang(np.array([], dtype=np.int64), 128)
        assert out.shape == (0, 2)


class TestDegradeEqualsRebinning:
    """The bit shift must equal re-projecting the coordinate. This is what
    makes the whole ladder one `ang2pix` pass plus shifts."""

    @pytest.mark.parametrize("fine", [128, 64, 32])
    def test_shifting_equals_rebinning_at_every_rung(self, fine):
        pix = HP.ang2pix(_LAT, _LON, fine)
        coarse = fine // 2
        while coarse >= 1:
            np.testing.assert_array_equal(
                HP.degrade(pix, fine, coarse),
                HP.ang2pix(_LAT, _LON, coarse),
                err_msg=f"degrade {fine}->{coarse} disagrees with re-binning",
            )
            coarse //= 2

    def test_degrading_to_the_same_nside_is_identity(self):
        pix = HP.ang2pix(_LAT, _LON, 64)
        np.testing.assert_array_equal(HP.degrade(pix, 64, 64), pix)

    def test_refusing_to_refine(self):
        with pytest.raises(ValueError, match="must not exceed"):
            HP.degrade(np.array([0]), 16, 32)


class TestChildrenNest:
    @pytest.mark.parametrize("n", [2, 8, 16, 64])
    def test_children_degrade_back_to_their_parent(self, n):
        parents = _RNG.integers(0, HP.npix(n), 500)
        kids = HP.children(parents)
        assert kids.shape == (500, 4)
        np.testing.assert_array_equal(
            HP.degrade(kids.ravel(), n * 2, n), np.repeat(parents, 4)
        )

    @pytest.mark.parametrize("n", [2, 8, 16, 64])
    def test_child_centres_re_bin_to_the_parent_cell(self, n):
        """The geometric half of the claim. Index arithmetic agreeing with
        itself is not enough -- H3's `cell_to_parent` does that too, and its
        children still straddle the parent's boundary."""
        parents = _RNG.integers(0, HP.npix(n), 500)
        kids = HP.children(parents).ravel()
        centres = HP.pix2ang(kids, n * 2)
        np.testing.assert_array_equal(
            HP.ang2pix(centres[:, 0], centres[:, 1], n), np.repeat(parents, 4)
        )

    def test_several_levels_at_once(self):
        parents = _RNG.integers(0, HP.npix(8), 50)
        kids = HP.children(parents, levels=3)
        assert kids.shape == (50, 64)
        np.testing.assert_array_equal(
            HP.degrade(kids.ravel(), 8 * 8, 8), np.repeat(parents, 64)
        )

    def test_the_children_of_a_cell_are_distinct_and_contiguous(self):
        kids = HP.children(np.array([7]))[0]
        assert sorted(kids.tolist()) == [28, 29, 30, 31]

    def test_the_whole_grid_is_covered_exactly_once(self):
        """Every cell at nside*2 is somebody's child, with no repeats."""
        kids = HP.children(np.arange(HP.npix(4))).ravel()
        np.testing.assert_array_equal(np.sort(kids), np.arange(HP.npix(8)))

    def test_levels_must_be_positive(self):
        with pytest.raises(ValueError, match="levels must be >= 1"):
            HP.children(np.array([1]), levels=0)


class TestNeighbours:
    @pytest.mark.parametrize("n", LADDER)
    def test_exactly_twenty_four_cells_have_a_missing_neighbour(self, n):
        """The base tessellation has 24 corners at every resolution. At
        nside=128 that is 0.012%; at nside=2 it is half the grid, so `-1` is
        not an edge case to skip."""
        nbrs = HP.neighbours(np.arange(HP.npix(n)), n)
        assert nbrs.shape == (HP.npix(n), 8)
        assert int((nbrs < 0).any(axis=1).sum()) == 24

    @pytest.mark.parametrize("n", [4, 16, 128])
    def test_adjacency_is_symmetric(self, n):
        seeds = _RNG.integers(0, HP.npix(n), 200)
        for s in seeds:
            for other in HP.neighbours(np.array([s]), n)[0]:
                if other < 0:
                    continue
                back = HP.neighbours(np.array([other]), n)[0]
                assert s in back, f"{s} -> {other} but not back"

    def test_empty_input(self):
        assert HP.neighbours(np.array([], dtype=np.int64), 16).shape == (0, 8)


class TestDisk:
    def test_k_zero_is_the_deduplicated_seeds(self):
        out = HP.disk(np.array([5, 5, 1]), 0, 16)
        np.testing.assert_array_equal(out, [1, 5])

    @pytest.mark.parametrize("n", [4, 16, 128])
    def test_a_single_seed_gives_nine_cells_or_eight_at_a_corner(self, n):
        sizes = {HP.disk(np.array([p]), 1, n).size for p in range(HP.npix(n))}
        assert sizes <= {8, 9}
        counts = [HP.disk(np.array([p]), 1, n).size for p in range(HP.npix(n))]
        assert counts.count(8) == 24

    def test_overlapping_disks_collapse(self):
        """Two adjacent seeds give 12, not 2x9 -- so callers expanding a top-k
        set need no deduplication of their own."""
        adj = HP.neighbours(np.array([100]), 16)[0]
        adj = adj[adj >= 0]
        pair = np.array([100, int(adj[0])])
        assert HP.disk(pair, 1, 16).size == 12

    def test_the_result_is_sorted_and_unique(self):
        out = HP.disk(_RNG.integers(0, HP.npix(32), 20), 2, 32)
        np.testing.assert_array_equal(out, np.unique(out))

    def test_growing_k_never_shrinks_the_disk(self):
        seeds = _RNG.integers(0, HP.npix(32), 5)
        prev = HP.disk(seeds, 0, 32)
        for k in range(1, 4):
            cur = HP.disk(seeds, k, 32)
            assert np.isin(prev, cur).all()
            assert cur.size >= prev.size
            prev = cur

    def test_the_seeds_are_always_included(self):
        seeds = _RNG.integers(0, HP.npix(64), 10)
        assert np.isin(seeds, HP.disk(seeds, 2, 64)).all()

    def test_a_disk_is_bounded_by_the_grid(self):
        """At nside=1 a ring-1 disk is 7 of 12 cells. Saturation must return
        the grid, not loop forever or emit duplicates."""
        out = HP.disk(np.array([0]), 12, 1)
        np.testing.assert_array_equal(out, np.arange(12))

    def test_negative_k_is_refused(self):
        with pytest.raises(ValueError, match="k must be >= 0"):
            HP.disk(np.array([0]), -1, 16)

    def test_every_disk_member_is_within_k_rings(self):
        """Cross-checks `disk` against `neighbours` directly, so a bug in the
        breadth-first bookkeeping cannot hide behind itself."""
        seed = 4242
        ring1 = HP.disk(np.array([seed]), 1, 64)
        expected = HP.neighbours(np.array([seed]), 64)[0]
        expected = np.unique(np.append(expected[expected >= 0], seed))
        np.testing.assert_array_equal(ring1, expected)


class TestCellRings:
    def test_one_ring_per_cell_with_four_edges(self):
        rings = HP.cell_rings(np.array([0, 1, 2]), 16, step=8)
        assert len(rings) == 3
        for r in rings:
            assert r.shape == (32, 2)

    def test_cell_rings_is_lon_then_lat(self):
        """The opposite order from `pix2ang`, which is why both are pinned."""
        pix = HP.ang2pix(41.88, -87.63, 128)
        ring = HP.cell_rings(pix, 128)[0]
        assert ring[:, 0].mean() == pytest.approx(-87.63, abs=1.0)
        assert ring[:, 1].mean() == pytest.approx(41.88, abs=1.0)

    def test_an_antimeridian_cell_stays_contiguous(self):
        """Mixed-sign longitudes would smear the polygon across the map."""
        pix = HP.ang2pix(np.array([0.0]), np.array([179.9]), 64)
        ring = HP.cell_rings(pix, 64)[0]
        assert ring[:, 0].max() - ring[:, 0].min() < 90.0

    def test_the_ring_encloses_its_own_centre(self):
        pix = HP.ang2pix(39.74, -104.99, 64)
        ring = HP.cell_rings(pix, 64)[0]
        lat, lon = HP.pix2ang(pix, 64)[0]
        assert ring[:, 0].min() <= lon <= ring[:, 0].max()
        assert ring[:, 1].min() <= lat <= ring[:, 1].max()

    def test_empty_input(self):
        assert HP.cell_rings(np.array([], dtype=np.int64), 16) == []


class TestDescribe:
    def test_it_reports_the_grid_not_a_guess(self):
        d = HP.describe(128)
        assert d["scheme"] == "healpix"
        assert d["nside"] == 128
        assert d["order"] == "nested"
        assert d["n_cells"] == 196_608
        assert d["cell_area_km2"] == pytest.approx(2594.3, abs=0.1)


class TestStandsAlone:
    """`scripts.libs.healpix` is imported by both the framework and the
    analysis layers, so it may depend on neither -- the same rule
    `scripts/analysis/v3/tests/test_layering.py` states for the CSV contract."""

    def test_it_pulls_no_consumer_package(self):
        code = (
            "import scripts.libs.healpix.grid, sys, json;"
            "print(json.dumps(sorted(m for m in sys.modules if"
            " m.startswith('scripts.framework') or m.startswith('scripts.benchmark')"
            " or m.startswith('scripts.analysis'))))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        import json

        leaked = json.loads(out.stdout.strip().splitlines()[-1])
        assert leaked == [], (
            f"scripts.libs.healpix pulled {leaked}; it is shared by the framework "
            f"and the analysis layers, so it may depend on neither"
        )

    def test_it_does_not_pull_scipy_for_one_constant(self):
        """`EARTH_RADIUS_KM` is defined locally rather than imported from
        `scripts.libs.cbg.rtt_model`, which drags in `scipy.optimize.linprog`
        -- 59 MB and 0.34 s for a single float, more than the astropy import
        this module keeps lazy."""
        code = (
            "import scripts.libs.healpix.grid, sys, json;"
            "print(json.dumps('scipy' in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        import json

        assert json.loads(out.stdout.strip().splitlines()[-1]) is False

    def test_the_closed_forms_do_not_need_astropy(self):
        """Areas, the ladder and `children` are all arithmetic, so a caller
        that wants only those should not pay 26 MB for astropy."""
        code = (
            "from scripts.libs.healpix import grid as HP;"
            "HP.npix(128); HP.pixel_area_km2(128); HP.nominal_cell_km(128);"
            "HP.children(__import__('numpy').array([1])); HP.describe(128);"
            "import sys, json; print(json.dumps('astropy_healpix' in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        import json

        assert json.loads(out.stdout.strip().splitlines()[-1]) is False


class TestAgreesWithV3:
    """v3 keeps its own copy of these primitives, and it is v3's
    `nominal_cell_km(128)` that produced the 50.9 km figure quoted in v4's
    README. v3 is deliberately not being refactored, so pin the agreement --
    an undetected divergence would silently desync two published numbers."""

    @pytest.mark.parametrize("n", (16, 32, 64, 128))
    def test_the_scalar_forms_match(self, n):
        from scripts.analysis.v3.modules import healpix as V3

        assert V3.npix(n) == HP.npix(n)
        assert V3.pixel_area_km2(n) == pytest.approx(HP.pixel_area_km2(n))
        assert V3.nominal_cell_km(n) == pytest.approx(HP.nominal_cell_km(n))

    @pytest.mark.parametrize("n", (16, 128))
    def test_the_quantisation_matches(self, n):
        from scripts.analysis.v3.modules import healpix as V3

        np.testing.assert_array_equal(
            V3.ang2pix(_LAT, _LON, n), HP.ang2pix(_LAT, _LON, n)
        )

    def test_degrade_matches(self):
        from scripts.analysis.v3.modules import healpix as V3

        pix = HP.ang2pix(_LAT, _LON, 128)
        for n in (64, 32, 16):
            np.testing.assert_array_equal(
                V3.degrade(pix, 128, n), HP.degrade(pix, 128, n)
            )
