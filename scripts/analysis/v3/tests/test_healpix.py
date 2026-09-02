"""HEALPix quantizer invariants (paper §7.3)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.v3.modules import healpix as hx


def test_paper_grid_constants():
    """nside=128 must reproduce the figures §7.3 quotes."""
    assert hx.npix(128) == 196_608
    assert hx.pixel_area_km2(128) == pytest.approx(2594, abs=1)
    assert hx.nominal_cell_km(128) == pytest.approx(51, abs=1)


@pytest.mark.parametrize("nside,km", [(128, 51), (64, 102), (32, 204), (16, 407)])
def test_hierarchy_pitches(nside, km):
    assert hx.nominal_cell_km(nside) == pytest.approx(km, abs=1)


def test_nside_must_be_power_of_two():
    """Nesting is a bit shift, so a non-power-of-two would silently misalign."""
    with pytest.raises(ValueError):
        hx.validate_nside(100)
    with pytest.raises(ValueError):
        hx.validate_nside(0)


def test_degrade_matches_direct_ang2pix():
    """`pix >> 2k` must equal quantizing at the coarser nside directly.

    This is the property that makes the multi-scale curve a single pass.
    """
    rng = np.random.default_rng(0)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 500)))
    lon = rng.uniform(-180, 180, 500)
    fine = hx.ang2pix(lat, lon, 128)
    for coarse in (64, 32, 16):
        assert np.array_equal(
            hx.degrade(fine, 128, coarse), hx.ang2pix(lat, lon, coarse)
        )


def test_degrade_rejects_refinement():
    with pytest.raises(ValueError):
        hx.degrade(np.array([0]), 16, 128)


def test_occupied_cells_is_monotone_non_increasing():
    """Coarsening can only merge cells, never split them."""
    rng = np.random.default_rng(1)
    lat = rng.uniform(25, 50, 300)
    lon = rng.uniform(-125, -70, 300)
    counts = hx.occupied_cell_hierarchy(lat, lon)
    ordered = [counts[n] for n in sorted(counts, reverse=True)]
    assert all(a >= b for a, b in zip(ordered, ordered[1:]))


def test_spherical_centroid_handles_dateline():
    """Averaging lon directly would land at 0°; the unit-vector mean must not."""
    lat, lon = hx.spherical_centroid([0.0, 0.0], [179.0, -179.0])
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert abs(lon) == pytest.approx(180.0, abs=1e-9)


def test_spherical_centroid_of_identical_points_is_that_point():
    lat, lon = hx.spherical_centroid([41.9742] * 3, [-87.9073] * 3)
    assert lat == pytest.approx(41.9742)
    assert lon == pytest.approx(-87.9073)
