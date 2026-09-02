"""The `Grid` contract, held identically against both tessellations.

Everything parametrized over `GRID_NAMES` is a rule the answer space relies on
regardless of which grid is in use — determinism, monotone coarsening, ragged
but well-formed rings, id survival through CSV. The grid-specific tests at the
bottom cover the two places the grids genuinely differ: H3's resolution band and
its non-nesting, and HEALPix's exact nesting (which lives in `test_healpix.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_NAMES,
    Grid,
    get_grid,
    resolutions_for,
    ring_lonlat,
)

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
US = [CHI, SJC, NYC, (32.7767, -96.7970), (38.9072, -77.0369)]


@pytest.fixture(params=GRID_NAMES)
def grid(request) -> Grid:
    return get_grid(request.param)


# ---- registry ---------------------------------------------------------------


def test_h3_is_the_default_grid():
    assert DEFAULT_GRID == "h3"
    assert get_grid(DEFAULT_GRID).name == "h3"


def test_registry_resolves_both_names(grid):
    assert grid.name in GRID_NAMES
    assert get_grid(grid.name).name == grid.name


def test_unknown_grid_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown grid"):
        get_grid("s2")


def test_every_grid_declares_its_own_defaults(grid):
    """A resolution is only meaningful paired with its grid.

    `4` means ~45 km on H3 and ~10,000 km on HEALPix, so nothing may carry a
    resolution without the grid that produced it.
    """
    assert grid.DEFAULT_RESOLUTION in grid.HIERARCHY
    assert grid.resolution_arg
    assert grid.validate_resolution(grid.DEFAULT_RESOLUTION) == grid.DEFAULT_RESOLUTION


# ---- resolution -------------------------------------------------------------


def test_hierarchy_is_ordered_finest_first(grid):
    assert list(grid.HIERARCHY) == sorted(grid.HIERARCHY, reverse=True)


def test_coarsening_ladder_never_includes_a_finer_rung(grid):
    """The regression this pins: reporting occupancy for grids never built."""
    for res in grid.HIERARCHY:
        ladder = grid.coarsening_ladder(res)
        assert ladder[0] == res
        assert all(x <= res for x in ladder)
        assert list(ladder) == sorted(ladder, reverse=True)


def test_coarsest_rung_has_a_single_step_ladder(grid):
    assert grid.coarsening_ladder(grid.HIERARCHY[-1]) == (grid.HIERARCHY[-1],)


def test_resolutions_for_defaults_to_the_grids_own(grid):
    assert resolutions_for(grid, [], sweep=False) == [grid.DEFAULT_RESOLUTION]
    assert resolutions_for(grid, None, sweep=False) == [grid.DEFAULT_RESOLUTION]


def test_resolutions_for_sweep_is_the_whole_hierarchy(grid):
    assert resolutions_for(grid, [], sweep=True) == list(grid.HIERARCHY)


def test_resolutions_for_dedupes_but_keeps_order(grid):
    a, b = grid.HIERARCHY[0], grid.HIERARCHY[1]
    assert resolutions_for(grid, [a, b, a], sweep=False) == [a, b]


# ---- quantization -----------------------------------------------------------


def test_cell_ids_are_deterministic(grid):
    lats = [c[0] for c in US]
    lons = [c[1] for c in US]
    r = grid.DEFAULT_RESOLUTION
    assert list(grid.cell_ids(lats, lons, r)) == list(grid.cell_ids(lats, lons, r))


def test_cell_ids_do_not_depend_on_row_order(grid):
    """seed_id is the rank of a sorted cell id, so ids must be order-free."""
    lats = [c[0] for c in US]
    lons = [c[1] for c in US]
    r = grid.DEFAULT_RESOLUTION
    fwd = grid.cell_ids(lats, lons, r)
    rev = grid.cell_ids(lats[::-1], lons[::-1], r)
    assert list(fwd) == list(rev)[::-1]


def test_colocated_points_share_a_cell(grid):
    ids = grid.cell_ids([CHI[0]] * 3, [CHI[1]] * 3, grid.DEFAULT_RESOLUTION)
    assert len(set(ids)) == 1


def test_distant_points_do_not_share_a_cell(grid):
    ids = grid.cell_ids(
        [CHI[0], SJC[0], NYC[0]], [CHI[1], SJC[1], NYC[1]], grid.DEFAULT_RESOLUTION
    )
    assert len(set(ids)) == 3


def test_cell_ids_survive_a_csv_roundtrip(tmp_path, grid):
    """CSV is untyped; `coerce_cell_ids` is the only thing that knows the dtype.

    HEALPix ids are int64 and H3's are hex strings, so pandas' per-column
    inference gets one of the two wrong without this.
    """
    ids = grid.cell_ids([c[0] for c in US], [c[1] for c in US], grid.DEFAULT_RESOLUTION)
    path = tmp_path / "cells.csv"
    pd.DataFrame({"cell_id": grid.coerce_cell_ids(ids)}).to_csv(path, index=False)
    back = grid.coerce_cell_ids(pd.read_csv(path)["cell_id"])
    assert list(back) == list(grid.coerce_cell_ids(ids))


# ---- scale ------------------------------------------------------------------


def test_coarser_resolutions_have_bigger_cells(grid):
    areas = [grid.cell_area_km2(r) for r in grid.HIERARCHY]
    pitches = [grid.nominal_cell_km(r) for r in grid.HIERARCHY]
    # HIERARCHY is finest-first, so both must increase along it.
    assert areas == sorted(areas)
    assert pitches == sorted(pitches)


def test_coarser_resolutions_have_fewer_cells(grid):
    counts = [grid.n_cells(r) for r in grid.HIERARCHY]
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize(
    "grid_name,resolution,pitch_km",
    [
        ("h3", 5, 17.1),
        ("h3", 4, 45.2),
        ("h3", 3, 119.5),
        ("h3", 2, 316.1),
        ("healpix", 128, 50.9),
        ("healpix", 64, 101.9),
        ("healpix", 32, 203.7),
        ("healpix", 16, 407.5),
    ],
)
def test_documented_pitches(grid_name, resolution, pitch_km):
    """The scale table in the README and plan is load-bearing, so pin it.

    Note the two grids define pitch differently on purpose: HEALPix uses
    sqrt(area) because its cells are exactly equal-area, H3 uses
    centre-to-centre because a hexagon's sqrt(area) understates its spacing.
    """
    assert get_grid(grid_name).nominal_cell_km(resolution) == pytest.approx(
        pitch_km, abs=0.15
    )


def test_h3_res4_is_the_nearest_rung_to_healpix_nside128():
    """Why res 4 is the default: swapping grids must not move the merge scale."""
    healpix = get_grid("healpix").nominal_cell_km(128)
    h3 = get_grid("h3")
    gaps = {r: abs(h3.nominal_cell_km(r) - healpix) for r in h3.HIERARCHY}
    assert min(gaps, key=gaps.get) == h3.DEFAULT_RESOLUTION


# ---- occupancy hierarchy ----------------------------------------------------


def test_occupied_cells_is_monotone_non_increasing(grid):
    rng = np.random.default_rng(7)
    lats = rng.uniform(25, 49, 300)
    lons = rng.uniform(-124, -67, 300)
    counts = list(
        grid.occupied_cell_hierarchy(lats, lons, grid.HIERARCHY).values()
    )
    assert counts == sorted(counts, reverse=True)


def test_occupied_cell_hierarchy_keys_match_the_request(grid):
    ladder = grid.coarsening_ladder(grid.DEFAULT_RESOLUTION)
    got = grid.occupied_cell_hierarchy(
        [c[0] for c in US], [c[1] for c in US], ladder
    )
    assert tuple(got) == tuple(int(r) for r in ladder)


# ---- geometry ---------------------------------------------------------------


def test_cell_boundaries_returns_one_ring_per_cell(grid):
    r = grid.DEFAULT_RESOLUTION
    ids = grid.cell_ids([c[0] for c in US], [c[1] for c in US], r)
    rings = grid.cell_boundaries(ids, r)
    assert len(rings) == len(ids)


def test_cell_boundaries_rings_are_well_formed(grid):
    r = grid.DEFAULT_RESOLUTION
    ids = grid.cell_ids([c[0] for c in US], [c[1] for c in US], r)
    for ring in grid.cell_boundaries(ids, r):
        assert ring.ndim == 2 and ring.shape[1] == 2  # (V, 2) as (lon, lat)
        assert ring.shape[0] >= 4
        assert np.isfinite(ring).all()
        lat = ring[:, 1]
        assert (lat >= -90.0).all() and (lat <= 90.0).all()


def test_cell_boundaries_ring_encloses_its_own_cell_centre(grid):
    """Catches a lat/lon swap, which is otherwise a silently plausible map.

    H3's `cell_to_boundary` returns (lat, lng); every plotting call wants
    (lon, lat). Transposed rings still render, just in the wrong hemisphere.
    """
    r = grid.DEFAULT_RESOLUTION
    ids = grid.cell_ids([CHI[0]], [CHI[1]], r)
    ring = grid.cell_boundaries(ids, r)[0]
    lon, lat = ring[:, 0], ring[:, 1]
    assert lon.min() <= CHI[1] <= lon.max()
    assert lat.min() <= CHI[0] <= lat.max()


def test_cell_boundaries_of_nothing_is_nothing(grid):
    assert grid.cell_boundaries([], grid.DEFAULT_RESOLUTION) == []


def test_ring_lonlat_keeps_a_dateline_ring_contiguous():
    """A ring split across +/-180 would otherwise smear across the whole map."""
    ring = ring_lonlat([179.9, 179.95, -179.95, -179.9], [0.0, 0.1, 0.1, 0.0])
    lon = ring[:, 0]
    assert lon.max() - lon.min() < 1.0


def test_dateline_cells_do_not_produce_a_giant_ring(grid):
    r = grid.DEFAULT_RESOLUTION
    ids = grid.cell_ids([0.0], [179.98], r)
    ring = grid.cell_boundaries(ids, r)[0]
    span = ring[:, 0].max() - ring[:, 0].min()
    assert span < 10.0, f"ring spans {span:.1f}° of longitude"


# ---- self description -------------------------------------------------------


def test_describe_names_its_own_scheme_and_resolution(grid):
    meta = grid.describe(grid.DEFAULT_RESOLUTION)
    assert meta["scheme"] == grid.name
    assert meta["resolution"] == grid.DEFAULT_RESOLUTION
    assert meta["n_cells"] > 0
    assert meta["cell_area_km2"] > 0
    assert meta["nominal_cell_km"] > 0


def test_describe_states_how_pitch_was_derived(grid):
    """The two grids compute pitch differently; the artifact has to say which."""
    assert "nominal_cell_km_note" in grid.describe(grid.DEFAULT_RESOLUTION)


# ---- grid-specific ----------------------------------------------------------


@pytest.mark.parametrize("bad", [1, 6, 0, 15, -1])
def test_h3_rejects_resolutions_outside_the_supported_band(bad):
    with pytest.raises(ValueError, match="h3 resolution must be one of"):
        get_grid("h3").validate_resolution(bad)


def test_h3_supports_two_through_five():
    g = get_grid("h3")
    assert [g.validate_resolution(r) for r in (2, 3, 4, 5)] == [2, 3, 4, 5]


def test_h3_reports_that_its_cells_are_not_equal_area():
    """HEALPix's equal-area guarantee is exact; H3's is not, so measure it."""
    g = get_grid("h3")
    r = g.DEFAULT_RESOLUTION
    lats = [c[0] for c in US]
    lons = [c[1] for c in US]
    diag = g.occupancy_diagnostics(g.cell_ids(lats, lons, r), lats, lons, r)
    assert diag["cell_area_km2_max"] >= diag["cell_area_km2_min"]
    assert diag["cell_area_km2_max_over_min"] >= 1.0
    assert diag["n_occupied_pentagons"] == 0  # pentagons sit over ocean


def test_h3_measures_its_own_non_nesting():
    """The aperture-7 caveat as a number, not an argument.

    `cell_to_parent` is exact on the index but is not a geometric container, so
    lineage and re-binning can disagree. The count may legitimately be 0 on a
    small target set — what matters is that it is reported per coarser rung.
    """
    g = get_grid("h3")
    rng = np.random.default_rng(3)
    lats = rng.uniform(25, 49, 400)
    lons = rng.uniform(-124, -67, 400)
    diag = g.occupancy_diagnostics(g.cell_ids(lats, lons, 5), lats, lons, 5)
    lineage = diag["parent_lineage_disagreements"]
    assert set(lineage) == {"4", "3", "2"}
    assert all(v >= 0 for v in lineage.values())


def test_healpix_has_nothing_to_disclose():
    """Exactly equal-area and exactly nested, so the diagnostic is empty."""
    g = get_grid("healpix")
    lats = [c[0] for c in US]
    lons = [c[1] for c in US]
    assert g.occupancy_diagnostics(g.cell_ids(lats, lons, 128), lats, lons, 128) == {}


def test_healpix_records_its_ordering():
    """RING ids would not coarsen by bit shift, so the ordering must be stated."""
    assert get_grid("healpix").describe(128)["order"] == "nested"


# ---- CLI front door ---------------------------------------------------------


def test_cli_front_door_returns_grid_and_validated_resolutions(grid):
    from scripts.analysis.v3.modules.grid import resolve_cli_grid

    g, res = resolve_cli_grid(grid.name, [], sweep=False)
    assert g.name == grid.name
    assert res == [grid.DEFAULT_RESOLUTION]


def test_cli_front_door_turns_a_bad_grid_into_a_cli_error():
    """A traceback is the wrong answer to a typo in --grid."""
    typer = pytest.importorskip("typer")
    from scripts.analysis.v3.modules.grid import resolve_cli_grid

    with pytest.raises(typer.BadParameter, match="--grid must be one of"):
        resolve_cli_grid("s2", [], sweep=False)


def test_cli_front_door_turns_a_bad_resolution_into_a_cli_error():
    typer = pytest.importorskip("typer")
    from scripts.analysis.v3.modules.grid import resolve_cli_grid

    with pytest.raises(typer.BadParameter, match="h3 resolution must be one of"):
        resolve_cli_grid("h3", [9], sweep=False)
    with pytest.raises(typer.BadParameter, match="power of two"):
        resolve_cli_grid("healpix", [100], sweep=False)
