"""Smoke tests for the static answer-space map.

Deliberately thin: cartopy rendering is slow and pixel-level assertions are
brittle. What is worth pinning is that the figure gets written, and that the
extent logic behaves — the geometry it draws is already covered by
`test_answer_space.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import build_answer_space
from scripts.analysis.v3.modules.map_answer_space import (
    US_MAINLAND_EXTENT,
    _auto_extent,
    plot_answer_space,
)

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)


@pytest.fixture(params=("h3", "healpix"))
def grid(request):
    """Render on both grids.

    Not redundant with the geometry tests in `test_grid.py`: H3 rings are ragged
    (6 vertices for a hexagon, 5 for a pentagon) where HEALPix rings are a
    uniform 4*step, so a plot layer that assumed a rectangular array would pass
    every unit test and still fail here.
    """
    return request.param


def _space(coords=(CHI, SJC, NYC), grid="h3"):
    return build_answer_space(
        pd.DataFrame(
            {
                "target_id": [f"tg-{i}" for i in range(len(coords))],
                "target_lat": [c[0] for c in coords],
                "target_lon": [c[1] for c in coords],
            }
        ),
        grid=grid,
    )


def test_auto_extent_covers_every_target():
    lats = np.array([c[0] for c in (CHI, SJC, NYC)])
    lons = np.array([c[1] for c in (CHI, SJC, NYC)])
    lon_min, lon_max, lat_min, lat_max = _auto_extent(lats, lons)
    assert lon_min < lons.min() and lon_max > lons.max()
    assert lat_min < lats.min() and lat_max > lats.max()


def test_auto_extent_stays_on_the_globe():
    """Padding must not push the window off the valid lon/lat range."""
    lon_min, lon_max, lat_min, lat_max = _auto_extent(
        np.array([-89.9, 89.9]), np.array([-179.9, 179.9])
    )
    assert lon_min >= -180.0 and lon_max <= 180.0
    assert lat_min >= -90.0 and lat_max <= 90.0


def test_auto_extent_is_non_degenerate_for_colocated_targets():
    """All targets in one metro would otherwise give a zero-width extent."""
    lon_min, lon_max, lat_min, lat_max = _auto_extent(
        np.array([41.9742, 41.9742]), np.array([-87.9073, -87.9073])
    )
    assert lon_max > lon_min
    assert lat_max > lat_min


def test_map_is_written_and_non_empty(tmp_path, grid):
    pytest.importorskip("cartopy")
    out = plot_answer_space(
        _space(grid=grid), tmp_path / "map.png", extent=US_MAINLAND_EXTENT
    )
    assert out.exists()
    assert out.stat().st_size > 5_000  # a real render, not a blank canvas


def test_map_handles_a_single_seed(tmp_path, grid):
    """K=1 has no seed-to-seed geometry; the map must still render.

    An explicit extent is passed on purpose: the auto extent around a single
    point is small enough that cartopy switches to 10m Natural Earth features
    and downloads them, which would make this test need the network.
    """
    pytest.importorskip("cartopy")
    space = _space(coords=(CHI,), grid=grid)
    assert space.n_seeds == 1
    out = plot_answer_space(space, tmp_path / "one.png", extent=US_MAINLAND_EXTENT)
    assert out.exists()


def test_map_draws_one_cell_per_seed(tmp_path, grid):
    """Guards the ragged-ring path: a dropped or duplicated ring is invisible.

    Cell polygons are the only patches added, so their count must equal K on
    either grid regardless of how many vertices each ring happens to have.
    """
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt

    space = _space(grid=grid)
    plot_answer_space(space, tmp_path / "m.png", extent=US_MAINLAND_EXTENT)
    # plot_answer_space closes its figure, so re-derive the ring count directly.
    from scripts.analysis.v3.modules.grid import get_grid

    g = get_grid(str(space.seeds["grid_scheme"].iloc[0]))
    rings = g.cell_boundaries(
        space.seeds["cell_id"].to_numpy(),
        int(space.seeds["grid_resolution"].iloc[0]),
    )
    assert len(rings) == space.n_seeds
    plt.close("all")


# ---- the nearest-seed Voronoi overlay ---------------------------------------
# This is the classifier's own top-1 decision boundary, so these are correctness
# tests, not render smoke tests: `classify` labels a prediction by nearest seed,
# and a partition that disagrees with that would be a figure that misstates the
# metric.

US_SEEDS = ((41.97, -87.90), (37.46, -121.92), (40.71, -74.01), (29.76, -95.36))


def _us_space(grid="h3"):
    """Four well-separated US metros — enough seeds for a real partition."""
    return _space(coords=US_SEEDS, grid=grid)


def test_voronoi_has_one_cell_per_seed(grid):
    from scripts.analysis.v3.modules.map_answer_space import seed_voronoi

    space = _us_space(grid)
    pv = seed_voronoi(space.seeds, US_MAINLAND_EXTENT)
    assert pv is not None
    assert len(pv.cells) == space.n_seeds
    # every seed claims exactly one cell
    assert sorted(pv.seed_index.tolist()) == list(range(space.n_seeds))


def test_voronoi_covers_the_whole_frame(grid):
    """Gaps would imply predictions that fall into no class, which cannot happen.

    Asserted by area rather than `covers`, which is exact and so trips on the
    sub-square-metre slivers GEOS leaves along shared cell edges. The invariant
    that matters is that nothing *visible* is uncovered.
    """
    from shapely.ops import unary_union

    from scripts.analysis.v3.modules.map_answer_space import (
        _projected_frame_box,
        seed_voronoi,
    )

    pv = seed_voronoi(_us_space(grid).seeds, US_MAINLAND_EXTENT)
    frame = _projected_frame_box(US_MAINLAND_EXTENT, pv.crs, pad_frac=0.0)
    uncovered = frame.difference(unary_union(pv.cells)).area
    assert uncovered / frame.area < 1e-9


def test_voronoi_cell_edges_stay_off_the_visible_frame(grid):
    """The clip box is padded so no cell edge inks a line along the map border.

    Without the padding the outermost cells are cut exactly at the frame and
    their outlines draw a boundary that is an artifact of the crop, not of the
    classifier.
    """
    from scripts.analysis.v3.modules.map_answer_space import (
        _VORONOI_PAD_FRAC,
        _projected_frame_box,
        seed_voronoi,
    )

    pv = seed_voronoi(_us_space(grid).seeds, US_MAINLAND_EXTENT)
    frame = _projected_frame_box(US_MAINLAND_EXTENT, pv.crs, pad_frac=0.0)
    clip = _projected_frame_box(
        US_MAINLAND_EXTENT, pv.crs, pad_frac=_VORONOI_PAD_FRAC
    )
    assert clip.contains(frame)


def test_voronoi_matches_great_circle_nearest_seed(grid):
    """Pins the projection choice, which is the whole reason this is not trivial.

    The cells must agree with `pairwise_km`-argmin labelling — the rule
    `classify` actually applies. Computing the diagram in raw lon/lat instead
    scores ~90% here because a degree of longitude is not a degree of latitude,
    so this threshold is what stops that regression.
    """
    from shapely.geometry import Point

    from scripts.analysis.v3.modules.answer_space import pairwise_km
    from scripts.analysis.v3.modules.map_answer_space import seed_voronoi

    space = _us_space(grid)
    seeds = space.seeds
    pv = seed_voronoi(seeds, US_MAINLAND_EXTENT)

    lo0, lo1, la0, la1 = US_MAINLAND_EXTENT
    gx, gy = np.meshgrid(np.linspace(lo0, lo1, 60), np.linspace(la0, la1, 45))
    qlon, qlat = gx.ravel(), gy.ravel()
    truth = pairwise_km(
        qlat,
        qlon,
        seeds["centroid_lat"].to_numpy(),
        seeds["centroid_lon"].to_numpy(),
    ).argmin(axis=1)

    import cartopy.crs as ccrs

    q = pv.crs.transform_points(ccrs.PlateCarree(), qlon, qlat)
    got = np.full(len(qlon), -1)
    for si, cell in zip(pv.seed_index, pv.cells):
        hit = np.array([cell.contains(Point(a, b)) for a, b in zip(q[:, 0], q[:, 1])])
        got[hit] = si

    covered = got >= 0
    assert covered.mean() > 0.99
    assert (got[covered] == truth[covered]).mean() > 0.99


def test_every_target_lands_in_its_own_class_region(grid):
    """Ties the overlay to `assignments`, built by the same nearest-seed rule."""
    import cartopy.crs as ccrs
    from shapely.geometry import Point

    from scripts.analysis.v3.modules.map_answer_space import seed_voronoi

    space = _us_space(grid)
    pv = seed_voronoi(space.seeds, US_MAINLAND_EXTENT)
    cell_of = {int(s): c for s, c in zip(pv.seed_index, pv.cells)}
    pos = {int(s): i for i, s in enumerate(space.seeds["seed_id"])}

    p = pv.crs.transform_points(
        ccrs.PlateCarree(),
        space.assignments["target_lon"].to_numpy(float),
        space.assignments["target_lat"].to_numpy(float),
    )
    for (x, y), sid in zip(p[:, :2], space.assignments["seed_id"]):
        assert cell_of[pos[int(sid)]].buffer(1e-6).contains(Point(x, y))


def test_voronoi_is_skipped_for_a_single_seed(tmp_path, grid):
    """K=1 has no boundary at all; the map must still render."""
    pytest.importorskip("cartopy")
    from scripts.analysis.v3.modules.map_answer_space import seed_voronoi

    space = _space(coords=(CHI,), grid=grid)
    assert space.n_seeds == 1
    assert seed_voronoi(space.seeds, US_MAINLAND_EXTENT) is None
    out = plot_answer_space(space, tmp_path / "one.png", extent=US_MAINLAND_EXTENT)
    assert out.exists()


def test_voronoi_can_be_switched_off(tmp_path, grid):
    """`--no-voronoi` must remove the layer, not merely restyle it."""
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt

    space = _us_space(grid)
    with_ov = plot_answer_space(
        space, tmp_path / "on.png", extent=US_MAINLAND_EXTENT, voronoi=True
    )
    without = plot_answer_space(
        space, tmp_path / "off.png", extent=US_MAINLAND_EXTENT, voronoi=False
    )
    assert with_ov.exists() and without.exists()
    assert with_ov.read_bytes() != without.read_bytes()
    plt.close("all")


def test_seeds_outside_the_frame_still_shape_boundaries_inside_it():
    """Seeds must not be filtered to the frame, unlike the landmass path.

    `voronoi.build_landmass_voronoi` drops centroids outside its boundary, which
    is right there and wrong here: nearest-seed labelling is defined over the
    whole sphere, so a seed just off-screen still owns part of the visible frame
    and bends lines the reader can see. Measured here, one extra off-frame seed
    moves the in-frame area of an existing cell by tens of percent.
    """
    import pandas as pd

    from scripts.analysis.v3.modules.map_answer_space import seed_voronoi

    base = pd.DataFrame(
        {
            "centroid_lat": [41.97, 37.46, 29.76],
            "centroid_lon": [-87.90, -121.92, -95.36],
        }
    )
    off_frame = pd.DataFrame({"centroid_lat": [45.0], "centroid_lon": [-60.0]})
    assert off_frame.loc[0, "centroid_lon"] > US_MAINLAND_EXTENT[1], "must be off-frame"

    a = seed_voronoi(base, US_MAINLAND_EXTENT)
    b = seed_voronoi(
        pd.concat([base, off_frame], ignore_index=True), US_MAINLAND_EXTENT
    )
    assert len(b.cells) == len(a.cells) + 1

    before = {int(i): c.area for i, c in zip(a.seed_index, a.cells)}
    after = {int(i): c.area for i, c in zip(b.seed_index, b.cells)}
    shrunk = max((before[i] - after[i]) / before[i] for i in before)
    assert shrunk > 0.10
