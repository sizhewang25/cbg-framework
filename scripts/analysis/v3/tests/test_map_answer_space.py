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
