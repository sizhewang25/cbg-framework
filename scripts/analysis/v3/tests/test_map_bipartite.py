"""Tests for the bipartite figures and the shared mapping primitives.

The rendering tests are deliberately thin, following
`test_map_answer_space.py`: cartopy is slow and pixel assertions are brittle, so
what is pinned is that each figure gets written on **both** grids. That is not
redundant with the geometry tests -- H3 cell rings are ragged (6 vertices for a
hexagon, 5 for one of its 12 pentagons) where HEALPix rings are a uniform
`4 * step`, so a layer that assumed a rectangular array passes every unit test
and fails only here.

`great_circle_segments` gets real assertions, because it is geometry and the way
it goes wrong is silent: a straight lon/lat chord renders as a plausible line
that is simply in the wrong place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import build_answer_space, elementwise_km
from scripts.analysis.v3.modules.bipartite import build_bipartite
from scripts.analysis.v3.modules.map_bipartite import (
    plot_bipartite_flows,
    plot_bipartite_nodes,
    plot_distance_cdf,
)
from scripts.analysis.v3.modules.mapping import great_circle_segments

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8968, -97.0380)
SEA = (47.4502, -122.3088)


@pytest.fixture(params=("h3", "healpix"))
def grid(request):
    return request.param


def _fixture(grid):
    tg = [CHI, SJC, NYC, DFW]
    vp = [SEA, DFW, NYC]
    space = build_answer_space(
        pd.DataFrame(
            {
                "target_id": [f"tg-{i}" for i in range(len(tg))],
                "target_lat": [c[0] for c in tg],
                "target_lon": [c[1] for c in tg],
            }
        ),
        grid=grid,
    )
    vps = pd.DataFrame(
        {
            "vp_id": [f"vp-{i}" for i in range(len(vp))],
            "vp_lat": [c[0] for c in vp],
            "vp_lon": [c[1] for c in vp],
            "vp_asn": [7018] * len(vp),
        }
    )
    t = space.assignments.set_index("target_id")
    rows = [
        {
            "vp_id": f"vp-{vi}",
            "vp_lat": vp[vi][0],
            "vp_lon": vp[vi][1],
            "target_id": f"tg-{ti}",
            "target_lat": t.loc[f"tg-{ti}", "target_lat"],
            "target_lon": t.loc[f"tg-{ti}", "target_lon"],
        }
        for vi in range(len(vp))
        for ti in range(len(tg))
    ]
    graph = build_bipartite(space, vps, pd.DataFrame(rows))
    return space, graph


# ---- great-circle interpolation ---------------------------------------------

def test_the_path_bows_where_an_independent_geodesic_says_it_does():
    """The reason the flow map does not draw straight lon/lat segments.

    Checked against `pyproj.Geod.npts`, an independent (and ellipsoidal)
    implementation, rather than against a remembered number. CHI->SEA bows
    1.313 deg north of its lon/lat chord there and 1.308 deg here -- the ~0.005
    deg residual is the spherical model this package assumes, and 1.3 deg is
    ~146 km, i.e. tens of pixels of misplaced line at continental scale.

    pyproj is not a new dependency: cartopy requires it, and these tests already
    render through cartopy.
    """
    from pyproj import Geod

    path = great_circle_segments(
        np.array([CHI[0]]), np.array([CHI[1]]), np.array([SEA[0]]), np.array([SEA[1]])
    )[0]
    mid_lon, mid_lat = path[len(path) // 2]

    ref_lon, ref_lat = Geod(ellps="WGS84").npts(
        CHI[1], CHI[0], SEA[1], SEA[0], 1
    )[0]
    assert mid_lat == pytest.approx(ref_lat, abs=0.01)
    assert mid_lon == pytest.approx(ref_lon, abs=0.01)
    # ... and that really is poleward of the straight lon/lat line.
    assert mid_lat > (CHI[0] + SEA[0]) / 2


def test_every_interpolated_point_lies_on_the_path(grid):
    """Cumulative segment length must equal the endpoint distance."""
    a, b = np.array([CHI[0]]), np.array([CHI[1]])
    c, d = np.array([SJC[0]]), np.array([SJC[1]])
    path = great_circle_segments(a, b, c, d, n_points=33)[0]
    hops = elementwise_km(path[:-1, 1], path[:-1, 0], path[1:, 1], path[1:, 0])
    total = float(elementwise_km(a, b, c, d)[0])
    assert hops.sum() == pytest.approx(total, rel=1e-6)


def test_the_endpoints_are_exact():
    path = great_circle_segments(
        np.array([CHI[0]]), np.array([CHI[1]]), np.array([NYC[0]]), np.array([NYC[1]])
    )[0]
    assert path[0] == pytest.approx((CHI[1], CHI[0]), abs=1e-9)
    assert path[-1] == pytest.approx((NYC[1], NYC[0]), abs=1e-9)


def test_a_path_across_the_antimeridian_stays_contiguous():
    """Same rule as `grid.ring_lonlat`: within half a turn of the first vertex.

    Without it the polyline's longitudes flip sign mid-path and the line is
    smeared clear across the map instead of crossing the edge.
    """
    path = great_circle_segments(
        np.array([20.0]), np.array([179.0]), np.array([20.0]), np.array([-179.0])
    )[0]
    assert np.abs(np.diff(path[:, 0])).max() < 10.0
    assert path[:, 0].max() > 180.0


def test_coincident_endpoints_do_not_divide_by_zero():
    """A VP measuring a target at its own coordinate is an ordinary case."""
    path = great_circle_segments(
        np.array([CHI[0]]), np.array([CHI[1]]), np.array([CHI[0]]), np.array([CHI[1]])
    )[0]
    assert np.all(np.isfinite(path))
    assert path[:, 0] == pytest.approx(CHI[1])


def test_mismatched_endpoint_arrays_are_refused():
    with pytest.raises(ValueError, match="disagree"):
        great_circle_segments(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([3.0]), np.array([3.0])
        )


# ---- the figures ------------------------------------------------------------

def test_all_three_figures_render(tmp_path, grid):
    space, graph = _fixture(grid)
    nodes = plot_bipartite_nodes(space, graph, tmp_path / "nodes.png")
    flows = plot_bipartite_flows(space, graph, tmp_path / "flows.png")
    cdf = plot_distance_cdf(graph, tmp_path / "cdf.png")
    for p in (nodes, flows, cdf):
        assert p.exists() and p.stat().st_size > 0


def test_the_optional_layers_can_be_dropped(tmp_path, grid):
    space, graph = _fixture(grid)
    out = plot_bipartite_nodes(
        space, graph, tmp_path / "bare.png", voronoi=False, vp_cells=False
    )
    assert out.exists() and out.stat().st_size > 0


def test_the_flow_map_can_be_capped(tmp_path, grid):
    """Deterministic, so a capped figure is still reproducible."""
    space, graph = _fixture(grid)
    a = plot_bipartite_flows(space, graph, tmp_path / "a.png", max_segments=4)
    b = plot_bipartite_flows(space, graph, tmp_path / "b.png", max_segments=4)
    assert a.read_bytes() == b.read_bytes()


def test_both_efficiency_panel_branches_render(tmp_path, grid):
    """All four real runs are degenerate at 1.00, so the other branch needs this.

    The degenerate branch draws a sentence instead of a CDF, because a vertical
    line at 1.0 would imply a distribution the data does not have. Only the
    non-degenerate branch draws a curve, and nothing in the repo's data reaches
    it.
    """
    from scripts.analysis.v3.modules.bipartite import build_bipartite

    space, degenerate = _fixture(grid)
    assert degenerate.meta["edges"]["measurement_efficiency"]["max"] == 1.0
    a = plot_distance_cdf(degenerate, tmp_path / "flat.png")

    # Same node sets, but every target measured only the farthest VP.
    tg = space.assignments
    vps = degenerate.vp_nodes[["vp_id", "vp_lat", "vp_lon", "vp_asn"]]
    far = vps.iloc[[0]]
    rows = [
        {
            "vp_id": far["vp_id"].iloc[0],
            "vp_lat": far["vp_lat"].iloc[0],
            "vp_lon": far["vp_lon"].iloc[0],
            "target_id": r.target_id,
            "target_lat": r.target_lat,
            "target_lon": r.target_lon,
        }
        for r in tg.itertuples()
    ]
    missed = build_bipartite(space, vps, pd.DataFrame(rows))
    assert missed.meta["edges"]["measurement_efficiency"]["max"] > 1.0
    b = plot_distance_cdf(missed, tmp_path / "spread.png")

    for out in (a, b):
        assert out.exists() and out.stat().st_size > 0
    assert a.read_bytes() != b.read_bytes()


def test_the_frame_covers_vps_outside_the_target_hull(tmp_path, grid):
    """`plot-answer-space` frames on targets alone; an off-hull VP is the point.

    Anchorage sits well north and west of every target here, so a target-only
    extent would crop the one node the topology figure exists to reveal.
    """
    from scripts.analysis.v3.modules.map_bipartite import _extent_for

    space, graph = _fixture(grid)
    graph.vp_nodes.loc[0, ["vp_lat", "vp_lon"]] = [61.17, -149.99]
    lon_min, lon_max, lat_min, lat_max = _extent_for(space, graph, None)
    assert lat_max > 61.17 and lon_min < -149.99
