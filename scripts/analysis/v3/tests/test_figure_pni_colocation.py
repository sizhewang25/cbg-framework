"""The co-location scatter: row filtering, replica collapsing and the stats block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import typer

from scripts.analysis.v3.modules import figure_pni_colocation as F


def _edge(target_id, x_km, y_km, *, is_sping, vp_id="v0", nearest=True):
    """One `pni_edges.csv` row with the two legs stated directly.

    The legs are columns `build-pni-graph` carries, so a fixture states them
    rather than placing coordinates and hoping the haversine reproduces them —
    this module recomputes nothing and the test should not either.
    """
    return {
        "vp_id": vp_id,
        "target_id": target_id,
        "rtt_ms": 5.0,
        "sel_pni_id": "p0",
        F.X_COLUMN: x_km,
        F.Y_COLUMN: y_km,
        "vp_to_tg_km": x_km + y_km,
        "vp_to_tg_via_pni_km": x_km + y_km,
        "sel_pni_is_tg_nearest": nearest,
        F.SPING_FLAG: is_sping,
    }


def _graph(tmp_path, rows):
    out = tmp_path / "pni-graph"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "pni_edges.csv", index=False)
    return out


def test_only_the_shortest_ping_edge_of_each_target_is_plotted(tmp_path):
    """The figure claims one point per target; a second edge would double-count
    that target's position and silently reweight the cloud toward well-measured
    targets."""
    graph = _graph(tmp_path, [
        _edge("t0", 10.0, 2.0, is_sping=True, vp_id="v0"),
        _edge("t0", 900.0, 2.0, is_sping=False, vp_id="v1"),
        _edge("t1", 20.0, 4.0, is_sping=True, vp_id="v0"),
        _edge("t1", 800.0, 4.0, is_sping=False, vp_id="v1"),
    ])

    points, diagnostics = F.load_sping_legs(graph)

    assert sorted(points["target_id"]) == ["t0", "t1"]
    assert sorted(points["x_km"]) == [10.0, 20.0]
    assert diagnostics["n_edges"] == 4
    assert diagnostics["n_sping_edges"] == 2


def test_a_string_valued_sping_flag_still_selects_the_right_rows(tmp_path):
    """A CSV round trip leaves `True`/`False` as strings whenever the column has
    a missing value, and a truthiness test on the raw strings would keep every
    row — including the ones written `False`."""
    rows = [
        _edge("t0", 10.0, 2.0, is_sping="True"),
        _edge("t1", 20.0, 4.0, is_sping="False"),
    ]
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))

    assert list(points["target_id"]) == ["t0"]


def test_coincident_targets_collapse_to_one_position_carrying_its_count(tmp_path):
    """The operator datasets put ~20 IP replicas on one coordinate. Drawing them
    as separate marks asserts a sample size the data does not have, so the
    replica count has to survive as a number rather than as overplotting."""
    rows = [_edge(f"t{i}", 10.0, 2.0, is_sping=True) for i in range(20)]
    rows.append(_edge("t99", 40.0, 8.0, is_sping=True))
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))

    distinct = F.collapse(points)

    assert len(points) == 21
    assert len(distinct) == 2
    assert sorted(distinct["n_targets"]) == [1, 20]


def test_the_points_csv_keeps_one_row_per_target_even_when_marks_are_collapsed(tmp_path):
    """Collapsing is a drawing decision. If it reached the artifact, a consumer
    joining on `target_id` would find 19 of every 20 targets missing."""
    rows = [_edge(f"t{i}", 10.0, 2.0, is_sping=True) for i in range(20)]
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))

    F.collapse(points)

    assert len(points) == 20
    assert points["target_id"].nunique() == 20


def test_the_colocation_radius_is_the_larger_leg_not_their_sum(tmp_path):
    """The claim is a corner: both legs small at once. A pair with one tiny leg
    and one huge one has a small *sum* relative to two medium legs, so summing
    would score the wrong geometry as the more co-located one."""
    rows = [
        _edge("t0", 1.0, 99.0, is_sping=True),   # sum 100, radius 99
        _edge("t1", 40.0, 40.0, is_sping=True),  # sum  80, radius 40
    ]
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))

    stats = F.series_stats(points)

    assert stats["colocation_radius_km_p50"] == pytest.approx(69.5)
    assert stats["routing_km_p50"] == pytest.approx(90.0)


def test_the_stats_report_both_the_target_count_and_the_distinct_count(tmp_path):
    """Both numbers are load-bearing: the first is the population, the second is
    how many independent positions the panel actually shows."""
    rows = [_edge(f"t{i}", 10.0, 2.0, is_sping=True) for i in range(20)]
    rows.append(_edge("t99", 40.0, 8.0, is_sping=True))
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))

    stats = F.series_stats(points)

    assert stats["n_targets"] == 21
    assert stats["n_distinct_points"] == 2
    assert stats["max_targets_per_point"] == 20


def test_the_nearest_site_collapse_is_measured_rather_than_assumed(tmp_path):
    """§8.1 predicts that argmin assignment and the target's own nearest site
    converge once targets sit on sites. Reporting the share keeps that a
    prediction; assuming it would make the figure unfalsifiable."""
    rows = [
        _edge("t0", 10.0, 2.0, is_sping=True, nearest=True),
        _edge("t1", 20.0, 4.0, is_sping=True, nearest=True),
        _edge("t2", 30.0, 6.0, is_sping=True, nearest=False),
        _edge("t3", 40.0, 8.0, is_sping=True, nearest=False),
    ]
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))

    assert F.series_stats(points)["sel_pni_is_tg_nearest_share"] == pytest.approx(0.5)


def test_count_scaled_marker_area_is_linear_in_the_replica_count(tmp_path):
    """Area is the channel the eye integrates, so a square-root mapping would
    render a 20x stack as a ~4.5x mark and understate the multiplicity the mode
    exists to expose."""
    distinct = pd.DataFrame({"x_km": [0.0, 1.0, 2.0], "y_km": [0.0, 1.0, 2.0],
                             "n_targets": [1, 10, 19]})

    sizes = F.marker_sizes(distinct, point_size=26.0, marker=F.MARKER_COUNT)

    assert sizes[0] == pytest.approx(F._AREA_MIN)
    assert sizes[-1] == pytest.approx(F._AREA_MAX)
    # 10 sits halfway between 1 and 19, so its area sits halfway between the ends.
    assert sizes[1] == pytest.approx((F._AREA_MIN + F._AREA_MAX) / 2)


def test_fixed_marker_area_ignores_the_replica_count(tmp_path):
    """The default mode draws what the caller asked for — equal dots — so the
    two modes cannot silently blend into one another."""
    distinct = pd.DataFrame({"x_km": [0.0, 1.0], "y_km": [0.0, 1.0], "n_targets": [1, 20]})

    sizes = F.marker_sizes(distinct, point_size=26.0, marker=F.MARKER_FIXED)

    assert np.allclose(sizes, 26.0)


def test_a_single_position_still_gets_a_drawable_marker_area(tmp_path):
    """One distinct position makes the count range degenerate; a naive
    normalisation would divide by zero and emit NaN areas, which matplotlib
    drops silently — an empty panel with no error."""
    distinct = pd.DataFrame({"x_km": [0.0], "y_km": [0.0], "n_targets": [7]})

    sizes = F.marker_sizes(distinct, point_size=26.0, marker=F.MARKER_COUNT)

    assert np.isfinite(sizes).all()
    assert sizes[0] > 0


def test_a_graph_with_no_sping_edge_is_refused_rather_than_drawn_empty(tmp_path):
    """Every target should carry exactly one flagged edge. None at all means the
    graph was written by something that did not flag them, and a blank panel
    would read as 'no co-location' instead of 'wrong input'."""
    rows = [_edge("t0", 10.0, 2.0, is_sping=False)]

    with pytest.raises(typer.BadParameter, match="flags no shortest-ping edge"):
        F.load_sping_legs(_graph(tmp_path, rows))


def test_an_older_graph_missing_a_leg_column_names_the_command_that_rewrites_it(tmp_path):
    """The legs are carried, not recomputed, so an artifact predating them can
    only be fixed upstream; the error has to say where."""
    out = tmp_path / "pni-graph"
    out.mkdir(parents=True)
    pd.DataFrame([{"target_id": "t0", F.SPING_FLAG: True}]).to_csv(
        out / "pni_edges.csv", index=False
    )

    with pytest.raises(typer.BadParameter, match="build-pni-graph"):
        F.load_sping_legs(out)


def test_a_missing_graph_directory_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(typer.BadParameter, match="build-pni-graph"):
        F.load_sping_legs(tmp_path / "pni-graph")


def test_non_finite_legs_are_dropped_and_counted(tmp_path):
    """A NaN leg is a broken assignment, not a point at the origin."""
    rows = [
        _edge("t0", 10.0, 2.0, is_sping=True),
        _edge("t1", float("nan"), 4.0, is_sping=True),
    ]
    points, diagnostics = F.load_sping_legs(_graph(tmp_path, rows))

    assert list(points["target_id"]) == ["t0"]
    assert diagnostics["n_dropped_non_finite"] == 1
    assert diagnostics["n_plotted"] == 1


def test_a_run_ids_role_suffix_is_not_printed_twice_in_the_legend():
    """`short_dataset` only strips an all-numeric tail, which the finals run ids
    do not have — they end in the role the label already states."""
    assert F._layer_label("as01-260728-260802-mesh", "mesh") == "as01 mesh"
    assert (
        F._layer_label("as01-260728-260802-weighted", "traffic-weighted")
        == "as01 traffic-weighted"
    )
    # A run id that is not a finals one survives intact rather than being
    # truncated at the first hyphen.
    assert F._layer_label("as7018_us_test01", "mesh") == "as7018_us_test01 mesh"


def test_a_randweight_overlay_says_its_weights_are_synthetic():
    """The only run that can populate the red layer today is the `.randweight`
    fixture, and a bare "traffic-weighted" legend would be read as evidence of
    real weights. No weight-bearing export has been collected."""
    assert F._weighted_label("as01-randweight-precomputed").endswith(
        "(SYNTHETIC weights)"
    )
    assert "SYNTHETIC" not in F._weighted_label("as01-260728-260802-weighted")


def test_the_figure_renders_both_layers_when_the_twin_is_supplied(tmp_path):
    rows = [_edge(f"t{i}", 10.0 * i + 1, 2.0 * i + 1, is_sping=True) for i in range(5)]
    mesh, _ = F.load_sping_legs(_graph(tmp_path, rows))
    tw = mesh.iloc[:2].copy()
    out_png = tmp_path / "out.png"

    F.build(
        mesh, tw, mesh_label="as01 mesh", tw_label="as01 traffic-weighted",
        stats={"mesh": F.series_stats(mesh), "traffic_weighted": F.series_stats(tw)},
        marker=F.MARKER_COUNT, point_size=26.0, alpha=0.55,
        x_max=200.0, y_max=50.0, log_axes=False, title=None, out_png=out_png,
    )

    assert out_png.exists() and out_png.stat().st_size > 0


def test_log_axes_clamp_marks_below_the_floor_and_report_how_many(tmp_path):
    """A log axis has no zero, and a VP sitting exactly on its site is a legal
    measurement. Dropping it would delete the most co-located point in the
    dataset; clamping it silently would pile the floor up as a false cluster."""
    rows = [
        _edge("t0", 0.0, 5.0, is_sping=True),
        _edge("t1", 50.0, 0.01, is_sping=True),
        _edge("t2", 50.0, 5.0, is_sping=True),
    ]
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))
    # The overlay is a subset of the mesh, and `n_clamped` sums over both
    # layers -- so this one is the single mesh position ABOVE the floor, which
    # keeps the expected count attributable to the mesh layer alone.
    tw = points.iloc[[2]]
    out_png = tmp_path / "out.log.png"

    n_clamped = F.build(
        points, tw, mesh_label="m", tw_label="w",
        stats={"mesh": F.series_stats(points), "traffic_weighted": F.series_stats(tw)},
        marker=F.MARKER_FIXED,
        point_size=26.0, alpha=0.55, x_max=None, y_max=None, log_axes=True,
        title=None, out_png=out_png,
    )

    assert n_clamped == 2
    assert out_png.exists() and out_png.stat().st_size > 0


def test_the_log_floor_never_reaches_the_statistics(tmp_path):
    """Clamping is a drawing decision. A quantile computed on clamped values
    would report a co-location radius the measurement never showed."""
    rows = [_edge(f"t{i}", 0.0, 0.0, is_sping=True) for i in range(3)]
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))
    tw = points.iloc[:1]

    F.build(
        points, tw, mesh_label="m", tw_label="w",
        stats={"mesh": F.series_stats(points), "traffic_weighted": F.series_stats(tw)},
        marker=F.MARKER_FIXED,
        point_size=26.0, alpha=0.55, x_max=None, y_max=None, log_axes=True,
        title=None, out_png=tmp_path / "out.png",
    )

    assert F.series_stats(points)["colocation_radius_km_p50"] == 0.0
    assert points["x_km"].max() == 0.0


def test_linear_axes_clamp_nothing(tmp_path):
    """The default scale has an origin, so a zero leg is drawn where it is."""
    rows = [_edge("t0", 0.0, 0.0, is_sping=True), _edge("t1", 9.0, 9.0, is_sping=True)]
    points, _ = F.load_sping_legs(_graph(tmp_path, rows))
    tw = points.iloc[:1]

    n_clamped = F.build(
        points, tw, mesh_label="m", tw_label="w",
        stats={"mesh": F.series_stats(points), "traffic_weighted": F.series_stats(tw)},
        marker=F.MARKER_FIXED,
        point_size=26.0, alpha=0.55, x_max=None, y_max=None, log_axes=False,
        title=None, out_png=tmp_path / "out.png",
    )

    assert n_clamped == 0
