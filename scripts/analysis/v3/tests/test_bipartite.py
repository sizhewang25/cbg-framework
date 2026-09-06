"""Tests for the bipartite-graph step: VP nodes, target nodes, measured edges.

Two things here are worth pinning beyond the arithmetic.

**The observed-only contract.** The module's whole point is that every distance
comes off a measured edge, and the failure mode is silent: adding an unmeasured
VP to the roster must move `density` (its column of the incidence matrix is
empty) and must *not* move any target's `nearest_measured_vp_km`, even when it
is parked on top of a target. A latent leak passes every other test.

**The `angular_features` duplicate.** `partvp/extract_features.py` has a private
copy of the same maths and reads an artifact the current output tree does not
have, so the two cannot share code. They are compared directly instead, which is
the only thing that keeps them from drifting.

Grid-parametrized throughout, as `test_answer_space.py` is: H3 ids are hex
strings and HEALPix ids are int64, and `load_bipartite` is the one place that
has to restore the difference after a CSV round trip.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import bipartite as bp
from scripts.analysis.v3.modules.answer_space import build_answer_space
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8968, -97.0380)


@pytest.fixture(params=("h3", "healpix"))
def grid(request):
    return request.param


def _space(coords, grid="h3"):
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


def _vps(coords, asn=7018):
    return pd.DataFrame(
        {
            "vp_id": [f"vp-{i}" for i in range(len(coords))],
            "vp_lat": [c[0] for c in coords],
            "vp_lon": [c[1] for c in coords],
            "vp_asn": [asn] * len(coords),
        }
    )


def _edges(pairs, vps, space):
    """`pairs` are `(vp_index, target_index)` tuples."""
    v = vps.set_index("vp_id")
    t = space.assignments.set_index("target_id")
    rows = []
    for vi, ti in pairs:
        vid, tid = f"vp-{vi}", f"tg-{ti}"
        rows.append(
            {
                "vp_id": vid,
                "vp_lat": v.loc[vid, "vp_lat"],
                "vp_lon": v.loc[vid, "vp_lon"],
                "target_id": tid,
                "target_lat": t.loc[tid, "target_lat"],
                "target_lon": t.loc[tid, "target_lon"],
            }
        )
    return pd.DataFrame(rows)


def _graph(vp_coords, tg_coords, pairs, grid="h3"):
    space = _space(tg_coords, grid=grid)
    vps = _vps(vp_coords)
    return space, vps, bp.build_bipartite(space, vps, _edges(pairs, vps, space))


# ---- angular geometry -------------------------------------------------------

def test_a_single_vp_leaves_the_whole_turn_empty():
    """360/0, and not NaN: one VP does constrain, just from one direction."""
    assert bp.angular_features([12.0]) == (360.0, 0.0)


def test_max_gap_includes_the_wraparound():
    """Bearings 10/40/75 leave 295 deg, which only the wrap term finds."""
    gap, _ = bp.angular_features([10.0, 40.0, 75.0])
    assert gap == pytest.approx(295.0)


def test_bracketing_vps_beat_clustered_ones_at_equal_degree():
    """§7.3's whole claim for the arrangement term."""
    clustered = bp.angular_features([10.0, 40.0, 75.0])
    bracketed = bp.angular_features([20.0, 140.0, 260.0])
    assert bracketed[0] < clustered[0]          # 120 deg vs 295 deg
    assert bracketed[1] > clustered[1]          # better surrounded


def test_circular_variance_is_zero_when_every_bearing_coincides():
    _, cv = bp.angular_features([33.0, 33.0, 33.0])
    assert cv == pytest.approx(0.0)


def test_no_bearings_is_nan_not_a_full_turn():
    gap, cv = bp.angular_features([])
    assert np.isnan(gap) and np.isnan(cv)


def test_angular_features_matches_the_partvp_copy():
    """The duplicate exists because partvp reads an artifact v2 does not write.

    Nothing enforces that the two stay equal except this, so it compares them on
    the real geometry rather than on synthetic bearings.
    """
    from scripts.analysis.partvp.extract_features import (
        _angular_features,
        _bearings_deg,
    )

    vlats = np.array([c[0] for c in (CHI, SJC, NYC, DFW)])
    vlons = np.array([c[1] for c in (CHI, SJC, NYC, DFW)])
    mine = bp.bearings_deg(*DFW, vlats, vlons)
    theirs = _bearings_deg(DFW[0], DFW[1], vlats, vlons)
    assert mine == pytest.approx(theirs)
    assert bp.angular_features(mine) == pytest.approx(_angular_features(theirs))


# ---- edge and node arithmetic -----------------------------------------------

def test_density_is_over_the_roster_not_over_the_edges(grid):
    """The §7.3 denominator is |VP roster| x |answer-space targets|.

    An unmeasured VP is measurement the campaign did not spend, so it belongs in
    the denominator; taking the denominator off the edge list instead would
    report every campaign as complete.
    """
    _, _, graph = _graph([CHI, SJC, NYC], [DFW, NYC], [(0, 0), (0, 1), (1, 0)], grid)
    assert graph.meta["edges"]["n_edges"] == 3
    assert graph.meta["edges"]["edge_density"] == pytest.approx(3 / 6)
    assert graph.meta["nodes"]["vps"]["n_with_no_edge"] == 1


def test_an_unmeasured_vp_never_moves_an_observed_distance(grid):
    """The observed-only contract, tested where a latent leak would show.

    The extra VP is placed *on* target 0, so a latent nearest-VP distance would
    read 0 km. `nearest_measured_vp_km` must not budge.
    """
    _, _, without = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (0, 1)], grid)
    _, _, with_extra = _graph(
        [CHI, SJC, DFW], [DFW, NYC], [(0, 0), (0, 1)], grid
    )
    a = without.target_nodes.set_index("target_id")["nearest_measured_vp_km"]
    b = with_extra.target_nodes.set_index("target_id")["nearest_measured_vp_km"]
    assert a.to_dict() == b.to_dict()
    assert b["tg-0"] > 100.0            # CHI->DFW, not the co-located VP
    # ... but the denominator did move, because the roster grew.
    assert with_extra.meta["edges"]["edge_density"] < without.meta["edges"]["edge_density"]


# ---- the latent half and measurement efficiency -----------------------------

def test_distances_are_reported_as_a_nested_pair(grid):
    """§7.3's pairing, enforced by structure rather than by a naming habit.

    Nested so that neither half can be read with the other out of view — the
    failure the paper names is reporting one and letting sampling bias pass for
    an algorithmic result.
    """
    _, _, graph = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (0, 1)], grid)
    e = graph.meta["edges"]
    for key in ("length_km", "nearest_vp_km"):
        assert set(e[key]) >= {"observed", "latent", "note"}, key
        assert e[key]["observed"]["n"] > 0 and e[key]["latent"]["n"] > 0
    # Degree gets no latent half, and each side says why in its own block
    # rather than in a stray sibling key.
    for side, key in (("vps", "degree_to_target"), ("targets", "degree_to_vp")):
        block = graph.meta["nodes"][side][key]
        assert "latent" not in block
        assert "carries nothing" in block["note"]


def test_the_latent_pair_count_is_the_full_cross_product(grid):
    """|VP| x |targets|, measured or not."""
    _, _, graph = _graph([CHI, SJC, NYC], [DFW, NYC], [(0, 0)], grid)
    e = graph.meta["edges"]
    assert e["length_km"]["n_latent_pairs"] == 6
    assert e["length_km"]["latent"]["n"] == 6
    assert e["length_km"]["observed"]["n"] == 1


def test_efficiency_is_one_when_the_nearest_vp_was_measured(grid):
    """The whole population of the four real runs, so it must be exact."""
    _, _, graph = _graph([CHI, SJC], [DFW], [(0, 0), (1, 0)], grid)
    e = graph.meta["edges"]
    assert graph.target_nodes["measured_nearest_vp_ratio"].tolist() == [1.0]
    assert e["measured_nearest_vp_ratio_per_target"]["max"] == 1.0
    assert bool(graph.target_nodes["nearest_vp_is_measured"].iloc[0])


def test_efficiency_exceeds_one_when_the_campaign_missed_the_nearest_vp(grid):
    """The code path the real datasets never exercise, since all four are 1.0.

    CHI is the nearest VP to a CHI target at 0 km, but only the SJC VP carries an
    edge, so the ratio is (CHI->SJC distance) / (0 km) -- which is the undefined
    case below. Offsetting the target slightly makes the denominator positive and
    the ratio finite and large, which is the ordinary miss.
    """
    near = (41.0, -87.0)
    far = (37.4675, -121.9215)
    space = _space([(41.05, -87.05)], grid=grid)
    vps = _vps([near, far])
    # Only the far VP measured this target.
    graph = bp.build_bipartite(space, vps, _edges([(1, 0)], vps, space))

    eff = float(graph.target_nodes["measured_nearest_vp_ratio"].iloc[0])
    latent = float(graph.target_nodes["nearest_vp_km"].iloc[0])
    measured = float(graph.target_nodes["nearest_measured_vp_km"].iloc[0])
    assert latent < 10.0 and measured > 2_000.0
    assert eff == pytest.approx(measured / latent, rel=1e-4)
    assert eff > 200.0
    assert not bool(graph.target_nodes["nearest_vp_is_measured"].iloc[0])


def test_efficiency_is_never_below_one(grid):
    """`>= 1` by construction: the measured min is over a subset of the latent.

    Pinned because it broke once for a non-obvious reason — edge lengths were
    rounded to 3 dp before the division while the latent denominator was not, so
    160 of as01's 399 targets came out a few 1e-4 below 1.0.
    """
    vp = [CHI, SJC, NYC, DFW, (33.9, -118.4), (47.45, -122.31)]
    # Offset off every VP coordinate, so the denominator is positive everywhere
    # and the ratio is defined for all four targets rather than NaN for the
    # co-located ones (that case is `test_a_coincident_but_unmeasured_vp_...`).
    tg = [(32.95, -97.10), (40.75, -74.05), (39.10, -94.60), (25.79, -80.29)]
    pairs = [(v, t) for v in range(len(vp)) for t in range(len(tg)) if (v + t) % 3]
    _, _, graph = _graph(vp, tg, pairs, grid)
    eff = graph.target_nodes["measured_nearest_vp_ratio"].to_numpy(dtype=float)
    assert np.isfinite(eff).all(), eff
    assert (eff >= 1.0).all(), eff
    # The fixture drops every third pair, so some target must actually miss.
    assert (eff > 1.0).any(), eff


def test_a_coincident_but_unmeasured_vp_is_counted_not_coerced(grid):
    """Ratio is infinite there; reporting 1.0 would invert the finding."""
    space = _space([CHI], grid=grid)
    vps = _vps([CHI, SJC])                       # vp-0 sits exactly on the target
    graph = bp.build_bipartite(space, vps, _edges([(1, 0)], vps, space))
    assert np.isnan(graph.target_nodes["measured_nearest_vp_ratio"].iloc[0])
    # The undefined target is excluded from the block, so `n` falls below the
    # target count by exactly that many -- which is how the count survives its
    # own removal from `meta.json`.
    assert graph.meta["edges"]["measured_nearest_vp_ratio_per_target"]["n"] == 0
    assert graph.meta["nodes"]["targets"]["count"] == 1


def test_a_coincident_and_measured_vp_is_exactly_one_not_zero_over_zero(grid):
    space = _space([CHI], grid=grid)
    vps = _vps([CHI])
    graph = bp.build_bipartite(space, vps, _edges([(0, 0)], vps, space))
    assert graph.target_nodes["measured_nearest_vp_ratio"].tolist() == [1.0]
    assert graph.meta["edges"]["measured_nearest_vp_ratio_per_target"]["n"] == 1


def test_efficiency_snaps_the_two_reductions_residue():
    """Sized from the measured 1.24e-9 relative residue; see `_EFFICIENCY_TOL`."""
    ratio, undefined = bp.measurement_efficiency(
        np.array([100.0 + 1.24e-7, 100.0, 110.0]), np.array([100.0, 100.0, 100.0])
    )
    assert undefined == 0
    assert ratio[0] == 1.0                       # snapped
    assert ratio[1] == 1.0
    assert ratio[2] == pytest.approx(1.1)        # a real miss survives


def test_the_latent_nearest_vp_ignores_the_edge_set(grid):
    """The latent half's defining property, and the observed half's complement.

    Same fixture as `test_an_unmeasured_vp_never_moves_an_observed_distance`,
    read the other way: the co-located VP the observed number must ignore is
    exactly the one the latent number must find.
    """
    _, _, graph = _graph([CHI, SJC, DFW], [DFW, NYC], [(0, 0), (0, 1)], grid)
    row = graph.target_nodes.set_index("target_id").loc["tg-0"]
    assert row["nearest_vp_km"] == pytest.approx(0.0)        # the DFW VP
    assert row["nearest_measured_vp_km"] > 100.0             # only CHI measured
    # tg-0's ratio is infinite (latent 0 km, measured 100+), so it drops out of
    # the block: 2 targets in, 1 described.
    assert graph.meta["edges"]["measured_nearest_vp_ratio_per_target"]["n"] == 1


def test_nearest_across_is_exact_under_chunking():
    """Chunking is what makes the latent half computable at deployment scale.

    A per-chunk `min` over VPs is exact by construction, so this is a guard on
    the bookkeeping (offsets, argmin indices) rather than on the maths.
    """
    rng = np.random.default_rng(7)
    a_lat = rng.uniform(25.0, 49.0, 500)
    a_lon = rng.uniform(-124.0, -67.0, 500)
    b_lat = rng.uniform(25.0, 49.0, 37)
    b_lon = rng.uniform(-124.0, -67.0, 37)
    whole = bp.nearest_across_km(a_lat, a_lon, b_lat, b_lon, chunk=10_000)
    chunked = bp.nearest_across_km(a_lat, a_lon, b_lat, b_lon, chunk=7)
    assert chunked[0] == pytest.approx(whole[0], rel=0, abs=0)
    assert chunked[1].tolist() == whole[1].tolist()


def test_nearest_across_an_empty_side_is_nan_not_a_crash():
    km, idx = bp.nearest_across_km(
        np.array([40.0]), np.array([-80.0]), np.array([]), np.array([])
    )
    assert np.isnan(km).all() and idx.tolist() == [-1]


def test_describe_p90_agrees_with_the_shared_describe_at_three_digits():
    """`describe_p90` computes its own block so it can widen the precision.

    That makes it a fork unless it is checked against the block shape the rest
    of the layer's `meta.json` files use, which is what this does.
    """
    from scripts.analysis.v3.modules.answer_space import _describe

    v = np.concatenate([np.arange(97, dtype=float), [1234.5678, 0.00049, 5.5]])
    mine, theirs = bp.describe_p90(v), _describe(v)
    assert {k: mine[k] for k in ("n", "min", "max", "mean")} == {
        k: theirs[k] for k in ("n", "min", "max", "mean")
    }
    for k, want in theirs["percentiles"].items():
        assert mine["percentiles"][k] == want, k


def test_dropped_keys_stay_dropped(grid):
    """Removed as redundant or unread; a re-add should be deliberate.

    `nearest_other_node_km` was all zeros on every operator run (399 targets at
    20 coordinates, so every target has a coincident twin) and `dispersion`
    answers the same question at a stated scale. `nearest_target_km_latent`
    answered a VP-siting question no §7.3 metric asks.
    """
    _, _, graph = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (1, 1)], grid)
    for side in ("vps", "targets"):
        assert "nearest_other_node_km" not in graph.meta["nodes"][side]
        assert "occupied_cells_by_resolution" not in graph.meta["nodes"][side]
    assert "nearest_target_km_latent" not in graph.meta["edges"]
    assert "nearest_other_vp_km" not in graph.vp_nodes.columns
    assert "nearest_other_target_km" not in graph.target_nodes.columns
    assert "nearest_target_km" not in graph.vp_nodes.columns


def test_a_ratio_block_carries_more_decimals_than_a_distance_block(grid):
    """3 dp hid as03's misses entirely; ratios live just above 1."""
    _, _, graph = _graph([CHI, SJC], [DFW], [(0, 0), (1, 0)], grid)
    e = graph.meta["edges"]
    assert bp._RATIO_DIGITS > 3
    # The distance blocks stay at the layer's 3 dp.
    assert e["length_km"]["observed"]["max"] == round(
        e["length_km"]["observed"]["max"], 3
    )


def test_edge_length_is_great_circle(grid):
    """Checked against haversine, which is a different formula.

    The module reduces to `EARTH_RADIUS_KM * arccos(a . b)` on unit vectors so
    that one matrix multiply covers all pairs; haversine is numerically better
    conditioned at small separations and is derived differently, so agreement to
    a metre is evidence rather than a tautology. (An ellipsoidal geodesic gives
    2,939 km for this pair -- the 0.24% gap is the spherical model's, which the
    whole package assumes.)
    """
    import math

    lat1, lon1, lat2, lon2 = map(math.radians, (CHI[0], CHI[1], SJC[0], SJC[1]))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    expected = 2 * 6371.0 * math.asin(math.sqrt(h))

    _, _, graph = _graph([CHI], [SJC], [(0, 0)], grid)
    assert graph.edge_segments["length_km"].iloc[0] == pytest.approx(expected, abs=1e-3)


def test_degree_counts_distinct_targets_not_observations(grid):
    """A repeated ping of one pair is one edge; §7.3's density is over pairs."""
    space = _space([DFW, NYC], grid=grid)
    vps = _vps([CHI])
    dup = pd.concat([_edges([(0, 0), (0, 1)], vps, space)] * 3, ignore_index=True)
    graph = bp.build_bipartite(space, vps, dup.drop_duplicates(["vp_id", "target_id"]))
    assert graph.vp_nodes["degree_to_target"].tolist() == [2]
    assert graph.meta["edges"]["n_edges"] == 2


def test_occupied_cells_never_exceed_the_node_count(grid):
    """Co-located VPs collapse -- the reason the count is the honest denominator."""
    _, _, graph = _graph([CHI, CHI, SJC], [DFW], [(0, 0), (1, 0), (2, 0)], grid)
    disp = graph.meta["nodes"]["vps"]["dispersion"]
    res = str(graph.meta["grid"]["resolution"])
    assert disp[res]["effective_count"] == 2 < 3
    # occupancy_ratio is effective_count / count, which is what makes it read as
    # "spread" vs "stacked" independently of set size.
    assert disp[res]["occupancy_ratio"] == pytest.approx(2 / 3, abs=1e-4)
    # And the ladder coarsens monotonically, since re-binning can only merge.
    rungs = sorted((k for k in disp if k != "note"), key=int, reverse=True)
    counts = [disp[k]["effective_count"] for k in rungs]
    assert counts == sorted(counts, reverse=True)


def test_target_occupied_cells_equal_the_seed_count(grid):
    """One occupied target cell is exactly one class, by construction."""
    space, _, graph = _graph([CHI], [DFW, NYC, SJC], [(0, 0), (0, 1), (0, 2)], grid)
    res = str(graph.meta["grid"]["resolution"])
    assert graph.meta["nodes"]["targets"]["dispersion"][res]["effective_count"] == space.n_seeds


def test_diameter_is_reported_with_its_p95(grid):
    """§7.3 requires both, because one node can set a diameter alone."""
    _, _, graph = _graph([CHI, SJC, NYC], [DFW], [(0, 0), (1, 0), (2, 0)], grid)
    v = graph.meta["nodes"]["vps"]
    assert v["geographic_diameter_km"] == pytest.approx(v["pairwise_km"]["max"])
    assert v["pairwise_p95_km"] <= v["geographic_diameter_km"]


# ---- components -------------------------------------------------------------

def test_a_split_edge_set_reports_two_components(grid):
    """§7.3 asks for this because traffic can concentrate regionally."""
    _, _, graph = _graph(
        [CHI, SJC], [DFW, NYC], [(0, 0), (1, 1)], grid
    )
    cc = graph.meta["edges"]["connected_components"]
    assert cc["n_components"] == 2
    assert cc["largest_component_node_share"] == pytest.approx(0.5)


def test_an_unmeasured_vp_is_its_own_component(grid):
    """A VP that measured nothing is disconnected from the experiment."""
    _, _, graph = _graph([CHI, SJC], [DFW], [(0, 0)], grid)
    cc = graph.meta["edges"]["connected_components"]
    assert cc["n_components"] == 2 and cc["n_isolated_nodes"] == 1


def test_more_than_127_nodes_do_not_overflow_the_incidence_matrix():
    """pandas hands back int8 codes for <= 127 categories.

    `n_v + target_code` then wrapped negative and scipy rejected the column --
    which is how as7018 (53 VPs, 78 targets) failed while as01 (134) passed.
    """
    lats = 30.0 + np.arange(60) * 0.4
    space = _space([(float(a), -95.0) for a in lats])
    vps = _vps([(float(a), -90.0) for a in lats])
    graph = bp.build_bipartite(space, vps, _edges([(i, i) for i in range(60)], vps, space))
    assert graph.meta["edges"]["connected_components"]["n_components"] == 60


# ---- flow segments ----------------------------------------------------------

def test_coincident_edges_collapse_into_one_segment(grid):
    """The flow map's honesty rule: duplicate IPs at one site are one line."""
    space = _space([CHI, CHI, CHI], grid=grid)      # three IPs, one facility
    vps = _vps([SJC])
    graph = bp.build_bipartite(space, vps, _edges([(0, 0), (0, 1), (0, 2)], vps, space))
    assert graph.meta["edges"]["n_edges"] == 3
    assert graph.meta["edges"]["n_distinct_geometry_edge"] == 1
    assert graph.edge_segments["n_edges"].tolist() == [3]


def test_distinct_targets_do_not_collapse(grid):
    _, _, graph = _graph([SJC], [CHI, NYC], [(0, 0), (0, 1)], grid)
    assert graph.meta["edges"]["n_distinct_geometry_edge"] == 2


# ---- joins and guards -------------------------------------------------------

def test_an_edge_naming_an_unknown_target_is_dropped_and_counted(grid):
    space = _space([DFW], grid=grid)
    vps = _vps([CHI])
    edges = _edges([(0, 0)], vps, space)
    stray = edges.iloc[[0]].assign(target_id="tg-not-in-the-answer-space")
    graph = bp.build_bipartite(space, vps, pd.concat([edges, stray], ignore_index=True))
    assert graph.meta["inputs"]["n_edges_dropped_target_not_in_answer_space"] == 1
    assert graph.meta["edges"]["n_edges"] == 1


def test_an_edge_naming_an_unknown_vp_is_dropped_and_counted(grid):
    space = _space([DFW], grid=grid)
    vps = _vps([CHI])
    edges = _edges([(0, 0)], vps, space)
    stray = edges.iloc[[0]].assign(vp_id="vp-not-in-the-roster")
    graph = bp.build_bipartite(space, vps, pd.concat([edges, stray], ignore_index=True))
    assert graph.meta["inputs"]["n_edges_dropped_vp_not_in_roster"] == 1
    assert graph.meta["edges"]["n_edges"] == 1


def test_a_graph_sharing_nothing_with_the_run_is_refused(grid):
    space = _space([DFW], grid=grid)
    vps = _vps([CHI])
    edges = _edges([(0, 0)], vps, space).assign(vp_id="vp-elsewhere")
    with pytest.raises(ValueError, match="do not describe the same graph"):
        bp.build_bipartite(space, vps, edges)


def test_one_id_with_two_coordinates_is_refused(tmp_path):
    """Otherwise every distance downstream is silently ambiguous."""
    csv = tmp_path / "canonical.csv"
    csv.write_text(
        "vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,10.0\n"
        "vp-0,42.0,-88.0,tg-1,40.0,-74.0,20.0\n"
    )
    with pytest.raises(ValueError, match="more than one coordinate"):
        bp.load_edges(csv)


def test_load_edges_collapses_repeated_observations(tmp_path):
    csv = tmp_path / "canonical.csv"
    csv.write_text(
        "vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,10.0\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,11.0\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,-1.0\n"
    )
    edges, n_obs = bp.load_edges(csv)
    assert n_obs == 2            # the non-positive RTT is not an observation
    assert len(edges) == 1
    assert "rtt_ms" not in edges.columns


# ---- round trip -------------------------------------------------------------

def test_write_then_load_round_trips(tmp_path, grid):
    _, _, graph = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (0, 1), (1, 1)], grid)
    out = graph.write(tmp_path / "bg")
    back = bp.load_bipartite(out)
    pd.testing.assert_frame_equal(back.edge_segments, graph.edge_segments)
    pd.testing.assert_frame_equal(back.edge_length_cdf, graph.edge_length_cdf)
    assert back.meta == graph.meta
    # cell_id dtype survives the CSV: hex strings for H3, int64 for HEALPix.
    assert back.vp_nodes["cell_id"].tolist() == graph.vp_nodes["cell_id"].tolist()


def test_load_without_the_artifacts_says_which_command_writes_them(tmp_path):
    with pytest.raises(MissingArtifactError, match="build-bipartite-graph"):
        bp.load_bipartite(tmp_path)


def test_the_cdfs_are_monotone_on_a_fixed_quantile_grid(grid):
    _, _, graph = _graph([CHI, SJC, NYC], [DFW, NYC], [(0, 0), (1, 0), (2, 1)], grid)
    for frame, col in (
        (graph.edge_length_cdf, "observed_edge_km"),
        (graph.edge_length_cdf, "latent_pair_km"),
        (graph.pairwise_distance_cdf, "vp_pairwise_km"),
        (graph.pairwise_distance_cdf, "target_pairwise_km"),
    ):
        v = frame[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        assert np.all(np.diff(v) >= -1e-9), col
    assert graph.edge_length_cdf["quantile"].iloc[[0, -1]].tolist() == [0.0, 1.0]


def test_describe_p90_reads_monotonically():
    d = bp.describe_p90(np.arange(101, dtype=float))
    assert list(d["percentiles"]) == ["p5", "p25", "p50", "p75", "p90", "p95"]
    assert d["percentiles"]["p90"] == pytest.approx(90.0)


# ---- source-csv resolution --------------------------------------------------

def test_the_source_csv_is_read_off_the_runs_own_eval_stats(tmp_path, monkeypatch):
    """v3 configs do not carry it (`benchmark: {}`), but the run records it."""
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup="setup")
    run.eval_source_dir.mkdir(parents=True)
    (run.eval_source_dir / "base_eval_per_target.csv").write_text("target_id\n")
    csv = tmp_path / "canonical.csv"
    csv.write_text("x\n")
    (run.eval_source_dir / "base_eval_stats.json").write_text(
        json.dumps({"csv": str(csv)})
    )
    assert bp.resolve_source_csv(run) == csv


def test_an_unlocatable_source_csv_names_the_override_flag(tmp_path):
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup="setup")
    run.eval_source_dir.mkdir(parents=True)
    (run.eval_source_dir / "gone_eval_per_target.csv").write_text("target_id\n")
    (run.eval_source_dir / "gone_eval_stats.json").write_text(
        json.dumps({"csv": "datasets/does-not-exist.csv"})
    )
    with pytest.raises(MissingArtifactError, match="--source-csv"):
        bp.resolve_source_csv(run)


def test_an_explicit_override_wins_and_is_checked(tmp_path):
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup="setup")
    csv = tmp_path / "mine.csv"
    csv.write_text("x\n")
    assert bp.resolve_source_csv(run, csv) == csv
    with pytest.raises(MissingArtifactError, match="does not exist"):
        bp.resolve_source_csv(run, tmp_path / "nope.csv")
