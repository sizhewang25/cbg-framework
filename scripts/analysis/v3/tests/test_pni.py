"""Invariants for the tripartite (VP, PNI, target) graph.

Two families here. The first pins the assignment itself — an argmin that a
chunk-boundary or gather bug would corrupt silently, because every wrong answer
is still a plausible site. The second pins the arithmetic that makes the
artifact readable: the inflation identity, and the fact that `gc_km` and the air
inflation are *carried* from `eval_source` rather than recomputed. A drift in
either speed constant is invisible in the column and fatal to the decomposition.

Deliberately not re-tested: the two-leg sum, the 2/3 c floor and the OLS fits,
which `test_figure_pni_delay.py` already owns.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import typer

from scripts.analysis.v3.modules import figure_pni_delay, pni
from scripts.analysis.v3.modules.answer_space import pairwise_km
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths
from scripts.benchmark.v2.eval_source import build_pairs, per_target_metrics
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

# Real metros, so a distance that looks wrong can be checked against intuition.
CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8968, -97.0380)
DEN = (39.8561, -104.6737)
MIA = (25.7807, -80.2952)

ASN = 15169


def _pni(rows, asn=ASN, **extra):
    """`rows` is (id, (lat, lon)); optional columns come in as equal-length lists."""
    frame = pd.DataFrame(
        {
            "pni_id": [r[0] for r in rows],
            "pni_lat": [r[1][0] for r in rows],
            "pni_lon": [r[1][1] for r in rows],
            "pni_asn": asn,
        }
    )
    for key, value in extra.items():
        frame[key] = value
    return frame.sort_values("pni_id").reset_index(drop=True)


def _pairs(rows, target_asn=None):
    """`rows` is (vp_id, (lat, lon), tg_id, (lat, lon), rtt_ms) -> a build_pairs frame."""
    df = pd.DataFrame(
        {
            "vp_id": [r[0] for r in rows],
            "vp_lat": [r[1][0] for r in rows],
            "vp_lon": [r[1][1] for r in rows],
            "target_id": [r[2] for r in rows],
            "target_lat": [r[3][0] for r in rows],
            "target_lon": [r[3][1] for r in rows],
            "rtt_ms": [r[4] for r in rows],
        }
    )
    if target_asn is not None:
        df["target_asn"] = target_asn
    return build_pairs(df)


def _soi_rtt(a, b, inflation=1.5):
    """An RTT `inflation` times the 2/3 c floor for the a->b geodesic."""
    km = pairwise_km(np.array([a[0]]), np.array([a[1]]), np.array([b[0]]), np.array([b[1]]))[0, 0]
    return THEORETICAL_SLOPE * km * inflation


_SITES = _pni([("pni-chi", CHI), ("pni-nyc", NYC), ("pni-sjc", SJC)])

#: Sites away from every endpoint used below, so the two-leg path is a genuine
#: detour rather than the direct distance rediscovered. With endpoints at sites,
#: both routes tie at the direct distance and no hairpin exists to observe.
_INLAND = _pni([("pni-den", DEN), ("pni-dfw", DFW)])


def _graph(pair_rows=None, sites=None, **kw):
    pairs = _pairs(
        pair_rows
        or [
            ("vp-w", SJC, "tg-e", NYC, _soi_rtt(SJC, NYC)),
            ("vp-e", NYC, "tg-e", NYC, _soi_rtt(NYC, NYC, 1.0) + 1.0),
            ("vp-c", CHI, "tg-e", NYC, _soi_rtt(CHI, NYC)),
        ]
    )
    return pni.build_pni_graph(pairs, sites if sites is not None else _SITES, **kw)


# ---- the assignment ---------------------------------------------------------


def test_the_selected_site_minimizes_the_two_leg_path():
    """Brute force against the vectorized argmin. Every wrong answer here is
    still a real site, so nothing downstream would look broken."""
    g = _graph()
    sites = _SITES
    for row in g.edges.itertuples():
        legs = [
            pairwise_km(np.array([row.vp_lat]), np.array([row.vp_lon]), np.array([la]), np.array([lo]))[0, 0]
            + pairwise_km(np.array([la]), np.array([lo]), np.array([row.target_lat]), np.array([row.target_lon]))[0, 0]
            for la, lo in zip(sites.pni_lat, sites.pni_lon)
        ]
        assert row.sel_pni_id == sites.pni_id.iloc[int(np.argmin(legs))]
        assert row.vp_to_tg_via_pni_km == pytest.approx(min(legs), rel=1e-6)


def test_chunking_does_not_change_the_assignment(monkeypatch):
    """The chunk boundary is where an index-arithmetic error hides, and it never
    fires on the runs in the repo."""
    whole = _graph().edges
    monkeypatch.setattr(pni, "_ARGMIN_CHUNK_CELLS", 3)
    chunked = _graph().edges
    pd.testing.assert_frame_equal(whole, chunked)


def test_site_row_order_does_not_change_the_result():
    """The tie rule is only deterministic if the sort inside load/assign is real."""
    shuffled = _SITES.iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(_graph().edges, _graph(sites=shuffled).edges)


def test_ties_are_flagged_and_broken_on_the_lowest_site_id():
    """Two records for one facility is the expected tie, not a pathology."""
    sites = _pni([("pni-a", CHI), ("pni-b", CHI)])
    g = _graph(sites=sites)
    assert (g.edges.sel_pni_id == "pni-a").all()
    assert g.edges.sel_pni_is_tied.all()
    assert g.meta["assignment"]["n_ties"] == len(g.edges)


# ---- the arithmetic ---------------------------------------------------------


def test_air_inflation_is_the_detour_ratio_times_the_routing_inflation():
    """The headline identity. Both inflations divide by THEORETICAL_SLOPE, so a
    drift in either denominator is invisible in the column but breaks the
    decomposition the paper reads off it."""
    g = _graph()
    assert g.meta["checks"]["inflation_identity"]["max_abs_rel_residual"] == pytest.approx(0, abs=1e-12)
    e = g.edges
    ok = (np.isfinite(e.vp_to_tg_air_inflation) & np.isfinite(e.vp_to_tg_routing_inflation)).to_numpy()
    product = (e.vp_to_tg_detour_ratio * e.vp_to_tg_routing_inflation).to_numpy()[ok]
    assert product == pytest.approx(e.vp_to_tg_air_inflation.to_numpy()[ok], rel=2e-6)


def test_air_inflation_and_the_direct_distance_are_v2s_columns_unchanged():
    """Recomputing either would fork THEORETICAL_SLOPE, which is the coupling
    `proximity._carry` exists to avoid."""
    pairs = _pairs([("vp-w", SJC, "tg-e", NYC, 40.0), ("vp-c", CHI, "tg-e", NYC, 20.0)])
    g = pni.build_pni_graph(pairs, _SITES)
    assert g.edges.vp_to_tg_km.to_numpy() == pytest.approx(
        pairs.gc_km.to_numpy().round(pni._KM_DIGITS), abs=1e-9
    )
    assert g.edges.vp_to_tg_air_inflation.to_numpy() == pytest.approx(pairs.inflation.to_numpy(), rel=1e-6)


def test_routing_inflation_never_exceeds_air_inflation():
    """detour >= 1 implies routing <= air; a violation means a units bug."""
    e = _graph().edges
    ok = np.isfinite(e.vp_to_tg_routing_inflation) & np.isfinite(e.vp_to_tg_air_inflation)
    assert (e.vp_to_tg_routing_inflation[ok] <= e.vp_to_tg_air_inflation[ok] + 1e-9).all()


def test_the_detour_ratio_floor_is_snapped_rather_than_left_to_rounding():
    """`via >= direct` is the triangle inequality, so a deficit is numerical. On
    as01 577 pairs land below 1 by under 5e-7 — inside the 6 dp the column is
    written at, so without the snap the floor would hold by accident."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        vp = (rng.uniform(25, 49), rng.uniform(-124, -67))
        tg = (rng.uniform(25, 49), rng.uniform(-124, -67))
        rows.append((f"vp-{i}", vp, f"tg-{i}", tg, rng.uniform(5, 90)))
    sites = _pni([(f"pni-{i}", (rng.uniform(25, 49), rng.uniform(-124, -67))) for i in range(6)])
    g = pni.build_pni_graph(_pairs(rows), sites)
    assert (g.edges.vp_to_tg_detour_ratio.dropna() >= 1.0).all()
    assert g.meta["checks"]["n_detour_below_one"] == 0


def test_the_identity_does_not_survive_the_per_target_minimum():
    """The three minima are attained by different VPs, which is why no
    min-detour-ratio column is emitted: it would invite this false arithmetic."""
    # vp-a sits at a site, so its path is direct and its routing inflation
    # equals its air inflation. vp-b hairpins, so its routing inflation is
    # divided down below vp-a's. Air is then minimized by vp-a and routing by
    # vp-b, which is the whole point.
    # One site, at Miami: far enough off the SJC->NYC great circle that vp-b is
    # forced into a real detour. Denver would not do -- it sits almost on that
    # line, so its detour ratio is 1.002 and the ordering never flips.
    g = pni.build_pni_graph(
        _pairs(
            [
                ("vp-a", MIA, "tg", NYC, _soi_rtt(MIA, NYC, 1.30)),
                ("vp-b", SJC, "tg", NYC, _soi_rtt(SJC, NYC, 1.50)),
            ]
        ),
        _pni([("pni-mia", MIA)]),
    )
    assert "min_vp_to_tg_detour_ratio" not in g.target_nodes.columns
    e = g.edges.set_index("vp_id")
    assert e.vp_to_tg_air_inflation.idxmin() == "vp-a"
    assert e.vp_to_tg_routing_inflation.idxmin() == "vp-b"
    row = g.target_nodes.iloc[0]
    product = row.min_vp_to_tg_routing_inflation * e.vp_to_tg_detour_ratio.min()
    assert product != pytest.approx(row.min_vp_to_tg_air_inflation, rel=1e-3)


# ---- agreement with the target's own nearest site ---------------------------


def test_agreement_is_true_exactly_when_the_selected_site_is_the_targets_nearest():
    g = _graph()
    e = g.edges
    assert (e.sel_pni_is_tg_nearest == (e.sel_pni_id == e.tg_nearest_pni_id)).all()


def test_a_far_vp_hairpins_through_its_own_side_and_disagrees():
    """The case that makes the headline share non-vacuous."""
    g = _graph(
        [("vp-w", SJC, "tg-w", SJC, 5.0), ("vp-w", SJC, "tg-e", NYC, 70.0)], sites=_INLAND
    )
    east = g.edges[g.edges.target_id == "tg-e"].iloc[0]
    # The argmin routes the west-coast VP through the western site; the target's
    # own nearest site is the eastern one. That disagreement is the hairpin.
    assert east.sel_pni_id == "pni-den"
    assert east.tg_nearest_pni_id == "pni-dfw"
    assert not east.sel_pni_is_tg_nearest
    assert east.vp_to_tg_detour_ratio > 1.0


def test_the_headline_share_is_the_column_mean():
    g = _graph()
    assert g.meta["agreement"]["sel_pni_is_tg_nearest_share"] == pytest.approx(
        g.edges.sel_pni_is_tg_nearest.mean(), abs=1e-6
    )


def test_two_records_for_one_facility_report_zero_distance_not_agreement():
    """The flag is identity, the distance is proximity. Folding the two together
    would report agreement the argmin never actually reached."""
    sites = _pni([("pni-a", NYC), ("pni-z", NYC)])
    g = _graph([("vp-e", NYC, "tg-e", NYC, 2.0)], sites=sites)
    row = g.edges.iloc[0]
    assert row.sel_pni_id == "pni-a" and row.tg_nearest_pni_id == "pni-a"
    assert row.sel_pni_to_tg_nearest_pni_km == 0.0


# ---- SOI, as a check on the assignment --------------------------------------


def test_routing_soi_violations_contain_the_air_side_ones():
    g = _graph()
    assert (
        g.meta["soi"]["routing_soi_violation_share"] >= g.meta["soi"]["soi_violation_share"]
    )


def test_an_rtt_between_the_two_floors_violates_only_the_routing_side():
    """This is what separates the two readings: the air share is a property of
    the data, the routing share is a verdict on the assignment."""
    g = _graph(
        [("vp-w", SJC, "tg-e", NYC, _soi_rtt(SJC, NYC, 1.05))],
        sites=_pni([("pni-dfw", DFW)]),
    )
    assert g.meta["soi"]["soi_violation_share"] == 0.0
    assert g.meta["soi"]["routing_soi_violation_share"] == 1.0


# ---- the cross-module pin ---------------------------------------------------


def test_the_per_target_min_air_inflation_reproduces_eval_sources_min_inflation():
    """If these drift, one of the two speed constants moved."""
    pairs = _pairs(
        [
            ("vp-w", SJC, "tg-e", NYC, 70.0),
            ("vp-c", CHI, "tg-e", NYC, 25.0),
            ("vp-d", DFW, "tg-e", NYC, 45.0),
        ]
    )
    g = pni.build_pni_graph(pairs, _SITES)
    ref = per_target_metrics(pairs).set_index("target_id")["min_inflation"]
    got = g.target_nodes.set_index("target_id")["min_vp_to_tg_air_inflation"]
    assert got.loc[ref.index].to_numpy() == pytest.approx(ref.to_numpy(), rel=1e-9)


# ---- degenerate inputs ------------------------------------------------------


def test_a_colocated_vp_and_target_yield_nan_not_infinity():
    """NaN and not inf: JSON cannot encode inf, and describe_p90 filters
    non-finite values, so an inf would vanish from a block without changing n."""
    g = _graph([("vp-e", NYC, "tg-e", NYC, 3.0)], sites=_pni([("pni-nyc", NYC)]))
    row = g.edges.iloc[0]
    assert np.isnan(row.vp_to_tg_detour_ratio)
    assert np.isnan(row.vp_to_tg_routing_inflation)
    assert g.meta["degenerate"]["n_pairs_colocated_vp_tg"] == 1
    assert g.meta["degenerate"]["n_pairs_zero_via_pni"] == 1
    json.dumps(g.meta)  # would raise on a non-finite value


def test_a_site_with_no_coordinate_is_dropped_rather_than_capturing_every_pair(tmp_path):
    """np.argmin over a row containing NaN returns the NaN column's index, so
    carrying the row would silently reassign the whole dataset."""
    csv = tmp_path / "sites.csv"
    pd.DataFrame(
        {
            "pni_id": ["pni-good", "pni-bad"],
            "pni_lat": [CHI[0], np.nan],
            "pni_lon": [CHI[1], CHI[1]],
            "pni_asn": ASN,
        }
    ).to_csv(csv, index=False)
    sites, diag = pni.load_pni_sites(csv)
    assert diag["n_pni_rows_dropped_nan_coords"] == 1
    assert sites.pni_id.tolist() == ["pni-good"]
    assert (_graph(sites=sites).edges.sel_pni_id == "pni-good").all()


def test_a_site_list_with_no_usable_row_is_refused(tmp_path):
    csv = tmp_path / "sites.csv"
    pd.DataFrame({"pni_id": [], "pni_lat": [], "pni_lon": [], "pni_asn": []}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="--pni-csv"):
        pni.load_pni_sites(csv)


def test_a_missing_required_column_names_it_and_the_flag(tmp_path):
    csv = tmp_path / "sites.csv"
    pd.DataFrame({"pni_id": ["a"], "pni_lat": [1.0], "pni_asn": [ASN]}).to_csv(csv, index=False)
    with pytest.raises(typer.BadParameter, match="pni_lon"):
        pni.load_pni_sites(csv)


def test_one_site_id_at_two_coordinates_is_refused(tmp_path):
    csv = tmp_path / "sites.csv"
    pd.DataFrame(
        {"pni_id": ["a", "a"], "pni_lat": [CHI[0], NYC[0]], "pni_lon": [CHI[1], NYC[1]], "pni_asn": ASN}
    ).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="more than one coordinate"):
        pni.load_pni_sites(csv)


def test_a_target_prefixed_header_is_accepted_by_renaming(tmp_path):
    """The file was specified as shaped like the canonical target CSV."""
    csv = tmp_path / "sites.csv"
    pd.DataFrame(
        {
            "target_id": ["pni-chi"],
            "target_lat": [CHI[0]],
            "target_lon": [CHI[1]],
            "target_asn": [ASN],
            "target_city": ["Chicago"],
        }
    ).to_csv(csv, index=False)
    sites, diag = pni.load_pni_sites(csv)
    assert diag["pni_renamed_from_target_prefix"]
    assert sites.pni_id.tolist() == ["pni-chi"] and sites.pni_city.tolist() == ["Chicago"]


# ---- the peer-ASN guard -----------------------------------------------------


def test_a_multi_asn_site_list_is_refused():
    """One run is one peer ASN, so one file cannot describe two peerings."""
    sites = pd.concat([_pni([("a", CHI)], asn=15169), _pni([("b", NYC)], asn=32934)])
    with pytest.raises(typer.BadParameter, match="peer ASN"):
        pni.resolve_peer_asn(_pairs([("v", CHI, "t", NYC, 20.0)]), sites)


def test_a_site_list_for_another_peer_is_refused_when_the_run_can_say_so():
    pairs = _pairs([("v", CHI, "t", NYC, 20.0)], target_asn=32934)
    with pytest.raises(typer.BadParameter, match="AS15169"):
        pni.resolve_peer_asn(pairs, _SITES)


def test_an_unverifiable_peer_asn_is_recorded_rather_than_blessed():
    """The operator runs lost target_asn to the parquet reconstruction, so there
    is nothing to check against; saying so beats implying a check happened."""
    _, diag = pni.resolve_peer_asn(_pairs([("v", CHI, "t", NYC, 20.0)]), _SITES)
    assert diag["peer_asn"] == ASN
    assert diag["peer_asn_verified_against"].startswith("unverified")


def test_peer_asn_contradicting_the_file_is_refused():
    with pytest.raises(typer.BadParameter, match="contradicts"):
        pni.resolve_peer_asn(_pairs([("v", CHI, "t", NYC, 20.0)]), _SITES, declared=32934)


# ---- capacity, artifacts, paths, CLI ----------------------------------------


def test_capacity_is_carried_and_never_reduced():
    """The scope decision is easy to erode by adding 'just a p50'."""
    sites = _pni([("pni-chi", CHI), ("pni-nyc", NYC), ("pni-sjc", SJC)], pni_capacity=[400, 200, 100])
    g = _graph(sites=sites)
    assert "sel_pni_capacity" in g.edges.columns
    assert "pni_capacity" in g.pni_nodes.columns
    assert not any("capacity" in k for k in (*g.meta["distances"], *g.meta["ratios"]))


def test_write_then_load_round_trips(tmp_path):
    g = _graph()
    g.write(tmp_path)
    back = pni.load_pni_graph(tmp_path)
    pd.testing.assert_frame_equal(g.edges, back.edges)
    pd.testing.assert_frame_equal(g.pni_nodes, back.pni_nodes)
    assert back.meta == g.meta


def test_loading_without_the_artifacts_names_the_command_that_writes_them(tmp_path):
    with pytest.raises(MissingArtifactError, match="build-pni-graph"):
        pni.load_pni_graph(tmp_path)


def test_the_output_directory_carries_no_grid_slug(tmp_path):
    """This is the deviation from every other accessor, so a later
    'consistency' edit would silently move the artifact."""
    run = RunPaths(run_id="r", root=tmp_path / "bench", source="generic_csv", setup="s")
    out = run.pni_graph_dir(root=tmp_path / "analysis")
    assert out.name == "pni-graph"
    assert "h3-" not in str(out) and "healpix-" not in str(out)


def test_the_command_declares_run_id_but_no_grid_options():
    """Both halves of the grid-free deviation, plus the `run_id` that
    `config.default_map` needs in order to inject a config's top-level run_id."""
    from scripts.analysis.v3.cli import app

    params = {p.name for p in typer.main.get_command(app).commands["build-pni-graph"].params}
    assert {"grid", "resolution", "sweep", "answer_space", "all_runs"} & params == set()
    assert {"run_id", "pni_csv"} <= params


def test_the_site_list_is_required_because_it_cannot_be_defaulted():
    """There is no repo convention for where a peer's sites live, so guessing
    one would silently describe the wrong peering."""
    from scripts.analysis.v3.cli import app

    params = {
        p.name: p for p in typer.main.get_command(app).commands["build-pni-graph"].params
    }
    assert params["pni_csv"].required and params["run_id"].required


# ---- the handoff to plot-pni-delay ------------------------------------------


def test_the_edge_csv_is_consumable_by_plot_pni_delay(tmp_path):
    """The two modules are only useful together, and `load_points` assigns its
    derived columns rather than validating them."""
    g = _graph()
    g.write(tmp_path)
    points, dropped = figure_pni_delay.load_points(
        tmp_path / pni.PNI_EDGES_CSV, pni_prefix="sel_pni", tg_prefix="target", rtt_col="rtt_ms"
    )
    assert dropped == 0 and len(points) == len(g.edges)
    assert points.d_via_pni_km.to_numpy() == pytest.approx(
        g.edges.vp_to_tg_via_pni_km.to_numpy(), rel=1e-5
    )


def test_our_columns_do_not_collide_with_the_figures_derived_names(tmp_path):
    """`load_points` does `df[col] = ...`, so a shared name is a silent
    overwrite agreeing only to haversine-vs-arccos precision."""
    derived = {
        "d_vp_pni_km", "d_pni_tg_km", "d_vp_tg_km", "d_via_pni_km",
        "prop_rtt_via_pni_ms", "prop_rtt_direct_ms", "min_rtt_ms", "residual_ms",
    }
    assert derived & set(_graph().edges.columns) == set()


def test_the_routing_soi_share_is_the_figures_below_floor_share(tmp_path):
    """The same predicate stated twice, so the meta scalar and the figure
    annotation are one number by construction."""
    g = _graph()
    g.write(tmp_path)
    points, _ = figure_pni_delay.load_points(
        tmp_path / pni.PNI_EDGES_CSV, pni_prefix="sel_pni", tg_prefix="target", rtt_col="rtt_ms"
    )
    stats = figure_pni_delay.plot(points, tmp_path / "fig.png")
    assert stats["n_below_floor"] / stats["n"] == pytest.approx(
        g.meta["soi"]["routing_soi_violation_share"], abs=1e-6
    )


def test_site_leg_frames_returns_the_matrices_assign_pni_reduces():
    """The two-leg matrices are shared with `pni_feasibility` and
    `pni_linearity`, so they had to come out of `assign_pni`.

    The tie rule is a property of the site frame's *order*, so a consumer that
    rebuilt these itself and sorted differently would silently pick different
    winners while every artifact still looked plausible. This pins that the
    factored function is the one `assign_pni` actually uses.
    """
    pairs = _pairs(
        [
            ("vp-a", DEN, "tg-1", NYC, 30.0),
            ("vp-b", SJC, "tg-1", NYC, 60.0),
            ("vp-a", DEN, "tg-2", MIA, 40.0),
        ]
    )
    frames = pni.site_leg_frames(pairs, _SITES)
    assign = pni.assign_pni(pairs, _SITES)

    assert frames["d_vp_p"].shape == (pairs["vp_id"].nunique(), len(_SITES))
    assert frames["d_tg_p"].shape == (pairs["target_id"].nunique(), len(_SITES))
    assert frames["d_pp"].shape == (len(_SITES), len(_SITES))
    # The reduction `assign_pni` performs, recomputed from the shared frames.
    cost = frames["d_vp_p"][frames["e_vp"], :] + frames["d_tg_p"][frames["e_tg"], :]
    assert np.array_equal(assign["sel"], cost.argmin(axis=1))
    assert assign["via_km"] == pytest.approx(cost.min(axis=1))
    np.testing.assert_array_equal(assign["e_vp"], frames["e_vp"])
    np.testing.assert_array_equal(assign["e_tg"], frames["e_tg"])


def test_site_leg_frames_are_stable_under_a_permutation_of_the_edges():
    """Why both node frames are sorted by id.

    The site axis carries `assign_pni`'s tie rule and every index a consumer
    gathers with, so the frames must be a function of the edge *set* rather than
    of its row order. Without the sort, feeding the same pairs in a different
    order would renumber the node rows and silently repoint every gather.
    """
    rows = [
        ("vp-b", SJC, "tg-2", MIA, 60.0),
        ("vp-a", DEN, "tg-1", NYC, 30.0),
        ("vp-a", DEN, "tg-2", MIA, 40.0),
    ]
    a_pairs, b_pairs = _pairs(rows), _pairs(list(reversed(rows)))
    forward = pni.site_leg_frames(a_pairs, _SITES)
    reverse = pni.site_leg_frames(b_pairs, _SITES)

    pd.testing.assert_frame_equal(forward["vp"], reverse["vp"])
    pd.testing.assert_frame_equal(forward["tg"], reverse["tg"])
    np.testing.assert_allclose(forward["d_vp_p"], reverse["d_vp_p"])
    np.testing.assert_allclose(forward["d_tg_p"], reverse["d_tg_p"])

    # The index arrays follow their own frame's row order, so they are checked
    # by what they resolve to rather than by position.
    for frames, pairs in ((forward, a_pairs), (reverse, b_pairs)):
        assert (
            frames["vp"]["vp_id"].to_numpy()[frames["e_vp"]] == pairs["vp_id"].to_numpy()
        ).all()
        assert (
            frames["tg"]["target_id"].to_numpy()[frames["e_tg"]]
            == pairs["target_id"].to_numpy()
        ).all()


def test_a_pure_strategy_assigns_a_fixed_site_per_target_or_per_vp():
    """`--strategy` is what makes the assignment a detected input rather than a
    hardcoded rule, and every number downstream inherits it.

    Under a pure rule the choice is a per-unit constant, so there is nothing to
    tie and no chunked scan; the test pins both the selection and that `tied` is
    empty, since a stray tie flag would misreport the assignment's confidence.
    """
    pairs = _pairs(
        [
            ("vp-a", DEN, "tg-1", NYC, 30.0),
            ("vp-b", SJC, "tg-1", NYC, 60.0),
            ("vp-a", DEN, "tg-2", MIA, 40.0),
        ]
    )
    tg = pni.assign_pni(pairs, _SITES, strategy="tg_nearest")
    vp = pni.assign_pni(pairs, _SITES, strategy="vp_nearest")

    # Fixed per target: both VPs of tg-1 get the same site.
    sel_tg = pd.Series(tg["sel"], index=pairs["target_id"].to_numpy())
    assert sel_tg.groupby(level=0).nunique().max() == 1
    # Fixed per VP: vp-a gets the same site for both its targets.
    sel_vp = pd.Series(vp["sel"], index=pairs["vp_id"].to_numpy())
    assert sel_vp.groupby(level=0).nunique().max() == 1

    assert not tg["tied"].any() and not vp["tied"].any()
    assert tg["strategy"] == "tg_nearest" and vp["strategy"] == "vp_nearest"
    # The two-leg sum is still the sum of the two legs it reports.
    assert tg["via_km"] == pytest.approx(
        tg["vp_to_sel_pni_km"] + tg["sel_pni_to_tg_km"], abs=1e-9
    )


def test_an_unknown_strategy_is_refused():
    """A typo must not fall through to the default and silently change the rule."""
    pairs = _pairs([("vp-a", DEN, "tg-1", NYC, 30.0)])
    with pytest.raises(ValueError, match="strategy must be one of"):
        pni.assign_pni(pairs, _SITES, strategy="nearest")


def test_a_partial_split_csv_is_refused_rather_than_defaulting_to_false():
    """A pair missing from the split would silently be scored as detect-half,
    re-denominating every study that filters on `is_holdout`."""
    pairs = _pairs(
        [("vp-a", DEN, "tg-1", NYC, 30.0), ("vp-b", SJC, "tg-1", NYC, 60.0)]
    )
    partial = pd.DataFrame(
        {"vp_id": ["vp-a"], "target_id": ["tg-1"], "is_holdout": [True]}
    )
    with pytest.raises(ValueError, match="absent from it"):
        pni.build_pni_graph(pairs, _SITES, pair_split=partial)
