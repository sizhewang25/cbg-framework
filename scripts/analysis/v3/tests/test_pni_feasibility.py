"""Invariants for the speed-of-internet feasibility test over PNI sites.

The module's whole value is that an exclusion is *sound*: every site it rules
out is one the pair could not physically have crossed. Three things could
silently break that and leave an artifact that still reads plausibly — a `k`
that is not an upper bound on speed, a hard intersection sneaking in as a
filter, and an empty feasible set being attributed to off-list serving when it
is really a bad coordinate. Each gets a test here.

The rest pin the geometry the reporting rests on: that a near VP is selective
and a far VP is not, and that its feasible ranks by proximity-to-target come out
non-contiguous, which is the reason the headline is restricted to the
shortest-ping VP rather than pooled over all pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import typer

from scripts.analysis.v3.modules import pni, pni_feasibility as F
from scripts.analysis.v3.modules.answer_space import pairwise_km
from scripts.analysis.v3.modules.paths import MissingArtifactError
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

# Real metros, so a distance that looks wrong can be checked against intuition.
CHI = (41.8534, -87.6180)
NYC = (40.7178, -74.0090)
DEN = (39.7436, -104.9903)
LAX = (34.0489, -118.2570)
SEA = (47.6146, -122.3390)
ATL = (33.7556, -84.3915)

ASN = 15169


def _km(a, b):
    return float(
        pairwise_km(np.array([a[0]]), np.array([a[1]]), np.array([b[0]]), np.array([b[1]]))[0, 0]
    )


def _sites(rows, asn=ASN):
    return pd.DataFrame(
        {
            "pni_id": [r[0] for r in rows],
            "pni_lat": [r[1][0] for r in rows],
            "pni_lon": [r[1][1] for r in rows],
            "pni_asn": asn,
        }
    ).sort_values("pni_id").reset_index(drop=True)


def _graph(pair_rows, sites, *, soi_meta=None):
    """A minimal `PniGraph` carrying only what `build_feasibility` reads."""
    pairs = pd.DataFrame(
        {
            "vp_id": [r[0] for r in pair_rows],
            "vp_lat": [r[1][0] for r in pair_rows],
            "vp_lon": [r[1][1] for r in pair_rows],
            "target_id": [r[2] for r in pair_rows],
            "target_lat": [r[3][0] for r in pair_rows],
            "target_lon": [r[3][1] for r in pair_rows],
            "rtt_ms": [r[4] for r in pair_rows],
        }
    )
    assign = pni.assign_pni(pairs, sites)
    pairs["sel_pni_id"] = sites["pni_id"].to_numpy()[assign["sel"]]
    pairs["vp_to_tg_via_pni_km"] = assign["via_km"]
    pairs["vp_to_tg_km"] = np.round(pairs_direct(pairs), pni._KM_DIGITS)
    pairs["is_sping_vp"] = [r[5] if len(r) > 5 else False for r in pair_rows]
    nodes = sites.copy()
    return pni.PniGraph(
        edges=pairs,
        target_nodes=pd.DataFrame(),
        pni_nodes=nodes,
        meta={"inputs": {"pni_csv": "datasets/pni/test-sites.csv", "peer_asn": ASN},
              "soi": soi_meta or {}},
    )


def pairs_direct(pairs):
    from scripts.analysis.v3.modules.answer_space import elementwise_km

    return elementwise_km(
        pairs["vp_lat"].to_numpy(float),
        pairs["vp_lon"].to_numpy(float),
        pairs["target_lat"].to_numpy(float),
        pairs["target_lon"].to_numpy(float),
    )


def _rtt_for(path_km, *, slack=1.0):
    """The RTT whose 2/3 c budget is exactly `slack` times `path_km`."""
    return THEORETICAL_SLOPE * path_km * slack


def test_a_near_vp_with_a_small_rtt_admits_only_the_targets_own_site():
    """The headline claim, in its smallest form.

    If this stops holding, "only nearby interconnects are physically possible"
    is no longer supported by anything, because the shortest-ping VP is the pair
    the paper quotes and this is that pair.
    """
    sites = _sites([("pni-chi", CHI), ("pni-den", DEN), ("pni-lax", LAX), ("pni-nyc", NYC)])
    vp = (39.90, -104.99)  # ~17 km north of the Denver site
    # 3 ms of budget: 300 km of path, which only the co-located site can meet.
    graph = _graph([("vp-a", vp, "tg-1", DEN, 3.0, True)], sites)
    out = F.build_feasibility(graph)
    row = out.pairs.iloc[0]

    assert row["n_feasible"] == 1
    assert row["is_feasible_set_singleton"]
    assert row["tg_nearest_pni_id"] == "pni-den"
    assert row["feasible_equals_tg_nearest_pni"]
    assert row["feasible_tg_site_ranks"] == "1"


def test_a_far_vp_admits_sites_that_are_far_from_the_target_at_non_contiguous_ranks():
    """Why the headline is restricted to the shortest-ping VP.

    A distant VP's ellipse elongates along the VP-target axis and admits sites
    near the *VP*. Their ranks by proximity-to-target then have gaps, so
    "the feasible set is the top-k nearest sites" is false for far VPs and any
    pooled version of the claim would be wrong.
    """
    sites = _sites(
        [("pni-atl", ATL), ("pni-chi", CHI), ("pni-den", DEN), ("pni-lax", LAX),
         ("pni-nyc", NYC), ("pni-sea", SEA)]
    )
    # Seattle VP to a Denver target, with enough budget to reach the west coast.
    direct = _km(SEA, DEN)
    graph = _graph([("vp-sea", SEA, "tg-1", DEN, _rtt_for(direct, slack=2.6))], sites)
    out = F.build_feasibility(graph)
    row = out.pairs.iloc[0]

    ranks = [int(v) for v in row["feasible_tg_site_ranks"].split("|")]
    assert row["n_feasible"] > 1
    assert ranks != list(range(1, len(ranks) + 1)), "expected a gap in the rank list"
    assert not row["feasible_ranks_are_contiguous"]
    # A site admitted here is far from the target, which is the point.
    assert row["feasible_pni_to_tg_km_max"] > 1000


def test_k_is_the_theoretical_slope_and_no_other_speed_appears():
    """Exclusion is sound only when k upper-bounds propagation speed.

    A calibrated or envelope-fitted speed is an average, so about half of all
    paths beat it and half the exclusions would be wrong. If a future change
    threads one in as the primary, this fails.
    """
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    graph = _graph([("vp-a", DEN, "tg-1", DEN, 5.0)], sites)
    out = F.build_feasibility(graph)

    assert out.pairs["k_ms_per_km"].unique().tolist() == [THEORETICAL_SLOPE]
    assert out.meta["k"]["k_ms_per_km"] == THEORETICAL_SLOPE
    assert "2/3 c" in out.meta["k"]["provenance"]
    assert "calibrated" not in json_dumps(out.meta["feasibility"])


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj)


def test_the_budget_is_the_rtt_at_two_thirds_c_so_a_path_at_the_bound_is_feasible():
    """The comparison is `<=`, not `<`, and the eps only absorbs float noise.

    A pair whose two-leg path is exactly its budget sits on the ellipse and is
    physically possible; excluding it would make the test unsound in the strict
    direction, which is harder to notice than the loose one.
    """
    sites = _sites([("pni-chi", CHI), ("pni-nyc", NYC)])
    path = _km(DEN, CHI) + _km(CHI, NYC)
    graph = _graph([("vp-a", DEN, "tg-1", NYC, _rtt_for(path))], sites)
    out = F.build_feasibility(graph)

    assert out.pairs.iloc[0]["n_feasible"] >= 1
    assert out.pairs.iloc[0]["sel_pni_is_feasible"]


def test_an_empty_set_caused_by_beating_the_direct_floor_is_not_the_falsifier():
    """The falsifier must not fire on a bad coordinate.

    Since `via >= direct` always, a pair whose RTT already beats 2/3 c on the
    direct geodesic has an empty feasible set by arithmetic. That is a data
    problem, which `build-pni-graph` reports as `soi_violation_share`, and
    counting it as evidence of off-list serving would turn data quality into a
    finding.
    """
    sites = _sites([("pni-chi", CHI), ("pni-nyc", NYC)])
    direct = _km(DEN, LAX)
    rows = [
        # Beats the direct floor: impossible RTT, so empty *and* SoI-violating.
        ("vp-bad", DEN, "tg-1", LAX, _rtt_for(direct, slack=0.5)),
        # Respects the direct floor but not the two-leg one: empty, unexplained.
        ("vp-ok", DEN, "tg-2", LAX, _rtt_for(direct, slack=1.01)),
    ]
    out = F.build_feasibility(_graph(rows, sites))
    bad, ok = out.pairs.iloc[0], out.pairs.iloc[1]

    assert bad["is_feasible_set_empty"] and bad["is_empty_and_soi_violating"]
    assert not bad["is_empty_unexplained_by_soi"]
    assert ok["is_feasible_set_empty"] and ok["is_empty_unexplained_by_soi"]
    assert not ok["is_empty_and_soi_violating"]
    assert out.meta["feasibility"]["empty_set_rate"]["n_empty"] == 2
    assert out.meta["feasibility"]["empty_set_rate"]["n_empty_unexplained_by_soi"] == 1


def test_the_hard_intersection_is_reported_but_never_used_to_filter():
    """`SphericalCircleMTL`'s failure mode, kept visible instead of inherited.

    One under-predicting constraint empties an intersection, so a target with a
    single anomalous VP would report "no site is possible". `vps_feasible_share`
    is the statistic; the intersection is a measured number beside it.
    """
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    # Target and VPs together, ~200 km north of the Denver site, so the two-leg
    # path is ~400 km while the direct distance is 0 -- no RTT here can violate
    # the *direct* floor, which keeps this test about the intersection alone.
    spot = (41.54, -104.99)
    rows = [
        ("vp-a", spot, "tg-1", spot, 5.0),
        ("vp-b", spot, "tg-1", spot, 5.0),
        # One VP whose budget (300 km) cannot cover the 400 km two-leg path.
        ("vp-c", spot, "tg-1", spot, 3.0),
    ]
    out = F.build_feasibility(_graph(rows, sites))
    den = out.sites[out.sites["pni_id"] == "pni-den"].iloc[0]

    assert den["n_vps"] == 3
    assert den["n_vps_feasible"] == 2
    assert den["vps_feasible_share"] == pytest.approx(2 / 3, abs=1e-6)
    # The intersection is empty, and that is reported rather than acted on.
    assert not den["feasible_under_all_vps"]
    assert out.meta["per_target_site_share"]["intersection"][
        "n_targets_with_empty_intersection"
    ] == 1


def test_a_site_no_vp_admits_is_suppressed_unless_it_is_the_targets_nearest():
    """The suppression rule, which controls the one artifact that grows with the site list.

    The nearest-site row must survive unconditionally so a downstream join on
    (target_id, tg_nearest_pni_id) can never miss, which is exactly the join
    `breakdown-sping-pni` performs.
    """
    sites = _sites([("pni-den", DEN), ("pni-nyc", NYC), ("pni-sea", SEA)])
    # A tight RTT at Denver: neither NYC nor Seattle is reachable.
    out = F.build_feasibility(_graph([("vp-a", DEN, "tg-1", DEN, 3.0)], sites))

    kept = set(out.sites["pni_id"])
    assert kept == {"pni-den"}
    assert out.meta["per_target_site_share"]["n_site_rows_suppressed_all_infeasible"] == 2

    # And with the target nearest a site no VP admits, that row still survives.
    out2 = F.build_feasibility(_graph([("vp-a", SEA, "tg-1", DEN, 0.5)], sites))
    assert "pni-den" in set(out2.sites["pni_id"])
    assert out2.sites[out2.sites["pni_id"] == "pni-den"].iloc[0]["feasible_under_no_vps"]


def test_the_legs_are_recomputed_from_coordinates_not_from_the_rounded_km_columns():
    """5e-4 km of rounding is 5e-6 ms, which is enough to flip a boundary case.

    The check block exists so the carried and recomputed forms are visibly the
    same quantity; if a refactor starts reading `vp_to_tg_km` instead, the
    disagreement silently becomes zero and the guarantee is gone.
    """
    sites = _sites([("pni-chi", CHI), ("pni-nyc", NYC)])
    graph = _graph([("vp-a", DEN, "tg-1", NYC, 40.0)], sites)
    # Corrupt the *rounded* carried column; the test must be unaffected by it.
    graph.edges["vp_to_tg_km"] = 999_999.0
    out = F.build_feasibility(graph)

    assert out.pairs.iloc[0]["vp_to_tg_km"] == pytest.approx(_km(DEN, NYC), abs=1e-3)
    assert out.meta["checks"]["max_abs_km_disagreement_vs_pni_edges"] > 1000


def test_chunking_over_pairs_does_not_change_any_pair_verdict():
    """Every wrong answer here is still a plausible site, so a boundary bug is silent.

    `test_pni.py` pins the argmin for the same reason; the feasibility pass runs
    the same chunk loop over a different reduction.
    """
    sites = _sites([("pni-atl", ATL), ("pni-chi", CHI), ("pni-den", DEN), ("pni-lax", LAX)])
    rows = [
        (f"vp-{i}", DEN, f"tg-{i % 3}", NYC, 20.0 + i) for i in range(37)
    ]
    graph = _graph(rows, sites)
    whole = F.build_feasibility(graph)

    original = pni._ARGMIN_CHUNK_CELLS
    try:
        pni._ARGMIN_CHUNK_CELLS = len(sites)  # one pair per chunk
        chunked = F.build_feasibility(graph)
    finally:
        pni._ARGMIN_CHUNK_CELLS = original

    assert chunked.meta["checks"]["chunk_pairs"] == 1
    pd.testing.assert_frame_equal(whole.pairs, chunked.pairs)


def test_the_decoy_null_is_reproducible_and_carries_the_observation_beside_it():
    """A null whose value moves between runs cannot be quoted next to a number."""
    sites = _sites([("pni-chi", CHI), ("pni-den", DEN), ("pni-nyc", NYC)])
    rows = [("vp-a", DEN, "tg-1", DEN, 4.0, True), ("vp-b", CHI, "tg-1", DEN, 30.0)]
    a = F.build_feasibility(_graph(rows, sites), decoy_trials=5)
    b = F.build_feasibility(_graph(rows, sites), decoy_trials=5)

    pd.testing.assert_frame_equal(a.null, b.null)
    assert (a.null["trial"] == -1).sum() >= 1, "the observation must be a row too"
    assert a.null[a.null["trial"] >= 0]["trial"].nunique() == 5
    assert a.meta["null_model"]["decoy_trials"] == 5
    assert a.meta["null_model"]["decoy_seed"] == F._DECOY_SEED


def test_running_without_a_null_says_so_rather_than_reporting_a_bare_number():
    """With few sites a singleton feasible set is as much about list sparsity as physics."""
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    out = F.build_feasibility(_graph([("vp-a", DEN, "tg-1", DEN, 3.0)], sites))

    assert out.meta["null_model"]["decoy_trials"] == 0
    assert "no null" in out.meta["null_model"]["warning"]


def test_a_synthetic_site_list_is_flagged_on_the_artifact():
    """Every number here is conditional on one small file; a fabricated one must say so."""
    sites = _sites([("pni-den", DEN)])
    graph = _graph([("vp-a", DEN, "tg-1", DEN, 3.0)], sites)
    graph.meta["inputs"]["pni_csv"] = "datasets/pni/SYNTHETIC-us-carrier-hotels.csv"
    out = F.build_feasibility(graph)

    assert out.meta["inputs"]["pni_site_list_is_synthetic"] is True


def test_an_edge_naming_a_site_absent_from_the_node_frame_is_refused():
    """Silently dropping it would change the denominator of every share."""
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    graph = _graph([("vp-a", DEN, "tg-1", DEN, 3.0)], sites)
    graph.edges["sel_pni_id"] = "pni-nowhere"

    with pytest.raises(ValueError, match="absent from pni_nodes"):
        F.build_feasibility(graph)


def test_loading_a_missing_artifact_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(MissingArtifactError, match="build-pni-feasibility"):
        F.load_feasibility(tmp_path)


def test_a_written_artifact_round_trips(tmp_path):
    """`is_sping_vp` and the other booleans have to survive the CSV round trip."""
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    rows = [("vp-a", DEN, "tg-1", DEN, 4.0, True), ("vp-b", CHI, "tg-1", DEN, 30.0)]
    out = F.build_feasibility(_graph(rows, sites), decoy_trials=2)
    out.write(tmp_path)
    back = F.load_feasibility(tmp_path)

    assert len(back.pairs) == len(out.pairs)
    assert back.meta["k"]["k_ms_per_km"] == THEORETICAL_SLOPE
    assert (tmp_path / F.META_JSON).read_text().endswith("\n")


def test_the_cli_refuses_both_run_id_and_all_runs():
    from scripts.analysis.v3.cli import app
    from typer.main import get_command
    from click.testing import CliRunner

    result = CliRunner().invoke(
        get_command(app), ["build-pni-feasibility", "--run-id", "x", "--all-runs"]
    )
    assert result.exit_code != 0
    assert "exactly one of --run-id or --all-runs" in result.output


def test_the_command_declares_no_grid_options():
    """Grid-free by design: no seed or answer space enters, so a sweep would
    write N byte-identical copies. `config.default_map` filters `common:` keys
    against a command's declared params, so this is also what keeps the shipped
    configs' `common: {grid, resolution}` from reaching it."""
    from scripts.analysis.v3.cli import app
    from typer.main import get_command

    cmd = get_command(app).commands["build-pni-feasibility"]
    names = {p.name for p in cmd.params}
    assert not names & {"grid", "resolution", "sweep"}
    assert "decoy_trials" in names
