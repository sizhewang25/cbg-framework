"""Invariants for the routing-vs-air linearity comparison.

Two families. The first pins the arithmetic a reader will quote: that the
pooled fit is the one `plot-pni-delay` draws, and that
`residual_variance_removed` is the proportional reduction in error it claims to
be. The second pins the two design decisions that keep the comparison honest —
the fixed-site guard axis, without which the argmin's per-pair minimization
partly manufactures the linearity, and the stratification, without which the
paired difference is diluted to zero by the co-located targets that dominate the
population under §7.3's own premise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import figure_pni_delay, pni, pni_linearity as L
from scripts.analysis.v3.modules.answer_space import pairwise_km
from scripts.analysis.v3.modules.bipartite import ols
from scripts.analysis.v3.modules.paths import MissingArtifactError
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

CHI = (41.8534, -87.6180)
NYC = (40.7178, -74.0090)
DEN = (39.7436, -104.9903)
LAX = (34.0489, -118.2570)
SEA = (47.6146, -122.3390)
MIA = (25.7807, -80.2952)

ASN = 15169


def _km(a, b):
    return float(
        pairwise_km(np.array([a[0]]), np.array([a[1]]), np.array([b[0]]), np.array([b[1]]))[0, 0]
    )


def _sites(rows):
    return pd.DataFrame(
        {
            "pni_id": [r[0] for r in rows],
            "pni_lat": [r[1][0] for r in rows],
            "pni_lon": [r[1][1] for r in rows],
            "pni_asn": ASN,
        }
    ).sort_values("pni_id").reset_index(drop=True)


def _graph(pair_rows, sites):
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
    from scripts.analysis.v3.modules.answer_space import elementwise_km

    pairs["sel_pni_id"] = sites["pni_id"].to_numpy()[assign["sel"]]
    pairs["vp_to_tg_via_pni_km"] = assign["via_km"]
    pairs["vp_to_tg_km"] = np.round(
        elementwise_km(
            pairs["vp_lat"].to_numpy(float), pairs["vp_lon"].to_numpy(float),
            pairs["target_lat"].to_numpy(float), pairs["target_lon"].to_numpy(float),
        ),
        pni._KM_DIGITS,
    )
    pairs["is_sping_vp"] = [r[5] if len(r) > 5 else False for r in pair_rows]
    return pni.PniGraph(
        edges=pairs, target_nodes=pd.DataFrame(), pni_nodes=sites.copy(),
        meta={"inputs": {"pni_csv": "test.csv"}},
    )


def _spread(anchor, n, *, step=0.7):
    """`n` points marching north from `anchor`, so a target has VP spread."""
    return [(anchor[0] + i * step, anchor[1]) for i in range(n)]


def test_the_pooled_air_fit_is_the_one_plot_pni_delay_draws():
    """The paper quotes one r-squared and shows the other; they cannot drift.

    Both go through `bipartite.ols`, and this is the assertion that keeps the
    shared definition shared.
    """
    sites = _sites([("pni-chi", CHI), ("pni-den", DEN)])
    rows = [
        (f"vp-{i}", p, "tg-1", NYC, 10.0 + 0.015 * _km(p, NYC))
        for i, p in enumerate(_spread(DEN, 12))
    ]
    out = L.build_linearity(_graph(rows, sites))
    air = out.fits[out.fits["axis"] == "air"].iloc[0]

    x = np.array([_km(p, NYC) for p in _spread(DEN, 12)])
    y = np.array([10.0 + 0.015 * v for v in x])
    slope, intercept, r, r2 = figure_pni_delay._ols(x, y)

    assert air["ols_slope_ms_per_km"] == pytest.approx(slope, abs=1e-9)
    assert air["r2"] == pytest.approx(r2, abs=1e-6)
    assert air["implied_km_per_ms"] == pytest.approx(2.0 / slope, abs=1e-3)


def test_residual_variance_removed_is_the_proportional_reduction_in_error():
    """0.834 to 0.887 must read as 31.9%, not as 0.053.

    The legibility of the headline number depends on this one line of algebra,
    so it is pinned against hand arithmetic rather than against itself.
    """
    fits = pd.DataFrame(
        [
            {"axis": "air", "r2": 0.833983, "implied_km_per_ms": 130.0, "residual_rmse_ms": 7.66},
            {"axis": "routing_selected", "r2": 0.887, "implied_km_per_ms": 122.0, "residual_rmse_ms": 7.0},
            {"axis": "routing_tg_nearest", "r2": 0.80, "implied_km_per_ms": 136.0, "residual_rmse_ms": 8.5},
        ]
    )
    row = L._compare_rows(fits).query("axis_num == 'routing_selected' and axis_den == 'air'").iloc[0]

    assert row["delta_r2"] == pytest.approx(0.053017, abs=1e-6)
    # The emitted value is rounded at 6 dp, like every ratio in this layer.
    assert row["residual_variance_removed"] == pytest.approx(
        1 - (1 - 0.887) / (1 - 0.833983), abs=1e-6
    )
    assert row["residual_variance_removed"] == pytest.approx(0.3193, abs=1e-4)


def test_a_planted_per_target_slope_is_recovered_within_the_target():
    """The paired design is the load-bearing statistic, so its estimator is pinned.

    Each target gets its own access floor here. A pooled fit would be dominated
    by the difference between the floors; the within-target correlation must not
    see them at all.
    """
    sites = _sites([("pni-chi", CHI), ("pni-den", DEN)])
    rows = []
    for t, (anchor, floor) in enumerate([(DEN, 2.0), (SEA, 40.0)]):
        for i, p in enumerate(_spread(anchor, 10)):
            rows.append((f"vp-{t}-{i}", p, f"tg-{t}", NYC, floor + 0.016 * _km(p, NYC)))
    out = L.build_linearity(_graph(rows, sites))

    # An exact affine relation within each target: r == 1 regardless of floor.
    assert out.targets["r_air"].to_numpy() == pytest.approx([1.0, 1.0], abs=1e-9)
    assert not out.targets["is_underpowered"].any()


def test_a_target_at_a_site_has_no_difference_between_the_axes_by_construction():
    """The dilution trap, pinned rather than discovered later.

    When `d(PNI, TG) == 0` the routing axes *are* the air axis, so a zero
    difference there is arithmetic, not evidence. Under §7.3's premise that is
    most targets, which is why the pooled paired median is not the headline.
    """
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    rows = [
        (f"vp-{i}", p, "tg-1", DEN, 2.0 + 0.016 * _km(p, DEN))
        for i, p in enumerate(_spread(SEA, 10))
    ]
    out = L.build_linearity(_graph(rows, sites))
    row = out.targets.iloc[0]

    assert row["tg_to_nearest_pni_km"] == pytest.approx(0.0, abs=1e-3)
    assert row["delta_r_tg_nearest"] == pytest.approx(0.0, abs=1e-6)
    assert row["r_routing_tg_nearest"] == pytest.approx(row["r_air"], abs=1e-9)


def test_the_fixed_site_axis_differs_from_the_argmin_axis_on_a_hairpin():
    """Without the guard, the argmin's per-pair minimum partly manufactures the fit.

    `routing_selected` is a lower envelope over the site list; `routing_tg_nearest`
    fixes the site per target. On a layout where the two rules disagree, the two
    x columns must too, or the guard is not guarding anything.
    """
    sites = _sites([("pni-chi", CHI), ("pni-mia", MIA)])
    # A Seattle target: its nearest site is Chicago, but a Miami VP's argmin is
    # Miami, so the two rules pick different sites for the same pair.
    rows = [("vp-mia", MIA, "tg-1", SEA, 90.0), ("vp-chi", CHI, "tg-1", SEA, 40.0)]
    graph = _graph(rows, sites)
    axes, _, diag = L.axis_columns(graph.edges, graph)

    assert not np.allclose(axes["routing_selected"], axes["routing_tg_nearest"])
    assert diag["tg_to_nearest_pni_km"][0] == pytest.approx(_km(SEA, CHI), abs=1e-3)


def test_the_interaction_statistic_does_not_read_the_bins():
    """The bins are presentation; perturbing them must not move the headline."""
    sites = _sites([("pni-chi", CHI), ("pni-den", DEN), ("pni-mia", MIA)])
    rows = []
    for t, tg in enumerate([DEN, CHI, SEA, LAX, NYC, MIA]):
        for i, p in enumerate(_spread(DEN, 9)):
            rows.append((f"vp-{t}-{i}", p, f"tg-{t}", tg, 3.0 + 0.016 * _km(p, tg)))
    graph = _graph(rows, sites)
    before = L.build_linearity(graph).meta["interaction"]

    original = L._N_PNI_DISTANCE_BINS
    try:
        L._N_PNI_DISTANCE_BINS = 2
        after = L.build_linearity(graph).meta["interaction"]
    finally:
        L._N_PNI_DISTANCE_BINS = original

    key = "spearman_delta_r_selected_vs_tg_to_nearest_pni_km"
    assert after[key] == before[key]


def test_an_underpowered_target_keeps_its_row_with_nan_correlations():
    """A dropped target is a silent change of population.

    The sparse targets are exactly the ones a reader would want to know were
    excluded, so they stay with `is_underpowered` set.
    """
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    rows = [
        ("vp-0", DEN, "tg-thin", NYC, 30.0),
        ("vp-1", SEA, "tg-thin", NYC, 60.0),
    ] + [
        (f"vp-f{i}", p, "tg-fat", NYC, 3.0 + 0.016 * _km(p, NYC))
        for i, p in enumerate(_spread(DEN, 10))
    ]
    out = L.build_linearity(_graph(rows, sites))
    thin = out.targets[out.targets["target_id"] == "tg-thin"].iloc[0]

    assert thin["is_underpowered"]
    assert np.isnan(thin["r_air"]) and np.isnan(thin["rho_air"])
    assert out.meta["checks"]["n_targets_underpowered"] == 1
    assert out.meta["paired"]["n_targets"] == 2
    assert out.meta["paired"]["n_targets_usable"] == 1


def test_the_air_axis_is_recomputed_from_coordinates():
    """Reading the rounded carried column instead would silently lose 5e-4 km."""
    sites = _sites([("pni-chi", CHI), ("pni-den", DEN)])
    rows = [
        (f"vp-{i}", p, "tg-1", NYC, 10.0 + 0.015 * _km(p, NYC))
        for i, p in enumerate(_spread(DEN, 10))
    ]
    graph = _graph(rows, sites)
    graph.edges["vp_to_tg_km"] = 12345.0
    out = L.build_linearity(graph)

    air = out.fits[out.fits["axis"] == "air"].iloc[0]
    assert np.isfinite(air["r2"]) and air["r2"] > 0.9
    assert out.meta["checks"]["max_abs_km_disagreement_vs_pni_edges"] > 1000


def test_group_corr_on_ranks_is_spearman():
    """Both per-target correlations share one implementation and one tie rule."""
    frame = pd.DataFrame(
        {"k": ["a"] * 5 + ["b"] * 5, "x": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
         "y": [2, 1, 4, 3, 5, 1, 2, 3, 4, 5]}
    )
    ranks = frame.groupby("k", sort=True)[["x", "y"]].rank(method="average")
    ranked = pd.concat([frame[["k"]], ranks.add_suffix("__r")], axis=1)
    got = L.group_corr(ranked, "k", "x__r", "y__r")

    from scripts.analysis.v3.modules.bipartite import spearman

    for key in ("a", "b"):
        sub = frame[frame["k"] == key]
        assert got[key] == pytest.approx(spearman(sub["x"], sub["y"]), abs=1e-9)


def test_loading_a_missing_artifact_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(MissingArtifactError, match="compare-pni-linearity"):
        L.load_linearity(tmp_path)


def test_a_written_artifact_round_trips(tmp_path):
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    rows = [
        (f"vp-{i}", p, "tg-1", NYC, 10.0 + 0.015 * _km(p, NYC), i == 0)
        for i, p in enumerate(_spread(DEN, 10))
    ]
    out = L.build_linearity(_graph(rows, sites))
    out.write(tmp_path)
    back = L.load_linearity(tmp_path)

    assert list(back.fits["axis"]) == list(L.AXES)
    assert len(back.targets) == 1
    assert (tmp_path / L.META_JSON).read_text().endswith("\n")


def test_the_command_is_grid_free_and_refuses_both_run_selectors():
    from click.testing import CliRunner
    from typer.main import get_command

    from scripts.analysis.v3.cli import app

    cmd = get_command(app).commands["compare-pni-linearity"]
    assert not {p.name for p in cmd.params} & {"grid", "resolution", "sweep"}

    result = CliRunner().invoke(
        get_command(app), ["compare-pni-linearity", "--run-id", "x", "--all-runs"]
    )
    assert result.exit_code != 0
    assert "exactly one of --run-id or --all-runs" in result.output


def test_holdout_only_scores_the_pairs_the_strategy_was_not_chosen_from():
    """The leakage guard.

    `detect-pni-strategy` picks the site rule by agreement with min-RTT, so
    scoring this study on the same pairs would make "min-RTT tracks routing
    distance" a fit evaluated on itself. The graph carries `is_holdout`; this
    flag is what spends it.
    """
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    rows = [
        (f"vp-{i}", p, "tg-1", NYC, 10.0 + 0.015 * _km(p, NYC))
        for i, p in enumerate(_spread(DEN, 20))
    ]
    graph = _graph(rows, sites)
    graph.edges["is_holdout"] = [i % 2 == 0 for i in range(len(graph.edges))]

    full = L.build_linearity(graph)
    held = L.build_linearity(graph, holdout_only=True)

    assert held.meta["inputs"]["n_pairs"] == 10
    assert held.meta["inputs"]["n_pairs_before_split"] == full.meta["inputs"]["n_pairs"]
    assert held.meta["inputs"]["holdout_only"] is True


def test_holdout_only_refuses_a_graph_that_carries_no_split():
    """Silently scoring on everything would defeat the guard without saying so."""
    sites = _sites([("pni-den", DEN), ("pni-chi", CHI)])
    rows = [
        (f"vp-{i}", p, "tg-1", NYC, 10.0 + 0.015 * _km(p, NYC))
        for i, p in enumerate(_spread(DEN, 10))
    ]
    with pytest.raises(ValueError, match="--split-csv"):
        L.build_linearity(_graph(rows, sites), holdout_only=True)
