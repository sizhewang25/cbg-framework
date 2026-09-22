"""Spotter Fig. 3 on a mesh CSV: the fidelity claims, and the drop contract.

Most of these guard a specific defect found by auditing
`scripts/libs/cbg_feasibility/spotter_normality_check.py` against the paper, so
each one names the thing it stops from coming back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import typer
import yaml

from scripts.analysis.v3.modules import figure_spotter_normality as F
from scripts.analysis.v3.modules.config import REPO_ROOT

#: The `spotter_cbg` ltd_kwargs the published operator runs fitted with. The
#: command's defaults must equal these, or the diagnostic describes a model
#: nobody deployed.
DEPLOYED_FIT_KWARGS = {
    "deg_mu": 3,
    "deg_sigma": 2,
    "bin_size_ms": 5.0,
    "cutoff_min_points": 5,
}


def _params():
    from scripts.analysis.v3.cli import app

    return {
        p.name: p
        for p in typer.main.get_command(app).commands["plot-spotter-normality"].params
    }


def test_the_fit_defaults_are_the_kwargs_the_operator_runs_deployed():
    """Read off the config rather than restated here, so the two cannot drift:
    a default that no longer matches `spotter_cbg` would make every number in
    the manifest describe a model that was never scored."""
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs/as01-260728-260802-mesh.yaml").read_text()
    )
    combo = next(
        c for c in cfg["benchmark"]["combos"] if c["combo_id"] == "spotter_cbg"
    )
    assert combo["ltd"] == "normal_dist"
    assert combo["ltd_kwargs"] == DEPLOYED_FIT_KWARGS

    defaults = {k: v.default for k, v in _params().items()}
    for key, want in DEPLOYED_FIT_KWARGS.items():
        assert defaults[key] == want, key
    # `target_coverage` is never exposed: k must stay 1.0 so panel (a)'s band
    # is the paper's mu +/- sigma rather than a coverage-calibrated annulus.
    assert "target_coverage" not in defaults


def test_the_population_defaults_keep_every_row_and_group_by_the_landmark():
    defaults = {k: v.default for k, v in _params().items()}
    # 0 = off. The operator meshes top out near 92 ms, already the paper's
    # regime, so capping at 80 would discard rows for no fidelity gain.
    assert defaults["rtt_max"] == 0.0
    assert defaults["rtt_min"] == 0.0
    # The audit's headline fix: the paper's claim is about the endpoint that
    # PERFORMED the measurement.
    assert defaults["group_by"] == "vp_id"
    assert defaults["n_quantiles"] == 49
    assert defaults["group_pick"] == "spread"
    assert defaults["sigma_ref_km"] == 50.0
    assert defaults["permute_strata"] == "rtt-bin"


# ---- the drop contract ------------------------------------------------------


def _tiny_sigma_fixture():
    """A fit whose sigma is positive but small at the short-RTT end.

    The old fixture made sigma *negative* below 5 ms, which is how as01's
    unconstrained deg-2 polynomial actually behaved. That is unreachable now --
    sigma is fitted as `log sigma`, so `exp` keeps it positive. The drop
    machinery these tests guard is still live through `sigma_ref_km`, so the
    fixture is restated in log space with sigma growing in d:

        log sigma(d) = 0.25*d + 1.6
        -> sigma(1, 2, 4) = 6.4, 8.2, 13.5   (below a 20 km reference)
        -> sigma(6, 10, 20) = 22.2, 60.3, 734
    """
    rtt = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 20.0])
    dist = np.array([50.0, 100.0, 300.0, 600.0, 1000.0, 2000.0])
    p_mu = np.array([0.0, 0.0, 100.0, 0.0])  # mu = 100*d
    p_log_sigma = np.array([0.25, 1.6])
    return rtt, dist, p_mu, p_log_sigma


def test_standardize_returns_a_full_length_array_and_names_every_drop():
    """The regression guard for `spotter_normality_check.standardize`, which
    returns a SHORTER array than its input with no count -- which is how a
    3.9% drop of as01's shortest-distance rows stayed invisible for months."""
    rtt, dist, p_mu, p_log_sigma = _tiny_sigma_fixture()

    s = F.standardize(rtt, dist, p_mu, p_log_sigma, sigma_ref_km=20.0)

    assert len(s.z) == len(rtt)
    assert len(s.valid) == len(rtt)
    assert len(s.reason) == len(rtt)
    assert list(s.valid) == [False, False, False, True, True, True]
    assert np.isnan(s.z[~s.valid]).all()
    assert set(s.reason[~s.valid]) == {F.REASON_SIGMA_BELOW_REF}
    assert s.diagnostics["n_sigma_below_ref"] == 3
    assert s.diagnostics["n_valid"] == 3
    assert s.diagnostics["share_dropped"] == pytest.approx(0.5)


def test_the_drop_is_reported_with_its_bias_not_just_its_size():
    """The dropped rows are the short-RTT, short-distance ones -- the
    near-target pairs a geolocation system is judged on -- so a bare count
    understates what removing them does to every downstream number."""
    rtt, dist, p_mu, p_log_sigma = _tiny_sigma_fixture()

    s = F.standardize(rtt, dist, p_mu, p_log_sigma, sigma_ref_km=20.0)

    assert s.diagnostics["dropped"]["distance_km_p50"] < s.diagnostics["kept"]["distance_km_p50"]
    assert s.diagnostics["dropped"]["rtt_ms_max"] == pytest.approx(4.0)


def test_a_nonpositive_sigma_is_now_only_reachable_by_underflow():
    """Two different objections, kept apart. A non-positive sigma used to mean
    a broken fit; under `log sigma` it can only arise when `exp` underflows to
    exactly zero, which needs log sigma around -745. A positive-but-tiny sigma
    is a different complaint and keeps its own reason code.

    This pins that the NONPOS branch still works -- it is unreachable from a
    real fit, not dead -- and that the two reasons do not collapse into one.
    """
    rtt = np.array([1.0, 2.0, 10.0, 20.0])
    dist = np.array([50.0, 100.0, 1000.0, 2000.0])
    p_mu = np.array([0.0, 0.0, 100.0, 0.0])
    # Deliberately extreme, because underflow is the only remaining route:
    # log sigma = -6d^2 + 180d - 1196 is about -1022 at d=1 and -860 at d=2
    # (both below the ~-745 underflow threshold, so exp is exactly 0.0), and
    # exactly 4 at d=10 and d=20, i.e. sigma = 54.6 km.
    p_log_sigma = np.array([-6.0, 180.0, -1196.0])
    with np.errstate(over="ignore"):
        assert np.exp(np.polyval(p_log_sigma, 1.0)) == 0.0
        assert np.exp(np.polyval(p_log_sigma, 2.0)) == 0.0

    s = F.standardize(rtt, dist, p_mu, p_log_sigma, sigma_ref_km=20.0)

    assert s.diagnostics["n_sigma_nonpos"] == 2
    assert set(s.reason[:2]) == {F.REASON_SIGMA_NONPOS}


# ---- estimators -------------------------------------------------------------


def test_the_three_estimators_agree_on_a_normal_and_split_in_the_known_order():
    """The audit claim that the paper's `1.035` is not comparable to a
    truncated moment rests on these three disagreeing systematically on
    non-normal data. On a true normal they must agree, or the comparison is
    meaningless in both directions."""
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 1.0, 40_000)

    est = F.estimators(z)

    for key in ("sigma_z_moment", "sigma_z_moment_trunc4", "sigma_z_density_ls"):
        assert est[key] == pytest.approx(1.0, abs=0.03), key

    # A tight core with a heavy tail: the moment sees the tail, the truncation
    # removes it, and the density fit follows the core.
    lepto = np.concatenate([rng.normal(0, 0.8, 90_000), rng.normal(0, 3.0, 10_000)])
    est2 = F.estimators(lepto)
    assert est2["sigma_z_moment"] > est2["sigma_z_moment_trunc4"] > est2["sigma_z_density_ls"]
    assert est2["excess_kurtosis_trunc4"] > 0


def test_the_density_fit_recovers_the_papers_published_pair():
    """Executable form of the digitization behind `PAPER`: Fig. 3b's red curve
    peaks at 0.386, which is 1/(1.035*sqrt(2*pi)) and not N(0,1)'s 0.399 --
    the evidence that the paper's (mu, sigma) is a density fit."""
    from scipy.stats import norm

    z = np.random.default_rng(11).normal(F.PAPER["mu_z"], F.PAPER["sigma_z"], 200_000)

    mu, sigma = F.density_ls_fit(z)

    assert mu == pytest.approx(F.PAPER["mu_z"], abs=0.01)
    assert sigma == pytest.approx(F.PAPER["sigma_z"], abs=0.01)
    assert float(norm.pdf(mu, mu, sigma)) == pytest.approx(0.3859, abs=0.002)
    assert float(norm.pdf(0.0, 0.0, 1.0)) == pytest.approx(0.3989, abs=0.002)


# ---- grouping ---------------------------------------------------------------


def _two_population_frame():
    """Two VPs whose z differ, each measuring the same two targets.

    Grouping on the VP separates them; grouping on the target cannot. That
    asymmetry is the whole point of the audit's first item.
    """
    rows = []
    for vp, (lat, lon), bias in (
        ("vp-a", (40.0, -74.0), 0.0),
        ("vp-b", (34.0, -118.0), 900.0),
    ):
        for tg, (tlat, tlon) in (("tg-1", (41.0, -87.0)), ("tg-2", (29.0, -95.0))):
            for i in range(60):
                rows.append(
                    {
                        "vp_id": vp, "target_id": tg,
                        "vp_lat": lat, "vp_lon": lon, "tg_lat": tlat, "tg_lon": tlon,
                        "rtt_ms": 10.0 + i * 0.5,
                        "distance_km": 1000.0 + bias + i,
                    }
                )
    return pd.DataFrame(rows)


def test_grouping_is_on_the_landmark_endpoint_and_coords_collapse_instances():
    df = _two_population_frame()

    assert set(F.group_keys(df, "vp_id")) == {"vp-a", "vp-b"}
    assert set(F.group_keys(df, "target_id")) == {"tg-1", "tg-2"}
    assert F.group_keys(df, "vp_coord").nunique() == 2
    with pytest.raises(typer.BadParameter):
        F.group_keys(df, "src")

    # Two vp_ids at ONE coordinate are one landmark: the paper's landmark is a
    # physical measurement origin, so an id split there invents a distinction.
    twinned = df.copy()
    twinned.loc[twinned["vp_id"] == "vp-b", ["vp_lat", "vp_lon"]] = [40.0, -74.0]
    assert F.group_keys(twinned, "vp_id").nunique() == 2
    assert F.group_keys(twinned, "vp_coord").nunique() == 1


def test_the_landmark_grouping_sees_a_split_the_target_grouping_cannot():
    df = _two_population_frame()
    p_mu = np.array([0.0, 1000.0])  # mu = 1000 km, flat
    p_log_sigma = np.array([0.0, np.log(300.0)])  # sigma = 300 km, flat
    std = F.standardize(df["rtt_ms"].to_numpy(), df["distance_km"].to_numpy(), p_mu, p_log_sigma)
    qs = F.quantile_grid(21)

    by_vp, _, _ = F.landmark_stats(
        df, std, F.group_keys(df, "vp_id"), qs=qs, min_per_group=10
    )
    by_tg, _, _ = F.landmark_stats(
        df, std, F.group_keys(df, "target_id"), qs=qs, min_per_group=10
    )

    # The bias is 3 sigma, so the two VPs' mean z must be ~3 apart while the
    # two targets' are identical -- each target saw both VPs.
    assert abs(np.diff(by_vp["mean_z"].astype(float))[0]) == pytest.approx(3.0, abs=0.1)
    assert abs(np.diff(by_tg["mean_z"].astype(float))[0]) == pytest.approx(0.0, abs=0.1)


def test_every_group_is_measured_and_only_the_named_ones_are_drawn():
    """A top-N report is how a landmark-independence claim gets cherry-picked.
    The CSV must carry the whole population whatever panel (c) draws."""
    rng = np.random.default_rng(3)
    stats = pd.DataFrame(
        {
            "group": [f"g{i:02d}" for i in range(20)],
            "used": True,
            "n_valid": rng.integers(100, 400, 20),
            "qq_slope": np.linspace(0.4, 2.4, 20),
        }
    )

    drawn = F.pick_groups(stats, n=5, how="spread", seed=42)

    assert len(stats) == 20 and len(drawn) == 5
    # min / p25 / median / p75 / max by slope, deterministically.
    assert drawn == ["g00", "g05", "g10", "g14", "g19"]
    assert F.pick_groups(stats, n=5, how="spread", seed=1) == drawn
    assert F.pick_groups(stats, n=3, how="count", seed=42) == (
        stats.nlargest(3, "n_valid")["group"].tolist()
    )


def test_a_group_below_the_minimum_is_measured_but_not_used():
    df = _two_population_frame()
    df = pd.concat([df, df.iloc[:3].assign(vp_id="vp-tiny")], ignore_index=True)
    p_mu = np.array([0.0, 1000.0])
    p_log_sigma = np.array([0.0, np.log(300.0)])
    std = F.standardize(df["rtt_ms"].to_numpy(), df["distance_km"].to_numpy(), p_mu, p_log_sigma)

    stats, _, curves = F.landmark_stats(
        df, std, F.group_keys(df, "vp_id"), qs=F.quantile_grid(21), min_per_group=10
    )

    tiny = stats[stats["group"] == "vp-tiny"].iloc[0]
    assert tiny["n"] == 3 and not tiny["used"]
    assert "vp-tiny" not in curves
    assert set(stats["group"]) == {"vp-a", "vp-b", "vp-tiny"}


# ---- the verdict ------------------------------------------------------------


def test_the_permutation_null_is_seeded_and_isolates_landmark_identity():
    """Why the null is stratified: `mu(d)` has structure in its residuals, so
    a landmark that only ever measures short RTTs differs in mean z from one
    that measures long RTTs even under a perfectly landmark-independent law.
    An unstratified null reads that RTT-coverage effect as a landmark effect.
    """
    rng = np.random.default_rng(5)
    n = 4000
    # z depends on RTT but NOT on the landmark. Groups differ only in which
    # RTTs they cover, so the truth is "landmark-independent".
    rtt = np.concatenate([rng.uniform(1, 20, n // 2), rng.uniform(40, 60, n // 2)])
    keys = np.array(["low"] * (n // 2) + ["high"] * (n // 2))
    z = 0.05 * rtt + rng.normal(0, 1, n)

    strat = F.permutation_null(
        z, keys, rtt, n_permutations=60, bin_size_ms=5.0, strata="rtt-bin", seed=42
    )
    plain = F.permutation_null(
        z, keys, rtt, n_permutations=60, bin_size_ms=5.0, strata="none", seed=42
    )

    assert strat["p_value"] > 0.05, "stratified null must not flag an RTT-coverage effect"
    assert plain["p_value"] <= 0.05, "the unstratified null sees it, which is the point"
    # Seeded: the same call twice is the same answer.
    again = F.permutation_null(
        z, keys, rtt, n_permutations=60, bin_size_ms=5.0, strata="rtt-bin", seed=42
    )
    assert again["p_value"] == strat["p_value"]


def test_eta_squared_is_zero_when_the_landmark_carries_no_information():
    rng = np.random.default_rng(9)
    z = rng.normal(0, 1, 6000)
    none = np.array(["a", "b", "c"] * 2000)
    split = np.where(z > 0, "hi", "lo")

    assert F.eta_squared(z, none) < 0.01
    assert F.eta_squared(z, split) > 0.5


def test_the_goodness_of_fit_reports_a_ratio_not_a_p_value():
    """At n ~ 5*10^4 every p-value is a foregone rejection, so the effect size
    has to be the reported quantity."""
    z = np.random.default_rng(13).normal(0, 1, 20_000)

    gof = F.pooled_gof(z, 0.0, 1.0)

    assert "p_value" not in gof
    assert gof["ks_critical_5pct"] == pytest.approx(1.36 / np.sqrt(20_000), abs=1e-6)
    assert gof["ks_d_over_critical_vs_n01"] < 2.0  # a true normal sits near the line


# ---- panels -----------------------------------------------------------------


def _spy(monkeypatch, drawn: dict):
    real_close = F.plt.close

    def spy(fig):
        ax = fig.axes[0]
        drawn["lines"] = [
            (line.get_label(), line.get_xdata(), line.get_ydata()) for line in ax.lines
        ]
        drawn["collections"] = [c.get_label() for c in ax.collections]
        drawn["xlabel"], drawn["ylabel"] = ax.get_xlabel(), ax.get_ylabel()
        real_close(fig)

    monkeypatch.setattr(F.plt, "close", spy)


def test_panel_b_draws_the_claim_and_the_papers_own_fit(tmp_path, monkeypatch):
    """One curve is not enough. `N(0,1)` alone makes any width error look like
    a modelling failure; the fitted normal alone hides it. The paper's own
    Fig. 3b draws the fitted one, which is what the ClickHouse arm omits."""
    z = np.random.default_rng(17).normal(0.2, 0.9, 20_000)
    est = F.estimators(z)
    drawn: dict = {}
    _spy(monkeypatch, drawn)

    F.build_panel_b(
        z, est, hist_bins=80, z_range=4.0, title=None,
        out_png=tmp_path / "b.png", note="n",
    )

    labels = [lbl for lbl, _, _ in drawn["lines"]]
    assert len(labels) == 2
    assert any("(0,1)" in lbl for lbl in labels)
    fitted = [(lbl, x, y) for lbl, x, y in drawn["lines"] if "fitted" in lbl]
    assert len(fitted) == 1
    # The drawn curve is the reported pair, not a re-fit or the standard normal.
    assert float(np.max(fitted[0][2])) == pytest.approx(
        1.0 / (est["sigma_z_density_ls"] * np.sqrt(2 * np.pi)), rel=0.01
    )
    assert (tmp_path / "b.png").exists()


def test_panel_c_puts_the_pooled_reference_on_the_x_axis(tmp_path, monkeypatch):
    """The orientation guard. `spotter_normality_check.plot_panel_c` draws
    observed on x, where the slope is sigma_pooled/sigma_group -- the inverse
    of what `notes/2026-05-17` reads off it. With pooled on x a group narrower
    than pooled has slope < 1, which is what a reader expects.
    """
    rng = np.random.default_rng(23)
    pooled = rng.normal(0, 1, 20_000)
    qs = F.quantile_grid(49)
    pooled_q = np.quantile(pooled, qs)
    curves = {
        "narrow": np.quantile(rng.normal(0, 0.5, 4000), qs),
        "wide": np.quantile(rng.normal(0, 2.0, 4000), qs),
    }
    core = (qs >= 0.05) & (qs <= 0.95)
    slopes = {
        g: float(np.polyfit(pooled_q[core], q[core], 1)[0]) for g, q in curves.items()
    }
    assert slopes["narrow"] < 1.0 < slopes["wide"]

    stats = pd.DataFrame(
        {
            "group": ["narrow", "wide"], "used": [True, True],
            "n_valid": [4000, 4000], "qq_slope": [slopes["narrow"], slopes["wide"]],
        }
    )
    drawn: dict = {}
    _spy(monkeypatch, drawn)

    F.build_panel_c(
        stats, pooled_q, curves, ["narrow", "wide"],
        group_by="vp_id", title=None, out_png=tmp_path / "c.png", note="n",
    )

    assert drawn["xlabel"].startswith("pooled")
    assert "vp_id" in drawn["ylabel"]
    # The identity line is the only Line2D; the groups are scatter collections
    # so the reader cannot mistake interpolation for data.
    assert len(drawn["lines"]) == 1
    assert len(drawn["collections"]) >= 2


def test_the_fit_carries_no_bin_summary():
    """Panel (a) used to draw per-bin moments beside the polynomial through
    them. There is no binning any more -- mu is fitted to the raw pairs -- so
    `Fit` must not carry, and the figure must not promise, bin moments."""
    rng = np.random.default_rng(29)
    rtt = rng.uniform(1, 80, 5000)
    dist = 45 * rtt + rng.normal(0, 300, 5000)

    fit = F.fit_pooled(
        rtt, np.abs(dist), deg_mu=3, deg_sigma=2,
        bin_size_ms=5.0, cutoff_min_points=5,
    )

    for gone in ("centers", "mus", "sigmas", "counts"):
        assert not hasattr(fit, gone), f"Fit still carries {gone}"
    # fit_rtt is the masked row set, so it may be shorter than the input
    # (the noise pushes some pairs below the 2/3 c line); it must be the same
    # length as fit_dist and non-empty.
    assert 0 < fit.fit_rtt.size <= rtt.size
    assert fit.fit_rtt.size == fit.fit_dist.size
    assert not hasattr(F, "BIN_CSV_SUFFIX")


def test_an_unphysical_row_is_counted_and_excluded_from_the_fit():
    """A sub-2/3-c pair is a data error, and `SpotterRTTModel.fit` drops it
    silently. The count has to survive, and `fit_rtt` has to be the rows the
    polynomials actually saw."""
    rng = np.random.default_rng(31)
    rtt = rng.uniform(10, 80, 2000)
    dist = 45 * rtt
    rtt[0], dist[0] = 1.0, 4000.0  # 4,000 km in 1 ms

    fit = F.fit_pooled(
        rtt, dist, deg_mu=3, deg_sigma=2,
        bin_size_ms=5.0, cutoff_min_points=5,
    )

    assert fit.n_unphysical == 1
    assert fit.fit_rtt.size == rtt.size - 1


# ---- population policy ------------------------------------------------------


def test_the_distance_helper_matches_the_one_the_deployed_model_fits_on():
    """`normal_dist` computes its own distances with `cbg.rtt_model`. If the
    two helpers disagreed, this diagnostic would describe a model fitted on
    different numbers -- and the ClickHouse arm's 6367 km haversine is exactly
    that mistake, at 6 km per 1,000."""
    from scripts.libs.cbg.rtt_model import haversine_distance

    rng = np.random.default_rng(37)
    lat_a, lon_a = rng.uniform(-60, 60, 200), rng.uniform(-180, 180, 200)
    lat_b, lon_b = rng.uniform(-60, 60, 200), rng.uniform(-180, 180, 200)

    mine = F.elementwise_km(lat_a, lon_a, lat_b, lon_b)
    theirs = haversine_distance(lat_a, lon_a, lat_b, lon_b)

    assert float(np.max(np.abs(mine - theirs))) < 1e-6


def test_the_discreteness_counters_come_from_coordinates_not_ids():
    """`f_d(s)` is a density in the paper and a discrete mixture here, and the
    bound is set by distinct COORDINATES -- 20 target sites behind 399 target
    ids on as01. Counting ids would overstate the geometry fivefold."""
    df = pd.DataFrame(
        {
            "vp_id": ["a", "b", "c", "d", "a", "b"],
            "target_id": ["t1", "t2", "t1", "t2", "t2", "t1"],
            "vp_lat": [10.0, 10.0, 20.0, 20.0, 10.0, 10.0],
            "vp_lon": [10.0, 10.0, 20.0, 20.0, 10.0, 10.0],
            "tg_lat": [30.0, 40.0, 30.0, 40.0, 40.0, 30.0],
            "tg_lon": [30.0, 40.0, 30.0, 40.0, 40.0, 30.0],
            "rtt_ms": [1.0] * 6,
            "distance_km": [100.0] * 6,
        }
    )

    d = F.discreteness(df)

    assert d["n_vp_ids"] == 4 and d["n_vp_coords"] == 2
    assert d["n_target_ids"] == 2 and d["n_target_coords"] == 2
    assert d["n_geometry_pairs"] == 4
    assert d["n_geometry_pairs_possible"] == 4
    assert d["geometry_pair_coverage"] == pytest.approx(1.0)


def test_rtt_max_filters_the_population_unlike_plot_distance_rtts_view_cuts(tmp_path):
    """The one deliberate asymmetry with `plot-distance-rtt`, which is read
    beside this figure: there every cut is view-only because clipping x
    attenuates Pearson r. Here the cut has to move the population, because the
    paper's calibration set IS its RTT range and a fit over a wider range is a
    different model."""
    rng = np.random.default_rng(41)
    n = 3000
    rtt = np.concatenate([rng.uniform(5, 70, n), rng.uniform(150, 200, n // 10)])
    # Distance has to come from the COORDINATES -- `load_mesh` derives it and
    # ignores any distance column -- so the VPs are spread along a meridian at
    # ~111.19 km per degree to make distance track RTT.
    dist = np.abs(45 * rtt + rng.normal(0, 200, rtt.size))
    df = pd.DataFrame(
        {
            "vp_id": [f"vp-{i % 20}" for i in range(rtt.size)],
            "target_id": [f"tg-{i % 8}" for i in range(rtt.size)],
            "vp_lat": dist / 111.19, "vp_lon": 0.0,
            "target_lat": 0.0, "target_lon": 0.0,
            "rtt_ms": rtt,
        }
    )
    csv = tmp_path / "frame.csv"
    df.to_csv(csv, index=False)

    loaded, _ = F.load_mesh(csv)
    capped = loaded[loaded["rtt_ms"] <= 80.0]

    unfiltered = F.fit_pooled(
        loaded["rtt_ms"].to_numpy(), loaded["distance_km"].to_numpy(),
        deg_mu=3, deg_sigma=2,
        bin_size_ms=5.0, cutoff_min_points=5,
    )
    filtered = F.fit_pooled(
        capped["rtt_ms"].to_numpy(), capped["distance_km"].to_numpy(),
        deg_mu=3, deg_sigma=2,
        bin_size_ms=5.0, cutoff_min_points=5,
    )

    assert unfiltered.model.rtt_max > 80.0
    assert filtered.model.rtt_max <= 80.0
    # Different populations must give a different model, or the flag is a lie.
    assert not np.allclose(unfiltered.model.p_mu, filtered.model.p_mu)


def test_load_mesh_keeps_the_identities_the_qq_plot_groups_on(tmp_path):
    """`figure_distance_rtt.load_pairs` drops `vp_id`, which is why this
    command has its own loader rather than reusing it."""
    csv = tmp_path / "m.csv"
    pd.DataFrame(
        {
            "vp_id": ["vp-a", "vp-b", "vp-c"],
            "vp_lat": [40.0, 34.0, np.nan],
            "vp_lon": [-74.0, -118.0, -90.0],
            "target_id": ["tg-1", "tg-1", "tg-1"],
            "target_lat": [41.0, 41.0, 41.0],
            "target_lon": [-87.0, -87.0, -87.0],
            "rtt_ms": [10.0, -1.0, 20.0],
        }
    ).to_csv(csv, index=False)

    df, dropped = F.load_mesh(csv)

    # A missing coordinate and a non-positive RTT both go, and are counted.
    assert dropped == 2
    assert list(df["vp_id"]) == ["vp-a"]
    assert {"vp_id", "target_id", "distance_km"} <= set(df.columns)
    assert df["distance_km"].iloc[0] == pytest.approx(1104.0, abs=5.0)
