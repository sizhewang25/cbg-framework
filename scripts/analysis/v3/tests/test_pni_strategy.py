"""Invariants for detecting which site-selection policy a peer uses.

The module's job is to replace an assumption with a measurement, so the tests
that matter are the ones that stop it from manufacturing one. Three families.

**It finds the rule that generated the data.** A world built from the target's
nearest site must be detected as `tg_nearest`, and one built from the VP's as
`vp_nearest`, or the detector is not measuring anything.

**It refuses when it cannot tell.** Where the rules pick the same site there is
nothing to separate, so the unit abstains by construction; and an ASN is named
only when its verdict is both concentrated and reproducible across the split.
The second gate is the load-bearing one — a clean majority of verdicts each
resting on a Spearman margin of 0.005 is what the synthetic site list actually
produces, and only the split exposes it.

**Ranks are invariant to the additive leg.** That is the whole reason this is a
rank test: the constant leg and the unit's access floor drop out with no floor
model and nothing fitted. If a per-unit offset ever moves a verdict, the
estimator has stopped being the one documented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import pni, pni_strategy as S
from scripts.analysis.v3.modules.answer_space import pairwise_km
from scripts.analysis.v3.modules.paths import MissingArtifactError
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

CHI = (41.8534, -87.6180)
NYC = (40.7178, -74.0090)
DEN = (39.7436, -104.9903)
LAX = (34.0489, -118.2570)
SEA = (47.6146, -122.3390)
ATL = (33.7556, -84.3915)

ASN = 15169
#: 20 VPs per target, so both halves of the split clear `_MIN_OBS`.
N_VPS = 20


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


#: Sites and endpoints deliberately offset from each other: if a target sits on
#: its own nearest site the routing axes collapse onto `air` and no policy is
#: distinguishable from any other.
SITES = _sites([("pni-atl", ATL), ("pni-chi", CHI), ("pni-lax", LAX), ("pni-nyc", NYC)])

#: 20 targets, because the per-VP grouping sees one observation per target and
#: the split halves that: three targets would leave every VP underpowered and
#: the VP-side verdict permanently undetermined.
TARGETS = {
    f"tg-{i}": (float(lat), float(lon))
    for i, (lat, lon) in enumerate(
        zip(np.linspace(31.5, 46.5, 20), np.linspace(-119.0, -76.0, 20))
    )
}


def _vp_grid(n=N_VPS):
    """VPs spread over a wide box so their nearest sites differ."""
    lats = np.linspace(30.0, 48.0, n)
    lons = np.linspace(-122.0, -72.0, n)
    return [(f"vp-{i}", (float(a), float(b))) for i, (a, b) in enumerate(zip(lats, lons))]


def _nearest(point, sites):
    d = [(_km(point, (r.pni_lat, r.pni_lon)), r.pni_id, (r.pni_lat, r.pni_lon))
         for r in sites.itertuples()]
    return min(d)[2]


def _world(rule, *, vp_floor=0.0, tg_floor=0.0, sites=SITES, targets=None):
    """min-RTT generated exactly by one policy, with optional per-unit floors."""
    targets = targets or TARGETS
    rows = []
    for vp_id, vp in _vp_grid():
        p_vp = _nearest(vp, sites)
        for tg_id, tg in targets.items():
            p_tg = _nearest(tg, sites)
            if rule == "tg_nearest":
                path = _km(vp, p_tg) + _km(p_tg, tg)
            elif rule == "vp_nearest":
                path = _km(vp, p_vp) + _km(p_vp, tg)
            elif rule == "air":
                path = _km(vp, tg)
            else:
                raise ValueError(rule)
            rows.append(
                (vp_id, vp, tg_id, tg, tg_floor + vp_floor + THEORETICAL_SLOPE * path)
            )
    return pd.DataFrame(
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


def test_a_world_built_from_the_targets_nearest_site_is_detected_as_such():
    """The detector has to find the rule that generated the data, or it is
    measuring nothing. Every target's correlation with the tg-nearest predictor
    is exact here, so a failure is a wiring bug rather than a power problem."""
    out = S.build_strategy(_world("tg_nearest"), SITES)
    t = out.target_verdicts

    # Every determinate target names the planted rule; the rest abstain, which
    # is what happens when a target sits on its own nearest site and `air` and
    # `routing_tg_nearest` become the same column.
    determinate = t[t["verdict"] != S.UNDETERMINED]
    assert len(determinate) > 0
    assert set(determinate["verdict"]) == {"tg_nearest"}
    assert t["rho_routing_tg_nearest"].to_numpy() == pytest.approx(1.0, abs=1e-9)
    assert out.strategy == "tg_nearest"
    assert out.meta["details"]["asn_verdict"]["is_declared"]


def test_a_world_built_from_the_vps_nearest_site_is_detected_as_such():
    """The other pure policy, which the per-VP grouping is there to catch."""
    out = S.build_strategy(_world("vp_nearest"), SITES)

    assert set(out.target_verdicts["verdict"]) <= {"vp_nearest", S.UNDETERMINED}
    assert set(out.vp_verdicts["verdict"]) <= {"vp_nearest", S.UNDETERMINED}
    assert out.strategy == "vp_nearest"


def test_a_world_with_no_pni_on_the_path_is_detected_as_direct():
    """`air` is a real hypothesis, not an "other" bucket -- on a well-peered
    dataset it may be the right answer, and it has to be nameable."""
    out = S.build_strategy(_world("air"), SITES)

    assert set(out.target_verdicts["verdict"]) <= {"direct_no_pni", S.UNDETERMINED}
    assert out.strategy == "direct_no_pni"
    assert "direct_no_pni" in S.VERDICTS


def test_a_per_unit_additive_floor_does_not_move_the_verdict():
    """The whole reason this is a rank test.

    An additive per-unit constant -- the fixed leg of the policy, and the
    unit's access floor -- cannot reorder that unit's observations, so it drops
    out with no floor model and nothing fitted. If an offset ever moves a
    verdict, the estimator is no longer the documented one.
    """
    plain = S.build_strategy(_world("tg_nearest"), SITES)
    floored = S.build_strategy(_world("tg_nearest", tg_floor=25.0), SITES)

    pd.testing.assert_series_equal(
        plain.target_verdicts["verdict"], floored.target_verdicts["verdict"]
    )
    assert floored.strategy == plain.strategy


def test_a_unit_whose_rules_never_disagree_abstains_by_construction():
    """Not a margin failing a cutoff: the three predictors are one column.

    With a single site every rule selects it, so no correlation can separate
    them however clean the data is.
    """
    one = _sites([("pni-chi", CHI)])
    out = S.build_strategy(_world("tg_nearest", sites=one), one)

    assert (out.target_verdicts["n_obs_rules_differ"] == 0).all()
    assert set(out.target_verdicts["verdict"]) == {S.UNDETERMINED}
    assert out.meta["details"]["by_target"]["undetermined_share"] == 1.0
    assert out.strategy == S.MIXED


def test_an_asn_is_not_declared_without_an_outright_majority():
    """A plurality on a 40/35/25 split is heterogeneous serving, not a strategy."""
    dist = {
        "shares": {"direct_no_pni": 0.30, "tg_nearest": 0.16,
                   "vp_nearest": 0.18, "argmin_or_mixed": 0.36},
    }
    verdict = S._declare(dist, dist, {"agreement_share": 0.99})

    assert verdict["is_concentrated"] is False
    assert verdict["strategy"] == S.MIXED
    assert verdict["fallback"] == "argmin"


def test_an_asn_is_not_declared_when_the_verdict_does_not_survive_the_split():
    """The load-bearing gate.

    A share can be a clean majority while every per-unit verdict rests on a
    Spearman margin of 0.005 — which is what the synthetic site list produces.
    A margin has no scale of its own to be judged against, so only the holdout
    half can expose it, and chance is the observed class distribution rather
    than a flat prior.
    """
    dist = {"shares": {"direct_no_pni": 0.05, "tg_nearest": 0.85,
                       "vp_nearest": 0.05, "argmin_or_mixed": 0.05}}
    chance = sum(s * s for s in dist["shares"].values())

    good = S._declare(dist, dist, {"agreement_share": chance + 0.2})
    bad = S._declare(dist, dist, {"agreement_share": chance - 0.01})

    assert good["is_concentrated"] and good["reproducibility"]["is_reproducible"]
    assert good["strategy"] == "tg_nearest"
    assert bad["is_concentrated"] and not bad["reproducibility"]["is_reproducible"]
    assert bad["strategy"] == S.MIXED
    assert bad["reproducibility"]["cohens_kappa"] < 0


def test_the_split_halves_every_target_and_is_reproducible():
    """Split within target rather than globally, so a sparse target cannot land
    wholly on one side and vanish from the detect half."""
    e_tg = np.repeat(np.arange(5), 9)
    a = S.half_split(e_tg)
    b = S.half_split(e_tg)

    np.testing.assert_array_equal(a, b)
    per_target = pd.Series(a).groupby(e_tg).sum()
    assert (per_target == 4).all(), "floor(9/2) held out of every target"
    assert S.half_split(e_tg, seed=1).tolist() != a.tolist()


def test_the_pair_split_covers_every_pair_exactly_once():
    """Downstream commands join on it, so a missing or duplicated pair would
    silently re-denominate the study it gates."""
    pairs = _world("tg_nearest")
    out = S.build_strategy(pairs, SITES)

    assert len(out.pair_split) == len(pairs)
    assert not out.pair_split.duplicated(["vp_id", "target_id"]).any()
    assert out.meta["details"]["split"]["n_detect"] + out.meta["details"]["split"]["n_holdout"] == len(pairs)


def test_the_discrimination_budget_is_reported_before_any_verdict():
    """The ceiling on what any comparison can resolve. Where the rules agree the
    three predictors are one column, so this bounds the whole exercise."""
    out = S.build_strategy(_world("tg_nearest"), SITES)
    d = out.meta["details"]["discrimination"]

    assert 0.0 <= d["pairs_rules_differ_share"] <= 1.0
    assert d["n_pairs_rules_differ"] == int(
        d["pairs_rules_differ_share"] * out.meta["inputs"]["n_pairs"] + 0.5
    )


def test_an_underpowered_unit_keeps_its_row_and_is_undetermined():
    """A dropped unit is a silent change of population."""
    pairs = _world("tg_nearest")
    thin = pairs[pairs["target_id"] == "tg-5"].head(6)
    out = S.build_strategy(pd.concat([pairs[pairs["target_id"] != "tg-5"], thin]), SITES)
    row = out.target_verdicts.query("target_id == 'tg-5'").iloc[0]

    assert row["is_underpowered"]
    assert row["verdict"] == S.UNDETERMINED
    assert out.meta["details"]["by_target"]["n_underpowered"] == 1


def test_the_argmin_uses_the_shared_frames_rather_than_rebuilding_them():
    """At a thousand sites a second build is hundreds of megabytes, and a second
    build that sorted differently would silently pick different tie winners."""
    pairs = _world("tg_nearest")
    cols = S.policy_columns(pairs, SITES)
    direct = pni.assign_pni(pairs, SITES)

    np.testing.assert_array_equal(cols["site_of_rule"]["argmin"], direct["sel"])


def test_loading_a_missing_artifact_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(MissingArtifactError, match="detect-pni-strategy"):
        S.load_strategy(tmp_path)


def test_a_written_artifact_round_trips(tmp_path):
    out = S.build_strategy(_world("tg_nearest"), SITES)
    out.write(tmp_path)
    back = S.load_strategy(tmp_path)

    assert back.strategy == out.strategy
    assert len(back.target_verdicts) == len(out.target_verdicts)
    assert (tmp_path / S.META_JSON).read_text().endswith("\n")


def test_the_command_is_grid_free_and_takes_no_all_runs():
    """One run is one peer ASN, so a single --pni-csv cannot describe several."""
    from typer.main import get_command

    from scripts.analysis.v3.cli import app

    cmd = get_command(app).commands["detect-pni-strategy"]
    names = {p.name for p in cmd.params}
    assert not names & {"grid", "resolution", "sweep", "all_runs"}
    assert {"run_id", "pni_csv"} <= names


def test_the_meta_opens_on_the_answer_rather_than_on_provenance():
    """A reader should not have to navigate to find the verdict.

    Four top-level keys: the answer, the tally behind it, the inputs, and
    everything else under `details`. The file is read by a person deciding what
    to pass to `build-pni-graph`, and that decision is two fields deep at most.
    """
    out = S.build_strategy(_world("tg_nearest"), SITES)

    assert set(out.meta) == {"answer", "votes", "inputs", "details", "source"}
    assert set(out.meta["answer"]) == {"strategy", "use_strategy", "why", "checks"}
    assert isinstance(out.meta["answer"]["why"], str)


def test_use_strategy_is_never_mixed_so_a_caller_can_pass_it_straight_through():
    """`mixed` is a verdict, not a strategy name; passing it to
    `build-pni-graph --strategy` would raise. The answer block resolves that
    so no caller has to know it."""
    declared = S.build_strategy(_world("tg_nearest"), SITES)
    one = _sites([("pni-chi", CHI)])
    undecided = S.build_strategy(_world("tg_nearest", sites=one), one)

    assert declared.use_strategy == "tg_nearest"
    assert undecided.strategy == S.MIXED
    assert undecided.use_strategy == "argmin"
    assert undecided.use_strategy in pni.STRATEGIES


def test_the_votes_tally_covers_every_unit_exactly_once():
    """Counts plus cant_tell must add to the unit count, or a verdict has been
    dropped somewhere between the per-unit table and the summary."""
    out = S.build_strategy(_world("tg_nearest"), SITES)
    votes = out.meta["votes"]["by_target"]

    assert sum(votes.values()) == len(out.target_verdicts)
    assert votes["cant_tell"] == int(
        (out.target_verdicts["verdict"] == S.UNDETERMINED).sum()
    )
