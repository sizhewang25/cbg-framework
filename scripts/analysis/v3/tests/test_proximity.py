"""The proximity diamond: seed-keyed distances, its implications, the tautology.

The invariants here are the ones every §8.1 stratum rests on. If the half-gap
rule stops implying the argmin rule, or the right chain stops agreeing with
`classify`'s shortest-ping score, the labels still *look* fine and every table
built on them is quietly wrong — so both are pinned rather than inspected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import build_answer_space, pairwise_km
from scripts.analysis.v3.modules.classify import _seed_distance_frame
from scripts.analysis.v3.modules.proximity import (
    FLAGS,
    IMPLICATIONS,
    TAXONOMY,
    build_proximity,
    load_proximity,
    seed_rank_matrix,
)

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
MKE = (43.0389, -87.9065)


def _space():
    return build_answer_space(
        pd.DataFrame(
            {
                "target_id": ["tg-chi", "tg-sjc", "tg-nyc"],
                "target_lat": [CHI[0], SJC[0], NYC[0]],
                "target_lon": [CHI[1], SJC[1], NYC[1]],
            }
        )
    )


def _vps(rows):
    return pd.DataFrame(
        [{"vp_id": i, "vp_lat": la, "vp_lon": lo} for i, la, lo in rows]
    )


def _obs(rows):
    """Canonical-CSV shape: one row per observation."""
    return pd.DataFrame(
        [
            {
                "vp_id": v,
                "vp_lat": vla,
                "vp_lon": vlo,
                "target_id": t,
                "target_lat": tla,
                "target_lon": tlo,
                "rtt_ms": rtt,
            }
            for v, vla, vlo, t, tla, tlo, rtt in rows
        ]
    )


def _eval(rows):
    """`io.load_sping_vp`'s frame — the diamond's right chain, as classify sees it.

    Built here in the shape that helper returns rather than in `eval_source`'s
    raw spelling, so the tests exercise the same contract the CLI does.
    """
    return pd.DataFrame(
        [
            {
                "target_id": t,
                "sping_vp_id": v,
                "sping_vp_lat": la,
                "sping_vp_lon": lo,
                "sping_vp_to_tg_km": np.nan,
                "min_inflation": infl,
            }
            for t, v, la, lo, infl in rows
        ]
    )


def _tiny():
    """One target per metro, one co-located VP each, each VP measuring its own.

    The trivially-good case: every flag should come out True.
    """
    space = _space()
    vps = _vps([("vp-chi", *CHI), ("vp-sjc", *SJC), ("vp-nyc", *NYC)])
    obs = _obs(
        [
            ("vp-chi", *CHI, "tg-chi", *CHI, 1.0),
            ("vp-sjc", *SJC, "tg-sjc", *SJC, 1.0),
            ("vp-nyc", *NYC, "tg-nyc", *NYC, 1.0),
        ]
    )
    ev = _eval(
        [
            ("tg-chi", "vp-chi", *CHI, 1.1),
            ("tg-sjc", "vp-sjc", *SJC, 1.2),
            ("tg-nyc", "vp-nyc", *NYC, 1.3),
        ]
    )
    return space, vps, obs, ev


# ---- the rank rule ----------------------------------------------------------


def test_seed_rank_matrix_gives_rank_zero_to_the_nearest_seed():
    d = np.array([[10.0, 30.0, 20.0], [50.0, 5.0, 40.0]])
    r = seed_rank_matrix(d)
    assert r[0].tolist() == [0, 2, 1]
    assert r[1].tolist() == [2, 0, 1]


def test_rank_ties_resolve_in_the_seeds_favour():
    """A VP exactly equidistant from two seeds ranks *both* 0.

    Matches `classify`'s rule, which is what lets the tautology below be exact
    rather than approximate: a tie must not make the label and the score
    disagree about which class the coordinate is consistent with.
    """
    r = seed_rank_matrix(np.array([[10.0, 10.0, 40.0]]))
    assert r[0].tolist() == [0, 0, 2]


# ---- the diamond ------------------------------------------------------------


def test_a_colocated_measuring_vp_sets_every_flag():
    space, vps, obs, ev = _tiny()
    out = build_proximity(space, vps, obs, ev)
    for flag in FLAGS:
        assert out.labels[flag].all(), flag
    assert (out.labels["tg_seed_best_rank"] == 0).all()
    assert (out.labels["sping_vp_tg_seed_rank"] == 0).all()


def test_implications_hold_on_random_geometry():
    """`4 ⟹ 2 ⟹ 1` and `4 ⟹ 3 ⟹ 1`, over VPs scattered without regard to seeds.

    The half-gap rule is a theorem, not a coincidence, so it must survive VPs
    that were not placed to satisfy it. Random placement is what makes this a
    test of the implication rather than of the fixture.
    """
    rng = np.random.default_rng(20260906)
    space = _space()
    tg = space.assignments
    vps = _vps(
        [
            (f"vp-{i}", float(la), float(lo))
            for i, (la, lo) in enumerate(
                zip(rng.uniform(30, 48, 40), rng.uniform(-124, -70, 40))
            )
        ]
    )
    rows, ev_rows = [], []
    for _, t in tg.iterrows():
        picks = rng.choice(len(vps), size=6, replace=False)
        rtts = rng.uniform(1, 90, size=picks.size)
        for k, rtt in zip(picks, rtts):
            v = vps.iloc[k]
            rows.append(
                (v.vp_id, v.vp_lat, v.vp_lon, t.target_id, t.target_lat, t.target_lon, float(rtt))
            )
        best = vps.iloc[picks[int(np.argmin(rtts))]]
        ev_rows.append((t.target_id, best.vp_id, best.vp_lat, best.vp_lon, 1.5))

    out = build_proximity(space, vps, _obs(rows), _eval(ev_rows))
    for ante, cons in IMPLICATIONS:
        assert not (out.labels[ante] & ~out.labels[cons]).any(), f"{ante} without {cons}"
    assert out.meta["implication_violations"] == {}


def test_half_gap_is_strictly_tighter_than_argmin():
    """A VP nearest its target's seed but outside the half-gap: rank 0, not discriminative.

    Places the VP just past `margin` along the line away from the two other
    metros, so it stays the seed's nearest neighbour while failing the
    guarantee. This is the cell that makes the two axes distinct rather than
    redundant, and its existence is why the diamond is not a chain.
    """
    space = _space()
    seeds = space.seeds.set_index("seed_id")
    chi_seed = int(space.assignments.set_index("target_id").loc["tg-chi", "seed_id"])
    margin = float(seeds.loc[chi_seed, "margin_km"])
    # Due north of the seed: SJC is west and NYC is east, so northward motion
    # increases the distance to both far faster than to CHI.
    lat = float(seeds.loc[chi_seed, "seed_lat"]) + (margin * 1.5) / 111.32
    lon = float(seeds.loc[chi_seed, "seed_lon"])

    vps = _vps([("vp-far", lat, lon)])
    obs = _obs([("vp-far", lat, lon, "tg-chi", *CHI, 5.0)])
    ev = _eval([("tg-chi", "vp-far", lat, lon, 2.0)])
    out = build_proximity(space, vps, obs, ev)
    row = out.labels[out.labels.target_id == "tg-chi"].iloc[0]

    assert row["tg_seed_best_rank"] == 0
    assert row["has_proximate_vp"]
    assert not row["has_discriminative_vp"]
    assert row["tg_seed_nearest_vp_km"] > row["tg_seed_margin_km"]
    assert out.meta["argmin_vs_half_gap"]["n_proximate_but_not_discriminative_vp"] >= 1


def test_the_two_left_chain_columns_can_name_different_vps():
    """Nearest-by-distance and best-by-rank are different minimizations.

    A VP wedged between two adjacent seeds can sit *closer* to the target's seed
    than any other measured VP while still ranking a neighbour first; a VP twice
    as far away, out in an empty region, ranks the target's seed 0. So the two
    ids are emitted separately rather than collapsed into one "closest VP" — the
    diamond's left column asks an existence question, not a distance one.

    Needs two adjacent seeds, hence the Milwaukee target: with three metros a
    thousand kilometres apart no VP can be near one seed and nearer another.
    """
    space = build_answer_space(
        pd.DataFrame(
            {
                "target_id": ["tg-chi", "tg-mke", "tg-sjc", "tg-nyc"],
                "target_lat": [CHI[0], MKE[0], SJC[0], NYC[0]],
                "target_lon": [CHI[1], MKE[1], SJC[1], NYC[1]],
            }
        )
    )
    seeds = space.seeds.set_index("seed_id")
    chi_seed = int(space.assignments.set_index("target_id").loc["tg-chi", "seed_id"])
    mke_seed = int(space.assignments.set_index("target_id").loc["tg-mke", "seed_id"])
    assert chi_seed != mke_seed
    c_lat, c_lon = (float(seeds.loc[chi_seed, "seed_lat"]),
                    float(seeds.loc[chi_seed, "seed_lon"]))
    m_lat, m_lon = (float(seeds.loc[mke_seed, "seed_lat"]),
                    float(seeds.loc[mke_seed, "seed_lon"]))

    # A: 55% of the way to the neighbour -> just past the boundary, rank >= 1.
    a_lat, a_lon = c_lat + 0.55 * (m_lat - c_lat), c_lon + 0.55 * (m_lon - c_lon)
    # B: farther in absolute terms, but away from every other seed -> rank 0.
    b_lat, b_lon = c_lat - 1.4 * abs(m_lat - c_lat), c_lon

    vps = _vps([("vp-a", a_lat, a_lon), ("vp-b", b_lat, b_lon)])
    obs = _obs(
        [
            ("vp-a", a_lat, a_lon, "tg-chi", *CHI, 3.0),
            ("vp-b", b_lat, b_lon, "tg-chi", *CHI, 9.0),
        ]
    )
    ev = _eval([("tg-chi", "vp-a", a_lat, a_lon, 2.0)])
    out = build_proximity(space, vps, obs, ev)
    row = out.labels[out.labels.target_id == "tg-chi"].iloc[0]

    d = pairwise_km(
        np.array([a_lat, b_lat]),
        np.array([a_lon, b_lon]),
        seeds["seed_lat"].to_numpy(),
        seeds["seed_lon"].to_numpy(),
    )
    col = list(seeds.index).index(chi_seed)
    ranks = seed_rank_matrix(d)
    assert d[0, col] < d[1, col]        # A is nearer the target's seed
    assert ranks[0, col] >= 1           # ...but ranks a neighbour first
    assert ranks[1, col] == 0           # B, farther out, ranks it first
    assert row["tg_seed_nearest_vp_id"] == "vp-a"
    assert row["tg_seed_best_rank_vp_id"] == "vp-b"
    assert row["tg_seed_best_rank"] == 0
    assert row["tg_seed_nearest_vp_km"] == pytest.approx(d[0, col], abs=1e-3)


# ---- the tautology ----------------------------------------------------------


def test_sping_flag_equals_classify_shortest_ping_top1():
    """`has_proximate_sping_vp` is Shortest-Ping's top-1 correctness, exactly.

    The self-check the design keeps the flag for. Both sides are computed here
    from the same eval_source coordinate through their own code path — this is
    the assertion that catches the two rank rules drifting apart.
    """
    space = _space()
    vps = _vps([("vp-chi", *CHI), ("vp-nyc", *NYC)])
    # tg-sjc's shortest-ping VP is in Chicago: the baseline is wrong there.
    obs = _obs(
        [
            ("vp-chi", *CHI, "tg-chi", *CHI, 1.0),
            ("vp-chi", *CHI, "tg-sjc", *SJC, 1.0),
            ("vp-nyc", *NYC, "tg-sjc", *SJC, 9.0),
            ("vp-nyc", *NYC, "tg-nyc", *NYC, 1.0),
        ]
    )
    ev = _eval(
        [
            ("tg-chi", "vp-chi", *CHI, 1.0),
            ("tg-sjc", "vp-chi", *CHI, 8.0),
            ("tg-nyc", "vp-nyc", *NYC, 1.0),
        ]
    )
    out = build_proximity(space, vps, obs, ev)

    truth = space.assignments.set_index("target_id")["seed_id"]
    scored = _seed_distance_frame(
        space,
        method="shortest_ping",
        target_id=ev["target_id"],
        fold=pd.Series([0] * len(ev)),
        status=pd.Series(["BASELINE"] * len(ev)),
        pred_lat=ev["sping_vp_lat"],
        pred_lon=ev["sping_vp_lon"],
        tg_seed_id=ev["target_id"].map(truth),
    )
    merged = out.labels.merge(scored, on="target_id", validate="1:1")
    assert (merged["has_proximate_sping_vp"] == (merged["tg_seed_rank"] == 0)).all()
    assert merged["sping_vp_tg_seed_rank"].tolist() == merged["tg_seed_rank"].tolist()
    assert merged["sping_vp_to_tg_seed_km"].to_numpy() == pytest.approx(
        merged["error_to_tg_seed_km"].to_numpy(), abs=1e-3
    )
    # And the case that makes the assertion non-vacuous.
    assert not merged.set_index("target_id").loc["tg-sjc", "has_proximate_sping_vp"]


# ---- edges, denominators and reporting --------------------------------------


def test_a_target_with_no_measured_edge_keeps_its_row():
    """Unmeasured targets stay in the denominator every §8.1 rate is taken over.

    Dropping them would inflate every stratum's accuracy by exactly the targets
    no VP ever looked at, which is the population §8.1 most wants counted.
    """
    space, vps, obs, ev = _tiny()
    out = build_proximity(space, vps, obs[obs.target_id != "tg-nyc"], ev)
    row = out.labels.set_index("target_id").loc["tg-nyc"]
    assert len(out.labels) == 3
    assert row["n_measured_vps"] == 0
    assert row["tg_seed_best_rank"] == -1
    assert not row["has_proximate_vp"]
    assert not row["has_discriminative_vp"]
    assert np.isnan(row["tg_seed_nearest_vp_km"])
    # The right chain is independent of the edge set: eval_source still names a
    # VP, so the baseline's own claim survives.
    assert row["has_proximate_sping_vp"]


def test_edges_outside_the_node_sets_are_dropped_and_counted():
    space, vps, obs, ev = _tiny()
    extra = _obs([("vp-ghost", 0.0, 0.0, "tg-chi", *CHI, 1.0)])
    out = build_proximity(space, vps, pd.concat([obs, extra], ignore_index=True), ev)
    assert out.meta["edges_dropped"]["edges_naming_an_unknown_vp"] == 1
    assert out.labels.set_index("target_id").loc["tg-chi", "n_measured_vps"] == 1


def test_constant_flags_are_named_rather_than_left_as_a_bare_share():
    """A 1.0 share must be reported as "no variance", never as "no effect".

    On as01 and as03 the two left-column flags really are constant; a consumer
    reading only `share` would report that VP proximity does not separate, when
    the correct statement is that there is nothing to separate on.
    """
    space, vps, obs, ev = _tiny()
    out = build_proximity(space, vps, obs, ev)
    assert set(out.meta["zero_variance"]) == set(FLAGS)
    for flag in FLAGS:
        block = out.meta["flags"][flag]
        assert block["n_true"] + block["n_false"] == len(out.labels)
        assert block["share"] == 1.0


def test_taxonomy_block_partitions_the_targets():
    """§8.2's three terms are exclusive and exhaustive — they are a chain cut twice."""
    space = _space()
    vps = _vps([("vp-chi", *CHI), ("vp-nyc", *NYC)])
    obs = _obs(
        [
            ("vp-chi", *CHI, "tg-chi", *CHI, 1.0),
            ("vp-chi", *CHI, "tg-sjc", *SJC, 1.0),
            ("vp-nyc", *NYC, "tg-nyc", *NYC, 1.0),
        ]
    )
    ev = _eval(
        [
            ("tg-chi", "vp-chi", *CHI, 1.0),
            ("tg-sjc", "vp-chi", *CHI, 8.0),
            ("tg-nyc", "vp-nyc", *NYC, 1.0),
        ]
    )
    out = build_proximity(space, vps, obs, ev)
    t = out.meta["section_8_2_taxonomy"]
    assert sum(t[k] for k in TAXONOMY) == out.meta["n_targets"]
    # geometry_only is defined against measured VPs, so a target the baseline
    # got right can never land there — the chain is cut, not partitioned twice.
    assert t["selection_hit"] == int(out.labels["has_proximate_sping_vp"].sum())


def test_the_flags_never_read_a_vp_to_target_distance(tmp_path):
    """Moving a target inside its own cell must not move any flag.

    The whole point of keying to the seed: quantization is what defines the
    class, so a target's offset within its cell is invisible to classification
    and must be invisible here too. If a flag ever tracked `closest_vp_to_tg_km`,
    this is what would catch it.
    """
    space, vps, obs, ev = _tiny()
    base = build_proximity(space, vps, obs, ev)

    shifted = obs.copy()
    moved = shifted.target_id == "tg-chi"
    shifted.loc[moved, "target_lat"] = CHI[0] + 0.05
    out = build_proximity(space, vps, shifted, ev)

    for flag in FLAGS:
        assert out.labels[flag].tolist() == base.labels[flag].tolist()
    assert out.labels["tg_seed_nearest_vp_km"].to_numpy() == pytest.approx(
        base.labels["tg_seed_nearest_vp_km"].to_numpy()
    )


def test_round_trip_through_disk(tmp_path):
    space, vps, obs, ev = _tiny()
    out = build_proximity(space, vps, obs, ev, source_label="unit/test")
    out.write(tmp_path)
    back = load_proximity(tmp_path)
    pd.testing.assert_frame_equal(back.labels, out.labels)
    assert back.meta["source"] == "unit/test"
    assert back.meta["top_n_context"] == 1


def test_sping_reader_is_shared_with_classify():
    """`io.load_sping_vp` is the one resolution of the baseline's VP.

    The tautology is only a usable self-check while `classify` and `proximity`
    read the *same row*. Two readers would agree today and be free to drift
    tomorrow, so this pins that `score_shortest_ping` goes through the helper —
    a regression that reintroduced a local `eval_per_target` read would still
    pass every numeric test above, and would fail here.
    """
    import inspect

    from scripts.analysis.v3.modules import classify as classify_mod
    from scripts.analysis.v3.modules import io as io_mod

    src = inspect.getsource(classify_mod.score_shortest_ping)
    assert "io.load_sping_vp(run)" in src
    assert "shortest_ping_vp_lat" not in src, (
        "classify resolved the baseline's VP itself; it must go through "
        "io.load_sping_vp so proximity's right chain cannot drift from it"
    )
    assert io_mod.SPING_VP_COLUMNS == (
        "target_id",
        "shortest_ping_vp_lat",
        "shortest_ping_vp_lon",
    )
