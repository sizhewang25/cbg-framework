"""Invariants for crossing method correctness with distance to the nearest PNI.

The claim this module supports is a trend in the *failures*, which is the only
place §8.1's mechanism is observable at all. Three ways that could become an
artifact, and one test each: the trend could be target difficulty rather than a
Shortest-Ping property (so every method is scored over the same strata), it
could be an artefact of where the bin edges fell (so the reported statistic is
continuous and must ignore them), and the population could quietly change (so a
target the PNI graph does not carry is counted rather than dropped).

The remaining tests pin the pipeline self-check. Under nearest-seed snapping
`has_proximate_sping_vp` and `shortest_ping` top-1 correctness are the same
quantity by construction, so their disagreement count is a drift detector — and
per `proximity.py`'s stance it is reported, never raised.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import pni_sping as S
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.paths import MissingArtifactError
from scripts.analysis.v3.modules.proximity import TAXONOMY

OCTANT = "octant_cbg_hull"


def _labels(rows):
    """`rows` is (target_id, sping_correct, pni_km, sping_km, has_proximate_vp)."""
    return pd.DataFrame(
        {
            "target_id": [r[0] for r in rows],
            "tg_seed_id": [f"seed-{i}" for i in range(len(rows))],
            "tg_seed_margin_km": 50.0,
            "tg_seed_nearest_vp_km": 20.0,
            "sping_vp_id": [f"vp-{i}" for i in range(len(rows))],
            "sping_vp_to_tg_km": [r[3] for r in rows],
            "has_proximate_vp": [r[4] for r in rows],
            "has_discriminative_vp": [r[4] for r in rows],
            "has_proximate_sping_vp": [r[1] for r in rows],
            "has_discriminative_sping_vp": [r[1] for r in rows],
            "n_measured_vps": 100,
            "min_inflation": 1.5,
        }
    )


def _nodes(rows, *, drop=()):
    """`target_nodes.csv`'s two columns this module reads."""
    keep = [r for r in rows if r[0] not in drop]
    return pd.DataFrame(
        {
            "target_id": [r[0] for r in keep],
            "tg_nearest_pni_id": "pni-a",
            "tg_to_nearest_pni_km": [r[2] for r in keep],
            "sping_vp_to_tg_km": [r[3] for r in keep],
        }
    )


def _membership(rows, *, extra=None, top_ns=(1,)):
    """One boolean column per method, indexed by target id."""
    idx = pd.Index([r[0] for r in rows], name="target_id")
    cols = {SHORTEST_PING: pd.Series([r[1] for r in rows], index=idx)}
    if extra:
        cols.update({k: pd.Series(v, index=idx) for k, v in extra.items()})
    frame = pd.DataFrame(cols).astype(bool)
    return {n: frame for n in top_ns}


#: Correctness falling off with distance to the nearest site -- §8.1's
#: prediction, planted so the estimator can be checked against a known sign.
_ROWS = [
    ("tg-0", True, 5.0, 10.0, True),
    ("tg-1", True, 12.0, 15.0, True),
    ("tg-2", True, 30.0, 25.0, True),
    ("tg-3", True, 80.0, 40.0, True),
    ("tg-4", False, 300.0, 400.0, True),
    ("tg-5", False, 500.0, 600.0, True),
    ("tg-6", False, 700.0, 800.0, False),
    ("tg-7", False, 900.0, 950.0, False),
]


def _build(rows=_ROWS, *, drop=(), extra=None, labels=None):
    return S.build_breakdown(
        labels if labels is not None else _labels(rows),
        _nodes(rows, drop=drop),
        _membership(rows, extra=extra),
        run_id="run-x",
        grid_meta={"scheme": "h3", "resolution": 4},
    )


def test_the_colocation_rate_equals_shortest_ping_top1_accuracy():
    """The tautology, used as a drift detector rather than reported as a finding.

    Under nearest-seed snapping the baseline predicts its VP's coordinate and
    `has_proximate_sping_vp` asks whether that VP's nearest seed is the target's.
    A nonzero disagreement means proximity's rule and classify's have diverged.
    """
    out = _build()
    colo = out.manifest["colocation"]

    assert colo["answer_region_colocation_rate"] == colo["self_check"][
        "shortest_ping_top1_accuracy"
    ]
    assert colo["self_check"]["n_disagreements_with_shortest_ping_membership"] == 0


def test_a_disagreement_between_proximity_and_membership_is_counted_not_raised():
    """A stack trace would hide the number; the reader needs to see it.

    Same stance as `proximity.py`'s implication violations: reported, so a
    pipeline drift shows up as a count on the artifact.
    """
    labels = _labels(_ROWS)
    labels.loc[0, "has_proximate_sping_vp"] = False  # membership still says True
    out = _build(labels=labels)

    assert out.manifest["colocation"]["self_check"][
        "n_disagreements_with_shortest_ping_membership"
    ] == 1


def test_every_method_is_scored_over_the_same_strata():
    """The control that separates mechanism from target difficulty.

    A trend present in all methods is "targets far from carrier hotels are far
    from everything"; only a trend specific to the baseline is evidence for
    §8.1's mechanism. Both have to be on the same table for a reader to tell.
    """
    flat = [True, False] * 4  # a method whose errors do not track distance
    out = _build(extra={OCTANT: flat})
    top1 = out.accuracy[out.accuracy["top_n"] == 1]

    assert set(top1["method"]) == {SHORTEST_PING, OCTANT}
    strata_of = {m: set(top1[top1["method"] == m]["stratum"]) for m in (SHORTEST_PING, OCTANT)}
    assert strata_of[SHORTEST_PING] == strata_of[OCTANT]
    for method in (SHORTEST_PING, OCTANT):
        sub = top1[top1["method"] == method]
        # The bins partition the targets; the pooled row repeats the whole set.
        assert sub.query("stratum_kind == 'pni_distance_bin'")["n_targets"].sum() == len(_ROWS)
        assert sub.query("stratum == 'pooled'")["n_targets"].iloc[0] == len(_ROWS)

    trend = out.manifest["mechanism"]["error_rate_trend"]["top_1"]
    assert trend[SHORTEST_PING]["spearman_wrong_vs_tg_to_nearest_pni_km"] > 0.8
    assert abs(trend[OCTANT]["spearman_wrong_vs_tg_to_nearest_pni_km"]) < 0.5


def test_the_reported_trend_does_not_read_the_bins():
    """The bins print a table; perturbing them must not move the headline."""
    before = _build().manifest["mechanism"]["error_rate_trend"]["top_1"]

    original = S._N_PNI_DISTANCE_BINS
    try:
        S._N_PNI_DISTANCE_BINS = 2
        after = _build().manifest["mechanism"]["error_rate_trend"]["top_1"]
    finally:
        S._N_PNI_DISTANCE_BINS = original

    key = "spearman_wrong_vs_tg_to_nearest_pni_km"
    assert after[SHORTEST_PING][key] == before[SHORTEST_PING][key]


def test_a_target_the_pni_graph_does_not_carry_is_counted_rather_than_dropped():
    """A silent change of denominator is the failure this module guards against.

    The population stays the one the classification was scored over, so the
    accuracy numbers keep meaning what `classify` said they meant.
    """
    out = _build(drop=("tg-3",))

    assert out.manifest["n_targets"] == len(_ROWS)
    assert out.manifest["n_targets_without_a_pni_row"] == 1
    pooled = out.accuracy.query("method == @SHORTEST_PING and stratum == 'pooled'").iloc[0]
    assert pooled["n_targets"] == len(_ROWS)


def test_the_strata_carry_the_geometry_confounds():
    """`tg_to_nearest_pni_km` correlates with rural, hence with VP sparsity.

    A rising error rate could be that instead of the mechanism, so the confounds
    sit on the same CSV as the claim rather than in a reviewer's question.
    """
    out = _build()
    wanted = {
        "tg_seed_margin_km_p50",
        "tg_seed_nearest_vp_km_p50",
        "has_proximate_vp_share",
        "n_measured_vps_p50",
        "answer_region_colocation_rate",
    }
    assert wanted <= set(out.strata.columns)
    assert out.strata[list(wanted)].notna().all().all()


def test_the_taxonomy_shares_come_from_breakdowns_partition():
    """One definition of the §8.2 terms, so this table cannot disagree with breakdown's."""
    out = _build()
    share_cols = [f"{t}_share" for t in TAXONOMY]

    assert set(share_cols) <= set(out.strata.columns)
    pooled = out.strata[out.strata["stratum"] == "pooled"].iloc[0]
    assert pooled[share_cols].sum() == pytest.approx(1.0, abs=1e-6)


def test_the_ecdf_is_emitted_separately_for_the_correct_and_the_wrong_targets():
    """Pooling the two hides exactly the contrast that makes it a claim."""
    out = _build()
    pops = set(out.ecdf["population"])

    assert pops == set(S.ECDF_POPULATIONS)
    med = out.ecdf[out.ecdf["quantile"] == 0.5].set_index("population")["sping_vp_to_tg_km"]
    assert med["shortest_ping_correct"] < med["shortest_ping_wrong"]
    n = out.ecdf.drop_duplicates("population").set_index("population")["n_targets"]
    assert n["shortest_ping_correct"] + n["shortest_ping_wrong"] == n["all"]


def test_the_two_carried_copies_of_the_sping_distance_are_compared():
    """Both trace to `io.load_sping_vp`, so a disagreement means drift."""
    out = _build()
    assert out.manifest["checks"]["n_sping_vp_to_tg_km_disagreements"] == 0

    rows = [(r[0], r[1], r[2], r[3], r[4]) for r in _ROWS]
    nodes = _nodes(rows)
    nodes.loc[0, "sping_vp_to_tg_km"] = 12345.0
    drifted = S.build_breakdown(_labels(rows), nodes, _membership(rows))
    assert drifted.manifest["checks"]["n_sping_vp_to_tg_km_disagreements"] == 1


def test_the_manifest_is_not_named_meta_json_because_the_directory_is_shared(tmp_path):
    """`target-cls-accuracy/` already holds classify's and breakdown's manifests.

    A `meta.json` here would collide with one of them, and every filename is
    `sping_`-prefixed for the same reason.
    """
    out = _build()
    out.write(tmp_path)

    assert (tmp_path / "sping_pni_manifest.json").exists()
    assert not (tmp_path / "meta.json").exists()
    assert all(p.name.startswith("sping_") for p in tmp_path.iterdir())


def test_the_pooled_baseline_cell_is_flagged_tautological():
    """It equals the co-location rate by construction, so it is a self-check."""
    out = _build()
    flagged = out.accuracy[out.accuracy["is_tautological"]]

    assert len(flagged) == 1
    row = flagged.iloc[0]
    assert (row["method"], row["top_n"], row["stratum"]) == (SHORTEST_PING, 1, "pooled")


def test_loading_a_missing_artifact_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(MissingArtifactError, match="breakdown-sping-pni"):
        S.load_breakdown(tmp_path)


def test_a_written_artifact_round_trips(tmp_path):
    out = _build()
    out.write(tmp_path)
    back = S.load_breakdown(tmp_path)

    assert len(back.accuracy) == len(out.accuracy)
    assert back.manifest["n_targets"] == len(_ROWS)
    assert (tmp_path / S.MANIFEST_JSON).read_text().endswith("\n")


def test_the_command_declares_the_grid_options_because_it_is_answer_space_keyed():
    """Unlike the other two PNI commands: correctness is scored against seeds,
    so the quantization is a parameter of every number here."""
    from typer.main import get_command

    from scripts.analysis.v3.cli import app

    cmd = get_command(app).commands["breakdown-sping-pni"]
    names = {p.name for p in cmd.params}
    assert {"grid", "resolution", "sweep"} <= names
