"""Scoring invariants: rank derivation, top-N, and the fallback policy (§7.2/§8.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import build_answer_space
from scripts.analysis.v3.modules.classify import (
    DEFAULT_TOPN,
    _seed_distance_frame,
    topn_summary,
)

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)


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


def _frame(space, preds, statuses, truth_seed_ids):
    n = len(preds)
    return _seed_distance_frame(
        space,
        method="m",
        target_id=pd.Series([f"t{i}" for i in range(n)]),
        fold=pd.Series([0] * n),
        status=pd.Series(statuses),
        pred_lat=pd.Series([p[0] for p in preds]),
        pred_lon=pd.Series([p[1] for p in preds]),
        truth_seed_id=pd.Series(truth_seed_ids),
    )


def test_distance_emitted_to_every_seed():
    space = _space()
    df = _frame(space, [CHI], ["SUCCESS"], [space.assignments.iloc[0]["seed_id"]])
    dist_cols = [c for c in df.columns if c.startswith("dist_km__seed_")]
    assert len(dist_cols) == space.n_seeds


def test_exact_hit_ranks_the_true_seed_first():
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [CHI], ["SUCCESS"], [truth])
    assert df.loc[0, "truth_seed_rank"] == 0
    assert df.loc[0, "pred_seed_id"] == truth
    assert df.loc[0, "error_to_truth_seed_km"] == pytest.approx(0.0, abs=1e-6)


def test_rank_counts_strictly_closer_seeds():
    """Predicting at NYC while the truth is Chicago must rank Chicago behind NYC."""
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [NYC], ["SUCCESS"], [truth])
    assert df.loc[0, "truth_seed_rank"] >= 1
    assert df.loc[0, "pred_seed_id"] != truth


def test_topn_accuracy_is_monotone_in_n():
    """Top-N is derivable from rank alone, and can only improve with N."""
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]
    df = _frame(
        space,
        [CHI, NYC, SJC],
        ["SUCCESS"] * 3,
        [truth["tg-chi"], truth["tg-chi"], truth["tg-nyc"]],
    )
    summary = topn_summary({"m": df}, ns=(1, 2, 3))
    accs = [summary.loc[0, f"accuracy_top{n}"] for n in (1, 2, 3)]
    assert accs == sorted(accs)
    assert accs[-1] == pytest.approx(1.0)


def test_fallback_rows_keep_distances_but_never_count_as_correct():
    """§7.2: a fallback carries a coordinate, yet must score as a failure.

    The raw frame stays neutral (distances present); only the summary applies
    the policy.
    """
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [CHI, CHI], ["SUCCESS", "FALLBACK"], [truth, truth])
    # Both rows are geometrically perfect...
    assert (df["truth_seed_rank"] == 0).all()
    assert df["error_to_truth_seed_km"].to_numpy() == pytest.approx([0.0, 0.0], abs=1e-6)
    # ...but the fallback must not be credited.
    s = topn_summary({"m": df}, ns=(1,)).iloc[0]
    assert s["accuracy_top1"] == pytest.approx(0.5)
    assert s["accuracy_top1_success_only"] == pytest.approx(1.0)
    assert s["fallback_rate"] == pytest.approx(0.5)
    assert s["n_targets"] == 2


def test_baseline_rows_are_all_treated_as_solved():
    """Shortest-Ping has no fit or fallback, so every row counts."""
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [CHI, NYC], ["BASELINE", "BASELINE"], [truth, truth])
    s = topn_summary({"m": df}, ns=(1,)).iloc[0]
    assert s["n_solved"] == 2
    assert s["accuracy_top1"] == pytest.approx(s["accuracy_top1_success_only"])


def test_missing_prediction_stays_in_the_denominator():
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [(np.nan, np.nan)], ["ERROR"], [truth])
    assert df.loc[0, "truth_seed_rank"] == -1
    assert df.loc[0, "pred_seed_id"] == -1
    s = topn_summary({"m": df}, ns=(1,)).iloc[0]
    assert s["n_targets"] == 1
    assert s["accuracy_top1"] == pytest.approx(0.0)


def test_unknown_truth_seed_is_rejected():
    """Predictions and answer space must agree on the seed set."""
    space = _space()
    with pytest.raises(ValueError, match="absent from the answer space"):
        _frame(space, [CHI], ["SUCCESS"], [999])


def test_default_topn_starts_at_one():
    assert DEFAULT_TOPN[0] == 1
