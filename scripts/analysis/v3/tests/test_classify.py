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


def _frame(space, preds, statuses, tg_seed_ids, target_ids=None):
    """Score `preds` against `space`.

    `target_ids` default to real ids from the answer space: `error_to_target_km`
    is measured against the target's own coordinate, so a synthetic id has no
    ground truth to measure against and `_seed_distance_frame` rejects it.
    """
    n = len(preds)
    if target_ids is None:
        target_ids = list(space.assignments["target_id"])[:n]
    return _seed_distance_frame(
        space,
        method="m",
        target_id=pd.Series(list(target_ids)),
        fold=pd.Series([0] * n),
        status=pd.Series(statuses),
        pred_lat=pd.Series([p[0] for p in preds]),
        pred_lon=pd.Series([p[1] for p in preds]),
        tg_seed_id=pd.Series(tg_seed_ids),
    )


def test_distance_emitted_to_every_seed():
    space = _space()
    df = _frame(space, [CHI], ["SUCCESS"], [space.assignments.iloc[0]["seed_id"]])
    dist_cols = [c for c in df.columns if c.startswith("dist_km__seed_")]
    assert len(dist_cols) == space.n_seeds


def test_exact_hit_ranks_the_true_seed_first():
    """A perfect prediction is top-1 correct, but is *not* zero from the seed.

    Seeds are cell centres, so landing exactly on the target still leaves the
    target's own quantization offset between the prediction and the seed. That
    floor is why `error_to_target_km`, not this column, is the error metric.
    """
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [CHI], ["SUCCESS"], [truth])
    assert df.loc[0, "tg_seed_rank"] == 0
    assert df.loc[0, "pred_seed_id"] == truth
    offset = space.assignments.set_index("target_id").loc["tg-chi", "cell_offset_km"]
    assert df.loc[0, "error_to_tg_seed_km"] == pytest.approx(offset, abs=1e-3)


def test_rank_counts_strictly_closer_seeds():
    """Predicting at NYC while the truth is Chicago must rank Chicago behind NYC."""
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [NYC], ["SUCCESS"], [truth])
    assert df.loc[0, "tg_seed_rank"] >= 1
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
    # Both rows are geometrically perfect — both land on the true seed's cell,
    # each at that cell's own quantization offset from the centre.
    assert (df["tg_seed_rank"] == 0).all()
    offset = space.assignments.set_index("target_id").loc["tg-chi", "cell_offset_km"]
    assert df["error_to_tg_seed_km"].to_numpy() == pytest.approx(
        [offset, offset], abs=1e-3
    )
    # ...but the fallback must not be credited: 1 of 2, not 2 of 2.
    s = topn_summary({"m": df}, ns=(1,)).iloc[0]
    assert s["accuracy_top1"] == pytest.approx(0.5)
    assert s["fallback_rate"] == pytest.approx(0.5)
    assert s["n_targets"] == 2
    assert s["n_solved"] == 1


def test_baseline_rows_are_all_treated_as_solved():
    """Shortest-Ping has no fit or fallback, so every row counts."""
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [CHI, NYC], ["BASELINE", "BASELINE"], [truth, truth])
    s = topn_summary({"m": df}, ns=(1,)).iloc[0]
    assert s["n_solved"] == 2
    assert s["n_fallback"] == 0
    # One of the two predictions is exact, the other is a different metro.
    assert s["accuracy_top1"] == pytest.approx(0.5)


def test_summary_has_no_success_only_columns():
    """The `_success_only` family was removed; `fallback_rate` carries the cost."""
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [CHI, CHI], ["SUCCESS", "FALLBACK"], [truth, truth])
    cols = set(topn_summary({"m": df}, ns=(1, 3)).columns)
    assert not any(c.endswith("_success_only") for c in cols)
    assert {"error_km_p50", "error_km_p90"} <= cols


def test_error_km_excludes_fallback_rows():
    """`error_km_p*` drops the suffix but keeps the solved-only denominator.

    A FALLBACK row's coordinate is the Shortest-Ping VP's, so its error is the
    baseline's, not a CBG error. Here the fallback is wildly wrong and the
    SUCCESS row is exact; the reported p50 must be the exact one.
    """
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    # Both rows predict Chicago, but the FALLBACK row's target is San Jose, so
    # its error is ~2,800 km. Only the exact SUCCESS row may reach the summary.
    df = _frame(
        space,
        [CHI, CHI],
        ["SUCCESS", "FALLBACK"],
        [truth, truth],
        target_ids=["tg-chi", "tg-sjc"],
    )
    assert df.loc[1, "error_to_target_km"] > 2_000.0
    s = topn_summary({"m": df}, ns=(1,)).iloc[0]
    assert s["error_km_p50"] == pytest.approx(0.0, abs=1e-6)


def test_missing_prediction_stays_in_the_denominator():
    space = _space()
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    df = _frame(space, [(np.nan, np.nan)], ["ERROR"], [truth])
    assert df.loc[0, "tg_seed_rank"] == -1
    assert df.loc[0, "pred_seed_id"] == -1
    s = topn_summary({"m": df}, ns=(1,)).iloc[0]
    assert s["n_targets"] == 1
    assert s["accuracy_top1"] == pytest.approx(0.0)


def test_unknown_tg_seed_is_rejected():
    """Predictions and answer space must agree on the seed set."""
    space = _space()
    with pytest.raises(ValueError, match="absent from the answer space"):
        _frame(space, [CHI], ["SUCCESS"], [999])


def test_default_topn_is_one_and_three():
    """Only top-1 and top-3 are reported; the parquet still supports any N."""
    assert DEFAULT_TOPN == (1, 3)


def test_error_to_target_is_measured_from_the_raw_target():
    """The error metric must not route through the seed.

    Chicago and Chicago-plus-a-nudge share a cell, so their seed is the centroid
    *between* them and sits on neither. A prediction landing exactly on one
    target therefore has zero error but a non-zero distance to its own seed —
    which is precisely the quantization offset that must stay out of the error
    figure.
    """
    # 0.05 deg keeps both Chicago targets inside one h3 res-4 cell (0.10 splits
    # them), so their seed is the midpoint and lies on neither target.
    nudge = (CHI[0] + 0.05, CHI[1] + 0.05)
    space = build_answer_space(
        pd.DataFrame(
            {
                "target_id": ["tg-chi", "tg-chi2", "tg-sjc"],
                "target_lat": [CHI[0], nudge[0], SJC[0]],
                "target_lon": [CHI[1], nudge[1], SJC[1]],
            }
        )
    )
    truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
    assert int((space.assignments["seed_id"] == truth).sum()) == 2, "need a shared cell"

    df = _frame(space, [CHI], ["SUCCESS"], [truth], target_ids=["tg-chi"])
    assert df.loc[0, "error_to_target_km"] == pytest.approx(0.0, abs=1e-6)
    assert df.loc[0, "error_to_tg_seed_km"] > 1.0


def test_error_to_target_ignores_the_seed_position_entirely():
    """Re-quantizing must move accuracy's geometry but not the error distance.

    The same prediction scored against a coarse and a fine grid gets different
    seeds, hence a different `error_to_tg_seed_km` — but `error_to_target_km`
    is a property of the prediction and the target alone and must be identical.
    """
    # At this offset res 4 keeps the two Chicago targets in separate cells (so
    # tg-chi's seed sits exactly on it) while res 3 merges them (so the seed
    # moves to their midpoint). Same prediction, two different seeds.
    targets = pd.DataFrame(
        {
            "target_id": ["tg-chi", "tg-chi2", "tg-sjc"],
            "target_lat": [CHI[0], CHI[0] + 0.10, SJC[0]],
            "target_lon": [CHI[1], CHI[1] + 0.10, SJC[1]],
        }
    )
    pred = (CHI[0] + 0.4, CHI[1] - 0.3)
    errs, seed_errs = [], []
    for res in (4, 3):
        space = build_answer_space(targets, grid="h3", resolution=res)
        truth = space.assignments.set_index("target_id")["seed_id"]["tg-chi"]
        df = _frame(space, [pred], ["SUCCESS"], [truth], target_ids=["tg-chi"])
        errs.append(df.loc[0, "error_to_target_km"])
        seed_errs.append(df.loc[0, "error_to_tg_seed_km"])
    assert errs[0] == pytest.approx(errs[1], abs=1e-6)
    assert seed_errs[0] != pytest.approx(seed_errs[1], abs=1e-6)


def test_scoring_a_target_outside_the_answer_space_is_rejected():
    """Silently emitting NaN error for an unknown target would hide a mismatch."""
    space = _space()
    truth = space.assignments.iloc[0]["seed_id"]
    with pytest.raises(ValueError, match="absent from the answer space"):
        _frame(space, [CHI], ["SUCCESS"], [truth], target_ids=["not-a-target"])
