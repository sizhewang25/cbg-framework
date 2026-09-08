"""`table-headline` — the row plan, the tie rule, and the reserved rows.

Three things here are easy to get quietly wrong and expensive to notice late: a
reserved row printing as `0.000` instead of `—`, a bolded argmax where the gap
is inside the noise, and a weighted run silently landing on the wrong dataset.
Each gets a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import typer

from scripts.analysis.v3.modules import headline_table as H
from scripts.analysis.v3.modules.classify import SHORTEST_PING


class _Run:
    """Stand-in for `RunPaths` — only `run_id` and `setup` are read here."""

    def __init__(self, run_id: str = "run", setup: str = "anchors_to_probes") -> None:
        self.run_id = run_id
        self.setup = setup


def _accuracy_frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal `accuracy_rows(include_counts=True)`-shaped frame.

    The count columns are derived from the rates a case states, rather than
    spelled out in every fixture, so a test says only the thing it is about.
    `classify` guarantees `n_targets == n_solved + n_fallback + n_error`, so the
    derivation keeps the fixtures on the same partition the real files are on.
    """
    filled = []
    for row in rows:
        row = dict(row)
        n = int(row["n_targets"])
        row.setdefault("n_fallback", int(round(row.get("fallback_rate", 0.0) * n)))
        row.setdefault("n_error", 0)
        row.setdefault("n_solved", n - row["n_fallback"] - row["n_error"])
        filled.append(row)
    return pd.DataFrame(filled)


# ---------------------------------------------------------------------------
# the row plan
# ---------------------------------------------------------------------------


def test_the_row_order_is_kind_major_with_the_aggregate_leading():
    plan = H.row_plan({"as01-260728-260802": _Run(), "as02-260728-260802": _Run()}, {})
    assert [(e["kind"], e["scope"], e["dataset"]) for e in plan] == [
        ("mesh", H.AGGREGATE, None),
        ("mesh", H.DATASET, "as01"),
        ("mesh", H.DATASET, "as02"),
        ("weighted", H.AGGREGATE, None),
        ("weighted", H.DATASET, "as01"),
        ("weighted", H.DATASET, "as02"),
    ]
    assert [e["row_index"] for e in plan] == list(range(len(plan)))


def test_a_single_dataset_table_emits_no_aggregate_row():
    """A micro-average over one dataset is that dataset, printed twice."""
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    assert all(e["scope"] == H.DATASET for e in plan)
    assert [(e["kind"], e["dataset"]) for e in plan] == [
        ("mesh", "as01"),
        ("weighted", "as01"),
    ]


def test_an_aggregate_row_carries_the_contributing_run_ids_not_a_single_run_id():
    plan = H.row_plan({"as01-260728-260802": _Run(), "as02-260728-260802": _Run()}, {})
    agg = next(e for e in plan if e["scope"] == H.AGGREGATE and e["kind"] == "mesh")
    assert agg["run_id"] is None
    assert agg["run_ids"] == ["as01-260728-260802", "as02-260728-260802"]
    assert agg["datasets"] == ["as01", "as02"]
    assert agg["n_expected"] == 2


def test_a_dataset_row_is_the_degenerate_one_run_aggregate():
    """`n_expected == 1` is what lets `pending` stay one rule across both scopes."""
    plan = H.row_plan({"as01-260728-260802": _Run(), "as02-260728-260802": _Run()}, {})
    row = next(e for e in plan if e["scope"] == H.DATASET and e["kind"] == "mesh")
    assert row["n_expected"] == 1
    assert row["run_ids"] == [row["run_id"]]


def test_two_mesh_runs_for_one_dataset_are_refused():
    """Silently dropping the second left it out of the rows and in the denominator."""
    with pytest.raises(typer.BadParameter) as exc:
        H.row_plan({"as01-260728-260802": _Run(), "as01-260801-260806": _Run()}, {})
    assert "as01" in str(exc.value)
    assert "one MESH row per dataset" in str(exc.value)


def test_an_aggregate_label_states_its_coverage_rather_than_the_group_size():
    assert H.row_label("mesh", H.AGGREGATE, None, 3, 3) == "MESH (3 ASes)"
    assert (
        H.row_label("weighted", H.AGGREGATE, None, 1, 3)
        == "TRAFFIC-WEIGHTED (1 of 3 ASes)"
    )
    assert (
        H.row_label("weighted", H.AGGREGATE, None, 0, 3)
        == "TRAFFIC-WEIGHTED (0 of 3 ASes)"
    )
    assert H.row_label("mesh", H.DATASET, "as01", 1, 1) == "· AS01"


def test_without_a_group_header_a_breakdown_row_names_its_own_kind():
    """Two rows both reading `· AS01` would name neither campaign."""
    assert (
        H.row_label("weighted", H.DATASET, "as01", 0, 1, grouped=False)
        == "AS01 TRAFFIC-WEIGHTED"
    )
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    assert [e["label"] for e in plan] == ["AS01 MESH", "AS01 TRAFFIC-WEIGHTED"]


def test_a_weighted_row_with_no_data_carries_no_run_id():
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    weighted = next(e for e in plan if e["kind"] == "weighted")
    assert weighted["run_id"] is None
    assert weighted["run_ids"] == []


def test_mesh_run_order_fixes_the_dataset_order():
    plan = H.row_plan(
        {"as03-260728-260802": _Run(), "as01-260728-260802": _Run()}, {}
    )
    breakdown = [e["dataset"] for e in plan if e["scope"] == H.DATASET]
    assert breakdown == ["as03", "as01", "as03", "as01"]


def test_a_weighted_run_fills_the_row_of_the_dataset_it_is_paired_to():
    plan = H.row_plan(
        {"as01-260728-260802": _Run()},
        {"as01": _Run(run_id="as01-weighted-260728")},
    )
    weighted = next(
        e for e in plan if e["kind"] == "weighted" and e["scope"] == H.DATASET
    )
    assert weighted["run_id"] == "as01-weighted-260728"


def test_a_weighted_run_paired_to_an_absent_dataset_is_refused():
    with pytest.raises(typer.BadParameter, match="no mesh run"):
        H.row_plan({"as01-260728-260802": _Run()}, {"as09": _Run()})


# ---------------------------------------------------------------------------
# the pairing is stated, never inferred
# ---------------------------------------------------------------------------


def test_a_pair_spec_splits_into_dataset_and_run():
    assert H.parse_weighted_pairs(["as01=as01-weighted-260728"]) == {
        "as01": "as01-weighted-260728"
    }


def test_whitespace_around_the_equals_is_tolerated():
    assert H.parse_weighted_pairs([" as01 = run-x "]) == {"as01": "run-x"}


@pytest.mark.parametrize("spec", ["as01-weighted-260728", "as01=", "=run-x", ""])
def test_a_spec_without_both_halves_is_refused(spec):
    """A bare run_id is refused rather than guessed at.

    `short_dataset` cannot reduce any plausible weighted run name to its
    dataset, so a fallback would file the row under the wrong dataset instead
    of failing.
    """
    with pytest.raises(typer.BadParameter, match="<dataset>=<run_id>"):
        H.parse_weighted_pairs([spec])


def test_short_dataset_cannot_infer_the_pairing_which_is_why_it_is_explicit():
    from scripts.analysis.v3.modules.cross import short_dataset

    assert short_dataset("as01-weighted-260728") == "as01-weighted-260728"
    assert short_dataset("as01-260728-260802-weighted") == (
        "as01-260728-260802-weighted"
    )
    assert short_dataset("as01w-260728-260802") == "as01w"


def test_two_weighted_runs_for_one_dataset_are_refused():
    with pytest.raises(typer.BadParameter, match="two weighted runs"):
        H.parse_weighted_pairs(["as01=run-a", "as01=run-b"])


def test_repeating_the_identical_pair_is_harmless():
    assert H.parse_weighted_pairs(["as01=run-a", "as01=run-a"]) == {"as01": "run-a"}


# ---------------------------------------------------------------------------
# the tie rule
# ---------------------------------------------------------------------------


def test_the_row_maximum_is_always_marked():
    marks = H.best_in_row(np.array([0.4, 0.7, 0.5]), 400)
    assert marks.tolist() == [False, True, False]


def test_a_near_tie_marks_both_rather_than_picking_a_winner():
    # 0.926 vs 0.926 on 458 targets: se = 0.0122, so both clear 0.9138.
    marks = H.best_in_row(np.array([0.926, 0.926, 0.880]), 458)
    assert marks.tolist() == [True, True, False]


def test_the_threshold_is_the_best_cells_own_standard_error():
    n, top = 458, 0.502
    se = np.sqrt(top * (1 - top) / n)
    inside, outside = top - se + 1e-6, top - se - 1e-6
    marks = H.best_in_row(np.array([top, inside, outside]), n)
    assert marks.tolist() == [True, True, False]


def test_a_smaller_sample_widens_the_band():
    values = np.array([0.50, 0.45])
    assert H.best_in_row(values, 10_000).tolist() == [True, False]
    assert H.best_in_row(values, 78).tolist() == [True, True]


def test_an_all_missing_row_marks_nothing():
    marks = H.best_in_row(np.array([np.nan, np.nan]), 0)
    assert not marks.any()


def test_a_row_with_no_targets_marks_nothing_even_with_values():
    """Guards against a zero-division producing an inf band that marks everything."""
    marks = H.best_in_row(np.array([0.5, 0.1]), 0)
    assert not marks.any()


# ---------------------------------------------------------------------------
# the long table
# ---------------------------------------------------------------------------


def _two_method_frame():
    return _accuracy_frame(
        [
            {
                "run_id": "as01-260728-260802",
                "dataset": "as01",
                "method": SHORTEST_PING,
                "n_targets": 400,
                "accuracy_top1": 0.60,
                "fallback_rate": 0.0,
            },
            {
                "run_id": "as01-260728-260802",
                "dataset": "as01",
                "method": "vanilla_cbg",
                "n_targets": 400,
                "accuracy_top1": 0.40,
                "fallback_rate": 0.25,
            },
        ]
    )


def test_a_reserved_row_is_nan_and_flagged_not_zero():
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    long = H.headline_long(
        _two_method_frame(), plan, top_n=1, methods=[SHORTEST_PING, "vanilla_cbg"]
    )
    reserved = long[long["kind"] == "weighted"]
    assert reserved["pending"].all()
    assert reserved["accuracy"].isna().all()
    assert not reserved["is_best"].any()


def test_a_missing_top_n_column_names_the_ones_available():
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    with pytest.raises(ValueError, match="accuracy_top1"):
        H.headline_long(_two_method_frame(), plan, top_n=5, methods=[SHORTEST_PING])


def test_every_requested_method_gets_a_cell_even_when_the_run_lacks_it():
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    long = H.headline_long(
        _two_method_frame(), plan, top_n=1, methods=[SHORTEST_PING, "spotter_cbg"]
    )
    mesh = long[long["kind"] == "mesh"].set_index("method")
    assert np.isnan(mesh.loc["spotter_cbg", "accuracy"])
    assert mesh.loc[SHORTEST_PING, "accuracy"] == 0.60


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _rendered(methods=(SHORTEST_PING, "vanilla_cbg")):
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    long = H.headline_long(
        _two_method_frame(), plan, top_n=1, methods=list(methods)
    )
    return H.render_markdown(
        long, top_n=1, grid="h3", resolution=4, methods=list(methods)
    )


def test_a_reserved_row_prints_the_pending_dash_never_a_number():
    text = _rendered()
    row = next(l for l in text.splitlines() if l.startswith("| AS01 TRAFFIC-WEIGHTED"))
    assert "0.000" not in row
    assert row.count(H.PENDING) == 3  # the n column plus both method cells


def test_the_pending_rows_are_explained_under_the_table():
    assert "reserved" in _rendered()
    assert "has_weight" in _rendered()


def test_a_non_zero_fallback_rate_rides_in_its_own_cell():
    text = _rendered()
    mesh = next(l for l in text.splitlines() if l.startswith("| AS01 MESH"))
    assert "(fb 0.25)" in mesh
    # and the five-sixths-zero column block is not printed
    assert "fallback" not in text.split("| dataset |")[1].splitlines()[0]


def test_a_zero_fallback_rate_prints_nothing():
    text = _rendered()
    assert "(fb 0.00)" not in text


def test_the_best_cell_is_bold_and_the_others_are_not():
    text = _rendered()
    mesh = next(l for l in text.splitlines() if l.startswith("| AS01 MESH"))
    assert "**0.600**" in mesh
    assert "**0.400**" not in mesh


def test_the_caption_states_the_tie_rule():
    assert "standard error" in _rendered()


def test_the_header_drops_the_redundant_cbg_suffix():
    text = _rendered()
    header = next(l for l in text.splitlines() if l.startswith("| dataset |"))
    assert "| Vanilla |" in header
    assert "Vanilla CBG" not in header


# ---------------------------------------------------------------------------
# the published-variant default
# ---------------------------------------------------------------------------


def test_the_default_method_set_is_the_six_published_variants():
    assert H.PUBLISHED_METHODS == (
        SHORTEST_PING,
        "million_scale_cbg",
        "vanilla_cbg",
        "octant_cbg_hull",
        "octant_cbg_spl",
        "spotter_cbg",
    )


def test_the_body_and_appendix_top_ns_are_distinct():
    assert H.BODY_TOP_N == 1
    assert H.APPENDIX_TOP_N == 3


# ---------------------------------------------------------------------------
# the pooled row
# ---------------------------------------------------------------------------


def _two_dataset_frame(
    *, a_acc=0.60, a_n=400, b_acc=0.20, b_n=100, a_fb=0.0, b_fb=0.0
) -> pd.DataFrame:
    """Two datasets whose micro- and macro-averages visibly disagree."""
    return _accuracy_frame(
        [
            {
                "run_id": "as01-260728-260802",
                "dataset": "as01",
                "method": SHORTEST_PING,
                "n_targets": a_n,
                "accuracy_top1": a_acc,
                "fallback_rate": a_fb,
            },
            {
                "run_id": "as02-260728-260802",
                "dataset": "as02",
                "method": SHORTEST_PING,
                "n_targets": b_n,
                "accuracy_top1": b_acc,
                "fallback_rate": b_fb,
            },
        ]
    )


def _table_body(text: str) -> str:
    """Only the rows. The caption names both marks, so a bare `mark in text`
    would pass on prose rather than on a cell."""
    return "\n".join(
        l for l in text.splitlines() if l.startswith("| ") and "---" not in l
    )


def _two_dataset_plan():
    return H.row_plan(
        {
            "as01-260728-260802": _Run("as01-260728-260802"),
            "as02-260728-260802": _Run("as02-260728-260802"),
        },
        {},
    )


def test_the_aggregate_sums_counts_rather_than_averaging_rates():
    """0.60 on 400 and 0.20 on 100 pool to 260/500, not to the mean of the two."""
    long = H.headline_long(
        _two_dataset_frame(), _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING]
    )
    agg = long[(long["scope"] == H.AGGREGATE) & (long["kind"] == "mesh")].iloc[0]
    assert agg["n_correct"] == 260
    assert agg["n_targets"] == 500
    assert agg["accuracy"] == pytest.approx(0.52)
    assert agg["accuracy"] != pytest.approx(0.40)  # the macro mean


def test_the_aggregate_n_is_the_sum_of_its_breakdown_rows():
    long = H.headline_long(
        _two_dataset_frame(), _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING]
    )
    mesh = long[long["kind"] == "mesh"]
    agg = mesh[mesh["scope"] == H.AGGREGATE].iloc[0]
    breakdown = mesh[mesh["scope"] == H.DATASET]
    assert agg["n_targets"] == breakdown["n_targets"].sum()
    assert agg["n_correct"] == breakdown["n_correct"].sum()


def test_the_aggregate_fallback_rate_pools_counts_not_rates():
    long = H.headline_long(
        _two_dataset_frame(a_fb=0.25, b_fb=0.10),
        _two_dataset_plan(),
        top_n=1,
        methods=[SHORTEST_PING],
    )
    agg = long[(long["scope"] == H.AGGREGATE) & (long["kind"] == "mesh")].iloc[0]
    # 100 of 400 plus 10 of 100 = 110 of 500
    assert agg["n_fallback"] == 110
    assert agg["fallback_rate"] == pytest.approx(0.22)
    assert agg["fallback_rate"] != pytest.approx(0.175)  # the macro mean


def test_a_breakdown_rows_accuracy_is_the_csv_value_verbatim_never_reconstructed():
    """The three real cells where the two routes round to different digits.

    `table-accuracy` prints the verbatim value, and the module's premise is that
    the two tables cannot disagree; recomputing a breakdown row from counts
    would move as01's Spotter cell from 0.389 to 0.388.
    """
    cases = [(399, 0.3885, "0.389"), (412, 0.3665, "0.366"), (412, 0.2985, "0.298")]
    for n, rate, printed in cases:
        frame = _accuracy_frame(
            [
                {
                    "run_id": "as01-260728-260802",
                    "dataset": "as01",
                    "method": SHORTEST_PING,
                    "n_targets": n,
                    "accuracy_top1": rate,
                    "fallback_rate": 0.0,
                }
            ]
        )
        plan = H.row_plan({"as01-260728-260802": _Run("as01-260728-260802")}, {})
        long = H.headline_long(frame, plan, top_n=1, methods=[SHORTEST_PING])
        row = long[long["scope"] == H.DATASET].iloc[0]
        assert row["accuracy"] == rate
        assert not row["accuracy_is_reconstructed"]
        assert f"{row['accuracy']:.3f}" == printed
        # and the reconstruction really would have differed
        assert f"{round(rate * n) / n:.3f}" != printed


def test_the_count_reconstruction_is_the_unique_integer_for_that_rate():
    """`round(acc * n)` is exact, not merely close, at 4 dp and these n."""
    for n in (399, 412, 458):
        for k in range(n + 1):
            rate = round(k / n, 4)
            candidates = [j for j in range(n + 1) if round(j / n, 4) == rate]
            assert candidates == [k]


def test_the_aggregate_of_a_kind_with_no_runs_is_pending():
    """`pending` stays "no run is paired here" even while a placeholder prints."""
    long = H.headline_long(
        _two_dataset_frame(), _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING]
    )
    weighted = long[(long["kind"] == "weighted") & (long["scope"] == H.AGGREGATE)]
    assert bool(weighted["pending"].all())
    # no denominator, so no standard error, so never marked best
    assert not bool(weighted["is_best"].any())
    assert weighted["n_targets"].isna().all()


def test_a_provisional_aggregate_prints_its_placeholder_and_is_marked():
    long = H.headline_long(
        _two_dataset_frame(), _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING]
    )
    row = long[(long["kind"] == "weighted") & (long["scope"] == H.AGGREGATE)].iloc[0]
    assert row["provisional"]
    assert row["accuracy"] == H.PROVISIONAL_WEIGHTED[1][SHORTEST_PING]["accuracy"]
    text = H.render_markdown(
        long, top_n=1, grid="h3", resolution=4, methods=[SHORTEST_PING]
    )
    assert H.PROVISIONAL_MARK in _table_body(text)
    assert "Provisional" in text


def test_the_appendix_top_n_has_no_placeholder_so_its_weighted_row_stays_blank():
    """Only top-1 rates were supplied; a top-1 figure must not print under a
    top-3 heading."""
    frame = _two_dataset_frame()
    frame["accuracy_top3"] = frame["accuracy_top1"]
    long = H.headline_long(
        frame, _two_dataset_plan(), top_n=3, methods=[SHORTEST_PING]
    )
    weighted = long[(long["kind"] == "weighted") & (long["scope"] == H.AGGREGATE)]
    assert not bool(weighted["provisional"].any())
    assert weighted["accuracy"].isna().all()


def test_a_real_weighted_run_wins_over_the_placeholder():
    """The placeholder is gated on the row having no run, so pairing one is a
    switch-over that needs no flag."""
    assert H.provisional_for("weighted", H.AGGREGATE, 1, 0) is not None
    assert H.provisional_for("weighted", H.AGGREGATE, 1, 2) is None
    assert H.provisional_for("mesh", H.AGGREGATE, 1, 0) is None
    assert H.provisional_for("weighted", H.DATASET, 1, 0) is None


def test_a_partial_aggregate_is_not_pending_and_states_its_coverage():
    """One weighted twin landing first must not print as though all three had."""
    frame = pd.concat(
        [
            _two_dataset_frame(),
            _accuracy_frame(
                [
                    {
                        "run_id": "as01-w",
                        "dataset": "as01-w",
                        "method": SHORTEST_PING,
                        "n_targets": 50,
                        "accuracy_top1": 0.80,
                        "fallback_rate": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    plan = H.row_plan(
        {
            "as01-260728-260802": _Run("as01-260728-260802"),
            "as02-260728-260802": _Run("as02-260728-260802"),
        },
        {"as01": _Run("as01-w")},
    )
    long = H.headline_long(frame, plan, top_n=1, methods=[SHORTEST_PING])
    agg = long[(long["kind"] == "weighted") & (long["scope"] == H.AGGREGATE)].iloc[0]
    assert not agg["pending"]
    assert agg["partial"]
    assert agg["n_targets"] == 50  # its own contributor, never the mesh row's 500
    assert "1 of 2 ASes" in agg["row_label"]


def test_an_aggregate_winner_that_does_not_lead_every_dataset_is_daggered():
    """The top-3 Simpson case: the pool and its members disagree about the winner."""
    frame = _accuracy_frame(
        [
            {"run_id": "as01-260728-260802", "dataset": "as01", "method": SHORTEST_PING,
             "n_targets": 100, "accuracy_top1": 0.50, "fallback_rate": 0.0},
            {"run_id": "as01-260728-260802", "dataset": "as01", "method": "vanilla_cbg",
             "n_targets": 100, "accuracy_top1": 0.90, "fallback_rate": 0.0},
            {"run_id": "as02-260728-260802", "dataset": "as02", "method": SHORTEST_PING,
             "n_targets": 900, "accuracy_top1": 0.80, "fallback_rate": 0.0},
            {"run_id": "as02-260728-260802", "dataset": "as02", "method": "vanilla_cbg",
             "n_targets": 900, "accuracy_top1": 0.40, "fallback_rate": 0.0},
        ]
    )
    long = H.headline_long(
        frame, _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING, "vanilla_cbg"]
    )
    agg = long[(long["scope"] == H.AGGREGATE) & (long["kind"] == "mesh")]
    winner = agg[agg["is_best"]].iloc[0]
    assert winner["method"] == SHORTEST_PING       # 770/1000 pooled
    assert winner["won_datasets"] == "as02"        # but it loses as01
    assert not winner["unanimous"]

    text = H.render_markdown(
        long, top_n=1, grid="h3", resolution=4, methods=[SHORTEST_PING, "vanilla_cbg"]
    )
    assert H.NON_UNANIMOUS_MARK in _table_body(text)
    assert "1 of 2 datasets" in text


def test_a_unanimous_aggregate_winner_is_not_daggered():
    frame = _accuracy_frame(
        [
            {"run_id": "as01-260728-260802", "dataset": "as01", "method": SHORTEST_PING,
             "n_targets": 100, "accuracy_top1": 0.90, "fallback_rate": 0.0},
            {"run_id": "as01-260728-260802", "dataset": "as01", "method": "vanilla_cbg",
             "n_targets": 100, "accuracy_top1": 0.20, "fallback_rate": 0.0},
            {"run_id": "as02-260728-260802", "dataset": "as02", "method": SHORTEST_PING,
             "n_targets": 100, "accuracy_top1": 0.90, "fallback_rate": 0.0},
            {"run_id": "as02-260728-260802", "dataset": "as02", "method": "vanilla_cbg",
             "n_targets": 100, "accuracy_top1": 0.20, "fallback_rate": 0.0},
        ]
    )
    long = H.headline_long(
        frame, _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING, "vanilla_cbg"]
    )
    agg = long[(long["scope"] == H.AGGREGATE) & (long["kind"] == "mesh")]
    winner = agg[agg["is_best"]].iloc[0]
    assert winner["unanimous"]
    text = H.render_markdown(
        long, top_n=1, grid="h3", resolution=4, methods=[SHORTEST_PING, "vanilla_cbg"]
    )
    assert H.NON_UNANIMOUS_MARK not in _table_body(text)


def test_an_aggregate_cell_is_pooled_over_the_runs_that_carry_the_method():
    """A missing method is scored on its own targets, and the cell says so."""
    frame = _accuracy_frame(
        [
            {"run_id": "as01-260728-260802", "dataset": "as01", "method": SHORTEST_PING,
             "n_targets": 400, "accuracy_top1": 0.60, "fallback_rate": 0.0},
            {"run_id": "as01-260728-260802", "dataset": "as01", "method": "vanilla_cbg",
             "n_targets": 400, "accuracy_top1": 0.50, "fallback_rate": 0.0},
            {"run_id": "as02-260728-260802", "dataset": "as02", "method": SHORTEST_PING,
             "n_targets": 100, "accuracy_top1": 0.20, "fallback_rate": 0.0},
        ]
    )
    long = H.headline_long(
        frame, _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING, "vanilla_cbg"]
    )
    agg = long[(long["scope"] == H.AGGREGATE) & (long["kind"] == "mesh")].set_index("method")
    assert agg.loc[SHORTEST_PING, "n_targets"] == 500
    assert bool(agg.loc[SHORTEST_PING, "denominator_complete"])
    assert agg.loc["vanilla_cbg", "n_targets"] == 400   # not the row's 500
    assert not bool(agg.loc["vanilla_cbg", "denominator_complete"])
    assert agg.loc["vanilla_cbg", "datasets"] == "as01"

    text = H.render_markdown(
        long, top_n=1, grid="h3", resolution=4, methods=[SHORTEST_PING, "vanilla_cbg"]
    )
    assert H.INCOMPLETE_MARK in _table_body(text)


def test_best_in_row_uses_the_winners_own_denominator_when_they_differ():
    """A scalar `n` still behaves exactly as before; an array picks the winner's."""
    acc = np.array([0.60, 0.55])
    assert list(H.best_in_row(acc, 100)) == list(H.best_in_row(acc, [100, 100]))
    # A tiny winner denominator widens the band enough to catch the runner-up.
    assert list(H.best_in_row(acc, [25, 10_000])) == [True, True]
    assert list(H.best_in_row(acc, [10_000, 25])) == [True, False]


def test_the_render_groups_on_row_index_so_a_null_dataset_cannot_drop_a_row():
    """`groupby(["dataset", ...])` drops NaN keys, which are every pooled row."""
    long = H.headline_long(
        _two_dataset_frame(), _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING]
    )
    assert long.loc[long["scope"] == H.AGGREGATE, "dataset"].isna().all()
    text = H.render_markdown(
        long, top_n=1, grid="h3", resolution=4, methods=[SHORTEST_PING]
    )
    body = _table_body(text).splitlines()
    assert sum(1 for l in body if l.startswith("| **MESH")) == 1
    assert sum(1 for l in body if l.startswith("| **TRAFFIC-WEIGHTED")) == 1
    # "unanalyzed" in the pending note contains the substring, so check the cells.
    assert "nan" not in _table_body(text)


def test_the_aggregate_n_prints_a_thousands_separator():
    long = H.headline_long(
        _two_dataset_frame(a_n=1200, b_n=69),
        _two_dataset_plan(),
        top_n=1,
        methods=[SHORTEST_PING],
    )
    text = H.render_markdown(
        long, top_n=1, grid="h3", resolution=4, methods=[SHORTEST_PING]
    )
    assert "| 1,269 |" in text


def test_the_weighting_check_records_the_macro_mean_it_is_not():
    long = H.headline_long(
        _two_dataset_frame(), _two_dataset_plan(), top_n=1, methods=[SHORTEST_PING]
    )
    check = H.weighting_check(long)
    entry = next(v for k, v in check.items() if k.startswith("MESH"))
    assert entry["pooled_micro"] == pytest.approx(0.52)
    assert entry["dataset_macro_mean"] == pytest.approx(0.40)
    assert entry["delta"] == pytest.approx(0.12)


def test_the_ranking_records_the_spread_between_first_and_third():
    frame = _accuracy_frame(
        [
            {"run_id": "as01-260728-260802", "dataset": "as01", "method": m,
             "n_targets": 100, "accuracy_top1": a, "fallback_rate": 0.0}
            for m, a in [(SHORTEST_PING, 0.90), ("vanilla_cbg", 0.70), ("spotter_cbg", 0.50)]
        ]
        + [
            {"run_id": "as02-260728-260802", "dataset": "as02", "method": m,
             "n_targets": 100, "accuracy_top1": a, "fallback_rate": 0.0}
            for m, a in [(SHORTEST_PING, 0.90), ("vanilla_cbg", 0.70), ("spotter_cbg", 0.50)]
        ]
    )
    methods = [SHORTEST_PING, "vanilla_cbg", "spotter_cbg"]
    long = H.headline_long(frame, _two_dataset_plan(), top_n=1, methods=methods)
    entry = next(v for k, v in H.ranking(long).items() if k.startswith("MESH"))
    assert [r["method"] for r in entry["ranked"]] == methods
    assert entry["spread_rank1_to_rank3"] == pytest.approx(0.40)
