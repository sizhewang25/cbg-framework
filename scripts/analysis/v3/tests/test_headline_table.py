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
    """A minimal `accuracy_rows`-shaped frame."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the row plan
# ---------------------------------------------------------------------------


def test_every_dataset_gets_a_mesh_row_and_a_weighted_row():
    plan = H.row_plan({"as01-260728-260802": _Run(), "as02-260728-260802": _Run()}, {})
    assert [(e["dataset"], e["kind"]) for e in plan] == [
        ("as01", "mesh"),
        ("as01", "weighted"),
        ("as02", "mesh"),
        ("as02", "weighted"),
    ]


def test_a_weighted_row_with_no_data_carries_no_run_id():
    plan = H.row_plan({"as01-260728-260802": _Run()}, {})
    weighted = next(e for e in plan if e["kind"] == "weighted")
    assert weighted["run_id"] is None


def test_mesh_run_order_fixes_the_dataset_order():
    plan = H.row_plan(
        {"as03-260728-260802": _Run(), "as01-260728-260802": _Run()}, {}
    )
    assert [e["dataset"] for e in plan] == ["as03", "as03", "as01", "as01"]


def test_a_weighted_run_fills_the_row_of_the_dataset_it_is_paired_to():
    plan = H.row_plan(
        {"as01-260728-260802": _Run()},
        {"as01": _Run(run_id="as01-weighted-260728")},
    )
    weighted = next(e for e in plan if e["kind"] == "weighted")
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
    row = next(l for l in text.splitlines() if l.startswith("| AS01 WEIGHTED"))
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
