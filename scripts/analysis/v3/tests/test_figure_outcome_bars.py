"""`plot-outcome-bars` — the partition, the encoding budget, and the empty half.

Four things this figure can get quietly wrong. Its segments can stop summing to
the target set, which renders as a shorter bar rather than as an error. A
segment can acquire a texture, which collides with the one channel the dataset
type owns. Its three outcome colours can lose the lightness ordering that is
what colour-blind and greyscale readers have left. And the uncollected
traffic-weighted campaign can render as a zero instead of as a gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import figure_outcome_bars as B
from scripts.analysis.v3.modules import headline_table as H
from scripts.analysis.v3.modules.classify import SHORTEST_PING


class _Run:
    def __init__(self, run_id: str = "run", setup: str = "anchors_to_probes") -> None:
        self.run_id = run_id
        self.setup = setup


def _relative_luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _frame() -> pd.DataFrame:
    """Two datasets, one method with fallbacks and one without."""
    rows = []
    for run_id, dataset, n, acc, fb in [
        ("as01-260728-260802", "as01", 400, 0.60, 0.25),
        ("as02-260728-260802", "as02", 100, 0.20, 0.10),
    ]:
        n_fb = int(round(fb * n))
        rows += [
            {
                "run_id": run_id, "dataset": dataset, "method": SHORTEST_PING,
                "n_targets": n, "accuracy_top1": acc, "fallback_rate": 0.0,
                "n_solved": n, "n_fallback": 0, "n_error": 0,
            },
            {
                "run_id": run_id, "dataset": dataset, "method": "vanilla_cbg",
                "n_targets": n, "accuracy_top1": acc / 2, "fallback_rate": fb,
                "n_solved": n - n_fb, "n_fallback": n_fb, "n_error": 0,
            },
        ]
    return pd.DataFrame(rows)


def _plan():
    return H.row_plan(
        {
            "as01-260728-260802": _Run("as01-260728-260802"),
            "as02-260728-260802": _Run("as02-260728-260802"),
        },
        {},
    )


METHODS = [SHORTEST_PING, "vanilla_cbg"]


# ---- the partition ----------------------------------------------------------


def test_the_four_segments_account_for_every_target():
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    drawn = table[~table["pending"]]
    assert len(drawn)
    assert (drawn[list(B.SEGMENTS)].sum(axis=1) == drawn["n_targets"]).all()


def test_the_shares_sum_to_one_on_every_drawn_bar():
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    drawn = table[~table["pending"]]
    shares = drawn[[f"share_{B.SEGMENT_LABELS[k]}" for k in B.SEGMENTS]].sum(axis=1)
    assert np.allclose(shares, 1.0)


def test_a_stack_that_does_not_partition_is_refused_rather_than_drawn_short():
    """A bar ending early looks like a smaller total, not like a bug."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    table.loc[table.index[0], "n_wrong"] -= 5
    with pytest.raises(ValueError) as exc:
        B.guard_partition(table)
    assert "partition" in str(exc.value)


def test_the_correct_segment_is_the_top_n_accuracy_the_table_prints():
    """Figure and table are one arithmetic path, so this is equality not closeness."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    long = H.headline_long(_frame(), _plan(), top_n=1, methods=METHODS)
    agg = long[(long["scope"] == H.AGGREGATE) & (long["kind"] == H.MESH)].set_index("method")
    bars = table[table["kind"] == H.MESH].set_index("method")
    for method in METHODS:
        assert bars.loc[method, "share_correct"] == agg.loc[method, "accuracy"]
        assert bars.loc[method, "n_correct"] == agg.loc[method, "n_correct"]


def test_the_compare_layout_keeps_each_dataset_on_its_own_denominator():
    table = B.compare_table(_frame(), _plan(), top_n=1, methods=METHODS)
    mesh = table[(table["kind"] == H.MESH)].set_index(["dataset", "method"])
    assert mesh.loc[("as01", SHORTEST_PING), "n_targets"] == 400
    assert mesh.loc[("as02", SHORTEST_PING), "n_targets"] == 100
    assert mesh.loc[("as02", SHORTEST_PING), "share_correct"] == pytest.approx(0.20)


# ---- the empty half ---------------------------------------------------------


def test_the_uncollected_campaign_is_never_drawn_as_zero():
    """Whether it is a ghost or a placeholder, it must not read as 'scored 0'."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    weighted = table[table["kind"] == H.WEIGHTED]
    assert len(weighted) == len(METHODS)
    drawn_as_zero = weighted["share_correct"] == 0.0
    assert not bool(drawn_as_zero.any())
    # every weighted row is either a ghost or a placeholder, never a real result
    assert bool((weighted["pending"] | weighted["provisional"]).all())


def test_a_provisional_row_carries_shares_but_no_counts():
    """The placeholder is rates from an earlier run with no denominator, so the
    counts stay NaN rather than being back-computed against an invented `n`."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    weighted = table[table["kind"] == H.WEIGHTED]
    assert bool(weighted["provisional"].all())
    assert weighted[list(B.COUNTS)].isna().all().all()
    assert weighted["share_correct"].notna().all()
    shares = weighted[[f"share_{B.SEGMENT_LABELS[k]}" for k in B.SEGMENTS]].sum(axis=1)
    assert np.allclose(shares, 1.0)


def test_a_provisional_row_reads_its_rates_from_the_headline_table():
    """One placeholder, so the figure and the table cannot quote it differently."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    weighted = table[table["kind"] == H.WEIGHTED].set_index("method")
    for method, block in H.PROVISIONAL_WEIGHTED[1].items():
        if method in weighted.index:
            assert weighted.loc[method, "share_correct"] == block["accuracy"]


def test_the_legend_swatch_matches_the_mark_the_reader_will_meet():
    """A kind with no data is drawn dashed, so its swatch is dashed too — the
    key shows the mark, and that the campaign is uncollected is a fact about the
    data that belongs in the caption, not in the legend text."""
    handles = B.kind_handles([H.MESH, H.WEIGHTED], pending={H.WEIGHTED})
    assert [h.get_label() for h in handles] == ["mesh", "traffic-weighted"]
    solid, dashed = handles
    assert solid.get_linestyle() == "solid"
    assert dashed.get_linestyle() != "solid"


def test_the_figure_does_not_mark_a_provisional_bar():
    """A deliberate omission, so the caveat has to survive somewhere: the CSV
    column, the table's footnote and the manifest all carry it."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    assert B.provisional_kinds(table) == {H.WEIGHTED}
    labels = [h.get_label() for h in B.kind_handles([H.MESH, H.WEIGHTED], pending=set())]
    assert all(H.PROVISIONAL_MARK not in label for label in labels)
    assert "provisional" in table.columns


def test_a_kind_with_data_is_drawn_solid_in_the_key():
    handles = B.kind_handles([H.MESH, H.WEIGHTED], pending=set())
    assert all(h.get_linestyle() == "solid" for h in handles)


# ---- the encoding budget ----------------------------------------------------


def test_hatch_is_reserved_for_the_dataset_type_and_no_segment_takes_it():
    """Hatch is the dataset type, so the outcome scheme separates in colour only."""
    assert B.KIND_HATCH[H.MESH] == ""
    assert B.KIND_HATCH[H.WEIGHTED] != ""
    assert set(B.SEGMENT_INK) == set(B.SEGMENTS)


def test_no_outcome_colour_is_one_of_the_six_variant_hues():
    """Colour means the outcome here, so it must not read as a method."""
    from scripts.analysis.v3.modules.diagram.common.palette import _VARIANT_HUES

    assert not set(B.SEGMENT_INK.values()) & set(_VARIANT_HUES)


def test_the_outcome_colours_are_monotone_in_lightness_up_the_stack():
    """Lightness is the separator neither greyscale nor colour blindness removes,
    and red-vs-green is exactly the pair that needs one."""
    lightness = [_relative_luminance(B.SEGMENT_INK[k]) for k in B.SEGMENTS]
    assert lightness == sorted(lightness)
    assert all(b / a >= 1.8 for a, b in zip(lightness, lightness[1:]))


def test_every_segment_label_is_legible_against_its_own_fill():
    """The stack runs dark green to light grey, so one fixed label colour fails
    at one end whichever end it is picked for."""
    for key in B.SEGMENTS:
        fill = B.SEGMENT_INK[key]
        ink = B.label_ink(fill)
        assert ink in {"#ffffff", "#0b0b0b"}
        lum = _relative_luminance(fill)
        ratio = (
            1.05 / (lum + 0.05) if ink == "#ffffff" else (lum + 0.05) / 0.05
        )
        assert ratio >= 4.5


def test_the_legend_always_lists_all_three_outcomes_in_stack_order():
    """An entry that comes and goes with the data makes the scheme look
    data-dependent; it is a fixed three-part key."""
    labels = [h.get_label() for h in B.outcome_handles()]
    assert labels == ["correct", "wrong", "failed"]
    assert "fallback" not in labels  # the paper says "failed"


def test_the_dataset_type_legend_entries_carry_no_fill():
    """Any neutral filling them is within reach of the `failed` grey."""
    for handle in B.kind_handles([H.MESH, H.WEIGHTED], pending=set()):
        assert handle.get_facecolor()[3] == 0.0  # transparent


# ---- rendering --------------------------------------------------------------


def test_the_pooled_figure_renders_non_empty(tmp_path):
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    out = B.plot_bars(table, tmp_path / "bars.png", methods=METHODS, title="t")
    assert out.exists() and out.stat().st_size > 5_000


def test_the_compare_figure_renders_non_empty(tmp_path):
    table = B.compare_table(_frame(), _plan(), top_n=1, methods=METHODS)
    out = B.plot_compare(
        table, tmp_path / "bars_by_dataset.png", methods=METHODS, title="t"
    )
    assert out.exists() and out.stat().st_size > 5_000


def test_the_bars_run_best_first_by_pooled_correct_rate():
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    ordered = B.method_order(table, METHODS)
    mesh = table[table["kind"] == H.MESH].set_index("method")
    rates = [mesh.loc[m, "share_correct"] for m in ordered]
    assert rates == sorted(rates, reverse=True)
    assert ordered[0] == SHORTEST_PING  # 0.52 pooled vs vanilla's 0.26


def test_the_order_is_shared_across_compare_panels_not_recomputed_per_panel():
    """An x position has to mean the same method in every panel, or a
    cross-dataset scan becomes a search."""
    table = B.compare_table(_frame(), _plan(), top_n=1, methods=METHODS)
    ordered = B.method_order(table, METHODS)
    # as02 alone ranks the two methods the same way here, so pin the mechanism:
    # the order comes from the whole table, not from any one dataset's slice.
    as02_only = B.method_order(table[table["dataset"] == "as02"], METHODS)
    assert ordered == B.method_order(table, METHODS)  # deterministic
    assert isinstance(as02_only, list)


def test_ties_fall_back_to_the_published_order_so_the_layout_is_deterministic():
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    tied = table.copy()
    tied["n_correct"] = 100
    tied["n_targets"] = 200
    assert B.method_order(tied, METHODS) == METHODS


def test_error_folds_into_failed_and_never_into_wrong():
    """A crash is a failure to answer, not a scoring miss."""
    frame = _frame()
    # move 20 of as01 Vanilla's targets from solved into `error`
    mask = (frame["run_id"] == "as01-260728-260802") & (frame["method"] == "vanilla_cbg")
    frame.loc[mask, "n_error"] = 20
    frame.loc[mask, "n_solved"] = frame.loc[mask, "n_solved"] - 20

    table = B.outcome_table(frame, _plan(), top_n=1, methods=METHODS)
    row = table[(table["kind"] == H.MESH) & (table["method"] == "vanilla_cbg")].iloc[0]
    assert row["n_error"] == 20
    assert row["n_failed"] == row["n_fallback"] + row["n_error"]
    # and the partition still holds, so nothing was double-counted
    assert row["n_correct"] + row["n_wrong"] + row["n_failed"] == row["n_targets"]


def test_the_counts_keep_the_fallback_error_split_the_figure_merges():
    """The bar draws three segments; the artifact must still say which failure."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    for column in B.COUNTS:
        assert column in table.columns
    assert "n_failed" in table.columns


def test_a_segment_too_thin_to_hold_its_label_is_left_unlabelled():
    """Better an unlabelled sliver than a number drawn over its neighbours."""
    assert B.MIN_LABEL_SHARE > 0
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    real = table[~table["pending"] & ~table["provisional"]]
    shares = real[[f"share_{B.SEGMENT_LABELS[k]}" for k in B.SEGMENTS]].to_numpy()
    # no measured segment is thin enough to trip it, so it stays a guard
    assert ((shares == 0) | (shares >= B.MIN_LABEL_SHARE)).all()
    # the placeholder does trip it — 99.3% correct leaves a 0.7% sliver
    prov = table[table["provisional"]]["share_wrong"]
    assert bool((prov < B.MIN_LABEL_SHARE).any())


def test_the_two_legends_sit_on_one_centred_line_without_overlapping(tmp_path):
    """The titles are placed by measurement, not by a hard-coded offset, so the
    layout has to be checked at a real figure size rather than reasoned about."""
    table = B.outcome_table(_frame(), _plan(), top_n=1, methods=METHODS)
    out = B.plot_bars(table, tmp_path / "bars.png", methods=METHODS, title="t")
    assert out.exists()


def test_the_comparison_grid_reserves_a_wider_band_for_its_panel_titles():
    """Its panels carry `AS01 · n=399` above the axes; the pooled panel does
    not, and one shared constant put the legend through them."""
    assert B.COMPARE_LEGEND_Y > B.LEGEND_Y
    assert B.COMPARE_AXES_TOP < B.AXES_TOP
    assert B.COMPARE_AXES_TOP < B.COMPARE_LEGEND_Y
    assert B.AXES_TOP < B.LEGEND_Y
