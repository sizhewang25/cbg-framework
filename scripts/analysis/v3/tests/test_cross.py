"""`cross` — the two guards on which run sets may legally be pooled.

`guard_disjoint_targets` lived in `figure_error_scatter` and had no direct test
while it had one caller. It has two now, and the second is a table command that
must not import matplotlib, so it moved here and this is its first test.
"""

from __future__ import annotations

import pytest
import typer

from scripts.analysis.v3.modules import cross


class _Run:
    def __init__(self, run_id: str, setup: str = "anchors_to_probes") -> None:
        self.run_id = run_id
        self.setup = setup


# ---- guard_disjoint_targets -------------------------------------------------


def test_disjoint_runs_pass_and_report_no_overlap():
    overlaps = cross.guard_disjoint_targets(
        {"as01": {"a", "b"}, "as02": {"c"}, "as03": {"d", "e"}}, remedy="Do X."
    )
    assert overlaps == {}


def test_runs_sharing_a_target_are_refused_rather_than_double_counted():
    with pytest.raises(typer.BadParameter) as exc:
        cross.guard_disjoint_targets(
            {"as01": {"a", "b"}, "as02": {"b", "c"}}, remedy="Do X."
        )
    message = str(exc.value)
    assert "1 ids in common between as01 and as02" in message
    assert "Do X." in message


def test_the_refusal_names_the_worst_pair_and_counts_them_all():
    with pytest.raises(typer.BadParameter) as exc:
        cross.guard_disjoint_targets(
            {"a": {1, 2, 3, 4}, "b": {3, 4, 5}, "c": {5, 6}}, remedy="Do X."
        )
    message = str(exc.value)
    assert "2 ids in common between a and b" in message  # the worst, not the first
    assert "2 overlapping pair(s)" in message


def test_the_remedy_is_the_callers_sentence_not_a_generic_one():
    """The useful advice differs per caller, so a default would be unactionable."""
    with pytest.raises(typer.BadParameter) as exc:
        cross.guard_disjoint_targets({"a": {1}, "b": {1}}, remedy="Use --layout compare.")
    assert "Use --layout compare." in str(exc.value)


def test_a_single_run_can_never_overlap_itself():
    assert cross.guard_disjoint_targets({"as01": {"a", "a", "b"}}, remedy="X") == {}


# ---- guard_one_setup --------------------------------------------------------


def test_one_setup_passes():
    cross.guard_one_setup(
        {"as01": _Run("as01"), "as02": _Run("as02")}, allow_mixed=False
    )


def test_mixed_setups_are_refused_and_named():
    with pytest.raises(typer.BadParameter) as exc:
        cross.guard_one_setup(
            {
                "as01": _Run("as01", "anchors_to_probes"),
                "as7018": _Run("as7018", "probes_to_anchors"),
            },
            allow_mixed=False,
        )
    message = str(exc.value)
    assert "anchors_to_probes" in message and "probes_to_anchors" in message


def test_allow_mixed_overrides_the_setup_guard():
    cross.guard_one_setup(
        {
            "as01": _Run("as01", "anchors_to_probes"),
            "as7018": _Run("as7018", "probes_to_anchors"),
        },
        allow_mixed=True,
    )
