"""The cost model: which columns are a cost, how they compose, and in what unit.

These tests exist because three of the model's policies are counter-intuitive
and were each chosen against a measured alternative: reduce-then-percentile,
max-not-sum for memory, and an all-rows cost denominator. They moved here from
`test_pareto.py` when `cost.py` was extracted -- three commands now depend on
these policies, so they are no longer one figure's business.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import cost as C
from scripts.analysis.v3.modules.paths import MissingArtifactError


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


def _cost_frame(ltd, mtl, ctr, status=None):
    n = len(ltd)
    return pd.DataFrame(
        {
            "target_id": [f"t{i}" for i in range(n)],
            "status": status or ["SUCCESS"] * n,
            "ltd_ms": ltd,
            "mtl_ms": mtl,
            "ctr_ms": ctr,
        }
    )


def test_reduce_happens_per_target_before_the_percentile():
    """The regression this pins: a median does not distribute over a sum.

    Per-stage medians are 1, 10, 1 -> 12, but the cheapest target costs 16 and
    the median target 52. Stacking the marginals understates by 4x here.
    """
    df = _cost_frame([1.0, 1.0, 5.0], [10.0, 50.0, 10.0], [100.0, 1.0, 1.0])
    got = C.per_target_cost(df, C.COST_SPECS["runtime"])
    assert sorted(got) == [16.0, 52.0, 111.0]
    assert float(np.median(got)) == 52.0
    naive = sum(float(np.median(df[c])) for c in ("ltd_ms", "mtl_ms", "ctr_ms"))
    assert naive != float(np.median(got))


def test_memory_max_reduce_is_at_most_the_sum_and_usually_less():
    spec = C.COST_SPECS["runtime"]  # reuse the ms columns as a stand-in
    df = _cost_frame([3.0, 1.0], [7.0, 1.0], [2.0, 1.0])
    mx = C.per_target_cost(df, spec, reduce="max")
    sm = C.per_target_cost(df, spec, reduce="sum")
    assert mx.tolist() == [7.0, 1.0]
    assert sm.tolist() == [12.0, 3.0]
    assert np.all(mx <= sm)


def test_a_null_stage_on_one_row_costs_nothing_but_keeps_the_row():
    """A skipped CTR on a FALLBACK row did no work; the target still exists."""
    df = _cost_frame([1.0, 2.0], [3.0, 4.0], [5.0, None])
    got = C.per_target_cost(df, C.COST_SPECS["runtime"])
    assert got.tolist() == [9.0, 6.0]


def test_an_uninstrumented_ltd_column_is_nan_not_zero():
    """The regression this guards: fillna(0) manufacturing a free method that
    then dominates the entire frontier."""
    df = _cost_frame([None, None], [3.0, 4.0], [5.0, 6.0])
    got = C.per_target_cost(df, C.COST_SPECS["runtime"])
    assert np.isnan(got).all()


def test_legacy_single_channel_schema_fails_loudly():
    df = pd.DataFrame({"target_id": ["t0"], "status": ["SUCCESS"], "ltd_peak_bytes": [1]})
    with pytest.raises(MissingArtifactError, match="cost columns"):
        C.per_target_cost(df, C.COST_SPECS["memory_alloc"])


def test_all_rows_and_solved_only_costs_differ_on_fallback_rows():
    """The as02 `vanilla_cbg` shape: `ctr_ms` null on exactly the FALLBACK rows.

    Solved-only reads higher, which is why the default denominator is `all` —
    it has to match `accuracy_topN`'s.
    """
    df = _cost_frame(
        [1.0, 1.0, 1.0, 1.0],
        [5.0, 5.0, 5.0, 5.0],
        [20.0, 20.0, None, None],
        status=["SUCCESS", "SUCCESS", "FALLBACK", "FALLBACK"],
    )
    spec = C.COST_SPECS["runtime"]
    all_rows = C.cost_stats(C.per_target_cost(df, spec))["p50"]
    solved = C.cost_stats(
        C.per_target_cost(df[df["status"] == "SUCCESS"], spec)
    )["p50"]
    assert all_rows == 16.0 and solved == 26.0


def test_bytes_scale_to_mb():
    spec = C.COST_SPECS["memory_alloc"]
    df = pd.DataFrame(
        {
            "target_id": ["t0"],
            "status": ["SUCCESS"],
            "ltd_alloc_peak_bytes": [2 * 1024 * 1024],
            "mtl_alloc_peak_bytes": [1024 * 1024],
            "ctr_alloc_peak_bytes": [0],
        }
    )
    assert C.per_target_cost(df, spec).tolist() == [2.0]  # max-reduced


def test_cost_stats_on_an_all_nan_input_is_nan_not_zero():
    st = C.cost_stats(np.array([np.nan, np.nan]))
    assert st["n"] == 0 and np.isnan(st["p50"])


# ---------------------------------------------------------------------------
# per-stage decomposition
# ---------------------------------------------------------------------------


def test_per_stage_keeps_the_three_stages_separate_over_one_denominator():
    df = _cost_frame([1.0, 2.0], [10.0, 20.0], [100.0, 200.0])
    got = C.per_stage_cost(df, C.COST_SPECS["runtime"])
    assert list(got.columns) == list(C.STAGES)
    assert len(got) == len(df)  # no per-stage dropna anywhere
    assert got["mtl"].tolist() == [10.0, 20.0]


def test_per_stage_columns_reduce_to_the_pipeline_value():
    """The identity that stops the two functions drifting apart.

    `per_target_cost` is *defined* as this frame reduced row-wise, so the null
    policy has one definition rather than two that can diverge.
    """
    df = _cost_frame([1.0, 5.0], [10.0, 2.0], [100.0, 3.0])
    for key, op in (("runtime", "sum"), ("memory_alloc", "max")):
        spec = C.COST_SPECS["runtime"]  # ms columns; reduce is what varies
        stages = C.per_stage_cost(df, spec)
        expected = stages.sum(axis=1) if op == "sum" else stages.max(axis=1)
        got = C.per_target_cost(df, spec, reduce=op)
        assert got.tolist() == expected.tolist(), key


def test_a_null_stage_fills_to_zero_in_its_own_column_only():
    df = _cost_frame([1.0, 2.0], [10.0, 20.0], [100.0, None],
                     status=["SUCCESS", "FALLBACK"])
    got = C.per_stage_cost(df, C.COST_SPECS["runtime"])
    assert got["ctr"].tolist() == [100.0, 0.0]
    assert got["ltd"].tolist() == [1.0, 2.0]  # untouched


def test_an_uninstrumented_ltd_column_nans_every_stage_not_just_ltd():
    """The regression this guards: a figure drawing MTL and CTR bars for a run
    that was never instrumented, which reads as a working measurement."""
    df = _cost_frame([None, None], [10.0, 20.0], [100.0, 200.0])
    got = C.per_stage_cost(df, C.COST_SPECS["runtime"])
    assert got.isna().all().all()


def test_stacking_per_stage_medians_overstates_or_understates_the_true_median():
    """The legacy defect this port exists to fix (`plot_phase_runtime.py:107-114`).

    Per-stage medians 1/10/1 sum to 12; the true per-target median is 52, and
    no target costs less than 16. The stacked marginals are not merely
    imprecise -- here they land below every observation.
    """
    df = _cost_frame([1.0, 1.0, 5.0], [10.0, 50.0, 10.0], [100.0, 1.0, 1.0])
    stages = C.per_stage_cost(df, C.COST_SPECS["runtime"])
    stacked = sum(float(stages[s].median()) for s in C.STAGES)
    pipeline = float(
        C.cost_stats(C.per_target_cost(df, C.COST_SPECS["runtime"]))["p50"]
    )
    assert stacked == 12.0
    assert pipeline == 52.0
    # Below the cheapest real target, not merely off the median.
    assert stacked < min(C.per_target_cost(df, C.COST_SPECS["runtime"]))


def test_stacking_per_stage_memory_peaks_exceeds_the_true_peak_by_construction():
    """`sum >= max` always, and strictly whenever two stages are non-zero, so a
    stacked memory bar has no statistic under it at any stat."""
    df = _cost_frame([9.0, 1.0], [1.0, 9.0], [1.0, 1.0])
    spec = C.COST_SPECS["runtime"]
    stages = C.per_stage_cost(df, spec)
    stacked = sum(float(np.percentile(stages[s], 95)) for s in C.STAGES)
    true_peak = float(
        np.percentile(C.per_target_cost(df, spec, reduce="max"), 95)
    )
    assert stacked > true_peak


def test_the_mean_stack_is_exactly_additive_for_sum_but_not_for_max():
    """This is what licenses `--layout stacked` at `--stat mean`, and only there.

    Linearity of expectation gives the sum case exactly; `mean(max)` is not
    `max(mean)`, so no stat rescues a stacked memory bar.
    """
    df = _cost_frame([9.0, 1.0], [1.0, 9.0], [1.0, 1.0])
    spec = C.COST_SPECS["runtime"]
    stages = C.per_stage_cost(df, spec)
    stage_means = sum(float(stages[s].mean()) for s in C.STAGES)
    assert stage_means == pytest.approx(
        float(C.per_target_cost(df, spec, reduce="sum").mean())
    )
    assert stage_means != pytest.approx(
        float(C.per_target_cost(df, spec, reduce="max").mean())
    )
