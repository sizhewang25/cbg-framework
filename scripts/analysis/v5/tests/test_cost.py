"""The cost model, ported from v3: which columns are a cost and how they compose.

Three of its policies are counter-intuitive and were each chosen against a
measured alternative: reduce-then-percentile, max-not-sum for memory, and an
all-rows denominator. The v3 tests pinning them are ported here unchanged in
substance; the loader tests are new, because the loader is v5's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import cost as C
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING

MB = 1024 * 1024


def _cost_frame(ltd, mtl, ctr, status=None):
    n = len(ltd)
    return pd.DataFrame(
        {
            "tg_id": [f"t{i}" for i in range(n)],
            "status": status or ["SUCCESS"] * n,
            "ltd_ms": ltd,
            "mtl_ms": mtl,
            "ctr_ms": ctr,
        }
    )


# ---- the model --------------------------------------------------------------


def test_reduce_happens_per_target_before_the_percentile():
    """Per-stage medians 1/10/1 sum to 12; the median target costs 52."""
    df = _cost_frame([1.0, 1.0, 5.0], [10.0, 50.0, 10.0], [100.0, 1.0, 1.0])
    got = C.per_target_cost(df, C.COST_SPECS["runtime"])
    assert sorted(got) == [16.0, 52.0, 111.0]
    assert float(np.median(got)) == 52.0
    naive = sum(float(np.median(df[c])) for c in ("ltd_ms", "mtl_ms", "ctr_ms"))
    assert naive == 12.0


def test_max_reduce_is_at_most_the_sum():
    spec = C.COST_SPECS["runtime"]
    df = _cost_frame([3.0, 1.0], [7.0, 1.0], [2.0, 1.0])
    mx = C.per_target_cost(df, spec, reduce="max")
    sm = C.per_target_cost(df, spec, reduce="sum")
    assert mx.tolist() == [7.0, 1.0]
    assert sm.tolist() == [12.0, 3.0]


def test_an_unknown_reduce_is_refused():
    with pytest.raises(ValueError, match="reduce"):
        C.per_target_cost(_cost_frame([1.0], [1.0], [1.0]), C.COST_SPECS["runtime"], reduce="mean")


def test_a_null_stage_on_one_row_costs_nothing_but_keeps_the_row():
    df = _cost_frame([1.0, 2.0], [3.0, 4.0], [5.0, None])
    assert C.per_target_cost(df, C.COST_SPECS["runtime"]).tolist() == [9.0, 6.0]


def test_an_uninstrumented_ltd_column_is_nan_not_zero():
    """fillna(0) would manufacture a free method."""
    df = _cost_frame([None, None], [3.0, 4.0], [5.0, 6.0])
    assert np.isnan(C.per_target_cost(df, C.COST_SPECS["runtime"])).all()


def test_an_uninstrumented_ltd_column_nans_every_stage():
    df = _cost_frame([None, None], [10.0, 20.0], [100.0, 200.0])
    assert C.per_stage_cost(df, C.COST_SPECS["runtime"]).isna().all().all()


def test_legacy_single_channel_schema_fails_loudly():
    df = pd.DataFrame({"tg_id": ["t0"], "status": ["SUCCESS"], "ltd_peak_bytes": [1]})
    with pytest.raises(MissingArtifactError, match="cost columns"):
        C.per_target_cost(df, C.COST_SPECS["memory_alloc"])


def test_bytes_scale_to_mb_and_memory_max_reduces():
    df = pd.DataFrame({
        "ltd_heap_peak_bytes": [2 * MB], "mtl_heap_peak_bytes": [MB], "ctr_heap_peak_bytes": [0],
    })
    assert C.per_target_cost(df, C.COST_SPECS["memory_heap"]).tolist() == [2.0]


def test_per_stage_columns_reduce_to_the_pipeline_value():
    """`per_target_cost` is defined as `per_stage_cost` reduced row-wise."""
    frames = {
        "runtime": _cost_frame([1.0, 5.0], [10.0, 2.0], [100.0, 3.0]),
        "memory_heap": pd.DataFrame({
            "ltd_heap_peak_bytes": [1 * MB, 5 * MB],
            "mtl_heap_peak_bytes": [10 * MB, 2 * MB],
            "ctr_heap_peak_bytes": [100 * MB, 3 * MB],
        }),
    }
    for key, op in (("runtime", "sum"), ("memory_heap", "max")):
        spec = C.COST_SPECS[key]
        stages = C.per_stage_cost(frames[key], spec)
        expected = stages.sum(axis=1) if op == "sum" else stages.max(axis=1)
        assert C.per_target_cost(frames[key], spec).tolist() == expected.tolist(), key


def test_cost_stats_on_an_all_nan_input_is_nan_not_zero():
    st = C.cost_stats(np.array([np.nan, np.nan]))
    assert st["n"] == 0 and np.isnan(st["p50"]) and np.isnan(st["p5"])


def test_cost_stats_carries_the_box_quantiles():
    st = C.cost_stats(np.arange(101, dtype=float))
    assert (st["p5"], st["p25"], st["p50"], st["p75"], st["p95"]) == (5, 25, 50, 75, 95)


def test_stage_cost_table_counts_the_nulls_it_filled():
    df = _cost_frame([1.0, 2.0], [10.0, 20.0], [100.0, None], status=["SUCCESS", "FALLBACK"])
    table = C.stage_cost_table(df, C.COST_SPECS["runtime"])
    assert table["ctr"]["n_null"] == 1 and table["ltd"]["n_null"] == 0
    assert table[C.PIPELINE]["p50"] == pytest.approx(np.median([111.0, 22.0]))


def test_require_measured_refuses_an_empty_channel():
    with pytest.raises(MissingArtifactError, match="no measurements"):
        C.require_measured(C.cost_stats(np.array([np.nan])), C.COST_SPECS["memory_heap"],
                           run_id="r", method="m")


# ---- the registry -----------------------------------------------------------


def test_every_cost_spec_names_three_real_stage_columns():
    for key, spec in C.COST_SPECS.items():
        assert spec.key == key
        for stage, col in zip(C.STAGES, spec.stage_cols):
            assert col.startswith(f"{stage}_"), (key, col)
        assert spec.reduce in C.REDUCERS


def test_memory_reduces_with_max_runtime_with_sum_and_rss_is_gone():
    assert C.COST_SPECS["runtime"].reduce == "sum"
    for key in C.MEMORY_SPECS:
        assert C.COST_SPECS[key].reduce == "max"
    assert "memory_rss" not in C.COST_SPECS


# ---- the loader -------------------------------------------------------------


def _write_combo(root, run_id, combo, fold, frame):
    d = root / run_id / "generic_csv" / "setup" / f"fold_{fold}" / combo
    d.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(d / "targets.parquet")


def _bench_frame(ids, status, ctr_ms):
    n = len(ids)
    return pd.DataFrame({
        "target_id": ids, "target_lat": [0.0] * n, "target_lon": [0.0] * n,
        "pred_lat": [0.0] * n, "pred_lon": [0.0] * n, "status": status,
        "ltd_ms": [1.0] * n, "mtl_ms": [2.0] * n, "ctr_ms": ctr_ms,
        "ltd_heap_peak_bytes": [MB] * n, "mtl_heap_peak_bytes": [2 * MB] * n,
        "ctr_heap_peak_bytes": [MB] * n,
    })


@pytest.fixture
def run(tmp_path):
    _write_combo(tmp_path, "r", "vanilla_cbg", 0, _bench_frame(["a", "b"], ["SUCCESS", "FALLBACK"], [3.0, None]))
    _write_combo(tmp_path, "r", "vanilla_cbg", 1, _bench_frame(["c"], ["SUCCESS"], [3.0]))
    return RunPaths("r", tmp_path, "generic_csv", "setup")


def test_load_pools_folds_and_both_channels_share_one_frame(run):
    specs = (C.COST_SPECS["runtime"], C.COST_SPECS["memory_heap"])
    df = C.load_cost_frame(run, "vanilla_cbg", specs)
    assert sorted(df["tg_id"]) == ["a", "b", "c"]
    assert C.per_target_cost(df, specs[0]).tolist() == [6.0, 3.0, 6.0]
    assert C.per_target_cost(df, specs[1]).tolist() == [2.0, 2.0, 2.0]


def test_solved_rows_drop_the_fallback(run):
    df = C.load_cost_frame(run, "vanilla_cbg", (C.COST_SPECS["runtime"],), rows="solved")
    assert sorted(df["tg_id"]) == ["a", "c"]


def test_shortest_ping_has_no_cost(run):
    with pytest.raises(ValueError, match="no cost"):
        C.load_cost_frame(run, SHORTEST_PING, (C.COST_SPECS["runtime"],))


def test_overlapping_folds_are_refused(run, tmp_path):
    _write_combo(tmp_path, "r", "vanilla_cbg", 2, _bench_frame(["a"], ["SUCCESS"], [3.0]))
    with pytest.raises(ValueError, match="more than one fold"):
        C.load_cost_frame(run, "vanilla_cbg", (C.COST_SPECS["runtime"],))
