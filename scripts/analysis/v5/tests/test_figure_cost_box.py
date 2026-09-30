"""The cost box figure: paired runtime/memory boxes, whiskers at p5/p95."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import cost as C
from scripts.analysis.v5.modules import figure_cost_box as B
from scripts.analysis.v5.modules.paths import RunPaths

MB = 1024 * 1024


def _frame(prefix, n, *, ms, heap_mb, fallback=0):
    status = ["FALLBACK"] * fallback + ["SUCCESS"] * (n - fallback)
    ctr = [None] * fallback + [1.0] * (n - fallback)
    return pd.DataFrame({
        "target_id": [f"{prefix}{i}" for i in range(n)],
        "target_lat": 0.0, "target_lon": 0.0, "pred_lat": 0.0, "pred_lon": 0.0,
        "status": status,
        "ltd_ms": 1.0, "mtl_ms": np.asarray(ms, dtype=float), "ctr_ms": ctr,
        "ltd_heap_peak_bytes": MB // 10, "mtl_heap_peak_bytes": np.asarray(heap_mb) * MB,
        "ctr_heap_peak_bytes": MB // 10,
        "ltd_alloc_peak_bytes": MB // 10, "mtl_alloc_peak_bytes": MB // 10,
        "ctr_alloc_peak_bytes": MB // 10,
    })


def _make_run(root, run_id, prefix, combos=("vanilla_cbg", "octant_cbg_hull")):
    for combo in combos:
        d = root / run_id / "generic_csv" / "setup" / "fold_0" / combo
        d.mkdir(parents=True, exist_ok=True)
        _frame(prefix, 101, ms=np.arange(101), heap_mb=np.linspace(1, 3, 101),
               fallback=10 if combo == "vanilla_cbg" else 0).to_parquet(d / "targets.parquet")
    return RunPaths(run_id, root, "generic_csv", "setup")


@pytest.fixture
def runs(tmp_path):
    bench = tmp_path / "bench"
    return [_make_run(bench, "ra", "a"), _make_run(bench, "rb", "b")]


def test_names_carry_channel_rows_and_pooled_infix():
    assert B.artifact_names(B.PER_RUN, "memory_heap", "all")["png"] == "cost_box.heap.png"
    assert (B.artifact_names(B.POOLED, "memory_alloc", "solved")["csv"]
            == "cost_box.pooled.alloc.solved.csv")


def test_bad_channel_or_rows_is_refused():
    with pytest.raises(ValueError, match="memory channel"):
        B.validate("memory_rss", "all")
    with pytest.raises(ValueError, match="rows"):
        B.validate("memory_heap", "some")


def test_box_whiskers_are_p5_and_p95():
    block = C.cost_stats(np.arange(101, dtype=float))
    box = B._box(block)
    assert (box["whislo"], box["q1"], box["med"], box["q3"], box["whishi"]) == (5, 25, 50, 75, 95)
    assert box["fliers"] == []


def test_shortest_ping_is_never_drawn(runs):
    assert B.costed_methods(runs[0], None) == ["vanilla_cbg", "octant_cbg_hull"]
    assert B.costed_methods(runs[0], ["shortest_ping", "vanilla_cbg"]) == ["vanilla_cbg"]
    with pytest.raises(ValueError, match="no cost"):
        B.costed_methods(runs[0], ["shortest_ping"])
    with pytest.raises(ValueError, match="holds no combo"):
        B.costed_methods(runs[0], ["spotter_cbg"])


def test_runtime_and_memory_share_rows_and_fallback_counts(runs):
    table = B.stats_table(B.load_run(runs[0], memory="memory_heap", rows="all"),
                          memory="memory_heap")
    van = table[(table["method"] == "vanilla_cbg") & (table["stage"] == C.PIPELINE)]
    assert set(van["n_rows"]) == {101} and set(van["n_solved"]) == {91}
    ctr = table[(table["method"] == "vanilla_cbg") & (table["stage"] == "ctr")
                & (table["channel"] == "runtime")]
    assert int(ctr["n_null"].iloc[0]) == 10


def test_solved_rows_shrink_both_channels(runs):
    table = B.stats_table(B.load_run(runs[0], memory="memory_heap", rows="solved"),
                          memory="memory_heap")
    van = table[(table["method"] == "vanilla_cbg") & (table["stage"] == C.PIPELINE)]
    assert set(van["n"]) == {91}


def test_pooling_concatenates_rows(runs):
    by_run = {r.run_id: B.load_run(r, memory="memory_heap", rows="all") for r in runs}
    pooled = B.stack_runs(by_run)
    assert len(pooled["vanilla_cbg"]) == 202


def test_pooling_refuses_a_method_missing_from_one_run(tmp_path):
    bench = tmp_path / "bench"
    a = _make_run(bench, "ra", "a")
    b = _make_run(bench, "rb", "b", combos=("vanilla_cbg",))
    by_run = {r.run_id: B.load_run(r, memory="memory_heap", rows="all") for r in (a, b)}
    with pytest.raises(ValueError, match="not scored in every run"):
        B.stack_runs(by_run)


def test_pooling_refuses_shared_tg_ids(tmp_path):
    bench = tmp_path / "bench"
    by_run = {
        r.run_id: B.load_run(r, memory="memory_heap", rows="all")
        for r in (_make_run(bench, "ra", "a"), _make_run(bench, "rb", "a"))
    }
    with pytest.raises(ValueError, match="share"):
        B.stack_runs(by_run)


def test_every_artifact_is_written_and_the_manifest_reads_back(runs, tmp_path):
    out = tmp_path / "analysis"
    sets = B.build_for_runs(runs, layouts=(B.PER_RUN, B.POOLED), analysis_root=out)
    assert len(sets) == 3
    for written in sets:
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0
    manifest = json.loads(sets[-1]["manifest"].read_text())
    assert manifest["layout"] == B.POOLED
    assert manifest["box"]["whiskers"] == "p5 / p95"
    assert manifest["pooling"]["n_tgs_by_run"] == {"ra": 101, "rb": 101}
    assert "shortest_ping" in manifest["excluded"]
    csv = pd.read_csv(sets[-1]["csv"])
    assert list(csv.columns) == list(B.CSV_COLUMNS)
    pipe = csv[(csv["stage"] == C.PIPELINE) & (csv["channel"] == "runtime")
               & (csv["method"] == "octant_cbg_hull")]
    # ltd 1 + mtl 0..100 + ctr 1: p5 over two identical runs is 5 + 2.
    assert float(pipe["p5"].iloc[0]) == pytest.approx(7.0)
