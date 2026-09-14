"""`plot-phase-cost` — the per-stage decomposition and its guards.

The figure's whole reason for existing is that the legacy stacked bars had no
statistic under them, so the tests here pin the properties that make the
replacement honest: one denominator across the four blocks, a pipeline value
that is the *reduce* rather than the sum, nulls surfaced rather than filled to a
plausible-looking zero, and filenames that carry every axis.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis.v3.modules import cost as C
from scripts.analysis.v3.modules import phase_cost as P
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths


def _make_run(tmp_path, *, combos=("vanilla_cbg",), folds=2, channel="heap", null_ctr=False):
    """A run carrying real byte columns, unlike `test_pareto._make_run`.

    `null_ctr` reproduces the FALLBACK shape: CTR never ran, so its column is
    NULL rather than 0 — the case the figure must not draw as "cheap".
    `channel` picks which of the three memory column families is populated; the
    other two are written all-NULL, which is exactly what a run of the wrong
    vintage looks like on disk.
    """
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup="setup")
    k = 0
    for f in range(folds):
        for combo in combos:
            d = run.combo_dir(combo, f"fold_{f}")
            d.mkdir(parents=True, exist_ok=True)
            n = 3
            cols = {
                "target_id": [f"t{k + i}" for i in range(n)],
                "status": ["SUCCESS", "SUCCESS", "FALLBACK" if null_ctr else "SUCCESS"],
                "ltd_ms": [1.0, 2.0, 3.0],
                "mtl_ms": [10.0, 20.0, 30.0],
                "ctr_ms": [0.5, 0.5, None if null_ctr else 0.5],
            }
            MB = 1024 * 1024
            for fam in ("alloc", "heap", "rss"):
                on = fam == channel
                cols[f"ltd_{fam}_peak_bytes"] = [1 * MB, 1 * MB, 1 * MB] if on else [None] * n
                cols[f"mtl_{fam}_peak_bytes"] = [4 * MB, 2 * MB, 2 * MB] if on else [None] * n
                cols[f"ctr_{fam}_peak_bytes"] = (
                    ([2 * MB, 2 * MB, None if null_ctr else 2 * MB]) if on else [None] * n
                )
            pq.write_table(pa.table(cols), d / "targets.parquet")
        k += 3
    return run


def _spec(key="memory_heap"):
    return C.COST_SPECS[key]


# ---- the four blocks -------------------------------------------------------

def test_every_stage_shares_one_denominator(tmp_path):
    """The audited defect: legacy bars and their reference line came from two
    different row sets. One `combo_stage_costs` call means one `load_folds` and
    one `rows` filter, so this cannot drift."""
    run = _make_run(tmp_path)
    df = P.collect(run, ["vanilla_cbg"], _spec(), rows="all", reduce="max")
    assert df["n_rows"].nunique() == 1
    assert set(df["stage"]) == {"ltd", "mtl", "ctr", C.PIPELINE}


def test_pipeline_is_the_reduce_not_the_sum_of_the_bars(tmp_path):
    """The claim the figure's pipeline rule makes. For memory the reduce is a
    per-target max, so it lands on the dearest stage — never on the total, which
    is what a stacked bar would have shown."""
    run = _make_run(tmp_path)
    df = P.collect(run, ["vanilla_cbg"], _spec(), rows="all", reduce="max")
    by = df.set_index("stage")["p50"]
    stages = by[["ltd", "mtl", "ctr"]]
    assert by[C.PIPELINE] == pytest.approx(max(stages))
    assert by[C.PIPELINE] < stages.sum()


def test_a_stage_is_summarised_over_the_rows_where_it_ran(tmp_path):
    """The deliberate two-denominator split. A null CTR fills to 0 for the
    pipeline's arithmetic (that target really did peak in MTL), but 0 has no
    position on a log axis and would drag the box's lower hinge to the floor.
    So the stage box drops those rows and says so; the pipeline box keeps them."""
    run = _make_run(tmp_path, null_ctr=True)
    df = P.collect(run, ["vanilla_cbg"], _spec(), rows="all", reduce="max")
    ctr = df[df.stage == "ctr"].iloc[0]
    pipe = df[df.stage == C.PIPELINE].iloc[0]
    assert ctr["n_null"] == 2                 # one per fold
    assert ctr["n"] == ctr["n_rows"] - 2      # box excludes them
    assert pipe["n"] == pipe["n_rows"]        # reduce keeps them
    assert df[df.stage == "mtl"].iloc[0]["n_null"] == 0
    # And no box hinge sits at zero, which is what makes it plottable on log.
    assert ctr["p5"] > 0


def test_the_box_carries_every_percentile(tmp_path):
    """`--cost-stat` is gone, so the row must hold the whole five-number
    summary rather than one chosen statistic."""
    run = _make_run(tmp_path)
    df = P.collect(run, ["vanilla_cbg"], _spec(), rows="all", reduce="max")
    for k in ("p5", "p25", "p50", "p75", "p95"):
        assert k in df.columns and np.isfinite(df[k]).all()
    row = df[df.stage == "mtl"].iloc[0]
    assert row["p5"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p95"]


# ---- the guards ------------------------------------------------------------

def test_a_channel_present_but_all_null_raises(tmp_path):
    """`memory_rss` on a run carrying the heap channel. Columns exist, so
    `_scaled_stages` returns None and the cost is NaN; without the guard the
    figure plotted an empty frontier and said nothing."""
    run = _make_run(tmp_path, channel="heap")
    with pytest.raises(MissingArtifactError, match="entirely NULL"):
        P.collect(run, ["vanilla_cbg"], _spec("memory_rss"), rows="all", reduce="max")


def test_an_absent_channel_raises_the_schema_error_not_arrow(tmp_path):
    """The friendly error in `_scaled_stages` was unreachable through
    `combo_stage_costs`: pyarrow raised `ArrowInvalid` on the unknown column
    first. `_rows_for` now intersects with the real schema so the message that
    names the columns survives."""
    run = _make_run(tmp_path)
    for d in run.setup_dir.glob("fold_*/vanilla_cbg"):
        t = pq.read_table(d / "targets.parquet")
        keep = [n for n in t.column_names if "heap_peak" not in n]
        pq.write_table(t.select(keep), d / "targets.parquet")
    with pytest.raises(MissingArtifactError, match="cost columns"):
        P.collect(run, ["vanilla_cbg"], _spec(), rows="all", reduce="max")


def test_deprecated_channel_warns(tmp_path):
    seen = []
    C.warn_if_deprecated(_spec("memory_rss"), echo=seen.append)
    assert seen and "DEPRECATED" in seen[0] and "memory_heap" in seen[0]
    seen.clear()
    C.warn_if_deprecated(_spec("memory_heap"), echo=seen.append)
    assert not seen


# ---- naming ----------------------------------------------------------------

def test_artifact_names_differ_across_every_parameter():
    """`pareto.artifact_name` omits rows/reduce, so its sweeps overwrite each
    other. Not inherited here. `cost_stat` is deliberately absent: the box shows
    every percentile, so it is no longer an axis."""
    variants = {
        P.artifact_name("memory_heap", "png", rows="all", reduce="max"),
        P.artifact_name("runtime", "png", rows="all", reduce="max"),
        P.artifact_name("memory_heap", "png", rows="solved", reduce="max"),
        P.artifact_name("memory_heap", "png", rows="all", reduce="sum"),
        P.artifact_name("memory_heap", "csv", rows="all", reduce="max"),
    }
    assert len(variants) == 5


# ---- end to end ------------------------------------------------------------

def test_figure_and_siblings_are_written(tmp_path):
    run = _make_run(tmp_path, combos=("vanilla_cbg", "octant_cbg_hull"))
    df = P.collect(run, ["vanilla_cbg", "octant_cbg_hull"], _spec(), rows="all", reduce="max")
    out = tmp_path / "phase_cost.png"
    P.plot(df, _spec(), out, rows="all", reduce="max", run_id="r")
    assert out.exists() and out.stat().st_size > 5_000


def test_method_order_follows_identity_not_magnitude(tmp_path):
    """Colour and position follow the entity, so a filter cannot reshuffle the
    survivors — the same rule `pareto` is held to."""
    ordered = P._order_methods(["spotter_cbg", "vanilla_cbg", "million_scale_cbg"])
    assert ordered == ["million_scale_cbg", "vanilla_cbg", "spotter_cbg"]
    assert P._order_methods(["zzz_unknown", "vanilla_cbg"]) == ["vanilla_cbg", "zzz_unknown"]
