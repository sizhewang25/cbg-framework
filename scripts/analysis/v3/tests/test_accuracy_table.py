"""The §8.1 headline table: no pooling, method order, and the context block."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import typer

from scripts.analysis.v3.modules.accuracy_table import (
    _method_order,
    accuracy_rows,
    build,
    dataset_context,
    render_markdown,
)
from scripts.analysis.v3.modules.classify import SHORTEST_PING, TOPN_CSV
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v3.modules.proximity import TAXONOMY

_TOPN_ROWS = [
    (SHORTEST_PING, 100, 0.60, 0.80, 0.00, 50.0, 900.0),
    ("vanilla_cbg", 100, 0.40, 0.62, 0.25, 75.0, 1200.0),
    ("octant_cbg_hull", 100, 0.72, 0.93, 0.00, 90.0, 500.0),
]


def _fake_run(tmp_path, run_id, *, setup="anchors_to_probes", rows=_TOPN_ROWS, n=100):
    """A run tree with just the two artifacts this command reads."""
    run = RunPaths(
        run_id=run_id,
        root=tmp_path / "bench",
        source="generic_csv",
        setup=setup,
    )
    analysis = tmp_path / "analysis"
    cls_dir = run.cls_accuracy_dir(root=analysis, grid="h3", resolution=4)
    pd.DataFrame(
        rows,
        columns=[
            "method", "n_targets", "accuracy_top1", "accuracy_top3",
            "fallback_rate", "error_km_p50", "error_km_p90",
        ],
    ).to_csv(cls_dir / TOPN_CSV, index=False)

    prox_dir = run.proximity_dir(root=analysis, grid="h3", resolution=4)
    (prox_dir / "target_labels.csv").write_text("target_id\n")
    (prox_dir / "meta.json").write_text(
        json.dumps(
            {
                "n_targets": n,
                "n_seeds": 18,
                "n_vps": 134,
                "zero_variance": ["has_proximate_vp"],
                "section_8_2_taxonomy": {
                    TAXONOMY[0]: 10,
                    TAXONOMY[1]: 30,
                    TAXONOMY[2]: 60,
                },
            }
        )
    )
    return run, analysis


def test_one_row_per_run_and_method_with_no_pooled_average(tmp_path):
    """as01/02/03 differ by peering and routing, and that difference *is* §8.1's
    finding — an average over them would delete it to produce a tidier number.
    So the table must stay long, and no aggregate row may appear."""
    r1, analysis = _fake_run(tmp_path, "as01-260728-260802")
    r2, _ = _fake_run(tmp_path, "as02-260728-260802")
    out = accuracy_rows(
        {"as01-260728-260802": r1, "as02-260728-260802": r2},
        analysis_root=analysis,
        grid="h3",
        resolution=4,
    )
    assert len(out) == 2 * len(_TOPN_ROWS)
    assert set(out["dataset"]) == {"as01", "as02"}
    assert out.groupby("run_id").size().tolist() == [3, 3]


def test_methods_print_baseline_first(tmp_path):
    """Reading order is the argument's order: the baseline, then increasingly
    fitted variants. Alphabetical would open on Octant."""
    run, analysis = _fake_run(tmp_path, "as01-260728-260802")
    out = accuracy_rows(
        {"as01-260728-260802": run}, analysis_root=analysis, grid="h3", resolution=4
    )
    assert out["method"].tolist()[0] == SHORTEST_PING
    assert _method_order(["spotter_cbg", SHORTEST_PING])[0] == SHORTEST_PING


def test_unknown_methods_sort_after_the_known_ones(tmp_path):
    assert _method_order(["zz_ablation", SHORTEST_PING, "vanilla_cbg"]) == [
        SHORTEST_PING,
        "vanilla_cbg",
        "zz_ablation",
    ]


def test_context_is_per_run_not_per_method(tmp_path):
    """The three shares describe the dataset. Repeating them down every method row
    would read as a property of the variant, which is the misreading the separate
    table exists to prevent."""
    r1, analysis = _fake_run(tmp_path, "as01-260728-260802")
    r2, _ = _fake_run(tmp_path, "as02-260728-260802")
    ctx = dataset_context(
        {"as01-260728-260802": r1, "as02-260728-260802": r2},
        analysis_root=analysis,
        grid="h3",
        resolution=4,
    )
    assert len(ctx) == 2
    assert ctx[f"{TAXONOMY[0]}_share"].tolist() == [0.1, 0.1]
    assert sum(ctx.iloc[0][f"{t}_n"] for t in TAXONOMY) == ctx.iloc[0]["n_targets"]


def test_zero_variance_flags_are_carried_into_the_context(tmp_path):
    """A constant flag is a fact about the run's VP fleet, so it belongs beside
    the run rather than being rediscovered by every consumer."""
    run, analysis = _fake_run(tmp_path, "as01-260728-260802")
    ctx = dataset_context(
        {"as01-260728-260802": run}, analysis_root=analysis, grid="h3", resolution=4
    )
    assert ctx.iloc[0]["zero_variance_flags"] == "has_proximate_vp"


def test_mixed_setups_are_noted_not_refused(tmp_path):
    """`plot-pareto` refuses this because one frontier over both would compare a
    134-VP fleet with a 53-VP one. Here each run keeps its own row, so no such
    average can form — and on as01-03 `setup` is a placeholder anyway."""
    r1, analysis = _fake_run(tmp_path, "as01-260728-260802")
    r2, _ = _fake_run(tmp_path, "as7018_us_test01", setup="probes_to_anchors")
    _, _, _, manifest = build(
        {"as01-260728-260802": r1, "as7018_us_test01": r2},
        analysis_root=analysis,
        grid="h3",
        resolution=4,
    )
    assert manifest["setups"] == ["anchors_to_probes", "probes_to_anchors"]
    assert manifest["setup_note"].startswith("mixed setups in one table")


def test_markdown_renders_both_tables(tmp_path):
    r1, analysis = _fake_run(tmp_path, "as01-260728-260802")
    accuracy, context, md, _ = build(
        {"as01-260728-260802": r1}, analysis_root=analysis, grid="h3", resolution=4
    )
    assert "| dataset | method | n | top-1 | top-3 |" in md
    assert "## Dataset context" in md
    assert "| as01 | Shortest-Ping | 100 | 0.600 | 0.800 |" in md.replace(
        " 0.000 | 50.0 | 900.0 |", ""
    )
    # Every accuracy row reaches the render; a silent drop would be invisible.
    assert md.count("| as01 | ") == len(accuracy) + len(context)


def test_a_missing_classify_run_names_the_command_to_fix_it(tmp_path):
    run = RunPaths(
        run_id="as01-260728-260802",
        root=tmp_path / "bench",
        source="generic_csv",
        setup="anchors_to_probes",
    )
    with pytest.raises(MissingArtifactError, match="run `classify`"):
        accuracy_rows(
            {"as01-260728-260802": run},
            analysis_root=tmp_path / "analysis",
            grid="h3",
            resolution=4,
        )


def test_missing_topn_columns_are_rejected(tmp_path):
    """Reading top-3 out of a run scored with `--topn 1` would silently print an
    empty column rather than say the run needs re-scoring."""
    rows = [(SHORTEST_PING, 100, 0.60, 0.00, 50.0, 900.0)]
    run = RunPaths(
        run_id="as01-260728-260802",
        root=tmp_path / "bench",
        source="generic_csv",
        setup="anchors_to_probes",
    )
    analysis = tmp_path / "analysis"
    cls_dir = run.cls_accuracy_dir(root=analysis, grid="h3", resolution=4)
    pd.DataFrame(
        rows,
        columns=[
            "method", "n_targets", "accuracy_top1",
            "fallback_rate", "error_km_p50", "error_km_p90",
        ],
    ).to_csv(cls_dir / TOPN_CSV, index=False)
    with pytest.raises(ValueError, match="accuracy_top3"):
        accuracy_rows(
            {"as01-260728-260802": run},
            analysis_root=analysis,
            grid="h3",
            resolution=4,
        )
