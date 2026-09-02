"""K-fold merge guarantees and set-overlap bookkeeping."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis.v3.modules import io, venn
from scripts.analysis.v3.modules.paths import RunPaths


def _make_run(tmp_path, fold_targets: dict[str, list[str]], combo="c1") -> RunPaths:
    """A minimal on-disk run: one combo, one targets.parquet per fold."""
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup="setup")
    for fold_id, ids in fold_targets.items():
        d = run.combo_dir(combo, fold_id)
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "target_id": ids,
                    "target_lat": [40.0] * len(ids),
                    "target_lon": [-80.0] * len(ids),
                    "pred_lat": [40.0] * len(ids),
                    "pred_lon": [-80.0] * len(ids),
                    "status": ["SUCCESS"] * len(ids),
                    "error_km": [1.0] * len(ids),
                }
            ),
            d / "targets.parquet",
        )
    return run


def test_folds_merge_and_are_labelled(tmp_path):
    run = _make_run(tmp_path, {"fold_0": ["a", "b"], "fold_1": ["c"]})
    df = io.load_folds(run, "c1")
    assert len(df) == 3
    assert df.groupby("fold").size().to_dict() == {0: 2, 1: 1}
    assert (df["combo_id"] == "c1").all()
    assert list(df.columns[:2]) == ["combo_id", "fold"]


def test_fold_order_is_numeric_not_lexicographic(tmp_path):
    """fold_10 must not sort between fold_1 and fold_2."""
    run = _make_run(
        tmp_path, {f"fold_{i}": [f"t{i}"] for i in (0, 1, 2, 10)}
    )
    assert run.fold_ids == ["fold_0", "fold_1", "fold_2", "fold_10"]
    assert list(io.load_folds(run, "c1")["fold"]) == [0, 1, 2, 10]


def test_overlapping_fold_test_sets_are_rejected(tmp_path):
    """Disjointness is the K-fold protocol; pooling a leak would double-count."""
    run = _make_run(tmp_path, {"fold_0": ["a", "b"], "fold_1": ["b"]})
    with pytest.raises(ValueError, match="more than one fold"):
        io.load_folds(run, "c1")


def test_a_missing_fold_is_an_error_not_a_silent_drop(tmp_path):
    run = _make_run(tmp_path, {"fold_0": ["a"], "fold_1": ["b"]})
    (run.combo_dir("c1", "fold_1") / "targets.parquet").unlink()
    with pytest.raises(io.MissingArtifactError, match="missing targets.parquet"):
        io.load_folds(run, "c1")


def test_fold_index_parsing():
    assert io.fold_index("fold_7") == 7


# ---- venn bookkeeping -------------------------------------------------------


def _membership():
    return pd.DataFrame(
        {
            "shortest_ping": [True, True, False, False],
            "vanilla_cbg": [True, False, True, False],
        },
        index=[f"t{i}" for i in range(4)],
    )


def test_intersections_partition_the_target_set():
    """Exact intersections must sum to the full denominator."""
    table = venn.intersection_table(_membership())
    assert table["n_targets"].sum() == 4
    assert table["share"].sum() == pytest.approx(1.0)


def test_intersection_labels_use_display_names():
    table = venn.intersection_table(_membership())
    assert "Shortest-Ping|Vanilla CBG" in set(table["methods"])
    assert "(none)" in set(table["methods"])


def test_pairwise_counts_are_complete_and_symmetricly_split():
    row = venn.pairwise_table(_membership()).iloc[0]
    assert row["both"] + row["a_only"] + row["b_only"] + row["neither"] == 4
    # a_only is the regression column when a is the baseline.
    assert row["a_only"] == 1
    assert row["b_only"] == 1
    # jaccard is rounded to 4dp on write, so compare at that precision.
    assert row["jaccard"] == pytest.approx(1 / 3, abs=1e-4)


def test_label_falls_back_to_the_raw_id():
    assert venn.label_for("shortest_ping") == "Shortest-Ping"
    assert venn.label_for("some_new_combo") == "some_new_combo"


def test_venn_rejects_arity_above_three():
    m = pd.DataFrame({c: [True] for c in "abcd"})
    with pytest.raises(ValueError, match="2 or 3 methods"):
        venn.plot_venn(m, pytest.importorskip("pathlib").Path("/tmp/x.png"), title="t")
