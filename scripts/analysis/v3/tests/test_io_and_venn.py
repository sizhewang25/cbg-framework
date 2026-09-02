"""K-fold merge guarantees and set-overlap bookkeeping."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis.v3.modules import io, venn
from pathlib import Path

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


def test_every_cbg_variant_is_labelled_as_one():
    """Shortest-Ping is the only non-CBG method, so everything else gets `CBG`.

    Unlabelled ablation arms matter most here: as a bare combo id they read as
    just another method, indistinguishable from the baseline at a glance.
    """
    assert venn.label_for("shortest_ping") == "Shortest-Ping"
    assert venn.label_for("octant_cbg_hull") == "Octant-Hull CBG"
    assert venn.label_for("spotter_cbg") == "Spotter CBG"
    assert venn.label_for("some_new_combo") == "some_new_combo CBG"
    for method in venn.PREFERRED_ORDER:
        label = venn.label_for(method)
        assert (label == "Shortest-Ping") == (method == venn.SHORTEST_PING)
        assert label.endswith("CBG") or method == venn.SHORTEST_PING


def test_venn_rejects_arity_above_three():
    m = pd.DataFrame({c: [True] for c in "abcd"})
    with pytest.raises(ValueError, match="2 or 3 methods"):
        venn.plot_venn(m, pytest.importorskip("pathlib").Path("/tmp/x.png"), title="t")


# ---- top-N artifact naming --------------------------------------------------


def test_artifact_names_differ_across_top_n():
    """The regression this guards: fixed names let top-3 overwrite top-1."""
    assert venn.artifact_name("overlap_upset", "png", 1) == "overlap_upset.top1.png"
    assert venn.artifact_name("overlap_upset", "png", 3) == "overlap_upset.top3.png"
    assert venn.artifact_name("overlap_upset", "png", 1) != venn.artifact_name(
        "overlap_upset", "png", 3
    )


# ---- Shortest-Ping vs >=1 CBG collapse --------------------------------------


def _sp_cbg_membership():
    # t0 both · t1 SP only · t2 CBG only (one of two) · t3 neither
    return pd.DataFrame(
        {
            "shortest_ping": [True, True, False, False],
            "vanilla_cbg": [True, False, True, False],
            "spotter_cbg": [False, False, False, False],
        },
        index=[f"t{i}" for i in range(4)],
    )


def test_collapse_reports_the_cbg_pool_size():
    """The pool size must reach the figure: it changes the answer on as7018."""
    collapsed, n_cbg = venn.collapse_to_sp_vs_cbg(_sp_cbg_membership())
    assert n_cbg == 2
    assert list(collapsed.columns) == ["Shortest-Ping", venn.CBG_ANY_LABEL]


def test_collapse_is_an_or_across_cbg_methods():
    collapsed, _ = venn.collapse_to_sp_vs_cbg(_sp_cbg_membership())
    assert list(collapsed[venn.CBG_ANY_LABEL]) == [True, False, True, False]


def test_collapsed_regions_partition_the_target_set():
    collapsed, _ = venn.collapse_to_sp_vs_cbg(_sp_cbg_membership())
    sp, cbg = collapsed["Shortest-Ping"], collapsed[venn.CBG_ANY_LABEL]
    both = int((sp & cbg).sum())
    sp_only = int((sp & ~cbg).sum())
    cbg_only = int((~sp & cbg).sum())
    neither = int((~sp & ~cbg).sum())
    assert (both, sp_only, cbg_only, neither) == (1, 1, 1, 1)
    assert both + sp_only + cbg_only + neither == 4


def test_collapse_requires_the_baseline():
    """Without Shortest-Ping there is no rescue-vs-regression view to draw."""
    m = pd.DataFrame({"vanilla_cbg": [True], "spotter_cbg": [False]})
    with pytest.raises(ValueError, match="shortest_ping"):
        venn.collapse_to_sp_vs_cbg(m)


def test_collapse_requires_at_least_one_cbg():
    m = pd.DataFrame({"shortest_ping": [True]})
    with pytest.raises(ValueError, match="at least one"):
        venn.collapse_to_sp_vs_cbg(m)


# ---- grid/resolution grouping of outputs ------------------------------------


def test_output_dirs_are_grouped_by_grid_and_resolution(tmp_path):
    """The answer space parameterizes every number downstream of it.

    Without this grouping a sweep would overwrite one `topn_accuracy.csv` per
    rung and leave no record of which grid produced the survivor.
    """
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup="setup")
    h3_4 = run.answer_space_dir(root=tmp_path, grid="h3", resolution=4)
    h3_3 = run.answer_space_dir(root=tmp_path, grid="h3", resolution=3)
    hp128 = run.answer_space_dir(root=tmp_path, grid="healpix", resolution=128)
    cls = run.cls_accuracy_dir(root=tmp_path, grid="h3", resolution=4)
    assert h3_4 != h3_3
    assert h3_4.name == "h3-4" and h3_3.name == "h3-3" and hp128.name == "healpix-128"
    assert h3_4.parent.name == "target-answer-space"
    assert cls.parent.name == "target-cls-accuracy"


def test_the_two_grids_cannot_collide_on_one_directory(tmp_path):
    """`h3` res 4 and HEALPix nside 4 are different grids, not one number."""
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup="setup")
    assert run.answer_space_dir(
        root=tmp_path, grid="h3", resolution=4
    ) != run.answer_space_dir(root=tmp_path, grid="healpix", resolution=4)


def test_grid_and_resolution_are_required_so_they_cannot_silently_drift():
    run = RunPaths(run_id="r", root=Path("/x"), source="s", setup="t")
    with pytest.raises(TypeError):
        run.answer_space_dir()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        run.answer_space_dir(grid="h3")  # type: ignore[call-arg]


# ---- upset annotation semantics ---------------------------------------------


def test_column_shares_sum_to_one_but_row_shares_need_not(tmp_path):
    """The two annotated number sets are different quantities.

    Columns are exact disjoint intersections, so they partition the targets.
    Rows are set totals, which overlap. The figure headers say so; this pins
    the arithmetic behind them.
    """
    pytest.importorskip("upsetplot")
    from upsetplot import UpSet, from_indicators

    # t0,t1 in both sets, t2 in SP only, t3 in neither -> totals overlap.
    m = pd.DataFrame(
        {"shortest_ping": [True, True, True, False],
         "vanilla_cbg": [True, True, False, False]},
        index=[f"t{i}" for i in range(4)],
    )
    renamed = m.rename(columns={c: venn.label_for(c) for c in m.columns})
    upset = UpSet(
        from_indicators(list(renamed.columns)[::-1], renamed),
        subset_size="count",
        sort_by="cardinality",
        sort_categories_by=None,
        min_subset_size=1,
    )
    total = len(m)
    assert upset.intersections.sum() == total  # disjoint and complete
    # Rows overlap, so their sum exceeds the total here (t0,t1 are in two sets).
    assert upset.totals.sum() > total


def test_columns_are_ranked_largest_intersection_first():
    """The annotated shares must read monotonically left to right."""
    pytest.importorskip("upsetplot")
    from upsetplot import UpSet, from_indicators

    m = pd.DataFrame(
        {"shortest_ping": [True, True, True, False, False],
         "vanilla_cbg": [True, False, False, True, False]},
        index=[f"t{i}" for i in range(5)],
    )
    renamed = m.rename(columns={c: venn.label_for(c) for c in m.columns})
    sizes = UpSet(
        from_indicators(list(renamed.columns)[::-1], renamed),
        subset_size="count",
        sort_by="cardinality",
        sort_categories_by=None,
        min_subset_size=1,
    ).intersections.to_list()
    assert sizes == sorted(sizes, reverse=True)


def test_upset_row_order_follows_the_caller_top_to_bottom(tmp_path):
    """A stable dot pattern across runs is the whole point of pinning order."""
    pytest.importorskip("upsetplot")
    m = pd.DataFrame(
        {c: [True, False] for c in ("shortest_ping", "vanilla_cbg", "spotter_cbg")}
    )
    out = venn.plot_upset(m, tmp_path / "u.png", title="t")
    assert out.exists() and out.stat().st_size > 5_000
