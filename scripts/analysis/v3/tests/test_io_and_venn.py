"""K-fold merge guarantees and set-overlap bookkeeping."""

from __future__ import annotations

import itertools
import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis.v3.modules import io, venn

# `venn` re-exports the public surface, so almost every test below reads it
# off there. These two helpers are private to the modules that own them, and
# a test that pins their contract should say which module it is pinning.
from scripts.analysis.v3.modules.diagram.common.draw import _draw_circles
from scripts.analysis.v3.modules.diagram.euler.plot import _euler_label_points
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
    assert list(collapsed.columns) == [venn.SP_VS_CBG_LABEL, venn.CBG_ANY_LABEL]


def test_collapse_is_an_or_across_cbg_methods():
    collapsed, _ = venn.collapse_to_sp_vs_cbg(_sp_cbg_membership())
    assert list(collapsed[venn.CBG_ANY_LABEL]) == [True, False, True, False]


def test_collapsed_regions_partition_the_target_set():
    collapsed, _ = venn.collapse_to_sp_vs_cbg(_sp_cbg_membership())
    sp, cbg = collapsed[venn.SP_VS_CBG_LABEL], collapsed[venn.CBG_ANY_LABEL]
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


# ---- cross-run pooling ------------------------------------------------------


def _make_cls_dir(
    tmp_path, run_id: str, per_method: dict[str, list[bool]], targets: list[str]
) -> RunPaths:
    """A run whose `target-cls-accuracy/h3-4/` holds one parquet per method.

    Writes only the three columns `build_membership` reads. `per_method` maps a
    method id to one correct/incorrect flag per target, so a test states the
    membership it wants directly instead of reverse-engineering ranks.
    """
    run = RunPaths(run_id=run_id, root=tmp_path, source="src", setup="setup")
    cls_dir = run.cls_accuracy_dir(root=tmp_path / "analysis", grid="h3", resolution=4)
    for method, flags in per_method.items():
        status = "BASELINE" if method == venn.SHORTEST_PING else "SUCCESS"
        pd.DataFrame(
            {
                "target_id": targets,
                "status": [status] * len(targets),
                "tg_seed_rank": [0 if f else 5 for f in flags],
            }
        ).to_parquet(cls_dir / f"{method}_seed_distances.parquet", index=False)
    return run


_TWO = (venn.SHORTEST_PING, "vanilla_cbg")


def _two_run_fixture(tmp_path):
    a = _make_cls_dir(
        tmp_path, "as01", {m: [True, False] for m in _TWO}, ["t0", "t1"]
    )
    b = _make_cls_dir(
        tmp_path,
        "as02",
        {venn.SHORTEST_PING: [True, True, False], "vanilla_cbg": [False, True, True]},
        ["t0", "t1", "t2"],
    )
    return {"as01": a, "as02": b}


def test_pooling_stacks_runs_rather_than_joining_them(tmp_path):
    """Rows add up: the runs contribute disjoint targets, not shared ones."""
    runs = _two_run_fixture(tmp_path)
    pooled, origin = venn.pooled_membership(
        runs, analysis_root=tmp_path / "analysis", grid="h3", resolution=4
    )
    assert len(pooled) == 5  # 2 + 3, not a 2x3 join and not a 3-row union
    assert pooled.index.is_unique
    assert origin.value_counts().to_dict() == {"as01": 2, "as02": 3}


def test_pooled_keys_carry_the_run_so_reused_target_ids_cannot_collide(tmp_path):
    """`t0` exists in both runs; without the prefix they would be one row."""
    runs = _two_run_fixture(tmp_path)
    pooled, _ = venn.pooled_membership(
        runs, analysis_root=tmp_path / "analysis", grid="h3", resolution=4
    )
    assert "as01::t0" in pooled.index and "as02::t0" in pooled.index


def test_pooling_refuses_runs_that_scored_different_methods(tmp_path):
    """An unscored method would read as wrong on every target of that run."""
    runs = _two_run_fixture(tmp_path)
    extra = _make_cls_dir(
        tmp_path,
        "as03",
        {m: [True] for m in (*_TWO, "spotter_cbg")},
        ["t0"],
    )
    with pytest.raises(ValueError, match="spotter_cbg"):
        venn.pooled_membership(
            {**runs, "as03": extra},
            analysis_root=tmp_path / "analysis",
            grid="h3",
            resolution=4,
        )


def test_an_extra_method_is_rejected_rather_than_silently_dropped(tmp_path):
    """Subset-testing would make the result depend on which run came first."""
    runs = _two_run_fixture(tmp_path)
    extra = _make_cls_dir(
        tmp_path, "as03", {m: [True] for m in (*_TWO, "spotter_cbg")}, ["t0"]
    )
    # Same two runs, opposite order: must fail both ways, not just one.
    for ordered in ({"as03": extra, **runs}, {**runs, "as03": extra}):
        with pytest.raises(ValueError, match="different methods"):
            venn.pooled_membership(
                ordered,
                analysis_root=tmp_path / "analysis",
                grid="h3",
                resolution=4,
            )


def test_pinning_methods_explicitly_allows_a_run_to_have_extras(tmp_path):
    """--method states the shared set, so extras are the caller's to exclude."""
    runs = _two_run_fixture(tmp_path)
    extra = _make_cls_dir(
        tmp_path, "as03", {m: [True] for m in (*_TWO, "spotter_cbg")}, ["t0"]
    )
    pooled, _ = venn.pooled_membership(
        {**runs, "as03": extra},
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        methods=list(_TWO),
    )
    assert list(pooled.columns) == list(_TWO)
    assert len(pooled) == 6


def test_pooling_needs_at_least_two_runs(tmp_path):
    runs = _two_run_fixture(tmp_path)
    with pytest.raises(ValueError, match=">= 2 runs"):
        venn.pooled_membership(
            {"as01": runs["as01"]},
            analysis_root=tmp_path / "analysis",
            grid="h3",
            resolution=4,
        )


def test_pooled_intersections_still_partition_the_pooled_population(tmp_path):
    runs = _two_run_fixture(tmp_path)
    pooled, _ = venn.pooled_membership(
        runs, analysis_root=tmp_path / "analysis", grid="h3", resolution=4
    )
    assert venn.intersection_table(pooled)["n_targets"].sum() == len(pooled)


# ---- ring venn --------------------------------------------------------------


def _ring_membership(n_rows: int = 4) -> pd.DataFrame:
    """Six methods, one row per distinct pattern the assertions below name."""
    methods = [
        venn.SHORTEST_PING, "million_scale_cbg", "vanilla_cbg",
        "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg",
    ]
    # Rows, as member index sets: a single, an adjacent pair, the all-six
    # centre, and one *non*-adjacent pair the ring cannot draw.
    patterns = [{0}, {0, 1}, {0, 1, 2, 3, 4, 5}, {0, 3}]
    rows = [[i in p for i in range(6)] for p in patterns]
    return pd.DataFrame(rows, columns=methods)


def _contiguous(combo, n: int) -> bool:
    """True when `combo` is an unbroken run of circles around the ring."""
    if len(combo) == n:
        return True
    return any(
        all(((start + j) % n) in combo for j in range(len(combo)))
        for start in range(n)
    )


def test_the_ring_draws_exactly_the_contiguous_subsets():
    """31 regions at six sets, not the 63 a mathematical 6-set Venn needs.

    Pins both the count and *which* combinations are realizable, since the
    coverage table's `drawn` column and the figure's honesty both rest on it.
    """
    regions = venn.ring_regions(6)
    assert len(regions) == 31 == 6 * 5 + 1
    assert all(_contiguous(c, 6) for c in regions)
    by_size = {}
    for combo in regions:
        by_size[len(combo)] = by_size.get(len(combo), 0) + 1
    assert by_size == {1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 1}


def test_every_ring_size_yields_n_times_n_minus_one_plus_one_regions():
    """The invariant holds across the supported range, so `n` is not special."""
    for n in range(venn.RING_MIN_SETS, venn.RING_MAX_SETS + 1):
        assert len(venn.ring_regions(n)) == n * (n - 1) + 1


def test_ring_geometry_is_independent_of_the_data():
    """The figure is a template: two different datasets get identical circles.

    This is the property that lets a reader compare two ring figures directly,
    and the one an area-encoding edit would silently destroy.
    """
    before = venn.ring_centres(6)
    a = _ring_membership()
    b = pd.DataFrame(~a.to_numpy(), columns=a.columns)
    assert venn.ring_order_for(a) == venn.ring_order_for(b)
    assert venn.ring_centres(6) == before
    assert venn.RING_CIRCLE_RADIUS == 1.0
    # First circle at 12 o'clock, then clockwise.
    assert before[0][0] == pytest.approx(0.0)
    assert before[0][1] == pytest.approx(1.0 / venn.RING_RATIO)
    assert before[1][0] > 0 and before[1][1] > 0


def test_ring_labels_are_the_counts_from_the_intersection_table(tmp_path):
    """The figure and the CSV must never disagree about a region."""
    m = _ring_membership()
    order = venn.ring_order_for(m)
    exact = venn.exact_combination_counts(m, order)
    table = venn.intersection_table(m)
    for combo, n in exact.items():
        if not combo:
            continue
        key = "|".join(venn.label_for(x) for x in order if x in combo)
        assert int(table.loc[table["methods"] == key, "n_targets"].iloc[0]) == n


def test_ring_coverage_partitions_the_solved_targets():
    """`drawn` + undrawn accounts for every target at least one method solved."""
    m = _ring_membership()
    coverage = venn.ring_coverage_table(m)
    n_solved = int(m.any(axis=1).sum())
    assert coverage["n_targets"].sum() == n_solved
    # The opposite-circles pair {0, 3} is real data with nowhere to go.
    undrawn = coverage.loc[~coverage["drawn"]]
    assert len(undrawn) == 1 and int(undrawn["n_targets"].iloc[0]) == 1
    assert int(coverage.loc[coverage["drawn"], "n_targets"].sum()) == n_solved - 1


def test_ring_order_must_be_a_permutation_not_a_subset():
    """Omitting a method would hide its targets from the undrawn count too."""
    m = _ring_membership()
    with pytest.raises(ValueError, match="permutation"):
        venn.ring_order_for(m, list(m.columns)[:5])
    reversed_order = list(m.columns)[::-1]
    assert venn.ring_order_for(m, reversed_order) == reversed_order


def test_ring_venn_renders_and_names_its_undrawn_regions(tmp_path):
    m = _ring_membership()
    out = venn.plot_ring_venn(
        m, tmp_path / "ring.png", title="t", coverage_ref="cov.csv"
    )
    assert out.exists() and out.stat().st_size > 5_000


def test_the_ring_caveat_denies_that_the_geometry_carries_data():
    """The layout looks area-proportional and is not; only this text says so."""
    assert "carry no data" in venn.RING_CAVEAT


def test_ring_venn_refuses_an_empty_population(tmp_path):
    m = _ring_membership().iloc[:0]
    with pytest.raises(ValueError, match="no targets"):
        venn.plot_ring_venn(m, tmp_path / "ring.png", title="t")


# ---- euler layout -----------------------------------------------------------


def _euler_membership(patterns, methods=None) -> pd.DataFrame:
    """One row per listed member-index set, over `n` named methods."""
    methods = methods or [
        venn.SHORTEST_PING, "million_scale_cbg", "vanilla_cbg",
    ]
    rows = [[i in p for i in range(len(methods))] for p in patterns]
    return pd.DataFrame(rows, columns=methods)


def test_lens_area_spans_disjoint_to_contained():
    """The area function the whole layout is inverted through."""
    assert venn.lens_area(1.0, 1.0, 2.0) == 0.0
    assert venn.lens_area(1.0, 1.0, 5.0) == 0.0
    assert venn.lens_area(1.0, 0.5, 0.4) == pytest.approx(math.pi * 0.25)
    half = venn.lens_area(1.0, 1.0, 1.0)
    assert 0 < half < math.pi
    # Monotone in the separation, which is what makes the bisection valid.
    seq = [venn.lens_area(1.0, 0.8, d) for d in (0.2, 0.6, 1.0, 1.4, 1.8)]
    assert seq == sorted(seq, reverse=True)


def test_separation_round_trips_through_the_area_it_was_solved_for():
    for r1, r2, target in ((1.0, 1.0, 1.0), (1.0, 0.4, 0.3), (0.7, 0.9, 0.5)):
        d = venn.separation_for_overlap(r1, r2, target)
        assert venn.lens_area(r1, r2, d) == pytest.approx(target, abs=1e-6)


def test_sets_that_never_co_occur_are_drawn_apart():
    """The rule the figure is read by: no shared targets, no shared area.

    `separation_for_overlap` puts a strict gap between them rather than making
    them tangent, because tangency reads as "they only just fail to overlap",
    which is a claim about the data that an empty intersection does not make.
    """
    d = venn.separation_for_overlap(1.0, 0.6, 0.0)
    assert d > 1.6
    assert venn.lens_area(1.0, 0.6, d) == 0.0


def test_a_contained_set_is_drawn_inside_its_container():
    inner = math.pi * 0.4 ** 2
    d = venn.separation_for_overlap(1.0, 0.4, inner)
    assert d + 0.4 <= 1.0 + 1e-9


def test_circle_area_carries_the_share_never_the_radius():
    """Twice the targets must be twice the ink, not twice the width."""
    r = venn.circle_radii(np.array([0.1, 0.2]))
    assert math.pi * r[0] ** 2 == pytest.approx(0.1)
    assert (r[1] / r[0]) ** 2 == pytest.approx(2.0)


def test_combination_shares_is_the_array_twin_of_the_intersection_table():
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}])
    order = list(m.columns)
    shares = venn.combination_shares(m, order)
    assert shares.sum() == pytest.approx(1.0)
    table = venn.intersection_table(m)
    for key in range(1, 1 << len(order)):
        members = [order[i] for i in range(len(order)) if key >> i & 1]
        name = "|".join(venn.label_for(x) for x in members)
        row = table.loc[table["methods"] == name, "n_targets"]
        expected = (float(row.iloc[0]) / len(m)) if len(row) else 0.0
        assert shares[key] == pytest.approx(expected)


def _disjoint_three() -> pd.DataFrame:
    """Three methods that never get the same target right."""
    return _euler_membership([{0}, {0}, {1}, {1}, {2}, {2}, set()])


def test_the_fit_separates_circles_whose_sets_never_intersect():
    """End to end: a zero intersection has to come out as zero drawn area.

    This is the property the figure is read by, and the one a plain
    area-proportional layout does not guarantee — it is only true here because
    the objective charges `EULER_EMPTY_WEIGHT` for area drawn on an empty
    combination.
    """
    m = _disjoint_three()
    order = list(m.columns)
    layout = venn.fit_euler_layout(m, order, restarts=1, grid=140, fit_grid=500)
    for i, j in combinations(range(3), 2):
        gap = float(np.hypot(*(layout.centres[i] - layout.centres[j])))
        assert gap >= layout.radii[i] + layout.radii[j]
    # Not exactly zero: areas are sampled, so a disk's measured area carries a
    # perimeter-sized quantization error even when the topology is exact.
    assert layout.misplaced == pytest.approx(0.0, abs=0.02)


def test_the_fit_nests_a_contained_set():
    """A ⊂ B is the relationship the ring template cannot draw at all."""
    m = _euler_membership([{0, 1}, {0, 1}, {1}, {1}, {1}, set()])
    order = list(m.columns)[:2]
    layout = venn.fit_euler_layout(m[order], order, restarts=1, grid=140, fit_grid=200)
    gap = float(np.hypot(*(layout.centres[0] - layout.centres[1])))
    assert gap + min(layout.radii) <= max(layout.radii) + 0.02


def test_misplaced_and_placed_are_one_distribution_distance():
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}, set()])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=200
    )
    expected = 0.5 * float(np.abs(layout.drawn[1:] - layout.observed[1:]).sum())
    assert layout.misplaced == pytest.approx(expected)
    assert layout.placed == pytest.approx(1.0 - layout.misplaced)
    assert 0.0 <= layout.misplaced <= 1.0


def test_the_fit_is_deterministic():
    """Same matrix, same figure — the MDS start and the jitters are both seeded."""
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}])
    order = list(m.columns)
    a = venn.fit_euler_layout(m, order, restarts=2, grid=140, fit_grid=200)
    b = venn.fit_euler_layout(m, order, restarts=2, grid=140, fit_grid=200)
    assert np.allclose(a.centres, b.centres)


def test_the_fit_table_names_the_regions_the_layout_invents():
    """Empty combinations with drawn area are the figure's own error term."""
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=200
    )
    table = venn.euler_fit_table(layout, len(m))
    assert list(table.columns) == [
        "methods", "region", "n_methods", "n_targets",
        "observed_share", "drawn_share", "delta",
    ]
    assert table["n_targets"].sum() == int(m.any(axis=1).sum())
    invented = table.loc[table["n_targets"] == 0]
    assert (invented["delta"] > 0).all()


def test_a_method_that_is_never_right_is_refused_rather_than_drawn_empty():
    m = _euler_membership([{0}, {0, 1}, {1}])
    with pytest.raises(ValueError, match="no target"):
        venn.fit_euler_layout(m, list(m.columns), restarts=1, grid=140)


def _label_dist(layout, i) -> float:
    """How far set `i`'s label sits from set `i`'s own centre."""
    lx, ly = _euler_label_points(layout)[i]
    return float(np.hypot(lx - layout.centres[i][0], ly - layout.centres[i][1]))


def test_every_label_sits_on_the_circle_it_names():
    """The property that lets the figure drop leader lines.

    A label is either in its set's exclusive lobe or just inside its own
    boundary; both are within the circle. Nothing has to be connected to the
    thing it names because it is already on it.
    """
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}, {1, 2}])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=200
    )
    points = _euler_label_points(layout)
    assert len(points) == len(layout.order)
    for i in range(len(layout.order)):
        assert _label_dist(layout, i) <= layout.radii[i] + 1e-9


def test_label_points_carry_no_leader_lines():
    """Pins the contract `plot_euler` relies on: bare (x, y), not (point, leader)."""
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=200
    )
    for point in _euler_label_points(layout):
        assert len(point) == 2
        assert all(isinstance(v, float) for v in point)


def test_a_set_with_its_own_lobe_is_labelled_inside_it():
    """Well-separated sets read like the reference: the name sits in the middle.

    Three disjoint circles each have their whole disk as an exclusive lobe, so
    the pole of inaccessibility is the centre.
    """
    m = _disjoint_three()
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=500
    )
    for i in range(3):
        assert _label_dist(layout, i) < 0.35 * layout.radii[i]


def test_near_coincident_circles_still_get_separated_labels():
    """The Shortest-Ping / SoI CBG case: two sets one target apart.

    Neither has a lobe big enough to hold a name, so both go to their own
    boundary — on nearly the same bearing. They must still be told apart, and
    each must still be on its own circle.
    """
    rows = [{0, 1}] * 12 + [{1}] + [set()]
    m = _euler_membership(rows, methods=[venn.SHORTEST_PING, "million_scale_cbg"])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=300
    )
    a, b = _euler_label_points(layout)
    assert math.hypot(a[0] - b[0], a[1] - b[1]) >= venn.LABEL_MIN_GAP * 0.9
    for i in range(2):
        assert _label_dist(layout, i) <= layout.radii[i] + 1e-9


def test_each_set_label_prints_its_own_drawn_area_as_a_percentage():
    """The printed share and the ink are the same quantity, not two derivations.

    `circle_radii` sets area = share, so `pi * r**2` recovers the share exactly.
    A test rather than a comment because an edit that scaled radius by share
    would still render, just wrong by a square.
    """
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}, set()])
    order = list(m.columns)
    layout = venn.fit_euler_layout(m, order, restarts=1, grid=140, fit_grid=200)
    for i, method in enumerate(order):
        printed = math.pi * layout.radii[i] ** 2
        assert printed == pytest.approx(float(m[method].mean()), abs=1e-9)


def test_only_sets_carry_percentages_never_intersections():
    """Set areas are exact; intersection areas are fitted and would disagree."""
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}, set()])
    order = list(m.columns)
    layout = venn.fit_euler_layout(m, order, restarts=1, grid=140, fit_grid=200)
    # Every singleton is drawn to its observed share...
    for i in range(len(order)):
        assert math.pi * layout.radii[i] ** 2 == pytest.approx(
            float(layout.observed[[k for k in range(1, 1 << len(order))
                                   if k >> i & 1]].sum()), abs=1e-9
        )
    # ...while at least one multi-set region is not, which is why none is
    # labelled and the caption reports the aggregate error instead.
    assert layout.misplaced > 0.0
    assert venn.OUTSIDE_LABEL == "None"


def test_circles_are_stacked_largest_first_so_small_sets_stay_visible():
    """Insertion order is the z-order at equal `zorder`, so this is the mechanism.

    Without it a 60%-of-the-population circle drawn after a 34% one buries the
    smaller set's outline in the larger's fill.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    order = ["a", "b", "c"]
    radii = np.array([0.2, 0.9, 0.5])
    centres = np.zeros((3, 2))
    fig, ax = plt.subplots()
    _draw_circles(ax, order, {m: "#000000" for m in order}, centres, radii)
    drawn = [p.get_radius() for p in ax.patches if isinstance(p, Circle)]
    plt.close(fig)
    # Two passes (fills then outlines), each descending by radius.
    assert drawn == [0.9, 0.5, 0.2, 0.9, 0.5, 0.2]


def test_equal_radii_leave_the_callers_order_untouched():
    """The ring shares this helper and its order is meaningful, not incidental."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    order = ["a", "b", "c"]
    centres = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    fig, ax = plt.subplots()
    _draw_circles(
        ax, order, {m: "#000000" for m in order}, centres, np.array([0.5, 0.5, 0.5])
    )
    xs = [p.center[0] for p in ax.patches if isinstance(p, Circle)]
    plt.close(fig)
    assert xs == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]


def test_the_outside_label_carries_the_never_correct_share():
    """`observed[0]` is the population outside every circle."""
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}, set(), set()])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=200
    )
    assert float(layout.observed[0]) == pytest.approx(2 / 6)
    assert venn.OUTSIDE_LABEL == "None"


def test_the_caption_is_a_choice_and_names_the_error_when_on(tmp_path):
    """Default on: the layout misplaces targets and must say so."""
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=200
    )
    on = venn.plot_euler(layout, tmp_path / "on.png", title="t")
    off = venn.plot_euler(layout, tmp_path / "off.png", title="t", caption=False)
    assert on.exists() and off.exists()
    # The caption is real ink: dropping it makes a strictly smaller image.
    assert off.stat().st_size < on.stat().st_size


def test_euler_figure_renders_and_states_its_own_error(tmp_path):
    m = _euler_membership([{0}, {0, 1}, {0, 1, 2}, {2}])
    layout = venn.fit_euler_layout(
        m, list(m.columns), restarts=1, grid=140, fit_grid=200
    )
    out = venn.plot_euler(layout, tmp_path / "euler.png", title="t")
    assert out.exists() and out.stat().st_size > 5_000
    # The caption states the one thing the geometry cannot say about itself:
    # what the areas mean. The fit's own error is added per-figure on top.
    assert "Circle area" in venn.EULER_CAPTION
    assert "share both get right" in venn.EULER_CAPTION


# ---- cross-run artifact naming ----------------------------------------------


def test_grid_slug_is_in_cross_names_and_absent_from_per_run_names():
    """The cross dir is keyed by dataset set alone, so it needs the slug."""
    per_run = venn.artifact_name("overlap_upset", "png", 1)
    cross_run = venn.artifact_name("overlap_upset", "png", 1, "h3-4")
    assert per_run == "overlap_upset.top1.png"  # unchanged: files exist on disk
    assert cross_run == "overlap_upset.h3-4.top1.png"
    assert cross_run != venn.artifact_name("overlap_upset", "png", 1, "healpix-128")


def test_cross_dir_is_order_independent_and_separates_kinds(tmp_path):
    from scripts.analysis.v3.modules import cross

    a = cross.cross_dir(tmp_path, ["as02", "as01"], kind="venn-diagram")
    b = cross.cross_dir(tmp_path, ["as01", "as02"], kind="venn-diagram")
    assert a == b and a.name == "as01+as02"
    assert cross.cross_dir(tmp_path, ["as01"], kind="cost-accuracy") != cross.cross_dir(
        tmp_path, ["as01"], kind="venn-diagram"
    )


def test_render_cross_overlap_writes_every_artifact_with_the_grid_slug(tmp_path):
    pytest.importorskip("upsetplot")
    runs = _two_run_fixture(tmp_path)
    out_dir = tmp_path / "cross"
    out_dir.mkdir()
    written = venn.render_cross_overlap(
        runs,
        out_dir,
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        top_n=1,
    )
    assert {"membership", "intersections", "pairwise",
            "venn", "manifest"} <= set(written)
    # Two methods is under the ring's arity floor, so it is skipped rather than
    # drawn degenerately — `plot_venn` already renders two sets exactly.
    assert "ring_venn" not in written
    assert all(p.exists() for p in written.values())
    assert all("h3-4.top1" in p.name for p in written.values())

    import json

    manifest = json.loads(written["manifest"].read_text())
    assert manifest["n_targets_total"] == 5
    assert manifest["n_targets_per_run"] == {"as01": 2, "as02": 3}
    # The membership CSV must say which run each pooled row came from.
    assert "run_id" in pd.read_csv(written["membership"]).columns


def test_pooled_render_draws_the_ring_once_there_are_enough_methods(tmp_path):
    """Six methods across two runs: the ring and its coverage table both land."""
    methods = [
        venn.SHORTEST_PING, "million_scale_cbg", "vanilla_cbg",
        "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg",
    ]
    runs = {
        "as01": _make_cls_dir(
            tmp_path, "as01",
            {m: [True, i % 2 == 0] for i, m in enumerate(methods)},
            ["t0", "t1"],
        ),
        "as02": _make_cls_dir(
            tmp_path, "as02",
            {m: [i < 3, True] for i, m in enumerate(methods)},
            ["t0", "t1"],
        ),
    }
    out_dir = tmp_path / "cross6"
    out_dir.mkdir()
    written = venn.render_cross_overlap(
        runs, out_dir, analysis_root=tmp_path / "analysis", grid="h3", resolution=4
    )
    assert {"ring_venn", "ring_coverage"} <= set(written)
    assert all(p.exists() for p in written.values())

    coverage = pd.read_csv(written["ring_coverage"])
    manifest = json.loads(written["manifest"].read_text())
    assert manifest["ring_order"] == methods
    assert manifest["n_regions_drawn"] == int(coverage["drawn"].sum())
    assert manifest["n_targets_undrawn"] == int(
        coverage.loc[~coverage["drawn"], "n_targets"].sum()
    )
    # Coverage plus the never-solved targets is the whole pooled population.
    assert (
        coverage["n_targets"].sum() + manifest["n_targets_none_correct"]
        == manifest["n_targets_total"]
    )


# ---- rescue view (the baseline's failures only) ------------------------------
#
# Two changes at once, and neither alone answers the question: drop the baseline
# column, and keep only the rows the baseline got wrong. The full matrix can say
# that CBG rescues what Shortest-Ping loses; only this population can say which
# variants do it, and in shares of what there was to rescue.


_SIX_METHODS = [
    venn.SHORTEST_PING, "million_scale_cbg", "vanilla_cbg",
    "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg",
]


def _rescue_flags():
    """Two runs whose restricted population exercises every branch.

    Hand-set rather than generated: the assertions below name exact counts, so
    the flags have to be readable as the answer. Shortest-Ping is right on 3 of
    the 9 targets, leaving 6; `t4` is rescued by nobody, so the "none correct"
    region is non-empty; and all five variants rescue at least one target, so
    the Euler fit has no zero-radius circle to refuse.
    """
    as01 = {
        venn.SHORTEST_PING: [True, False, False, False, False],
        "million_scale_cbg": [False, True, False, False, False],
        "vanilla_cbg": [False, True, True, False, False],
        "octant_cbg_hull": [False, False, True, True, False],
        "octant_cbg_spl": [True, True, False, True, False],
        "spotter_cbg": [False, False, True, False, False],
    }
    as02 = {
        venn.SHORTEST_PING: [True, True, False, False],
        "million_scale_cbg": [False, False, True, False],
        "vanilla_cbg": [False, False, False, True],
        "octant_cbg_hull": [True, False, True, False],
        "octant_cbg_spl": [False, False, False, True],
        "spotter_cbg": [False, False, True, True],
    }
    return as01, as02


def _rescue_runs(tmp_path):
    as01, as02 = _rescue_flags()
    return {
        "as01": _make_cls_dir(tmp_path, "as01", as01, ["t0", "t1", "t2", "t3", "t4"]),
        "as02": _make_cls_dir(tmp_path, "as02", as02, ["t0", "t1", "t2", "t3"]),
    }


def _rescue_single_run(tmp_path, flags=None, targets=None):
    """One run, plus the cls_dir `render_overlap` writes into."""
    as01, _ = _rescue_flags()
    run = _make_cls_dir(
        tmp_path, "as01", flags or as01, targets or ["t0", "t1", "t2", "t3", "t4"]
    )
    return run, run.cls_accuracy_dir(
        root=tmp_path / "analysis", grid="h3", resolution=4
    )


def _rescue_membership():
    """Six methods, Shortest-Ping right on `t0` alone.

    `_ring_membership` is unusable here: Shortest-Ping is a member of every one
    of its patterns, so the restricted population would be empty and each
    assertion below would hold vacuously.
    """
    as01, _ = _rescue_flags()
    return pd.DataFrame(
        {m: as01[m] for m in _SIX_METHODS}, index=[f"t{i}" for i in range(5)]
    )


def test_the_rescue_population_is_exactly_the_baselines_failures():
    m = _rescue_membership()
    rescue = venn.restrict_to_baseline_failures(m)
    assert list(rescue.index) == ["t1", "t2", "t3", "t4"]
    assert list(rescue.index) == list(m.index[~m[venn.SHORTEST_PING]])
    assert len(rescue) == len(m) - int(m[venn.SHORTEST_PING].sum())


def test_the_rescue_filter_drops_the_baseline_because_its_set_is_empty():
    """Not a second decision — over these rows the baseline is all-False."""
    m = _rescue_membership()
    rescue = venn.restrict_to_baseline_failures(m)
    assert venn.SHORTEST_PING not in rescue.columns
    assert len(rescue) == 4  # not vacuous: there are rows to check
    # The column it dropped would have contributed nothing but an empty circle.
    kept = m.loc[rescue.index, venn.SHORTEST_PING]
    assert not kept.any()


def test_rescue_shares_are_denominated_in_the_restricted_population():
    """The partition invariant, over the smaller total."""
    rescue = venn.restrict_to_baseline_failures(_rescue_membership())
    table = venn.intersection_table(rescue)
    assert table["n_targets"].sum() == len(rescue)
    assert table["share"].sum() == pytest.approx(1.0, abs=5e-4)


def test_the_rescue_filter_requires_the_baseline():
    m = _rescue_membership().drop(columns=[venn.SHORTEST_PING])
    with pytest.raises(ValueError, match="rescue view is defined"):
        venn.restrict_to_baseline_failures(m)


def test_render_overlap_writes_the_full_artifact_set(tmp_path):
    """The per-run path had no test; this is the baseline for the shared writer."""
    pytest.importorskip("upsetplot")
    run, cls_dir = _rescue_single_run(tmp_path)
    written = venn.render_overlap(run, cls_dir, top_n=1)
    assert {"membership", "intersections", "pairwise",
            "venn_spec", "venn", "upset"} <= set(written)
    assert all(p.exists() for p in written.values())
    # Per-run names carry the top-N but never the grid slug: the directory
    # they land in is already `target-cls-accuracy/h3-4/`.
    assert all("top1" in p.name and "h3-4" not in p.name for p in written.values())
    # No ring or Euler in the per-run directory, even at six methods.
    assert not {"ring_venn", "euler"} & set(written)


def test_rescue_view_is_off_by_default(tmp_path):
    pytest.importorskip("upsetplot")
    run, cls_dir = _rescue_single_run(tmp_path)
    written = venn.render_overlap(run, cls_dir, top_n=1)
    assert not [k for k in written if k.startswith("rescue_")]
    assert not list(cls_dir.glob("rescue_*"))


def test_render_overlap_adds_the_rescue_set_under_prefixed_keys(tmp_path):
    pytest.importorskip("upsetplot")
    run, cls_dir = _rescue_single_run(tmp_path)
    written = venn.render_overlap(run, cls_dir, top_n=1, rescue_view=True)
    assert {"rescue_membership", "rescue_intersections", "rescue_pairwise",
            "rescue_venn_spec", "rescue_upset"} <= set(written)
    # No baseline column survives the filter, so the Shortest-Ping collapse is
    # the one artifact the rescue pass drops — its left set would be empty by
    # construction.
    assert "rescue_venn" not in written
    assert all(
        p.name.startswith("rescue_overlap")
        for k, p in written.items() if k.startswith("rescue_")
    )
    assert written["rescue_upset"].stat().st_size > 5_000
    rescue = pd.read_csv(written["rescue_membership"], index_col=0)
    assert venn.LABELS[venn.SHORTEST_PING] not in rescue.columns
    assert len(rescue) == 4  # 5 targets, Shortest-Ping right on t0


def test_the_rescue_pass_never_touches_the_full_population_files(tmp_path):
    """New filenames, not an overwrite: the full view must be bit-for-bit equal."""
    pytest.importorskip("upsetplot")
    run_a, dir_a = _rescue_single_run(tmp_path / "plain")
    plain = venn.render_overlap(run_a, dir_a, top_n=1)
    baseline = {k: p.read_bytes() for k, p in plain.items()}

    run_b, dir_b = _rescue_single_run(tmp_path / "withrescue")
    both = venn.render_overlap(run_b, dir_b, top_n=1, rescue_view=True)
    for key, blob in baseline.items():
        assert both[key].read_bytes() == blob, key
    assert len(pd.read_csv(both["rescue_membership"], index_col=0)) < len(
        pd.read_csv(both["membership"], index_col=0)
    )


def test_cross_rescue_drops_the_sp_collapse_and_keeps_everything_else(tmp_path):
    pytest.importorskip("upsetplot")
    out_dir = tmp_path / "cross"
    out_dir.mkdir()
    written = venn.render_cross_overlap(
        _rescue_runs(tmp_path),
        out_dir,
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        rescue_view=True,
    )
    assert {"rescue_membership", "rescue_intersections", "rescue_pairwise",
            "rescue_venn_spec", "rescue_upset", "rescue_ring_coverage",
            "rescue_ring_venn", "rescue_euler_fit",
            "rescue_euler"} <= set(written)
    assert "rescue_venn" not in written
    assert all(p.exists() for p in written.values())
    assert all("h3-4.top1" in p.name for p in written.values())


def test_the_rescue_membership_csv_still_says_which_run_each_row_came_from(tmp_path):
    pytest.importorskip("upsetplot")
    out_dir = tmp_path / "cross"
    out_dir.mkdir()
    written = venn.render_cross_overlap(
        _rescue_runs(tmp_path),
        out_dir,
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        rescue_view=True,
    )
    frame = pd.read_csv(written["rescue_membership"])
    assert "run_id" in frame.columns
    # The restricted rows, split by origin: as01 loses t0, as02 loses t0 and t1.
    assert frame["run_id"].value_counts().to_dict() == {"as01": 4, "as02": 2}


def test_the_rescue_ring_order_is_the_full_order_minus_the_baseline(tmp_path):
    """A 6-method --ring-order cannot be reused verbatim on 5 columns."""
    pytest.importorskip("upsetplot")
    out_dir = tmp_path / "cross"
    out_dir.mkdir()
    reversed_order = list(reversed(_SIX_METHODS))
    written = venn.render_cross_overlap(
        _rescue_runs(tmp_path),
        out_dir,
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        ring_order=reversed_order,
        rescue_view=True,
    )
    manifest = json.loads(written["manifest"].read_text())
    assert manifest["ring_order"] == reversed_order
    # Filtered, not revalidated — the caller's clockwise sequence is preserved.
    assert manifest["rescue_view"]["ring_order"] == [
        m for m in reversed_order if m != venn.SHORTEST_PING
    ]


def test_rescue_coverage_partitions_the_restricted_population(tmp_path):
    pytest.importorskip("upsetplot")
    out_dir = tmp_path / "cross"
    out_dir.mkdir()
    written = venn.render_cross_overlap(
        _rescue_runs(tmp_path),
        out_dir,
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        rescue_view=True,
    )
    coverage = pd.read_csv(written["rescue_ring_coverage"])
    block = json.loads(written["manifest"].read_text())["rescue_view"]
    assert (
        coverage["n_targets"].sum() + block["n_targets_none_correct"]
        == block["n_targets"]
    )
    assert block["n_regions_drawn"] == int(coverage["drawn"].sum())


def test_the_manifest_names_the_rescue_denominator(tmp_path):
    """The block's own strings are what stop a cross-denominator comparison."""
    pytest.importorskip("upsetplot")
    out_dir = tmp_path / "cross"
    out_dir.mkdir()
    written = venn.render_cross_overlap(
        _rescue_runs(tmp_path),
        out_dir,
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        rescue_view=True,
    )
    manifest = json.loads(written["manifest"].read_text())
    block = manifest["rescue_view"]
    assert block["denominator"] == "restricted"
    assert block["baseline"] == venn.SHORTEST_PING
    assert venn.SHORTEST_PING not in block["methods"]
    # 9 pooled targets, Shortest-Ping right on 3 of them.
    assert block["n_targets"] == 6 == manifest["n_targets_total"] - 3
    assert block["n_targets_per_run"] == {"as01": 4, "as02": 2}
    assert block["n_rescued_by_any"] == 5  # t4 is rescued by nobody
    assert block["n_targets_none_correct"] == 1
    assert block["share_rescued_by_any"] == round(5 / 6, 4)
    assert block["n_correct_per_method"] == {
        "million_scale_cbg": 2, "vanilla_cbg": 3, "octant_cbg_hull": 3,
        "octant_cbg_spl": 3, "spotter_cbg": 3,
    }


def test_rescue_view_refuses_a_population_the_baseline_solved_entirely(tmp_path):
    run, cls_dir = _rescue_single_run(
        tmp_path,
        flags={m: [True, True] for m in _SIX_METHODS},
        targets=["t0", "t1"],
    )
    with pytest.raises(ValueError, match="nothing left to rescue"):
        venn.render_overlap(run, cls_dir, top_n=1, rescue_view=True)


def test_rescue_view_needs_two_cbg_variants_to_overlap(tmp_path):
    run, cls_dir = _rescue_single_run(tmp_path)
    with pytest.raises(ValueError, match=">= 2 CBG variants"):
        venn.render_overlap(
            run, cls_dir, methods=[venn.SHORTEST_PING, "vanilla_cbg"],
            top_n=1, rescue_view=True,
        )


def test_rescue_view_requires_the_baseline_among_the_chosen_methods(tmp_path):
    """`--method` can exclude the very column the view is defined against."""
    run, cls_dir = _rescue_single_run(tmp_path)
    with pytest.raises(ValueError, match="needs the .* baseline"):
        venn.render_overlap(
            run, cls_dir,
            methods=["vanilla_cbg", "spotter_cbg", "octant_cbg_hull"],
            top_n=1, rescue_view=True,
        )


def test_the_guards_fire_before_any_artifact_is_written(tmp_path):
    """A refused rescue view must not leave a half-written full pass behind."""
    run, cls_dir = _rescue_single_run(tmp_path)
    with pytest.raises(ValueError, match=">= 2 CBG variants"):
        venn.render_overlap(
            run, cls_dir, methods=[venn.SHORTEST_PING, "vanilla_cbg"],
            top_n=1, rescue_view=True,
        )
    assert not list(cls_dir.glob("overlap_*"))


def test_euler_is_skipped_rather_than_crashing_when_a_variant_rescues_nothing(
    tmp_path,
):
    """A zero-radius circle is refused by the fit; the ring draws it as 0.0%."""
    pytest.importorskip("upsetplot")
    as01, as02 = _rescue_flags()
    # Spotter now only ever agrees with the baseline, so it rescues nothing.
    as01["spotter_cbg"] = list(as01[venn.SHORTEST_PING])
    as02["spotter_cbg"] = list(as02[venn.SHORTEST_PING])
    runs = {
        "as01": _make_cls_dir(tmp_path, "as01", as01, ["t0", "t1", "t2", "t3", "t4"]),
        "as02": _make_cls_dir(tmp_path, "as02", as02, ["t0", "t1", "t2", "t3"]),
    }
    out_dir = tmp_path / "cross"
    out_dir.mkdir()
    written = venn.render_cross_overlap(
        runs,
        out_dir,
        analysis_root=tmp_path / "analysis",
        grid="h3",
        resolution=4,
        rescue_view=True,
    )
    assert "rescue_ring_venn" in written
    assert "rescue_euler" not in written and "rescue_euler_fit" not in written
    block = json.loads(written["manifest"].read_text())["rescue_view"]
    assert block["ring_drawn"] is True
    assert block["euler_drawn"] is False
    assert block["euler_skipped_methods"] == ["spotter_cbg"]
    assert block["euler_placed_share"] is None
    # The full population still has its Euler figure: only the restricted one
    # has an empty set.
    assert "euler" in written


# ---- venn spec (generic Venn-tool input) ------------------------------------


def _spec_membership() -> pd.DataFrame:
    """Three sets with a non-empty triple, so cumulative != exclusive."""
    return pd.DataFrame(
        {
            venn.SHORTEST_PING: [1, 1, 1, 0, 0, 0],
            "vanilla_cbg": [1, 1, 0, 1, 0, 0],
            "spotter_cbg": [1, 0, 0, 0, 1, 0],
        }
    ).astype(bool)


def test_venn_spec_has_a_value_for_every_combination_of_two_or_more():
    """A drawing tool needs all 2**n - 1 - n relations, empties included."""
    spec = venn.venn_spec(_spec_membership())
    assert spec["number_of_sets"] == 3
    assert [s["id"] for s in spec["sets"]] == ["A", "B", "C"]
    assert [s["size"] for s in spec["sets"]] == [3, 3, 2]
    assert set(spec["relations"]) == {"A ^ B", "A ^ C", "B ^ C", "A ^ B ^ C"}
    assert len(spec["relations"]) == 2**3 - 1 - 3


def test_venn_spec_relations_are_cumulative_by_default():
    """`A ^ B` is the full intersection — it counts the targets also in C."""
    spec = venn.venn_spec(_spec_membership())
    assert spec["relation_convention"] == "cumulative"
    assert spec["relations"]["A ^ B"] == 2  # rows 0 and 1
    assert spec["relations"]["A ^ B ^ C"] == 1  # row 0, counted in A^B too


def test_venn_spec_cumulative_relations_satisfy_inclusion_exclusion():
    """The property that lets a tool solve for the regions from this document."""
    m = _spec_membership()
    spec = venn.venn_spec(m)
    terms = {s["id"]: s["size"] for s in spec["sets"]}
    terms.update(spec["relations"])
    ids = [s["id"] for s in spec["sets"]]
    union = 0
    for k in range(1, len(ids) + 1):
        for combo in itertools.combinations(ids, k):
            union += (1 if k % 2 else -1) * terms[" ^ ".join(combo)]
    assert union == spec["n_targets"] - spec["n_none"] == int(m.any(axis=1).sum())


def test_venn_spec_exclusive_relations_partition_the_population():
    """The other convention: disjoint regions that sum back to n_targets."""
    m = _spec_membership()
    spec = venn.venn_spec(m, exclusive=True)
    assert spec["relation_convention"] == "exclusive"
    # A^B now excludes the triple, unlike the cumulative reading above.
    assert spec["relations"]["A ^ B"] == 1
    singles = {
        s["id"]: int((m[s["method"]] & ~m.drop(columns=[s["method"]]).any(axis=1)).sum())
        for s in spec["sets"]
    }
    assert (
        sum(singles.values()) + sum(spec["relations"].values()) + spec["n_none"]
        == spec["n_targets"]
    )


def test_venn_spec_exclusive_relations_match_the_intersection_table():
    """One population, two artifacts; they must not disagree about a region."""
    m = _spec_membership()
    spec = venn.venn_spec(m, exclusive=True)
    table = venn.intersection_table(m)
    by_id = {s["id"]: s["method"] for s in spec["sets"]}
    for key, n in spec["relations"].items():
        members = [by_id[i] for i in key.split(" ^ ")]
        label = "|".join(venn.label_for(x) for x in members)
        row = table.loc[table["methods"] == label, "n_targets"]
        assert (int(row.iloc[0]) if len(row) else 0) == n


def test_venn_spec_refuses_more_sets_than_it_has_letters():
    m = pd.DataFrame({f"m{i}": [True] for i in range(len(venn.SET_IDS) + 1)})
    with pytest.raises(ValueError, match="at most"):
        venn.venn_spec(m)
