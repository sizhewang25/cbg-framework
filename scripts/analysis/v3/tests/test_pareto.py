"""Cost model, encoding, and artifact naming for `plot-pareto`.

The cost-model tests live in `test_cost.py` -- `cost.py` owns those policies
now, since three commands depend on them.

The encoding tests exist because colour now carries variant identity, which
makes colour assignment a correctness question rather than a styling one: a
palette that shifts when the method pool is filtered silently relabels every
mark in the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis.v3.modules import pareto as P
from scripts.analysis.v3.modules.cost import COST_SPECS
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths


# ---------------------------------------------------------------------------
# accuracy input
# ---------------------------------------------------------------------------


def _acc_csv(tmp_path, methods=("shortest_ping", "vanilla_cbg"), ns=(1, 3)):
    data = {
        "method": list(methods),
        "n_targets": [412] * len(methods),
        "n_solved": [412] * len(methods),
        "n_fallback": [0] * len(methods),
        "fallback_rate": [0.0] * len(methods),
    }
    for i, n in enumerate(ns):
        data[f"accuracy_top{n}"] = [0.3 + 0.1 * i + 0.01 * j for j in range(len(methods))]
    p = tmp_path / "topn_accuracy.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p


def test_load_accuracy_picks_the_requested_n(tmp_path):
    df = P.load_accuracy(_acc_csv(tmp_path), top_n=3)
    assert list(df.index) == ["shortest_ping", "vanilla_cbg"]
    assert df.loc["shortest_ping", "accuracy"] == pytest.approx(0.4)


def test_an_unreported_top_n_names_the_available_ones(tmp_path):
    with pytest.raises(MissingArtifactError, match=r"reports top-N \[1, 3\]"):
        P.load_accuracy(_acc_csv(tmp_path), top_n=2)


def test_available_top_ns_ignores_other_columns(tmp_path):
    assert P.available_top_ns(pd.read_csv(_acc_csv(tmp_path))) == [1, 3]


def test_a_duplicated_method_row_is_ambiguous_and_raises(tmp_path):
    p = _acc_csv(tmp_path, methods=("vanilla_cbg", "vanilla_cbg"))
    with pytest.raises(ValueError, match="more than once"):
        P.load_accuracy(p, top_n=1)


# ---------------------------------------------------------------------------
# naming — every axis that parameterizes the numbers must be in the path
# ---------------------------------------------------------------------------


def _name(**kw):
    base = dict(cost_key="runtime", grid="healpix", resolution=128, top_n=1, ext="csv")
    return P.artifact_name(**{**base, **kw})


def test_artifact_names_differ_across_every_parameter():
    """The regression this guards: a sweep silently overwriting its own output."""
    base = _name()
    assert base == "pareto_runtime.healpix-128.top1.csv"
    for kw in (
        dict(top_n=3),
        dict(cost_key="memory_alloc"),
        dict(grid="h3", resolution=4),
        dict(resolution=64),
        dict(amortized=True),
        dict(x="throughput"),
        dict(ext="png"),
    ):
        assert _name(**kw) != base, kw


def test_dataset_sets_get_separate_directories(tmp_path):
    """The regression this guards: a single-run figure replacing the
    three-dataset one, since the filename cannot say which datasets it used."""
    a = P.cross_dir(tmp_path, ["as01-260728-260802", "as02-260728-260802"])
    b = P.cross_dir(tmp_path, ["as7018_us_test01"])
    assert a != b
    assert a.name == "as01+as02" and b.name == "as7018_us_test01"


def test_a_long_dataset_set_is_hashed_rather_than_unusable():
    slug = P.dataset_set_slug([f"run-{i:03d}-abcdefghij" for i in range(30)])
    assert len(slug) < 40 and slug.startswith("30sets-")


def test_short_label_drops_the_redundant_cbg_suffix():
    assert P.short_label("vanilla_cbg") == "Vanilla"
    assert P.short_label(SHORTEST_PING) == "Shortest-Ping"
    assert P.short_label("octant_cbg_top_geo") == "octant_cbg_top_geo"


# ---------------------------------------------------------------------------
# baseline policy
# ---------------------------------------------------------------------------


def test_shortest_ping_is_charged_exactly_zero_on_both_axes():
    row = pd.Series({"accuracy": 0.37, "n_targets": 412})
    for key in ("runtime", "memory_alloc", "memory_rss"):
        got = P.shortest_ping_row("as02", row, COST_SPECS[key])
        assert got["cost"] == 0.0 and got["cost_basis"] == "analytical"
        assert bool(got["is_baseline"])


# ---------------------------------------------------------------------------
# assembly + figure, on a synthetic on-disk run
# ---------------------------------------------------------------------------


def _make_run(tmp_path, *, combos=("vanilla_cbg",), folds=2, setup="setup"):
    run = RunPaths(run_id="r", root=tmp_path, source="src", setup=setup)
    k = 0
    for f in range(folds):
        for combo in combos:
            d = run.combo_dir(combo, f"fold_{f}")
            d.mkdir(parents=True, exist_ok=True)
            n = 3
            ids = [f"t{k + i}" for i in range(n)]
            pq.write_table(
                pa.table(
                    {
                        "target_id": ids,
                        "status": ["SUCCESS"] * n,
                        "ltd_ms": [1.0, 2.0, 3.0],
                        "mtl_ms": [10.0, 20.0, 30.0],
                        "ctr_ms": [0.5, 0.5, 0.5],
                    }
                ),
                d / "targets.parquet",
            )
        k += 3
    return run


def test_end_to_end_emits_a_csv_and_a_readable_figure(tmp_path):
    run = _make_run(tmp_path, combos=("vanilla_cbg",))
    csv = _acc_csv(tmp_path, methods=("shortest_ping", "vanilla_cbg"))
    long, notes = P.load_cost_accuracy(
        {"r": csv}, {"r": run}, top_n=1, spec=COST_SPECS["runtime"]
    )
    assert set(long["method"]) == {SHORTEST_PING, "vanilla_cbg"}
    wide = P.aggregate_methods(long, top_n=1)
    assert wide["accuracy_range"].max() == pytest.approx(0.0)  # one dataset

    out = P.order_columns(wide, top_n=1)
    assert list(out.columns)[:3] == ["method", "label", "is_baseline"]
    # The Pareto columns were removed with the frontier encoding.
    assert not {"on_frontier", "dominated_by", "frontier_margin_rel"} & set(out.columns)

    png = P.plot_pareto(
        wide, long, tmp_path / "f.png",
        spec=COST_SPECS["runtime"], top_n=1, subtitle="test",
    )
    assert png.exists() and png.stat().st_size > 5_000


def test_a_zero_cost_method_survives_the_log_axis(tmp_path):
    """The regression this guards: dropping the free baseline entirely because
    a log axis cannot render x=0."""
    run = _make_run(tmp_path)
    csv = _acc_csv(tmp_path, methods=("shortest_ping", "vanilla_cbg"))
    long, _ = P.load_cost_accuracy(
        {"r": csv}, {"r": run}, top_n=1, spec=COST_SPECS["runtime"]
    )
    wide = P.aggregate_methods(long, top_n=1)
    assert (wide.loc[wide["method"] == SHORTEST_PING, "cost"] == 0.0).all()
    png = P.plot_pareto(
        wide, long, tmp_path / "z.png",
        spec=COST_SPECS["runtime"], top_n=1, subtitle="test",
    )
    assert png.stat().st_size > 5_000


def test_a_method_missing_from_one_dataset_is_flagged(tmp_path):
    """Partial coverage has to be visible: that dataset's polyline skips the
    variant rather than interpolating across a gap it never measured."""
    r1 = _make_run(tmp_path / "a", combos=("vanilla_cbg", "spotter_cbg"))
    r2 = _make_run(tmp_path / "b", combos=("vanilla_cbg",))
    c1 = _acc_csv(tmp_path / "a", methods=("vanilla_cbg", "spotter_cbg"))
    c2 = _acc_csv(tmp_path / "b", methods=("vanilla_cbg", "spotter_cbg"))
    with pytest.warns(UserWarning, match="no combo dir on disk"):
        long, notes = P.load_cost_accuracy(
            {"r1": c1, "r2": c2}, {"r1": r1, "r2": r2},
            top_n=1, spec=COST_SPECS["runtime"],
        )
    wide = P.aggregate_methods(long, top_n=1).set_index("method")
    assert bool(wide.loc["spotter_cbg", "partial_coverage"])
    assert wide.loc["spotter_cbg", "n_datasets"] == 1
    assert not bool(wide.loc["vanilla_cbg", "partial_coverage"])
    assert wide.loc["vanilla_cbg", "n_datasets"] == 2
    assert notes


def test_strict_membership_turns_the_warning_into_a_failure(tmp_path):
    r1 = _make_run(tmp_path / "a", combos=("vanilla_cbg",))
    c1 = _acc_csv(tmp_path / "a", methods=("vanilla_cbg", "spotter_cbg"))
    with pytest.raises(MissingArtifactError, match="no combo dir on disk"):
        P.load_cost_accuracy(
            {"r1": c1}, {"r1": r1}, top_n=1,
            spec=COST_SPECS["runtime"], strict_membership=True,
        )


def test_amortizing_fit_into_peak_memory_is_refused(tmp_path):
    """A one-time fit's *peak* memory is not additive across targets."""
    run = _make_run(tmp_path)
    csv = _acc_csv(tmp_path, methods=("vanilla_cbg",))
    with pytest.raises(ValueError, match="runtime only"):
        P.load_cost_accuracy(
            {"r": csv}, {"r": run}, top_n=1,
            spec=COST_SPECS["memory_alloc"], amortize_fit=True,
        )


def test_accuracy_range_is_the_cross_dataset_spread(tmp_path):
    r1 = _make_run(tmp_path / "a")
    r2 = _make_run(tmp_path / "b")
    c1 = tmp_path / "a" / "one.csv"
    c2 = tmp_path / "b" / "two.csv"
    for p, acc in ((c1, 0.30), (c2, 0.55)):
        pd.DataFrame(
            {
                "method": ["vanilla_cbg"], "n_targets": [10], "n_solved": [10],
                "n_fallback": [0], "fallback_rate": [0.0], "accuracy_top1": [acc],
            }
        ).to_csv(p, index=False)
    long, _ = P.load_cost_accuracy(
        {"r1": c1, "r2": c2}, {"r1": r1, "r2": r2},
        top_n=1, spec=COST_SPECS["runtime"],
    )
    wide = P.aggregate_methods(long, top_n=1)
    assert wide["accuracy_range"].iloc[0] == pytest.approx(0.25)
    assert {"accuracy_top1_r1", "accuracy_top1_r2"} <= set(wide.columns)


def test_mixed_setups_are_refused_without_the_override(tmp_path):
    """SCHEMA.md §7: the two run families swap the VP and target roles."""
    typer = pytest.importorskip("typer")
    runs = {
        "op": RunPaths(run_id="op", root=tmp_path, source="s", setup="anchors_to_probes"),
        "ripe": RunPaths(run_id="ripe", root=tmp_path, source="s", setup="probes_to_anchors"),
    }
    with pytest.raises(typer.BadParameter, match="swap the VP/target roles"):
        P._guard_one_setup(runs, allow_mixed=False)
    P._guard_one_setup(runs, allow_mixed=True)  # override is honoured


# ---------------------------------------------------------------------------
# encoding — colour is variant identity, symbol/linestyle is dataset
# ---------------------------------------------------------------------------

_PUBLISHED = [
    "shortest_ping",
    "million_scale_cbg",
    "vanilla_cbg",
    "octant_cbg_hull",
    "octant_cbg_spl",
    "spotter_cbg",
]


def test_the_six_published_variants_each_get_their_own_hue():
    got = P.method_colors(_PUBLISHED)
    assert len(set(got.values())) == 6
    assert P._C_OTHER not in got.values()


def test_filtering_the_method_pool_does_not_repaint_the_survivors():
    """The regression this guards is the recolor-on-filter anti-pattern: colour
    must follow the entity, never its rank within the current selection.

    Assigning hues by position among the methods *present* meant that dropping
    one variant shifted the hue of every variant after it, silently relabelling
    every mark in the figure.
    """
    full = P.method_colors(_PUBLISHED)
    for drop in _PUBLISHED:
        kept = [m for m in _PUBLISHED if m != drop]
        assert P.method_colors(kept) == {m: full[m] for m in kept}, drop


def test_hue_order_follows_preferred_order_not_input_order():
    forward = P.method_colors(_PUBLISHED)
    reversed_ = P.method_colors(list(reversed(_PUBLISHED)))
    assert forward == reversed_


def test_the_two_octant_spline_ids_share_one_hue():
    """`octant_cbg_spl` and `octant_cbg` are one paper variant whose combo id
    differs per run, exactly as venn.LABELS already encodes."""
    got = P.method_colors(["octant_cbg_spl", "octant_cbg"])
    assert got["octant_cbg_spl"] == got["octant_cbg"]
    assert got["octant_cbg"] != P._C_OTHER


def test_ablation_arms_fold_into_one_other_bucket():
    """17 series cannot be coloured, so the published set keeps its hues and the
    rest share the de-emphasis grey rather than being handed generated ones."""
    methods = _PUBLISHED + [
        "vanilla_cbg_geo", "octant_cbg_top", "spotter_cbg_c80",
        "spotter_cbg_c100_geo", "million_scale_cbg_geo",
    ]
    got = P.method_colors(methods)
    assert all(got[m] != P._C_OTHER for m in _PUBLISHED)
    assert {got[m] for m in methods if m not in _PUBLISHED} == {P._C_OTHER}


def test_every_variant_hue_is_a_distinct_validated_slot():
    assert len(set(P._VARIANT_HUES)) == len(P._VARIANT_HUES)
    assert P._C_OTHER not in P._VARIANT_HUES


# ---- dataset polylines ----


def _spread(methods_costs, per_dataset):
    """(wide, long) for the given {method: cost} and {run: {method: acc}}."""
    wide = pd.DataFrame(
        {
            "method": list(methods_costs),
            "cost": [methods_costs[m] for m in methods_costs],
        }
    )
    rows = [
        {"run_id": r, "method": m, "accuracy": a}
        for r, accs in per_dataset.items()
        for m, a in accs.items()
    ]
    return wide, pd.DataFrame(rows)


def test_dataset_lines_yields_one_cost_ascending_polyline_per_dataset():
    wide, long = _spread(
        {"b": 100.0, "a": 10.0, "c": 1000.0},
        {"r1": {"a": 0.1, "b": 0.2, "c": 0.3}, "r2": {"a": 0.4, "b": 0.5, "c": 0.6}},
    )
    lines = P.dataset_lines(wide, long)
    assert set(lines) == {"r1", "r2"}
    for xs, ys in lines.values():
        assert np.all(np.diff(xs) > 0), "vertices must run cheap -> expensive"
        assert len(xs) == 3 and len(ys) == 3
    assert lines["r1"][1].tolist() == [0.1, 0.2, 0.3]


def test_a_variant_one_dataset_lacks_is_skipped_in_that_line_only():
    """The regression this guards: interpolating a dataset's polyline across a
    variant it never measured, which draws a number that does not exist."""
    wide, long = _spread(
        {"a": 10.0, "b": 100.0, "c": 1000.0},
        {"r1": {"a": 0.1, "b": 0.2, "c": 0.3}, "r2": {"a": 0.4, "c": 0.6}},
    )
    lines = P.dataset_lines(wide, long)
    assert lines["r1"][0].tolist() == [10.0, 100.0, 1000.0]
    assert lines["r2"][0].tolist() == [10.0, 1000.0]


def test_dataset_lines_drops_non_finite_points():
    wide, long = _spread(
        {"a": 10.0, "b": float("nan")},
        {"r1": {"a": 0.1, "b": 0.2}},
    )
    assert P.dataset_lines(wide, long)["r1"][0].tolist() == [10.0]


def test_the_zero_cost_baseline_is_a_polyline_vertex_like_any_other():
    wide, long = _spread(
        {SHORTEST_PING: 0.0, "vanilla_cbg": 40.0},
        {"r1": {SHORTEST_PING: 0.37, "vanilla_cbg": 0.41}},
    )
    xs, _ = P.dataset_lines(wide, long)["r1"]
    assert xs.tolist() == [0.0, 40.0]


# ---- figure smoke ----


def test_figure_renders_with_one_dataset_so_the_cost_bar_is_degenerate(tmp_path):
    run = _make_run(tmp_path, combos=("vanilla_cbg",))
    csv = _acc_csv(tmp_path, methods=("shortest_ping", "vanilla_cbg"))
    long, _ = P.load_cost_accuracy(
        {"r": csv}, {"r": run}, top_n=1, spec=COST_SPECS["runtime"]
    )
    wide = P.aggregate_methods(long, top_n=1)
    assert (wide["cost_min"] == wide["cost_max"]).all()  # nothing to span
    out = P.plot_pareto(
        wide, long, tmp_path / "one.png",
        spec=COST_SPECS["runtime"], top_n=1, subtitle="one dataset",
    )
    assert out.stat().st_size > 5_000


def test_figure_renders_with_the_other_bucket_populated(tmp_path):
    combos = tuple(_PUBLISHED[1:]) + ("vanilla_cbg_geo", "octant_cbg_top")
    run = _make_run(tmp_path, combos=combos)
    csv = _acc_csv(tmp_path, methods=("shortest_ping",) + combos)
    long, _ = P.load_cost_accuracy(
        {"r": csv}, {"r": run}, top_n=1, spec=COST_SPECS["runtime"]
    )
    wide = P.aggregate_methods(long, top_n=1)
    colors = P.method_colors(list(wide["method"]))
    assert sum(1 for v in colors.values() if v == P._C_OTHER) == 2
    out = P.plot_pareto(
        wide, long, tmp_path / "many.png",
        spec=COST_SPECS["runtime"], top_n=1, subtitle="overflow",
    )
    assert out.stat().st_size > 5_000


def test_throughput_maps_a_zero_cost_to_nan_not_inf(tmp_path):
    """The regression this pins: `inf` throughput propagating into the axis
    limits and raising "Axis limits cannot be NaN or Inf".

    Shortest-Ping costs 0, so its throughput is unbounded and no finite axis can
    place it. It drops out of the throughput figure (and keeps its exact `inf`
    in the CSV) rather than taking the whole render down.
    """
    run = _make_run(tmp_path, combos=("vanilla_cbg",))
    csv = _acc_csv(tmp_path, methods=("shortest_ping", "vanilla_cbg"))
    long, _ = P.load_cost_accuracy(
        {"r": csv}, {"r": run}, top_n=1, spec=COST_SPECS["runtime"]
    )
    wide = P.aggregate_methods(long, top_n=1)
    tw, tl = P.to_throughput(wide, long)
    sp = tw.loc[tw["method"] == SHORTEST_PING, "cost"]
    assert sp.isna().all() and not np.isinf(tw["cost"]).any()
    out = P.plot_pareto(
        tw, tl, tmp_path / "thru.png",
        spec=COST_SPECS["runtime"], top_n=1, subtitle="throughput",
    )
    assert out.stat().st_size > 5_000
