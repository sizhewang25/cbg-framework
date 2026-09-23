"""`plot-error-cdf` — the row filter, the distance column, and the log floor.

The three ways this figure can lie: pooling fallback rows (so a variant
inherits the baseline's error where it failed), measuring through the class
seed (so the grid's quantization enters a distance that has its own ground
truth), and clamping the log floor over real data (so the left tail flattens
and p5 becomes the floor). One test each, plus the baseline case that a
hand-written `status == "SUCCESS"` filter silently drops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import figure_error_cdf as F
from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.diagram.common.labels import PUBLISHED_METHODS


def _scored(statuses, errors, *, seed_errors=None):
    """A `load_scored`-shaped frame."""
    n = len(statuses)
    return pd.DataFrame(
        {
            "target_id": [f"tg-{i}" for i in range(n)],
            "status": statuses,
            "tg_seed_id": [0] * n,
            "pred_seed_id": [0] * n,
            "tg_seed_rank": [0] * n,
            "error_to_target_km": errors,
            "error_to_tg_seed_km": seed_errors
            if seed_errors is not None
            else [e + 18.0 for e in errors],
            "error_to_pred_seed_km": [0.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# which rows are plotted
# ---------------------------------------------------------------------------


def test_fallback_rows_are_excluded():
    df = _scored(["SUCCESS", "FALLBACK", "SUCCESS"], [10.0, 999.0, 20.0])
    mask = io.solved_mask(df)
    assert mask.tolist() == [True, False, True]
    assert 999.0 not in df.loc[mask, F.ERROR_COLUMN].tolist()


def test_the_baseline_is_solved_despite_never_being_SUCCESS():
    """Shortest-Ping's rows are all BASELINE.

    A hand-written `status == "SUCCESS"` filter returns an all-false mask here,
    which would drop the baseline curve from the figure entirely.
    """
    df = _scored(["BASELINE"] * 3, [1.0, 2.0, 3.0])
    assert io.solved_mask(df).all()
    assert not df["status"].isin({"SUCCESS"}).any()


def test_an_empty_frame_yields_an_empty_mask_rather_than_raising():
    assert io.solved_mask(_scored([], [])).tolist() == []


# ---------------------------------------------------------------------------
# which distance
# ---------------------------------------------------------------------------


def test_the_figure_reads_the_raw_target_distance_not_the_seed_one():
    """Routing through the seed would add cell_offset_km to every answer."""
    assert F.ERROR_COLUMN == "error_to_target_km"
    df = _scored(["SUCCESS"], [10.0], seed_errors=[28.0])
    assert df.loc[0, F.ERROR_COLUMN] == 10.0


# ---------------------------------------------------------------------------
# the log floor
# ---------------------------------------------------------------------------


def test_the_floor_sits_below_the_observed_minimum_on_the_operator_runs():
    """0.135 km is the smallest error measured across as01/02/03.

    A 1 km floor — the v2 plotter's — clamps 8 to 20 rows per method there, up
    to 4.4% of a run, and pins their p5 to the clamp.
    """
    assert F.X_MIN_KM < 0.135


def test_the_clamp_applies_to_the_curve_only_not_the_percentiles():
    values = np.array([0.01, 0.02, 5.0, 500.0])
    xs, _ = F._cdf(values, 1.0)
    assert xs.min() == 1.0  # the drawn curve is clamped up to the floor

    table = F.percentile_table(
        {"m": values}, {"m": {"n_total": 4, "n_solved": 4, "n_fallback": 0}}
    )
    p5 = table.loc[0, "error_km_p5"]
    assert p5 < 1.0  # the table is not — it still sees below the floor
    assert p5 == pytest.approx(round(float(np.percentile(values, 5)), 3))


def test_the_cdf_rises_to_one_and_is_monotone():
    xs, ys = F._cdf(np.array([3.0, 1.0, 2.0]))
    assert xs.tolist() == [1.0, 2.0, 3.0]
    assert ys[-1] == pytest.approx(1.0)
    assert np.all(np.diff(ys) > 0)


# ---------------------------------------------------------------------------
# the percentile table
# ---------------------------------------------------------------------------


def test_percentiles_match_classifys_interpolation_exactly():
    """Both files call the column `error_km_p50`, so both must compute it the same.

    `classify.topn_summary` uses numpy's default (linear). Switching this to
    `method="nearest"` moved p50 by 0.7-1.3 km against `topn_accuracy.csv` —
    the file the paper's accuracy table reads.
    """
    values = np.array([1.0, 2.0, 3.0, 10.0, 500.0, 501.0, 900.0])
    table = F.percentile_table(
        {"m": values}, {"m": {"n_total": 7, "n_solved": 7, "n_fallback": 0}}
    )
    for p in F.PERCENTILES:
        assert table.loc[0, f"error_km_p{p}"] == pytest.approx(
            round(float(np.percentile(values, p)), 3)
        )


def test_the_table_carries_the_denominator_the_curve_was_drawn_over():
    counts = {"m": {"n_total": 412, "n_solved": 337, "n_fallback": 75}}
    table = F.percentile_table({"m": np.arange(337.0)}, counts)
    assert int(table.loc[0, "n_total"]) == 412
    assert int(table.loc[0, "n_plotted"]) == 337


def test_the_baseline_row_is_flagged():
    table = F.percentile_table(
        {SHORTEST_PING: np.array([1.0]), "vanilla_cbg": np.array([2.0])},
        {},
    )
    flagged = table.set_index("method")["is_baseline"]
    assert flagged[SHORTEST_PING]
    assert not flagged["vanilla_cbg"]


def test_an_all_empty_method_reports_nan_rather_than_crashing():
    table = F.percentile_table(
        {"m": np.array([])}, {"m": {"n_total": 5, "n_solved": 0, "n_fallback": 5}}
    )
    assert np.isnan(table.loc[0, "error_km_p50"])


# ---------------------------------------------------------------------------
# ordering and defaults
# ---------------------------------------------------------------------------


def test_published_order_leads_and_unknown_methods_follow():
    order = F.method_order(["zzz_cbg", "octant_cbg_hull", SHORTEST_PING])
    assert order == [SHORTEST_PING, "octant_cbg_hull", "zzz_cbg"]


def test_the_default_methods_are_the_shared_published_six():
    assert F.PUBLISHED_METHODS is PUBLISHED_METHODS
    assert len(PUBLISHED_METHODS) == 6
    assert PUBLISHED_METHODS[0] == SHORTEST_PING


def test_the_threshold_guides_avoid_the_variant_hues():
    """Green/orange/red guides would read as Octant-Hull/Vanilla/Spotter."""
    from scripts.analysis.v3.modules.diagram.common.palette import METHOD_HUES

    assert F.THRESHOLDS_KM == (100, 500, 1000)
    # Guides are drawn in grid ink, never in a method hue.
    assert "#e1e0d9" not in METHOD_HUES.values()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_the_figure_renders_to_a_non_empty_png(tmp_path):
    errors = {
        SHORTEST_PING: np.array([1.0, 50.0, 400.0]),
        "octant_cbg_hull": np.array([0.5, 20.0, 120.0]),
    }
    counts = {m: {"n_total": 3, "n_solved": 3, "n_fallback": 0} for m in errors}
    table = F.percentile_table(errors, counts)
    out = F.plot_error_cdf(
        errors, table, tmp_path / "cdf.png", title="t", subtitle="s"
    )
    assert out.exists() and out.stat().st_size > 5_000


def test_a_method_with_no_solved_rows_is_skipped_not_drawn_as_a_flat_line(tmp_path):
    errors = {SHORTEST_PING: np.array([1.0, 2.0]), "vanilla_cbg": np.array([])}
    counts = {
        SHORTEST_PING: {"n_total": 2, "n_solved": 2, "n_fallback": 0},
        "vanilla_cbg": {"n_total": 2, "n_solved": 0, "n_fallback": 2},
    }
    table = F.percentile_table(errors, counts)
    out = F.plot_error_cdf(
        errors, table, tmp_path / "cdf.png", title="t", subtitle="s"
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# cross-run layouts: pooled and compare
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import typer  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from scripts.analysis.v3.modules import headline_table as H  # noqa: E402
from scripts.analysis.v3.modules.diagram.common.draw import plt  # noqa: E402
from scripts.analysis.v3.modules.diagram.common.palette import (  # noqa: E402
    method_colors,
)


def _run(run_id, setup="probes_to_anchors"):
    """A `RunPaths` stand-in: `_load_solved` only asks for these three."""
    return SimpleNamespace(
        run_id=run_id,
        setup=setup,
        cls_accuracy_dir=lambda **_: Path(run_id),
    )


def _frame(ids, statuses, errors):
    df = _scored(statuses, errors)
    df["target_id"] = ids
    return df


@pytest.fixture
def scored(monkeypatch):
    """`{(run_id, method): frame}`, served in place of the classify parquets."""
    store: dict[tuple[str, str], pd.DataFrame] = {}
    monkeypatch.setattr(F, "load_scored", lambda d, m: store[(str(d), m)])
    monkeypatch.setattr(
        F,
        "available_methods",
        lambda d: sorted(m for r, m in store if r == str(d)),
    )
    return store


def _load(**kw):
    return dict(analysis_root=None, grid="h3", resolution=4, **kw)


def test_pooling_concatenates_the_solved_rows_and_sums_the_counts(scored):
    scored[("as01-1-2", "vanilla_cbg")] = _frame(
        ["a", "b", "c"], ["SUCCESS", "FALLBACK", "SUCCESS"], [10.0, 999.0, 20.0]
    )
    scored[("as02-1-2", "vanilla_cbg")] = _frame(["d", "e"], ["SUCCESS"] * 2, [5.0, 7.0])
    pooled = F.pool_runs([_run("as01-1-2"), _run("as02-1-2")], **_load())

    assert sorted(pooled["errors"]["vanilla_cbg"].tolist()) == [5.0, 7.0, 10.0, 20.0]
    assert 999.0 not in pooled["errors"]["vanilla_cbg"]  # fallback stays out
    assert pooled["counts"]["vanilla_cbg"] == {
        "n_total": 5, "n_solved": 4, "n_fallback": 1,
    }
    assert set(pooled["per_run_counts"]) == {"as01-1-2", "as02-1-2"}


def test_the_baseline_survives_pooling_despite_never_being_SUCCESS(scored):
    for rid, ids in (("as01-1-2", ["a"]), ("as02-1-2", ["b"])):
        scored[(rid, SHORTEST_PING)] = _frame(ids, ["BASELINE"], [3.0])
    pooled = F.pool_runs([_run("as01-1-2"), _run("as02-1-2")], **_load())
    assert pooled["counts"][SHORTEST_PING]["n_solved"] == 2


def test_a_method_missing_from_one_run_is_dropped_and_named(scored):
    scored[("as01-1-2", "vanilla_cbg")] = _frame(["a"], ["SUCCESS"], [1.0])
    scored[("as01-1-2", "spotter_cbg")] = _frame(["a"], ["SUCCESS"], [2.0])
    scored[("as02-1-2", "vanilla_cbg")] = _frame(["b"], ["SUCCESS"], [3.0])
    pooled = F.pool_runs([_run("as01-1-2"), _run("as02-1-2")], **_load())
    assert list(pooled["errors"]) == ["vanilla_cbg"]
    assert pooled["methods_absent_in_some_runs"] == ["spotter_cbg"]


def test_runs_sharing_a_target_refuse_to_pool(scored):
    scored[("as01-1-2", "vanilla_cbg")] = _frame(["a", "b"], ["SUCCESS"] * 2, [1.0, 2.0])
    scored[("as02-1-2", "vanilla_cbg")] = _frame(["b"], ["SUCCESS"], [3.0])
    with pytest.raises(typer.BadParameter, match="share targets"):
        F.pool_runs([_run("as01-1-2"), _run("as02-1-2")], **_load())


def _mesh_and_weighted(scored, *, weighted=True):
    """as01/as02 mesh, and as01's weighted twin over a subset of its targets."""
    for rid, ids in (("as01-1-2", ["a", "b"]), ("as02-1-2", ["c", "d"])):
        scored[(rid, SHORTEST_PING)] = _frame(ids, ["BASELINE"] * 2, [4.0, 40.0])
        scored[(rid, "vanilla_cbg")] = _frame(ids, ["SUCCESS"] * 2, [2.0, 20.0])
    mesh = {rid: _run(rid) for rid in ("as01-1-2", "as02-1-2")}
    wtd = {}
    if weighted:
        scored[("as01-w", SHORTEST_PING)] = _frame(["a"], ["BASELINE"], [4.0])
        scored[("as01-w", "vanilla_cbg")] = _frame(["a"], ["SUCCESS"], [2.0])
        wtd = {"as01": _run("as01-w")}
    return mesh, wtd


def test_a_weighted_twin_overlapping_its_mesh_targets_is_not_refused(scored, tmp_path):
    mesh, wtd = _mesh_and_weighted(scored)
    rendered, manifest = F.build_cross(
        mesh, wtd, analysis_root=tmp_path, resolution=4, layouts=(F.POOLED, F.COMPARE)
    )
    assert all(png.exists() for png, _ in rendered.values())
    table = rendered[F.POOLED][1]
    assert set(table["kind"]) == {H.MESH, H.WEIGHTED}
    assert not any(c["pending"] for c in manifest["curves"] if c["layout"] == F.POOLED)


def test_no_weighted_run_leaves_the_kind_pending_with_no_curve(scored, tmp_path):
    mesh, wtd = _mesh_and_weighted(scored, weighted=False)
    rendered, manifest = F.build_cross(
        mesh, wtd, analysis_root=tmp_path, resolution=4, layouts=(F.POOLED,)
    )
    assert set(rendered[F.POOLED][1]["kind"]) == {H.MESH}
    pending = [c for c in manifest["curves"] if c["pending"]]
    assert [c["kind"] for c in pending] == [H.WEIGHTED]


def test_a_weighted_run_without_its_mesh_twin_is_refused(scored, tmp_path):
    mesh, _ = _mesh_and_weighted(scored)
    with pytest.raises(typer.BadParameter):
        F.build_cross(
            mesh, {"as09": _run("as01-w")}, analysis_root=tmp_path, resolution=4
        )


def test_line_style_is_the_kind_and_the_baseline_keeps_its_own_hue():
    """Mesh solid, weighted dashed — the baseline included, in its variant hue."""
    entries = [
        {"kind": H.MESH, "errors": {SHORTEST_PING: np.array([1.0, 2.0]),
                                    "vanilla_cbg": np.array([3.0])}},
        {"kind": H.WEIGHTED, "errors": {SHORTEST_PING: np.array([1.5]),
                                        "vanilla_cbg": np.array([2.5])}},
    ]
    fig, ax = plt.subplots()
    F._draw_kind_curves(ax, entries, F.X_MIN_KM)
    lines = {line.get_gid(): line for line in ax.get_lines()}
    plt.close(fig)

    for method in (SHORTEST_PING, "vanilla_cbg"):
        assert lines[f"{H.MESH}:{method}"].get_linestyle() == "-"
        assert lines[f"{H.WEIGHTED}:{method}"].get_linestyle() == "--"
    hue = method_colors([SHORTEST_PING])[SHORTEST_PING]
    assert to_hex(lines[f"{H.MESH}:{SHORTEST_PING}"].get_color()) == to_hex(hue)
    # Drawn last, so it reads on top of every kind's variants.
    assert list(lines)[-2:] == [f"{H.MESH}:{SHORTEST_PING}", f"{H.WEIGHTED}:{SHORTEST_PING}"]


def test_the_pending_note_names_missing_and_partial_kinds():
    mesh = {"kind": H.MESH, "dataset": None, "datasets": ["as01", "as02"], "pending": False}
    partial = {"kind": H.WEIGHTED, "dataset": None, "datasets": ["as01"], "pending": False}
    missing = {"kind": H.WEIGHTED, "dataset": None, "datasets": [], "pending": True}
    assert "1 of 2 datasets" in F.pending_note([mesh, partial])
    assert "not collected" in F.pending_note([mesh, missing])
    assert F.pending_note([mesh]) == ""
