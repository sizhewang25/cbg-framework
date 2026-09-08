"""`plot-proximity-inflation` — the two axes, the seam, and the missing half.

Five things this figure can get quietly wrong. Its x axis has two candidate
columns that answer different questions, and drawing one under the other's label
would turn a claim about geometry into a claim about routing. Its x axis is
logarithmic and its source columns may legally be zero, so a VP sitting on a
seed can take the whole transform out. Its y column is *carried* rather than
computed, so a `build-proximity` run that could not find its source CSV yields a
frame of NaN that would plot as an empty panel with no complaint. Its uncollected
dataset type has no points, so it falls out of anything derived from the data and
can vanish from the legend. And `share_proximate_sping_vp` is Shortest-Ping's own
top-1 accuracy by construction — the seam between this figure and the headline
table — which stops being true silently if either side changes its rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules import figure_proximity_inflation as P
from scripts.analysis.v3.modules import headline_table as H
from scripts.analysis.v3.modules.diagram.common.palette import _VARIANT_HUES

CLOSEST_COLUMN = P.X_METRICS[P.CLOSEST]["column"]
SPING_COLUMN = P.X_METRICS[P.SPING]["column"]


class _Run:
    """Enough of `RunPaths` for this module: an id and a proximity directory."""

    def __init__(self, run_id: str, root: Path) -> None:
        self.run_id = run_id
        self.setup = "anchors_to_probes"
        self._root = root

    def proximity_dir(self, root=None, *, grid: str, resolution: int) -> Path:
        path = self._root / self.run_id / "target-proximity" / f"{grid}-{resolution}"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _labels(n: int = 6, **overrides) -> pd.DataFrame:
    """Six targets whose two x metrics differ: the closest VP is always nearer
    than the one RTT picked, and the two coincide on the first two targets."""
    frame = pd.DataFrame(
        {
            "target_id": [f"t{i}" for i in range(n)],
            "tg_seed_id": [f"s{i % 2}" for i in range(n)],
            "tg_seed_margin_km": np.linspace(80.0, 240.0, n),
            CLOSEST_COLUMN: np.geomspace(5.0, 90.0, n),
            SPING_COLUMN: np.geomspace(20.0, 900.0, n),
            "tg_seed_nearest_vp_id": [f"v{i}" for i in range(n)],
            "sping_vp_id": ["v0", "v1"] + [f"w{i}" for i in range(2, n)],
            "has_proximate_vp": [True] * n,
            "has_discriminative_vp": [True] * n,
            "has_proximate_sping_vp": [True, True, False, False, False, False][:n],
            "has_discriminative_sping_vp": [True, False, False, False, False, False][:n],
            "n_measured_vps": np.arange(n) + 3,
        }
    )
    frame[P.Y_COLUMN] = np.linspace(1.2, 1.9, n)
    for column, values in overrides.items():
        frame[column] = values
    return frame


def _write(root: Path, run_id: str, labels: pd.DataFrame) -> _Run:
    run = _Run(run_id, root)
    labels.to_csv(run.proximity_dir(grid="h3", resolution=4) / "target_labels.csv",
                  index=False)
    return run


def _plan(runs: dict[str, _Run], weighted: dict | None = None) -> list[dict]:
    plan = H.row_plan(runs, weighted or {})
    by_run = {**runs, **{r.run_id: r for r in (weighted or {}).values()}}
    for entry in plan:
        entry["run"] = by_run.get(entry["run_id"]) if entry["run_id"] else None
    return plan


def _points(tmp_path: Path, **kwargs) -> tuple[pd.DataFrame, dict]:
    runs = {
        "as01-260728-260802": _write(tmp_path, "as01-260728-260802", _labels()),
        "as02-260728-260802": _write(tmp_path, "as02-260728-260802", _labels()),
    }
    return P.load_points(_plan(runs), grid="h3", resolution=4, **kwargs)


# ---- the frame --------------------------------------------------------------


def test_one_row_per_target_tagged_by_dataset_and_kind(tmp_path):
    points, _ = _points(tmp_path)
    assert len(points) == 12
    assert sorted(points["dataset"].unique()) == ["as01", "as02"]
    assert set(points["kind"]) == {H.MESH}
    assert {"y_inflation", "target_id", "run_id"} <= set(points.columns)


def test_both_x_metrics_load_regardless_of_which_one_is_drawn(tmp_path):
    """One points CSV describes the target set; the axis choice is a render-time
    decision, and the summary's two rank correlations need both columns."""
    points, _ = _points(tmp_path)
    for metric in P.X_METRIC_ORDER:
        assert P.x_column(metric) in points.columns
    assert not points[P.x_column(P.CLOSEST)].equals(points[P.x_column(P.SPING)])


def test_the_aggregate_plan_rows_contribute_no_points(tmp_path):
    """Pooling here is concatenation; a scatter has no denominator to average."""
    points, _ = _points(tmp_path)
    assert points.groupby("dataset").size().to_dict() == {"as01": 6, "as02": 6}


def test_the_flags_ride_along_so_the_seam_to_the_table_is_in_the_artifact(tmp_path):
    points, _ = _points(tmp_path)
    for column in P.CARRIED:
        assert column in points.columns


def test_whether_rtt_returned_the_geometrically_closest_vp_is_derived(tmp_path):
    """The categorical form of the gap between the two x axes."""
    points, _ = _points(tmp_path)
    assert points["sping_is_closest_vp"].tolist() == [
        True, True, False, False, False, False
    ] * 2


def test_an_unknown_x_metric_is_refused_naming_the_ones_that_exist():
    with pytest.raises(ValueError) as excinfo:
        P.x_metric_spec("nearest")
    assert P.CLOSEST in str(excinfo.value)
    assert P.SPING in str(excinfo.value)


# ---- the axes that can blow up ----------------------------------------------


def test_a_vp_sitting_on_its_seed_is_clamped_to_the_floor_not_dropped(tmp_path):
    """`log(0)` would take the transform out; dropping would lose the best target."""
    labels = _labels()
    labels.loc[0, CLOSEST_COLUMN] = 0.0
    runs = {"as01-260728-260802": _write(tmp_path, "as01-260728-260802", labels)}
    points, diagnostics = P.load_points(_plan(runs), grid="h3", resolution=4)
    assert len(points) == 6
    assert points[P.x_column(P.CLOSEST)].min() == P.MIN_X_KM
    assert diagnostics["n_clamped_to_floor"] == 1


def test_non_finite_rows_are_dropped_and_counted(tmp_path):
    labels = _labels()
    labels.loc[1, P.Y_COLUMN] = np.nan
    runs = {"as01-260728-260802": _write(tmp_path, "as01-260728-260802", labels)}
    points, diagnostics = P.load_points(_plan(runs), grid="h3", resolution=4)
    assert len(points) == 5
    assert diagnostics["n_dropped_non_finite"] == 1
    assert diagnostics["runs"]["as01-260728-260802"]["n_targets"] == 6


def test_a_row_missing_either_x_metric_is_dropped_from_both(tmp_path):
    """The two axes describe one target set, so a row drawn on one axis and
    absent from the other would give the summary two denominators."""
    labels = _labels()
    labels.loc[2, SPING_COLUMN] = np.nan
    runs = {"as01-260728-260802": _write(tmp_path, "as01-260728-260802", labels)}
    points, _ = P.load_points(_plan(runs), grid="h3", resolution=4)
    assert len(points) == 5
    assert points[P.x_column(P.CLOSEST)].notna().all()


def test_an_all_nan_inflation_column_is_refused_by_run_with_the_remedy(tmp_path):
    """`min_inflation` is carried from eval_source, so it is NaN when the source
    CSV was not found — a whole run of NaN plots as an empty panel in silence."""
    labels = _labels()
    labels[P.Y_COLUMN] = np.nan
    runs = {"as01-260728-260802": _write(tmp_path, "as01-260728-260802", labels)}
    with pytest.raises(ValueError) as excinfo:
        P.load_points(_plan(runs), grid="h3", resolution=4)
    assert "as01-260728-260802" in str(excinfo.value)
    assert "--source-csv" in str(excinfo.value)


def test_labels_written_by_an_older_build_are_refused_by_column_name(tmp_path):
    labels = _labels().drop(columns=[CLOSEST_COLUMN])
    runs = {"as01-260728-260802": _write(tmp_path, "as01-260728-260802", labels)}
    with pytest.raises(ValueError) as excinfo:
        P.load_points(_plan(runs), grid="h3", resolution=4)
    assert CLOSEST_COLUMN in str(excinfo.value)
    assert "build-proximity" in str(excinfo.value)


# ---- the summary ------------------------------------------------------------


def test_the_pooled_row_appears_only_when_a_kind_spans_more_than_one_dataset(tmp_path):
    points, _ = _points(tmp_path)
    pooled = P.summarize(points)
    assert list(pooled["dataset"]) == ["as01", "as02", H.AGGREGATE]

    one = points[points["dataset"] == "as01"]
    assert list(P.summarize(one)["dataset"]) == ["as01"]


def test_one_row_per_population_carrying_both_metrics_not_one_row_per_metric(tmp_path):
    """Splitting the rows would duplicate the shared y quantiles and invite the
    two copies to be read as two measurements."""
    summary = P.summarize(_points(tmp_path)[0])
    assert len(summary) == 3
    for metric in P.X_METRIC_ORDER:
        assert f"spearman_rho_{metric}" in summary.columns
        assert f"share_inside_margin_{metric}" in summary.columns
        assert f"{P.X_METRICS[metric]['prefix']}_p50" in summary.columns
    assert "min_inflation_p50" in summary.columns


def test_the_two_metrics_are_summarised_apart(tmp_path):
    """Same targets, same y — the difference is the axis alone, which is the
    whole reason both are reported."""
    summary = P.summarize(_points(tmp_path)[0])
    row = summary[summary["dataset"] == H.AGGREGATE].iloc[0]
    assert row["closest_vp_to_seed_km_p50"] < row["sping_vp_to_seed_km_p50"]
    assert row["share_inside_margin_closest"] >= row["share_inside_margin_sping"]


def test_share_proximate_sping_vp_is_shortest_pings_own_top1_accuracy(tmp_path):
    """The seam to the headline table: same computation on the same input, so a
    reader can tie this figure's near cluster to that table's first column."""
    summary = P.summarize(_points(tmp_path)[0])
    row = summary[summary["dataset"] == H.AGGREGATE].iloc[0]
    assert row["share_proximate_sping_vp"] == pytest.approx(2 / 6)
    assert row["n_targets"] == 12


def test_spearman_is_the_pearson_correlation_of_the_ranks():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert P._spearman(x, x * 3) == pytest.approx(1.0)
    assert P._spearman(x, -x) == pytest.approx(-1.0)
    ranked = pd.Series([10.0, 1.0, 100.0, 50.0, 3.0])
    assert P._spearman(x, ranked) == pytest.approx(
        np.corrcoef(x.rank(), ranked.rank())[0, 1]
    )


def test_spearman_is_nan_rather_than_a_divide_by_zero_without_variance():
    x = pd.Series([1.0, 2.0, 3.0])
    assert np.isnan(P._spearman(x, pd.Series([2.0, 2.0, 2.0])))
    assert np.isnan(P._spearman(pd.Series([1.0]), pd.Series([2.0])))


# ---- the encoding -----------------------------------------------------------


def test_an_uncollected_dataset_type_keeps_its_legend_entry(tmp_path):
    """It has no points, so it falls out of anything derived from the data."""
    assert P.legend_kinds([H.MESH], [H.WEIGHTED]) == list(H.KINDS)
    handles = P.kind_handles(list(H.KINDS), pending=[H.WEIGHTED])
    labels = [h.get_label() for h in handles]
    assert any("not collected" in label for label in labels)
    assert any(label == "mesh" for label in labels)


def test_the_pending_entry_is_an_outline_and_the_drawn_one_is_filled():
    handles = P.kind_handles(list(H.KINDS), pending=[H.WEIGHTED])
    mesh, weighted = handles[0], handles[1]
    assert mesh.get_markerfacecolor() == P.KIND_INK[H.MESH]
    assert weighted.get_markerfacecolor() == "none"
    assert weighted.get_markeredgecolor() == P.KIND_INK[H.WEIGHTED]


def test_no_variant_hue_is_spent_on_a_dataset_type():
    """Nothing here runs a method, so a variant hue would name a thing absent
    from the figure."""
    assert not set(P.KIND_INK.values()) & set(_VARIANT_HUES)


def test_the_two_dataset_type_inks_separate_on_white():
    def luminance(hex_colour: str) -> float:
        rgb = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    for ink in P.KIND_INK.values():
        assert 1.05 / (luminance(ink) + 0.05) >= 3.0
    mesh, weighted = (luminance(P.KIND_INK[k]) for k in (H.MESH, H.WEIGHTED))
    assert mesh != weighted


def test_the_weighted_series_is_drawn_over_the_mesh_one():
    """It is the mesh set filtered, so drawing it under would bury it."""
    assert P.KIND_Z[H.WEIGHTED] > P.KIND_Z[H.MESH]


def test_every_x_metric_carries_its_own_label_title_and_stem():
    """Four things move together when an axis is added; three are easy to miss."""
    stems = set()
    for metric in P.X_METRIC_ORDER:
        spec = P.x_metric_spec(metric)
        assert {"column", "points", "prefix", "axis_label", "title", "stem"} == set(spec)
        stems.add(spec["stem"])
    assert len(stems) == len(P.X_METRIC_ORDER)


# ---- the limits -------------------------------------------------------------


def test_the_y_axis_always_includes_the_soi_floor_and_leaves_it_off_the_spine(tmp_path):
    """The axis is read as "how far above the physical bound"; cropping it to the
    data rescales that reading per figure, and setting the limit *to* the floor
    hides the rule under the spine."""
    points, _ = _points(tmp_path)
    _, (bottom, top) = P._limits(points, P.CLOSEST)
    assert bottom < P.SOI_FLOOR
    assert top > points["y_inflation"].max()


def test_each_metric_gets_its_own_x_range(tmp_path):
    """`closest` spans two decades and `sping` nearly four; one shared range
    would squeeze the first into a stripe."""
    points, _ = _points(tmp_path)
    (closest_left, closest_right), _ = P._limits(points, P.CLOSEST)
    (sping_left, sping_right), _ = P._limits(points, P.SPING)
    assert closest_right < sping_right
    assert closest_left < sping_left


def test_the_x_limit_stays_positive_so_the_log_transform_holds(tmp_path):
    labels = _labels()
    labels.loc[0, CLOSEST_COLUMN] = 0.0
    runs = {"as01-260728-260802": _write(tmp_path, "as01-260728-260802", labels)}
    points, _ = P.load_points(_plan(runs), grid="h3", resolution=4)
    (left, right), _ = P._limits(points, P.CLOSEST)
    assert left > 0
    assert right > points[P.x_column(P.CLOSEST)].max()


def test_the_margin_band_spans_both_dataset_types_targets(tmp_path):
    """The margin is a property of the answer space, not of the campaign, so
    both series are measured against one threshold."""
    from scripts.analysis.v3.modules.diagram.common.draw import plt

    points, _ = _points(tmp_path)
    fig, ax = plt.subplots()
    P._margin_band(ax, points)
    xs = ax.patches[0].get_xy()[:, 0]
    lo, hi = points[P.MARGIN_COLUMN].quantile([0.25, 0.75])
    assert xs.min() == pytest.approx(lo)
    assert xs.max() == pytest.approx(hi)
    plt.close(fig)


def test_a_panel_without_the_margin_column_still_draws(tmp_path):
    from scripts.analysis.v3.modules.diagram.common.draw import plt

    points, _ = _points(tmp_path)
    fig, ax = plt.subplots()
    P._margin_band(ax, points.drop(columns=[P.MARGIN_COLUMN]))
    assert not ax.patches
    plt.close(fig)


# ---- rendering --------------------------------------------------------------


def test_both_layouts_render_to_distinct_non_empty_files(tmp_path):
    points, _ = _points(tmp_path)
    pooled = tmp_path / "pooled.png"
    compare = tmp_path / "compare.png"
    P.plot_scatter(points, pooled, kinds=[H.MESH], pending=[H.WEIGHTED], title="t")
    P.plot_compare(points, compare, kinds=[H.MESH], pending=[H.WEIGHTED], title="t")
    assert pooled.stat().st_size > 0
    assert compare.stat().st_size > 0
    assert pooled.read_bytes() != compare.read_bytes()


def test_the_two_metrics_render_different_pictures(tmp_path):
    """Same points, same layout — if these matched, the axis would not be being
    read."""
    points, _ = _points(tmp_path)
    closest = tmp_path / "closest.png"
    sping = tmp_path / "sping.png"
    for metric, path in ((P.CLOSEST, closest), (P.SPING, sping)):
        P.plot_scatter(
            points, path, kinds=[H.MESH], pending=[H.WEIGHTED], title="t",
            x_metric=metric,
        )
    assert closest.read_bytes() != sping.read_bytes()


def test_build_reports_the_uncollected_kind_rather_than_omitting_it(tmp_path):
    runs = {
        "as01-260728-260802": _write(tmp_path, "as01-260728-260802", _labels()),
        "as02-260728-260802": _write(tmp_path, "as02-260728-260802", _labels()),
    }
    points, summary, manifest = P.build(runs, {}, grid="h3", resolution=4)
    assert manifest["kinds_drawn"] == [H.MESH]
    assert manifest["kinds_pending"] == [H.WEIGHTED]
    assert "has_weight" in manifest["pending_note"]
    assert manifest["n_points"] == len(points) == 12
    assert len(summary) == 3
    assert set(manifest["axes"]["x"]) == set(P.X_METRIC_ORDER)
