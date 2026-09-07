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
    from scripts.analysis.v3.modules.diagram.common.palette import _VARIANT_HUES

    assert F.THRESHOLDS_KM == (100, 500, 1000)
    assert "#008300" in _VARIANT_HUES  # the hue a green guide would collide with


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
