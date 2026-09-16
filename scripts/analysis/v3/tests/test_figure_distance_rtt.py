"""§8.1 figure 1a: the view it ships with, and what clipping may not touch."""

from __future__ import annotations

import pandas as pd
import pytest
import typer

from scripts.analysis.v3.modules import figure_distance_rtt as F

#: The frame this figure shares with `plot-pni-delay`. x there is this same
#: VP-to-target line bent through an interconnect, so the two are read side by
#: side and have to be cut AND ruled alike -- a shared cut on two different
#: gridline steps still puts comparable pictures on two rulers. The
#: `plot-pni-delay` half is pinned per run in `configs/*.yaml`, since that
#: command has no single dataset to default for.
SHARED_FRAME = {"x_max": 5000.0, "x_tick": 1000.0, "y_max": 100.0, "y_tick": 20.0}


def test_the_default_view_is_the_frame_shared_with_plot_pni_delay():
    from scripts.analysis.v3.cli import app

    params = {
        p.name: p
        for p in typer.main.get_command(app).commands["plot-distance-rtt"].params
    }
    assert {k: params[k].default for k in SHARED_FRAME} == SHARED_FRAME


def _pairs(distances, rtts):
    return pd.DataFrame({"distance_km": distances, "rtt_ms": rtts})


def test_the_fit_is_over_all_pairs_and_the_cuts_only_count_what_they_hide():
    """Restricting x compresses its range, which attenuates Pearson r on its
    own -- so a clipped-subset fit would report range restriction as worse
    agreement. The cut may move the view and the off-view count, nothing else.
    """
    pairs = _pairs([100, 1000, 2000, 9000], [5.0, 12.0, 25.0, 95.0])

    full = F.series_stats(pairs, x_max=None, y_max=None, n_dropped=0)
    clipped = F.series_stats(pairs, x_max=5000.0, y_max=100.0, n_dropped=0)

    for key in ("n", "pearson_r", "r2", "ols_slope_ms_per_km", "residual_rmse_ms"):
        assert clipped[key] == pytest.approx(full[key])
    assert full["n_outside_view"] == 0
    assert clipped["n_outside_view"] == 1
    assert clipped["share_outside_view"] == pytest.approx(0.25)


def test_a_pair_past_both_cuts_is_counted_once():
    """y is counted only inside the x window, so the off-view share cannot
    exceed 1 and a far, slow pair is one hidden point rather than two."""
    pairs = _pairs([9000, 100], [200.0, 5.0])

    stats = F.series_stats(pairs, x_max=5000.0, y_max=100.0, n_dropped=0)

    assert stats["n_outside_view"] == 1


def test_the_tick_steps_are_applied_to_the_drawn_axes(tmp_path, monkeypatch):
    """A fixed cut with an auto step is the failure this guards: matplotlib
    picks the step from the range it is given, so two panels cut alike could
    still be ruled differently."""
    mesh = _pairs([100, 1000, 2000, 4500], [5.0, 12.0, 25.0, 60.0])
    stats = {"mesh": F.series_stats(mesh, x_max=5000.0, y_max=100.0, n_dropped=0)}
    out_png = tmp_path / "fig.png"

    # `build` closes its own figure, so the axes are read on the way out.
    drawn = {}
    real_close = F.plt.close

    def spy(fig):
        ax = fig.axes[0]
        drawn["x"] = list(ax.get_xticks())
        drawn["y"] = list(ax.get_yticks())
        drawn["xlim"], drawn["ylim"] = ax.get_xlim(), ax.get_ylim()
        real_close(fig)

    monkeypatch.setattr(F.plt, "close", spy)
    F.build(
        mesh, None,
        x_max=SHARED_FRAME["x_max"], y_max=SHARED_FRAME["y_max"],
        x_tick=SHARED_FRAME["x_tick"], y_tick=SHARED_FRAME["y_tick"],
        mesh_label="mesh", tw_label="tw", title=None,
        mesh_point_size=11.0, tw_point_size=2.2,
        mesh_alpha=0.22, tw_alpha=0.45,
        out_png=out_png, stats=stats,
    )

    assert out_png.exists()
    assert drawn["xlim"] == (0.0, 5000.0) and drawn["ylim"] == (0.0, 100.0)
    # Ticks every 1,000 km and 20 ms across the pinned range. Matplotlib keeps
    # one step beyond each end in the list, so filter to the visible window.
    assert [t for t in drawn["x"] if 0 <= t <= 5000] == [
        0, 1000, 2000, 3000, 4000, 5000
    ]
    assert [t for t in drawn["y"] if 0 <= t <= 100] == [0, 20, 40, 60, 80, 100]
