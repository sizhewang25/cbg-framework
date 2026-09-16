"""The two-leg delay scatter: geometry, row filtering and the fit it reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import typer

from scripts.analysis.v3.modules import figure_pni_delay as F
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE


def _triple(vp, pni, tg, rtt, tid="t0"):
    return {
        "vp_id": "v0", "vp_lat": vp[0], "vp_lon": vp[1],
        "pni_id": "p0", "pni_lat": pni[0], "pni_lon": pni[1],
        "tg_id": tid, "tg_lat": tg[0], "tg_lon": tg[1],
        "min_rtt": rtt,
    }


def _write(tmp_path, rows, name="pairs.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_via_pni_delay_is_the_sum_of_the_two_legs(tmp_path):
    """The bent path is longer than the straight one, and priced at 2/3 c."""
    csv = _write(tmp_path, [_triple((48.85, 2.35), (52.52, 13.40), (51.51, -0.13), 30.0)])
    df, dropped = F.load_points(csv)

    assert dropped == 0
    row = df.iloc[0]
    assert row["d_via_pni_km"] == pytest.approx(row["d_vp_pni_km"] + row["d_pni_tg_km"])
    assert row["d_via_pni_km"] > row["d_vp_tg_km"]  # Paris→Berlin→London detours
    assert row["prop_rtt_via_pni_ms"] == pytest.approx(
        row["d_via_pni_km"] * THEORETICAL_SLOPE
    )
    assert row["residual_ms"] == pytest.approx(30.0 - row["prop_rtt_via_pni_ms"])


def test_non_positive_rtts_and_missing_coords_are_dropped_and_counted(tmp_path):
    """0 and -1 are measurement sentinels, not sub-light paths."""
    rows = [
        _triple((48.85, 2.35), (50.11, 8.68), (51.51, -0.13), 12.0, "keep"),
        _triple((48.85, 2.35), (50.11, 8.68), (51.51, -0.13), 0.0, "zero"),
        _triple((48.85, 2.35), (50.11, 8.68), (51.51, -0.13), -1.0, "neg"),
        _triple((48.85, 2.35), (50.11, np.nan), (51.51, -0.13), 12.0, "nan"),
    ]
    df, dropped = F.load_points(_write(tmp_path, rows))

    assert dropped == 3
    assert df["tg_id"].tolist() == ["keep"]


def test_a_row_below_the_floor_is_flagged_by_a_negative_residual(tmp_path):
    """An RTT under the bent path's minimum means the PNI is not on that path."""
    csv = _write(tmp_path, [_triple((48.85, 2.35), (40.71, -74.01), (51.51, -0.13), 5.0)])
    df, _ = F.load_points(csv)
    assert df.iloc[0]["residual_ms"] < 0

    stats = F.plot(df, tmp_path / "fig.png")
    assert stats["n_below_floor"] == 1.0


def test_prefixes_and_rtt_column_are_configurable(tmp_path):
    """A CSV naming its intermediate `ixp_*` needs a flag, not a rename."""
    rows = [{
        "vp_lat": 48.85, "vp_lon": 2.35,
        "ixp_lat": 50.11, "ixp_lon": 8.68,
        "tg_lat": 51.51, "tg_lon": -0.13,
        "rtt_min": 14.0,
    }]
    csv = _write(tmp_path, rows)

    with pytest.raises(typer.BadParameter, match="missing"):
        F.load_points(csv)

    df, _ = F.load_points(csv, pni_prefix="ixp", rtt_col="rtt_min")
    assert df.iloc[0]["min_rtt_ms"] == 14.0


def test_fit_recovers_a_planted_slope_and_beats_the_no_pni_control(tmp_path):
    """The control exists to make the PNI's contribution a comparison.

    Targets are placed so the straight VP→TG distance is a poor proxy for the
    path actually taken; the via-PNI fit must then explain more variance.
    """
    rng = np.random.default_rng(7)
    rows = []
    for i in range(150):
        vp = (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18)))
        pni = (vp[0] + float(rng.normal(0, 2)), vp[1] + float(rng.normal(0, 2)))
        tg = (pni[0] + float(rng.normal(0, 3)), pni[1] + float(rng.normal(0, 3)))
        rows.append(_triple(vp, pni, tg, 1.0, f"t{i}"))
    df = pd.DataFrame(rows)
    csv = _write(tmp_path, rows)
    df, _ = F.load_points(csv)
    df["min_rtt_ms"] = 1.4 * df["prop_rtt_via_pni_ms"] + 3.0
    df["residual_ms"] = df["min_rtt_ms"] - df["prop_rtt_via_pni_ms"]

    # The slope was planted in delay space, so it is dimensionless there.
    stats = F.plot(df, tmp_path / "fig.png", x_unit="ms")

    assert stats["via_pni_slope"] == pytest.approx(1.4, abs=1e-6)
    assert stats["via_pni_intercept_ms"] == pytest.approx(3.0, abs=1e-6)
    assert stats["via_pni_r2"] == pytest.approx(1.0, abs=1e-9)
    assert stats["direct_r2"] < stats["via_pni_r2"]
    assert (tmp_path / "fig.png").exists()

    # In km the same fit is the same line in other units: the slope carries
    # ms/km, the intercept and r2 are untouched, and 2/slope is a speed.
    km = F.plot(df, tmp_path / "fig_km.png", x_unit="km")
    assert km["via_pni_slope"] == pytest.approx(1.4 * THEORETICAL_SLOPE, rel=1e-9)
    assert km["via_pni_intercept_ms"] == pytest.approx(3.0, abs=1e-6)
    assert km["via_pni_r2"] == pytest.approx(stats["via_pni_r2"], rel=1e-12)
    assert km["implied_km_per_ms"] == pytest.approx(2 / (1.4 * THEORETICAL_SLOPE), rel=1e-9)
    # The below-floor set is a property of the points, not of the units.
    assert km["n_below_floor"] == stats["n_below_floor"]


def test_degenerate_input_yields_nan_fit_rather_than_raising(tmp_path):
    """One row cannot support a line; the figure is still written."""
    csv = _write(tmp_path, [_triple((48.85, 2.35), (50.11, 8.68), (51.51, -0.13), 12.0)])
    df, _ = F.load_points(csv)

    stats = F.plot(df, tmp_path / "fig.png")

    assert np.isnan(stats["via_pni_slope"])
    assert (tmp_path / "fig.png").exists()


def test_command_is_registered_and_writes_both_outputs(tmp_path):
    """End-to-end through the CLI, which is how this script is meant to be run."""
    from typer.testing import CliRunner

    from scripts.analysis.v3.cli import app

    rng = np.random.default_rng(1)
    rows = [
        _triple(
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            (50.11, 8.68),
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            float(rng.uniform(5, 40)),
            f"t{i}",
        )
        for i in range(20)
    ]
    csv = _write(tmp_path, rows)

    result = CliRunner().invoke(app, ["plot-pni-delay", "--csv", str(csv)])

    assert result.exit_code == 0, result.output
    # The stem records the x-axis choice, so a via-pni and a direct run of the
    # same input cannot overwrite each other.
    assert (tmp_path / "pairs_pni_delay.via-pni.km.png").exists()
    assert (tmp_path / "pairs_pni_delay.via-pni.km_points.csv").exists()
    assert "via_pni_pearson_r" in result.output


def test_all_rows_filtered_out_is_a_clear_error(tmp_path):
    from typer.testing import CliRunner

    from scripts.analysis.v3.cli import app

    csv = _write(tmp_path, [_triple((48.85, 2.35), (50.11, 8.68), (51.51, -0.13), 0.0)])

    result = CliRunner().invoke(app, ["plot-pni-delay", "--csv", str(csv)])

    assert result.exit_code != 0
    assert "no usable rows" in result.output


def test_where_restricts_to_a_boolean_column_without_counting_it_as_dropped(tmp_path):
    """A subset is a deliberate choice, not unusable data, so it must not land in
    the dropped count the figure annotates itself with."""
    csv = tmp_path / "edges.csv"
    pd.DataFrame(
        {
            "vp_lat": [40.0, 41.0, 42.0], "vp_lon": [-74.0, -75.0, -76.0],
            "pni_lat": [40.5, 41.5, 42.5], "pni_lon": [-74.5, -75.5, -76.5],
            "tg_lat": [41.0, 42.0, 43.0], "tg_lon": [-75.0, -76.0, -77.0],
            "min_rtt": [10.0, 20.0, 30.0],
            "keep": [True, False, True],
        }
    ).to_csv(csv, index=False)
    df, dropped = F.load_points(csv, where="keep")
    assert len(df) == 2 and dropped == 0
    assert df.min_rtt.tolist() == [10.0, 30.0]


def test_a_where_column_read_back_as_strings_is_still_a_mask(tmp_path):
    """One NaN makes the round trip `object`, and a naive df[col] mask would then
    select the string "False" as truthy."""
    csv = tmp_path / "edges.csv"
    pd.DataFrame(
        {
            "vp_lat": [40.0, 41.0], "vp_lon": [-74.0, -75.0],
            "pni_lat": [40.5, 41.5], "pni_lon": [-74.5, -75.5],
            "tg_lat": [41.0, 42.0], "tg_lon": [-75.0, -76.0],
            "min_rtt": [10.0, 20.0],
            "keep": ["True", "False"],
        }
    ).to_csv(csv, index=False)
    df, _ = F.load_points(csv, where="keep")
    assert df.min_rtt.tolist() == [10.0]


def test_an_unknown_where_column_is_named(tmp_path):
    csv = tmp_path / "edges.csv"
    pd.DataFrame(
        {
            "vp_lat": [40.0], "vp_lon": [-74.0], "pni_lat": [40.5], "pni_lon": [-74.5],
            "tg_lat": [41.0], "tg_lon": [-75.0], "min_rtt": [10.0],
        }
    ).to_csv(csv, index=False)
    with pytest.raises(typer.BadParameter, match="nope"):
        F.load_points(csv, where="nope")


def test_the_x_axis_can_be_the_direct_geodesic_instead(tmp_path):
    """`--x-axis direct` draws the no-PNI control as the figure itself. Both fits
    keep their own key names, so a stat never changes meaning with the axis."""
    from typer.testing import CliRunner

    from scripts.analysis.v3.cli import app

    rng = np.random.default_rng(2)
    rows = [
        _triple(
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            (50.11, 8.68),
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            float(rng.uniform(5, 40)),
            f"t{i}",
        )
        for i in range(20)
    ]
    csv = _write(tmp_path, rows)
    df, _ = F.load_points(csv)

    via = F.plot(df, tmp_path / "via.png", x_axis="via-pni")
    direct = F.plot(df, tmp_path / "direct.png", x_axis="direct")

    # Same two fits either way; only which is drawn moves.
    for key in ("via_pni_r2", "direct_r2", "via_pni_slope", "direct_slope"):
        assert via[key] == pytest.approx(direct[key])
    assert via["x_axis"] == "via-pni" and direct["x_axis"] == "direct"
    # The below-floor count follows the axis on screen, so it differs.
    assert direct["n_below_floor"] == float((df.min_rtt_ms < df.prop_rtt_direct_ms).sum())
    assert via["n_below_floor"] == float((df.min_rtt_ms < df.prop_rtt_via_pni_ms).sum())

    result = CliRunner().invoke(app, ["plot-pni-delay", "--csv", str(csv), "--x-axis", "direct"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "pairs_pni_delay.direct.km.png").exists()


def test_an_unknown_x_axis_is_refused_by_name(tmp_path):
    df = pd.DataFrame(
        {"prop_rtt_via_pni_ms": [1.0], "prop_rtt_direct_ms": [1.0], "min_rtt_ms": [2.0],
         "residual_ms": [1.0]}
    )
    with pytest.raises(typer.BadParameter, match="sideways"):
        F.plot(df, tmp_path / "x.png", x_axis="sideways")


def test_a_below_floor_row_can_clear_the_geodesic_and_is_listed_with_its_ids(tmp_path):
    """The band between the two floors, which is where the outliers live.

    Paris→London is 344 km, so 8 ms is a legal RTT on the geodesic — there is no
    `min_rtt - 0.01 d(VP,TG)` violation anywhere in this row. Routed through
    Frankfurt the path is 1,119 km, whose floor is 11.2 ms, so the same row is
    below the bent floor. That is the assignment being refuted, not the
    measurement, and `detour_ratio > air_inflation` is the arithmetic of it.
    """
    csv = _write(tmp_path, [_triple((48.85, 2.35), (50.11, 8.68), (51.51, -0.13), 8.0)])
    df, _ = F.load_points(csv)

    assert df.iloc[0]["residual_direct_ms"] > 0  # legal on the straight line
    assert df.iloc[0]["residual_ms"] < 0  # illegal through Frankfurt
    assert df.iloc[0]["detour_ratio"] > df.iloc[0]["air_inflation"]

    below = F.below_floor(df)
    assert len(below) == 1
    row = below.iloc[0]
    assert (row["vp_id"], row["pni_id"], row["tg_id"]) == ("v0", "p0", "t0")
    assert row["below_by_ms"] == pytest.approx(-df.iloc[0]["residual_ms"])
    # The file opens on the identity of the offending triple, each id followed
    # by its place, measured pair first and the assigned site last.
    assert list(below.columns)[:6] == [
        "vp_id", "vp_loc", "tg_id", "tg_loc", "pni_id", "pni_loc",
    ]

    # The same row is *not* a direct-axis violation, since that floor is lower.
    assert F.below_floor(df, x_axis="direct").empty


def test_the_below_floor_listing_matches_the_count_the_figure_annotates(tmp_path):
    """The listing and the figure's `n_below_floor` read the same mask."""
    rng = np.random.default_rng(11)
    rows = [
        _triple(
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            (50.11, 8.68),
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            float(rng.uniform(1, 30)),
            f"t{i}",
        )
        for i in range(60)
    ]
    df, _ = F.load_points(_write(tmp_path, rows))

    for x_axis in ("via-pni", "direct"):
        stats = F.plot(df, tmp_path / f"{x_axis}.png", x_axis=x_axis)
        below = F.below_floor(df, x_axis=x_axis)
        assert stats["n_below_floor"] == float(len(below))
        # Worst first, and every listed row really is under that floor.
        assert (below["below_by_ms"] > 0).all()
        assert below["below_by_ms"].is_monotonic_decreasing
    # The floors nest: via >= direct, so every direct violation is a via one.
    via = set(F.below_floor(df)["tg_id"])
    assert set(F.below_floor(df, x_axis="direct")["tg_id"]) <= via


def test_a_csv_without_ids_is_listed_by_coordinates_instead(tmp_path):
    """`--pni-prefix ixp` inputs may carry no id column; the row must still be
    identifiable, so the listing falls back to the coordinates."""
    rows = [{
        "vp_lat": 48.85, "vp_lon": 2.35,
        "pni_lat": 50.11, "pni_lon": 8.68,
        "tg_lat": 51.51, "tg_lon": -0.13,
        "min_rtt": 8.0,
    }]
    df, _ = F.load_points(_write(tmp_path, rows))
    below = F.below_floor(df)

    assert list(below.columns)[:9] == [
        "vp_lat", "vp_lon", "vp_loc",
        "tg_lat", "tg_lon", "tg_loc",
        "pni_lat", "pni_lon", "pni_loc",
    ]
    assert "48.850" in F.below_floor_table(below, width=400)


def test_the_command_writes_and_prints_the_below_floor_rows(tmp_path):
    from typer.testing import CliRunner

    from scripts.analysis.v3.cli import app

    rows = [
        _triple((48.85, 2.35), (50.11, 8.68), (51.51, -0.13), 8.0, "bent"),
        _triple((48.85, 2.35), (48.86, 2.36), (48.87, 2.37), 40.0, "fine"),
    ]
    csv = _write(tmp_path, rows)

    result = CliRunner().invoke(app, ["plot-pni-delay", "--csv", str(csv)])

    assert result.exit_code == 0, result.output
    out_below = tmp_path / "pairs_pni_delay.via-pni.km_below_floor.csv"
    assert out_below.exists()
    listed = pd.read_csv(out_below)
    assert listed["tg_id"].tolist() == ["bent"]
    assert "1 of 2 rows below the via-pni floor" in result.output
    assert "bent" in result.output


def test_no_below_floor_rows_writes_no_file_and_prints_nothing(tmp_path):
    """The absence of violations is the good case; it must not leave an empty
    file behind for a downstream reader to mistake for a run that found none."""
    from typer.testing import CliRunner

    from scripts.analysis.v3.cli import app

    csv = _write(tmp_path, [
        _triple((48.85, 2.35), (48.86, 2.36), (48.87, 2.37), 40.0, "a"),
        _triple((48.85, 2.35), (48.90, 2.40), (48.95, 2.45), 50.0, "b"),
    ])

    result = CliRunner().invoke(app, ["plot-pni-delay", "--csv", str(csv)])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "pairs_pni_delay.via-pni.km_below_floor.csv").exists()
    assert "below the via-pni floor" not in result.output


def test_an_overlong_title_is_shrunk_to_fit_rather_than_clipped():
    """matplotlib clips a too-wide title at both ends, which reads as a
    deliberate label that happens to be missing its first and last words.

    `--title` is free text and `create_analysis_artifacts.sh` passes a
    per-strategy one, so the figure has to survive a caption longer than the
    default. A title that already fits must be left exactly alone.
    """
    from scripts.analysis.v3.modules.diagram.common.draw import plt

    fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=200)
    try:
        ax.set_title("min-RTT vs two-leg delay -- strategy vp_nearest")
        default = ax.title.get_fontsize()
        F._fit_title(fig, ax)
        assert ax.title.get_fontsize() == default

        ax.set_title("min-RTT vs two-leg propagation delay -- " + "long " * 30)
        F._fit_title(fig, ax)
        assert ax.title.get_fontsize() < default

        # And it fits now, which is the property the caller actually wants.
        fig.canvas.draw()
        assert ax.title.get_window_extent().width <= ax.get_window_extent().width + 1
    finally:
        plt.close(fig)


def _drawn_xlim(monkeypatch, df, out_png, **kwargs):
    """The x limits `plot()` actually drew. It closes its own figure, so the
    limits are read off the way out."""
    captured = {}
    real_close = F.plt.close

    def spy(fig):
        captured["xlim"] = fig.axes[0].get_xlim()
        real_close(fig)

    monkeypatch.setattr(F.plt, "close", spy)
    F.plot(df, out_png, **kwargs)
    return captured["xlim"]


def test_the_x_axis_ends_at_the_data_not_at_the_floor_lines_reach(tmp_path, monkeypatch):
    """The reference lines span the x DATA, not the y range.

    Priced at 2/3 c the floor reaches an observed 80 ms only at 8,000 km, so
    drawing the lines that far autoscaled x to several times the widest path in
    the input: half the panel came out empty with the cloud squeezed into the
    rest.
    """
    rows = [
        _triple((40.0, -74.0), (41.0, -75.0), (42.0, -76.0), 80.0, "slow"),
        _triple((40.0, -74.0), (40.1, -74.1), (40.2, -74.2), 5.0, "near"),
    ]
    df, _ = F.load_points(_write(tmp_path, rows))
    widest = float(df["d_via_pni_km"].max())
    floors_reach = 80.0 / THEORETICAL_SLOPE
    assert floors_reach > 3 * widest  # the fixture reproduces the old blow-up

    left, right = _drawn_xlim(monkeypatch, df, tmp_path / "fig.png")

    assert left == 0.0
    assert right == pytest.approx(widest * 1.02)


def test_x_max_sets_the_right_edge_even_when_the_data_runs_past_it(tmp_path, monkeypatch):
    """Otherwise the three per-strategy panels get three different scales."""
    rows = [
        _triple((40.0, -74.0), (41.0, -75.0), (42.0, -76.0), 60.0, "far"),
        _triple((40.0, -74.0), (40.1, -74.1), (40.2, -74.2), 5.0, "near"),
    ]
    df, _ = F.load_points(_write(tmp_path, rows))

    _, right = _drawn_xlim(monkeypatch, df, tmp_path / "fig.png", x_max=100.0)

    assert right == pytest.approx(100.0)


def test_x_max_clips_the_view_only(tmp_path):
    """Clipping x compresses its range, which attenuates r on its own. The fit
    has to stay over every row or two panels cannot be compared."""
    rng = np.random.default_rng(3)
    rows = [
        _triple(
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            (50.11, 8.68),
            (float(rng.uniform(40, 55)), float(rng.uniform(-8, 18))),
            float(rng.uniform(5, 40)),
            f"t{i}",
        )
        for i in range(40)
    ]
    df, _ = F.load_points(_write(tmp_path, rows))
    cut = float(df["d_via_pni_km"].quantile(0.6))

    full = F.plot(df, tmp_path / "full.png")
    clipped = F.plot(df, tmp_path / "clipped.png", x_max=cut)

    for key in ("n", "via_pni_r2", "via_pni_slope", "direct_r2", "n_below_floor"):
        assert clipped[key] == pytest.approx(full[key])
    assert full["n_outside_view"] == 0.0
    assert clipped["n_outside_view"] == float((df["d_via_pni_km"] > cut).sum())
    assert clipped["n_outside_view"] > 0
    assert clipped["share_outside_view"] == pytest.approx(
        clipped["n_outside_view"] / len(df)
    )


def test_a_point_past_both_cuts_is_counted_once(tmp_path):
    """`figure_distance_rtt`'s convention: y is counted only inside the x
    window, so the off-view share cannot exceed 1."""
    rows = [
        _triple((40.0, -74.0), (41.0, -75.0), (42.0, -76.0), 90.0, "far_and_slow"),
        _triple((40.0, -74.0), (40.1, -74.1), (40.2, -74.2), 5.0, "near"),
    ]
    df, _ = F.load_points(_write(tmp_path, rows))
    cut = float(df["d_via_pni_km"].min()) + 1.0

    stats = F.plot(df, tmp_path / "fig.png", x_max=cut, y_max=50.0)
    assert stats["n_outside_view"] == 1.0


def test_the_command_takes_the_cuts_and_records_them(tmp_path):
    from typer.testing import CliRunner

    from scripts.analysis.v3.cli import app

    rows = [
        _triple((40.0, -74.0), (41.0, -75.0), (42.0, -76.0), 60.0, "far"),
        _triple((40.0, -74.0), (40.1, -74.1), (40.2, -74.2), 5.0, "near"),
    ]
    csv = _write(tmp_path, rows)

    result = CliRunner().invoke(
        app, ["plot-pni-delay", "--csv", str(csv), "--x-max", "100", "--y-max", "50"]
    )

    assert result.exit_code == 0, result.output
    assert "n_outside_view" in result.output


def test_a_declared_place_is_used_and_a_missing_one_is_looked_up(tmp_path):
    """`pni_edges.csv` names the site's city off the operator's list, and that
    label is authoritative. Nothing upstream names a VP's or a target's, so
    those come from the coordinate — and the two must not be confused."""
    rows = [{
        "vp_id": "v0", "vp_lat": 33.7556, "vp_lon": -84.3915,
        "pni_id": "p0", "pni_lat": 40.0, "pni_lon": -75.0,
        "pni_country": "US", "pni_region": "Georgia", "pni_city": "Atlanta",
        "tg_id": "t0", "tg_lat": 25.7932, "tg_lon": -80.2906,
        "min_rtt": 8.0,
    }]
    df, _ = F.load_points(_write(tmp_path, rows))

    assert F.loc_provenance(df, "pni") == "declared"
    assert F.loc_provenance(df, "vp") == "derived"
    assert F.loc_provenance(df, "tg") == "derived"

    placed = F.add_loc_columns(df, "vp", "tg", "pni")
    # Declared: carried through verbatim, not re-derived. The site's own
    # coordinate is in Philadelphia here, so a lookup would disagree.
    assert placed.loc[0, "pni_loc"] == "US-Georgia-Atlanta"
    # Derived: country and region from the nearest cities1000 entry.
    assert placed.loc[0, "vp_loc"].startswith("US-Georgia-")
    assert placed.loc[0, "tg_loc"].startswith("US-Florida-")


def test_a_partial_declared_place_does_not_emit_empty_separators(tmp_path):
    """A site list may carry a country and no city."""
    rows = [{
        "vp_id": "v0", "vp_lat": 48.85, "vp_lon": 2.35,
        "pni_id": "p0", "pni_lat": 50.11, "pni_lon": 8.68, "pni_country": "DE",
        "tg_id": "t0", "tg_lat": 51.51, "tg_lon": -0.13,
        "min_rtt": 8.0,
    }]
    df, _ = F.load_points(_write(tmp_path, rows))

    assert F.add_loc_columns(df, "pni").loc[0, "pni_loc"] == "DE"


def test_the_printed_table_wraps_instead_of_running_off_the_terminal(tmp_path):
    """Three ids, three places and nine numbers is wider than a terminal, and a
    row the emulator soft-wraps is unreadable in a way a stacked block is
    not."""
    rows = [_triple((48.85, 2.35), (40.71, -74.01), (51.51, -0.13), 8.0, "t0")]
    df, _ = F.load_points(_write(tmp_path, rows))
    below = F.below_floor(df)

    narrow = F.below_floor_table(below, width=100)
    assert max(len(line) for line in narrow.splitlines()) <= 100
    # Every column survives the wrap; it is stacked, not truncated.
    for col in ("vp_loc", "tg_loc", "pni_loc", "below_by_ms", "air_inflation"):
        assert col in narrow


def test_an_empty_below_floor_set_still_gets_its_columns(tmp_path):
    """The regression this file missed the first time: zero listed rows is the
    NORMAL case for argmin, and a row-wise join over an empty frame returns a
    DataFrame that cannot be assigned to a column."""
    rows = [{
        "vp_id": "v0", "vp_lat": 48.85, "vp_lon": 2.35,
        "pni_id": "p0", "pni_lat": 48.86, "pni_lon": 2.36,
        "pni_country": "FR", "pni_region": "Ile-de-France", "pni_city": "Paris",
        "tg_id": "t0", "tg_lat": 48.87, "tg_lon": 2.37,
        "min_rtt": 40.0,
    }]
    df, _ = F.load_points(_write(tmp_path, rows))
    assert df.loc[0, "residual_ms"] > 0  # nothing is below the floor

    below = F.below_floor(df)

    assert below.empty
    assert {"vp_loc", "tg_loc", "pni_loc"} <= set(below.columns)
    assert F.below_floor_table(below) == "no rows below the floor"
