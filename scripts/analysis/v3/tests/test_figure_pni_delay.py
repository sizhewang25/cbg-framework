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
