"""The per-dataset RTT CDF, and the one thing a truncated linear axis hides.

The load-bearing class is `TestTheXCutIsReported`. A linear axis cut at 200 ms
renders a dataset whose tail reaches 2 s exactly like one whose largest RTT is
92 ms: the curve leaves the right edge below 100% and nothing on the panel says
why. Every other v5 figure that truncates does it on a log axis, where the
clipped mass is at least visibly compressed; here it is simply gone. So the
share within the cut is computed once, reaches the CSV, the manifest and an
in-panel warning, and is pinned here.

`TestNoPoolingGuard` pins an absence. The sibling cross figures call
`cross.guard_disjoint_tgs`, and a later reader is entitled to assume this one
forgot. It did not: nothing here shares a denominator, so a TG in two datasets
belongs on both curves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_rtt_cdf as R


@dataclass(frozen=True)
class FakeRun:
    """The slice of `RunPaths` this figure touches: a run id and nothing else.

    `load` reaches the measurements through `edges.load_min_rtt`, which the
    tests bypass with `source_csv`, so no benchmark tree has to exist.
    """

    run_id: str


def _csv(path: Path, rtts: list[float], *, tgs: int = 2) -> Path:
    """A canonical CSV spreading `rtts` over `tgs` targets and one VP each."""
    rows = []
    for i, rtt in enumerate(rtts):
        t = i % tgs
        rows.append(
            {
                "vp_id": f"vp-{i}",
                "vp_lat": 40.0,
                "vp_lon": -80.0,
                "target_id": f"tg-{t}",
                "target_lat": 41.0,
                "target_lon": -81.0,
                "rtt_ms": rtt,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _long(per_run: dict[str, list[float]]) -> pd.DataFrame:
    """The frame `load` returns, built directly."""
    rows = []
    for run_id, rtts in per_run.items():
        for i, rtt in enumerate(rtts):
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": run_id.split("-")[0],
                    "tg_id": f"tg-{i % 2}",
                    "vp_id": f"vp-{i}",
                    "rtt_ms": rtt,
                }
            )
    return pd.DataFrame(rows)


class TestLoad:
    def test_reads_each_runs_csv_and_tags_the_dataset(self, tmp_path):
        a = _csv(tmp_path / "a.csv", [1.0, 2.0, 3.0])
        b = _csv(tmp_path / "b.csv", [10.0, 20.0])
        long, meta = R.load(
            [FakeRun("as01-260728-260802-mesh"), FakeRun("as02-260728-260802-mesh")],
            source_csv={"as01-260728-260802-mesh": a, "as02-260728-260802-mesh": b},
        )
        assert len(long) == 5
        assert sorted(long["dataset"].unique()) == ["as01", "as02"]
        assert meta["datasets"] == ["as01", "as02"]

    def test_no_runs_raises(self):
        with pytest.raises(ValueError, match="at least one run"):
            R.load([])

    def test_non_positive_rtts_are_dropped_as_the_benchmark_drops_them(self, tmp_path):
        """`load_min_rtt` goes through `load_canonical_csv`, so the rows here
        are the rows the solvers ran on -- a 0 ms edge is not one of them."""
        path = _csv(tmp_path / "a.csv", [1.0, 0.0, -5.0, 4.0])
        long, _ = R.load([FakeRun("as01-mesh")], source_csv={"as01-mesh": path})
        assert sorted(long["rtt_ms"]) == [1.0, 4.0]


class TestStats:
    def test_row_order_follows_the_callers_run_order(self):
        """Not sort order: the CSV, the legend and --run-id are one sequence."""
        stats = R.stats_table(_long({"as03-mesh": [1.0], "as01-mesh": [2.0]}))
        assert stats["dataset"].tolist() == ["as03", "as01"]

    def test_extrema_are_observed_not_interpolated(self):
        stats = R.stats_table(_long({"as01-mesh": [1.0, 2.0, 3.0, 100.0]}))
        assert stats.loc[0, "min_ms"] == 1.0
        assert stats.loc[0, "max_ms"] == 100.0

    def test_percentiles_are_emitted_for_every_requested_rung(self):
        stats = R.stats_table(_long({"as01-mesh": list(range(1, 101))}))
        for p in R.PERCENTILES:
            assert f"p{p}_ms" in stats.columns
        assert stats.loc[0, "p50_ms"] == pytest.approx(50.5)

    def test_counts_expose_the_replica_oversampling(self):
        """n_edges >> n_tgs is the caveat the docstring is about, so both are
        in the table rather than only the one the curve is drawn from."""
        stats = R.stats_table(_long({"as01-mesh": [1.0, 2.0, 3.0, 4.0]}))
        assert stats.loc[0, "n_edges"] == 4
        assert stats.loc[0, "n_tgs"] == 2
        assert stats.loc[0, "n_vps"] == 4


class TestTheXCutIsReported:
    """A truncated linear axis cannot show what it truncated."""

    def test_share_within_the_cut_is_computed_per_dataset(self):
        long = _long({"as01-mesh": [1.0, 2.0], "as02-mesh": [1.0, 500.0]})
        stats = R.stats_table(long, x_max_ms=200.0).set_index("dataset")
        assert stats.loc["as01", "share_within_xmax_pct"] == pytest.approx(100.0)
        assert stats.loc["as02", "share_within_xmax_pct"] == pytest.approx(50.0)

    def test_clipped_names_only_the_datasets_that_overflow(self):
        stats = R.stats_table(
            _long({"as01-mesh": [1.0, 2.0], "as02-mesh": [1.0, 500.0]}), x_max_ms=200.0
        )
        assert R.clipped(stats)["dataset"].tolist() == ["as02"]

    def test_nothing_is_clipped_when_the_cut_clears_the_tail(self):
        """The as01-03 case: 200 ms against a 92 ms maximum warns about nothing."""
        stats = R.stats_table(_long({"as01-mesh": [1.0, 92.4]}), x_max_ms=200.0)
        assert R.clipped(stats).empty

    def test_manifest_carries_the_share_and_the_observed_max(self):
        """The figure cannot show the clipped mass, so the numbers must exist."""
        stats = R.stats_table(
            _long({"as01-mesh": [1.0, 500.0]}), x_max_ms=200.0
        )
        body = json.loads(
            R._manifest({"run_ids": ["as01-mesh"], "datasets": ["as01"]},
                        stats, x_max_ms=200.0)
        )
        axis = body["x_axis"]
        assert axis["share_within_xmax_pct"]["as01"] == pytest.approx(50.0)
        assert axis["observed_max_ms"]["as01"] == 500.0
        assert axis["clipped_datasets"] == ["as01"]

    def test_a_clipped_dataset_gets_a_note_naming_it(self):
        """The earlier wording named no dataset; on four curves that is
        useless, since *which* curve stops short is the whole question."""
        stats = R.stats_table(_long({"as01-mesh": [1.0, 500.0]}), x_max_ms=200.0)
        (text, colour), = R.clip_notes(stats)
        assert text.startswith("PRO AS01")
        assert colour == R.DISPLAY["as01"][1]

    def test_the_note_carries_p99_not_the_max(self):
        """A p99 above the axis says the truncation is structural; a max above
        it can be one packet."""
        stats = R.stats_table(_long({"as01-mesh": [1.0] * 99 + [900.0]}),
                              x_max_ms=200.0)
        (text, _), = R.clip_notes(stats)
        assert "p99" in text
        assert "900" not in text

    def test_an_unclipped_dataset_gets_no_note(self):
        """The as01-03 case at 200 ms: a 92 ms maximum warns about nothing."""
        stats = R.stats_table(_long({"as01-mesh": [1.0, 92.4]}), x_max_ms=200.0)
        assert R.clip_notes(stats) == []

    def test_only_the_clipped_datasets_are_named(self):
        stats = R.stats_table(
            _long({"as01-mesh": [1.0, 2.0], "as7018-ripe-mesh": [1.0, 500.0]}),
            x_max_ms=200.0,
        )
        texts = [t for t, _ in R.clip_notes(stats)]
        assert len(texts) == 1
        assert texts[0].startswith("RIPE MIX-ASN")

    def test_each_note_takes_its_own_curves_colour(self):
        """So the note attaches to a line without needing a second legend."""
        stats = R.stats_table(
            _long({"as01-mesh": [1.0, 500.0], "as7018-ripe-mesh": [1.0, 500.0]}),
            x_max_ms=200.0,
        )
        colours = [c for _, c in R.clip_notes(stats)]
        assert colours == [R.DISPLAY["as01"][1], R.DISPLAY["as7018"][1]]

    def test_the_drawn_panel_carries_the_notes(self, tmp_path):
        """`clip_notes` being right is worth nothing if `plot` ignores it."""
        import matplotlib.pyplot as plt

        long = _long({"as01-mesh": [1.0, 500.0]})
        stats = R.stats_table(long, x_max_ms=200.0)
        drawn: list[str] = []
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            drawn.extend(t.get_text() for ax in self.axes for t in ax.texts)
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": ["as01-mesh"]},
                   out_png=tmp_path / "x.png", x_max_ms=200.0)
        finally:
            plt.Figure.savefig = real_savefig
        assert any(t.startswith("PRO AS01") and "p99" in t for t in drawn)


class TestTheScaleIsValidated:
    """symlog's linear window is where it can lie about being logarithmic."""

    def test_an_unknown_scale_is_refused(self):
        with pytest.raises(ValueError, match="--x-scale must be one of"):
            R.validate_axis("log", x_max_ms=100.0, linthresh_ms=1.0)

    def test_linear_is_the_default_and_ignores_linthresh(self):
        """A nonsense linthresh must not block a linear axis that never uses it."""
        assert R.validate_axis("linear", x_max_ms=100.0, linthresh_ms=999.0) == "linear"

    def test_a_linthresh_swallowing_the_axis_is_refused(self):
        """linthresh >= x_max makes the whole axis linear under a log label."""
        with pytest.raises(ValueError, match="at or above"):
            R.validate_axis("symlog", x_max_ms=100.0, linthresh_ms=100.0)

    def test_a_non_positive_linthresh_is_refused(self):
        with pytest.raises(ValueError, match="--linthresh must be positive"):
            R.validate_axis("symlog", x_max_ms=100.0, linthresh_ms=0.0)

    def test_the_share_inside_the_linear_window_is_reported(self):
        """The axis can still be linear-in-disguise for one dataset and not
        another, so the share is per dataset rather than a single verdict."""
        long = _long({"as01-mesh": [0.5, 0.6, 50.0, 60.0], "as02-mesh": [50.0, 60.0]})
        stats = R.stats_table(long, linthresh_ms=1.0).set_index("dataset")
        assert stats.loc["as01", "share_below_linthresh_pct"] == pytest.approx(50.0)
        assert stats.loc["as02", "share_below_linthresh_pct"] == pytest.approx(0.0)

    def test_manifest_carries_the_symlog_block_only_for_symlog(self):
        stats = R.stats_table(_long({"as01-mesh": [0.5, 50.0]}), linthresh_ms=1.0)
        meta = {"run_ids": ["as01-mesh"], "datasets": ["as01"]}
        sym = json.loads(R._manifest(meta, stats, x_max_ms=100.0,
                                     x_scale="symlog", linthresh_ms=1.0))["x_axis"]
        assert sym["scale"] == "symlog"
        assert sym["symlog"]["linthresh_ms"] == 1.0
        assert sym["symlog"]["share_below_linthresh_pct"]["as01"] == pytest.approx(50.0)

        lin = json.loads(R._manifest(meta, stats, x_max_ms=100.0))["x_axis"]
        assert lin["scale"] == "linear"
        assert "symlog" not in lin

    def test_symlog_draws_and_keeps_the_axis_bounds(self, tmp_path):
        """symlog must still start at 0 -- that is the whole reason it is the
        offered scale rather than log."""
        import matplotlib.pyplot as plt

        long = _long({"as01-mesh": [0.5, 5.0, 50.0]})
        stats = R.stats_table(long, x_max_ms=100.0, linthresh_ms=1.0)
        seen = {}
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            ax = self.axes[0]
            seen["scale"] = ax.get_xscale()
            seen["xlim"] = ax.get_xlim()
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": ["as01-mesh"]},
                   out_png=tmp_path / "x.png", x_max_ms=100.0,
                   x_scale="symlog", linthresh_ms=1.0)
        finally:
            plt.Figure.savefig = real_savefig
        assert seen["scale"] == "symlog"
        assert seen["xlim"] == (0.0, 100.0)


class TestTheYAxisIsACdf:
    """A fraction on the axis, a percentage in the prose -- and no mixing."""

    def test_ecdf_returns_a_fraction_not_a_percentage(self):
        _, y = R._ecdf(np.array([3.0, 1.0, 2.0, 4.0]))
        assert y.tolist() == [0.25, 0.5, 0.75, 1.0]

    def test_the_curve_reaches_exactly_one(self):
        """Nothing filtered, nothing double-counted."""
        _, y = R._ecdf(np.arange(1.0, 1001.0))
        assert y[-1] == pytest.approx(1.0)

    def test_the_axis_is_zero_to_one_in_quarters(self, tmp_path):
        import matplotlib.pyplot as plt

        long = _long({"as01-mesh": [1.0, 5.0, 50.0]})
        stats = R.stats_table(long)
        seen = {}
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            ax = self.axes[0]
            seen["ylim"] = ax.get_ylim()
            # MultipleLocator emits ticks past the view limits (-0.25, 1.25)
            # which are clipped at draw time; compare the visible ones.
            lo, hi = ax.get_ylim()
            seen["ticks"] = [t for t in ax.get_yticks() if lo <= t <= hi]
            seen["label"] = ax.get_ylabel()
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": ["as01-mesh"]},
                   out_png=tmp_path / "x.png")
        finally:
            plt.Figure.savefig = real_savefig
        assert seen["ylim"] == (0.0, 1.0)
        assert seen["ticks"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
        assert seen["label"] == "CDF"

    def test_the_csv_still_reports_shares_in_percent(self):
        """The axis changed units; the quotable numbers did not. A reader
        citing `share_within_xmax_pct` as 99.8 must not get 0.998."""
        stats = R.stats_table(_long({"as01-mesh": [1.0, 500.0]}), x_max_ms=200.0)
        assert stats.loc[0, "share_within_xmax_pct"] == pytest.approx(50.0)


class TestTheLegend:
    """Names, not numbers -- and a label no run id could have produced."""

    def test_the_named_datasets_get_their_labels(self):
        assert R._style("as01", 0)[0] == "PRO AS01"
        assert R._style("as02", 1)[0] == "PRO AS02"
        assert R._style("as03", 2)[0] == "PRO AS03"
        assert R._style("as7018", 3)[0] == "RIPE MIX-ASN"

    def test_the_operator_meshes_are_solid_and_the_ripe_mesh_is_grey_dashed(self):
        for d in ("as01", "as02", "as03"):
            assert R._style(d, 0)[2] == "-", d
        _, colour, linestyle = R._style("as7018", 0)
        assert linestyle == "--"
        assert colour == "#8a8a8a"

    def test_solid_lines_are_not_separated_by_hue_alone_on_the_risky_pair(self):
        """The three solid meshes lose their linestyle cue, so their hues must
        not include the orange/green pair deuteranopes cannot tell apart."""
        hues = {R._style(d, 0)[1] for d in ("as01", "as02", "as03")}
        assert "#2f8f4e" not in hues

    def test_an_unknown_dataset_draws_rather_than_raising(self):
        label, colour, linestyle = R._style("as99", 0)
        assert label == "AS99"
        assert linestyle == ":"
        assert colour

    def test_the_drawn_legend_is_names_only(self, tmp_path):
        """Pins "remove data": no n=, no p50= in any legend entry."""
        import matplotlib.pyplot as plt

        long = _long({"as01-mesh": [1.0, 5.0], "as7018-ripe-mesh": [2.0, 6.0]})
        stats = R.stats_table(long)
        seen: list[str] = []
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            seen.extend(t.get_text() for t in self.axes[0].get_legend().get_texts())
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": list(stats.run_id)},
                   out_png=tmp_path / "x.png")
        finally:
            plt.Figure.savefig = real_savefig
        assert seen == ["PRO AS01", "RIPE MIX-ASN"]

    def test_legend_order_follows_the_callers_run_order(self):
        """The four-curve figure reads PRO AS01/02/03 then RIPE MIX-ASN only
        because that is the --run-id order; nothing sorts it."""
        long = _long({
            "as01-mesh": [1.0], "as02-mesh": [2.0],
            "as03-mesh": [3.0], "as7018-ripe-mesh": [4.0],
        })
        stats = R.stats_table(long)
        labels = [R._style(d, i)[0] for i, d in enumerate(stats["dataset"])]
        assert labels == ["PRO AS01", "PRO AS02", "PRO AS03", "RIPE MIX-ASN"]


class TestClippingAnnotationIsOptOut:
    """On by default; suppressing it must not suppress the manifest record."""

    def test_no_notes_when_suppressed(self, tmp_path):
        import matplotlib.pyplot as plt

        long = _long({"as01-mesh": [1.0, 500.0]})
        stats = R.stats_table(long, x_max_ms=200.0)
        drawn: list[str] = []
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            drawn.extend(t.get_text() for ax in self.axes for t in ax.texts)
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": ["as01-mesh"]},
                   out_png=tmp_path / "x.png", x_max_ms=200.0,
                   annotate_clipping=False, show_medians=False)
        finally:
            plt.Figure.savefig = real_savefig
        assert not any("p99" in t for t in drawn)

    def test_the_manifest_records_the_clipping_regardless(self):
        stats = R.stats_table(_long({"as01-mesh": [1.0, 500.0]}), x_max_ms=200.0)
        body = json.loads(R._manifest(
            {"run_ids": ["as01-mesh"], "datasets": ["as01"]}, stats, x_max_ms=200.0
        ))["x_axis"]
        assert body["clipped_datasets"] == ["as01"]
        assert body["share_within_xmax_pct"]["as01"] == pytest.approx(50.0)
        assert body["observed_max_ms"]["as01"] == 500.0

    def test_the_manifest_says_whether_the_panel_annotated(self):
        """A reader must be able to tell a silent panel from an unclipped one."""
        stats = R.stats_table(_long({"as01-mesh": [1.0, 500.0]}), x_max_ms=200.0)
        meta = {"run_ids": ["as01-mesh"], "datasets": ["as01"]}
        on = json.loads(R._manifest(meta, stats, x_max_ms=200.0,
                                    annotate_clipping=True))["x_axis"]
        off = json.loads(R._manifest(meta, stats, x_max_ms=200.0,
                                     annotate_clipping=False))["x_axis"]
        assert on["annotated_on_figure"] is True
        assert off["annotated_on_figure"] is False


class TestMedianDroplines:
    """The median comes back onto the panel, at its own x rather than in a list."""

    @staticmethod
    def _drawn(tmp_path, long, **kwargs):
        import matplotlib.pyplot as plt

        stats = R.stats_table(long, x_max_ms=kwargs.get("x_max_ms", 200.0))
        seen = {"texts": [], "vlines": 0}
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            ax = self.axes[0]
            seen["texts"] = [t.get_text() for t in ax.texts]
            seen["vlines"] = len(ax.collections)
            seen["hlines"] = [ln.get_ydata()[0] for ln in ax.lines
                              if len(set(ln.get_ydata())) == 1]
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": list(stats.run_id)},
                   out_png=tmp_path / "x.png", **kwargs)
        finally:
            plt.Figure.savefig = real_savefig
        return seen

    def test_each_dataset_gets_a_labelled_dropline(self, tmp_path):
        long = _long({"as01-mesh": [10.0, 30.0], "as7018-ripe-mesh": [40.0, 60.0]})
        seen = self._drawn(tmp_path, long)
        assert seen["vlines"] == 2
        assert "20.0 ms" in seen["texts"]
        assert "50.0 ms" in seen["texts"]

    def test_the_half_reference_is_drawn_so_the_verticals_mean_something(self, tmp_path):
        seen = self._drawn(tmp_path, _long({"as01-mesh": [10.0, 30.0]}))
        assert any(y == pytest.approx(0.5) for y in seen["hlines"])

    def test_no_medians_draws_neither_the_lines_nor_the_reference(self, tmp_path):
        long = _long({"as01-mesh": [10.0, 30.0]})
        seen = self._drawn(tmp_path, long, show_medians=False)
        assert seen["vlines"] == 0
        assert not any(y == pytest.approx(0.5) for y in seen["hlines"])
        assert not any(t == "20.0 ms" for t in seen["texts"])

    def test_a_median_beyond_the_cut_is_skipped_not_clamped(self, tmp_path):
        """A label pinned at x_max would read as "the median is x_max", which
        is a stronger and wronger claim than drawing nothing."""
        # The dataset is also clipped, so a p99 note is drawn; assert on the
        # median label specifically rather than on "ms" appearing anywhere.
        long = _long({"as01-mesh": [400.0, 600.0]})
        seen = self._drawn(tmp_path, long, x_max_ms=200.0)
        assert seen["vlines"] == 0
        assert not any(t == "500.0 ms" for t in seen["texts"])

    def test_labels_share_one_anchor_clear_of_every_dropline(self):
        """Per-label `p50 * 1.05` put each label on the next dataset's line."""
        stats = R.stats_table(_long({
            "as01-mesh": [34.7], "as02-mesh": [35.9],
            "as03-mesh": [38.4], "as7018-ripe-mesh": [42.0],
        }))
        anchor = R.label_anchor(stats, x_max_ms=200.0)
        assert anchor > stats["p50_ms"].max()

    def test_no_anchor_when_every_median_is_beyond_the_cut(self):
        stats = R.stats_table(_long({"as01-mesh": [500.0]}), x_max_ms=200.0)
        assert R.label_anchor(stats, x_max_ms=200.0) is None

    def test_the_anchor_ignores_medians_past_the_cut(self):
        """An off-panel median must not push the column off the panel too."""
        stats = R.stats_table(_long({"as01-mesh": [20.0], "as02-mesh": [500.0]}),
                              x_max_ms=200.0)
        assert R.label_anchor(stats, x_max_ms=200.0) == pytest.approx(20.0 * 1.08)

    def test_labels_are_stepped_down_so_close_medians_do_not_overprint(self):
        """34.7 / 35.9 / 38.4 / 42.0 ms is a few mm on a symlog axis."""
        ys = [R._MEDIAN_LABEL_TOP - R._MEDIAN_LABEL_STEP * i for i in range(4)]
        assert len(set(ys)) == 4
        assert all(0 < y < 0.5 for y in ys)


class TestLinearTickStep:
    """`--x-step` is linear-only, and must not silently do nothing on symlog."""

    def test_a_non_positive_step_is_refused(self):
        with pytest.raises(ValueError, match="--x-step must be positive"):
            R.validate_axis("linear", x_max_ms=100.0, linthresh_ms=1.0, x_step_ms=0.0)

    def test_a_step_wider_than_the_axis_is_refused(self):
        """One tick at 0 is not an axis."""
        with pytest.raises(ValueError, match="exceeds --x-max"):
            R.validate_axis("linear", x_max_ms=100.0, linthresh_ms=1.0, x_step_ms=200.0)

    def test_unset_is_allowed_and_leaves_the_automatic_locator(self):
        assert R.validate_axis("linear", x_max_ms=100.0, linthresh_ms=1.0,
                               x_step_ms=None) == "linear"

    def test_the_ticks_land_on_the_step(self, tmp_path):
        import matplotlib.pyplot as plt

        long = _long({"as01-mesh": [10.0, 30.0, 90.0]})
        stats = R.stats_table(long, x_max_ms=100.0)
        seen = {}
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            ax = self.axes[0]
            lo, hi = ax.get_xlim()
            seen["ticks"] = [t for t in ax.get_xticks() if lo <= t <= hi]
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": ["as01-mesh"]},
                   out_png=tmp_path / "x.png", x_max_ms=100.0, x_step_ms=10.0)
        finally:
            plt.Figure.savefig = real_savefig
        assert seen["ticks"] == pytest.approx([float(v) for v in range(0, 101, 10)])

    def test_symlog_ignores_the_step_rather_than_fighting_it(self, tmp_path):
        """A MultipleLocator on a log axis produces ticks that crowd into the
        left decade; symlog keeps its own locator regardless of --x-step."""
        import matplotlib.pyplot as plt

        long = _long({"as01-mesh": [1.0, 10.0, 90.0]})
        stats = R.stats_table(long, x_max_ms=100.0)
        seen = {}
        real_savefig = plt.Figure.savefig

        def spy(self, *a, **k):
            lo, hi = self.axes[0].get_xlim()
            seen["ticks"] = [t for t in self.axes[0].get_xticks() if lo <= t <= hi]
            return real_savefig(self, *a, **k)

        plt.Figure.savefig = spy
        try:
            R.plot(long, stats, meta={"run_ids": ["as01-mesh"]},
                   out_png=tmp_path / "x.png", x_max_ms=100.0,
                   x_scale="symlog", linthresh_ms=1.0, x_step_ms=10.0)
        finally:
            plt.Figure.savefig = real_savefig
        assert seen["ticks"] != pytest.approx([float(v) for v in range(0, 101, 10)])

    def test_the_manifest_records_the_step(self):
        stats = R.stats_table(_long({"as01-mesh": [1.0, 2.0]}))
        meta = {"run_ids": ["as01-mesh"], "datasets": ["as01"]}
        with_step = json.loads(R._manifest(meta, stats, x_max_ms=100.0,
                                           x_step_ms=10.0))["x_axis"]
        auto = json.loads(R._manifest(meta, stats, x_max_ms=100.0))["x_axis"]
        assert with_step["tick_step_ms"] == 10.0
        assert auto["tick_step_ms"] is None


class TestNoPoolingGuard:
    def test_a_tg_in_two_datasets_is_not_an_error(self):
        """Pins an absence. Nothing shares a denominator across runs here, so
        the sibling figures' `guard_disjoint_tgs` must not be wired in."""
        long = _long({"as01-mesh": [1.0, 2.0], "as02-mesh": [3.0, 4.0]})
        assert set(long.loc[long.run_id == "as01-mesh", "tg_id"]) == set(
            long.loc[long.run_id == "as02-mesh", "tg_id"]
        )
        stats = R.stats_table(long)
        assert len(stats) == 2
        assert stats["n_edges"].tolist() == [2, 2]


class TestPrivacy:
    def test_no_output_column_carries_a_location(self):
        stats = R.stats_table(_long({"as01-mesh": [1.0, 2.0]}))
        forbidden = ("lat", "lon", "city", "country", "site", "region", "asn")
        for col in stats.columns:
            assert not any(f in col.lower() for f in forbidden), col

    def test_manifest_holds_no_coordinate(self):
        """Keys and values, not raw substrings -- "population" contains "lat"."""
        stats = R.stats_table(_long({"as01-mesh": [1.0, 2.0]}))
        body = json.loads(
            R._manifest({"run_ids": ["as01-mesh"], "datasets": ["as01"]},
                        stats, x_max_ms=200.0)
        )
        forbidden = {"lat", "lon", "latitude", "longitude", "city", "country",
                     "site", "site_id", "region", "asn", "coordinates"}
        coord_pair = re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}")

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    assert k.lower() not in forbidden, f"{path}.{k}"
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif isinstance(node, str):
                assert not coord_pair.search(node), f"{path}: {node!r}"

        walk(body)


class TestBuild:
    def test_writes_png_csv_and_manifest(self, tmp_path):
        a = _csv(tmp_path / "a.csv", [1.0, 2.0, 3.0])
        b = _csv(tmp_path / "b.csv", [10.0, 20.0])
        out = tmp_path / "analysis"
        pngs = R.build_for_runs(
            [FakeRun("as01-260728-260802-mesh"), FakeRun("as02-260728-260802-mesh")],
            analysis_root=out,
            source_csv={"as01-260728-260802-mesh": a, "as02-260728-260802-mesh": b},
        )
        assert len(pngs) == 1 and pngs[0].exists() and pngs[0].stat().st_size > 0
        d = pngs[0].parent
        assert d.name == "as01+as02@260728-260802-mesh"
        assert (d / R.CSV_NAME).exists() and (d / R.MANIFEST_NAME).exists()

    def test_a_non_positive_x_max_raises_before_anything_is_read(self):
        """FakeRun carries no CSV, so reaching `load` would raise something
        else -- the axis check must come first."""
        with pytest.raises(ValueError, match="--x-max must be positive"):
            R.build_for_runs([FakeRun("as01-mesh")], x_max_ms=0.0)

    def test_a_bad_scale_raises_before_anything_is_read(self):
        with pytest.raises(ValueError, match="--x-scale must be one of"):
            R.build_for_runs([FakeRun("as01-mesh")], x_scale="log")

    def test_symlog_writes_the_same_three_artifacts(self, tmp_path):
        a = _csv(tmp_path / "a.csv", [0.5, 2.0, 30.0])
        out = tmp_path / "analysis"
        pngs = R.build_for_runs(
            [FakeRun("as01-260728-260802-mesh")],
            x_max_ms=100.0, x_scale="symlog", linthresh_ms=1.0,
            analysis_root=out,
            source_csv={"as01-260728-260802-mesh": a},
        )
        d = pngs[0].parent
        assert pngs[0].exists() and (d / R.CSV_NAME).exists()
        body = json.loads((d / R.MANIFEST_NAME).read_text())
        assert body["x_axis"]["scale"] == "symlog"
