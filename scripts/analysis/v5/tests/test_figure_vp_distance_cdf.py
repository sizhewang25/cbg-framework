"""The three-curve VP distance CDF, and the two things it must not hide.

The load-bearing tests are `TestZeroGapIsNotDropped` and
`TestPointwiseInequality`. Both guard failures that a log axis makes
*invisible* rather than obvious, which is the only reason they need pinning.

A gap of exactly 0 cannot be placed on a log axis. The naive fix -- filter to
`gap > 0` before taking the ECDF -- silently deletes 243 of 1,269 TGs on
as01-03, and deletes precisely the TGs the surrounding argument is about, so
the curve would then start at 0% and read as "no TG has a small gap" when the
truth is "a fifth of TGs have none at all".

`d_geo <= d_sp` is a per-TG inequality. Two marginal CDFs cannot express it,
and a negative gap cannot be drawn on a log axis, so a frame-alignment bug
would produce a plausible-looking figure. `population` raises instead.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_vp_distance_cdf as C


def _long(geo: list[float], sping: list[float], *, methods: int = 2) -> pd.DataFrame:
    """A long frame: `methods` methods over the same TGs, one run.

    The two distances are TG properties repeated onto every method's row,
    which is the shape `load` returns and the reason `population` dedups.
    """
    n = len(geo)
    return pd.DataFrame(
        {
            "run_id": ["r"] * n * methods,
            "tg_id": [f"tg-{i}" for i in range(n)] * methods,
            "method": sum(([f"m_{m}"] * n for m in range(methods)), []),
            "geo_vp_dist_to_tg_km": geo * methods,
            "sping_vp_dist_to_tg_km": sping * methods,
        }
    )


class TestPopulationDeduplicates:
    def test_methods_collapse_to_one_row_per_tg(self):
        """Four TGs scored by three methods is still four TGs."""
        pop = C.population(_long([1.0, 2.0, 3.0, 4.0], [1.0, 9.0, 3.0, 40.0], methods=3))
        assert len(pop) == 4
        assert sorted(pop.tg_id) == ["tg-0", "tg-1", "tg-2", "tg-3"]

    def test_gap_is_the_difference(self):
        pop = C.population(_long([1.0, 2.0], [1.0, 9.0]))
        assert pop[C.GAP].tolist() == [0.0, 7.0]

    def test_same_tg_id_in_two_runs_is_two_rows(self):
        """Keyed on (run_id, tg_id), so a relaxed disjointness guard cannot
        silently merge two runs' TGs into one."""
        long = _long([1.0, 2.0], [5.0, 6.0], methods=1)
        other = long.assign(run_id="s")
        pop = C.population(pd.concat([long, other], ignore_index=True))
        assert len(pop) == 4


class TestPointwiseInequality:
    def test_violation_raises_rather_than_drawing(self):
        """d_sp < d_geo is arithmetically impossible and must not be plotted."""
        with pytest.raises(ValueError, match="impossible"):
            C.population(_long([5.0, 2.0], [1.0, 9.0]))

    def test_equality_is_not_a_violation(self):
        """The whole point is that equality is common; it is not an error."""
        pop = C.population(_long([3.0, 3.0], [3.0, 3.0]))
        assert (pop[C.GAP] == 0).all()

    def test_manifest_carries_min_gap_and_the_verdict(self):
        """The figure cannot show a negative gap, so the number must exist."""
        pop = C.population(_long([1.0, 2.0], [1.0, 9.0]))
        body = json.loads(C._manifest(
            {"run_ids": ["r"], "nside": 128, "n_tgs": 2}, pop, C.stats_table(pop)
        ))
        assert body["pointwise_inequality"]["min_gap_km"] == 0.0
        assert body["pointwise_inequality"]["holds"] is True


class TestZeroGapIsNotDropped:
    """The zero mass is the finding; a log axis cannot draw it."""

    @staticmethod
    def _mixed() -> pd.DataFrame:
        # 4 of 10 TGs have coincident VPs, so the gap curve must start at 40%.
        geo = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        sping = [1.0, 2.0, 3.0, 4.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0]
        return C.population(_long(geo, sping, methods=1))

    def test_curve_starts_at_the_zero_share_not_at_zero(self):
        pop = self._mixed()
        x, y, zero_share = C._gap_curve(pop[C.GAP].to_numpy())
        assert zero_share == pytest.approx(40.0)
        assert x[0] == C.X_MIN_KM
        assert y[0] == pytest.approx(40.0)

    def test_every_tg_is_accounted_for(self):
        """The curve must reach 100%: nothing filtered, nothing double-counted."""
        pop = self._mixed()
        _, y, _ = C._gap_curve(pop[C.GAP].to_numpy())
        assert y[-1] == pytest.approx(100.0)

    def test_dropping_zeros_would_lose_the_share(self):
        """Pins the bug this class exists for, by exhibiting it."""
        gap = self._mixed()[C.GAP].to_numpy()
        naive = np.sort(gap[gap > 0])
        assert len(naive) == 6            # 4 TGs silently gone
        _, y, _ = C._gap_curve(gap)
        assert y[0] > 0                   # the honest curve does not start at 0

    def test_all_zero_gaps_give_a_flat_full_curve(self):
        """Every VP coincident: the curve is 100% from the axis floor."""
        pop = C.population(_long([1.0, 2.0], [1.0, 2.0], methods=1))
        x, y, zero_share = C._gap_curve(pop[C.GAP].to_numpy())
        assert zero_share == pytest.approx(100.0)
        assert len(x) == 1 and y[0] == pytest.approx(100.0)


class TestStats:
    def test_distinct_counts_report_the_quantization(self):
        """1,269 TGs carrying 43 values is the reason this is a CDF."""
        pop = C.population(_long([7.0] * 5 + [9.0], [7.0] * 5 + [99.0], methods=1))
        stats = C.stats_table(pop).set_index("series")
        assert stats.loc[C.GEO, "n_distinct"] == 2
        assert stats.loc[C.GEO, "n"] == 6

    def test_zero_share_is_emitted_for_the_gap(self):
        pop = C.population(_long([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 33.0, 44.0], methods=1))
        stats = C.stats_table(pop).set_index("series")
        assert stats.loc[C.GAP, "n_zero"] == 2
        assert stats.loc[C.GAP, "zero_share_pct"] == pytest.approx(50.0)

    def test_min_positive_skips_the_zeros(self):
        """The flat run to the first positive gap is real, so it is reported."""
        pop = C.population(_long([1.0, 2.0, 3.0], [1.0, 20.0, 33.0], methods=1))
        stats = C.stats_table(pop).set_index("series")
        assert stats.loc[C.GAP, "min_positive_km"] == pytest.approx(18.0)


class TestPrivacy:
    def test_no_output_column_carries_a_location(self):
        """Target-based analysis only: no coordinates, no place names."""
        pop = C.population(_long([1.0, 2.0], [1.0, 9.0]))
        forbidden = ("lat", "lon", "city", "country", "site", "region", "asn")
        for col in C.stats_table(pop).columns:
            assert not any(f in col.lower() for f in forbidden), col

    def test_manifest_holds_no_coordinate(self):
        """Scan keys and values, not raw substrings -- "population" contains
        "lat", so a substring scan over the whole document is a false
        positive generator and would be switched off the first time it fired.
        """
        pop = C.population(_long([1.0, 2.0], [1.0, 9.0]))
        body = json.loads(C._manifest(
            {"run_ids": ["r"], "nside": 128, "n_tgs": 2}, pop, C.stats_table(pop)
        ))
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


class TestPlot:
    def test_it_writes_a_png(self, tmp_path):
        pop = C.population(_long([1.0, 2.0, 3.0], [1.0, 20.0, 33.0], methods=1))
        png = C.plot(pop, meta={"run_ids": ["r"], "n_tgs": 3},
                     out_png=tmp_path / "cdf.png")
        assert png.exists() and png.stat().st_size > 0
