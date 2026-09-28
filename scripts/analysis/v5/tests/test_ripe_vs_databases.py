"""The databases-on-the-CDF figure: what a lookup contributes, and what it may not.

Three things carry the figure's meaning:

  * a **database is measured exactly like a method** -- great-circle to the
    same `tg_lat`/`tg_lon` the CBG rows were scored against, so a number on one
    curve is comparable with a number on another;
  * a TG the database has **no record for is unanswered**, never an error of
    zero and never (0, 0) -- the Gulf of Guinea is not a geolocation; and
  * the databases stay **out of the shared method tables** -- adding a term to
    `methods.METHOD_TERMS` would break its pin to `LABEL_HUES`, and adding a
    hue would re-order `TERM_ORDER`, hence every other figure's legend.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_error_cdf as E
from scripts.analysis.v5.modules import methods as MT
from scripts.analysis.v5.modules import ripe_vs_databases as RVD
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

NSIDE = 128

#: Four TGs at known coordinates, so a hand-computed offset is checkable.
TGS = {
    "tg-0": (40.0, -100.0),
    "tg-1": (41.0, -100.0),
    "tg-2": (42.0, -100.0),
    "tg-3": (43.0, -100.0),
}


def _tgs_frame(errors=None) -> pd.DataFrame:
    """A scored frame over `TGS`, every row SUCCESS."""
    ids = list(TGS)
    dist = list(errors) if errors is not None else [10.0 * (i + 1) for i in range(len(ids))]
    return pd.DataFrame(
        {
            "tg_id": ids,
            "tg_lat": [TGS[i][0] for i in ids],
            "tg_lon": [TGS[i][1] for i in ids],
            "status": ["SUCCESS"] * len(ids),
            "pred_lat": [40.0] * len(ids),
            "pred_lon": [-100.0] * len(ids),
            C.GRID_OFFSET: [0] * len(ids),
            "cell_label": ["correct"] * len(ids),
            "pred_dist_to_tg_km": dist,
            "pred_dist_to_seed_km": dist,
        }
    )


def _write_run(root, *, methods=("million_scale_cbg",)) -> RunPaths:
    run = RunPaths(run_id="ripe-x-mesh", root=root, source="s", setup="t")
    out = run.classify_dir(NSIDE, root=root)
    frames = {m: _tgs_frame() for m in methods}
    for method, df in frames.items():
        df.to_parquet(out / C.TGS_PARQUET.format(method=method), index=False)
    C.summarize(frames, NSIDE).to_csv(out / C.ACCURACY_CSV, index=False)
    return run


def _write_dbs(db_dir, *, ipinfo: dict | None = None, maxmind: dict | None = None):
    """Write both shipped-shaped JSONs. Defaults put each DB on its TG exactly."""
    db_dir.mkdir(parents=True, exist_ok=True)
    ipinfo = TGS if ipinfo is None else ipinfo
    maxmind = TGS if maxmind is None else maxmind
    (db_dir / RVD.DATABASES[RVD.IPINFO][0]).write_text(
        json.dumps({ip: {"ip": ip, "loc": f"{lat},{lon}"} for ip, (lat, lon) in ipinfo.items()})
    )
    (db_dir / RVD.DATABASES[RVD.MAXMIND][0]).write_text(
        json.dumps({ip: [lat, lon] for ip, (lat, lon) in maxmind.items()})
    )


class TestParsing:
    def test_ipinfo_loc_string_is_split_into_a_pair(self, tmp_path):
        _write_dbs(tmp_path)
        assert RVD.load_database(RVD.IPINFO, tmp_path)["tg-0"] == (40.0, -100.0)

    def test_maxmind_pair_is_read_in_order(self, tmp_path):
        _write_dbs(tmp_path)
        assert RVD.load_database(RVD.MAXMIND, tmp_path)["tg-0"] == (40.0, -100.0)

    def test_an_unparseable_record_is_dropped_not_defaulted(self, tmp_path):
        """A missing `loc` must not become (0, 0) -- that is a real coordinate."""
        (tmp_path / RVD.DATABASES[RVD.IPINFO][0]).write_text(
            json.dumps({"tg-0": {"ip": "tg-0"}, "tg-1": {"loc": "not,a,number"},
                        "tg-2": {"loc": "42.0,-100.0"}})
        )
        got = RVD.load_database(RVD.IPINFO, tmp_path)
        assert set(got) == {"tg-2"}

    def test_a_missing_file_names_where_it_ships(self, tmp_path):
        with pytest.raises(MissingArtifactError, match="static_datasets"):
            RVD.load_database(RVD.IPINFO, tmp_path)


class TestDistances:
    def test_an_exact_lookup_is_zero_error(self, tmp_path):
        run = _write_run(tmp_path)
        roster = RVD.load_roster(run, NSIDE, analysis_root=tmp_path)
        entry = RVD.database_errors(roster, dict(TGS))
        # atol, not exact: `geodesy.elementwise_km` goes through unit vectors,
        # so a point against itself lands within a metre rather than on zero.
        assert np.allclose(entry["errors"], 0.0, atol=1e-3)
        assert entry["n_solved"] == len(TGS) and entry["n_failed"] == 0

    def test_a_one_degree_offset_is_about_111_km(self, tmp_path):
        run = _write_run(tmp_path)
        roster = RVD.load_roster(run, NSIDE, analysis_root=tmp_path)
        shifted = {ip: (lat + 1.0, lon) for ip, (lat, lon) in TGS.items()}
        entry = RVD.database_errors(roster, shifted)
        assert np.allclose(entry["errors"], 111.2, atol=0.5)

    def test_an_uncovered_tg_is_unanswered_not_zero(self, tmp_path):
        run = _write_run(tmp_path)
        roster = RVD.load_roster(run, NSIDE, analysis_root=tmp_path)
        entry = RVD.database_errors(roster, {"tg-0": TGS["tg-0"]})
        assert entry["n_tgs"] == 4 and entry["n_solved"] == 1 and entry["n_failed"] == 3
        assert len(entry["errors"]) == 1

    def test_uncovered_tgs_reach_the_sentinel_not_the_curve(self, tmp_path):
        """The censoring path must see a database's gaps as unanswered rows."""
        run = _write_run(tmp_path)
        roster = RVD.load_roster(run, NSIDE, analysis_root=tmp_path)
        entry = RVD.database_errors(roster, {"tg-0": TGS["tg-0"]})
        drawn = E.censor({"db": entry}, policy=E.SENTINEL)["db"]
        assert drawn["n_censored"] == 3
        assert list(drawn["errors"]).count(E.SENTINEL_KM) == 3

    def test_the_roster_ground_truth_is_the_scored_one(self, tmp_path):
        """Read from the parquet, so both sides share one ground truth."""
        run = _write_run(tmp_path)
        roster = RVD.load_roster(run, NSIDE, analysis_root=tmp_path)
        assert list(roster.columns) == list(RVD.ROSTER_COLUMNS)
        assert dict(zip(roster["tg_id"], zip(roster["tg_lat"], roster["tg_lon"]))) == TGS


class TestSeries:
    def test_databases_are_labelled_by_their_own_terms(self):
        assert RVD.series_label(RVD.MAXMIND) == "MM"
        assert RVD.series_label(RVD.IPINFO) == "IPI"

    def test_methods_keep_the_package_terms(self):
        assert RVD.series_label("million_scale_cbg") == "SOI"

    def test_databases_are_drawn_after_every_method(self):
        order = RVD.series_order(
            {RVD.IPINFO: {}, "million_scale_cbg": {}, RVD.MAXMIND: {}, "vanilla_cbg": {}}
        )
        assert order[-2:] == [RVD.MAXMIND, RVD.IPINFO]
        assert [MT.method_label(m) for m in order[:2]] == ["SOI", "VAN"]

    def test_databases_are_dashed_and_methods_are_not(self):
        colors = MT.method_colors(["million_scale_cbg"])
        assert RVD.series_style(RVD.MAXMIND, colors)["linestyle"] != "-"
        assert RVD.series_style("million_scale_cbg", colors)["linestyle"] == "-"

    def test_database_hues_are_not_method_hues(self):
        """A database sharing a method's hex would read as that method."""
        assert not set(RVD.DB_HUES.values()) & set(MT.LABEL_HUES.values())

    def test_the_shared_tables_are_left_alone(self):
        """`test_methods` pins METHOD_TERMS to LABEL_HUES; TERM_ORDER is that
        table's order, and every other figure's legend inherits it."""
        for series in RVD.DATABASES:
            assert series not in MT.METHOD_LABELS
        for term in RVD.DB_TERMS:
            assert term not in MT.METHOD_TERMS
            assert term not in MT.TERM_ORDER


class TestArtifacts:
    def test_it_writes_the_triple_and_names_the_databases(self, tmp_path):
        run = _write_run(tmp_path, methods=("million_scale_cbg", "vanilla_cbg"))
        db_dir = tmp_path / "static"
        _write_dbs(db_dir)
        png = RVD.build_for_run(run, nside=NSIDE, db_dir=db_dir, analysis_root=tmp_path)
        assert png.exists()

        out = png.parent
        table = pd.read_csv(out / RVD.artifact_names()[1])
        assert set(table.loc[table["is_database"], "method_label"]) == {"MM", "IPI"}
        # An exact lookup: every percentile is zero to the CSV's 3dp.
        assert (table.loc[table["is_database"], E.pcol(50)] == 0).all()

        body = json.loads((out / RVD.artifact_names()[2]).read_text())
        assert set(body["databases"]) == set(RVD.DATABASES)
        assert body["databases"][RVD.IPINFO]["n_covered"] == len(TGS)
        assert body["terms"]["MM"] == RVD.DB_TERMS["MM"]
        # The glossary must still carry the methods on the panel.
        assert body["terms"]["SOI"] == MT.METHOD_TERMS["SOI"]

    def test_the_sentinel_run_does_not_overwrite_the_other(self, tmp_path):
        run = _write_run(tmp_path)
        db_dir = tmp_path / "static"
        _write_dbs(db_dir)
        RVD.build_for_run(run, nside=NSIDE, db_dir=db_dir, analysis_root=tmp_path)
        png = RVD.build_for_run(
            run, nside=NSIDE, db_dir=db_dir, analysis_root=tmp_path, unanswered=E.SENTINEL
        )
        out = png.parent
        assert {p.name for p in out.glob("ripe_vs_databases*.png")} == {
            RVD.artifact_names()[0],
            RVD.artifact_names(E.SENTINEL)[0],
        }

    def test_partial_coverage_is_reported_not_hidden(self, tmp_path):
        run = _write_run(tmp_path)
        db_dir = tmp_path / "static"
        _write_dbs(db_dir, maxmind={"tg-0": TGS["tg-0"]})
        png = RVD.build_for_run(run, nside=NSIDE, db_dir=db_dir, analysis_root=tmp_path)
        body = json.loads((png.parent / RVD.artifact_names()[2]).read_text())
        assert body["databases"][RVD.MAXMIND]["n_covered"] == 1
        assert body["databases"][RVD.MAXMIND]["n_uncovered"] == 3
