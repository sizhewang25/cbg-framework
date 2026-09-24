"""The answer space: both partitions, placed TGs, and the artifacts round trip."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.geodesy import pairwise_km

SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)
EWR = (40.6895, -74.1745)
JFK = (40.6413, -73.7781)
HONOLULU = (21.31, -157.86)


def _tgs(coords=(SEATTLE, OMAHA, EWR, JFK), per=3) -> pd.DataFrame:
    rows = [
        {"tg_id": f"tg-{i}-{k}", "tg_lat": lat, "tg_lon": lon}
        for i, (lat, lon) in enumerate(coords)
        for k in range(per)
    ]
    return pd.DataFrame(rows)


def _space(nside=128, **kw):
    return A.build_answer_space(_tgs(**kw), nside=nside, run_id="test-run")


class TestInputContract:
    def test_missing_columns_are_named(self):
        with pytest.raises(ValueError, match="tg_lon"):
            A.build_answer_space(_tgs().drop(columns="tg_lon"), run_id="r")

    def test_duplicate_tg_id_is_refused(self):
        t = _tgs()
        t.loc[1, "tg_id"] = t.loc[0, "tg_id"]
        with pytest.raises(ValueError, match="duplicate tg_id"):
            A.build_answer_space(t, run_id="r")

    def test_a_site_off_the_landmass_is_refused(self):
        with pytest.raises(ValueError, match="outside the landmass"):
            _space(coords=(SEATTLE, HONOLULU))


class TestBothPartitions:
    def test_replicas_are_one_site(self):
        s = _space()
        assert s.meta["n_tgs"] == 12 and s.meta["n_sites"] == 4

    def test_ewr_and_jfk_are_two_grids_but_one_seed(self):
        """The two partitions disagree here, which is the point of having both."""
        s = _space()
        by_coord = s.sites.set_index(["site_lat", "site_lon"])
        ewr, jfk = by_coord.loc[EWR], by_coord.loc[JFK]
        assert ewr["grid_id"] != jfk["grid_id"]
        assert ewr["seed_id"] == jfk["seed_id"]
        assert s.meta["n_seeds"] == 3 and s.meta["n_sites_merged"] == 2

    def test_tg_seed_is_its_sites_seed(self):
        s = _space()
        merged = s.tgs.merge(s.sites[["site_id", "seed_id"]], on="site_id")
        assert (merged["tg_seed_id"] == merged["seed_id"]).all()

    def test_tg_dist_to_seed_is_half_the_ewr_jfk_gap(self):
        s = _space()
        ny = s.tgs.loc[s.tgs["tg_lat"].round(4) == round(EWR[0], 4), "tg_dist_to_seed_km"]
        gap = pairwise_km([EWR[0]], [EWR[1]], [JFK[0]], [JFK[1]])[0, 0]
        assert ny.iloc[0] == pytest.approx(gap / 2, abs=0.05)

    def test_seeds_never_split_as_the_grid_grows(self):
        counts = [_space(nside=n).n_seeds for n in G.NSIDE_LADDER]
        assert counts == sorted(counts, reverse=True)

    def test_meta_carries_the_glossary_and_landmass(self):
        m = _space().meta
        assert m["glossary"] == A.GLOSSARY
        assert m["landmass"]["buffer_km"] == pytest.approx(G.grid_km(128), abs=1e-3)
        assert m["seed_rule"]["diameter_km"] == pytest.approx(G.grid_km(128), abs=1e-3)


class TestRoundTrip:
    def test_every_artifact_round_trips(self, tmp_path):
        s = _space()
        back = A.load_answer_space(s.write(tmp_path))
        assert back.nside == s.nside
        for name in ("grids", "sites", "seeds", "tgs"):
            pd.testing.assert_frame_equal(
                getattr(back, name), getattr(s, name), check_dtype=False, check_exact=False
            )
        assert back.tgs["tg_seed_id"].dtype == "int64"

    def test_sweep_row_reports_both_partitions(self):
        row = A.sweep_row(_space())
        assert row["n_grids"] == 4 and row["n_seeds"] == 3
