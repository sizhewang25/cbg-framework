"""A traffic-weighted run is scored over its **mesh's** answer space.

Its evaluated TGs survived a flow filter, so they are a subset of the mesh. Built
from them, the answer space loses every site the filter emptied, and each lost
seed's cell goes to a neighbour: a prediction landing at a dropped site turns
from `wrong` into `correct`. v3 fixed this, v4 and v5 lost it; these pin the
port and every load path that refuses a space built the old way.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_outcome_map as OM
from scripts.analysis.v5.modules import map_answer_space as MA
from scripts.analysis.v5.modules import map_mtl as MM
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

NSIDE = 128
COMBO = "octant_cbg"
REPLICAS = 3

#: Five sites, far enough apart that each is its own seed at nside 128.
CHICAGO = (41.9742, -87.9073)
LOS_ANGELES = (34.0489, -118.2570)
NEW_YORK = (40.7178, -74.0090)
HOUSTON = (29.7520, -95.3660)
SEATTLE = (47.6146, -122.3390)
SPREAD = [CHICAGO, LOS_ANGELES, NEW_YORK, HOUSTON, SEATTLE]

#: The traffic filter keeps Chicago and New York, and empties the other three.
KEPT = (0, 2)


def _tg_id(site: int, k: int) -> str:
    return f"tg-{site}-{k}"


def _mesh_rows(coords=SPREAD):
    return [
        {"target_id": _tg_id(i, k), "target_lat": lat, "target_lon": lon}
        for i, (lat, lon) in enumerate(coords)
        for k in range(REPLICAS)
    ]


def _write_mesh_csv(path, coords=SPREAD):
    """Canonical (VP, TG) rows: every TG repeated once per VP, as the real ones are."""
    rec = [
        {"vp_id": f"vp-{v}", "vp_lat": 0.0, "vp_lon": 0.0, **r, "rtt_ms": 10.0, "weight": 1.0}
        for r in _mesh_rows(coords)
        for v in range(2)
    ]
    pd.DataFrame(rec).to_csv(path, index=False)


def _make_run(
    tmp_path, *, source=A.WEIGHTED_SOURCE, kept=KEPT, preds=None, cfg=None, mesh=True,
    run_id="wrun",
):
    """A one-fold run on disk. `preds` maps tg_id -> (lat, lon); default = the TG's own site."""
    mesh_csv = tmp_path / f"{run_id}-mesh.csv"
    if mesh:
        _write_mesh_csv(mesh_csv)
    setup = tmp_path / "outputs" / run_id / source / "anchors_to_probes"
    combo = setup / "fold_0" / COMBO
    combo.mkdir(parents=True)

    rows = [r for r in _mesh_rows() if int(r["target_id"].split("-")[1]) in kept]
    t = pd.DataFrame(rows)
    preds = preds or {}
    t["pred_lat"] = [preds.get(i, (la, lo))[0] for i, la, lo in
                     zip(t["target_id"], t["target_lat"], t["target_lon"])]
    t["pred_lon"] = [preds.get(i, (la, lo))[1] for i, la, lo in
                     zip(t["target_id"], t["target_lat"], t["target_lon"])]
    t["status"] = "SUCCESS"
    t.to_parquet(combo / "targets.parquet", index=False)

    body = cfg if cfg is not None else {
        "run_id": run_id,
        "benchmark": {"source_kwargs": {"mesh_csv_path": str(mesh_csv)}},
    }
    cfg_path = tmp_path / f"{run_id}.yaml"
    cfg_path.write_text(yaml.safe_dump(body))
    (setup / "target_space.json").write_text(json.dumps({"config": str(cfg_path)}))
    return RunPaths(run_id=run_id, root=tmp_path / "outputs", source=source,
                    setup="anchors_to_probes")


def _build(run, tmp_path):
    (space,) = A.build_for_run(run, nsides=(NSIDE,), analysis_root=tmp_path / "analysis")
    return space


class TestTheUniverseIsTheMesh:
    def test_every_mesh_site_is_a_seed(self, tmp_path):
        space = _build(_make_run(tmp_path), tmp_path)
        assert space.meta["n_sites"] == 5 and space.n_seeds == 5
        assert space.meta["n_tgs"] == 5 * REPLICAS

    def test_provenance_records_the_counterfactual(self, tmp_path):
        prov = _build(_make_run(tmp_path), tmp_path).meta["targets_provenance"]
        assert prov["targets_source"].endswith("wrun-mesh.csv")
        assert prov["n_tgs_in_space"] == 5 * REPLICAS
        assert prov["n_tgs_scored_by_run"] == len(KEPT) * REPLICAS
        assert prov["n_seeds_if_built_from_run_tgs"] == len(KEPT)
        assert prov["n_sites_scored_by_run"] == len(KEPT)
        assert prov["n_seeds_with_no_scored_tg"] == 5 - len(KEPT)

    def test_sites_carry_their_scored_count(self, tmp_path):
        _build(_make_run(tmp_path), tmp_path)
        space = A.load_answer_space(tmp_path / "analysis" / "wrun" / "answer-space" / "healpix-128")
        scored = A.site_n_scored(space)
        at = space.sites.assign(n=scored).set_index(["site_lat", "site_lon"])["n"]
        assert at.loc[CHICAGO] == REPLICAS and at.loc[NEW_YORK] == REPLICAS
        assert at.loc[HOUSTON] == 0 and at.loc[SEATTLE] == 0 and at.loc[LOS_ANGELES] == 0

    def test_ids_equal_the_mesh_arms(self, tmp_path):
        """Same sites, same seeds, same ids: the arms differ only in who is scored."""
        weighted = _build(_make_run(tmp_path), tmp_path)
        mesh_tgs = pd.DataFrame(_mesh_rows()).rename(columns=A.BENCHMARK_TG_COLUMNS)
        mesh = A.build_answer_space(mesh_tgs, nside=NSIDE, run_id="wrun-mesh")
        cols = ["site_id", "site_lat", "site_lon", "seed_id"]
        pd.testing.assert_frame_equal(weighted.sites[cols], mesh.sites[cols])
        pd.testing.assert_frame_equal(weighted.seeds, mesh.seeds)

    def test_a_mesh_run_is_untouched(self, tmp_path):
        run = _make_run(tmp_path, source="generic_csv", cfg={"run_id": "wrun"})
        space = _build(run, tmp_path)
        assert space.n_seeds == len(KEPT)
        assert "targets_provenance" not in space.meta
        assert (A.site_n_scored(space) == space.sites["n_tgs"]).all()


class TestItIsRefusedNotDefaulted:
    def test_no_mesh_csv_path(self, tmp_path):
        run = _make_run(tmp_path, cfg={"run_id": "wrun", "benchmark": {"source_kwargs": {}}})
        with pytest.raises(MissingArtifactError, match="mesh_csv_path"):
            _build(run, tmp_path)

    def test_mesh_csv_missing_on_disk(self, tmp_path):
        with pytest.raises(MissingArtifactError, match="does not exist"):
            _build(_make_run(tmp_path, mesh=False), tmp_path)

    def test_no_config_reachable(self, tmp_path):
        run = _make_run(tmp_path)
        run.target_space_json.write_text(json.dumps({}))
        with pytest.raises(MissingArtifactError, match="no config is reachable"):
            _build(run, tmp_path)

    def test_a_scored_tg_absent_from_the_mesh(self, tmp_path):
        run = _make_run(tmp_path)
        _write_mesh_csv(tmp_path / "wrun-mesh.csv", coords=SPREAD[:1])
        with pytest.raises(ValueError, match="absent from"):
            _build(run, tmp_path)

    def test_a_scored_tg_at_another_coordinate(self, tmp_path):
        run = _make_run(tmp_path)
        moved = [SEATTLE, *SPREAD[1:]]
        _write_mesh_csv(tmp_path / "wrun-mesh.csv", coords=moved)
        with pytest.raises(ValueError, match="different coordinate"):
            _build(run, tmp_path)


class TestTheBugItFixes:
    #: A Chicago TG predicted at Houston. Houston's site is filtered out.
    PRED = {_tg_id(0, 0): HOUSTON}

    def test_a_prediction_at_a_dropped_site_is_wrong(self, tmp_path):
        run = _make_run(tmp_path, preds=self.PRED)
        _build(run, tmp_path)
        summary = C.score_rung(run, NSIDE, methods=[COMBO], analysis_root=tmp_path / "analysis")
        row = summary.iloc[0]
        assert row["n_tgs"] == len(KEPT) * REPLICAS
        assert row["n_cell_wrong"] == 1
        assert row["n_cell_correct"] == len(KEPT) * REPLICAS - 1

    def test_the_old_filtered_space_would_have_called_it_correct(self, tmp_path):
        """The counterfactual, stated: this is the inflation the port removes."""
        run = _make_run(tmp_path, preds=self.PRED)
        filtered = A.build_answer_space(A.load_tgs(run), nside=NSIDE, run_id=run.run_id)
        scored = C.score_method(C.load_method_frame(run, COMBO), filtered)
        assert (scored["cell_label"] == "correct").all()

    def test_the_manifest_names_the_universe(self, tmp_path):
        run = _make_run(tmp_path)
        _build(run, tmp_path)
        C.score_rung(run, NSIDE, methods=[COMBO], analysis_root=tmp_path / "analysis")
        man = json.loads((run.classify_dir(NSIDE, root=tmp_path / "analysis")
                          / C.MANIFEST_JSON).read_text())
        assert man["n_sites"] == 5 and man["n_sites_scored"] == len(KEPT)
        assert man["n_seeds"] == 5 and man["n_seeds_scored"] == len(KEPT)
        assert man["targets_source"].endswith("wrun-mesh.csv")


class TestAStaleSpaceIsRefusedEverywhere:
    """A weighted space written before the port has no provenance. Every loader
    that hands seeds to a figure refuses it: classify, the answer-space map
    (the outcome map and the contest family load through it), and the MTL map."""

    @pytest.fixture
    def stale(self, tmp_path):
        run = _make_run(tmp_path)
        old = A.build_answer_space(A.load_tgs(run), nside=NSIDE, run_id=run.run_id)
        old.write(run.answer_space_dir(NSIDE, root=tmp_path / "analysis"))
        return run, tmp_path / "analysis"

    def test_classify(self, stale):
        run, root = stale
        with pytest.raises(MissingArtifactError, match="build-answer-space"):
            C.score_rung(run, NSIDE, methods=[COMBO], analysis_root=root)

    def test_map_answer_space_load_rung(self, stale):
        run, root = stale
        with pytest.raises(MissingArtifactError, match="post-filter"):
            MA.load_rung(run, NSIDE, analysis_root=root)

    def test_outcome_map_goes_through_the_same_loader(self):
        assert OM.load_rung is MA.load_rung

    def test_map_mtl_load_rung(self, stale):
        run, root = stale
        with pytest.raises(MissingArtifactError, match="post-filter"):
            MM.load_rung(run, NSIDE, analysis_root=root)


class TestFiguresOverTheMeshSpace:
    def test_outcome_map_seed_rows_skip_seeds_with_no_scored_tg(self, tmp_path):
        run = _make_run(tmp_path, preds=TestTheBugItFixes.PRED)
        space = _build(run, tmp_path)
        scored = C.score_method(C.load_method_frame(run, COMBO), space)
        rows = OM.cohort_rows(scored, "correct")
        counts = {"run_id": run.run_id}
        out = OM.seed_rows(rows, space.seeds, counts)
        assert out["n_tgs"].sum() == len(rows)
        assert set(out["seed_id"]) == set(rows["tg_seed_id"])
        assert out[["seed_lat", "seed_lon"]].notna().all().all()

    def test_answer_space_map_counts_both(self, tmp_path):
        run = _make_run(tmp_path)
        space = _build(run, tmp_path)
        counts = MA.rung_counts(space, polygons={}, agreement=1.0)
        assert counts["n_sites"] == 5 and counts["n_sites_scored"] == len(KEPT)
        assert counts["n_tgs_scored"] == len(KEPT) * REPLICAS
        assert counts["targets_source"].endswith("wrun-mesh.csv")

    def test_site_n_scored_falls_back_to_n_tgs(self):
        tgs = pd.DataFrame(_mesh_rows()).rename(columns=A.BENCHMARK_TG_COLUMNS)
        space = A.build_answer_space(tgs, nside=NSIDE, run_id="r")
        assert A.SCORED_COL not in space.sites.columns
        assert np.array_equal(A.site_n_scored(space), space.sites["n_tgs"])
