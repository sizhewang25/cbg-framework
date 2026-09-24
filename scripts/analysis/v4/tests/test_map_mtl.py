"""Tests for the MTL case viewer.

Four of these are load-bearing rather than smoke.

`TestStatusIsTheSamePartitionAsTheBars` pins the viewer's per-target verdict
against `classify.summarize`'s five outcome counts. They are the same five
buckets, and a map whose popup said `ring1` while the bar chart counted the
target under `beyond` would be worse than either being wrong alone.

`TestVerdictRings` pins the drawn neighbourhood against `ring_distance` — the
set form and the pair form of one BFS, which is the cross-check that makes
keeping both implementations safe.

`TestWindingIsClockwise` pins the shoelace on every ring the payload emits. A
counter-clockwise filled ring makes Plotly fill the antipodal complement, which
is invisible until something is filled and then flattens the whole map.

`TestRetiredKeysAreGone` walks the serialized payload for nearest-seed
vocabulary. v3's viewer was built entirely around a rule v4 retired, and the
failure mode of this port is not a crash — it is a page that renders a retired
verdict convincingly.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import edges as E
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules import map_mtl as M
from scripts.analysis.v4.modules.answer_space import build_answer_space
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths, resolve_run

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8998, -97.0403)
ARCTIC = (63.74, -97.47)  # the tg-e1a1545 case, which must score `beyond`

_TARGETS = (CHI, SJC, NYC, DFW)
NSIDE = 128
METHOD = "test_cbg"


def _space(coords=_TARGETS, nside=NSIDE):
    return build_answer_space(
        pd.DataFrame(
            {
                "target_id": [f"tg-{i}" for i in range(len(coords))],
                "target_lat": [c[0] for c in coords],
                "target_lon": [c[1] for c in coords],
            }
        ),
        nside=nside,
    )


class _Run:
    """The handful of `RunPaths` attributes `build_payload` actually reads."""

    run_id = "test-run"
    source = "generic_csv"
    setup = "anchors_to_probes"


def _cell_centre(cell: int, nside: int = NSIDE) -> tuple[float, float]:
    lat, lon = H.pix2ang(np.array([cell]), nside)[0]
    return float(lat), float(lon)


def _preds_at_rings(space, rings: list[int]) -> list[tuple[float, float] | None]:
    """One prediction per target, planted `rings[i]` cells from its truth cell.

    `None` means no coordinate at all. A ring index above `MAX_RING` is planted
    in the Arctic, which is the case the metric exists to catch.
    """
    out = []
    for cell, k in zip(space.assignments["cell_id"], rings):
        if k is None:
            out.append(None)
        elif k > H.MAX_RING:
            out.append(ARCTIC)
        else:
            nbhd = H.ring_cells(int(cell), NSIDE)
            out.append(_cell_centre(nbhd[k][0]))
    return out


def _scored(space, *, rings=None, statuses=None):
    """A frame shaped exactly as `classify.score_method` returns one."""
    assign = space.assignments
    n = len(assign)
    rings = rings if rings is not None else [0] * n
    preds = _preds_at_rings(space, rings)
    frame = pd.DataFrame(
        {
            "target_id": assign["target_id"].to_numpy(),
            "target_lat": assign["target_lat"].to_numpy(),
            "target_lon": assign["target_lon"].to_numpy(),
            "pred_lat": [p[0] if p else np.nan for p in preds],
            "pred_lon": [p[1] if p else np.nan for p in preds],
            "status": statuses or ["SUCCESS"] * n,
            "fold": 0,
        }
    )
    return C.score_method(frame, space)


def _vps():
    """Three VPs, so all three marker classes are exercised: vp-0 is the
    shortest-ping VP, vp-1 is measured-but-not-selected, vp-2 is never measured
    and so is latent for every target."""
    return pd.DataFrame(
        {
            "vp_id": ["vp-0", "vp-1", "vp-2"],
            "vp_lat": [41.8, 37.3, 25.0],
            "vp_lon": [-87.6, -121.8, -80.2],
            "vp_asn": ["64500", "64501", None],
            "vp_country": ["US", "US", None],
        }
    )


def _edges(space, vps=None):
    vps = _vps() if vps is None else vps
    rows = []
    for tid in space.assignments["target_id"]:
        for vp, rtt in (("vp-0", 5.0), ("vp-1", 22.5)):
            rows.append({"target_id": tid, "vp_id": vp, "rtt_ms": rtt})
    return pd.DataFrame(rows)


def _context(space):
    n = len(space.assignments)
    return pd.DataFrame(
        {
            "target_id": space.assignments["target_id"].to_numpy(),
            "sping_vp_id": ["vp-0"] * n,
            "min_inflation": [1.5] * n,
        }
    )


def _payload(space, scored, **kw):
    kw.setdefault("edges", _edges(space))
    kw.setdefault("vps", _vps())
    kw.setdefault("context", _context(space))
    kw.setdefault("voronoi", [])
    return M.build_payload(_Run(), space, METHOD, scored=scored, **kw)


def _walk(node):
    """Every key at every depth of a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _shoelace(ring) -> float:
    """Signed area in an (x=lon, y=lat) frame. Positive is counter-clockwise."""
    a = 0.0
    for (la_a, lo_a), (la_b, lo_b) in zip(ring, ring[1:] + ring[:1]):
        a += lo_a * la_b - lo_b * la_a
    return a / 2.0


class TestStatusIsTheSamePartitionAsTheBars:
    def test_status_of_covers_the_ladder(self):
        assert M.status_of(True, 0) == "ring0"
        assert M.status_of(True, 1) == "ring1"
        assert M.status_of(True, 2) == "ring2"
        assert M.status_of(True, -1) == "beyond"
        assert M.status_of(False, 0) == "failed"
        assert set(M.STATUSES) == {"ring0", "ring1", "ring2", "beyond", "failed"}

    def test_unsolved_outranks_the_ring(self):
        """A FALLBACK carries the Shortest-Ping VP's coordinate, which can land
        in the truth's own cell. The method still did not answer."""
        assert M.status_of(False, 0) == "failed"

    def test_the_histogram_equals_summarize(self):
        space = _space()
        scored = _scored(
            space,
            rings=[0, 1, 2, 9],
            statuses=["SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS"],
        )
        payload = _payload(space, scored)
        got = pd.Series([t["status"] for t in payload["targets"]]).value_counts()
        row = C.summarize({METHOD: scored}, NSIDE).iloc[0]
        for status in M.STATUSES:
            assert int(got.get(status, 0)) == int(row[f"n_{status}"]), status

    def test_fallbacks_land_in_the_failed_bucket_of_both(self):
        space = _space()
        scored = _scored(
            space, rings=[0, 0, 1, 2],
            statuses=["FALLBACK", "SUCCESS", "SUCCESS", "SUCCESS"],
        )
        payload = _payload(space, scored)
        got = pd.Series([t["status"] for t in payload["targets"]]).value_counts()
        row = C.summarize({METHOD: scored}, NSIDE).iloc[0]
        assert int(got.get("failed", 0)) == int(row["n_failed"]) == 1
        # ...and NOT under ring0, even though its coordinate is in the cell.
        assert int(got.get("ring0", 0)) == int(row["n_ring0"]) == 1

    def test_baseline_rows_are_solved_not_failed(self):
        space = _space()
        scored = _scored(space, statuses=["BASELINE"] * 4)
        payload = _payload(space, scored)
        assert {t["status"] for t in payload["targets"]} == {"ring0"}


class TestVerdictRings:
    """What the PAYLOAD carries. `ring_cells`' own grid properties are pinned
    in `test_healpix.py`, beside the primitive they describe."""

    def test_payload_rings_are_the_truths_neighbourhood(self):
        space = _space()
        scored = _scored(space)
        payload = _payload(space, scored)
        for t in payload["targets"]:
            assert t["ring_cells"][0] == [t["tg_cell"]]
            assert t["ring_cells"] == H.ring_cells(t["tg_cell"], NSIDE)

    def test_the_ring_column_comes_from_classify(self):
        space = _space()
        scored = _scored(space, rings=[0, 1, 2, 9]).set_index("target_id")
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for t in payload["targets"]:
            assert t["ring"] == int(scored.loc[t["target_id"], "ring"])

    def test_neighbourhoods_are_memoised_per_cell(self, monkeypatch):
        """Targets in one class share a neighbourhood exactly; recomputing it
        per target is ~399 calls where ~18 will do."""
        calls = []
        real = H.ring_cells
        monkeypatch.setattr(
            H, "ring_cells", lambda p, n, *a, **k: (calls.append(p), real(p, n, *a, **k))[1]
        )
        # Two targets in one cell, one elsewhere.
        space = _space((CHI, CHI, SJC))
        M.ring_neighbourhood(space.assignments["cell_id"], NSIDE)
        assert len(calls) == len(set(calls)) == 2


class TestCellTable:
    def test_every_referenced_cell_is_present(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for t in payload["targets"]:
            for ring in t["ring_cells"]:
                for c in ring:
                    assert str(c) in payload["cells"]
            if t["pred_cell"] >= 0:
                assert str(t["pred_cell"]) in payload["cells"]
        for s in payload["seeds"]:
            assert str(s["cell_id"]) in payload["cells"]

    def test_no_unreferenced_entries(self):
        space = _space()
        payload = _payload(space, _scored(space))
        used = {str(s["cell_id"]) for s in payload["seeds"]}
        for t in payload["targets"]:
            used.update(str(c) for ring in t["ring_cells"] for c in ring)
            if t["pred_cell"] >= 0:
                used.add(str(t["pred_cell"]))
        assert set(payload["cells"]) == used

    def test_a_missing_prediction_adds_no_cell(self):
        space = _space()
        scored = _scored(space, rings=[0, 0, 0, None])
        payload = _payload(space, scored)
        missing = [t for t in payload["targets"] if t["pred"] is None]
        assert len(missing) == 1
        assert missing[0]["pred_cell"] == -1

    def test_shared_cells_are_serialized_once(self):
        space = _space((CHI, CHI, CHI))
        payload = _payload(space, _scored(space))
        assert payload["n_seeds"] == 1
        # One class, one neighbourhood: 1 + 8 + 16 cells, not three copies.
        assert len(payload["cells"]) == 25


class TestWindingIsClockwise:
    def test_cw_ring_flips_a_counter_clockwise_input(self):
        ccw = M._cw_ring([0, 1, 1, 0], [0, 0, 1, 1])
        assert _shoelace(ccw[:-1]) <= 0

    def test_cw_ring_closes_the_ring(self):
        ring = M._cw_ring([0, 1, 1, 0], [0, 0, 1, 1])
        assert ring[0] == ring[-1]

    def test_every_payload_cell_is_clockwise(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for cid, ring in payload["cells"].items():
            assert ring[0] == ring[-1], cid
            assert _shoelace(ring[:-1]) <= 0, cid

    def test_a_density_argmax_cell_is_clockwise(self):
        rings = H.cell_rings(np.array([1000]), NSIDE)
        ring = M._cw_ring(rings[0][:, 0], rings[0][:, 1])
        assert _shoelace(ring[:-1]) <= 0


class TestRetiredKeysAreGone:
    RETIRED = {"top_seeds", "margin_km", "seeds_crossed", "proximity", "voronoi_rank"}

    def test_no_nearest_seed_key_at_any_depth(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        assert not (set(_walk(payload)) & self.RETIRED)

    def test_the_module_exports_no_retired_helper(self):
        for name in ("_top_seeds", "_proximity_label", "seed_crossing_matrix"):
            assert not hasattr(M, name), name


class TestVoronoiIsContextNotTheVerdict:
    def test_it_is_seeded_from_the_class_centres(self):
        """v4's seeds ARE the occupied cells' centres, so the overlay is the
        partition over cell centroids the request asked for."""
        space = _space()
        centres = H.pix2ang(space.seeds["cell_id"].to_numpy(), NSIDE)
        assert np.allclose(space.seeds["seed_lat"], centres[:, 0])
        assert np.allclose(space.seeds["seed_lon"], centres[:, 1])

    def test_fewer_than_two_seeds_yields_nothing(self):
        """A partition of one is not a partition, and scipy would raise."""
        assert M.conus_voronoi_rings(_space((CHI,)).seeds) == []

    def test_rings_are_clockwise_and_closed(self):
        rings = M.conus_voronoi_rings(_space().seeds)
        assert rings
        for ring in rings:
            assert ring[0] == ring[-1]
            assert _shoelace(ring[:-1]) <= 0


class TestPayloadIsStrictJson:
    def test_nan_predictions_serialize(self):
        space = _space()
        scored = _scored(space, rings=[0, 0, 0, None])
        json.dumps(_payload(space, scored), allow_nan=False)

    def test_a_partial_eval_source_degrades_rather_than_raising(self, tmp_path):
        """`eval_per_target.csv` carrying one of the two context columns must
        give the other as null, not a KeyError out of `build_payload`."""
        run = _write_run(tmp_path / "bench")
        space = _space()
        pd.DataFrame({
            "target_id": space.assignments["target_id"],
            "min_inflation": 1.5,          # and no shortest_ping_vp_id
        }).to_csv(run.eval_source_dir / "ds_eval_per_target.csv", index=False)

        ctx = M.eval_context(run)
        assert list(ctx.columns) == ["target_id", "sping_vp_id", "min_inflation"]
        payload = _payload(space, _scored(space), context=ctx)
        # None, not the string "None" -- the viewer would print that as a VP id.
        assert all(t["sping_vp_id"] is None for t in payload["targets"])
        assert all(t["min_inflation"] == 1.5 for t in payload["targets"])
        json.dumps(payload, allow_nan=False)

    def test_missing_context_serializes(self):
        space = _space()
        payload = _payload(
            space, _scored(space), context=pd.DataFrame(columns=["target_id"])
        )
        assert all(t["min_inflation"] is None for t in payload["targets"])
        json.dumps(payload, allow_nan=False)

    def test_render_escapes_a_closing_script_tag(self):
        space = _space()
        html = M.render_html(_payload(space, _scored(space)))
        for token in ("__PAYLOAD__", "__SCRIPT__", "__TITLE__"):
            assert token not in html
        body = re.search(
            r'<script id="data" type="application/json">([\s\S]*?)</script>', html
        )
        assert body and "</script>" not in body.group(1)

    def test_the_title_names_the_rung(self):
        space = _space()
        html = M.render_html(_payload(space, _scored(space)))
        assert f"healpix nside {NSIDE}" in html


class TestObservationLayer:
    def test_one_entry_per_measured_pair(self):
        space = _space()
        payload = _payload(space, _scored(space))
        for t in payload["targets"]:
            assert t["n_measured"] == 2
            assert t["n_total"] == 3        # vp-2 is latent everywhere
            assert {o[0] for o in t["obs"]} == {"vp-0", "vp-1"}

    def test_inflation_is_rtt_over_the_soi_rtt(self):
        from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE

        assert M._inflation(10.0, 500.0) == pytest.approx(
            10.0 / (THEORETICAL_SLOPE * 500.0)
        )

    def test_absent_ids_are_null_not_the_string_none(self):
        assert M._safe_str(None) is None
        assert M._safe_str(float("nan")) is None
        assert M._safe_str(np.nan) is None
        assert M._safe_str("vp-0") == "vp-0"

    def test_inflation_is_undefined_at_zero_distance(self):
        assert M._inflation(10.0, 0.0) is None
        assert M._inflation(10.0, -1.0) is None


class TestShortestPingRenders:
    def test_the_baseline_has_no_constraints_or_region(self):
        space = _space()
        scored = _scored(space, statuses=["BASELINE"] * 4)
        payload = M.build_payload(
            _Run(), space, M.SHORTEST_PING, scored=scored, edges=_edges(space),
            vps=_vps(), context=_context(space), voronoi=[],
        )
        assert payload["is_baseline"] is True
        for t in payload["targets"]:
            assert t["rings"] == [] and t["n_kept"] == 0 and t["region"] is None

    def test_the_title_says_classification_not_mtl(self):
        space = _space()
        scored = _scored(space, statuses=["BASELINE"] * 4)
        html = M.render_html(M.build_payload(
            _Run(), space, M.SHORTEST_PING, scored=scored, edges=_edges(space),
            vps=_vps(), context=_context(space), voronoi=[],
        ))
        assert "Classification map" in html
        assert "MTL map" not in html


class TestRegionMode:
    def test_an_unknown_mode_is_refused(self):
        space = _space()
        with pytest.raises(ValueError, match="region_mode"):
            _payload(space, _scored(space), region_mode="probabilistic")

    def test_region_is_pred_cell_only_when_the_grids_coincide(self):
        space = _space()
        scored = _scored(space)
        same = _payload(
            space, scored, region_mode=M.REGION_DENSITY, density_nside=NSIDE
        )
        other = _payload(
            space, scored, region_mode=M.REGION_DENSITY, density_nside=64
        )
        geometric = _payload(space, scored, region_mode=M.REGION_GEOMETRIC)
        assert same["region_is_pred_cell"] is True
        assert other["region_is_pred_cell"] is False
        assert geometric["region_is_pred_cell"] is False

    def test_the_two_nsides_stay_separate_fields(self):
        """`nside` is the rung scored on; `density_nside` is the MTL's own
        hypothesis grid. Folding them would mislabel the drawn cell."""
        space = _space()
        payload = _payload(
            space, _scored(space), region_mode=M.REGION_DENSITY, density_nside=64
        )
        assert payload["nside"] == NSIDE and payload["density_nside"] == 64


# ---- a real run on disk -----------------------------------------------------


def _write_run(root: Path, *, method: str = METHOD, mtl: str = "planar_circle") -> RunPaths:
    """A benchmark tree with one fold, nested constraint columns and all."""
    run = RunPaths(run_id="r", root=root, source="generic_csv", setup="s")
    combo = run.combo_dir(method, "fold_0")
    combo.mkdir(parents=True, exist_ok=True)
    space = _space()
    scored = _scored(space, rings=[0, 1, 2, 9])

    participant = pa.struct([
        ("vp_id", pa.string()), ("vp_lat", pa.float64()), ("vp_lon", pa.float64()),
        ("rtt_ms", pa.float64()), ("echoed_upper_km", pa.float64()),
        ("echoed_lower_km", pa.float64()),
    ])
    ltd = pa.struct([
        ("vp_id", pa.string()), ("success", pa.bool_()),
        ("upper_km", pa.float64()), ("lower_km", pa.float64()),
    ])
    n = len(scored)
    table = pa.table({
        "target_id": pa.array(scored["target_id"].astype(str)),
        "target_lat": pa.array(scored["target_lat"].astype(float)),
        "target_lon": pa.array(scored["target_lon"].astype(float)),
        "pred_lat": pa.array(scored["pred_lat"].astype(float)),
        "pred_lon": pa.array(scored["pred_lon"].astype(float)),
        "status": pa.array(scored["status"].astype(str)),
        "mtl_participants": pa.array(
            [[{"vp_id": "vp-0", "vp_lat": 41.8, "vp_lon": -87.6, "rtt_ms": 5.0,
               "echoed_upper_km": 500.0, "echoed_lower_km": 0.0}]] * n,
            type=pa.list_(participant),
        ),
        "ltd_predictions": pa.array(
            [[{"vp_id": "vp-0", "success": True, "upper_km": 500.0, "lower_km": 0.0},
              {"vp_id": "vp-1", "success": True, "upper_km": 900.0, "lower_km": 0.0}]] * n,
            type=pa.list_(ltd),
        ),
    })
    pq.write_table(table, combo / "targets.parquet")
    (combo / "run.json").write_text(json.dumps({"mtl": mtl, "mtl_kwargs": {}}))

    # The canonical edge CSV, recorded where `edges.resolve_source_csv` looks
    # first. Written into the run tree rather than under datasets/ so the
    # fixture is self-contained.
    csv = run.setup_dir / "edges.csv"
    vps = _vps()
    rows = []
    for tid, tlat, tlon in zip(
        space.assignments["target_id"], space.assignments["target_lat"],
        space.assignments["target_lon"],
    ):
        for vp, rtt in (("vp-0", 5.0), ("vp-1", 22.5)):
            v = vps[vps.vp_id == vp].iloc[0]
            rows.append({
                "vp_id": vp, "vp_lat": v.vp_lat, "vp_lon": v.vp_lon,
                "target_id": tid, "target_lat": tlat, "target_lon": tlon,
                "rtt_ms": rtt,
            })
    pd.DataFrame(rows).to_csv(csv, index=False)
    run.eval_source_dir.mkdir(parents=True, exist_ok=True)
    (run.eval_source_dir / "ds_eval_stats.json").write_text(
        json.dumps({"csv": str(csv)})
    )

    _vps().to_csv(run.setup_dir / "vps.csv", index=False)
    pd.DataFrame({
        "target_id": space.assignments["target_id"],
        "target_lat": space.assignments["target_lat"],
        "target_lon": space.assignments["target_lon"],
    }).to_csv(run.setup_dir / "targets.csv", index=False)
    return run


class TestLoadMtlSpecs:
    def test_it_reads_the_combo_run_json(self, tmp_path):
        run = _write_run(tmp_path)
        specs = M.load_mtl_specs(run, METHOD)
        assert list(specs) == [0]
        assert specs[0][0] == "planar_circle"
        assert json.loads(specs[0][1]) == {}

    def test_an_unreadable_spec_is_skipped_not_raised(self, tmp_path):
        run = _write_run(tmp_path)
        (run.combo_dir(METHOD, "fold_0") / "run.json").write_text("{not json")
        assert M.load_mtl_specs(run, METHOD) == {}

    def test_a_missing_run_json_degrades_to_geometric(self, tmp_path):
        run = _write_run(tmp_path)
        (run.combo_dir(METHOD, "fold_0") / "run.json").unlink()
        assert M.combo_region_mode(run, METHOD) == M.REGION_GEOMETRIC


class TestRegionCache:
    def _specs(self):
        return {0: ("planar_circle", '{"n_pts": 64}')}

    def test_a_memoised_empty_stays_skipped(self, tmp_path):
        M._write_region_cache(tmp_path, METHOD, "tg-0", None)
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") == {}

    def test_a_changed_spec_drops_the_cache(self, tmp_path):
        M._write_region_cache(tmp_path, METHOD, "tg-0", {"kind": "polygon"})
        M._invalidate_stale_cache(tmp_path, METHOD, self._specs())
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") is not None
        M._invalidate_stale_cache(
            tmp_path, METHOD, {0: ("planar_circle", '{"n_pts": 256}')}
        )
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") is None

    def test_a_missing_sidecar_is_adopted(self, tmp_path):
        """Caches written before the guard existed must survive it."""
        M._write_region_cache(tmp_path, METHOD, "tg-0", {"kind": "polygon"})
        M._invalidate_stale_cache(tmp_path, METHOD, self._specs())
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") == {"kind": "polygon"}

    def test_a_corrupt_cache_entry_reads_as_uncached(self, tmp_path):
        path = M._region_cache_path(tmp_path, METHOD, "tg-0")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{truncated")
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") is None


class TestCacheIsRungFree:
    def test_the_cache_dir_carries_no_nside(self, tmp_path):
        run = _write_run(tmp_path / "bench")
        cache = run.mtl_region_cache_dir(root=tmp_path / "out")
        assert "healpix" not in str(cache)
        # ...while the rendered pages do separate by rung.
        a = run.mtl_map_dir(128, root=tmp_path / "out")
        b = run.mtl_map_dir(64, root=tmp_path / "out")
        assert a != b and cache.parent == a.parent == b.parent

    def test_a_second_rung_reuses_the_replay(self, tmp_path):
        """The whole point of the rung-free cache: `--nside 64` must not re-pay
        a replay measured in tens of minutes."""
        run = _write_run(tmp_path / "bench")
        out = tmp_path / "out"
        scored = _scored(_space())
        scored["mtl_participants"] = [
            [{"vp_id": "vp-0", "vp_lat": 41.8, "vp_lon": -87.6, "rtt_ms": 5.0,
              "echoed_upper_km": 500.0, "echoed_lower_km": 0.0}]
        ] * len(scored)
        cache = run.mtl_region_cache_dir(root=out)
        M.replay_mtl(run, METHOD, scored, cache_dir=cache, workers=1)
        n_cached = len(list((cache / METHOD).glob("tg-*.json")))
        assert n_cached == len(scored)

        calls = []
        real = M._region_task
        try:
            M._region_task = lambda task: calls.append(task) or real(task)
            M.replay_mtl(run, METHOD, scored, cache_dir=cache, workers=1)
        finally:
            M._region_task = real
        assert calls == [], "a second render replayed regions it had cached"


class TestBuildForRun:
    def test_it_writes_one_page_per_method(self, tmp_path):
        run = _write_run(tmp_path / "bench")
        out = tmp_path / "out"
        rendered = M.build_for_run(
            run, methods=[METHOD], nside=NSIDE, analysis_root=out, regions=False
        )
        assert len(rendered) == 1
        method, path, payload = rendered[0]
        assert path.name == f"mtl_map.{METHOD}.html"
        assert path.parent.name == "healpix-128"
        assert payload["nside"] == NSIDE
        html = path.read_text()
        for token in ("__PAYLOAD__", "__SCRIPT__", "__TITLE__"):
            assert token not in html

    def test_no_regions_writes_no_cache(self, tmp_path):
        run = _write_run(tmp_path / "bench")
        out = tmp_path / "out"
        M.build_for_run(
            run, methods=[METHOD], nside=NSIDE, analysis_root=out, regions=False
        )
        assert not list(run.mtl_region_cache_dir(root=out).glob("**/tg-*.json"))

    def test_an_absent_edge_csv_drops_one_layer_not_the_map(self, tmp_path):
        """The map's headline claim is that it renders on a bare benchmark run.
        The canonical CSV lives outside that tree, so its absence must cost the
        per-VP RTT layer and nothing else."""
        run = _write_run(tmp_path / "bench")
        (run.eval_source_dir / "ds_eval_stats.json").unlink()
        notes = []
        _, _, payload = M.build_for_run(
            run, methods=[METHOD], nside=NSIDE, analysis_root=tmp_path / "out",
            regions=False, progress=notes.append,
        )[0]
        assert any("per-VP RTTs omitted" in n for n in notes)
        assert all(t["obs"] == [] and t["n_measured"] == 0 for t in payload["targets"])
        # The verdict layer is untouched: it comes from the run's own parquets.
        assert all(t["ring_cells"][0] == [t["tg_cell"]] for t in payload["targets"])

    def test_a_mesh_superset_is_refused_rather_than_degraded(self, tmp_path):
        """The opposite case: that CSV parses and every RTT in it is real, but
        they are edges this arm never measured."""
        run = _write_run(tmp_path / "bench")
        (run.eval_source_dir / "ds_eval_stats.json").unlink()
        run.target_space_json.write_text(json.dumps({
            "csv": str(run.setup_dir / "edges.csv"), "csv_is_mesh_superset": True,
        }))
        with pytest.raises(E.MeshSupersetError):
            M.build_for_run(
                run, methods=[METHOD], nside=NSIDE,
                analysis_root=tmp_path / "out", regions=False,
            )

    def test_a_bad_nside_is_refused(self, tmp_path):
        run = _write_run(tmp_path / "bench")
        with pytest.raises(ValueError, match="power of two"):
            M.build_for_run(run, methods=[METHOD], nside=100, regions=False,
                            analysis_root=tmp_path / "out")


# ---- real runs --------------------------------------------------------------

MESH_RUNS = [
    "as01-260728-260802-mesh",
    "as02-260728-260802-mesh",
    "as03-260728-260802-mesh",
]


def _real(run_id: str):
    try:
        return resolve_run(run_id)
    except MissingArtifactError as exc:
        pytest.skip(f"benchmark output not present: {exc}")


class TestOnRealRuns:
    @pytest.mark.parametrize("run_id", MESH_RUNS)
    def test_the_map_agrees_with_accuracy_csv(self, run_id, tmp_path):
        """The cross-artifact assertion. `accuracy.csv` is written by a
        different command from a separate read of the same parquets; if the
        viewer's five counts differ from its five, one of them is lying."""
        run = _real(run_id)
        method = "vanilla_cbg"
        if method not in run.combo_ids:
            pytest.skip(f"{method} not in {run_id}")
        rendered = M.build_for_run(
            run, methods=[method], nside=NSIDE, analysis_root=tmp_path, regions=False
        )
        _, _, payload = rendered[0]
        got = pd.Series([t["status"] for t in payload["targets"]]).value_counts()

        space = build_answer_space(
            __import__(
                "scripts.analysis.v4.modules.answer_space", fromlist=["load_targets"]
            ).load_targets(run),
            nside=NSIDE,
        )
        scored = C.score_method(
            C.load_method_frame(run, method), space
        )
        row = C.summarize({method: scored}, NSIDE).iloc[0]
        for status in M.STATUSES:
            assert int(got.get(status, 0)) == int(row[f"n_{status}"]), status

    def test_the_page_stays_openable_over_file_url(self, tmp_path):
        run = _real(MESH_RUNS[0])
        _, path, _ = M.build_for_run(
            run, methods=["vanilla_cbg"], nside=NSIDE,
            analysis_root=tmp_path, regions=False,
        )[0]
        assert path.stat().st_size < 12_000_000


# ---- the viewer itself ------------------------------------------------------


def _run_viewer(html_path: Path) -> dict:
    """Execute the real viewer over a rendered page, or skip without node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    harness = Path(__file__).with_name("test_map_mtl_viewer.js")
    proc = subprocess.run(
        [node, str(harness), str(html_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """One rendered page, driven through every control once."""
    tmp = tmp_path_factory.mktemp("viewer")
    run = _write_run(tmp / "bench")
    _, path, _ = M.build_for_run(
        run, methods=[METHOD], nside=NSIDE,
        analysis_root=tmp / "out", regions=False,
    )[0]
    return _run_viewer(path)


class TestTheViewerExecutes:
    """Python cannot check the viewer: a typo in draw() renders a blank page and
    every payload assertion still passes. So the real JS is executed."""

    def test_it_drew_every_target(self, report):
        assert report["targets"] == 4
        assert report["draws"] > 0

    def test_the_ring_layers_are_drawn(self, report):
        for counts in report["ringStructure"]:
            assert counts[0] == 1
            assert 0 < counts[1] <= 8
            assert 0 < counts[2] <= 16

    def test_the_verdict_agrees_with_the_drawn_ring0(self, report):
        assert report["verdictChecked"] > 0

    def test_the_voronoi_is_labelled_context(self, report):
        assert any("context" in name for name in report["allLayers"])

    def test_the_ring_layers_are_named(self, report):
        joined = " ".join(report["allLayers"])
        assert "ring 0" in joined and "ring 1" in joined and "ring 2" in joined

    def test_no_second_plotly_tooltip(self, report):
        assert report["tooltipTraces"] == []

    def test_the_status_filter_is_the_ring_ladder(self, report):
        assert "ring0" in report["statusOptions"]
        assert "within1" in report["statusOptions"]
        assert "fail" not in report["statusOptions"]

    def test_every_mark_reports_through_a_panel(self, report):
        assert report["popups"] == report["hovers"] > 0
        assert set(report["panels"]) >= {"target", "pred", "vp"}

    def test_the_meta_strip_names_the_rung(self, report):
        assert "nside" in report["meta"]
