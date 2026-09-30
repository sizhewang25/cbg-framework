"""Tests for the MTL case viewer.

Five of these are load-bearing rather than smoke.

`TestStatusIsTheSamePartitionAsTheBars` pins the viewer's per-TG grid verdict
against `classify.summarize`'s five outcome counts, and its cell verdict
against the three cell counts. A map whose popup said `ring1` while the bar
chart counted the TG under `beyond` would be worse than either being wrong
alone.

`TestCellLabelIsMasked` is the one that caught a real bug. `score_method`
labels any row carrying a coordinate, and a FALLBACK row carries the
Shortest-Ping VP's -- so the raw column credits a method with the baseline's
answers exactly where it gave up. On as01 `vanilla_cbg` that is 270 `correct`
against `accuracy.csv`'s 163.

`TestVerdictGrids` pins the drawn neighbourhood against `grid_offset` -- the
set form and the pair form of one BFS, which is the cross-check that makes
keeping both implementations safe.

`TestWindingIsClockwise` pins the shoelace on every ring the payload emits,
grids and cells both. A counter-clockwise filled ring makes Plotly fill the
antipodal complement; it is invisible until something is filled and then
flattens the whole map. It matters more for a cell (up to 59 degrees across)
than for a grid (51 km).

`TestRetiredKeysAreGone` walks the serialized payload for v4 vocabulary. The
failure mode of this port is not a crash -- it is a page that convincingly
labels a HEALPix grid with the word v5 uses for a serving region.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import cells as CL
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import edges as E
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import map_mtl as M
from scripts.analysis.v5.modules.answer_space import build_answer_space
from scripts.analysis.v5.modules.paths import MissingArtifactError, resolve_run

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8998, -97.0403)
ARCTIC = (63.74, -97.47)  # the tg-e1a1545 case, which must score `beyond`

_TGS = (CHI, SJC, NYC, DFW)
NSIDE = 128
METHOD = "test_cbg"
RUN_ID = "test-run"


def _space(coords=_TGS, nside=NSIDE):
    return build_answer_space(
        pd.DataFrame(
            {
                "tg_id": [f"tg-{i}" for i in range(len(coords))],
                "tg_lat": [c[0] for c in coords],
                "tg_lon": [c[1] for c in coords],
            }
        ),
        nside=nside,
        run_id=RUN_ID,
    )


class _Run:
    """The handful of `RunPaths` attributes `build_payload` actually reads."""

    run_id = RUN_ID
    source = "generic_csv"
    setup = "anchors_to_probes"


def _grid_centre(grid: int, nside: int = NSIDE) -> tuple[float, float]:
    lat, lon = G.pix2ang(np.array([grid]), nside)[0]
    return float(lat), float(lon)


def _preds_at_rings(space, rings: list[int]) -> list[tuple[float, float] | None]:
    """One prediction per TG, planted `rings[i]` grids from its own grid.

    `None` means no coordinate at all. A ring index above `MAX_RING` is planted
    in the Arctic, which is the case the metric exists to catch.
    """
    out = []
    for grid, k in zip(space.tgs["tg_grid_id"], rings):
        if k is None:
            out.append(None)
        elif k > C.MAX_RING:
            out.append(ARCTIC)
        else:
            nbhd = G.ring_grids(int(grid), NSIDE, C.MAX_RING)
            out.append(_grid_centre(nbhd[k][0]))
    return out


def _scored(space, *, rings=None, statuses=None):
    """A frame shaped exactly as `classify.score_method` returns one."""
    tgs = space.tgs
    n = len(tgs)
    rings = rings if rings is not None else [0] * n
    preds = _preds_at_rings(space, rings)
    frame = pd.DataFrame(
        {
            "tg_id": tgs["tg_id"].to_numpy(),
            "tg_lat": tgs["tg_lat"].to_numpy(),
            "tg_lon": tgs["tg_lon"].to_numpy(),
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
    and so is latent for every TG."""
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
    rows = []
    for tid in space.tgs["tg_id"]:
        for vp, rtt in (("vp-0", 5.0), ("vp-1", 22.5)):
            rows.append({"tg_id": tid, "vp_id": vp, "rtt_ms": rtt})
    return pd.DataFrame(rows)


def _context(space):
    n = len(space.tgs)
    return pd.DataFrame(
        {
            "tg_id": space.tgs["tg_id"].to_numpy(),
            "sping_vp_id": ["vp-0"] * n,
            "min_inflation": [1.5] * n,
        }
    )


def _cells(space, extent=M.CELL_FRAME):
    polys, frame = M.fit_extent(space.seeds, extent)
    polys = M.simplify_cells(polys)
    return M.cell_rings(polys), frame, CL.agreement_with_nearest_seed(space.seeds, polys, frame)


def _payload(space, scored, **kw):
    cells, frame, agreement = _cells(space)
    kw.setdefault("edges", _edges(space))
    kw.setdefault("vps", _vps())
    kw.setdefault("context", _context(space))
    kw.setdefault("cells", cells)
    kw.setdefault("cell_frame", frame)
    kw.setdefault("cell_agreement", agreement)
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


def _real_run(run_id: str = "as01-260728-260802-mesh"):
    """A real benchmark run, or a skip. The map reads parquets and vps.csv, so
    the paths-and-refusals tests need a real tree rather than a stub."""
    try:
        run = resolve_run(run_id)
    except MissingArtifactError as exc:
        pytest.skip(f"{run_id} not available: {exc}")
    if not run.combo_ids:
        pytest.skip(f"{run_id} has no scored combo")
    return run


def _shoelace(ring) -> float:
    a = np.asarray(ring, dtype=float)
    return float(np.sum(a[:, 1] * np.roll(a[:, 0], -1) - np.roll(a[:, 1], -1) * a[:, 0]))


class TestStatusIsTheSamePartitionAsTheBars:
    """The viewer's two verdicts against `summarize`'s counts, bucket for bucket."""

    def test_the_five_grid_statuses_match_the_outcome_counts(self):
        space = _space()
        scored = _scored(space, rings=[0, 1, 2, 9])
        payload = _payload(space, scored)
        summary = C.summarize({METHOD: scored}, NSIDE).iloc[0]
        drawn = {s: 0 for s in M.STATUSES}
        for t in payload["tgs"]:
            drawn[t["status"]] += 1
        for name in M.STATUSES:
            key = "n_failed" if name == "failed" else f"n_{name}"
            assert drawn[name] == int(summary[key]), name

    def test_the_three_cell_labels_match_the_cell_counts(self):
        space = _space()
        scored = _scored(space, rings=[0, 1, 2, 9])
        payload = _payload(space, scored)
        summary = C.summarize({METHOD: scored}, NSIDE).iloc[0]
        drawn = {lab: 0 for lab in C.CELL_LABELS}
        for t in payload["tgs"]:
            drawn[t["cell_label"]] += 1
        for lab in C.CELL_LABELS:
            assert drawn[lab] == int(summary[f"n_cell_{lab}"]), lab

    def test_both_partitions_close_on_the_same_denominator(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        n = len(payload["tgs"])
        assert sum(1 for t in payload["tgs"] if t["status"] in M.STATUSES) == n
        assert sum(1 for t in payload["tgs"] if t["cell_label"] in C.CELL_LABELS) == n

    def test_a_fallback_row_is_failed_not_ring0(self):
        """Its coordinate is the Shortest-Ping VP's, so it can sit in the TG's
        own grid. The METHOD still declined to answer."""
        space = _space()
        scored = _scored(space, rings=[0, 0, 0, 0], statuses=["FALLBACK"] * 4)
        payload = _payload(space, scored)
        assert {t["status"] for t in payload["tgs"]} == {"failed"}

    def test_the_baseline_is_not_scored_as_universally_failed(self):
        """An all-`BASELINE` frame is wholly solved; an inline
        `status == "SUCCESS"` would score the control as 100% failed."""
        space = _space()
        scored = _scored(space, rings=[0, 1, 2, 9], statuses=["BASELINE"] * 4)
        payload = _payload(space, scored)
        assert not any(t["status"] == "failed" for t in payload["tgs"])


class TestCellLabelIsMasked:
    """`score_method`'s raw label vs the one `summarize` counts.

    A FALLBACK row carries the Shortest-Ping VP's coordinate, so `score_method`
    labels it `correct`/`wrong`. `summarize` ANDs with `answered`, so it lands
    in `n_cell_unanswered`. The map must apply the same mask or it credits a
    method's cell accuracy with the baseline's answers.
    """

    def test_a_fallback_is_unanswered_on_the_cell_axis(self):
        space = _space()
        scored = _scored(space, rings=[0, 0, 0, 0], statuses=["FALLBACK"] * 4)
        payload = _payload(space, scored)
        assert {t["cell_label"] for t in payload["tgs"]} == {C.UNANSWERED}

    def test_the_unmasked_label_is_kept_but_never_counted(self):
        space = _space()
        scored = _scored(space, rings=[0, 0, 0, 0], statuses=["FALLBACK"] * 4)
        payload = _payload(space, scored)
        # The fallback coordinates were planted in the TGs' own grids, so the
        # raw label really is `correct` -- which is exactly why the mask has
        # to exist rather than the raw label being harmless.
        assert {t["cell_label_unmasked"] for t in payload["tgs"]} == {"correct"}

    def test_unanswered_equals_failed_by_construction(self):
        space = _space()
        scored = _scored(space, rings=[0, 1, 2, 9], statuses=["SUCCESS", "FALLBACK", "ERROR", "SUCCESS"])
        payload = _payload(space, scored)
        n_failed = sum(1 for t in payload["tgs"] if t["status"] == "failed")
        n_unans = sum(1 for t in payload["tgs"] if t["cell_label"] == C.UNANSWERED)
        assert n_failed == n_unans == 2

    def test_cell_label_agrees_with_the_seed_ids_on_answered_rows(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for t in payload["tgs"]:
            if t["cell_label"] == C.UNANSWERED:
                continue
            expected = "correct" if t["tg_seed_id"] == t["pred_seed_id"] else "wrong"
            assert t["cell_label"] == expected, t["tg_id"]


class TestGridOffsetIsCarriedExactly:
    """The uncapped offset, which is what makes `beyond` readable."""

    def test_the_offset_matches_grid_offset(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for t in payload["tgs"]:
            if t["pred_grid"] < 0:
                continue
            expected = int(G.grid_offset([t["tg_grid"]], [t["pred_grid"]], NSIDE)[0])
            assert t["grid_offset"] == expected, t["tg_id"]

    def test_beyond_carries_a_real_number_not_a_sentinel(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        beyond = [t for t in payload["tgs"] if t["status"] == "beyond"]
        assert beyond
        for t in beyond:
            assert t["grid_offset"] > C.MAX_RING

    def test_no_prediction_is_minus_one(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, None, 9]))
        missing = [t for t in payload["tgs"] if t["pred"] is None]
        assert missing
        assert all(t["grid_offset"] == -1 for t in missing)

    def test_banding_the_offset_reproduces_the_status(self):
        space = _space()
        scored = _scored(space, rings=[0, 1, 2, 9])
        payload = _payload(space, scored)
        for t in payload["tgs"]:
            band = M.status_of(t["status"] != "failed", t["grid_offset"])
            assert band == t["status"], t["tg_id"]


class TestVerdictGrids:
    """The drawn neighbourhood against the pair-form BFS."""

    def test_every_grid_at_ring_k_is_offset_k(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for t in payload["tgs"]:
            for k, shell in enumerate(t["ring_grids"]):
                for g in shell:
                    assert int(G.grid_offset([t["tg_grid"]], [g], NSIDE)[0]) == k

    def test_ring_zero_is_the_tgs_own_grid(self):
        space = _space()
        payload = _payload(space, _scored(space))
        for t in payload["tgs"]:
            assert t["ring_grids"][0] == [t["tg_grid"]]

    def test_the_rings_are_disjoint(self):
        space = _space()
        payload = _payload(space, _scored(space))
        for t in payload["tgs"]:
            flat = [g for shell in t["ring_grids"] for g in shell]
            assert len(flat) == len(set(flat))

    def test_the_neighbourhood_stops_at_max_ring(self):
        """Never grown to the offset. The disk at offset 68 is ~15,000 grids."""
        space = _space()
        payload = _payload(space, _scored(space, rings=[9, 9, 9, 9]))
        for t in payload["tgs"]:
            assert len(t["ring_grids"]) == C.MAX_RING + 1


class TestGridTable:
    """`grids` is shared, and every id any layer draws is in it."""

    def test_every_drawn_grid_has_geometry(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for t in payload["tgs"]:
            for shell in t["ring_grids"]:
                for g in shell:
                    assert str(g) in payload["grids"]
            if t["pred_grid"] >= 0:
                assert str(t["pred_grid"]) in payload["grids"]

    def test_the_occupied_grids_are_listed_and_drawn(self):
        space = _space()
        payload = _payload(space, _scored(space))
        assert sorted(payload["tg_grids"]) == sorted(int(g) for g in space.grids["grid_id"])
        for g in payload["tg_grids"]:
            assert str(g) in payload["grids"]

    def test_geometry_is_not_repeated_per_tg(self):
        """Two TGs in one grid must share the ring, not carry a copy each."""
        space = _space(coords=(CHI, CHI, NYC, DFW))
        payload = _payload(space, _scored(space))
        assert len({t["tg_grid"] for t in payload["tgs"]}) < len(payload["tgs"])


class TestCellsAreAVerdictNotContext:
    """v4's `TestVoronoiIsContextNotTheVerdict`, inverted.

    In v4 the Voronoi overlay was decoration and the test asserted nothing was
    scored on it. In v5 it IS the cell axis, so the assertions are the
    opposite: every seed has a cell, the cells are keyed so the viewer can name
    them, and the drawn polygon agrees with the label.
    """

    def test_every_seed_gets_a_cell(self):
        space = _space()
        payload = _payload(space, _scored(space))
        assert set(payload["cells"]) == {str(s) for s in space.seeds["seed_id"]}
        assert len(payload["cells"]) == payload["n_seeds"]

    def test_the_cells_are_keyed_by_seed_id_not_anonymous(self):
        """v4 shipped a flat list. Without the key the viewer cannot draw the
        TG's own cell against the one the prediction fell in, which is the
        whole of `cell_label`."""
        space = _space()
        payload = _payload(space, _scored(space))
        assert isinstance(payload["cells"], dict)
        for t in payload["tgs"]:
            assert str(t["tg_seed_id"]) in payload["cells"]

    def test_a_correct_prediction_is_inside_the_drawn_tg_cell(self):
        import shapely

        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for t in payload["tgs"]:
            if t["cell_label"] != "correct" or not t["pred"]:
                continue
            ring = payload["cells"][str(t["tg_seed_id"])]
            poly = shapely.Polygon([(lon, lat) for lat, lon in ring])
            assert poly.contains(shapely.Point(t["pred"][1], t["pred"][0])), t["tg_id"]

    def test_the_cells_are_single_polygons_with_no_holes(self):
        polys, _frame = M.fit_extent(_space().seeds, M.CELL_FRAME)
        for geom in M.simplify_cells(polys).values():
            assert geom.geom_type == "Polygon"
            assert len(geom.interiors) == 0

    def test_a_multipolygon_is_refused_rather_than_walked(self):
        import shapely

        bad = {0: shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)])}
        with pytest.raises(AssertionError, match="not a Polygon"):
            M.simplify_cells(bad)

    def test_a_hole_is_refused(self):
        import shapely

        donut = shapely.Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [[(3, 3), (3, 6), (6, 6), (6, 3)]],
        )
        with pytest.raises(AssertionError, match="holes"):
            M.simplify_cells({0: donut})

    def test_the_page_states_how_faithfully_it_draws_the_rule(self):
        space = _space()
        payload = _payload(space, _scored(space))
        meta = payload["cell_meta"]
        assert 0.9 <= meta["agreement"] <= 1.0
        assert meta["projected_crs"] == "EPSG:5070"
        assert meta["simplify_deg"] == M.SIMPLIFY_DEG
        assert "RENDERING bound" in meta["note"]

    def test_the_sites_are_carried_with_their_seed(self):
        """Every site rides with the seed it was grouped into, so the viewer
        can show why a cross sits between two dots."""
        space = _space()
        payload = _payload(space, _scored(space))
        assert len(payload["sites"]) == len(space.sites)
        seed_ids = {s["id"] for s in payload["seeds"]}
        for lat, lon, seed_id, n_scored in payload["sites"]:
            assert seed_id in seed_ids
            assert -90 <= lat <= 90 and -180 <= lon <= 180
            # Built from its own TGs, so every site is scored in full.
            assert n_scored > 0

    def test_the_frame_is_carried_and_drawn(self):
        space = _space()
        payload = _payload(space, _scored(space))
        assert payload["cell_frame"] == dict(
            zip(("lon_min", "lon_max", "lat_min", "lat_max"), M.CELL_FRAME)
        )
        ring = payload["cell_frame_ring"]
        assert ring[0] == ring[-1] and len(ring) > 4

    def test_simplification_keeps_the_table_small(self):
        space = _space()
        payload = _payload(space, _scored(space))
        size = len(json.dumps(payload["cells"], separators=(",", ":")).encode())
        assert size < 64 * 1024, f"{size/1024:.1f} KiB"


class TestFitExtent:
    """The shrink-to-fit backstop around EPSG:5070's usable domain."""

    def test_the_default_frame_needs_no_shrinking(self):
        _polys, frame = M.fit_extent(_space().seeds, M.CELL_FRAME)
        assert frame == M.CELL_FRAME

    def test_an_oversized_request_converges_to_a_smaller_frame(self):
        seeds = _space().seeds
        huge = (-180.0, -20.0, -10.0, 85.0)
        polys, frame = M.fit_extent(seeds, huge)
        assert frame != huge
        assert len(polys) == len(seeds)
        # Strictly inside the request, and still containing every seed.
        assert frame[0] > huge[0] and frame[3] < huge[3]
        assert frame[0] <= seeds["seed_lon"].min() and frame[1] >= seeds["seed_lon"].max()

    def test_shrink_extent_pulls_toward_the_centre(self):
        out = M.shrink_extent((-10.0, 10.0, -10.0, 10.0), (0.0, 0.0), 0.5)
        assert out == (-5.0, 5.0, -5.0, 5.0)


class TestWindingIsClockwise:
    """Grids and cells both. A CCW filled ring fills the antipodal complement."""

    def test_cw_ring_flips_a_counter_clockwise_input(self):
        ccw = M._cw_ring([0, 1, 1, 0], [0, 0, 1, 1])
        assert _shoelace(ccw[:-1]) <= 0

    def test_cw_ring_closes_the_ring(self):
        ring = M._cw_ring([0, 1, 1, 0], [0, 0, 1, 1])
        assert ring[0] == ring[-1]

    def test_every_payload_grid_is_clockwise(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        for gid, ring in payload["grids"].items():
            assert ring[0] == ring[-1], gid
            assert _shoelace(ring[:-1]) <= 0, gid

    def test_every_payload_cell_is_clockwise(self):
        """The one that matters most: a cell spans up to 59 degrees."""
        space = _space()
        payload = _payload(space, _scored(space))
        for sid, ring in payload["cells"].items():
            assert ring[0] == ring[-1], sid
            assert _shoelace(ring[:-1]) <= 0, sid

    def test_the_cell_frame_ring_is_clockwise(self):
        space = _space()
        payload = _payload(space, _scored(space))
        ring = payload["cell_frame_ring"]
        assert _shoelace(ring[:-1]) <= 0


class TestRetiredKeysAreGone:
    """v4's vocabulary must not survive into a v5 payload.

    `cell` and `seed` both changed meaning between the packages, so a leftover
    `tg_cell` does not crash -- it labels a HEALPix grid with the word v5 uses
    for a serving region, and a reader has no way to spot that.
    """

    RETIRED = {
        # v4 grid-as-"cell" names
        "tg_cell", "pred_cell", "ring_cells", "cell_id", "cell_km", "cell_offset_km",
        "region_is_pred_cell",
        # v4/v3 benchmark names
        "target_id", "error_km", "targets",
        # v3's retired nearest-seed machinery
        "top_seeds", "margin_km", "seeds_crossed", "proximity", "voronoi_rank", "voronoi",
    }

    def test_no_retired_key_at_any_depth(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, 2, 9]))
        found = self.RETIRED & set(_walk(payload))
        assert not found, f"retired keys survive: {sorted(found)}"

    def test_a_seed_carries_no_grid_id(self):
        """v4's `seeds[].cell_id` only worked because a seed WAS a grid centre.
        A v5 seed is a complete-linkage centroid of sites and has no grid."""
        space = _space()
        payload = _payload(space, _scored(space))
        for s in payload["seeds"]:
            assert "cell_id" not in s and "grid_id" not in s
            assert {"id", "lat", "lon", "n_sites", "n_tgs", "n_tgs_scored"} == set(s)

    def test_the_templates_carry_no_retired_wording(self):
        for name in ("mtl_map.html", "mtl_map.js"):
            text = (M._TEMPLATE_DIR / name).read_text(encoding="utf-8")
            for phrase in ("truth cell", "class cells", "showVoronoi", "voronoiCells"):
                assert phrase not in text, f"{name} still says {phrase!r}"


class TestPayloadIsStrictJson:
    def test_it_serializes_without_nan(self):
        space = _space()
        payload = _payload(space, _scored(space, rings=[0, 1, None, 9]))
        json.dumps(payload, allow_nan=False)

    def test_the_rendered_page_parses_its_own_blob(self):
        space = _space()
        html = M.render_html(_payload(space, _scored(space, rings=[0, 1, 2, 9])))
        blob = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
        assert json.loads(blob.replace("<\\/", "</"))["n_seeds"] == 4

    def test_a_script_close_cannot_escape_the_data_block(self):
        space = _space()
        payload = _payload(space, _scored(space))
        payload["method"] = "</script><script>alert(1)</script>"
        html = M.render_html(payload)
        body = html.split('<script id="data" type="application/json">')[1]
        assert "</script><script>alert(1)" not in body.split("</script>")[0]

    def test_every_token_is_substituted(self):
        space = _space()
        html = M.render_html(_payload(space, _scored(space)))
        for token in ("__PAYLOAD__", "__SCRIPT__", "__TITLE__"):
            assert token not in html


class TestObservationLayer:
    def test_measured_vps_carry_rtt_and_inflation(self):
        space = _space()
        payload = _payload(space, _scored(space))
        for t in payload["tgs"]:
            assert t["n_measured"] == 2
            assert t["n_total"] == 3
            for vp_id, rtt, infl in t["obs"]:
                assert vp_id in payload["vps"]
                assert rtt > 0 and (infl is None or infl > 0)

    def test_an_absent_edge_table_drops_only_that_layer(self):
        space = _space()
        payload = _payload(
            space, _scored(space), edges=pd.DataFrame(columns=list(E.MIN_RTT_COLUMNS))
        )
        assert all(t["obs"] == [] for t in payload["tgs"])
        # The verdicts are unaffected: they come from the parquets and the space.
        assert {t["status"] for t in payload["tgs"]} == {"ring0"}

    def test_context_is_optional(self):
        space = _space()
        payload = _payload(
            space, _scored(space),
            context=pd.DataFrame(columns=["tg_id", "sping_vp_id", "min_inflation"]),
        )
        assert all(t["sping_vp_id"] is None for t in payload["tgs"])


class TestTheAnswerSpaceIsLoadedNotRebuilt:
    """v5's convention, and the reason the map cannot draw the wrong seeds.

    v4 rebuilt the answer space in-process and advertised running on a bare
    benchmark run. A v5 seed is a complete-linkage cluster centroid, so
    rebuilding risks drawing cells that are not the cells `classify` scored.
    """

    def test_a_missing_rung_names_the_command(self, tmp_path):
        run = _real_run()
        with pytest.raises(MissingArtifactError, match="build-answer-space"):
            M.load_rung(run, NSIDE, analysis_root=tmp_path)

    def test_build_for_run_refuses_before_build_answer_space(self, tmp_path):
        run = _real_run()
        with pytest.raises(MissingArtifactError, match="build-answer-space"):
            M.build_for_run(run, methods=[run.combo_ids[0]], analysis_root=tmp_path, regions=False)


class TestRegionCache:
    def test_a_region_round_trips(self, tmp_path):
        region = {"kind": "polygon", "rings": [{"outer": [[1.0, 2.0]], "holes": []}]}
        M._write_region_cache(tmp_path, METHOD, "tg-0", region)
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") == region

    def test_an_empty_result_is_memoized_not_recomputed(self):
        """`{}` means "this TG has no region", and must stay distinguishable
        from `None`, which means "not cached"."""
        import tempfile

        d = Path(tempfile.mkdtemp())
        M._write_region_cache(d, METHOD, "tg-0", None)
        assert M._read_region_cache(d, METHOD, "tg-0") == {}
        assert M._read_region_cache(d, METHOD, "tg-missing") is None

    def test_a_changed_mtl_spec_drops_the_cache(self, tmp_path):
        old = {0: ("octant", '{"a": 1}')}
        new = {0: ("octant", '{"a": 2}')}
        M._invalidate_stale_cache(tmp_path, METHOD, old)
        M._write_region_cache(tmp_path, METHOD, "tg-0", {"kind": "polygon", "rings": []})
        M._invalidate_stale_cache(tmp_path, METHOD, new)
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") is None

    def test_an_unchanged_spec_keeps_the_cache(self, tmp_path):
        spec = {0: ("octant", '{"a": 1}')}
        M._invalidate_stale_cache(tmp_path, METHOD, spec)
        M._write_region_cache(tmp_path, METHOD, "tg-0", {"kind": "polygon", "rings": []})
        M._invalidate_stale_cache(tmp_path, METHOD, spec)
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") is not None

    def test_a_cache_with_no_sidecar_is_adopted(self, tmp_path):
        """Caches written before the guard existed must survive."""
        M._write_region_cache(tmp_path, METHOD, "tg-0", {"kind": "polygon", "rings": []})
        M._invalidate_stale_cache(tmp_path, METHOD, {0: ("octant", "{}")})
        assert M._read_region_cache(tmp_path, METHOD, "tg-0") is not None


class TestCacheIsRungFree:
    """No nside enters `replay_mtl`, so a second rung must not re-pay it."""

    def test_the_cache_dir_carries_no_rung(self, tmp_path):
        run = _real_run()
        d = run.mtl_region_cache_dir(root=tmp_path)
        assert "healpix" not in str(d)
        assert d.name == "regions"

    def test_the_map_dir_does_carry_one(self, tmp_path):
        run = _real_run()
        assert "healpix-128" in str(run.mtl_map_dir(128, root=tmp_path))
        assert "healpix-64" in str(run.mtl_map_dir(64, root=tmp_path))


class _Sub:
    def __init__(self, cutoff_rtt, fitted=True):
        self.cutoff_rtt, self.fitted = cutoff_rtt, fitted


class _PerVp:
    def __init__(self, subs):
        self._submodels = subs


class _Pooled:
    def __init__(self, cutoff_rtt):
        self._model = _Sub(cutoff_rtt)


def _scored_with_constraints(space):
    """`_scored` plus the nested columns, both VPs constraining every TG.
    `_edges` gives vp-0 5.0 ms and vp-1 22.5 ms."""
    scored = _scored(space)
    n = len(scored)
    scored["ltd_predictions"] = [
        [
            {"vp_id": "vp-0", "success": True, "upper_km": 400.0, "lower_km": 0.0},
            {"vp_id": "vp-1", "success": True, "upper_km": 2500.0, "lower_km": 100.0},
        ]
    ] * n
    scored["mtl_participants"] = [[{"vp_id": "vp-0"}, {"vp_id": "vp-1"}]] * n
    return scored


class TestLtdCutoff:
    """The cutoff is fitted state read off the checkpoint, and a constraint is
    past it on the models' own strict `rtt > cutoff_rtt` gate."""

    def test_per_vp_cutoffs_skip_unset_and_unfitted(self):
        model = _PerVp({"vp-0": _Sub(10.0), "vp-1": _Sub(0.0), "vp-2": _Sub(9.0, fitted=False)})
        assert M.ltd_cutoffs(model) == {"vp-0": 10.0}

    def test_pooled_cutoff_is_one_number(self):
        assert M.ltd_cutoffs(_Pooled(90.5)) == 90.5

    def test_unset_or_stateless_is_no_cutoff(self):
        assert M.ltd_cutoffs(_Pooled(0.0)) is None
        assert M.ltd_cutoffs(_PerVp({"vp-0": _Sub(0.0)})) is None
        assert M.ltd_cutoffs(object()) is None

    def test_rows_carry_the_cutoff_and_the_past_flag(self):
        space = _space()
        # vp-0 at 5.0 ms is inside its 10 ms cutoff; vp-1 at 22.5 ms is past 20.
        payload = _payload(
            space, _scored_with_constraints(space), cutoffs={0: {"vp-0": 10.0, "vp-1": 20.0}}
        )
        for t in payload["tgs"]:
            by_vp = {r[0]: r for r in t["rings"]}
            assert by_vp["vp-0"][4:] == [10.0, 0]
            assert by_vp["vp-1"][4:] == [20.0, 1]
        lc = payload["ltd_cutoff"]
        assert lc["scope"] == "per_vp"
        assert lc["n_past"] == lc["n_past_kept"] == len(space.tgs)

    def test_the_gate_is_strict(self):
        space = _space()
        payload = _payload(space, _scored_with_constraints(space), cutoffs={0: 22.5})
        assert all(r[5] == 0 for t in payload["tgs"] for r in t["rings"])
        assert payload["ltd_cutoff"]["scope"] == "pooled"
        assert payload["ltd_cutoff"]["pooled_ms_by_fold"] == {"0": 22.5}

    def test_no_cutoffs_means_no_flag(self):
        space = _space()
        payload = _payload(space, _scored_with_constraints(space))
        assert payload["ltd_cutoff"] is None
        assert all(r[4:] == [None, 0] for t in payload["tgs"] for r in t["rings"])

    def test_real_checkpoints(self):
        run = _real_run()
        octant = M.load_cutoffs(run, "octant_cbg_hull")
        spotter = M.load_cutoffs(run, "spotter_cbg")
        assert octant and all(isinstance(c, dict) and c for c in octant.values())
        assert spotter and all(isinstance(c, float) for c in spotter.values())
        # `speed_of_internet` is stateless: no checkpoint state, no cutoff.
        assert M.load_cutoffs(run, "million_scale_cbg") == {}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """One real run rendered into a temp root, regions skipped.

    Module-scoped: `build_for_run` reads every fold's parquet and the canonical
    edge CSV, which is seconds rather than milliseconds. `--no-regions` is the
    equivalent of the CLI flag -- the replay is ~7 s per TG and is not what any
    of these assert on.
    """
    run = _real_run()
    root = tmp_path_factory.mktemp("v5map")
    from scripts.analysis.v5.modules import answer_space as A

    A.build_for_run(run, analysis_root=root)
    C.score_for_run(run, analysis_root=root)
    out = M.build_for_run(
        run,
        methods=[*run.combo_ids, M.SHORTEST_PING],
        analysis_root=root,
        regions=False,
    )
    return run, root, out


class TestOnRealRuns:
    def test_both_axes_match_accuracy_csv(self, rendered):
        run, root, out = rendered
        acc = pd.read_csv(run.classify_dir(NSIDE, root=root) / C.ACCURACY_CSV).set_index("method")
        for method, _path, payload in out:
            row = acc.loc[method]
            grid = {s: 0 for s in M.STATUSES}
            cell = {lab: 0 for lab in C.CELL_LABELS}
            for t in payload["tgs"]:
                grid[t["status"]] += 1
                cell[t["cell_label"]] += 1
            for name in M.STATUSES:
                key = "n_failed" if name == "failed" else f"n_{name}"
                assert grid[name] == int(row[key]), f"{method} {name}"
            for lab in C.CELL_LABELS:
                assert cell[lab] == int(row[f"n_cell_{lab}"]), f"{method} cell {lab}"

    def test_the_offset_percentiles_match_accuracy_csv(self, rendered):
        run, root, out = rendered
        acc = pd.read_csv(run.classify_dir(NSIDE, root=root) / C.ACCURACY_CSV).set_index("method")
        for method, _path, payload in out:
            offs = [
                t["grid_offset"]
                for t in payload["tgs"]
                if t["status"] != "failed" and t["grid_offset"] >= 0
            ]
            row = acc.loc[method]
            assert round(float(np.quantile(offs, 0.50)), 1) == row[f"{C.GRID_OFFSET}_p50"]
            assert round(float(np.quantile(offs, 0.90)), 1) == row[f"{C.GRID_OFFSET}_p90"]
            assert max(offs) == int(row[f"{C.GRID_OFFSET}_max"])

    def test_beyond_is_a_spread_not_a_bucket(self, rendered):
        """The whole reason the map reads `pred_dist_to_tg_grid`: `ring` pooled
        all of this into one `-1`."""
        _run, _root, out = rendered
        for method, _path, payload in out:
            offs = sorted(t["grid_offset"] for t in payload["tgs"] if t["status"] == "beyond")
            if not offs:
                continue
            assert offs[0] > C.MAX_RING
            assert offs[-1] > offs[0], f"{method} has a flat beyond bucket"

    def test_the_two_axes_are_not_the_same_partition(self, rendered):
        """If every `correct` were also `ring0`, the cell axis would carry no
        information and v5 would be a relabelling of v4."""
        _run, _root, out = rendered
        off_diagonal = 0
        for _method, _path, payload in out:
            for t in payload["tgs"]:
                if t["status"] == "beyond" and t["cell_label"] == "correct":
                    off_diagonal += 1
        assert off_diagonal > 0

    def test_every_page_is_self_contained(self, rendered):
        _run, _root, out = rendered
        for method, path, _payload in out:
            text = path.read_text(encoding="utf-8")
            assert path.stat().st_size > 100_000, method
            # Plotly from CDN is the one external reference; nothing else may
            # be fetched, or the page stops opening over `file://`.
            assert text.count("<script src=") == 1
            assert "cdn.plot.ly" in text
            assert not list(path.parent.glob("*.json")), "a sibling JSON was written"

    def test_all_pages_of_a_run_share_one_cell_frame(self, rendered):
        """The cells belong to the answer space, not to the method."""
        _run, _root, out = rendered
        frames = {json.dumps(p["cell_frame"], sort_keys=True) for _m, _p, p in out}
        cells = {json.dumps(p["cells"], sort_keys=True) for _m, _p, p in out}
        assert len(frames) == 1 and len(cells) == 1

    def test_the_shortest_ping_control_renders(self, rendered):
        _run, _root, out = rendered
        payload = next(p for m, _p, p in out if m == M.SHORTEST_PING)
        assert payload["is_baseline"] is True
        assert all(t["rings"] == [] for t in payload["tgs"])
        assert all(t["region"] is None for t in payload["tgs"])


_HARNESS = Path(__file__).resolve().parent / "test_map_mtl_viewer.js"


def _run_viewer(html_path: Path) -> dict:
    """Drive the real viewer JS under node, or skip.

    Python cannot check the viewer: a typo in `draw()` renders a blank page and
    every payload assertion still passes. The harness stubs the DOM and Plotly,
    evals the IIFE, and sweeps every TG, projection, toggle and filter.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    proc = subprocess.run(
        [node, str(_HARNESS), str(html_path), str(M._JS_TEMPLATE_PATH)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"viewer harness failed:\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def report(rendered):
    """The harness run once, for the whole class: it sweeps 399 TGs x every
    control, which is seconds."""
    _run, _root, out = rendered
    path = next(p for m, p, _pl in out if m == "octant_cbg_hull")
    return _run_viewer(path)


class TestTheViewerExecutes:
    def test_it_drew_every_tg(self, report):
        assert report["targets"] > 100
        assert report["draws"] >= report["targets"]

    def test_the_cell_layers_are_drawn(self, report):
        layers = " | ".join(report["allLayers"])
        assert "serving cells" in layers
        assert "the TG's cell (seed #" in layers
        assert "landed in seed #" in layers, "no wrong-cell highlight was ever drawn"
        assert "cell frame (rendering bound" in layers

    def test_the_past_cutoff_layer_is_drawn(self, report):
        """as01's Octant folds carry a handful of kept constraints past their
        VP's cutoff, so the layer must appear on at least one TG."""
        assert any(n.startswith("past LTD cutoff (") for n in report["allLayers"])

    def test_the_grid_layers_use_grid_wording(self, report):
        layers = report["allLayers"]
        joined = " | ".join(layers)
        assert "the TG's own grid" in joined
        assert "prediction grid #" in joined
        assert any("grids out" in name for name in layers)

    def test_no_layer_calls_a_healpix_grid_a_cell(self, report):
        """The rename that matters. In v5 "cell" means a serving region, so a
        layer named `prediction cell #59612` for a HEALPix grid is wrong in a
        way no reader can catch."""
        for name in report["allLayers"]:
            assert "prediction cell #" not in name
            assert "class cell" not in name
            assert "ring 0 · the truth's cell" not in name
            assert "Voronoi" not in name

    def test_the_seeds_are_drawn(self, report):
        """`cell_label` is defined as "nearest seed", so without the seeds on
        the map the cell verdict is asserted rather than shown."""
        layers = " | ".join(report["allLayers"])
        assert "seeds (" in layers
        assert "the TG's seed #" in layers
        assert "sites (" in layers

    def test_the_drawn_tg_cell_agrees_with_cell_label(self, report):
        """Allowing the measured planar/spherical drift, which the harness
        budgets rather than asserting an exactness it cannot have."""
        assert report["cellChecked"] > 100
        assert report["cellDisagreed"] <= report["cellBudget"], report["cellDisagreements"]

    def test_the_drawn_ring0_agrees_with_the_status(self, report):
        assert report["verdictChecked"] > 100

    def test_both_verdicts_reach_the_prose(self, report):
        meta = report["meta"]
        assert "grid offset=" in meta
        assert "TG grid" in meta and "TG cell" in meta
        assert "agree with the great-circle nearest-seed rule" in meta
        pred = report["panels"]["pred"]
        assert "grid verdict" in pred and "cell verdict" in pred

    def test_the_cross_tab_has_off_diagonal_mass(self, report):
        """Both axes carry information; neither is a relabelling of the other."""
        assert report["offDiagonal"] > 0

    def test_the_cells_are_all_present_and_measured(self, report):
        assert report["nCells"] > 0
        assert 0.9 <= report["cellAgreement"] <= 1.0

    def test_hovering_a_vp_lifts_its_constraint(self, report):
        assert report["anyRings"] is True
        assert report["highlighted"] > 0

    def test_the_baseline_hides_the_controls_it_cannot_use(self, rendered):
        _run, _root, out = rendered
        path = next(p for m, p, _pl in out if m == M.SHORTEST_PING)
        report = _run_viewer(path)
        assert "showRings" in report["hiddenControls"]
        assert "failed" not in report["statusOptions"]
