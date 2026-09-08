"""Tests for the interactive MTL map.

Thin on rendering, as the other map tests are, but the *derivations* get real
assertions: this command recomputes the answer space, the top-1 verdict, the
crossing count and the proximity ladder rather than reading the CSVs that hold
them, so a drift here would be silent — the page would still render, just with
the wrong verdict on it.

No grid fixture: the command draws grid cells and is H3-only by construction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v3.modules.answer_space import build_answer_space, pairwise_km
from scripts.analysis.v3.modules.map_mtl import (
    SHORTEST_PING,
    _cw_ring,
    _nested,
    _proximity_label,
    _safe_float,
    _status_of,
    _top_seeds,
    build_payload,
    conus_boundary,
    conus_voronoi_rings,
    render_html,
    seed_rings,
)

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8998, -97.0403)
ANC = (61.1743, -149.9962)  # Anchorage — the Alaska case CONUS must exclude
HNL = (21.3187, -157.9224)  # Honolulu — likewise Hawaii

_TARGETS = (CHI, SJC, NYC, DFW)


def _space(coords=_TARGETS):
    return build_answer_space(
        pd.DataFrame(
            {
                "target_id": [f"tg-{i}" for i in range(len(coords))],
                "target_lat": [c[0] for c in coords],
                "target_lon": [c[1] for c in coords],
            }
        ),
        grid="h3",
    )


class _Run:
    """The handful of `RunPaths` attributes `build_payload` actually reads."""

    run_id = "test-run"
    source = "generic_csv"
    setup = "anchors_to_probes"


def _scores(space, *, statuses=None, preds=None):
    """A `classify`-shaped seed-distance frame for the fixture's targets."""
    assign = space.assignments
    n = len(assign)
    statuses = statuses or ["SUCCESS"] * n
    preds = preds or list(zip(assign["target_lat"], assign["target_lon"]))
    seeds = space.seeds
    dist = pairwise_km(
        np.array([p[0] for p in preds]), np.array([p[1] for p in preds]),
        seeds["seed_lat"].to_numpy(), seeds["seed_lon"].to_numpy(),
    )
    frame = pd.DataFrame(
        {
            "method": "test_cbg",
            "target_id": assign["target_id"].to_numpy(),
            "fold": 0,
            "status": statuses,
            "pred_lat": [p[0] for p in preds],
            "pred_lon": [p[1] for p in preds],
            "tg_seed_id": assign["seed_id"].to_numpy(),
            "pred_seed_id": dist.argmin(axis=1),
            "tg_seed_rank": [
                int((dist[i] < dist[i, assign["seed_id"].iloc[i]] - 1e-9).sum())
                for i in range(n)
            ],
            "error_to_target_km": 0.0,
        }
    )
    for k, sid in enumerate(seeds["seed_id"]):
        frame[f"dist_km__seed_{int(sid)}"] = dist[:, k]
    return frame


def _labels(space, **overrides):
    n = len(space.assignments)
    base = {
        "target_id": space.assignments["target_id"].to_numpy(),
        "has_proximate_vp": [True] * n,
        "has_discriminative_vp": [False] * n,
        "sping_vp_id": ["vp-0"] * n,
        "min_inflation": [1.5] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _vps():
    """Three VPs, so all three marker classes are exercised: vp-0 is the
    shortest-ping VP, vp-1 is measured-but-not-selected, vp-2 is never
    measured and so is latent for every target."""
    return pd.DataFrame(
        {
            "vp_id": ["vp-0", "vp-1", "vp-2"],
            "vp_lat": [CHI[0], DFW[0], SJC[0]],
            "vp_lon": [CHI[1], DFW[1], SJC[1]],
            "vp_asn": ["7018", "7018", "7018"],
            "vp_country": ["US", "US", "US"],
        }
    )


def _edges(space):
    """vp-0 and vp-1 measure every target; vp-2 measures none."""
    a = space.assignments
    n = len(a)
    return pd.DataFrame(
        {
            "target_id": list(a["target_id"]) * 2,
            "vp_id": ["vp-0"] * n + ["vp-1"] * n,
            "rtt_ms": list(np.linspace(5.0, 40.0, n)) + list(np.linspace(9.0, 55.0, n)),
        }
    )


def _payload(space, **kw):
    return build_payload(
        _Run(), space, "test_cbg",
        scores=kw.pop("scores", None) if "scores" in kw else _scores(space),
        labels=kw.pop("labels", None) if "labels" in kw else _labels(space),
        edges=_edges(space),
        crossing=kw.pop("crossing", None),
        rings=seed_rings(space.seeds),
        voronoi=[],
        vps=_vps(),
        **kw,
    )


# ---- verdict derivation -----------------------------------------------------


def test_status_reads_rank_only_when_the_row_was_solved():
    assert _status_of(True, 0) == "correct"
    assert _status_of(True, 1) == "wrong"
    # A fallback is a failure even when it happens to land on the right seed:
    # scoring it as a success would floor CBG's accuracy at Shortest-Ping's,
    # which is the comparison the paper's RQ2 rests on (§7.2).
    assert _status_of(False, 0) == "failed"
    assert _status_of(True, None) == "wrong"


def test_the_shortest_ping_baseline_is_not_scored_as_universally_failed():
    """The regression this stops: `status == "SUCCESS"` written inline.

    `classify.score_shortest_ping` emits an all-`BASELINE` frame, so the
    inline test is False on every row and the baseline — the denominator of
    every claim in the paper — would render as 100% failed.
    """
    from scripts.analysis.v3.modules import io

    frame = pd.DataFrame({"status": ["BASELINE"] * 3})
    solved = io.solved_mask(frame)
    assert list(solved) == [True, True, True]
    assert [_status_of(bool(s), 0) for s in solved] == ["correct"] * 3


@pytest.mark.parametrize(
    ("disc", "prox", "expected"),
    [
        (True, True, "discriminative"),
        (False, True, "proximate"),
        (False, False, "no proximity"),
        # Forbidden by `proximity.IMPLICATIONS` (discriminative => proximate).
        # Pinned anyway: the ladder must degrade to the stronger term rather
        # than fall through to "no proximity" if the invariant ever breaks.
        (True, False, "discriminative"),
    ],
)
def test_proximity_ladder(disc, prox, expected):
    row = {"has_discriminative_vp": disc, "has_proximate_vp": prox}
    assert _proximity_label(row) == expected


def test_top_seeds_is_the_prediction_distance_ordering():
    row = np.array([300.0, 100.0, 200.0, 50.0])
    assert _top_seeds(row, 3) == [3, 1, 2]


def test_top_seeds_skips_unscored_seeds_instead_of_ranking_nan_first():
    row = np.array([np.nan, 100.0, np.nan, 50.0])
    assert _top_seeds(row, 3) == [3, 1]
    assert _top_seeds(np.array([np.nan, np.nan])) == []


def test_safe_float_scrubs_what_allow_nan_false_would_reject():
    assert _safe_float(np.nan) is None
    assert _safe_float(np.inf) is None
    assert _safe_float(None) is None
    assert _safe_float("nope") is None
    assert _safe_float(np.float64(2.5)) == 2.5


def test_nested_accepts_the_numpy_object_arrays_parquet_returns():
    """`value or []` on a pyarrow list-of-struct cell raises, and only on
    non-empty rows — so it survives any fixture that uses an empty one."""
    assert _nested(None) == []
    assert _nested(np.array([{"a": 1}, {"a": 2}], dtype=object)) == [{"a": 1}, {"a": 2}]


# ---- payload ----------------------------------------------------------------


def test_payload_carries_every_layer():
    space = _space()
    pl = _payload(space)
    assert pl["grid"] == "h3" and pl["resolution"] == 4
    assert len(pl["seeds"]) == space.n_seeds
    assert len(pl["targets"]) == len(space.assignments)
    assert set(pl["vps"]) == {"vp-0", "vp-1", "vp-2"}
    for s in pl["seeds"]:
        assert s["ring"][0] == s["ring"][-1], "cell ring must be closed"
    t = pl["targets"][0]
    assert t["n_total"] == 3 and t["n_measured"] == 2, "latent VPs count in n_total only"
    assert t["proximity"] == "proximate"
    assert sorted(o[0] for o in t["obs"]) == ["vp-0", "vp-1"]


def test_margin_is_half_the_top1_to_top2_geodesic():
    space = _space()
    pl = _payload(space)
    mesh = space.seed_mesh_km.to_numpy()
    for t in pl["targets"]:
        if len(t["top_seeds"]) < 2:
            continue
        a, b = t["top_seeds"][0], t["top_seeds"][1]
        assert t["margin_km"] == pytest.approx(0.5 * mesh[a, b], abs=1e-6)


def test_per_vp_inflation_matches_the_speed_of_internet_definition():
    """RTT over the 2/3-c RTT for the same great circle — and None, never
    inf, when the VP sits on its target."""
    from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE, haversine_distance

    space = _space()
    pl = _payload(space)
    coords = {t["target_id"]: t["true"] for t in pl["targets"]}
    seen_colocated = False
    for t in pl["targets"]:
        for vp_id, rtt, infl in t["obs"]:
            vp = pl["vps"][vp_id]
            km = haversine_distance(coords[t["target_id"]][0], coords[t["target_id"]][1],
                                    vp[0], vp[1])
            if km == 0:
                # A VP sitting on its own target: inflation is undefined, not
                # infinite. `allow_nan=False` rejects an inf, so letting one
                # through here would fail the whole page at serialization.
                seen_colocated = True
                assert infl is None
                continue
            assert infl == pytest.approx(rtt / (THEORETICAL_SLOPE * km), abs=1e-3)
    assert seen_colocated, "fixture no longer exercises the zero-distance case"


def test_seeds_crossed_is_read_off_the_crossing_matrix():
    space = _space()
    crossing = np.arange(space.n_seeds ** 2).reshape(space.n_seeds, space.n_seeds)
    pl = _payload(space, crossing=crossing)
    for t in pl["targets"]:
        assert t["seeds_crossed"] == crossing[t["tg_seed_id"], t["top_seeds"][0]]


def test_seeds_crossed_is_null_rather_than_zero_when_the_matrix_is_skipped():
    """Above `MAX_SEEDS_FOR_CROSSING` the matrix is not built. Emitting 0 would
    read as "adjacent classes", which is the opposite of "not computed"."""
    pl = _payload(_space(), crossing=None)
    assert all(t["seeds_crossed"] is None for t in pl["targets"])


def test_failures_sort_first_so_the_page_opens_on_a_bad_case():
    space = _space()
    n = len(space.assignments)
    scores = _scores(space)
    scores.loc[scores.index[-1], "status"] = "FALLBACK"
    pl = build_payload(
        _Run(), space, "test_cbg", scores=scores, labels=_labels(space),
        edges=_edges(space), crossing=None, rings=seed_rings(space.seeds),
        voronoi=[], vps=_vps(),
    )
    order = [t["status"] for t in pl["targets"]]
    assert order[0] == "failed"
    assert order == sorted(order, key=lambda s: {"failed": 0, "wrong": 1, "correct": 2}[s])
    assert len(order) == n


def test_the_baseline_renders_without_constraint_or_region_layers():
    space = _space()
    pl = build_payload(
        _Run(), space, SHORTEST_PING, scores=_scores(space), labels=_labels(space),
        edges=_edges(space), crossing=None, rings=seed_rings(space.seeds),
        voronoi=[], vps=_vps(),
    )
    assert pl["mtl_kind"] == "disk"
    assert all(t["rings"] == [] and t["region"] is None for t in pl["targets"])


# ---- rendering --------------------------------------------------------------


def _embedded(html):
    blob = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html, re.S
    ).group(1)
    return json.loads(blob.replace("<\\/", "</"))


def test_render_round_trips_the_payload_through_the_data_block():
    space = _space()
    pl = _payload(space)
    html = render_html(pl)
    assert "__PAYLOAD__" not in html and "__SCRIPT__" not in html and "__TITLE__" not in html
    assert _embedded(html) == pl


def test_a_closing_script_tag_in_the_payload_cannot_end_the_data_block():
    """Escaping `</` is what keeps a hostile-looking target id from truncating
    the JSON — the browser would then parse a prefix and fail silently."""
    space = _space()
    pl = _payload(space)
    pl["targets"][0]["target_id"] = "tg-</script><b>x"
    html = render_html(pl)
    assert "</script><b>x" not in html
    assert _embedded(html)["targets"][0]["target_id"] == "tg-</script><b>x"


def test_render_is_deterministic():
    space = _space()
    pl = _payload(space)
    assert render_html(pl) == render_html(pl)


# ---- CONUS geometry ---------------------------------------------------------


def test_conus_boundary_drops_alaska_and_hawaii():
    from shapely.geometry import Point

    boundary = conus_boundary()
    assert boundary.contains(Point(CHI[1], CHI[0]))
    assert not boundary.contains(Point(ANC[1], ANC[0]))
    assert not boundary.contains(Point(HNL[1], HNL[0]))


def test_conus_voronoi_returns_lonlat_degrees_not_projected_metres():
    """The diagram is computed in an azimuthal-equidistant frame; forgetting to
    unproject yields rings in metres, which Plotly would silently clamp to the
    poles rather than reject."""
    space = _space()
    rings = conus_voronoi_rings(space.seeds)
    assert rings
    for ring in rings:
        for lat, lon in ring:
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 180.0


def test_conus_voronoi_needs_at_least_two_seeds():
    assert conus_voronoi_rings(_space(coords=(CHI,)).seeds) == []


# ---- region cache -----------------------------------------------------------


def test_a_changed_mtl_spec_drops_the_cached_regions(tmp_path):
    """The cache is keyed on (method, target), which alone is not enough.

    Re-running the benchmark for the same run_id with different `mtl_kwargs`
    would otherwise leave the map drawing the old configuration's regions next
    to the new configuration's predictions.
    """
    from scripts.analysis.v3.modules.map_mtl import (
        REGION_SPEC_JSON,
        _invalidate_stale_cache,
        _read_region_cache,
        _write_region_cache,
    )

    specs = {(0, "test_cbg"): ("planar_circle", '{"n_pts": 64}')}
    _invalidate_stale_cache(tmp_path, "test_cbg", specs)
    _write_region_cache(tmp_path, "test_cbg", "tg-0", {"kind": "polygon", "rings": []})
    assert (tmp_path / "test_cbg" / REGION_SPEC_JSON).exists()
    assert _read_region_cache(tmp_path, "test_cbg", "tg-0") is not None

    # Same spec: the cache survives.
    _invalidate_stale_cache(tmp_path, "test_cbg", specs)
    assert _read_region_cache(tmp_path, "test_cbg", "tg-0") is not None

    # Different kwargs: it does not.
    _invalidate_stale_cache(tmp_path, "test_cbg", {(0, "test_cbg"): ("planar_circle", '{"n_pts": 32}')})
    assert _read_region_cache(tmp_path, "test_cbg", "tg-0") is None


def test_a_cache_predating_the_guard_is_adopted_rather_than_discarded(tmp_path):
    from scripts.analysis.v3.modules.map_mtl import (
        _invalidate_stale_cache,
        _read_region_cache,
        _write_region_cache,
    )

    _write_region_cache(tmp_path, "test_cbg", "tg-0", {"kind": "polygon", "rings": []})
    _invalidate_stale_cache(tmp_path, "test_cbg", {(0, "test_cbg"): ("planar_circle", "{}")})
    assert _read_region_cache(tmp_path, "test_cbg", "tg-0") is not None


def test_an_empty_result_is_memoized_so_it_is_not_recomputed_forever(tmp_path):
    from scripts.analysis.v3.modules.map_mtl import _read_region_cache, _write_region_cache

    _write_region_cache(tmp_path, "test_cbg", "tg-0", None)
    cached = _read_region_cache(tmp_path, "test_cbg", "tg-0")
    assert cached == {}, "must be a hit, not a miss"
    assert cached is not None


# ---- the viewer itself ------------------------------------------------------


def test_the_viewer_executes_against_a_real_payload(tmp_path):
    """Run `templates/mtl_map.js` under node, over every control state.

    Python can only check that the payload is well-formed; a mistake in
    `draw()` produces a blank page while every payload assertion above still
    passes. The harness stubs Plotly and enough DOM to execute the IIFE, then
    drives every target, projection, layer toggle, status filter and clickable
    trace — and rejects NaN coordinates, which Plotly would otherwise drop
    silently, making a ring simply vanish.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    space = _space()
    scores = _scores(space)
    scores.loc[scores.index[-1], "status"] = "FALLBACK"  # exercise the failed branch
    pl = build_payload(
        _Run(), space, "test_cbg", scores=scores, labels=_labels(space),
        edges=_edges(space), crossing=np.zeros((space.n_seeds, space.n_seeds), dtype=int),
        rings=seed_rings(space.seeds), voronoi=conus_voronoi_rings(space.seeds),
        vps=_vps(),
        # Every target, because the payload sorts failures first — pinning the
        # region to assignment 0 would leave it off the target the page opens on.
        regions={
            tid: {
                "kind": "polygon",
                "rings": [{"outer": [[41.0, -88.0], [42.0, -88.0], [42.0, -87.0], [41.0, -88.0]],
                           "holes": []}],
            }
            for tid in space.assignments["target_id"]
        },
        # vp-0's constraint is kept, vp-1's is dropped by the inclusion filter,
        # so both ring styles and both popup branches are exercised.
        folds=pd.DataFrame(
            {
                "target_id": space.assignments["target_id"].to_numpy(),
                "ltd_predictions": [
                    [
                        {"vp_id": "vp-0", "success": True, "upper_km": 900.0, "lower_km": 100.0},
                        {"vp_id": "vp-1", "success": True, "upper_km": 3000.0, "lower_km": 0.0},
                    ]
                ] * len(space.assignments),
                "mtl_participants": [
                    [{"vp_id": "vp-0"}]
                ] * len(space.assignments),
            }
        ),
    )
    out = tmp_path / "map.html"
    out.write_text(render_html(pl), encoding="utf-8")

    harness = Path(__file__).with_name("test_map_mtl_viewer.js")
    proc = subprocess.run(
        [node, str(harness), str(out)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["targets"] == len(space.assignments)
    assert report["clicks"] >= 3, "target, prediction and VP traces must all be clickable"
    layers = " | ".join(report["layers"])
    for expected in ("Voronoi cells", "seed regions", "top-1 seed", "margin",
                     "measured VPs", "latent VPs", "shortest-ping VP",
                     "feasible region", "outer bounds", "true target", "prediction"):
        assert expected in layers, f"{expected!r} missing from the default view {layers!r}"

    # Only reachable with `post-filter only` unchecked, which is why the harness
    # reports the union over every control state rather than the first draw.
    every = " | ".join(report["allLayers"])
    assert "dropped by inclusion filter" in every, every
    assert "dropped by inclusion filter" not in layers, "dropped rings must be off by default"

    # Hovering a VP lifts its own constraint out of the bundle, and unhovering
    # puts it back; the harness fails outright if either half stops firing.
    # p5 must be the near-miss and p95 the disaster. The dropdown is ordered
    # failures-first, so a percentile taken over *it* reads the scale backwards;
    # the harness fails outright if the sequence stops being monotone.
    errs = report["percentileErrors"]
    assert errs == sorted(errs), errs

    assert report["hovers"] > 0
    assert report["anyRings"] is True
    assert report["highlighted"] > 0, "hovering a VP never highlighted its LTD ring"


# ---- fill winding -----------------------------------------------------------


def _signed_area_lonlat(ring):
    """Shoelace in (lon, lat) over a `[lat, lon]` ring. Positive is CCW."""
    return 0.5 * sum(
        lon_a * lat_b - lon_b * lat_a
        for (lat_a, lon_a), (lat_b, lon_b) in zip(ring, ring[1:])
    )


def test_cw_ring_reverses_only_counter_clockwise_input():
    ccw = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    assert _signed_area_lonlat(ccw) > 0
    assert _signed_area_lonlat(_cw_ring(ccw)) < 0
    cw = ccw[::-1]
    assert _cw_ring(cw) is cw, "an already-clockwise ring must not be copied or flipped"


def test_every_filled_ring_is_clockwise():
    """The regression: a CCW ring under `fill: "toself"`.

    Plotly treats a closed lat/lon path as a spherical polygon and fills the
    side to the right of the walk, so a counter-clockwise ring fills the
    antipodal complement — the whole globe minus the shape. H3 emits cell
    boundaries counter-clockwise, so all K seed hexagons did exactly that and
    the map rendered as one flat wash of colour. The outline is identical
    either way, so no other assertion here would catch it.
    """
    space = _space()
    for ring in seed_rings(space.seeds):
        assert ring[0] == ring[-1]
        assert _signed_area_lonlat(ring) < 0
    for ring in conus_voronoi_rings(space.seeds):
        assert _signed_area_lonlat(ring) < 0


# ---- the inclusion filter ---------------------------------------------------


def test_rings_carry_the_inclusion_filter_verdict():
    """`mtl_participants[]` is the post-`filter_redundant_outer_disks` set.

    Reading it back is exact. Recomputing the heuristic client-side — what
    `mtl_world_map` does — is a mirror that can drift from the geometry it
    claims to describe, and the drift would be invisible: both versions draw a
    plausible-looking set of circles.
    """
    space = _space()
    n = len(space.assignments)
    folds = pd.DataFrame(
        {
            "target_id": space.assignments["target_id"].to_numpy(),
            "ltd_predictions": [
                [
                    {"vp_id": "vp-0", "success": True, "upper_km": 900.0, "lower_km": 0.0},
                    {"vp_id": "vp-1", "success": True, "upper_km": 4000.0, "lower_km": 0.0},
                    # Dropped upstream by the LTD stage, so it is not a ring at all.
                    {"vp_id": "vp-2", "success": False, "upper_km": None, "lower_km": 0.0},
                ]
            ] * n,
            "mtl_participants": [[{"vp_id": "vp-0"}]] * n,
        }
    )
    pl = build_payload(
        _Run(), space, "test_cbg", scores=_scores(space), labels=_labels(space),
        edges=_edges(space), crossing=None, rings=seed_rings(space.seeds),
        voronoi=[], vps=_vps(), folds=folds,
    )
    for t in pl["targets"]:
        assert [r[0] for r in t["rings"]] == ["vp-0", "vp-1"], "failed LTDs are not rings"
        assert [r[3] for r in t["rings"]] == [1, 0]
        assert t["n_kept"] == 1


def test_a_run_without_the_filter_keeps_every_constraint():
    """`enable_circle_filter=False` makes participants == the admitted set, so
    nothing is marked dropped and the default view is unchanged."""
    space = _space()
    n = len(space.assignments)
    folds = pd.DataFrame(
        {
            "target_id": space.assignments["target_id"].to_numpy(),
            "ltd_predictions": [
                [
                    {"vp_id": "vp-0", "success": True, "upper_km": 900.0, "lower_km": 0.0},
                    {"vp_id": "vp-1", "success": True, "upper_km": 4000.0, "lower_km": 0.0},
                ]
            ] * n,
            "mtl_participants": [[{"vp_id": "vp-0"}, {"vp_id": "vp-1"}]] * n,
        }
    )
    pl = build_payload(
        _Run(), space, "test_cbg", scores=_scores(space), labels=_labels(space),
        edges=_edges(space), crossing=None, rings=seed_rings(space.seeds),
        voronoi=[], vps=_vps(), folds=folds,
    )
    for t in pl["targets"]:
        assert all(r[3] == 1 for r in t["rings"])
        assert t["n_kept"] == len(t["rings"]) == 2


def test_the_baseline_reports_no_kept_constraints():
    space = _space()
    pl = build_payload(
        _Run(), space, SHORTEST_PING, scores=_scores(space), labels=_labels(space),
        edges=_edges(space), crossing=None, rings=seed_rings(space.seeds),
        voronoi=[], vps=_vps(),
    )
    assert all(t["n_kept"] == 0 and t["rings"] == [] for t in pl["targets"])
