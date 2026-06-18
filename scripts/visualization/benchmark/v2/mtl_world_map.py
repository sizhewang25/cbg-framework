"""Build an interactive world-map viewer for MTL output of any v2 combo.

Per (fold, eval target) it draws on a Plotly geo map:
  - every participating VP with its LTD-predicted constraint as a great-circle
    ring (single ring for disk combos; outer + dashed inner for annulus combos)
  - the MTL/CTR predicted target location (red diamond)
  - the true target location (gold star)

Disk vs. annulus is auto-detected from `lower_km > 0` in the LTD predictions:
disk combos (low_envelope, speed_of_internet) leave `lower_km = 0`; annulus
combos (bounded_spline, normal_dist) emit a non-zero inner radius.

Math reference: spherical-cap → great-circle ring as in
`scripts/visualization/mtl/circle_intersections/inspect_percentile_cases.py`
:: great_circle_polygon. Same closed-form rendered client-side in JS so the
embedded JSON only carries `(vp_id, radius_km)` pairs.

CLI:
    python -m scripts.visualization.benchmark.v2.mtl_world_map \\
        --config scripts/benchmark/v2/config/north_america_as7018.yaml \\
        --combo vanilla_cbg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from shapely.geometry import MultiPolygon, Polygon

# Importing `scripts.framework.v2` populates LTD/MTL/CTR_REGISTRY via the
# `@register_*` decorators on each subclass module.
import scripts.framework.v2  # noqa: F401
from scripts.framework.v2.ltd.base import LTDResult
from scripts.framework.v2.registry import MTL_REGISTRY
from scripts.framework.v2.types import Coord, Distance, Error, Latency, VpId
from scripts.libs.cbg.rtt_model import haversine_distance
from scripts.processing.ripe_atlas.continents import continent_of


REPO_ROOT = Path(__file__).resolve().parents[4]
EARTH_RADIUS_KM = 6371.0

# HTML shell (markup + CSS) and the viewer JS live as standalone files under
# `templates/` so each can be edited in isolation. `render_html` assembles them
# into the single self-contained page written per combo. Placeholders:
#   mtl_world_map.html → __TITLE__, __PAYLOAD__, __SCRIPT__
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HTML_TEMPLATE_PATH = _TEMPLATE_DIR / "mtl_world_map.html"
_JS_TEMPLATE_PATH = _TEMPLATE_DIR / "mtl_world_map.js"
FILTERED_ANCHORS_PATH = REPO_ROOT / "datasets" / "ripe_atlas" / "filtered_anchors.json"

# Run-id prefix → canonical continent name for the same/rest selector.
# Other prefixes leave the selector disabled (continent unknown for this run).
_PREFIX_CONTINENT = {
    "north_america": "North America",
    "europe": "Europe",
}


def _same_continent_for_run(run_id: str) -> str | None:
    for prefix, canon in _PREFIX_CONTINENT.items():
        if run_id.startswith(prefix):
            return canon
    return None


def _load_ip_to_continent() -> dict[str, str]:
    """{address_v4: canonical continent name} from filtered_anchors.json,
    via `continent_of(country_code)`. Anchors with no `country_code` map to
    `"Unknown"`."""
    records = json.loads(FILTERED_ANCHORS_PATH.read_text())
    out: dict[str, str] = {}
    for r in records:
        ip = r.get("address_v4")
        if ip:
            out[ip] = continent_of(r.get("country_code"))
    return out


# ---- intersection polygons -------------------------------------------------
#
# The MTL intersection (Shapely Polygon/MultiPolygon for planar; list[Coord]
# of feasible-region vertices for spherical) is the actual feasible set the
# CTR stage collapses to a single coord. We recompute it from the saved
# ltd_predictions (re-instantiating the MTL via MTL_REGISTRY with the
# combo's mtl_kwargs) and write per-target JSON under
# `<viz_out>/<run_id>/static/<combo>/<fold>__<target_id>.json`.
# The HTML lazy-fetches them on target select to keep the page lean.

def _polygon_to_ring(poly: Polygon) -> dict[str, Any]:
    """Shapely stores coords as (x, y) = (lon, lat) in degree space (the
    planar combos build polygons in `_circle_to_planar_polygon`). Swap to
    `[lat, lon]` so the JSON matches the rest of the payload.

    Force every ring (outer + holes) to CW orientation in (lon, lat).
    Plotly scattergeo's `fill: "toself"` uses the right-hand convention
    on closed lat/lon paths, and the JS renders holes as a *separate*
    ocean-colored fill that subtracts visually — both rings need their
    bounded interior to be the filled side, so both must be CW. Source
    MTLs disagree on winding (Shapely ops can return either), so
    normalize here instead of relying on the upstream convention.
    """
    def _cw_coords(ring: Any) -> list[tuple[float, float]]:
        pts = list(ring.coords)
        return pts[::-1] if ring.is_ccw else pts

    outer = [[round(y, 4), round(x, 4)] for x, y in _cw_coords(poly.exterior)]
    holes = [
        [[round(y, 4), round(x, 4)] for x, y in _cw_coords(inner)]
        for inner in poly.interiors
    ]
    return {"outer": outer, "holes": holes}


def _serialize_intersection(intersection: Any) -> dict[str, Any] | None:
    """Convert MTLResult.intersection to a JSON-serializable dict.

    Returns None when the intersection is empty / unsupported, so the caller
    can skip writing the file (the JS treats a missing file as "no polygon").
    """
    if intersection is None:
        return None
    # Spherical (list[Coord]) — feasible-region vertices on the sphere. The
    # region is convex, so ordering by azimuth around the centroid yields a
    # closed polygon that approximates the (arc-bounded) boundary; coarse but
    # consistent with the existing planar approximation.
    #
    # Emit CW in (lon, lat). Plotly scattergeo's `fill: "toself"` treats a
    # closed lat/lon path as a *spherical* polygon and uses the right-hand
    # convention (interior on the right of the walk). CCW vertices end up
    # filling the antipodal hemisphere (whole world minus the diamond);
    # `reverse=True` flips the atan2 ascending order to CW so the small
    # feasible region is the one filled.
    if isinstance(intersection, list):
        if not intersection:
            return None
        lat_c = sum(c.lat for c in intersection) / len(intersection)
        lon_c = sum(c.lon for c in intersection) / len(intersection)
        ordered = sorted(
            intersection,
            key=lambda c: math.atan2(c.lat - lat_c, c.lon - lon_c),
            reverse=True,
        )
        outer = [[round(c.lat, 4), round(c.lon, 4)] for c in ordered]
        # Close the ring.
        if outer and outer[0] != outer[-1]:
            outer.append(outer[0])
        return {"kind": "spherical_vertices",
                "rings": [{"outer": outer, "holes": []}]}
    if isinstance(intersection, Polygon):
        return {"kind": "polygon", "rings": [_polygon_to_ring(intersection)]}
    if isinstance(intersection, MultiPolygon):
        return {"kind": "multipolygon",
                "rings": [_polygon_to_ring(p) for p in intersection.geoms]}
    return None


def _reconstruct_ltd_result(
    pred: dict[str, Any],
    vp_coord: Coord,
    latency_ms: float | None,
) -> LTDResult:
    """Rebuild an LTDResult from a `targets.parquet:ltd_predictions` row.

    The MTL stage reads `vp_coord`, `tg_distance.upper_km`, and (for
    weighted annulus) `latency` / `tg_distance.lower_km`. We rebuild the
    fields the MTL needs and fill the rest with what the saved row knows.
    """
    error_name = pred.get("error")
    error = Error[error_name] if error_name else None
    upper = pred.get("upper_km")
    lower = pred.get("lower_km") or 0.0
    if upper is not None:
        tg_distance: Distance | None = Distance(upper_km=float(upper),
                                                lower_km=float(lower))
    else:
        tg_distance = None
    vp_id_raw = pred.get("vp_id")
    return LTDResult(
        success=bool(pred.get("success")),
        error=error,
        vp_id=VpId(str(vp_id_raw)) if vp_id_raw is not None else None,
        vp_coord=vp_coord,
        latency=Latency(float(latency_ms)) if latency_ms is not None else None,
        tg_distance=tg_distance,
    )


def _run_mtl_for_target(
    mtl_name: str,
    mtl_kwargs: dict[str, Any],
    ltd_predictions: list[dict[str, Any]],
    vps: dict[str, list[float]],
    latency_by_vp: dict[str, float],
) -> Any:
    """Instantiate the MTL with its saved kwargs and re-run multilateration.

    Returns `MTLResult.intersection` (Shapely geometry / list[Coord] / None).
    """
    if mtl_name not in MTL_REGISTRY:
        raise KeyError(f"MTL {mtl_name!r} not registered")
    mtl = MTL_REGISTRY[mtl_name](**(mtl_kwargs or {}))

    ok: list[LTDResult] = []
    for pred in ltd_predictions:
        if not pred.get("success"):
            continue
        vp_id = str(pred["vp_id"])
        coord = vps.get(vp_id)
        if coord is None:
            continue
        upper = pred.get("upper_km")
        if upper is None or upper <= 0:
            continue
        ok.append(
            _reconstruct_ltd_result(
                pred,
                Coord(lat=coord[0], lon=coord[1]),
                latency_by_vp.get(vp_id),
            )
        )
    if not ok:
        return None
    result = mtl.multilaterate(ok)
    if not result.success:
        return None
    return result.intersection


def _fold_input_dir(source: str, run_id: str, setup: str, fold: str) -> Path:
    return REPO_ROOT / "scripts" / "benchmark" / "v2" / "inputs" / source / run_id / setup / fold


def _fold_output_dir(
    source: str, run_id: str, setup: str, fold: str, combo: str
) -> Path:
    return (
        REPO_ROOT / "scripts" / "benchmark" / "v2" / "outputs" / run_id / source
        / setup / fold / combo
    )


def _kept_after_filter(
    preds: list[tuple[str, float, float, float]],
) -> set[str]:
    """Return the set of vp_ids that survive `circle_preprocessing`.

    Mirrors [scripts/framework/geometry.py::circle_preprocessing] (the path
    `SphericalCircleMTL` walks when `enable_circle_filter=True`): a disk is
    redundant — and therefore dropped — when it fully contains another disk,
    i.e. `radius_i > haversine(center_i, center_j) + radius_j`. We break out
    of the inner loop as soon as `i` is marked redundant; later removals are
    still picked up when those entries take their turn as the outer.

    `preds` items are `(vp_id, lat, lon, radius_km)`.
    """
    ignored: set[str] = set()
    n = len(preds)
    for i in range(n):
        vp_i, lat_i, lon_i, r_i = preds[i]
        if vp_i in ignored:
            continue
        for j in range(i + 1, n):
            vp_j, lat_j, lon_j, r_j = preds[j]
            if vp_j in ignored:
                continue
            d = float(haversine_distance(lat_i, lon_i, lat_j, lon_j))
            if r_i > d + r_j:
                ignored.add(vp_i)
                break
            if r_j > d + r_i:
                ignored.add(vp_j)
    return {p[0] for p in preds} - ignored


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _build_fold_payload(
    source: str,
    run_id: str,
    setup: str,
    fold: str,
    combo: str,
    static_dir: Path | None = None,
) -> dict[str, Any]:
    in_dir = _fold_input_dir(source, run_id, setup, fold)
    out_dir = _fold_output_dir(source, run_id, setup, fold, combo)

    vp_configs = pd.read_parquet(in_dir / "vp_configs.parquet")
    targets = pd.read_parquet(out_dir / "targets.parquet")
    eval_obs = pd.read_parquet(in_dir / "eval_observations.parquet")

    # Load the combo's MTL spec from run.json so we can re-instantiate it
    # with the exact kwargs used at bench time (n_pts, enable_circle_filter,
    # speed_ratio, …). Read once per fold.
    run_meta = json.loads((out_dir / "run.json").read_text())
    mtl_name = run_meta.get("mtl")
    mtl_kwargs = run_meta.get("mtl_kwargs") or {}

    vps = {
        str(row.vp_id): [round(float(row.lat), 4), round(float(row.lon), 4)]
        for row in vp_configs.itertuples(index=False)
    }

    # Per-target shortest-ping VP: idxmin on latency_ms within each target_id.
    sp_idx = eval_obs.groupby("target_id")["latency_ms"].idxmin()
    sp_rows = eval_obs.loc[sp_idx, ["target_id", "vp_id", "latency_ms"]]
    shortest_ping_by_target = {
        str(r.target_id): (str(r.vp_id), float(r.latency_ms))
        for r in sp_rows.itertuples(index=False)
    }

    # Per-(target_id, vp_id) latency map — weighted-annulus MTL derives its
    # constraint weight from `latency`, so re-running MTL needs the real
    # value, not a placeholder. Tiny memory hit (<1 MB per fold).
    latency_lookup: dict[tuple[str, str], float] = {
        (str(r.target_id), str(r.vp_id)): float(r.latency_ms)
        for r in eval_obs[["target_id", "vp_id", "latency_ms"]].itertuples(index=False)
    }

    target_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        # Collect (vp_id, lat, lon, outer_km, inner_km) per surviving
        # prediction. Annulus combos populate inner_km > 0; disk combos
        # leave it at 0. The compact JSON form is
        # [vp_id, outer, inner, isKept, rtt_ms] — rtt_ms is the latency that
        # produced this constraint, surfaced in the VP hover popup.
        raw: list[tuple[str, float, float, float, float]] = []
        for p in row.ltd_predictions:
            if not p.get("success"):
                continue
            vp_id = str(p["vp_id"])
            coord = vps.get(vp_id)
            if coord is None:
                continue
            outer = _safe_float(p.get("upper_km"))
            if outer is None or outer <= 0:
                continue
            inner = _safe_float(p.get("lower_km")) or 0.0
            raw.append((vp_id, coord[0], coord[1], float(outer), float(inner)))

        # `_kept_after_filter` operates on outer disks. The same heuristic
        # runs as the pre-filter inside both PlanarCircleMTL/SphericalCircleMTL
        # AND PlanarAnnulusMTL/PlanarAnnulusWeightedMTL (via
        # `filter_redundant_outer_disks` on `enable_circle_filter=True`), so
        # we can apply it uniformly here regardless of mtl_kind.
        kept = _kept_after_filter([(v, la, lo, r) for v, la, lo, r, _ in raw])
        tid = str(row.target_id)
        preds = []
        for vp_id, _lat, _lon, outer, inner in raw:
            rtt = latency_lookup.get((tid, vp_id))
            preds.append(
                [
                    vp_id,
                    round(outer, 4),
                    round(inner, 4),
                    1 if vp_id in kept else 0,
                    round(rtt, 4) if rtt is not None else None,
                ]
            )

        pred_lat = _safe_float(row.pred_lat)
        pred_lon = _safe_float(row.pred_lon)
        pred = (
            [round(pred_lat, 4), round(pred_lon, 4)]
            if pred_lat is not None and pred_lon is not None
            else None
        )
        sp = shortest_ping_by_target.get(str(row.target_id))
        if sp is not None and sp[0] in vps:
            shortest_ping = {"vp_id": sp[0], "latency_ms": round(sp[1], 4)}
        else:
            shortest_ping = None

        # Recompute the MTL intersection from saved LTD predictions and write
        # it to a per-target JSON for the HTML to lazy-fetch. We do this even
        # when the bench-time MTL succeeded but the row's own MTL kind was
        # something Plotly can't render (e.g. degenerate); the JS just skips
        # rendering when the file is missing or empty.
        has_polygon = False
        if static_dir is not None and mtl_name:
            latency_by_vp: dict[str, float] = {}
            for p in row.ltd_predictions:
                vp_id_str = str(p["vp_id"])
                lat_ms = latency_lookup.get((str(row.target_id), vp_id_str))
                if lat_ms is not None:
                    latency_by_vp[vp_id_str] = lat_ms
            intersection = _run_mtl_for_target(
                mtl_name=mtl_name,
                mtl_kwargs=mtl_kwargs,
                ltd_predictions=list(row.ltd_predictions),
                vps=vps,
                latency_by_vp=latency_by_vp,
            )
            poly_json = _serialize_intersection(intersection)
            if poly_json is not None:
                out_path = static_dir / f"{fold}__{row.target_id}.json"
                out_path.write_text(
                    json.dumps(poly_json, allow_nan=False, separators=(",", ":"))
                )
                has_polygon = True

        target_rows.append(
            {
                "target_id": str(row.target_id),
                "true": [round(float(row.target_lat), 4), round(float(row.target_lon), 4)],
                "pred": pred,
                "status": str(row.status),
                "error_km": _safe_float(row.error_km),
                "intersection_kind": str(row.mtl_intersection_kind),
                "n_obs": int(row.n_obs),
                "n_ltd_success": int(row.n_ltd_success),
                "predictions": preds,
                "shortest_ping": shortest_ping,
                "has_polygon": has_polygon,
            }
        )

    return {"vps": vps, "targets": target_rows}


def build_payload(
    config_path: Path,
    combo_id: str,
    static_dir: Path | None = None,
) -> dict[str, Any]:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    run_id = cfg["run_id"]
    source = cfg["source"]
    setup = cfg["setup"]
    slices = cfg["slices"]

    same_continent = _same_continent_for_run(run_id)
    ip_to_continent = _load_ip_to_continent() if same_continent else {}

    if static_dir is not None:
        static_dir.mkdir(parents=True, exist_ok=True)

    merged_vps: dict[str, list[float]] = {}
    merged_targets: list[dict[str, Any]] = []
    for fold in slices:
        print(f"  fold {fold}: loading…", flush=True)
        fold_pl = _build_fold_payload(
            source, run_id, setup, fold, combo_id, static_dir=static_dir
        )
        # VPs are identical across folds for an ASN corpus (k-fold splits
        # anchors, not probes); first-seen wins on any conflict.
        for vp_id, coord in fold_pl["vps"].items():
            merged_vps.setdefault(vp_id, coord)
        for t in fold_pl["targets"]:
            t["fold"] = fold
            if same_continent:
                t["continent"] = ip_to_continent.get(t["target_id"], "Unknown")
            merged_targets.append(t)

    # Rank ASC by error_km; targets without a finite error_km drop to the end.
    def _err_sort_key(t: dict[str, Any]) -> tuple[int, float]:
        e = t.get("error_km")
        return (1, 0.0) if e is None else (0, float(e))

    merged_targets.sort(key=_err_sort_key)

    # Annulus combos always emit lower_km > 0 in their LTD predictions
    # (BoundedSplineLTD, NormalDistLTD). Disk combos leave inner=0.
    mtl_kind = "disk"
    for t in merged_targets:
        if any(pred[2] > 0 for pred in t["predictions"]):
            mtl_kind = "annulus"
            break

    # The HTML lazy-fetches polygons from this prefix on target select. Sibling
    # of the HTML (both under `<viz_out>/<run_id>/`) so it works against a
    # local static server with no rewriting.
    polygon_url_prefix = f"static/{combo_id}/" if static_dir is not None else None

    return {
        "run_id": run_id,
        "combo_id": combo_id,
        "source": source,
        "setup": setup,
        "earth_radius_km": EARTH_RADIUS_KM,
        "same_continent": same_continent,
        "mtl_kind": mtl_kind,
        "polygon_url_prefix": polygon_url_prefix,
        "vps": merged_vps,
        "targets": merged_targets,
    }


def render_html(payload: dict[str, Any]) -> str:
    """Assemble the standalone page from the HTML shell + viewer JS templates.

    The JS is injected first (it contains neither `__TITLE__` nor
    `__PAYLOAD__`), then the title, then the JSON payload last — the payload
    blob escapes `</` so an embedded `</script>` can't close the data block
    early.
    """
    title = f"{payload['run_id']} \u00b7 {payload['combo_id']}"
    html = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    js = _JS_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__SCRIPT__", js).replace("__TITLE__", title)
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return html.replace("__PAYLOAD__", blob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "scripts"
        / "benchmark"
        / "v2"
        / "config"
        / "north_america_as7018.yaml",
    )
    parser.add_argument("--combo", type=str, default="vanilla_cbg")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory under which <run_id>/<combo>_map.html is written.",
    )
    args = parser.parse_args()

    print(f"Loading config: {args.config}")

    # Build the run/combo output dirs first so build_payload can stream per-
    # target polygon JSONs into <run_dir>/static/<combo>/ as it walks folds.
    with open(args.config) as fh:
        run_id = yaml.safe_load(fh)["run_id"]
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    static_dir = run_dir / "static" / args.combo

    payload = build_payload(args.config, args.combo, static_dir=static_dir)

    out_path = run_dir / f"{payload['combo_id']}_map.html"
    html = render_html(payload)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    n_poly = sum(1 for t in payload["targets"] if t.get("has_polygon"))
    print(
        f"Wrote {out_path} ({size_mb:.2f} MB) "
        f"+ {n_poly} polygon JSONs under {static_dir}"
    )


if __name__ == "__main__":
    main()
