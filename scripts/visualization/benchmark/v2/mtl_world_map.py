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
    `[lat, lon]` so the JSON matches the rest of the payload."""
    outer = [[round(y, 4), round(x, 4)] for x, y in poly.exterior.coords]
    holes = [
        [[round(y, 4), round(x, 4)] for x, y in inner.coords]
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
    if isinstance(intersection, list):
        if not intersection:
            return None
        lat_c = sum(c.lat for c in intersection) / len(intersection)
        lon_c = sum(c.lon for c in intersection) / len(intersection)
        ordered = sorted(
            intersection, key=lambda c: math.atan2(c.lat - lat_c, c.lon - lon_c)
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
        # leave it at 0. The compact JSON form is [vp_id, outer, inner, isKept].
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
        preds = [
            [vp_id, round(outer, 4), round(inner, 4), 1 if vp_id in kept else 0]
            for vp_id, _lat, _lon, outer, inner in raw
        ]

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


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MTL world map — __TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 16px; color: #222; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 12px 0; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
              margin-bottom: 12px; }
  .controls label { font-size: 13px; }
  .controls select { font-size: 13px; padding: 4px 6px; }
  .controls select.wide { min-width: 360px; }
  .meta { font-size: 12px; color: #555; margin: 4px 0 12px 0; line-height: 1.5; }
  #plot { width: 100%; height: 720px; }
</style>
</head>
<body>
<h1>MTL world map — __TITLE__</h1>
<div class="controls">
  <label id="contLabel" style="display:none">Continent <select id="cont">
    <option value="all" selected>all</option>
    <option value="same">same continent</option>
    <option value="rest">rest of world</option>
  </select></label>
  <label>Percentile <select id="pct">
    <option value="" selected>(none)</option>
    <option value="5">p5</option>
    <option value="25">p25</option>
    <option value="50">p50</option>
    <option value="75">p75</option>
    <option value="95">p95</option>
  </select></label>
  <label><input type="checkbox" id="successOnly"> SUCCESS only</label>
  <label>Eval target <select id="target" class="wide"></select></label>
  <label id="keptOnlyLabel"><input type="checkbox" id="keptOnly"> only post-filter circles</label>
  <label id="maxRLabel">Hide circles ≥ <select id="maxR">
    <option value="10000" selected>10 000 km (¼ Earth circumf.)</option>
    <option value="20000">20 000 km (½ Earth circumf.)</option>
    <option value="5000">5 000 km</option>
    <option value="2000">2 000 km</option>
    <option value="0">(show all — large radii wrap to antipode)</option>
  </select></label>
  <label>Projection <select id="proj">
    <option value="natural earth">natural earth</option>
    <option value="orthographic">orthographic</option>
    <option value="equirectangular">equirectangular</option>
    <option value="robinson">robinson</option>
  </select></label>
  <label><input type="checkbox" id="showRegion"> feasible region</label>
  <span id="ringHint" style="display:none; font-size:12px; color:#555;">outer ring solid · inner ring dashed</span>
</div>
<div id="meta" class="meta"></div>
<div id="plot"></div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const EARTH = data.earth_radius_km;
  const vps = data.vps;
  const sameContinent = data.same_continent;  // null when not applicable
  const isAnnulus = data.mtl_kind === "annulus";
  if (isAnnulus) {
    // The outer-disk pre-filter (`filter_redundant_outer_disks`) runs for
    // annulus combos too, so the kept toggle stays active. We just relabel
    // and add the ring-style hint.
    const maxRLabel = document.getElementById("maxRLabel");
    maxRLabel.firstChild.nodeValue = "Hide rings ≥ ";
    document.getElementById("ringHint").style.display = "";
  }
  const contSel = document.getElementById("cont");
  const contLabel = document.getElementById("contLabel");
  if (sameContinent) {
    contLabel.style.display = "";
    // Show the continent name so "same/rest" isn't ambiguous.
    contSel.options[1].textContent = `same continent (${sameContinent})`;
    contSel.options[2].textContent = `rest of world (≠ ${sameContinent})`;
  }
  const pctSel = document.getElementById("pct");
  const successOnly = document.getElementById("successOnly");
  const targetSel = document.getElementById("target");
  const maxRSel = document.getElementById("maxR");
  const keptOnly = document.getElementById("keptOnly");
  const projSel = document.getElementById("proj");
  const showRegion = document.getElementById("showRegion");
  const metaDiv = document.getElementById("meta");
  const plotDiv = document.getElementById("plot");

  // ---- feasible-region lazy loader ----
  // Per-target JSON written by build_payload(). Key is "<fold>__<target_id>".
  // Values: a polygon dict on success, `null` on 404 (no feasible region),
  // or a Promise during fetch (handled by chaining .then on draw()).
  const POLY_URL_PREFIX = data.polygon_url_prefix;  // null when --static-dir wasn't passed
  const REGION_FILL = "rgba(220,40,60,0.18)";      // crimson, low alpha — fits the predicted-diamond color
  const REGION_LINE = "rgba(160,20,40,0.55)";
  // Ocean color matches the Plotly geo background so holes "subtract" cleanly.
  const HOLE_FILL = "rgb(225,235,245)";
  const HOLE_LINE = "rgba(160,20,40,0.45)";
  const polyCache = new Map();

  function polyKey(t) { return `${t.fold}__${t.target_id}`; }
  function polyUrl(t) { return `${POLY_URL_PREFIX}${polyKey(t)}.json`; }

  function ensurePolygon(t) {
    // Returns a Promise<polygon|null>. Cached for repeated draws of the
    // same target. has_polygon=false → resolve immediately without a fetch.
    if (!POLY_URL_PREFIX || !t.has_polygon) return Promise.resolve(null);
    const key = polyKey(t);
    if (polyCache.has(key)) {
      const v = polyCache.get(key);
      return v instanceof Promise ? v : Promise.resolve(v);
    }
    const p = fetch(polyUrl(t))
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .then((json) => {
        polyCache.set(key, json);
        return json;
      });
    polyCache.set(key, p);
    return p;
  }

  function foldLabel(f) {
    return f.startsWith("fold_") ? f.slice(5) : f;
  }
  function statusLabel(s) { return s === "SUCCESS" ? "SUCC" : "FAILED"; }
  function targetLabel(t) {
    const err = t.error_km != null ? t.error_km.toFixed(1) : "—";
    return `${t.target_id} (fold ${foldLabel(t.fold)}), error=${err} km, ${statusLabel(t.status)}`;
  }
  function percentileIndex(p, n) {
    // Match numpy.percentile(method="nearest") exactly — including its tie
    // rule (round half to even, a.k.a. banker's rounding). plot_error_cdf.py's
    // percentile table uses the same method, so the CDF table values and the
    // map's p-value bookmarks land on the same sample for any subset.
    if (n === 0) return 0;
    const raw = (p / 100) * (n - 1);
    const lo = Math.floor(raw);
    const frac = raw - lo;
    let i;
    if (frac < 0.5)       i = lo;
    else if (frac > 0.5)  i = lo + 1;
    else                  i = (lo % 2 === 0) ? lo : lo + 1;  // half-to-even
    return Math.max(0, Math.min(n - 1, i));
  }

  // ---- great-circle ring sampler (port of great_circle_polygon) ----
  function cross(a, b) {
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  }
  function norm(v) { return Math.hypot(v[0], v[1], v[2]); }

  function ringLatLon(latC, lonC, radiusKm, n) {
    const r = radiusKm / EARTH;
    const lat = latC * Math.PI/180;
    const lon = lonC * Math.PI/180;
    const c = [Math.cos(lat)*Math.cos(lon), Math.cos(lat)*Math.sin(lon), Math.sin(lat)];
    const tmp = Math.abs(c[2]) < 0.9 ? [0,0,1] : [1,0,0];
    let e1 = cross(c, tmp);
    const e1n = norm(e1);
    e1 = [e1[0]/e1n, e1[1]/e1n, e1[2]/e1n];
    const e2 = cross(c, e1);
    const cosR = Math.cos(r), sinR = Math.sin(r);
    const lats = new Array(n+1), lons = new Array(n+1);
    for (let i = 0; i <= n; i++) {
      const t = 2*Math.PI*i/n;
      const ct = Math.cos(t), st = Math.sin(t);
      const x = cosR*c[0] + sinR*(ct*e1[0] + st*e2[0]);
      const y = cosR*c[1] + sinR*(ct*e1[1] + st*e2[1]);
      const z = cosR*c[2] + sinR*(ct*e1[2] + st*e2[2]);
      lats[i] = Math.asin(Math.max(-1, Math.min(1, z))) * 180/Math.PI;
      lons[i] = Math.atan2(y, x) * 180/Math.PI;
    }
    // Break the polyline at antimeridian jumps so Plotly doesn't draw
    // a straight segment all the way across the map.
    const outLat = [lats[0]], outLon = [lons[0]];
    for (let i = 1; i < lats.length; i++) {
      if (Math.abs(lons[i] - lons[i-1]) > 180) {
        outLat.push(null); outLon.push(null);
      }
      outLat.push(lats[i]); outLon.push(lons[i]);
    }
    return { lats: outLat, lons: outLon };
  }

  // ---- dropdown population ----
  // `data.targets` is pre-sorted ASC by error_km (Python side). Any filtered
  // subset preserves that order, so percentile bookmarks are just index math.
  let currentList = data.targets;

  function activeList() {
    const cont = sameContinent ? contSel.value : "all";
    return data.targets.filter((t) => {
      if (successOnly.checked && t.status !== "SUCCESS") return false;
      if (cont === "same" && t.continent !== sameContinent) return false;
      // "rest" excludes the run's continent AND drops Unknowns so the
      // partition same+rest is clean (matches plot_error_cdf semantics).
      if (cont === "rest" && (t.continent === sameContinent || t.continent === "Unknown")) return false;
      return true;
    });
  }

  function populateTargets() {
    const prev = currentList[+targetSel.value];
    currentList = activeList();
    targetSel.innerHTML = "";
    currentList.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = targetLabel(t);
      targetSel.appendChild(o);
    });
    let idx = 0;
    if (pctSel.value !== "") {
      idx = percentileIndex(+pctSel.value, currentList.length);
    } else if (prev) {
      // Preserve previous selection if still present in the filtered list.
      const pIdx = currentList.findIndex(
        (t) => t.target_id === prev.target_id && t.fold === prev.fold,
      );
      if (pIdx >= 0) idx = pIdx;
    }
    targetSel.value = String(idx);
  }

  function applyPercentile() {
    if (pctSel.value === "") return;
    const idx = percentileIndex(+pctSel.value, currentList.length);
    targetSel.value = String(idx);
  }

  // ---- main draw ----
  function draw() {
    if (currentList.length === 0) {
      metaDiv.innerHTML = "<i>no eval targets match the current filters</i>";
      Plotly.react(plotDiv, [], { geo: { projection: { type: projSel.value } } }, { responsive: true });
      return;
    }
    const tIdx = +targetSel.value || 0;
    const t = currentList[tIdx];
    const maxR = +maxRSel.value;  // 0 = no cutoff

    const traces = [];

    // 1) per-VP great-circle rings (filtered by maxR, post-filter toggle, and
    // a hard cap covering the whole Earth — r >= π·R ≈ 20015 km — which has
    // no meaningful ring even with "show all" picked).
    // Prediction tuple: [vp_id, outer_km, inner_km, isKept]. For disk combos
    // inner_km == 0 and only the outer ring is drawn.
    const fullEarthKm = Math.PI * EARTH;  // ≈ 20015 km
    const onlyKept = keptOnly.checked;
    const totalKept = t.predictions.reduce((n, p) => n + (p[3] ? 1 : 0), 0);
    const totalDropped = t.predictions.length - totalKept;
    function included(pred) {
      const outer = pred[1], isKept = pred[3];
      if (outer >= fullEarthKm) return false;
      if (maxR > 0 && outer >= maxR) return false;
      if (onlyKept && !isKept) return false;
      return true;
    }

    const outerLats = [], outerLons = [];
    const innerLats = [], innerLons = [];
    const visibleInners = [], visibleOuters = [];
    let kept = 0;
    for (const pred of t.predictions) {
      if (!included(pred)) continue;
      const coord = vps[pred[0]];
      if (!coord) continue;
      kept++;
      const outer = pred[1], inner = pred[2];
      visibleOuters.push(outer);
      const oRing = ringLatLon(coord[0], coord[1], outer, 96);
      outerLats.push(...oRing.lats, null);
      outerLons.push(...oRing.lons, null);
      if (isAnnulus && inner > 0) {
        visibleInners.push(inner);
        const iRing = ringLatLon(coord[0], coord[1], inner, 96);
        innerLats.push(...iRing.lats, null);
        innerLons.push(...iRing.lons, null);
      }
    }
    if (outerLats.length) {
      traces.push({
        type: "scattergeo",
        mode: "lines",
        lat: outerLats, lon: outerLons,
        line: { width: 0.8, color: "rgba(60,90,160,0.35)" },
        name: isAnnulus ? `outer rings (${kept})` : `LTD circles (${kept})`,
        hoverinfo: "skip",
      });
    }
    if (innerLats.length) {
      traces.push({
        type: "scattergeo",
        mode: "lines",
        lat: innerLats, lon: innerLons,
        line: { width: 0.7, color: "rgba(60,90,160,0.55)", dash: "dash" },
        name: `inner rings (${visibleInners.length})`,
        hoverinfo: "skip",
      });
    }

    // 1b) feasible-region polygon (cached lazy fetch). Drawn after VP rings so
    //     the rings stay visible underneath, before markers so the markers sit
    //     on top. Holes are rendered as separate ocean-colored fills (Plotly's
    //     scattergeo doesn't natively support polygon-with-holes).
    const cached = polyCache.get(polyKey(t));
    const poly = (cached && !(cached instanceof Promise)) ? cached : null;
    if (showRegion.checked && poly) {
      const fillRings = [];   // outer rings (feasible)
      const holeRings = [];   // inner rings (holes inside feasible)
      for (const ring of (poly.rings || [])) {
        if (ring.outer && ring.outer.length) fillRings.push(ring.outer);
        for (const h of (ring.holes || [])) {
          if (h.length) holeRings.push(h);
        }
      }
      function ringsToTrace(rings, fillColor, lineColor) {
        const lat = [], lon = [];
        for (const r of rings) {
          for (const [la, lo] of r) { lat.push(la); lon.push(lo); }
          // null breaks separate multiple rings in one trace.
          lat.push(null); lon.push(null);
        }
        return {
          type: "scattergeo", mode: "lines",
          lat, lon,
          fill: "toself", fillcolor: fillColor,
          line: { color: lineColor, width: 1.2 },
          hoverinfo: "skip",
        };
      }
      if (fillRings.length) {
        traces.push(Object.assign(
          ringsToTrace(fillRings, REGION_FILL, REGION_LINE),
          { name: `feasible region (${fillRings.length})` },
        ));
      }
      if (holeRings.length) {
        // Holes drawn on top of the fill in the ocean color → visually
        // subtractive. Border keeps the hole edge legible.
        traces.push(Object.assign(
          ringsToTrace(holeRings, HOLE_FILL, HOLE_LINE),
          { name: `holes (${holeRings.length})`, showlegend: false },
        ));
      }
    }

    // 2) per-VP markers (only those whose ring is shown).
    const mkLat = [], mkLon = [], mkText = [];
    for (const pred of t.predictions) {
      if (!included(pred)) continue;
      const coord = vps[pred[0]];
      if (!coord) continue;
      mkLat.push(coord[0]); mkLon.push(coord[1]);
      const outer = pred[1], inner = pred[2];
      let line = `VP ${pred[0]}<br>outer = ${outer.toFixed(1)} km`;
      if (isAnnulus) {
        line += inner > 0 ? `<br>inner = ${inner.toFixed(1)} km` : `<br>inner = 0`;
      }
      line += `<br>${pred[3] ? "kept" : "dropped by pre-filter"}`;
      mkText.push(line);
    }
    traces.push({
      type: "scattergeo",
      mode: "markers",
      lat: mkLat, lon: mkLon,
      text: mkText,
      hoverinfo: "text",
      marker: { size: 5, color: "rgba(40,60,120,0.75)" },
      name: `VPs (${kept})`,
    });

    const hidden = t.predictions.length - kept;

    // 3) shortest-ping VP marker (drawn before true target so the star sits on top).
    if (t.shortest_ping) {
      const sp = t.shortest_ping;
      const spCoord = vps[sp.vp_id];
      if (spCoord) {
        traces.push({
          type: "scattergeo",
          mode: "markers",
          lat: [spCoord[0]], lon: [spCoord[1]],
          marker: { size: 14, color: "dodgerblue", symbol: "triangle-up",
                    line: { color: "white", width: 1.5 } },
          name: `shortest-ping VP (${sp.latency_ms} ms)`,
          text: [`shortest-ping VP ${sp.vp_id}<br>latency = ${sp.latency_ms} ms`],
          hoverinfo: "text",
        });
      }
    }

    // 4) true target.
    traces.push({
      type: "scattergeo",
      mode: "markers",
      lat: [t.true[0]], lon: [t.true[1]],
      marker: { size: 16, color: "gold", symbol: "star",
                line: { color: "black", width: 1 } },
      name: "true target",
      text: [`true: ${t.target_id}<br>(${t.true[0]}, ${t.true[1]})`],
      hoverinfo: "text",
    });

    // 5) predicted target.
    if (t.pred) {
      traces.push({
        type: "scattergeo",
        mode: "markers",
        lat: [t.pred[0]], lon: [t.pred[1]],
        marker: { size: 12, color: "crimson", symbol: "diamond",
                  line: { color: "white", width: 1.5 } },
        name: "predicted target",
        text: [`predicted<br>(${t.pred[0]}, ${t.pred[1]})`],
        hoverinfo: "text",
      });

      // 6) connector from predicted to true (straight-projection segment).
      traces.push({
        type: "scattergeo",
        mode: "lines",
        lat: [t.pred[0], t.true[0]], lon: [t.pred[1], t.true[1]],
        line: { width: 1.5, color: "crimson", dash: "dot" },
        showlegend: false,
        hoverinfo: "skip",
      });
    }

    const layout = {
      geo: {
        projection: { type: projSel.value },
        showland: true, landcolor: "rgb(243,243,238)",
        showocean: true, oceancolor: "rgb(225,235,245)",
        showcountries: true, countrycolor: "rgb(190,190,190)",
        coastlinecolor: "rgb(120,120,120)", coastlinewidth: 0.6,
        showframe: false,
        center: { lat: t.true[0], lon: t.true[1] },
      },
      margin: { l: 0, r: 0, t: 30, b: 0 },
      legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)" },
      title: { text: `${t.fold} · ${t.target_id} · ${t.status}`, font: { size: 14 } },
    };

    const errStr = t.error_km != null ? `${t.error_km.toFixed(2)} km` : "—";
    const predStr = t.pred ? `(${t.pred[0]}, ${t.pred[1]})` : "(none)";
    const unit = isAnnulus ? "ring(s)" : "circle(s)";
    const filterNote = hidden > 0
      ? ` &nbsp;|&nbsp; <span style="color:#b00">${hidden} ${unit} hidden</span>`
      : "";
    const pctNote = pctSel.value !== ""
      ? ` &nbsp;|&nbsp; rank ${tIdx + 1}/${currentList.length} (p${pctSel.value})`
      : ` &nbsp;|&nbsp; rank ${tIdx + 1}/${currentList.length}`;
    const median = (arr) => {
      if (!arr.length) return null;
      const s = arr.slice().sort((a, b) => a - b);
      const n = s.length;
      return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    };
    let constraintsClause =
      `pre-filter kept ${totalKept}/${t.predictions.length} (${totalDropped} dropped)`;
    if (isAnnulus) {
      const mi = median(visibleInners), mo = median(visibleOuters);
      const miStr = mi != null ? `${mi.toFixed(1)} km` : "—";
      const moStr = mo != null ? `${mo.toFixed(1)} km` : "—";
      constraintsClause +=
        ` &nbsp;|&nbsp; annular bounds (visible): median inner=${miStr}, median outer=${moStr}`;
    }
    metaDiv.innerHTML =
      `<b>${t.target_id}</b> (fold ${foldLabel(t.fold)}) — status=${t.status}, ` +
      `intersection=${t.intersection_kind}, ` +
      `n_ltd_success/n_obs=${t.n_ltd_success}/${t.n_obs}, ` +
      `${constraintsClause}<br>` +
      `true=(${t.true[0]}, ${t.true[1]})  ·  predicted=${predStr}  ·  error=${errStr}` +
      pctNote + filterNote;

    Plotly.react(plotDiv, traces, layout, { responsive: true });

    // Kick off the lazy fetch only when the cache hasn't seen this target yet.
    // Once the fetch resolves it populates the cache and re-calls draw(); on
    // that second pass the cache hit is consumed above and we MUST NOT re-fire
    // — otherwise the resolved-Promise path loops draw() infinitely.
    if (
      showRegion.checked && t.has_polygon && POLY_URL_PREFIX
      && !polyCache.has(polyKey(t))
    ) {
      const cur = polyKey(t);
      ensurePolygon(t).then((p) => {
        if (!p) return;
        const tNow = currentList[+targetSel.value || 0];
        if (tNow && polyKey(tNow) === cur) draw();
      });
    }
  }

  pctSel.addEventListener("change", () => { applyPercentile(); draw(); });
  successOnly.addEventListener("change", () => { populateTargets(); draw(); });
  contSel.addEventListener("change", () => { populateTargets(); draw(); });
  targetSel.addEventListener("change", draw);
  maxRSel.addEventListener("change", draw);
  keptOnly.addEventListener("change", draw);
  projSel.addEventListener("change", draw);
  showRegion.addEventListener("change", draw);

  populateTargets();
  draw();
})();
</script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    title = f"{payload['run_id']} · {payload['combo_id']}"
    body = HTML_TEMPLATE.replace("__TITLE__", title)
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return body.replace("__PAYLOAD__", blob)


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
