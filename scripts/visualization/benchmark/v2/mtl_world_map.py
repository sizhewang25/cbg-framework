"""Build an interactive world-map viewer for MTL output of `vanilla_cbg`.

Per (fold, eval target) it draws on a Plotly geo map:
  - every participating VP with its LTD-predicted radius as a great-circle ring
  - the MTL/CTR predicted target location (red diamond)
  - the true target location (gold star)

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


REPO_ROOT = Path(__file__).resolve().parents[4]
EARTH_RADIUS_KM = 6371.0


def _fold_input_dir(source: str, run_id: str, setup: str, fold: str) -> Path:
    return REPO_ROOT / "scripts" / "benchmark" / "v2" / "inputs" / source / run_id / setup / fold


def _fold_output_dir(
    source: str, run_id: str, setup: str, fold: str, combo: str
) -> Path:
    return (
        REPO_ROOT / "scripts" / "benchmark" / "v2" / "outputs" / run_id / source
        / setup / fold / combo
    )


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
    source: str, run_id: str, setup: str, fold: str, combo: str
) -> dict[str, Any]:
    in_dir = _fold_input_dir(source, run_id, setup, fold)
    out_dir = _fold_output_dir(source, run_id, setup, fold, combo)

    vp_configs = pd.read_parquet(in_dir / "vp_configs.parquet")
    targets = pd.read_parquet(out_dir / "targets.parquet")
    eval_obs = pd.read_parquet(in_dir / "eval_observations.parquet")

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

    target_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        preds = []
        for p in row.ltd_predictions:
            if not p.get("success"):
                continue
            vp_id = str(p["vp_id"])
            if vp_id not in vps:
                continue
            radius = _safe_float(p.get("upper_km"))
            if radius is None or radius <= 0:
                continue
            preds.append([vp_id, round(radius, 4)])

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
            }
        )

    target_rows.sort(key=lambda t: t["target_id"])
    return {"vps": vps, "targets": target_rows}


def build_payload(config_path: Path, combo_id: str) -> dict[str, Any]:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    run_id = cfg["run_id"]
    source = cfg["source"]
    setup = cfg["setup"]
    slices = cfg["slices"]

    folds: dict[str, Any] = {}
    for fold in slices:
        print(f"  fold {fold}: loading…", flush=True)
        folds[fold] = _build_fold_payload(source, run_id, setup, fold, combo_id)

    return {
        "run_id": run_id,
        "combo_id": combo_id,
        "source": source,
        "setup": setup,
        "earth_radius_km": EARTH_RADIUS_KM,
        "folds": folds,
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
  <label>Fold <select id="fold"></select></label>
  <label>Eval target <select id="target" class="wide"></select></label>
  <label>Hide circles ≥ <select id="maxR">
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
</div>
<div id="meta" class="meta"></div>
<div id="plot"></div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const EARTH = data.earth_radius_km;
  const foldSel = document.getElementById("fold");
  const targetSel = document.getElementById("target");
  const maxRSel = document.getElementById("maxR");
  const projSel = document.getElementById("proj");
  const metaDiv = document.getElementById("meta");
  const plotDiv = document.getElementById("plot");

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
  Object.keys(data.folds).sort().forEach((f) => {
    const o = document.createElement("option");
    o.value = f; o.textContent = f; foldSel.appendChild(o);
  });

  function populateTargets() {
    targetSel.innerHTML = "";
    const fold = data.folds[foldSel.value];
    fold.targets.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      const err = t.error_km != null ? t.error_km.toFixed(0) : "—";
      o.textContent = `${t.target_id} — ${t.status}, error=${err} km, n=${t.n_ltd_success}/${t.n_obs}`;
      targetSel.appendChild(o);
    });
  }

  // ---- main draw ----
  function draw() {
    const fold = data.folds[foldSel.value];
    const tIdx = +targetSel.value || 0;
    const t = fold.targets[tIdx];
    const vps = fold.vps;
    const maxR = +maxRSel.value;  // 0 = no cutoff

    const traces = [];

    // 1) per-VP great-circle rings (filtered by maxR).
    // Anything whose cap covers ~the entire Earth (r >= π·R ≈ 20015 km) has
    // no meaningful ring — always skip those even when "show all" is picked.
    const fullEarthKm = Math.PI * EARTH;  // ≈ 20015 km
    const ringLats = [], ringLons = [];
    let kept = 0, hidden = 0;
    for (const [vpId, radius] of t.predictions) {
      if (radius >= fullEarthKm) { hidden++; continue; }
      if (maxR > 0 && radius >= maxR) { hidden++; continue; }
      const coord = vps[vpId];
      if (!coord) continue;
      kept++;
      const ring = ringLatLon(coord[0], coord[1], radius, 96);
      ringLats.push(...ring.lats, null);
      ringLons.push(...ring.lons, null);
    }
    if (ringLats.length) {
      traces.push({
        type: "scattergeo",
        mode: "lines",
        lat: ringLats, lon: ringLons,
        line: { width: 0.8, color: "rgba(60,90,160,0.35)" },
        name: `LTD circles (${kept})`,
        hoverinfo: "skip",
      });
    }

    // 2) per-VP markers (only those whose circle is shown).
    const mkLat = [], mkLon = [], mkText = [];
    for (const [vpId, radius] of t.predictions) {
      if (radius >= fullEarthKm) continue;
      if (maxR > 0 && radius >= maxR) continue;
      const coord = vps[vpId];
      if (!coord) continue;
      mkLat.push(coord[0]); mkLon.push(coord[1]);
      mkText.push(`VP ${vpId}<br>radius = ${radius.toFixed(1)} km`);
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
      title: { text: `${foldSel.value} · ${t.target_id} · ${t.status}`, font: { size: 14 } },
    };

    const errStr = t.error_km != null ? `${t.error_km.toFixed(2)} km` : "—";
    const predStr = t.pred ? `(${t.pred[0]}, ${t.pred[1]})` : "(none)";
    const filterNote = hidden > 0
      ? ` &nbsp;|&nbsp; <span style="color:#b00">${hidden} circle(s) hidden by ≥${maxR} km filter</span>`
      : "";
    metaDiv.innerHTML =
      `<b>${t.target_id}</b> — status=${t.status}, intersection=${t.intersection_kind}, ` +
      `n_ltd_success/n_obs=${t.n_ltd_success}/${t.n_obs}<br>` +
      `true=(${t.true[0]}, ${t.true[1]})  ·  predicted=${predStr}  ·  error=${errStr}` +
      filterNote;

    Plotly.react(plotDiv, traces, layout, { responsive: true });
  }

  foldSel.addEventListener("change", () => { populateTargets(); draw(); });
  targetSel.addEventListener("change", draw);
  maxRSel.addEventListener("change", draw);
  projSel.addEventListener("change", draw);

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
    payload = build_payload(args.config, args.combo)

    out_dir = args.out_dir / payload["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{payload['combo_id']}_map.html"

    html = render_html(payload)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
