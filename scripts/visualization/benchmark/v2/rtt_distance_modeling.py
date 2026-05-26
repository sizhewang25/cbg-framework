"""Build an interactive HTML viewer for `vanilla_cbg` (low-envelope) LTD fits.

Reads a benchmark-v2 run produced by `scripts/benchmark/v2/Snakefile`, walks
every fold, and emits a single self-contained HTML file. The page has three
dropdowns — fold, VP, eval target — and re-draws an RTT-vs-distance scatter
with the 2/3·c baseline and the per-VP LP low-envelope line. Picking an eval
target overlays the true and predicted (rtt, distance) points with dashed
indicator lines to the axes.

CLI:
    python -m scripts.visualization.benchmark.v2.rtt_distance_modeling \\
        --config scripts/benchmark/v2/config/north_america_as7018.yaml \\
        --combo vanilla_cbg
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.framework.v2.ltd.low_envelope import LowEnvelopeLTD  # noqa: F401  (pickle resolution)
from scripts.libs.cbg.rtt_model import (  # noqa: F401  (pickle resolution)
    THEORETICAL_SLOPE,
    RTTDistanceModel,
    haversine_distance,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _fold_input_dir(source: str, run_id: str, setup: str, fold: str) -> Path:
    return REPO_ROOT / "scripts" / "benchmark" / "v2" / "inputs" / source / run_id / setup / fold


def _fold_output_dir(
    source: str, run_id: str, setup: str, fold: str, combo: str
) -> Path:
    return (
        REPO_ROOT
        / "scripts"
        / "benchmark"
        / "v2"
        / "outputs"
        / run_id
        / source
        / setup
        / fold
        / combo
    )


def _build_fold_payload(
    source: str, run_id: str, setup: str, fold: str, combo: str
) -> dict[str, Any]:
    in_dir = _fold_input_dir(source, run_id, setup, fold)
    out_dir = _fold_output_dir(source, run_id, setup, fold, combo)

    fit_samples = pd.read_parquet(in_dir / "fit_samples.parquet")
    vp_configs = pd.read_parquet(in_dir / "vp_configs.parquet")
    eval_obs = pd.read_parquet(in_dir / "eval_observations.parquet")

    with open(out_dir / "fit_checkpoint.pkl", "rb") as fh:
        model: LowEnvelopeLTD = pickle.load(fh)

    fit_samples = fit_samples.assign(
        distance_km=haversine_distance(
            fit_samples["vp_lat"].to_numpy(),
            fit_samples["vp_lon"].to_numpy(),
            fit_samples["probe_lat"].to_numpy(),
            fit_samples["probe_lon"].to_numpy(),
        )
    )
    eval_obs = eval_obs.assign(
        true_dist_km=haversine_distance(
            eval_obs["vp_lat"].to_numpy(),
            eval_obs["vp_lon"].to_numpy(),
            eval_obs["target_lat"].to_numpy(),
            eval_obs["target_lon"].to_numpy(),
        )
    )

    samples_by_vp = {
        vp: g[["latency_ms", "distance_km"]].to_numpy()
        for vp, g in fit_samples.groupby("vp_id", sort=False)
    }
    evals_by_vp = {vp: g for vp, g in eval_obs.groupby("vp_id", sort=False)}

    vps: dict[str, Any] = {}
    for row in vp_configs.itertuples(index=False):
        vp_id = str(row.vp_id)
        sub = model._submodels.get(vp_id)
        pairs = samples_by_vp.get(vp_id, np.empty((0, 2)))
        samples = [
            [round(float(rtt), 4), round(float(dist), 4)]
            for rtt, dist in pairs
        ]
        eval_targets: list[dict[str, Any]] = []
        eg = evals_by_vp.get(vp_id)
        if eg is not None:
            for r in eg.itertuples(index=False):
                eval_targets.append(
                    {
                        "target_id": str(r.target_id),
                        "rtt": round(float(r.latency_ms), 4),
                        "true_dist": round(float(r.true_dist_km), 4),
                    }
                )
            eval_targets.sort(key=lambda d: d["target_id"])

        vps[vp_id] = {
            "lat": float(row.lat),
            "lon": float(row.lon),
            "samples": samples,
            "slope": (
                float(sub.slope) if sub is not None and sub.slope is not None else None
            ),
            "intercept": (
                float(sub.intercept)
                if sub is not None and sub.intercept is not None
                else None
            ),
            "n_measurements": int(sub.n_measurements) if sub is not None else 0,
            "fitted": bool(sub.fitted) if sub is not None else False,
            "fit_message": (sub.fit_message if sub is not None else "VP not in fit"),
            "eval_targets": eval_targets,
        }

    return {"vps": vps}


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
        "theoretical_slope": float(THEORETICAL_SLOPE),
        "folds": folds,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Low-envelope LTD viewer — __TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 16px; color: #222; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 12px 0; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
              margin-bottom: 12px; }
  .controls label { font-size: 13px; }
  .controls select { font-size: 13px; padding: 4px 6px; min-width: 180px; }
  .meta { font-size: 12px; color: #555; margin: 4px 0 12px 0; }
  #plot { width: 100%; height: 620px; }
  .warn { color: #b00; font-weight: 600; }
</style>
</head>
<body>
<h1>Low-envelope LTD viewer — __TITLE__</h1>
<div class="controls">
  <label>Fold <select id="fold"></select></label>
  <label>VP <select id="vp"></select></label>
  <label>Eval target <select id="target"></select></label>
</div>
<div id="meta" class="meta"></div>
<div id="plot"></div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const foldSel = document.getElementById("fold");
  const vpSel = document.getElementById("vp");
  const targetSel = document.getElementById("target");
  const metaDiv = document.getElementById("meta");
  const plotDiv = document.getElementById("plot");
  const theoSlope = data.theoretical_slope;

  Object.keys(data.folds).sort().forEach((f) => {
    const o = document.createElement("option");
    o.value = f; o.textContent = f; foldSel.appendChild(o);
  });

  function populateVps() {
    vpSel.innerHTML = "";
    const fold = data.folds[foldSel.value];
    const ids = Object.keys(fold.vps).sort();
    ids.forEach((id) => {
      const o = document.createElement("option");
      const sub = fold.vps[id];
      const tag = sub.fitted ? "" : " [unfitted]";
      o.value = id;
      o.textContent = `${id} (n=${sub.samples.length})${tag}`;
      vpSel.appendChild(o);
    });
  }

  function populateTargets() {
    targetSel.innerHTML = "";
    const none = document.createElement("option");
    none.value = ""; none.textContent = "(none)";
    targetSel.appendChild(none);
    const vp = data.folds[foldSel.value].vps[vpSel.value];
    vp.eval_targets.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = `${t.target_id} — rtt=${t.rtt} ms, true=${t.true_dist} km`;
      targetSel.appendChild(o);
    });
  }

  function draw() {
    const fold = data.folds[foldSel.value];
    const vp = fold.vps[vpSel.value];
    const samples = vp.samples;
    const rtts = samples.map((p) => p[0]);
    const dists = samples.map((p) => p[1]);
    const dMax = dists.length ? Math.max(...dists) : 1.0;
    const rttMax = rtts.length ? Math.max(...rtts) : 1.0;
    const dGrid = [0, dMax];

    const traces = [];

    traces.push({
      x: rtts, y: dists,
      mode: "markers", type: "scattergl",
      name: `samples (n=${samples.length})`,
      marker: { size: 5, color: "rgba(110,110,110,0.45)" },
      hovertemplate: "rtt=%{x:.2f} ms<br>dist=%{y:.2f} km<extra></extra>",
    });

    traces.push({
      x: dGrid.map((d) => theoSlope * d),
      y: dGrid,
      mode: "lines",
      name: `2/3·c baseline (${theoSlope.toFixed(4)} ms/km)`,
      line: { color: "black", dash: "dash", width: 1.5 },
      hoverinfo: "skip",
    });

    if (vp.fitted && vp.slope != null && vp.intercept != null) {
      traces.push({
        x: dGrid.map((d) => vp.slope * d + vp.intercept),
        y: dGrid,
        mode: "lines",
        name: `low envelope: ${vp.slope.toFixed(5)}·d + ${vp.intercept.toFixed(2)}`,
        line: { color: "crimson", width: 2 },
        hoverinfo: "skip",
      });
    }

    let xRangeMax = rttMax * 1.05 || 1.0;
    let yRangeMax = dMax * 1.05 || 1.0;

    const tIdx = targetSel.value;
    let evalNote = "";
    if (tIdx !== "" && vp.eval_targets[+tIdx]) {
      const t = vp.eval_targets[+tIdx];
      const predDist =
        vp.fitted && vp.slope && vp.slope > 0
          ? Math.max(0, (t.rtt - vp.intercept) / vp.slope)
          : null;

      xRangeMax = Math.max(xRangeMax, t.rtt * 1.1);
      const yCandidates = [yRangeMax, t.true_dist * 1.1];
      if (predDist != null) yCandidates.push(predDist * 1.1);
      yRangeMax = Math.max(...yCandidates);

      traces.push({
        x: [t.rtt], y: [t.true_dist],
        mode: "markers",
        name: `true: ${t.target_id}`,
        marker: { size: 12, color: "royalblue", symbol: "circle",
                  line: { color: "white", width: 1 } },
        hovertemplate: "true<br>rtt=%{x:.2f} ms<br>dist=%{y:.2f} km<extra></extra>",
      });

      if (predDist != null) {
        traces.push({
          x: [t.rtt], y: [predDist],
          mode: "markers",
          name: `predicted: ${predDist.toFixed(2)} km`,
          marker: { size: 12, color: "darkorange", symbol: "diamond",
                    line: { color: "white", width: 1 } },
          hovertemplate: "predicted<br>rtt=%{x:.2f} ms<br>dist=%{y:.2f} km<extra></extra>",
        });

        // dashed indicators from (rtt, predDist) to both axes
        traces.push({
          x: [t.rtt, t.rtt], y: [0, predDist],
          mode: "lines", showlegend: false, hoverinfo: "skip",
          line: { color: "darkorange", dash: "dot", width: 1.5 },
        });
        traces.push({
          x: [0, t.rtt], y: [predDist, predDist],
          mode: "lines", showlegend: false, hoverinfo: "skip",
          line: { color: "darkorange", dash: "dot", width: 1.5 },
        });

        evalNote =
          ` — eval target <b>${t.target_id}</b>: ` +
          `rtt=${t.rtt.toFixed(2)} ms, true=${t.true_dist.toFixed(2)} km, ` +
          `predicted=${predDist.toFixed(2)} km, ` +
          `Δ=${(predDist - t.true_dist).toFixed(2)} km`;
      } else {
        evalNote =
          ` — eval target <b>${t.target_id}</b>: ` +
          `rtt=${t.rtt.toFixed(2)} ms, true=${t.true_dist.toFixed(2)} km (no prediction: VP unfitted)`;
      }
    }

    const layout = {
      xaxis: { title: "RTT (ms)", range: [0, xRangeMax], gridcolor: "#eee" },
      yaxis: { title: "Distance (km)", range: [0, yRangeMax], gridcolor: "#eee" },
      legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)" },
      margin: { l: 60, r: 20, t: 30, b: 50 },
      plot_bgcolor: "white",
      title: { text: `${foldSel.value} · VP ${vpSel.value}`, font: { size: 14 } },
    };

    const fitStr = vp.fitted
      ? `slope=${vp.slope.toFixed(5)} ms/km, intercept=${vp.intercept.toFixed(2)} ms, n=${vp.n_measurements}`
      : `<span class="warn">VP not fitted</span> — ${vp.fit_message}`;
    metaDiv.innerHTML =
      `VP @ (${vp.lat.toFixed(4)}, ${vp.lon.toFixed(4)}) — ${fitStr}` + evalNote;

    Plotly.react(plotDiv, traces, layout, { responsive: true });
  }

  foldSel.addEventListener("change", () => {
    populateVps();
    populateTargets();
    draw();
  });
  vpSel.addEventListener("change", () => { populateTargets(); draw(); });
  targetSel.addEventListener("change", draw);

  populateVps();
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
    # Escape </script> just in case any string contains it.
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
        help="Directory under which <run_id>/<combo>.html is written.",
    )
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    payload = build_payload(args.config, args.combo)

    out_dir = args.out_dir / payload["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{payload['combo_id']}.html"

    html = render_html(payload)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
