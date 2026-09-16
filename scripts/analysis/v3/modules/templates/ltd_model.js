/* LTD fit viewer — fold / VP selection over a prebuilt payload.
 *
 * This file draws and nothing else. Every band, centre line and distance in the
 * payload was computed in Python against the real fitted model objects
 * (`figure_ltd_model.build_payload`), which is the deliberate inversion of the
 * legacy viewer: that one shipped slope/intercept and reimplemented the fit
 * here, so it rendered exactly one LTD family and drifted from the Python the
 * moment a second one existed. Adding an LTD variant must not require touching
 * this file.
 *
 * Axes are RTT (x) against distance km (y), matching the static primitives in
 * scripts/visualization/ltd/.
 */

(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);

  const C_SAMPLES = "#898781";
  const C_BAND = "#2a78d6";
  const C_BAND_FILL = "rgba(42,120,214,0.16)";
  const C_CENTER = "#4a3aa7";
  const C_BASELINE = "#0b0b0b";
  const C_EVAL = "#e34948";
  const C_EVAL_BAND = "#eda100";

  const $ = (id) => document.getElementById(id);

  const foldSel = $("fold");
  const vpSel = $("vp");
  const vpSearch = $("vpSearch");
  const targetSel = $("targets");
  const targetSearch = $("targetSearch");
  const onlyThisVp = $("onlyThisVp");
  const showScatter = $("showScatter");
  const showBand = $("showBand");
  const showCenter = $("showCenter");
  const showBaseline = $("showBaseline");
  const logY = $("logY");
  const sharedAxes = $("sharedAxes");

  /* The user's own target picks, kept here rather than read back off the
     <select>: repopulating the list on a fold or VP change destroys the
     option nodes, and with them the selection, so it has to live outside the
     DOM to survive. */
  let picked = new Set(DATA.preselect_targets || []);

  function fold() {
    return DATA.folds[foldSel.value] || DATA.folds[DATA.fold_ids[0]];
  }

  function vpIds() {
    return Object.keys(fold().vps).sort();
  }

  /* ---- control population ------------------------------------------------ */

  function fillFolds() {
    foldSel.innerHTML = "";
    for (const f of DATA.fold_ids) {
      const d = DATA.folds[f];
      const o = document.createElement("option");
      o.value = f;
      o.textContent = `${f} — ${Object.keys(d.vps).length} VPs, ${d.n_targets} targets`;
      foldSel.appendChild(o);
    }
    if (!foldSel.value) foldSel.value = DATA.fold_ids[0];
  }

  function fillVps() {
    const prev = vpSel.value;
    const q = (vpSearch.value || "").trim().toLowerCase();
    const f = fold();
    const all = vpIds();
    vpSel.innerHTML = "";
    const shown = [];
    for (const vp of all) {
      if (q && !vp.toLowerCase().includes(q)) continue;
      const v = f.vps[vp];
      const o = document.createElement("option");
      o.value = vp;
      /* The fit state is in the label rather than a separate readout: an
         unfitted VP is the single most common explanation for an empty band,
         and it has to be visible while choosing, not after. */
      const fit = v.fitted === false ? " · UNFITTED" : "";
      o.textContent = `${vp} — ${v.n_samples} samples${fit}`;
      vpSel.appendChild(o);
      shown.push(vp);
    }
    $("vpCount").textContent = `${shown.length} / ${all.length} VPs`;
    if (prev && shown.includes(prev)) vpSel.value = prev;
    else vpSel.value = shown.length ? shown[0] : "";
  }

  function fillTargets() {
    const q = (targetSearch.value || "").trim().toLowerCase();
    const vp = vpSel.value;
    const targets = fold().targets || {};
    const all = Object.keys(targets).sort();
    targetSel.innerHTML = "";
    let shown = 0;
    for (const tg of all) {
      const t = targets[tg];
      if (onlyThisVp.checked && vp && !(vp in t.vps)) continue;
      const hay = `${tg} ${t.status || ""}`.toLowerCase();
      if (q && !hay.includes(q)) continue;
      const o = document.createElement("option");
      o.value = tg;
      const err = t.error_km === null ? "n/a" : `${t.error_km.toFixed(0)} km`;
      o.textContent = `${tg} — ${t.status} · err ${err} · ${t.n_participants} VPs`;
      o.selected = picked.has(tg);
      targetSel.appendChild(o);
      shown++;
    }
    $("targetCount").textContent = `${shown} / ${all.length} targets`;
  }

  /* Read the multi-select back into `picked`. Only the options currently on
     screen are authoritative -- a target filtered out of the list keeps
     whatever state it had, so narrowing the search does not silently drop it
     from the overlay. */
  function syncPicked() {
    for (const o of targetSel.options) {
      if (o.selected) picked.add(o.value);
      else picked.delete(o.value);
    }
  }

  /* ---- traces ------------------------------------------------------------ */

  /* Split a band into contiguous runs of defined points.
   *
   * A gap is real signal, not missing data: the pooled Spotter model declines
   * every RTT below its fitted minimum, and the Octant band declines above its
   * cutoff -- on as01/fold_4 that is 86 of 134 VPs for Spotter and all 134 for
   * Octant. Plotly would bridge straight across a null with a filled polygon,
   * inventing a claim the model does not make, so each run is emitted as its
   * own filled pair of traces. */
  function bandSegments(band) {
    const runs = [];
    let cur = [];
    for (const [rtt, lo, hi] of band) {
      if (lo === null || hi === null) {
        if (cur.length) runs.push(cur);
        cur = [];
      } else {
        cur.push([rtt, lo, hi]);
      }
    }
    if (cur.length) runs.push(cur);
    return runs;
  }

  function bandTraces(band) {
    const out = [];
    let first = true;
    for (const seg of bandSegments(band)) {
      const rtt = seg.map((p) => p[0]);
      out.push({
        x: rtt,
        y: seg.map((p) => p[1]),
        mode: "lines",
        line: { color: C_BAND, width: 1.5 },
        name: "fitted band (lower)",
        legendgroup: "band",
        showlegend: false,
        hovertemplate: "lower %{y:.0f} km @ %{x:.1f} ms<extra></extra>",
      });
      out.push({
        x: rtt,
        y: seg.map((p) => p[2]),
        mode: "lines",
        line: { color: C_BAND, width: 1.5 },
        fill: "tonexty",
        fillcolor: C_BAND_FILL,
        name: "fitted band",
        legendgroup: "band",
        showlegend: first,
        hovertemplate: "upper %{y:.0f} km @ %{x:.1f} ms<extra></extra>",
      });
      first = false;
    }
    return out;
  }

  function baselineTrace(xMax) {
    /* rtt = slope * km, so km = rtt / slope. Drawn across the x range rather
       than the observed points so it stays a reference even when a VP's
       samples sit far off it. */
    const s = DATA.theoretical_slope;
    return {
      x: [0, xMax],
      y: [0, xMax / s],
      mode: "lines",
      line: { color: C_BASELINE, width: 1.2, dash: "dash" },
      name: "2/3 c",
      hoverinfo: "skip",
    };
  }

  function evalTraces(vp) {
    const targets = fold().targets || {};
    const xs = [];
    const ys = [];
    const text = [];
    const bandX = [];
    const bandY = [];
    for (const tg of [...picked].sort()) {
      const t = targets[tg];
      if (!t) continue;
      const p = t.vps[vp];
      if (!p) continue;
      const [rtt, km, lo, hi] = p;
      xs.push(rtt);
      ys.push(km);
      const err = t.error_km === null ? "n/a" : `${t.error_km.toFixed(0)} km`;
      text.push(
        `${tg}<br>${t.status} · err ${err}<br>true ${km.toFixed(0)} km @ ${rtt.toFixed(1)} ms`
      );
      /* The echoed bounds are what this VP contributed to the
         multilateration, so they are drawn as a vertical extent at the
         measured RTT -- the constraint the target had to satisfy, against
         where it actually was. */
      if (lo !== null && hi !== null) {
        bandX.push(rtt, rtt, null);
        bandY.push(lo, hi, null);
      }
    }
    const out = [];
    if (bandX.length) {
      out.push({
        x: bandX,
        y: bandY,
        mode: "lines",
        line: { color: C_EVAL_BAND, width: 3 },
        name: "echoed LTD bound",
        hoverinfo: "skip",
      });
    }
    if (xs.length) {
      out.push({
        x: xs,
        y: ys,
        mode: "markers",
        marker: { color: C_EVAL, size: 10, symbol: "x", line: { width: 1 } },
        name: "eval target (true)",
        text,
        hovertemplate: "%{text}<extra></extra>",
      });
    }
    return out;
  }

  /* ---- draw -------------------------------------------------------------- */

  let frozen = null;

  function draw() {
    const vp = vpSel.value;
    const f = fold();
    const v = f.vps[vp];
    if (!v) {
      $("meta").innerHTML = "<span class='warn'>no VP matches the current filter</span>";
      Plotly.react("plot", [], {}, { displaylogo: false });
      return;
    }

    const traces = [];
    if (showScatter.checked && v.samples.length) {
      traces.push({
        x: v.samples.map((p) => p[0]),
        y: v.samples.map((p) => p[1]),
        mode: "markers",
        marker: { color: C_SAMPLES, size: 4, opacity: 0.55 },
        name: `fit samples (${v.n_samples})`,
        hovertemplate: "%{y:.0f} km @ %{x:.1f} ms<extra></extra>",
      });
    }
    if (showBand.checked) traces.push(...bandTraces(v.band));
    if (showCenter.checked && v.center.length) {
      traces.push({
        x: v.center.map((p) => p[0]),
        y: v.center.map((p) => p[1]),
        mode: "lines",
        line: { color: C_CENTER, width: 2 },
        name: "centre",
        hovertemplate: "centre %{y:.0f} km @ %{x:.1f} ms<extra></extra>",
      });
    }

    const xObs = v.samples.map((p) => p[0]);
    const xMax = (xObs.length ? Math.max(...xObs) : 100) * 1.05;
    if (showBaseline.checked) traces.push(baselineTrace(xMax));
    traces.push(...evalTraces(vp));

    /* Freezing the axes is opt-in because both readings are wanted: free axes
       show each VP's own fit in full, frozen axes make two VPs comparable by
       eye. The frozen window is taken from the first draw after the box is
       ticked, so it is a VP the user chose rather than an arbitrary global. */
    if (!sharedAxes.checked) frozen = null;
    else if (!frozen) frozen = { x: [0, xMax] };

    const layout = {
      margin: { l: 64, r: 16, t: 10, b: 48 },
      xaxis: {
        title: "min RTT (ms)",
        rangemode: "tozero",
        gridcolor: "#e1e0d9",
        zeroline: false,
        range: frozen ? frozen.x : undefined,
      },
      yaxis: {
        title: "distance VP → target (km)",
        type: logY.checked ? "log" : "linear",
        rangemode: logY.checked ? "normal" : "tozero",
        gridcolor: "#e1e0d9",
        zeroline: false,
      },
      hovermode: "closest",
      showlegend: true,
      legend: { orientation: "h", y: -0.16 },
      plot_bgcolor: "#fff",
      paper_bgcolor: "#fff",
    };
    Plotly.react("plot", traces, layout, { displaylogo: false, responsive: true });
    drawMeta(vp, v, f);
  }

  function drawMeta(vp, v, f) {
    const prov = (DATA.provenance || {})[foldSel.value] || {};
    const routeLabel =
      prov.route === "materialized_inputs"
        ? `materialized inputs (<code>${prov.path}</code>)`
        : prov.route === "rebuilt_from_dataset_csv"
        ? `rebuilt from <code>${prov.csv}</code>, folds pinned by <code>${prov.stratification}</code>`
        : "unknown";
    const sc = f.status_counts || {};
    const counts = Object.keys(sc)
      .map((k) => `${k} ${sc[k]}`)
      .join(" · ");
    const defined = v.band.filter((p) => p[1] !== null).length;
    const gap = v.band.length - defined;
    const parts = [
      `<b>${DATA.method_label}</b> · ltd <code>${DATA.ltd}</code> ${JSON.stringify(
        DATA.ltd_kwargs
      )} · mtl <code>${DATA.mtl}</code> · ctr <code>${DATA.ctr}</code>`,
      `${DATA.run_id} · ${DATA.source}/${DATA.setup} · ${foldSel.value} · ${f.n_fit_samples} fit samples · ${counts}`,
      `VP <b>${vp}</b> at ${v.lat}, ${v.lon} · ${v.n_samples} samples${
        v.samples.length < v.n_samples ? ` (${v.samples.length} drawn)` : ""
      }${v.n_measurements !== null ? ` · model saw ${v.n_measurements}` : ""}`,
      `fit scatter: ${routeLabel}`,
    ];
    if (!v.center.length) {
      parts.push(
        "this model exposes no centre line — only the band it constrains with"
      );
    }
    if (v.fitted === false) {
      parts.push(
        `<span class='warn'>this VP is UNFITTED${
          v.fit_message ? `: ${v.fit_message}` : ""
        } — the model falls back rather than constraining from it</span>`
      );
    }
    if (gap > 0) {
      parts.push(
        `<span class='warn'>${gap} of ${v.band.length} band samples have no prediction</span> — the model declines that RTT range; drawn as a gap, not as zero`
      );
    }
    if (f.model_rebuilt_stateless) {
      parts.push(
        "model reconstructed from <code>run.json</code> — this LTD is stateless, so the run wrote a <code>.stateless</code> marker rather than a pickle"
      );
    }
    const skipped = Object.keys(DATA.skipped_folds || {});
    if (skipped.length) {
      parts.push(`<span class='warn'>folds not built: ${skipped.join(", ")}</span>`);
    }
    $("meta").innerHTML = parts.join("<br>");
  }

  /* ---- wiring ------------------------------------------------------------ */

  fillFolds();
  fillVps();
  fillTargets();
  draw();

  foldSel.addEventListener("change", () => {
    frozen = null;
    fillVps();
    fillTargets();
    draw();
  });
  vpSel.addEventListener("change", () => {
    fillTargets();
    draw();
  });
  vpSearch.addEventListener("input", () => {
    fillVps();
    fillTargets();
    draw();
  });
  targetSearch.addEventListener("input", fillTargets);
  onlyThisVp.addEventListener("change", fillTargets);
  targetSel.addEventListener("change", () => {
    syncPicked();
    draw();
  });
  for (const el of [
    showScatter,
    showBand,
    showCenter,
    showBaseline,
    logY,
    sharedAxes,
  ]) {
    el.addEventListener("change", draw);
  }

  // Headless-test hooks. `typeof module` is undefined in a browser, so this is
  // dead code there; under node it lets a harness drive draw() without a real
  // DOM. See scripts/analysis/v3/tests/test_ltd_model_viewer.js.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      redraw: draw,
      refillVps: fillVps,
      refillTargets: fillTargets,
      syncPicked: syncPicked,
      pick: (tg) => picked.add(tg),
      clearPicks: () => picked.clear(),
      segments: bandSegments,
    };
  }
})();
