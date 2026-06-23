(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const EARTH = data.earth_radius_km;
  const vps = data.vps;
  const centroids = data.centroids || [];
  const voronoiCells = data.voronoi_cells || [];
  const radiusKm = data.radius_km || 50;
  const mechLabels = data.mech_labels || {};
  const isAnnulus = data.mtl_kind === "annulus";
  if (isAnnulus) {
    document.getElementById("maxRLabel").firstChild.nodeValue = "Hide rings ≥ ";
    document.getElementById("ringHint").style.display = "";
  }

  const outcomeSel = document.getElementById("outcome");
  const mechSel = document.getElementById("mech");
  const pctSel = document.getElementById("pct");
  const targetSel = document.getElementById("target");
  const maxRSel = document.getElementById("maxR");
  const keptOnly = document.getElementById("keptOnly");
  const projSel = document.getElementById("proj");
  const showRegion = document.getElementById("showRegion");
  const showCells = document.getElementById("showCells");
  const showVoronoi = document.getElementById("showVoronoi");
  const metaDiv = document.getElementById("meta");
  const plotDiv = document.getElementById("plot");

  // ---- feasible-region lazy loader (per-target JSON; see mtl_world_map.js) ----
  const POLY_URL_PREFIX = data.polygon_url_prefix;
  const REGION_FILL = "rgba(220,40,60,0.18)";
  const REGION_LINE = "rgba(160,20,40,0.55)";
  const HOLE_FILL = "rgb(225,235,245)";
  const HOLE_LINE = "rgba(160,20,40,0.45)";
  const polyCache = new Map();

  const RING_OUTER = "rgba(60,90,160,0.35)";
  const RING_OUTER_DIM = "rgba(60,90,160,0.06)";
  const RING_INNER = "rgba(60,90,160,0.55)";
  const RING_INNER_DIM = "rgba(60,90,160,0.10)";
  const HL_RING = "rgba(20,40,140,0.95)";

  // answer-space colours
  const CELL_DOT = "rgba(120,120,120,0.45)";   // all centroids
  const VORONOI_LINE = "rgba(220,30,40,0.85)"; // nearest-hub snapping fences
  const VORONOI_FILL = "rgba(0,0,0,0)";
  const TRUTH_CELL_LINE = "rgba(200,160,0,0.9)";
  const TRUTH_CELL_FILL = "rgba(255,215,0,0.10)";
  const PRED_CELL_LINE = "rgba(200,20,60,0.9)";
  const PRED_CELL_FILL = "rgba(220,40,60,0.08)";

  function polyKey(t) { return `${t.fold}__${t.target_id}`; }
  function polyUrl(t) { return `${POLY_URL_PREFIX}${polyKey(t)}.json`; }
  function ensurePolygon(t) {
    if (!POLY_URL_PREFIX || !t.has_polygon) return Promise.resolve(null);
    const key = polyKey(t);
    if (polyCache.has(key)) {
      const v = polyCache.get(key);
      return v instanceof Promise ? v : Promise.resolve(v);
    }
    const p = fetch(polyUrl(t))
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .then((json) => { polyCache.set(key, json); return json; });
    polyCache.set(key, p);
    return p;
  }

  function foldLabel(f) { return f.startsWith("fold_") ? f.slice(5) : f; }
  function outcomeTag(t) {
    if (t.match) return "MATCH";
    return t.outcome === "GIVE_UP" ? "GIVE-UP" : "WRONG";
  }
  function mechTag(t) {
    return t.mechanism ? (mechLabels[t.mechanism] || t.mechanism) : "—";
  }
  function targetLabel(t) {
    const e = t.error_to_centroid_km;
    const eStr = e != null ? `${e.toFixed(0)} km` : "—";
    const tail = t.match ? "MATCH" : `${outcomeTag(t)}/${mechTag(t)}`;
    return `${t.target_id} (fold ${foldLabel(t.fold)}) — d→cell=${eStr} — ${tail}`;
  }
  function percentileIndex(p, n) {
    if (n === 0) return 0;
    const raw = (p / 100) * (n - 1);
    const lo = Math.floor(raw); const frac = raw - lo;
    let i;
    if (frac < 0.5) i = lo; else if (frac > 0.5) i = lo + 1;
    else i = (lo % 2 === 0) ? lo : lo + 1;
    return Math.max(0, Math.min(n - 1, i));
  }

  // ---- great-circle ring sampler ----
  function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
  function norm(v) { return Math.hypot(v[0], v[1], v[2]); }
  function ringLatLon(latC, lonC, radiusKmArg, n) {
    const r = radiusKmArg / EARTH;
    const lat = latC * Math.PI/180, lon = lonC * Math.PI/180;
    const c = [Math.cos(lat)*Math.cos(lon), Math.cos(lat)*Math.sin(lon), Math.sin(lat)];
    const tmp = Math.abs(c[2]) < 0.9 ? [0,0,1] : [1,0,0];
    let e1 = cross(c, tmp); const e1n = norm(e1);
    e1 = [e1[0]/e1n, e1[1]/e1n, e1[2]/e1n];
    const e2 = cross(c, e1);
    const cosR = Math.cos(r), sinR = Math.sin(r);
    const lats = new Array(n+1), lons = new Array(n+1);
    for (let i = 0; i <= n; i++) {
      const t = 2*Math.PI*i/n, ct = Math.cos(t), st = Math.sin(t);
      const x = cosR*c[0] + sinR*(ct*e1[0] + st*e2[0]);
      const y = cosR*c[1] + sinR*(ct*e1[1] + st*e2[1]);
      const z = cosR*c[2] + sinR*(ct*e1[2] + st*e2[2]);
      lats[i] = Math.asin(Math.max(-1, Math.min(1, z))) * 180/Math.PI;
      lons[i] = Math.atan2(y, x) * 180/Math.PI;
    }
    const outLat = [lats[0]], outLon = [lons[0]];
    for (let i = 1; i < lats.length; i++) {
      if (Math.abs(lons[i] - lons[i-1]) > 180) { outLat.push(null); outLon.push(null); }
      outLat.push(lats[i]); outLon.push(lons[i]);
    }
    return { lats: outLat, lons: outLon };
  }

  // ---- dropdown population (targets are pre-sorted failures-first in Python) ----
  let currentList = data.targets;
  function activeList() {
    const oc = outcomeSel.value, mc = mechSel.value;
    return data.targets.filter((t) => {
      if (oc === "fail" && t.match) return false;
      if (oc === "WRONG" && !(t.outcome === "WRONG" && !t.match)) return false;
      if (oc === "GIVE_UP" && t.outcome !== "GIVE_UP") return false;
      if (oc === "MATCH" && !t.match) return false;
      if (mc !== "all" && t.mechanism !== mc) return false;
      return true;
    });
  }
  function populateTargets() {
    const prev = currentList[+targetSel.value];
    currentList = activeList();
    targetSel.innerHTML = "";
    currentList.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = String(i); o.textContent = targetLabel(t);
      targetSel.appendChild(o);
    });
    let idx = 0;
    if (pctSel.value !== "") idx = percentileIndex(+pctSel.value, currentList.length);
    else if (prev) {
      const pIdx = currentList.findIndex(
        (t) => t.target_id === prev.target_id && t.fold === prev.fold);
      if (pIdx >= 0) idx = pIdx;
    }
    targetSel.value = String(idx);
  }
  function applyPercentile() {
    if (pctSel.value === "") return;
    targetSel.value = String(percentileIndex(+pctSel.value, currentList.length));
  }

  // ---- VP-dot hover highlighting ----
  function onVpHover(ev, ctx) {
    const pt = ev.points && ev.points[0];
    if (!pt || pt.curveNumber !== ctx.vpMarkerIdx) return;
    const pred = pt.customdata; if (!pred) return;
    const coord = vps[pred[0]]; if (!coord) return;
    const o = ringLatLon(coord[0], coord[1], pred[1], 96);
    Plotly.restyle(plotDiv, { lat: [o.lats], lon: [o.lons] }, [ctx.hlOuterIdx]);
    if (ctx.hlInnerIdx >= 0) {
      if (isAnnulus && pred[2] > 0) {
        const ir = ringLatLon(coord[0], coord[1], pred[2], 96);
        Plotly.restyle(plotDiv, { lat: [ir.lats], lon: [ir.lons] }, [ctx.hlInnerIdx]);
      } else Plotly.restyle(plotDiv, { lat: [[]], lon: [[]] }, [ctx.hlInnerIdx]);
    }
    if (ctx.outerIdx >= 0) Plotly.restyle(plotDiv, { "line.color": RING_OUTER_DIM }, [ctx.outerIdx]);
    if (ctx.innerIdx >= 0) Plotly.restyle(plotDiv, { "line.color": RING_INNER_DIM }, [ctx.innerIdx]);
  }
  function onVpUnhover(ctx) {
    Plotly.restyle(plotDiv, { lat: [[]], lon: [[]] }, [ctx.hlOuterIdx]);
    if (ctx.hlInnerIdx >= 0) Plotly.restyle(plotDiv, { lat: [[]], lon: [[]] }, [ctx.hlInnerIdx]);
    if (ctx.outerIdx >= 0) Plotly.restyle(plotDiv, { "line.color": RING_OUTER }, [ctx.outerIdx]);
    if (ctx.innerIdx >= 0) Plotly.restyle(plotDiv, { "line.color": RING_INNER }, [ctx.innerIdx]);
  }

  function cellTrace(centroid, lineColor, fillColor, name) {
    const ring = ringLatLon(centroid[0], centroid[1], radiusKm, 72);
    return {
      type: "scattergeo", mode: "lines", lat: ring.lats, lon: ring.lons,
      fill: "toself", fillcolor: fillColor, line: { width: 1.6, color: lineColor },
      name, hoverinfo: "skip",
    };
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
    const maxR = +maxRSel.value;
    const traces = [];
    let outerIdx = -1, innerIdx = -1, hlOuterIdx = -1, hlInnerIdx = -1, vpMarkerIdx = -1;

    // 0a) nearest-hub snapping fences (Voronoi partition of the centroids,
    //     clipped to the landmass) — drawn first so it sits underneath.
    if (showVoronoi && showVoronoi.checked && voronoiCells.length) {
      const lat = [], lon = [];
      for (const ring of voronoiCells) {
        for (const [la, lo] of ring) { lat.push(la); lon.push(lo); }
        lat.push(null); lon.push(null);
      }
      traces.push({
        type: "scattergeo", mode: "lines", lat, lon, fill: "toself",
        fillcolor: VORONOI_FILL, line: { width: 0.7, color: VORONOI_LINE },
        name: `nearest-hub cells (${voronoiCells.length})`, hoverinfo: "skip",
      });
    }

    // 0) answer-space centroids (faint dots) — context for the cells.
    if (showCells.checked && centroids.length) {
      traces.push({
        type: "scattergeo", mode: "markers",
        lat: centroids.map((c) => c[0]), lon: centroids.map((c) => c[1]),
        marker: { size: 3, color: CELL_DOT },
        name: `centroids (${centroids.length})`, hoverinfo: "skip",
      });
    }

    // 1) per-VP great-circle rings.
    const fullEarthKm = Math.PI * EARTH;
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
    const outerLats = [], outerLons = [], innerLats = [], innerLons = [];
    let kept = 0;
    for (const pred of t.predictions) {
      if (!included(pred)) continue;
      const coord = vps[pred[0]]; if (!coord) continue;
      kept++;
      const oRing = ringLatLon(coord[0], coord[1], pred[1], 96);
      outerLats.push(...oRing.lats, null); outerLons.push(...oRing.lons, null);
      if (isAnnulus && pred[2] > 0) {
        const iRing = ringLatLon(coord[0], coord[1], pred[2], 96);
        innerLats.push(...iRing.lats, null); innerLons.push(...iRing.lons, null);
      }
    }
    if (outerLats.length) {
      outerIdx = traces.length;
      traces.push({ type: "scattergeo", mode: "lines", lat: outerLats, lon: outerLons,
        line: { width: 0.8, color: RING_OUTER },
        name: isAnnulus ? `outer rings (${kept})` : `LTD circles (${kept})`, hoverinfo: "skip" });
    }
    if (innerLats.length) {
      innerIdx = traces.length;
      traces.push({ type: "scattergeo", mode: "lines", lat: innerLats, lon: innerLons,
        line: { width: 0.7, color: RING_INNER, dash: "dash" },
        name: "inner rings", hoverinfo: "skip" });
    }

    // 1b) feasible-region polygon (lazy fetched).
    const cached = polyCache.get(polyKey(t));
    const poly = (cached && !(cached instanceof Promise)) ? cached : null;
    if (showRegion.checked && poly) {
      const fillRings = [], holeRings = [];
      for (const ring of (poly.rings || [])) {
        if (ring.outer && ring.outer.length) fillRings.push(ring.outer);
        for (const h of (ring.holes || [])) if (h.length) holeRings.push(h);
      }
      function ringsToTrace(rings, fillColor, lineColor) {
        const lat = [], lon = [];
        for (const r of rings) { for (const [la, lo] of r) { lat.push(la); lon.push(lo); } lat.push(null); lon.push(null); }
        return { type: "scattergeo", mode: "lines", lat, lon, fill: "toself",
          fillcolor: fillColor, line: { color: lineColor, width: 1.2 }, hoverinfo: "skip" };
      }
      if (fillRings.length) traces.push(Object.assign(
        ringsToTrace(fillRings, REGION_FILL, REGION_LINE), { name: `feasible region (${fillRings.length})` }));
      if (holeRings.length) traces.push(Object.assign(
        ringsToTrace(holeRings, HOLE_FILL, HOLE_LINE), { name: `holes (${holeRings.length})`, showlegend: false }));
    }

    // 1c) highlight ring traces (restyled on VP hover).
    hlOuterIdx = traces.length;
    traces.push({ type: "scattergeo", mode: "lines", lat: [], lon: [],
      line: { width: 2.2, color: HL_RING }, hoverinfo: "skip", showlegend: false });
    if (isAnnulus) {
      hlInnerIdx = traces.length;
      traces.push({ type: "scattergeo", mode: "lines", lat: [], lon: [],
        line: { width: 1.8, color: HL_RING, dash: "dash" }, hoverinfo: "skip", showlegend: false });
    }

    // 1d) answer cells: truth's cell (gold) and prediction's cell (crimson).
    if (showCells.checked && t.truth_centroid) {
      traces.push(cellTrace(t.truth_centroid, TRUTH_CELL_LINE, TRUTH_CELL_FILL,
        `truth cell (R=${radiusKm} km)`));
    }
    if (showCells.checked && t.pred_centroid && !t.match) {
      traces.push(cellTrace(t.pred_centroid, PRED_CELL_LINE, PRED_CELL_FILL, "predicted cell"));
    }

    // 2) per-VP markers.
    const mkLat = [], mkLon = [], mkText = [], mkPred = [];
    for (const pred of t.predictions) {
      if (!included(pred)) continue;
      const coord = vps[pred[0]]; if (!coord) continue;
      mkLat.push(coord[0]); mkLon.push(coord[1]); mkPred.push(pred);
      let line = `VP ${pred[0]}`;
      line += pred[4] != null ? `<br>RTT = ${pred[4].toFixed(2)} ms` : "<br>RTT = —";
      line += `<br>outer = ${pred[1].toFixed(1)} km`;
      if (isAnnulus) line += pred[2] > 0 ? `<br>inner = ${pred[2].toFixed(1)} km` : "<br>inner = 0";
      line += `<br>${pred[3] ? "kept" : "dropped by pre-filter"}`;
      mkText.push(line);
    }
    vpMarkerIdx = traces.length;
    traces.push({ type: "scattergeo", mode: "markers", lat: mkLat, lon: mkLon,
      text: mkText, customdata: mkPred, hoverinfo: "text",
      marker: { size: 5, color: "rgba(40,60,120,0.75)" }, name: `VPs (${kept})` });
    const hidden = t.predictions.length - kept;

    // 3) shortest-ping VP.
    if (t.shortest_ping) {
      const spCoord = vps[t.shortest_ping.vp_id];
      if (spCoord) traces.push({ type: "scattergeo", mode: "markers",
        lat: [spCoord[0]], lon: [spCoord[1]],
        marker: { size: 14, color: "dodgerblue", symbol: "triangle-up", line: { color: "white", width: 1.5 } },
        name: `shortest-ping VP (${t.shortest_ping.latency_ms} ms)`,
        text: [`shortest-ping VP ${t.shortest_ping.vp_id}<br>latency = ${t.shortest_ping.latency_ms} ms`],
        hoverinfo: "text" });
    }

    // 4) truth centroid marker (the correct answer cell centre).
    if (t.truth_centroid) traces.push({ type: "scattergeo", mode: "markers",
      lat: [t.truth_centroid[0]], lon: [t.truth_centroid[1]],
      marker: { size: 9, color: "gold", symbol: "circle", line: { color: "rgba(140,110,0,1)", width: 1 } },
      name: "truth centroid", text: ["truth centroid"], hoverinfo: "text" });

    // 5) true target.
    traces.push({ type: "scattergeo", mode: "markers", lat: [t.true[0]], lon: [t.true[1]],
      marker: { size: 16, color: "gold", symbol: "star", line: { color: "black", width: 1 } },
      name: "true target", text: [`true: ${t.target_id}<br>(${t.true[0]}, ${t.true[1]})`], hoverinfo: "text" });

    // 6) predicted target + connector to truth.
    if (t.pred) {
      traces.push({ type: "scattergeo", mode: "markers", lat: [t.pred[0]], lon: [t.pred[1]],
        marker: { size: 12, color: "crimson", symbol: "diamond", line: { color: "white", width: 1.5 } },
        name: "predicted target", text: [`predicted<br>(${t.pred[0]}, ${t.pred[1]})`], hoverinfo: "text" });
      traces.push({ type: "scattergeo", mode: "lines", lat: [t.pred[0], t.true[0]], lon: [t.pred[1], t.true[1]],
        line: { width: 1.5, color: "crimson", dash: "dot" }, showlegend: false, hoverinfo: "skip" });
    }

    const layout = {
      geo: { projection: { type: projSel.value },
        showland: true, landcolor: "rgb(243,243,238)",
        showocean: true, oceancolor: "rgb(225,235,245)",
        showcountries: true, countrycolor: "rgb(190,190,190)",
        coastlinecolor: "rgb(120,120,120)", coastlinewidth: 0.6,
        showframe: false, center: { lat: t.true[0], lon: t.true[1] } },
      margin: { l: 0, r: 0, t: 30, b: 0 },
      legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)" },
      title: { text: `${t.fold} · ${t.target_id} · ${outcomeTag(t)}${t.match ? "" : " / " + mechTag(t)}`, font: { size: 14 } },
    };

    const f = t.feat || {};
    const fnum = (v, u) => (v != null ? `${v.toFixed(u === "x" ? 2 : (u === "f" ? 2 : 0))}${u === "x" ? "×" : (u === "f" ? "" : " km")}` : "—");
    const eStr = t.error_to_centroid_km != null ? `${t.error_to_centroid_km.toFixed(1)} km` : "—";
    const badgeColor = t.match ? "#2a8" : (t.outcome === "GIVE_UP" ? "#d39" : "#d33");
    const pctNote = pctSel.value !== ""
      ? ` &nbsp;|&nbsp; rank ${tIdx + 1}/${currentList.length} (p${pctSel.value})`
      : ` &nbsp;|&nbsp; rank ${tIdx + 1}/${currentList.length}`;
    const unit = isAnnulus ? "ring(s)" : "circle(s)";
    const filterNote = hidden > 0 ? ` &nbsp;|&nbsp; <span style="color:#b00">${hidden} ${unit} hidden</span>` : "";
    metaDiv.innerHTML =
      `<span class="badge" style="background:${badgeColor};color:#fff">${outcomeTag(t)}</span> ` +
      `mechanism=<b>${mechTag(t)}</b> &nbsp;|&nbsp; d(pred→truth cell)=<b>${eStr}</b> ` +
      `(match iff ≤ snap) &nbsp;|&nbsp; nearest VP=${fnum(f.avail_min_vp_km)} · ` +
      `cell gap=${fnum(f.nearest_other_centroid_km)} · RTT infl=${fnum(f.part_min_infl, "x")} · ` +
      `blockers=${fnum(f.frac_blockers, "f")}<br>` +
      `pre-filter kept ${totalKept}/${t.predictions.length} (${totalDropped} dropped), ` +
      `n_ltd_success/n_obs=${t.n_ltd_success}/${t.n_obs}, intersection=${t.intersection_kind}` +
      pctNote + filterNote;

    Plotly.react(plotDiv, traces, layout, { responsive: true });

    if (plotDiv.removeAllListeners) {
      plotDiv.removeAllListeners("plotly_hover");
      plotDiv.removeAllListeners("plotly_unhover");
    }
    if (vpMarkerIdx >= 0) {
      const hoverCtx = { outerIdx, innerIdx, hlOuterIdx, hlInnerIdx, vpMarkerIdx };
      plotDiv.on("plotly_hover", (ev) => onVpHover(ev, hoverCtx));
      plotDiv.on("plotly_unhover", () => onVpUnhover(hoverCtx));
    }

    if (showRegion.checked && t.has_polygon && POLY_URL_PREFIX && !polyCache.has(polyKey(t))) {
      const cur = polyKey(t);
      ensurePolygon(t).then((p) => {
        if (!p) return;
        const tNow = currentList[+targetSel.value || 0];
        if (tNow && polyKey(tNow) === cur) draw();
      });
    }
  }

  pctSel.addEventListener("change", () => { applyPercentile(); draw(); });
  outcomeSel.addEventListener("change", () => { populateTargets(); draw(); });
  mechSel.addEventListener("change", () => { populateTargets(); draw(); });
  targetSel.addEventListener("change", draw);
  maxRSel.addEventListener("change", draw);
  keptOnly.addEventListener("change", draw);
  projSel.addEventListener("change", draw);
  if (showVoronoi) showVoronoi.addEventListener("change", draw);
  showRegion.addEventListener("change", draw);
  showCells.addEventListener("change", draw);

  populateTargets();
  draw();
})();
