(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const EARTH = data.earth_radius_km;
  const vps = data.vps || {};
  const seeds = data.seeds || [];
  const voronoiCells = data.voronoi || [];
  const isAnnulus = data.mtl_kind === "annulus";

  const statusSel = document.getElementById("status");
  const pctSel = document.getElementById("pct");
  const targetSel = document.getElementById("target");
  const targetSearch = document.getElementById("targetSearch");
  const targetCount = document.getElementById("targetCount");
  const projSel = document.getElementById("proj");
  const maxRSel = document.getElementById("maxR");
  const showVoronoi = document.getElementById("showVoronoi");
  const showSeeds = document.getElementById("showSeeds");
  const showMargin = document.getElementById("showMargin");
  const showRings = document.getElementById("showRings");
  const keptOnly = document.getElementById("keptOnly");
  const showRegion = document.getElementById("showRegion");
  const showLatent = document.getElementById("showLatent");
  const metaDiv = document.getElementById("meta");
  const plotDiv = document.getElementById("plot");
  const popup = document.getElementById("popup");

  // ---- palette ----
  const VORONOI_LINE = "rgba(220,30,40,0.85)";
  const SEED_RANK_LINE = ["rgba(214,39,40,0.95)", "rgba(255,140,0,0.95)", "rgba(218,165,32,0.95)"];
  const SEED_RANK_FILL = ["rgba(214,39,40,0.22)", "rgba(255,140,0,0.20)", "rgba(255,215,0,0.20)"];
  const SEED_OTHER_LINE = "rgba(130,130,130,0.75)";
  const SEED_OTHER_FILL = "rgba(150,150,150,0.10)";
  const MARGIN_LINE = "rgba(214,39,40,0.85)";
  const MARGIN_FILL = "rgba(214,39,40,0.10)";
  const SEED_LINK = "rgba(110,110,110,0.9)";
  const RING_OUTER = "rgba(60,90,160,0.45)";
  const RING_INNER = "rgba(60,90,160,0.60)";
  // Dropped constraints, when shown, must recede: they are the ones the MTL
  // decided are not binding, and at 3-20% keep rates they outnumber the rest.
  const RING_DROPPED = "rgba(120,120,120,0.16)";
  // Hovering one VP dims the rest of the bundle rather than hiding it, so the
  // highlighted constraint is read against its neighbours instead of alone.
  const RING_OUTER_DIM = "rgba(60,90,160,0.07)";
  const RING_INNER_DIM = "rgba(60,90,160,0.10)";
  const RING_DROPPED_DIM = "rgba(120,120,120,0.05)";
  const HL_RING = "rgba(20,40,140,0.95)";
  const HL_RING_DROPPED = "rgba(150,60,20,0.9)";
  const HL_TRACE_OUTER = "vp-constraint-highlight";
  const HL_TRACE_INNER = "vp-constraint-highlight-inner";
  const REGION_FILL = "rgba(20,120,90,0.20)";
  const REGION_LINE = "rgba(15,90,70,0.75)";
  const VP_MEASURED = "rgba(105,105,105,0.95)";
  const VP_LATENT = "rgba(140,140,140,0.85)";
  const VP_SPING = "rgba(30,110,220,0.95)";
  const ERR_LINK = "rgba(90,90,90,0.9)";
  const OK_GREEN = "rgba(34,150,90,1)";
  const BAD_RED = "rgba(214,39,40,1)";
  const STATUS_COLOR = { correct: "#2a8f5f", wrong: "#d33", failed: "#b0389a" };

  // Trace tags read back by the click handler. `curveNumber` is the only handle
  // Plotly gives, and the trace list is rebuilt every draw, so the mapping is
  // recorded per draw rather than assumed.
  let clickKind = {};
  let onPlotClick = () => {};
  let onPlotHover = () => {};
  let onPlotUnhover = () => {};

  // ---- small utilities (ported from cluster_world_map.js) ----
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
      // Negative t walks the ring clockwise in (lon, lat). Plotly's
      // `fill: "toself"` fills the side to the right of the walk on a
      // spherical path, so a counter-clockwise circle would fill the whole
      // globe *except* the disk -- which is invisible on an unfilled
      // constraint ring and catastrophic on the filled margin circle.
      const t = -2*Math.PI*i/n, ct = Math.cos(t), st = Math.sin(t);
      const x = cosR*c[0] + sinR*(ct*e1[0] + st*e2[0]);
      const y = cosR*c[1] + sinR*(ct*e1[1] + st*e2[1]);
      const z = cosR*c[2] + sinR*(ct*e1[2] + st*e2[2]);
      lats[i] = Math.asin(Math.max(-1, Math.min(1, z))) * 180/Math.PI;
      lons[i] = Math.atan2(y, x) * 180/Math.PI;
    }
    // Break the polyline at the antimeridian; otherwise Plotly draws a
    // horizontal smear right across the map.
    const outLat = [lats[0]], outLon = [lons[0]];
    for (let i = 1; i < lats.length; i++) {
      if (Math.abs(lons[i] - lons[i-1]) > 180) { outLat.push(null); outLon.push(null); }
      outLat.push(lats[i]); outLon.push(lons[i]);
    }
    return { lats: outLat, lons: outLon };
  }
  function compactSearch(s) { return String(s || "").toLowerCase().replace(/[^a-z0-9]/g, ""); }
  function isSubsequence(needle, haystack) {
    let j = 0;
    for (let i = 0; i < haystack.length && j < needle.length; i++) if (haystack[i] === needle[j]) j++;
    return j === needle.length;
  }
  function percentileIndex(p, n) {
    if (n === 0) return 0;
    const raw = (p/100) * (n-1);
    const lo = Math.floor(raw); const frac = raw - lo;
    let i;
    if (frac < 0.5) i = lo; else if (frac > 0.5) i = lo + 1; else i = (lo % 2 === 0) ? lo : lo + 1;
    return Math.max(0, Math.min(n-1, i));
  }
  function num(v, digits, unit) {
    if (v === null || v === undefined) return "—";
    return v.toFixed(digits) + (unit ? " " + unit : "");
  }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function targetLabel(t) {
    const e = t.error_km != null ? `${t.error_km.toFixed(0)} km` : "—";
    return `${t.target_id} (fold ${t.fold}) — ${t.status} — err=${e}`;
  }
  function searchScore(t, rawQuery) {
    const q = String(rawQuery || "").trim().toLowerCase();
    if (!q) return 0;
    const id = String(t.target_id || "").toLowerCase();
    const label = targetLabel(t).toLowerCase();
    const cq = compactSearch(q), cid = compactSearch(id), clabel = compactSearch(label);
    if (id === q) return 1000;
    if (id.startsWith(q)) return 900 - id.length;
    if (id.includes(q)) return 800 - id.indexOf(q);
    if (cq && cid === cq) return 700;
    if (cq && cid.startsWith(cq)) return 650 - cid.length;
    if (cq && cid.includes(cq)) return 600 - cid.indexOf(cq);
    if (label.includes(q)) return 500;
    if (cq && clabel.includes(cq)) return 400;
    if (cq.length >= 3 && isSubsequence(cq, clabel)) return 300 - cq.length;
    return -1;
  }

  // ---- target list ----
  let currentList = data.targets;
  function activeList() {
    const sc = statusSel.value;
    const q = targetSearch ? targetSearch.value : "";
    const rows = data.targets.map((t, i) => ({ t, i, score: searchScore(t, q) })).filter((r) => {
      const t = r.t;
      if (sc === "fail" && t.status === "correct") return false;
      if ((sc === "correct" || sc === "wrong" || sc === "failed") && t.status !== sc) return false;
      if (q.trim() && r.score < 0) return false;
      return true;
    });
    if (q.trim()) rows.sort((a, b) => (b.score - a.score) || (a.i - b.i));
    return rows.map((r) => r.t);
  }
  // pN means "the target at the Nth percentile of error distance", so p5 is a
  // near-miss and p95 is a disaster. The ranking is recomputed here rather than
  // read off the dropdown's positions: the two agree while the list is in its
  // payload order, but a Status filter can leave a prefix that is no longer the
  // whole distribution, and the percentile must follow the list on screen.
  //
  // `failed` rows are excluded rather than pooled. On a fallback the benchmark
  // fills `error_km` with the *Shortest-Ping VP's* error, not the method's, so
  // mixing the two answers a question about neither -- on as01 `vanilla_cbg`
  // its 106 fallbacks pull p50 from 75.1 km down to 51.1 km. Excluding them
  // makes these percentiles agree with `topn_accuracy.csv`'s `error_km_*`,
  // which is computed over solved rows for the same reason. Filter Status to
  // `failed` to walk those instead; the control then falls back to the
  // dropdown's own severity order, since there is no method error to rank on.
  function errorRanking() {
    return currentList
      .map((t, i) => ({ i, e: t.error_km, ok: t.status !== "failed" }))
      .filter((r) => r.ok && r.e !== null && r.e !== undefined)
      .sort((a, b) => (a.e - b.e) || (a.i - b.i));
  }
  function percentileTargetIndex(p) {
    const ranked = errorRanking();
    if (!ranked.length) return percentileIndex(p, currentList.length);
    return ranked[percentileIndex(p, ranked.length)].i;
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
    if (pctSel.value !== "") idx = percentileTargetIndex(+pctSel.value);
    else if (prev) {
      const p = currentList.findIndex((t) => t.target_id === prev.target_id && t.fold === prev.fold);
      if (p >= 0) idx = p;
    }
    targetSel.value = String(idx);
    if (targetCount) targetCount.textContent = `${currentList.length}/${data.targets.length}`;
  }

  // ---- click popup ----
  function hidePopup() { popup.style.display = "none"; }
  function showPopup(ev, title, rows) {
    const body = rows.map(([k, v]) => `<span class="k">${esc(k)}:</span> ${v}`).join("<br>");
    popup.innerHTML = `<span class="close" title="close">×</span>` +
                      `<div class="ttl">${esc(title)}</div>${body}`;
    popup.style.display = "block";
    const mx = (ev && ev.event ? ev.event.pageX : 0) + 14;
    const my = (ev && ev.event ? ev.event.pageY : 0) + 14;
    popup.style.left = Math.max(8, Math.min(mx, window.scrollX + window.innerWidth - 360)) + "px";
    popup.style.top = my + "px";
    const c = popup.querySelector(".close");
    if (c) c.addEventListener("click", hidePopup);
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hidePopup(); });

  function targetPopupRows(t) {
    const pct = t.n_total ? (100 * t.n_measured / t.n_total).toFixed(1) : "—";
    const seed = seeds[t.tg_seed_id];
    return [
      ["target", `<b>${esc(t.target_id)}</b>`],
      ["fold", esc(t.fold)],
      ["coords", `${t.true[0]}, ${t.true[1]}`],
      ["true seed", seed ? `#${seed.id} (cell ${esc(seed.cell_id)}, ${seed.n_targets} targets)` : "—"],
      ["offset to seed", num(t.cell_offset_km, 1, "km")],
      ["measured VPs", `${t.n_measured}/${t.n_total} (${pct}%)`],
      ["status", `<b style="color:${STATUS_COLOR[t.status]}">${t.status}</b>`],
      ["VP proximity", `<b>${esc(t.proximity || "—")}</b>`],
      ["min-RTT inflation", num(t.min_inflation, 2, "×")],
    ];
  }
  function predPopupRows(t) {
    return [
      ["error distance", `<b>${num(t.error_km, 1, "km")}</b>`],
      ["seeds crossed", t.seeds_crossed === null || t.seeds_crossed === undefined
        ? "—" : `<b>${t.seeds_crossed}</b>`],
      ["predicted seed", t.top_seeds.length ? `#${t.top_seeds[0]}` : "—"],
      ["true seed", t.tg_seed_id === null ? "—" : `#${t.tg_seed_id}`],
      ["margin (½ top1↔top2)", num(t.margin_km, 1, "km")],
      ["coords", t.pred ? `${t.pred[0]}, ${t.pred[1]}` : "—"],
    ];
  }
  function vpPopupRows(vpId, t, obs) {
    const meta = vps[vpId] || [];
    const ring = (t.rings || []).find((r) => r[0] === vpId);
    const constraint = !ring
      ? "— (no LTD constraint)"
      : (ring[3] === 1
          ? `kept · upper ${ring[1].toFixed(0)} km` + (ring[2] > 0 ? ` · inner ${ring[2].toFixed(0)} km` : "")
          : `dropped by the inclusion filter · upper ${ring[1].toFixed(0)} km`);
    return [
      ["vp", `<b>${esc(vpId)}</b>`],
      ["coords", meta.length ? `${meta[0]}, ${meta[1]}` : "—"],
      ["asn", esc(meta[2] || "—")],
      ["country", esc(meta[3] || "—")],
      ["RTT", obs ? num(obs[1], 2, "ms") : "— (latent: no observation for this target)"],
      ["shortest_ping", `<b>${vpId === t.sping_vp_id}</b>`],
      ["min-RTT inflation", obs ? num(obs[2], 2, "×") : "—"],
      ["constraint", constraint],
    ];
  }

  // ---- trace builders ----
  function ringsToTrace(rings, fillColor, lineColor, width, dash) {
    const lat = [], lon = [];
    for (const r of rings) {
      for (const [la, lo] of r) { lat.push(la); lon.push(lo); }
      lat.push(null); lon.push(null);
    }
    const line = { color: lineColor, width };
    if (dash) line.dash = dash;
    return { type: "scattergeo", mode: "lines", lat, lon, fill: "toself",
             fillcolor: fillColor, line, hoverinfo: "skip" };
  }
  // No `text`: every mark on this map reports through the click popup, so a
  // second, differently-styled Plotly tooltip on top of it is just noise.
  // `hoverinfo: "none"` rather than `"skip"` keeps the trace clickable.
  function connectorTrace(from, to, color) {
    return { type: "scattergeo", mode: "lines",
             lat: [from[0], to[0]], lon: [from[1], to[1]],
             line: { width: 1.6, color, dash: "dash" },
             hoverinfo: "none", showlegend: false };
  }

  // ---- main draw ----
  function draw() {
    hidePopup();
    if (currentList.length === 0) {
      metaDiv.innerHTML = "<i>no targets match the current filters</i>";
      Plotly.react(plotDiv, [], { geo: { projection: { type: projSel.value } } }, { responsive: true });
      return;
    }
    const tIdx = +targetSel.value || 0;
    const t = currentList[tIdx];
    const maxR = +maxRSel.value;
    const traces = [];
    clickKind = {};
    // `curveNumber` is the only handle Plotly hands back, and the trace list is
    // rebuilt from scratch every draw, so the indices the hover handler needs
    // are recorded here rather than assumed.
    let outerIdx = -1, innerIdx = -1, droppedIdx = -1, hlOuterIdx = -1, hlInnerIdx = -1;

    // 1) nearest-seed Voronoi partition over the continental US — the
    //    classifier's own top-1 decision boundary, so it sits under everything.
    if (showVoronoi.checked && voronoiCells.length) {
      traces.push(Object.assign(
        ringsToTrace(voronoiCells, "rgba(0,0,0,0)", VORONOI_LINE, 0.8, "dash"),
        { name: `Voronoi cells (${voronoiCells.length})` }));
    }

    // 2) seed regions, as the grid cells they are. The prediction's top-3
    //    candidates are coloured; everything else is grey context.
    if (showSeeds.checked && seeds.length) {
      const rank = new Map();
      (t.top_seeds || []).forEach((sid, i) => { if (i < 3) rank.set(sid, i); });
      const other = [];
      for (const s of seeds) if (!rank.has(s.id)) other.push(s.ring);
      if (other.length) {
        traces.push(Object.assign(
          ringsToTrace(other, SEED_OTHER_FILL, SEED_OTHER_LINE, 0.6),
          { name: `seed regions (${other.length})` }));
      }
      for (const [sid, i] of [...rank.entries()].sort((a, b) => b[1] - a[1])) {
        const s = seeds[sid];
        if (!s) continue;
        traces.push(Object.assign(
          ringsToTrace([s.ring], SEED_RANK_FILL[i], SEED_RANK_LINE[i], 1.6),
          { name: `top-${i + 1} seed #${sid}` }));
      }
    }

    // 3) the margin: half the geodesic from top-1 to top-2. Any coordinate
    //    inside it snaps to top-1, so this is the discriminative range.
    if (showMargin.checked && t.margin_km && t.top_seeds.length >= 2) {
      const s1 = seeds[t.top_seeds[0]], s2 = seeds[t.top_seeds[1]];
      if (s1 && s2) {
        const ring = ringLatLon(s1.lat, s1.lon, t.margin_km, 96);
        traces.push({ type: "scattergeo", mode: "lines", lat: ring.lats, lon: ring.lons,
          fill: "toself", fillcolor: MARGIN_FILL,
          line: { width: 1.2, color: MARGIN_LINE, dash: "dash" },
          name: `margin ${t.margin_km.toFixed(0)} km`, hoverinfo: "skip" });
        traces.push({ type: "scattergeo", mode: "lines",
          lat: [s1.lat, s2.lat], lon: [s1.lon, s2.lon],
          line: { width: 1.2, color: SEED_LINK, dash: "dot" },
          name: `top1↔top2 (${(2 * t.margin_km).toFixed(0)} km)`,
          hoverinfo: "skip" });
      }
    }

    // 4) LTD constraints. Disk LTDs leave lower_km at 0; annulus LTDs get a
    //    dashed inner ring as well.
    // A disk that fully contains another is redundant — the contained one is
    // strictly tighter — so every MTL drops it via
    // `geometry.filter_redundant_outer_disks` before intersecting, and records
    // the survivors. `r[3]` is that flag, read back from `mtl_participants[]`
    // rather than recomputed. Keep rates run 3-20%, so showing the dropped set
    // by default buries the handful of constraints that decide the answer.
    const fullEarthKm = Math.PI * EARTH;
    let shown = 0, hidden = 0, dropped = 0;
    if (showRings.checked) {
      const oLat = [], oLon = [], iLat = [], iLon = [], dLat = [], dLon = [];
      for (const r of (t.rings || [])) {
        const coord = vps[r[0]];
        if (!coord) continue;
        const isKept = r[3] === 1;
        if (!isKept && keptOnly.checked) { dropped++; continue; }
        if (r[1] >= fullEarthKm || (maxR > 0 && r[1] >= maxR)) { hidden++; continue; }
        const o = ringLatLon(coord[0], coord[1], r[1], 96);
        if (!isKept) {
          dropped++;
          dLat.push(...o.lats, null); dLon.push(...o.lons, null);
          continue;
        }
        shown++;
        oLat.push(...o.lats, null); oLon.push(...o.lons, null);
        if (isAnnulus && r[2] > 0) {
          const inn = ringLatLon(coord[0], coord[1], r[2], 96);
          iLat.push(...inn.lats, null); iLon.push(...inn.lons, null);
        }
      }
      // Dropped first, so the binding constraints draw over them.
      if (dLat.length) {
        droppedIdx = traces.length;
        traces.push({ type: "scattergeo", mode: "lines", lat: dLat, lon: dLon,
          line: { width: 0.6, color: RING_DROPPED }, hoverinfo: "skip",
          name: `dropped by inclusion filter (${dropped})` });
      }
      if (oLat.length) {
        outerIdx = traces.length;
        traces.push({ type: "scattergeo", mode: "lines", lat: oLat, lon: oLon,
          line: { width: 1.0, color: RING_OUTER }, hoverinfo: "skip",
          name: isAnnulus ? `outer bounds (${shown})` : `LTD disks (${shown})` });
      }
      if (iLat.length) {
        innerIdx = traces.length;
        traces.push({ type: "scattergeo", mode: "lines", lat: iLat, lon: iLon,
          line: { width: 0.9, color: RING_INNER, dash: "dash" }, hoverinfo: "skip",
          name: "inner bounds" });
      }
    }

    // Empty placeholders, restyled in place on hover. Allocating them once per
    // draw rather than adding a trace per hover keeps `Plotly.react` out of the
    // hover path, which is what makes the highlight feel instant.
    // Named, though hidden from the legend, so a test can tell them apart from
    // the other `showlegend: false` traces (the error connector, region holes).
    hlOuterIdx = traces.length;
    traces.push({ type: "scattergeo", mode: "lines", lat: [], lon: [],
      line: { width: 2.4, color: HL_RING }, hoverinfo: "skip",
      showlegend: false, name: HL_TRACE_OUTER });
    hlInnerIdx = traces.length;
    traces.push({ type: "scattergeo", mode: "lines", lat: [], lon: [],
      line: { width: 1.8, color: HL_RING, dash: "dash" }, hoverinfo: "skip",
      showlegend: false, name: HL_TRACE_INNER });

    // 5) MTL feasible region. Holes are a separate ocean-coloured fill drawn on
    //    top, which is how Plotly's `toself` can express a polygon with a hole.
    if (showRegion.checked && t.region) {
      const fillRings = [], holeRings = [];
      for (const ring of (t.region.rings || [])) {
        if (ring.outer && ring.outer.length) fillRings.push(ring.outer);
        for (const h of (ring.holes || [])) if (h.length) holeRings.push(h);
      }
      if (fillRings.length) traces.push(Object.assign(
        ringsToTrace(fillRings, REGION_FILL, REGION_LINE, 1.3),
        { name: `feasible region (${t.region.kind})` }));
      // A hole is drawn as an outline over the same fill, not as a punched-out
      // patch. Plotly cannot express a real hole on a `toself` path, so the
      // previous version overpainted it in the ocean colour — which reads as a
      // lake wherever the region sits on land, and as nothing at all wherever
      // it sits on water. An outline says "excluded" without claiming a colour
      // the basemap does not have.
      if (holeRings.length) traces.push(Object.assign(
        ringsToTrace(holeRings, "rgba(0,0,0,0)", REGION_LINE, 1.0, "dot"),
        { name: `excluded (${holeRings.length})`, showlegend: false }));
    }

    // 6) VPs. Latent = in the roster but with no observation for THIS target,
    //    drawn hollow: Plotly markers cannot carry a dashed outline.
    const obsByVp = new Map();
    for (const o of (t.obs || [])) obsByVp.set(o[0], o);
    const mLat = [], mLon = [], mIds = [];
    const lLat = [], lLon = [], lIds = [];
    let spCoord = null;
    for (const vpId of Object.keys(vps)) {
      const c = vps[vpId];
      if (vpId === t.sping_vp_id) { spCoord = c; continue; }
      const o = obsByVp.get(vpId);
      if (o) {
        mLat.push(c[0]); mLon.push(c[1]); mIds.push(vpId);
      } else if (showLatent.checked) {
        lLat.push(c[0]); lLon.push(c[1]); lIds.push(vpId);
      }
    }
    if (lLat.length) {
      clickKind[traces.length] = "vp";
      traces.push({ type: "scattergeo", mode: "markers", lat: lLat, lon: lLon,
        customdata: lIds, hoverinfo: "none",
        marker: { size: 6, symbol: "circle-open", color: VP_LATENT, line: { width: 1.1, color: VP_LATENT } },
        name: `latent VPs (${lLat.length})` });
    }
    if (mLat.length) {
      clickKind[traces.length] = "vp";
      traces.push({ type: "scattergeo", mode: "markers", lat: mLat, lon: mLon,
        customdata: mIds, hoverinfo: "none",
        marker: { size: 7, color: VP_MEASURED, line: { width: 0.8, color: "white" } },
        name: `measured VPs (${mLat.length})` });
    }
    if (spCoord) {
      // Same click popup as every other VP; only the marker colour differs.
      clickKind[traces.length] = "vp";
      traces.push({ type: "scattergeo", mode: "markers", lat: [spCoord[0]], lon: [spCoord[1]],
        customdata: [t.sping_vp_id], hoverinfo: "none",
        marker: { size: 10, color: VP_SPING, line: { width: 1.2, color: "white" } },
        name: "shortest-ping VP" });
    }

    // 7) the error itself, then the two endpoints on top of it.
    if (t.pred) {
      clickKind[traces.length] = "pred";
      traces.push(connectorTrace(t.pred, t.true, ERR_LINK));
    }
    clickKind[traces.length] = "target";
    traces.push({ type: "scattergeo", mode: "markers", lat: [t.true[0]], lon: [t.true[1]],
      marker: { size: 13, color: "gold", symbol: "star", line: { color: "black", width: 1 } },
      hoverinfo: "none", name: "true target" });
    if (t.pred) {
      clickKind[traces.length] = "pred";
      traces.push({ type: "scattergeo", mode: "markers", lat: [t.pred[0]], lon: [t.pred[1]],
        marker: { size: 12, symbol: "triangle-up",
                  color: t.status === "correct" ? OK_GREEN : BAD_RED,
                  line: { color: "white", width: 1.4 } },
        hoverinfo: "none", name: "prediction" });
    }

    const layout = {
      geo: { projection: { type: projSel.value },
        showland: true, landcolor: "rgb(243,243,238)",
        showocean: true, oceancolor: "rgb(225,235,245)",
        showcountries: true, countrycolor: "rgb(190,190,190)",
        showsubunits: true, subunitcolor: "rgb(210,210,210)",
        coastlinecolor: "rgb(120,120,120)", coastlinewidth: 0.6,
        showframe: false },
      margin: { l: 0, r: 0, t: 30, b: 0 },
      legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)" },
      title: { text: `fold ${t.fold} · ${t.target_id} · ${t.status}`, font: { size: 14 } },
    };
    // `albers usa` has a fixed frame; only the free projections take a centre.
    if (projSel.value !== "albers usa") layout.geo.center = { lat: t.true[0], lon: t.true[1] };

    const pct = t.n_total ? (100 * t.n_measured / t.n_total).toFixed(1) : "—";
    const hiddenNote = hidden > 0 ? ` &nbsp;|&nbsp; <span style="color:#b00">${hidden} ring(s) hidden</span>` : "";
    metaDiv.innerHTML =
      `<span class="badge" style="background:${STATUS_COLOR[t.status]}">${t.status}</span> ` +
      `<b>${esc(t.target_id)}</b> &nbsp;|&nbsp; error=<b>${num(t.error_km, 1, "km")}</b>` +
      ` &nbsp;|&nbsp; seeds crossed=<b>${t.seeds_crossed == null ? "—" : t.seeds_crossed}</b>` +
      ` &nbsp;|&nbsp; margin=<b>${num(t.margin_km, 1, "km")}</b>` +
      ` &nbsp;|&nbsp; VP proximity=<b>${esc(t.proximity || "—")}</b>` +
      ` &nbsp;|&nbsp; min-RTT infl=<b>${num(t.min_inflation, 2, "×")}</b><br>` +
      `measured VPs ${t.n_measured}/${t.n_total} (${pct}%) · ` +
      `LTD constraints ${t.n_kept}/${(t.rings || []).length} kept by the inclusion ` +
      `filter, ${shown} drawn${dropped ? `, ${dropped} dropped` : ""} · ` +
      `region=${t.region ? t.region.kind : "none"} · ` +
      `top-3 seeds ${(t.top_seeds || []).map((s) => "#" + s).join(", ") || "—"} · ` +
      `rank ${tIdx + 1}/${currentList.length} by error` +
      (pctSel.value !== "" ? ` (p${pctSel.value} by error, solved only)` : "") + hiddenNote;

    Plotly.react(plotDiv, traces, layout, { responsive: true });

    // Hovering a VP marker pulls its own constraint out of the bundle. Drawn
    // even when `post-filter only` is hiding it, because "which disk did the
    // filter drop, and how big was it?" is exactly the question a VP that looks
    // close but contributed nothing raises — the highlight turns rust-coloured
    // to say the constraint is a dropped one.
    const ringByVp = new Map();
    for (const r of (t.rings || [])) ringByVp.set(r[0], r);

    function dimBundle(dim) {
      if (outerIdx >= 0) {
        Plotly.restyle(plotDiv, { "line.color": dim ? RING_OUTER_DIM : RING_OUTER }, [outerIdx]);
      }
      if (innerIdx >= 0) {
        Plotly.restyle(plotDiv, { "line.color": dim ? RING_INNER_DIM : RING_INNER }, [innerIdx]);
      }
      if (droppedIdx >= 0) {
        Plotly.restyle(plotDiv, { "line.color": dim ? RING_DROPPED_DIM : RING_DROPPED }, [droppedIdx]);
      }
    }
    function clearHighlight() {
      Plotly.restyle(plotDiv, { lat: [[], []], lon: [[], []] }, [hlOuterIdx, hlInnerIdx]);
    }

    onPlotHover = (ev) => {
      const pt = ev.points && ev.points[0];
      if (!pt || clickKind[pt.curveNumber] !== "vp") return;
      const vpId = pt.customdata;
      const coord = vps[vpId];
      const ring = ringByVp.get(vpId);
      // A latent VP, or one whose LTD failed, has no constraint to show.
      if (!coord || !ring) { clearHighlight(); dimBundle(false); return; }
      const colour = ring[3] === 1 ? HL_RING : HL_RING_DROPPED;
      const o = ringLatLon(coord[0], coord[1], ring[1], 96);
      Plotly.restyle(plotDiv, { lat: [o.lats], lon: [o.lons], "line.color": colour }, [hlOuterIdx]);
      if (isAnnulus && ring[2] > 0) {
        const inn = ringLatLon(coord[0], coord[1], ring[2], 96);
        Plotly.restyle(plotDiv, { lat: [inn.lats], lon: [inn.lons], "line.color": colour }, [hlInnerIdx]);
      } else {
        Plotly.restyle(plotDiv, { lat: [[]], lon: [[]] }, [hlInnerIdx]);
      }
      dimBundle(true);
    };
    onPlotUnhover = () => { clearHighlight(); dimBundle(false); };

    // `Plotly.react` keeps existing handlers, so they must be cleared or every
    // redraw stacks another copy onto the same div.
    if (plotDiv.removeAllListeners) {
      plotDiv.removeAllListeners("plotly_click");
      plotDiv.removeAllListeners("plotly_hover");
      plotDiv.removeAllListeners("plotly_unhover");
    }
    onPlotClick = (ev) => {
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      const kind = clickKind[pt.curveNumber];
      if (kind === "target") showPopup(ev, `Target ${t.target_id}`, targetPopupRows(t));
      else if (kind === "pred") showPopup(ev, "Prediction", predPopupRows(t));
      else if (kind === "vp") {
        const vpId = pt.customdata;
        showPopup(ev, `VP ${vpId}`, vpPopupRows(vpId, t, obsByVp.get(vpId)));
      }
    };
    plotDiv.on("plotly_click", onPlotClick);
    plotDiv.on("plotly_hover", onPlotHover);
    plotDiv.on("plotly_unhover", onPlotUnhover);
  }

  pctSel.addEventListener("change", () => {
    if (pctSel.value !== "") targetSel.value = String(percentileTargetIndex(+pctSel.value));
    draw();
  });
  statusSel.addEventListener("change", () => { populateTargets(); draw(); });
  targetSearch.addEventListener("input", () => { pctSel.value = ""; populateTargets(); draw(); });
  targetSel.addEventListener("change", draw);
  projSel.addEventListener("change", draw);
  maxRSel.addEventListener("change", draw);
  for (const el of [showVoronoi, showSeeds, showMargin, showRings, keptOnly,
                    showRegion, showLatent]) {
    el.addEventListener("change", draw);
  }

  populateTargets();
  draw();

  // Headless-test hooks. `typeof module` is undefined in a browser, so this is
  // dead code there; under node it lets a harness drive draw()/click without a
  // real DOM. See scripts/analysis/v3/tests/test_map_mtl_viewer.js.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      redraw: draw,
      repopulate: populateTargets,
      clickKind: () => clickKind,
      dispatchClick: (ev) => onPlotClick(ev),
      dispatchHover: (ev) => onPlotHover(ev),
      selected: () => currentList[+targetSel.value || 0],
      dispatchUnhover: () => onPlotUnhover(),
    };
  }
})();
