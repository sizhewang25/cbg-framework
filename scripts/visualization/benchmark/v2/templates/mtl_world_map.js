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

  // ---- VP-ring colors (full vs. dimmed-on-hover) + highlight ring ----
  // Base rings draw at the "full" alpha; hovering a VP dot restyles the base
  // ring trace(s) to the "dim" alpha and paints the hovered VP's ring(s) into
  // a dedicated bold highlight trace so a single constraint stands out.
  const RING_OUTER = "rgba(60,90,160,0.35)";
  const RING_OUTER_DIM = "rgba(60,90,160,0.06)";
  const RING_INNER = "rgba(60,90,160,0.55)";
  const RING_INNER_DIM = "rgba(60,90,160,0.10)";
  const HL_RING = "rgba(20,40,140,0.95)";  // saturated blue, contrasts the dim

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

  // ---- VP-dot hover highlighting ----
  // On hover over a VP marker: paint that VP's ring(s) into the bold highlight
  // trace and dim the base rings so the hovered constraint stands out. On
  // unhover: clear the highlight and restore full-alpha base rings. `ctx`
  // carries this draw's trace indices (rebound after every Plotly.react).
  function onVpHover(ev, ctx) {
    const pt = ev.points && ev.points[0];
    if (!pt || pt.curveNumber !== ctx.vpMarkerIdx) return;
    const pred = pt.customdata;
    if (!pred) return;
    const coord = vps[pred[0]];
    if (!coord) return;
    const o = ringLatLon(coord[0], coord[1], pred[1], 96);
    Plotly.restyle(plotDiv, { lat: [o.lats], lon: [o.lons] }, [ctx.hlOuterIdx]);
    if (ctx.hlInnerIdx >= 0) {
      if (isAnnulus && pred[2] > 0) {
        const ir = ringLatLon(coord[0], coord[1], pred[2], 96);
        Plotly.restyle(plotDiv, { lat: [ir.lats], lon: [ir.lons] }, [ctx.hlInnerIdx]);
      } else {
        Plotly.restyle(plotDiv, { lat: [[]], lon: [[]] }, [ctx.hlInnerIdx]);
      }
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
    // Trace indices captured as traces are pushed, so the hover handler can
    // restyle the right ones regardless of which optional traces are present.
    let outerIdx = -1, innerIdx = -1, hlOuterIdx = -1, hlInnerIdx = -1, vpMarkerIdx = -1;

    // 1) per-VP great-circle rings (filtered by maxR, post-filter toggle, and
    // a hard cap covering the whole Earth — r >= π·R ≈ 20015 km — which has
    // no meaningful ring even with "show all" picked).
    // Prediction tuple: [vp_id, outer_km, inner_km, isKept, rtt_ms]. For disk
    // combos inner_km == 0 and only the outer ring is drawn.
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
      outerIdx = traces.length;
      traces.push({
        type: "scattergeo",
        mode: "lines",
        lat: outerLats, lon: outerLons,
        line: { width: 0.8, color: RING_OUTER },
        name: isAnnulus ? `outer rings (${kept})` : `LTD circles (${kept})`,
        hoverinfo: "skip",
      });
    }
    if (innerLats.length) {
      innerIdx = traces.length;
      traces.push({
        type: "scattergeo",
        mode: "lines",
        lat: innerLats, lon: innerLons,
        line: { width: 0.7, color: RING_INNER, dash: "dash" },
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

    // 1c) highlight ring traces — empty until a VP dot is hovered, then the
    //     hover handler restyles them with that VP's ring(s) on top of the
    //     dimmed base rings. Always present so their indices are stable.
    hlOuterIdx = traces.length;
    traces.push({
      type: "scattergeo", mode: "lines", lat: [], lon: [],
      line: { width: 2.2, color: HL_RING },
      hoverinfo: "skip", showlegend: false,
    });
    if (isAnnulus) {
      hlInnerIdx = traces.length;
      traces.push({
        type: "scattergeo", mode: "lines", lat: [], lon: [],
        line: { width: 1.8, color: HL_RING, dash: "dash" },
        hoverinfo: "skip", showlegend: false,
      });
    }

    // 2) per-VP markers (only those whose ring is shown). `customdata` carries
    //    the full prediction tuple so the hover handler can rebuild the ring.
    const mkLat = [], mkLon = [], mkText = [], mkPred = [];
    for (const pred of t.predictions) {
      if (!included(pred)) continue;
      const coord = vps[pred[0]];
      if (!coord) continue;
      mkLat.push(coord[0]); mkLon.push(coord[1]);
      mkPred.push(pred);
      const outer = pred[1], inner = pred[2], rtt = pred[4];
      let line = `VP ${pred[0]}`;
      line += rtt != null ? `<br>RTT = ${rtt.toFixed(2)} ms` : `<br>RTT = —`;
      line += `<br>outer = ${outer.toFixed(1)} km`;
      if (isAnnulus) {
        line += inner > 0 ? `<br>inner = ${inner.toFixed(1)} km` : `<br>inner = 0`;
      }
      line += `<br>${pred[3] ? "kept" : "dropped by pre-filter"}`;
      mkText.push(line);
    }
    vpMarkerIdx = traces.length;
    traces.push({
      type: "scattergeo",
      mode: "markers",
      lat: mkLat, lon: mkLon,
      text: mkText,
      customdata: mkPred,
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

    // Re-bind hover highlighting for this draw's trace layout. Remove prior
    // listeners first so they don't stack across Plotly.react calls — trace
    // indices shift as optional traces (rings, feasible region) come and go.
    if (plotDiv.removeAllListeners) {
      plotDiv.removeAllListeners("plotly_hover");
      plotDiv.removeAllListeners("plotly_unhover");
    }
    if (vpMarkerIdx >= 0) {
      const hoverCtx = { outerIdx, innerIdx, hlOuterIdx, hlInnerIdx, vpMarkerIdx };
      plotDiv.on("plotly_hover", (ev) => onVpHover(ev, hoverCtx));
      plotDiv.on("plotly_unhover", () => onVpUnhover(hoverCtx));
    }

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
