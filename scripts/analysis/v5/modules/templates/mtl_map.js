(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const EARTH = data.earth_radius_km;
  const vps = data.vps || {};
  const seeds = data.seeds || [];
  // The Voronoi cell of each seed, KEYED BY seed_id -- not v4's anonymous
  // ring list. That keying is what lets the map draw the cell verdict: the
  // TG's own cell against the cell the prediction fell in, which is exactly
  // what `cell_label` compares. v4 could only draw the partition as decoration
  // because it had no id to join on.
  const cells = data.cells || {};
  const cellFrameRing = data.cell_frame_ring || [];
  const cellMeta = data.cell_meta || {};
  const CELL_COLOR = data.cell_palette || {};
  // Site coordinates, as [lat, lon, seed_id]. A seed is the spherical centroid
  // of the sites complete linkage grouped, so where a cross sits between two
  // dots, two sites within one grid_km were merged into one serving region.
  const sites = data.sites || [];
  const tgGrids = data.tg_grids || [];
  // Grid geometry is shared, not repeated per TG: 399 TGs occupy ~18 grids,
  // so a run-level table is a few hundred rings where per-TG copies would be
  // ~10k. Every layer that draws a grid indexes this, so none of them can draw
  // the same grid two different ways.
  const grids = data.grids || {};
  const NSIDE = data.nside;
  const GRID_KM = data.grid_km;
  const MAX_RING = data.max_ring;
  // True when the density MTL reported on the very rung the map is scored at,
  // so the argmax grid and the prediction's grid are the SAME polygon. Without
  // this the viewer stacks two identical fills under two different labels.
  const regionIsPredGrid = data.region_is_pred_grid === true;
  const isAnnulus = data.mtl_kind === "annulus";
  // Shortest-Ping emits no LTD constraint and no feasible region, and its
  // prediction *is* the shortest-ping VP's own coordinate. Set in Python off
  // the method name rather than sniffed from the data here, so a CBG run that
  // happened to produce nothing is not mistaken for the baseline.
  const isBaseline = data.is_baseline === true;
  // A density MTL (Spotter's Eq. 2) answers with a probability field over a
  // HEALPix grid, not with a feasible set. There is still a region to draw --
  // the grid cell the argmax fell in, which Python recovers by re-binning the
  // stored prediction -- but it is a quantisation of the point estimate rather
  // than a feasible set, and there is no inclusion filter to report. Stated by
  // Python off the MTL's registry family, for the same reason `isBaseline` is:
  // an empty `region` alone cannot distinguish "this method has no region"
  // from "this target had none".
  const isDensity = data.region_mode === "density";

  const statusSel = document.getElementById("status");
  const pctSel = document.getElementById("pct");
  const targetSel = document.getElementById("target");
  const targetSearch = document.getElementById("targetSearch");
  const targetCount = document.getElementById("targetCount");
  const projSel = document.getElementById("proj");
  const maxRSel = document.getElementById("maxR");
  const cellStatusSel = document.getElementById("cellStatus");
  const sortSel = document.getElementById("sort");
  const showCells = document.getElementById("showCells");
  const showFrame = document.getElementById("showFrame");
  const showGrids = document.getElementById("showGrids");
  // `showNbhd`, not `showRings`: that id is the LTD constraint layer. Reusing
  // it would silently bind the two -- both would still draw, so nothing would
  // look broken.
  const showNbhd = document.getElementById("showNbhd");
  const showRings = document.getElementById("showRings");
  const keptOnly = document.getElementById("keptOnly");
  const showRegion = document.getElementById("showRegion");
  const showLatent = document.getElementById("showLatent");
  const metaDiv = document.getElementById("meta");
  const plotDiv = document.getElementById("plot");
  const popup = document.getElementById("popup");

  // ---- palette ----
  // Cell edges take `mapping.CELL_EDGE`, so the interactive map and the static
  // answer-space figure draw the partition in one ink.
  const CELL_LINE = "rgba(82,81,78,0.55)";
  const CELL_FRAME_LINE = "rgba(82,81,78,0.55)";
  const GRID_OTHER_LINE = "rgba(130,130,130,0.75)";
  const GRID_OTHER_FILL = "rgba(150,150,150,0.10)";
  const SITE_INK = "rgba(11,11,11,0.75)";

  // The ring ramp arrives from Python (`figure_outcome_bars.SEGMENT_INK`)
  // rather than being copied here, so the map and the outcome bars cannot
  // drift into two greens for the same outcome. It is ordinal light-to-dark in
  // the direction the neighbourhood nests.
  const STATUS_COLOR = data.palette || {};
  function hexToRgba(hex, alpha) {
    const h = String(hex || "#888").replace("#", "");
    const v = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
    const r = parseInt(v.slice(0, 2), 16), g = parseInt(v.slice(2, 4), 16),
          b = parseInt(v.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  // The ramp was validated for adjacent AREAS, not for text. #79bf9b is 2.2:1
  // against white and #d8d7cf is 1.44:1, so a badge with hard-coded white text
  // would be unreadable on the two lightest statuses. Pick the ink from the
  // background's relative luminance instead of dropping the palette.
  function inkOn(hex) {
    const h = String(hex || "#888").replace("#", "");
    const v = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
    const lin = [0, 2, 4].map((i) => {
      const c = parseInt(v.slice(i, i + 2), 16) / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    const L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    return L > 0.45 ? "#0b0b0b" : "#ffffff";
  }
  const RING_FILL = [0, 1, 2].map((k) => hexToRgba(STATUS_COLOR["ring" + k], 0.22));
  const RING_LINE = [0, 1, 2].map((k) => hexToRgba(STATUS_COLOR["ring" + k], 0.95));
  // Outline only, no fill: the prediction's cell frequently coincides with
  // ring 0 or one of the ring-1 cells, and a second fill there just darkens the
  // square into a colour that matches no legend swatch.
  const PRED_GRID_LINE = "rgba(11,11,11,0.90)";
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
  // Kept constraints whose RTT is past the LTD's `cutoff_rtt`. Their band is
  // the sentinel extension, not the fit, so they get their own ink rather
  // than hiding in the blue bundle.
  const RING_PAST = "rgba(200,110,20,0.80)";
  const RING_PAST_DIM = "rgba(200,110,20,0.12)";
  const HL_RING_PAST = "rgba(170,80,0,0.95)";
  // `per_vp` (Octant), `pooled` (Spotter), or null when the LTD has no cutoff.
  const ltdCutoff = data.ltd_cutoff || null;
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
  // A symbol channel beside the colour: the ramp is four greens, and a reader
  // should not have to tell #17724a from #0b4d2c to know a prediction missed.
  const STATUS_SYMBOL = {
    ring0: "triangle-up", ring1: "triangle-up", ring2: "triangle-up",
    beyond: "x-thin-open", failed: "circle-open",
  };

  // Trace tags read back by the click handler. `curveNumber` is the only handle
  // Plotly gives, and the trace list is rebuilt every draw, so the mapping is
  // recorded per draw rather than assumed.
  let markKind = {};
  let ringIdx = { ring0: -1, ring1: -1, ring2: -1, predGrid: -1 };
  // Which trace is the TG's cell and which the prediction's, recorded for
  // the same reason `ringIdx` is: the trace list is rebuilt every draw.
  let cellIdx = { tg: -1, pred: -1 };
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
  // Cell ids arrive as numbers in `ring_cells` and as strings in the `cells`
  // table (JSON object keys). Missing ids are dropped rather than pushed as
  // `undefined`, which would put a NaN into the trace and silently blank it.
  function gridRingsFor(ids) {
    const out = [];
    for (const id of (ids || [])) {
      const ring = grids[String(id)];
      if (ring && ring.length) out.push(ring);
    }
    return out;
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
    const e = t.pred_dist_to_tg_km != null ? `${t.pred_dist_to_tg_km.toFixed(0)} km` : "—";
    return `${t.tg_id} (fold ${t.fold}) — ${t.status} — err=${e}`;
  }
  function searchScore(t, rawQuery) {
    const q = String(rawQuery || "").trim().toLowerCase();
    if (!q) return 0;
    const id = String(t.tg_id || "").toLowerCase();
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

  // ---- shape the control bar to what the method can actually have ----
  //
  // The control is hidden *and* disabled, never removed: all seven checkbox
  // handles are dereferenced at load and re-read on every draw, so deleting one
  // from the shell throws and blanks the page. Disabling is also what makes the
  // state observable to a test and stops the control being reached by keyboard.
  function hideControl(id) {
    const node = document.getElementById(id);
    if (!node) return;
    node.disabled = true;
    const label = node.closest ? node.closest("label") : null;
    (label || node).style.display = "none";
  }
  function dropStatusOption(value) {
    const opt = statusSel.querySelector
      ? statusSel.querySelector(`option[value="${value}"]`)
      : null;
    if (opt && opt.remove) opt.remove();
  }
  if (isDensity && !isBaseline) {
    // The annuli stay -- they are the per-landmark constraints and still worth
    // reading -- and `showRegion` now has the argmax grid to toggle. Only
    // `post-filter only` goes: it would filter by an inclusion filter that
    // never ran, so it reads as broken rather than as inapplicable.
    hideControl("keptOnly");
  }
  if (isBaseline) {
    for (const id of ["showRings", "keptOnly", "showRegion", "maxR"]) hideControl(id);
    // `failed` means a fallback, and Shortest-Ping has no pipeline to fall back
    // in: `io.solved_mask` reads its all-`BASELINE` frame as wholly solved, so
    // the filter would always come back empty and `wrong + failed` is just
    // `wrong` under another name.
    dropStatusOption("failed");
    const hint = document.getElementById("hoverHint");
    if (hint) hint.textContent = "· hover any marker for detail";
  }

  // ---- target list ----
  let currentList = data.tgs;
  function activeList() {
    const sc = statusSel.value;
    const q = targetSearch ? targetSearch.value : "";
    const rows = data.tgs.map((t, i) => ({ t, i, score: searchScore(t, q) })).filter((r) => {
      const t = r.t;
      // Cumulative where that is the useful cut: "within 1 ring" is the
      // tolerance the Euler figure sweeps, not a single bucket.
      if (sc === "ring0" && t.status !== "ring0") return false;
      if (sc === "within1" && !(t.status === "ring0" || t.status === "ring1")) return false;
      if (sc === "within2" && !(t.status === "ring0" || t.status === "ring1" || t.status === "ring2")) return false;
      if (sc === "missed" && t.status === "ring0") return false;
      if ((sc === "beyond" || sc === "failed") && t.status !== sc) return false;
      // The cell axis, filtered INDEPENDENTLY of the grid axis, which is the
      // point of having two. Crossing them is how the interesting cases are
      // found: "right serving region but 14 grids out" is a different failure
      // from "wrong serving region and 1 grid out", and neither axis alone
      // separates them.
      const cs = cellStatusSel ? cellStatusSel.value : "all";
      if (cs !== "all" && t.cell_label !== cs) return false;
      if (q.trim() && r.score < 0) return false;
      return true;
    });
    if (q.trim()) rows.sort((a, b) => (b.score - a.score) || (a.i - b.i));
    else sortRows(rows);
    return rows.map((r) => r.t);
  }
  // The payload arrives sorted by error distance (see `build_payload`). The
  // other two orders are recomputed here rather than shipped as extra columns.
  // `failed` always sinks to the bottom whatever the key: on a fallback every
  // one of these numbers describes the Shortest-Ping VP rather than the
  // method, so ranking it against the rest answers a question about neither.
  function sortRows(rows) {
    const key = sortSel ? sortSel.value : "error";
    if (key === "error") return;
    const of = (t) => {
      if (key === "offset") return t.grid_offset < 0 ? Infinity : t.grid_offset;
      return t.pred_dist_to_seed_km === null || t.pred_dist_to_seed_km === undefined
        ? Infinity : t.pred_dist_to_seed_km;
    };
    rows.sort((a, b) => {
      const fa = a.t.status === "failed" ? 1 : 0, fb = b.t.status === "failed" ? 1 : 0;
      return (fa - fb) || (of(a.t) - of(b.t)) || (a.i - b.i);
    });
  }
  // pN means "the target at the Nth percentile of error distance", so p5 is a
  // near-miss and p95 is a disaster. The ranking is recomputed here rather than
  // read off the dropdown's positions: the two agree while the list is in its
  // payload order, but a Status filter can leave a prefix that is no longer the
  // whole distribution, and the percentile must follow the list on screen.
  //
  // `beyond` rows ARE included: they carry a real, large `pred_dist_to_tg_km`, and they
  // are precisely what a p95 lookup should land on. Only `failed` is excluded.
  //
  // `failed` rows are excluded rather than pooled. On a fallback the benchmark
  // fills `pred_dist_to_tg_km` with the *Shortest-Ping VP's* error, not the method's, so
  // mixing the two answers a question about neither -- on as01 `vanilla_cbg`
  // its 106 fallbacks pull p50 from 75.1 km down to 51.1 km. Excluding them
  // makes these percentiles agree with `accuracy.csv`'s distance percentiles,
  // which is computed over solved rows for the same reason. Filter Status to
  // `failed` to walk those instead; the control then falls back to the
  // dropdown's own severity order, since there is no method error to rank on.
  function errorRanking() {
    return currentList
      .map((t, i) => ({ i, e: t.pred_dist_to_tg_km, ok: t.status !== "failed" }))
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
      const p = currentList.findIndex((t) => t.tg_id === prev.tg_id && t.fold === prev.fold);
      if (p >= 0) idx = p;
    }
    targetSel.value = String(idx);
    if (targetCount) targetCount.textContent = `${currentList.length}/${data.tgs.length}`;
  }

  // ---- click popup ----
  function hidePopup() { popup.style.display = "none"; }
  function showPopup(ev, title, rows) {
    const body = rows.map(([k, v]) => `<span class="k">${esc(k)}:</span> ${v}`).join("<br>");
    popup.innerHTML = `<div class="ttl">${esc(title)}</div>${body}`;
    popup.style.display = "block";
    const mx = (ev && ev.event ? ev.event.pageX : 0) + 14;
    const my = (ev && ev.event ? ev.event.pageY : 0) + 14;
    popup.style.left = Math.max(8, Math.min(mx, window.scrollX + window.innerWidth - 360)) + "px";
    popup.style.top = my + "px";
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hidePopup(); });

  const seedById = new Map(seeds.map((s) => [s.id, s]));
  function targetPopupRows(t) {
    const pct = t.n_total ? (100 * t.n_measured / t.n_total).toFixed(1) : "—";
    const seed = seedById.get(t.tg_seed_id);
    const rows = [
      ["TG", `<b>${esc(t.tg_id)}</b>`],
      ["fold", esc(t.fold)],
      ["coords", `${t.true[0]}, ${t.true[1]}`],
      ["TG grid", `#${t.tg_grid} · nside ${NSIDE} · ${GRID_KM} km`],
      // The quantization floor: no estimator scores better than this against
      // the grid centre, however good it is.
      ["offset to grid centre", num(t.tg_dist_to_grid_centre_km, 1, "km")],
      ["TG cell", `seed #${t.tg_seed_id}` +
        (seed ? ` · ${seed.n_sites} site(s), ${seed.n_tgs} TG(s)` +
          // Only a traffic-weighted run scores fewer TGs than its mesh holds.
          (seed.n_tgs_scored != null && seed.n_tgs_scored !== seed.n_tgs
            ? `, ${seed.n_tgs_scored} scored` : "") : "")],
      ["offset to its seed", num(t.tg_dist_to_seed_km, 1, "km")],
      ["measured VPs", `${t.n_measured}/${t.n_total} (${pct}%)`],
      ["min-RTT inflation", num(t.min_inflation, 2, "×")],
    ];
    return rows;
  }
  const VERDICT_TEXT = {
    ring0: "in the TG's own grid",
    ring1: "one of its immediate neighbour grids",
    ring2: "the second ring of grids out",
    beyond: `further out than ${MAX_RING} grids`,
    failed: "no prediction of the method's own",
  };
  const CELL_TEXT = {
    correct: "the TG's own serving region",
    wrong: "another seed's serving region",
    unanswered: "no prediction to place",
  };
  function statusBadge(status) {
    const bg = STATUS_COLOR[status] || "#888";
    return `<span class="badge" style="background:${bg};color:${inkOn(bg)}">` +
      `${esc(status)}</span>`;
  }
  function cellBadge(label) {
    const bg = CELL_COLOR[label] || "#888";
    return `<span class="badge" style="background:${bg};color:${inkOn(bg)}">` +
      `${esc(label)}</span>`;
  }
  // The grid offset, spelled out. `beyond` is a bucket covering 3 to 68 grids
  // out on these runs, so the bucket name alone is the one thing a case viewer
  // must not stop at -- `classify` keeps the exact number for precisely this.
  // Header fragment: how many of this TG's constraints are past the cutoff.
  // `past` is the drawn kept count; the total covers the dropped ones too.
  function cutoffNote(t, past) {
    if (!ltdCutoff) return "";
    const total = (t.rings || []).filter((r) => r[5] === 1).length;
    const pooled = ltdCutoff.scope === "pooled"
      ? ` (pooled ${num((ltdCutoff.pooled_ms_by_fold || {})[String(t.fold)], 1, "ms")})`
      : "";
    return `past LTD cutoff${pooled} ${total}` + (total ? `, ${past} drawn` : "") + " · ";
  }
  function offsetText(t) {
    if (t.grid_offset === null || t.grid_offset === undefined || t.grid_offset < 0) {
      return "— (no prediction)";
    }
    const n = t.grid_offset;
    return `<b>${n}</b> grid${n === 1 ? "" : "s"} out`;
  }
  // BOTH verdicts, always, and adjacent. Either one alone is a half-answer:
  // the grid axis bounds the distance but says nothing about direction, and
  // the cell axis is unbounded and will say `correct` at any distance.
  function predPopupRows(t) {
    return [
      ["grid verdict", statusBadge(t.status) + " " + esc(VERDICT_TEXT[t.status] || "")],
      ["grid offset", offsetText(t)],
      ["cell verdict", cellBadge(t.cell_label) + " " + esc(CELL_TEXT[t.cell_label] || "")],
      // Bolded harder once the prediction has left the neighbourhood: at that
      // point "how far" is the only question the tier can no longer answer.
      ["error distance", t.status === "beyond"
        ? `<b style="font-size:13px">${num(t.pred_dist_to_tg_km, 1, "km")}</b>`
        : `<b>${num(t.pred_dist_to_tg_km, 1, "km")}</b>`],
      ["to the TG's seed", num(t.pred_dist_to_seed_km, 1, "km")],
      ["prediction grid", t.pred_grid >= 0 ? `#${t.pred_grid}` : "— (no coordinate)"],
      ["TG grid", `#${t.tg_grid}`],
      ["prediction cell", t.pred_seed_id >= 0 ? `seed #${t.pred_seed_id}` : "—"],
      ["TG cell", `seed #${t.tg_seed_id}`],
      ["coords", t.pred ? `${t.pred[0]}, ${t.pred[1]}` : "—"],
    ];
  }
  // Shortest-Ping's estimate is the shortest-ping VP's own coordinate, so the
  // triangle and that VP are one mark and report as one panel. The inflation
  // here is this VP's, not `t.min_inflation` — that is a minimum over every
  // measured VP and belongs to the target's panel.
  function baselinePredPopupRows(t) {
    const vpId = t.sping_vp_id;
    const meta = vps[vpId] || [];
    const obs = (t.obs || []).find((o) => o[0] === vpId);
    return [
      ["shortest-ping VP", `<b>${esc(vpId || "—")}</b>`],
      ["coords", meta.length ? `${meta[0]}, ${meta[1]}` : "—"],
      ["asn", esc(meta[2] || "—")],
      ["country", esc(meta[3] || "—")],
      ["RTT", obs ? num(obs[1], 2, "ms") : "—"],
      ["RTT inflation at this VP", obs ? num(obs[2], 2, "×") : "—"],
    ].concat(predPopupRows(t).filter(([k]) => k !== "coords"));
  }
  function vpPopupRows(vpId, t, obs) {
    const meta = vps[vpId] || [];
    const ring = (t.rings || []).find((r) => r[0] === vpId);
    const constraint = !ring
      ? "— (no LTD constraint)"
      : (ring[3] === 1
          ? `kept · upper ${ring[1].toFixed(0)} km` + (ring[2] > 0 ? ` · inner ${ring[2].toFixed(0)} km` : "")
          : `dropped by the inclusion filter · upper ${ring[1].toFixed(0)} km`);
    // ring[4] = this VP's cutoff_rtt (ms), ring[5] = 1 when the RTT is past it.
    let cutoffText = null;
    if (ltdCutoff) {
      const scope = ltdCutoff.scope === "pooled" ? "pooled over all VPs" : "this VP's own";
      if (!ring || ring[4] === null || ring[4] === undefined) {
        cutoffText = "— (no fitted cutoff for this VP)";
      } else if (ring[5] === 1) {
        cutoffText = `<b>${ring[4].toFixed(1)} ms</b> (${scope}) · <b>RTT past it</b>: ` +
          "outer bound on the sentinel line (∥ 2/3·c), inner held at its cutoff value";
      } else {
        cutoffText = `${ring[4].toFixed(1)} ms (${scope}) · RTT within the fitted range`;
      }
    }
    const rows = [
      ["vp", `<b>${esc(vpId)}</b>`],
      ["coords", meta.length ? `${meta[0]}, ${meta[1]}` : "—"],
      ["asn", esc(meta[2] || "—")],
      ["country", esc(meta[3] || "—")],
      ["RTT", obs ? num(obs[1], 2, "ms") : "— (latent: no observation for this target)"],
      // This VP's own ratio. `min-RTT inflation` is the minimum over every
      // measured VP and belongs to the target's panel; naming both the same
      // invited reading a per-VP number as the target's routing statistic.
      ["RTT inflation at this VP", obs ? num(obs[2], 2, "×") : "—"],
    ];
    // On the baseline map the shortest-ping VP is drawn as the prediction and
    // is not in this bucket at all, so both of these would be constant.
    if (!isBaseline) {
      rows.splice(5, 0, ["shortest_ping", `<b>${vpId === t.sping_vp_id ? "yes" : "no"}</b>`]);
      rows.push(["constraint", constraint]);
      if (cutoffText !== null) rows.push(["LTD cutoff", cutoffText]);
    }
    return rows;
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
    markKind = {};
    // `curveNumber` is the only handle Plotly hands back, and the trace list is
    // rebuilt from scratch every draw, so what each trace *is* — and the
    // indices the highlight restyles — are recorded here rather than assumed.
    let outerIdx = -1, innerIdx = -1, droppedIdx = -1, hlOuterIdx = -1, hlInnerIdx = -1;
    let pastOuterIdx = -1, pastInnerIdx = -1;
    // Recorded so the headless harness can assert on the ring layers by index
    // rather than by matching a name like `ring 1 · 7 neighbours`, which
    // changes with the cell.
    ringIdx = { ring0: -1, ring1: -1, ring2: -1, predGrid: -1 };

    // 1) the cell partition -- HALF THE VERDICT, not context. v4 drew this as
    //    a decorative overlay because it had retired the nearest-seed rule;
    //    v5 grades on it, so it is drawn as a graded layer and named as one.
    //
    //    Three traces, not one, because they say different things: every other
    //    cell as a hairline, the TG's own cell FILLED (the region a `correct`
    //    prediction has to land in), and the cell it actually fell in outlined
    //    when those differ. The two highlights get their own single-ring
    //    traces rather than being null-concatenated into the background trace
    //    -- a limb-clipped multi-ring `toself` fill is where Plotly's geo fill
    //    misbehaves, and a per-cell legend entry comes free.
    const tgCellRing = cells[String(t.tg_seed_id)];
    const predCellRing = t.pred_seed_id >= 0 ? cells[String(t.pred_seed_id)] : null;
    cellIdx = { tg: -1, pred: -1 };
    if (showCells.checked) {
      const others = [];
      for (const k of Object.keys(cells)) {
        const id = +k;
        if (id === t.tg_seed_id || id === t.pred_seed_id) continue;
        if (cells[k] && cells[k].length) others.push(cells[k]);
      }
      if (others.length) {
        traces.push(Object.assign(
          ringsToTrace(others, "rgba(0,0,0,0)", CELL_LINE, 0.7),
          { name: `serving cells (${others.length} others)` }));
      }
      // The prediction's cell first, so the truth's fill draws over it when
      // they are the same polygon and the reader sees one shape, not two.
      if (predCellRing && predCellRing.length && t.pred_seed_id !== t.tg_seed_id) {
        cellIdx.pred = traces.length;
        traces.push(Object.assign(
          ringsToTrace([predCellRing], hexToRgba(CELL_COLOR.wrong, 0.10),
                       hexToRgba(CELL_COLOR.wrong, 0.95), 2.0, "dot"),
          { name: `landed in seed #${t.pred_seed_id}'s cell` }));
      }
      if (tgCellRing && tgCellRing.length) {
        cellIdx.tg = traces.length;
        traces.push(Object.assign(
          ringsToTrace([tgCellRing], hexToRgba(CELL_COLOR.correct, 0.13),
                       hexToRgba(CELL_COLOR.correct, 0.95), 2.0),
          { name: `the TG's cell (seed #${t.tg_seed_id})` }));
      }
    }

    // 1b) the frame the cells were cut to. A RENDERING bound and nothing else:
    //     every cell would continue past it. Drawn because this viewer pans
    //     and offers an orthographic projection, so without the rectangle a
    //     reader sees the cells stop in mid-Atlantic and reads the stop as a
    //     cell edge -- the one misreading the partition must not invite.
    if (showFrame.checked && cellFrameRing.length) {
      traces.push(Object.assign(
        ringsToTrace([cellFrameRing], "rgba(0,0,0,0)", CELL_FRAME_LINE, 1.0, "dot"),
        { name: "cell frame (rendering bound — cells continue past it)" }));
    }

    // 2) the grid neighbourhood -- the grid verdict, drawn as the grids it is
    //    defined over. OUTERMOST FIRST so the inner rings composite on top of
    //    them; ring 2 is already a set difference in Python, so no fill is
    //    stacked on another.
    //
    //    CAPPED AT MAX_RING, and deliberately. `pred_dist_to_tg_grid` is
    //    uncapped and reaches 68 on these runs, but the disk at that offset is
    //    ~15,000 grids: undrawable, and meaningless as a shaded region. Past
    //    the second ring the offset is a number in the popup, never a polygon.
    //
    //    Ring 1 holds SEVEN grids rather than eight at the 24 base-face corner
    //    grids of every nside, and ring 2 is short to match. Lengths are read,
    //    never assumed.
    const nbhd = t.ring_grids || [[t.tg_grid], [], []];
    if (showNbhd.checked) {
      for (let k = Math.min(MAX_RING, nbhd.length - 1); k >= 0; k--) {
        const rings = gridRingsFor(nbhd[k]);
        if (!rings.length) continue;
        const label = k === 0
          ? "0 grids out · the TG's own grid"
          : `${k} grid${k === 1 ? "" : "s"} out (${rings.length} grid${rings.length === 1 ? "" : "s"})`;
        ringIdx["ring" + k] = traces.length;
        traces.push(Object.assign(
          ringsToTrace(rings, RING_FILL[k], RING_LINE[k], k === 0 ? 1.8 : 0.9),
          { name: label }));
      }
    }

    // 3) the rest of the occupied TG grids, as grey context. The truth's own
    //    grid is excluded: it is already drawn at ring 0, and two fills on one
    //    square make a colour that matches no legend swatch.
    //
    //    Fed from `data.tg_grids` rather than from the seeds. v4 read
    //    `seeds[].cell_id` here, which only worked because a v4 seed WAS a
    //    grid centre. A v5 seed is a complete-linkage centroid of sites and
    //    has no grid; it is the wrong object to enumerate grids with.
    if (showGrids.checked && tgGrids.length) {
      const other = [];
      for (const g of tgGrids) {
        if (g === t.tg_grid) continue;
        const ring = grids[String(g)];
        if (ring && ring.length) other.push(ring);
      }
      if (other.length) {
        traces.push(Object.assign(
          ringsToTrace(other, GRID_OTHER_FILL, GRID_OTHER_LINE, 0.6),
          { name: `TG grids (${other.length})` }));
      }
    }

    // 3b) the GRID the prediction actually fell in -- what the grid offset
    //     compares
    //     against. Outline only; see PRED_GRID_LINE. Skipped when the region
    //     layer is about to draw the very same polygon (a density MTL
    //     reporting on this exact rung), which `regionIsPredGrid` states.
    if (showNbhd.checked && t.pred_grid >= 0 && !(regionIsPredGrid && showRegion.checked && t.region)) {
      const ring = grids[String(t.pred_grid)];
      if (ring && ring.length) {
        ringIdx.predGrid = traces.length;
        traces.push(Object.assign(
          ringsToTrace([ring], "rgba(0,0,0,0)", PRED_GRID_LINE, 2.2, "dot"),
          { name: `prediction grid #${t.pred_grid}` }));
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
    let shown = 0, hidden = 0, dropped = 0, past = 0;
    if (showRings.checked) {
      const oLat = [], oLon = [], iLat = [], iLon = [], dLat = [], dLon = [];
      // Kept constraints past the LTD cutoff (`r[5]`) go to their own pair of
      // traces: their band is the sentinel extension, not the fit.
      const pLat = [], pLon = [], piLat = [], piLon = [];
      for (const r of (t.rings || [])) {
        const coord = vps[r[0]];
        if (!coord) continue;
        const isKept = r[3] === 1;
        if (!isKept) dropped++;
        if (!isKept && keptOnly.checked) continue;
        if (r[1] >= fullEarthKm || (maxR > 0 && r[1] >= maxR)) { hidden++; continue; }
        const o = ringLatLon(coord[0], coord[1], r[1], 96);
        if (!isKept) {
          dLat.push(...o.lats, null); dLon.push(...o.lons, null);
          continue;
        }
        shown++;
        const isPast = r[5] === 1;
        if (isPast) past++;
        (isPast ? pLat : oLat).push(...o.lats, null);
        (isPast ? pLon : oLon).push(...o.lons, null);
        if (isAnnulus && r[2] > 0) {
          const inn = ringLatLon(coord[0], coord[1], r[2], 96);
          (isPast ? piLat : iLat).push(...inn.lats, null);
          (isPast ? piLon : iLon).push(...inn.lons, null);
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
      // Last, so the extrapolated constraints are never buried under the fit.
      if (pLat.length) {
        pastOuterIdx = traces.length;
        traces.push({ type: "scattergeo", mode: "lines", lat: pLat, lon: pLon,
          line: { width: 1.4, color: RING_PAST }, hoverinfo: "skip",
          name: `past LTD cutoff (${past})` });
      }
      if (piLat.length) {
        pastInnerIdx = traces.length;
        traces.push({ type: "scattergeo", mode: "lines", lat: piLat, lon: piLon,
          line: { width: 1.1, color: RING_PAST, dash: "dash" }, hoverinfo: "skip",
          name: "inner bounds past LTD cutoff" });
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
        { name: isDensity
            ? (regionIsPredGrid
                // Same polygon as the prediction's cell, so it is drawn once
                // and named for both rather than stacked under two labels.
                ? `argmax grid = prediction grid #${t.pred_grid} (nside ${data.density_nside})`
                : `argmax grid (nside ${data.density_nside})`)
            : `feasible region (${t.region.kind})` }));
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

    // 5b) seeds, and the sites they were built from. `cell_label` is literally
    //     "is the prediction's nearest seed the TG's seed", so without the
    //     seeds on the map that verdict is asserted rather than shown. Same
    //     encoding as the static answer-space figure: sites are dots, seeds
    //     are crosses, and a cross sitting between two dots is two sites
    //     within one grid_km that were grouped into one serving region.
    if (showCells.checked && seeds.length) {
      // A site's 4th field is how many of its TGs this run scores; 0 marks a
      // traffic-weighted run's mesh-only site, drawn hollow. Payloads from
      // before the field existed have no 4th entry and every site is scored.
      const scoredSites = sites.filter((s) => s[3] == null || s[3] > 0);
      const meshOnly = sites.filter((s) => s[3] === 0);
      if (scoredSites.length) {
        traces.push({ type: "scattergeo", mode: "markers",
          lat: scoredSites.map((s) => s[0]), lon: scoredSites.map((s) => s[1]),
          marker: { size: 4, color: SITE_INK, line: { width: 0 } },
          hoverinfo: "skip", name: `sites (${scoredSites.length})` });
      }
      if (meshOnly.length) {
        traces.push({ type: "scattergeo", mode: "markers",
          lat: meshOnly.map((s) => s[0]), lon: meshOnly.map((s) => s[1]),
          marker: { size: 5, color: "rgba(0,0,0,0)", symbol: "circle-open",
                    line: { width: 1, color: SITE_INK } },
          hoverinfo: "skip", name: `mesh sites, no scored TG (${meshOnly.length})` });
      }

      const others2 = seeds.filter((s) => s.id !== t.tg_seed_id);
      if (others2.length) {
        traces.push({ type: "scattergeo", mode: "markers",
          lat: others2.map((s) => s.lat), lon: others2.map((s) => s.lon),
          marker: { size: 7, color: SITE_INK, symbol: "x-thin",
                    line: { width: 1.2, color: SITE_INK } },
          hoverinfo: "skip", name: `seeds (${others2.length})` });
      }
      const tg = seedById.get(t.tg_seed_id);
      if (tg) {
        traces.push({ type: "scattergeo", mode: "markers",
          lat: [tg.lat], lon: [tg.lon],
          marker: { size: 11, color: hexToRgba(CELL_COLOR.correct, 0.95),
                    symbol: "x-thin", line: { width: 2.2,
                    color: hexToRgba(CELL_COLOR.correct, 0.95) } },
          hoverinfo: "skip", name: `the TG's seed #${tg.id}` });
      }
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
      markKind[traces.length] = "vp";
      traces.push({ type: "scattergeo", mode: "markers", lat: lLat, lon: lLon,
        customdata: lIds, hoverinfo: "none",
        marker: { size: 6, symbol: "circle-open", color: VP_LATENT, line: { width: 1.1, color: VP_LATENT } },
        name: `latent VPs (${lLat.length})` });
    }
    if (mLat.length) {
      markKind[traces.length] = "vp";
      traces.push({ type: "scattergeo", mode: "markers", lat: mLat, lon: mLon,
        customdata: mIds, hoverinfo: "none",
        marker: { size: 7, color: VP_MEASURED, line: { width: 0.8, color: "white" } },
        name: `measured VPs (${mLat.length})` });
    }
    // On the baseline the prediction sits on this exact coordinate — it *is*
    // this VP — so drawing both stacks two marks on one point and lets Plotly
    // decide which one the hover reaches. The triangle stands for both there.
    if (spCoord && !isBaseline) {
      // Same panel as every other VP; only the marker colour differs.
      markKind[traces.length] = "vp";
      traces.push({ type: "scattergeo", mode: "markers", lat: [spCoord[0]], lon: [spCoord[1]],
        customdata: [t.sping_vp_id], hoverinfo: "none",
        marker: { size: 10, color: VP_SPING, line: { width: 1.2, color: "white" } },
        name: "shortest-ping VP" });
    }

    // 7) the error itself, then the two endpoints on top of it.
    if (t.pred) {
      markKind[traces.length] = "pred";
      traces.push(connectorTrace(t.pred, t.true, ERR_LINK));
    }
    markKind[traces.length] = "target";
    traces.push({ type: "scattergeo", mode: "markers", lat: [t.true[0]], lon: [t.true[1]],
      marker: { size: 13, color: "gold", symbol: "star", line: { color: "black", width: 1 } },
      hoverinfo: "none", name: "true target" });
    if (t.pred) {
      markKind[traces.length] = "pred";
      traces.push({ type: "scattergeo", mode: "markers", lat: [t.pred[0]], lon: [t.pred[1]],
        marker: { size: 12, symbol: STATUS_SYMBOL[t.status] || "triangle-up",
                  color: STATUS_COLOR[t.status] || "#888",
                  line: { color: "white", width: 1.4 } },
        hoverinfo: "none",
        name: isBaseline ? "prediction · shortest-ping VP" : "prediction" });
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
      title: { text: `fold ${t.fold} · ${t.tg_id} · ${t.status}`, font: { size: 14 } },
    };
    // `albers usa` has a fixed frame; only the free projections take a centre.
    if (projSel.value !== "albers usa") layout.geo.center = { lat: t.true[0], lon: t.true[1] };

    const pct = t.n_total ? (100 * t.n_measured / t.n_total).toFixed(1) : "—";
    const hiddenNote = hidden > 0 ? ` &nbsp;|&nbsp; <span style="color:#b00">${hidden} ring(s) hidden</span>` : "";
    // `nside` is obligatory here, not decorative: one page is one rung, and
    // nothing else on the page says which one the verdict was taken at.
    // Both verdicts on the first line, in that order. The grid offset is
    // printed as the number `classify` records, never as the tier name: the
    // `beyond` bucket spans 3 to 68 grids out and the bucket alone is the
    // thing a case viewer must not stop at.
    metaDiv.innerHTML =
      statusBadge(t.status) + " " + cellBadge(t.cell_label) + " " +
      `<b>${esc(t.tg_id)}</b> &nbsp;|&nbsp; error=<b>${num(t.pred_dist_to_tg_km, 1, "km")}</b>` +
      ` &nbsp;|&nbsp; grid offset=<b>${t.grid_offset < 0 ? "—" : t.grid_offset}</b>` +
      ` &nbsp;|&nbsp; TG grid <b>#${t.tg_grid}</b> / pred grid ` +
      `<b>${t.pred_grid >= 0 ? "#" + t.pred_grid : "—"}</b>` +
      ` &nbsp;|&nbsp; TG cell <b>#${t.tg_seed_id}</b> / pred cell ` +
      `<b>${t.pred_seed_id >= 0 ? "#" + t.pred_seed_id : "—"}</b>` +
      ` &nbsp;|&nbsp; nside <b>${NSIDE}</b> · ${GRID_KM} km grids` +
      ` &nbsp;|&nbsp; min-RTT infl=<b>${num(t.min_inflation, 2, "×")}</b><br>` +
      `measured VPs ${t.n_measured}/${t.n_total} (${pct}%) · ` +
      (isBaseline
        ? ""
        : isDensity
        ? // No inclusion filter runs, so every constraint participates and
          // `n_kept` carries no information. The region is named for what it
          // is -- the argmax's own cell -- so it is not read as a feasible set.
          `LTD constraints ${(t.rings || []).length} (all contribute, no ` +
          `inclusion filter), ${shown} drawn · ` + cutoffNote(t, past) + `region=` +
          `${t.region ? `argmax grid (healpix nside=${data.density_nside})` : "none"}` +
          ` · `
        : `LTD constraints ${t.n_kept}/${(t.rings || []).length} kept by the inclusion ` +
          `filter, ${shown} drawn${dropped ? `, ${dropped} dropped` : ""} · ` +
          cutoffNote(t, past) +
          `region=${t.region ? t.region.kind : "none"} · `) +
      `rank ${tIdx + 1}/${currentList.length} by ${sortSel ? sortSel.value : "error"}` +
      (pctSel.value !== "" ? ` (p${pctSel.value} by error, answered only)` : "") + hiddenNote +
      // The cells are DRAWN as a planar Voronoi in EPSG:5070 but SCORED as a
      // great-circle nearest seed. Stating the gap is the same honesty the
      // answer-space manifest carries: the figure says how faithfully it draws
      // the rule rather than asserting it draws it exactly.
      ` &nbsp;|&nbsp; ${Object.keys(cells).length} cells drawn, ` +
      `${(100 * (cellMeta.agreement ?? 0)).toFixed(1)}% agree with the ` +
      `great-circle nearest-seed rule classify scores`;

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
      const pastIdx = [pastOuterIdx, pastInnerIdx].filter((i) => i >= 0);
      if (pastIdx.length) {
        Plotly.restyle(plotDiv, { "line.color": dim ? RING_PAST_DIM : RING_PAST }, pastIdx);
      }
    }
    function clearHighlight() {
      Plotly.restyle(plotDiv, { lat: [[], []], lon: [[], []] }, [hlOuterIdx, hlInnerIdx]);
    }

    // One gesture: hovering a mark opens its panel, and a VP additionally
    // lifts its constraint. Unhover undoes both.
    onPlotHover = (ev) => {
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      const kind = markKind[pt.curveNumber];
      if (kind === "target") { showPopup(ev, `Target ${t.tg_id}`, targetPopupRows(t)); return; }
      if (kind === "pred") {
        showPopup(
          ev,
          isBaseline ? "Prediction · shortest-ping VP" : "Prediction",
          isBaseline ? baselinePredPopupRows(t) : predPopupRows(t),
        );
        return;
      }
      if (kind !== "vp") return;

      const vpId = pt.customdata;
      showPopup(ev, `VP ${vpId}`, vpPopupRows(vpId, t, obsByVp.get(vpId)));

      const coord = vps[vpId];
      const ring = ringByVp.get(vpId);
      // A latent VP, or one whose LTD failed, has no constraint to lift.
      if (!coord || !ring) { clearHighlight(); dimBundle(false); return; }
      const colour = ring[3] !== 1 ? HL_RING_DROPPED : (ring[5] === 1 ? HL_RING_PAST : HL_RING);
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
    onPlotUnhover = () => { hidePopup(); clearHighlight(); dimBundle(false); };

    // `Plotly.react` keeps existing handlers, so they must be cleared or every
    // redraw stacks another copy onto the same div.
    if (plotDiv.removeAllListeners) {
      plotDiv.removeAllListeners("plotly_hover");
      plotDiv.removeAllListeners("plotly_unhover");
    }
    plotDiv.on("plotly_hover", onPlotHover);
    plotDiv.on("plotly_unhover", onPlotUnhover);
  }

  pctSel.addEventListener("change", () => {
    if (pctSel.value !== "") targetSel.value = String(percentileTargetIndex(+pctSel.value));
    draw();
  });
  statusSel.addEventListener("change", () => { populateTargets(); draw(); });
  // Both re-filter or re-order the list, so both must repopulate the dropdown
  // before redrawing -- `draw()` reads `currentList[targetSel.value]`, and a
  // stale index there silently shows a different TG than the one selected.
  if (cellStatusSel) cellStatusSel.addEventListener("change", () => { populateTargets(); draw(); });
  if (sortSel) sortSel.addEventListener("change", () => { populateTargets(); draw(); });
  targetSearch.addEventListener("input", () => { pctSel.value = ""; populateTargets(); draw(); });
  targetSel.addEventListener("change", draw);
  projSel.addEventListener("change", draw);
  maxRSel.addEventListener("change", draw);
  for (const el of [showCells, showFrame, showGrids, showNbhd, showRings, keptOnly,
                    showRegion, showLatent]) {
    el.addEventListener("change", draw);
  }

  populateTargets();
  draw();

  // Headless-test hooks. `typeof module` is undefined in a browser, so this is
  // dead code there; under node it lets a harness drive draw()/click without a
  // real DOM. See scripts/analysis/v5/tests/test_map_mtl_viewer.js.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      redraw: draw,
      repopulate: populateTargets,
      markKind: () => markKind,
      // By index, not by name: a layer label carries its own grid count
      // (`1 grid out (7 grids)`), so name-matching would be brittle.
      layerIndex: () => ringIdx,
      // The two highlighted cells, for the same reason.
      cellIndex: () => cellIdx,
      dispatchHover: (ev) => onPlotHover(ev),
      selected: () => currentList[+targetSel.value || 0],
      dispatchUnhover: () => onPlotUnhover(),
    };
  }
})();
