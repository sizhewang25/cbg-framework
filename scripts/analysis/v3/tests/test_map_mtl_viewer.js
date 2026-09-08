// Headless smoke test for templates/mtl_map.js.
//
// The viewer is the one artifact in this layer that Python cannot check: a
// typo in draw() renders a blank page, and every Python-side assertion about
// the payload still passes. So this stubs just enough DOM and Plotly to
// actually execute the IIFE, then drives every target, projection, layer
// toggle, status filter and clickable trace.
//
// Driven by tests/test_map_mtl.py, which renders the fixture HTML first and
// skips when node is unavailable. Run directly with:
//   node scripts/analysis/v3/tests/test_map_mtl_viewer.js <rendered.html>

const fs = require("fs");
const path = require("path");

const htmlPath = process.argv[2];
const jsPath = process.argv[3] ||
  path.join(__dirname, "..", "modules", "templates", "mtl_map.js");
const html = fs.readFileSync(htmlPath, "utf8");
const payload = html.match(
  /<script id="data" type="application\/json">([\s\S]*?)<\/script>/
)[1];
const js = fs.readFileSync(jsPath, "utf8");

// Initial control state is read out of the shell's own markup rather than
// invented here, so the harness exercises the real default view.
const checkedIds = new Set(
  [...html.matchAll(/id="([A-Za-z0-9_]+)"[^>]*checked/g)].map((m) => m[1])
);
const store = {};
function el(id) {
  if (store[id]) return store[id];
  store[id] = {
    id, value: "", checked: checkedIds.has(id), innerHTML: "", textContent: "",
    style: {}, options: [], firstChild: { nodeValue: "" },
    appendChild(o) { this.options.push(o); },
    addEventListener() {},
    querySelector() { return { addEventListener() {} }; },
    removeAllListeners() {}, on() {},
  };
  return store[id];
}
el("data").textContent = payload;
for (const [, id, v] of html.matchAll(
  /<select id="([A-Za-z0-9_]+)"[\s\S]*?<option value="([^"]*)"[^>]*selected/g
)) el(id).value = v;

// `null` separators split one trace into several rings.
function splitRings(lats, lons) {
  const rings = [];
  let cur = [];
  for (let i = 0; i < lats.length; i++) {
    if (lats[i] === null) { if (cur.length > 2) rings.push(cur); cur = []; continue; }
    cur.push([lats[i], lons[i]]);
  }
  if (cur.length > 2) rings.push(cur);
  return rings;
}

// Shoelace in (lon, lat): > 0 is counter-clockwise.
function signedArea(ring) {
  let a = 0;
  for (let i = 0; i < ring.length; i++) {
    const [laA, loA] = ring[i];
    const [laB, loB] = ring[(i + 1) % ring.length];
    a += loA * laB - loB * laA;
  }
  return a / 2;
}

function isTransparent(color) {
  if (!color) return true;
  const m = /rgba\([^,]+,[^,]+,[^,]+,\s*([0-9.]+)\s*\)/.exec(color);
  return m ? parseFloat(m[1]) === 0 : false;
}

let reactCalls = 0;
let lastTraces = null;
const restyleCalls = [];
const tooltipTraces = new Set();
const HIGHLIGHT_NAMES = new Set([
  "vp-constraint-highlight", "vp-constraint-highlight-inner",
]);
// Union over every draw, so layers that only appear under a non-default
// control state (a dropped-constraint ring, an empty-list message) are
// still observable to the caller.
const allLayers = new Set();
global.window = { scrollX: 0, innerWidth: 1400 };
global.document = {
  getElementById: el,
  addEventListener() {},
  createElement: () => ({ value: "", textContent: "" }),
};
global.Plotly = {
  react(div, traces, layout) {
    reactCalls++;
    lastTraces = traces;
    for (const t of traces) allLayers.add(t.name || "(unnamed)");
    if (!layout.geo) throw new Error("no geo layout");
    for (const t of traces) {
      if (t.lat && t.lon && t.lat.length !== t.lon.length) {
        throw new Error(`trace "${t.name}" has a lat/lon length mismatch`);
      }
      // A NaN slips through as a silently dropped point rather than an error,
      // so the ring simply disappears; catch it here instead.
      for (const v of (t.lat || [])) {
        if (v !== null && (typeof v !== "number" || !isFinite(v))) {
          throw new Error(`trace "${t.name}" has a non-finite lat: ${v}`);
        }
      }
      // Winding. `fill: "toself"` on scattergeo fills the side to the right of
      // the walk on a spherical path, so a counter-clockwise ring fills the
      // antipodal complement -- the whole globe minus the shape. Stack a few
      // and the map is one flat wash of colour. The outline is identical
      // either way, so nothing else here would notice.
      // No trace may render a Plotly tooltip: detail is the click panel's job,
      // and a second differently-styled box on top of it is noise.
      if (t.hoverinfo === "text" || t.text) {
        tooltipTraces.add(t.name || "(unnamed)");
      }
      if (t.fill === "toself" && !isTransparent(t.fillcolor)) {
        for (const [k, ring] of splitRings(t.lat, t.lon).entries()) {
          if (signedArea(ring) > 0) {
            throw new Error(
              `trace "${t.name}" ring ${k} is counter-clockwise and filled ` +
              `with ${t.fillcolor}; it will fill the whole globe`
            );
          }
        }
      }
    }
  },
  restyle(div, update, indices) {
    restyleCalls.push({ update, indices });
    // Mirror the restyle onto the trace list so a hover highlight can be
    // observed the way the browser would render it.
    for (const [k, idx] of (indices || []).entries()) {
      const t = lastTraces[idx];
      if (!t) continue;
      for (const [key, vals] of Object.entries(update)) {
        const v = Array.isArray(vals) ? vals[Math.min(k, vals.length - 1)] : vals;
        if (key === "lat") t.lat = v;
        else if (key === "lon") t.lon = v;
        else if (key === "line.color") t.line = Object.assign({}, t.line, { color: v });
      }
    }
  },
};

const module_ = { exports: {} };
const api = (function () {
  const module = module_;
  eval(js);
  return module.exports;
})();

const targetSel = el("target");
const statusSel = el("status");
const nOpts = targetSel.options.length;
if (nOpts === 0) throw new Error("no targets in the dropdown");
if (!lastTraces || lastTraces.length === 0) throw new Error("first draw produced no traces");
const firstLayers = lastTraces.map((t) => t.name || "(unnamed)");

for (let i = 0; i < nOpts; i++) { targetSel.value = String(i); api.redraw(); }

targetSel.value = "0";
for (const proj of ["albers usa", "natural earth", "orthographic", "equirectangular", "robinson"]) {
  el("proj").value = proj;
  api.redraw();
}
el("proj").value = "albers usa";

const layers = ["showVoronoi", "showSeeds", "showMargin", "showRings", "keptOnly",
                "showRegion", "showLatent"];
for (const id of layers) el(id).checked = false;
api.redraw();
for (const id of layers) { el(id).checked = true; api.redraw(); }

for (const v of ["all", "fail", "correct", "wrong", "failed"]) {
  statusSel.value = v;
  api.repopulate();
  api.redraw();
}

// Percentiles run over the error distribution of the answered targets, with
// the fallbacks excluded. Both ends must select a real target and the
// direction must be monotone increasing: p5 the near-miss, p95 the disaster.
statusSel.value = "all";
api.repopulate();
const pctSel = el("pct");
const selectedTarget = () => api.selected();
const errorAt = (p) => {
  pctSel.value = String(p);
  api.repopulate();
  api.redraw();
  return selectedTarget().error_km;
};
const errs = [5, 25, 50, 75, 95].map(errorAt);
if (errs.some((e) => e === null || e === undefined)) {
  throw new Error(`a percentile selected a target with no error: ${JSON.stringify(errs)}`);
}
for (let i = 1; i < errs.length; i++) {
  if (errs[i] < errs[i - 1]) {
    throw new Error(
      `percentile error is not monotone increasing: ${JSON.stringify(errs)} ` +
      `- p5 must be the small error, p95 the large one`
    );
  }
}
pctSel.value = "";
api.repopulate();

statusSel.value = "all";
api.repopulate();
targetSel.value = "0";
api.redraw();

// One gesture drives everything: hovering a mark opens its panel, a VP also
// lifts its constraint, and unhovering undoes both. Walk every point of every
// interactive trace.
let hovers = 0;
let popups = 0;
let highlighted = 0;
const kindsSeen = new Set();
const drawnHighlights = () => lastTraces.filter(
  (t) => HIGHLIGHT_NAMES.has(t.name) && t.lat && t.lat.length > 3
).length;

for (const [curve, kind] of Object.entries(api.markKind())) {
  const trace = lastTraces[+curve];
  if (!trace) continue;
  const points = trace.customdata && trace.customdata.length
    ? trace.customdata
    : [undefined];
  for (const cd of points) {
    el("popup").innerHTML = "";
    el("popup").style.display = "none";
    api.dispatchHover({
      points: [{ curveNumber: +curve, customdata: cd }],
      event: { pageX: 10, pageY: 10 },
    });
    hovers++;
    if (!el("popup").innerHTML) {
      throw new Error(`hovering the ${kind} trace produced no panel`);
    }
    if (el("popup").style.display !== "block") {
      throw new Error(`the ${kind} panel was filled but left hidden`);
    }
    popups++;
    kindsSeen.add(kind);
    if (kind === "vp" && drawnHighlights()) highlighted++;

    api.dispatchUnhover();
    if (el("popup").style.display !== "none") {
      throw new Error(`unhover left the ${kind} panel open`);
    }
    const left = drawnHighlights();
    if (left) throw new Error(`unhover left ${left} highlight ring(s) drawn`);
  }
}
for (const kind of ["target", "pred", "vp"]) {
  if (!kindsSeen.has(kind)) throw new Error(`no ${kind} mark was hoverable`);
}
// The Shortest-Ping baseline emits no LTD constraints, so there is nothing for
// a hover to lift out; only demand a highlight where rings actually exist.
const anyRings = JSON.parse(payload).targets.some((t) => (t.rings || []).length > 0);
if (anyRings && highlighted === 0) {
  throw new Error("hovering a VP never populated a highlight ring");
}

console.log(JSON.stringify({
  targets: nOpts, draws: reactCalls, hovers, popups, highlighted, anyRings,
  percentileErrors: errs,
  tooltipTraces: [...tooltipTraces],
  // `hoverinfo: "none"` keeps hover events flowing; `"skip"` would not.
  hoverableVpTraces: Object.entries(api.markKind())
    .filter(([c, k]) => k === "vp" && lastTraces[+c] && lastTraces[+c].hoverinfo === "none")
    .length,
  layers: firstLayers, allLayers: [...allLayers],
}));
