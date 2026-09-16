// Headless smoke test for templates/ltd_model.js.
//
// The viewer is the one artifact in this layer that Python cannot check: a typo
// in draw() renders a blank page, and every Python-side assertion about the
// payload still passes. So this stubs just enough DOM and Plotly to actually
// execute the IIFE, then drives every fold, VP, layer toggle and target pick.
//
// It also enforces the one drawing invariant the payload cannot express on its
// own: a band gap must stay a gap. `band_for_vp` emits null where the model
// declines an RTT, and Plotly would bridge straight across a null inside one
// filled trace -- drawing a claim the model does not make, at exactly the short
// RTTs the accuracy story turns on. The check below is that no emitted trace
// carries a null, i.e. that the segmentation really split them.
//
// Driven by tests/test_figure_ltd_model.py, which renders the fixture HTML
// first and skips when node is unavailable. Run directly with:
//   node scripts/analysis/v3/tests/test_ltd_model_viewer.js <rendered.html>

const fs = require("fs");
const path = require("path");

const htmlPath = process.argv[2];
const jsPath =
  process.argv[3] ||
  path.join(__dirname, "..", "modules", "templates", "ltd_model.js");
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
  const node = {
    id,
    value: "",
    checked: checkedIds.has(id),
    innerHTML: "",
    textContent: "",
    style: {},
    options: [],
    appendChild(o) {
      this.options.push(o);
    },
    addEventListener() {},
  };
  // `innerHTML = ""` is how the viewer clears a <select>; mirror that onto the
  // option list so a repopulate does not silently accumulate.
  Object.defineProperty(node, "innerHTML", {
    get() {
      return node._html || "";
    },
    set(v) {
      node._html = v;
      if (v === "") node.options = [];
    },
  });
  store[id] = node;
  return node;
}
el("data").textContent = payload;

let reactCalls = 0;
let lastTraces = null;
const allLayers = new Set();
const nullBearing = new Set();

global.window = { innerWidth: 1400 };
global.document = {
  getElementById: el,
  addEventListener() {},
  createElement: () => ({ value: "", textContent: "", selected: false }),
};
global.Plotly = {
  react(div, traces, layout) {
    reactCalls++;
    lastTraces = traces;
    for (const t of traces) {
      allLayers.add(t.name || "(unnamed)");
      if (t.x && t.y && t.x.length !== t.y.length) {
        throw new Error(`trace "${t.name}" has an x/y length mismatch`);
      }
      // A NaN slips through as a silently dropped point rather than an error,
      // so the curve simply disappears; catch it here instead.
      for (const v of t.y || []) {
        if (v !== null && (typeof v !== "number" || !isFinite(v))) {
          throw new Error(`trace "${t.name}" has a non-finite y: ${v}`);
        }
      }
      // The band must never carry a null: `bandSegments` is supposed to have
      // split it into separate filled traces already. The eval-bound trace is
      // exempt -- there the nulls are deliberate per-target separators on an
      // unfilled line, which is the one place Plotly's own break is wanted.
      if (t.name !== "echoed LTD bound" && (t.y || []).some((v) => v === null)) {
        nullBearing.add(t.name || "(unnamed)");
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

const foldSel = el("fold");
const vpSel = el("vp");
const targetSel = el("targets");

if (!lastTraces || lastTraces.length === 0) {
  throw new Error("first draw produced no traces");
}
const firstLayers = lastTraces.map((t) => t.name || "(unnamed)");
const folds = foldSel.options.map((o) => o.value);
if (folds.length === 0) throw new Error("no folds in the dropdown");
const vpsPerFold = {};

// Every fold, and within each fold every VP.
for (const f of folds) {
  foldSel.value = f;
  api.refillVps();
  api.refillTargets();
  const vps = vpSel.options.map((o) => o.value);
  vpsPerFold[f] = vps.length;
  if (vps.length === 0) throw new Error(`fold ${f} has no VPs`);
  for (const vp of vps) {
    vpSel.value = vp;
    api.refillTargets();
    api.redraw();
  }
}

// Back to a known state, then the layer toggles.
foldSel.value = folds[0];
api.refillVps();
api.refillTargets();
const toggleIds = [
  "showScatter",
  "showBand",
  "showCenter",
  "showBaseline",
  "logY",
  "sharedAxes",
];
for (const id of toggleIds) el(id).checked = false;
api.redraw();
const tracesAllOff = lastTraces.length;
for (const id of toggleIds) {
  el(id).checked = true;
  api.redraw();
}

// Target overlay. The VP is chosen from the payload rather than left wherever
// the sweep above finished: a VP that participated in no target lists none, so
// picking one of those would assert the overlay is missing when the viewer is
// behaving correctly. Take the VP with the most participations.
const DATA = JSON.parse(payload);
let best = null;
for (const f of DATA.fold_ids) {
  const targets = DATA.folds[f].targets || {};
  const counts = {};
  for (const tg of Object.keys(targets)) {
    for (const vp of Object.keys(targets[tg].vps)) counts[vp] = (counts[vp] || 0) + 1;
  }
  for (const vp of Object.keys(counts)) {
    if (!best || counts[vp] > best.n) best = { fold: f, vp, n: counts[vp] };
  }
}
let overlaid = 0;
let overlayLayers = [];
if (best) {
  foldSel.value = best.fold;
  api.refillVps();
  vpSel.value = best.vp;
  api.clearPicks();
  api.refillTargets();
  for (const o of targetSel.options) {
    api.pick(o.value);
    overlaid++;
  }
  api.redraw();
  overlayLayers = lastTraces.map((t) => t.name || "(unnamed)");
  if (overlaid === 0) {
    throw new Error(`VP ${best.vp} participates in ${best.n} targets but the picker listed none`);
  }
}

// A search that matches nothing must not throw -- it leaves no VP selected,
// which the viewer has to report rather than crash on.
el("vpSearch").value = "zzz-no-such-vp";
api.refillVps();
api.redraw();

// Interior-gap segmentation, driven against the real function rather than a
// reimplementation. No run produces this shape -- every observed gap is at one
// end of the axis -- so it is unreachable through the payload and has to be
// called directly. An interior gap drawn as one filled trace is the failure the
// whole null-splitting design exists to prevent.
const interior = api.segments([
  [1.0, 0.0, 10.0],
  [2.0, null, null],
  [3.0, 0.0, 30.0],
  [4.0, 0.0, 40.0],
]);
const interiorShape = interior.map((r) => r.length);
// Leading and trailing gaps must not produce empty runs either.
const edgeShape = api
  .segments([
    [1.0, null, null],
    [2.0, 0.0, 20.0],
    [3.0, null, null],
  ])
  .map((r) => r.length);

console.log(
  JSON.stringify({
    reactCalls,
    folds: folds.length,
    vpsPerFold,
    firstLayers,
    allLayers: [...allLayers],
    overlayLayers,
    overlaid,
    tracesAllOff,
    nullBearing: [...nullBearing],
    interiorShape,
    edgeShape,
  })
);
