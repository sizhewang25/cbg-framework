# analysis/v5: two partitions, two labels

v4 graded a prediction on one axis, **distance**: HEALPix grid steps. That axis
has no direction. A miss one ring out can land inside the TG's own serving
region or inside a neighbour's, and an operator needs to know which. v5 adds a
second partition, the **unbounded** Voronoi **cell**, so every prediction
carries `(pred_dist_to_tg_grid, cell_label)`. One resolution: nside 128.

Scope: the answer space, classify, the answer-space map, the outcome bars,
the outcome map, the error-distance CDF, the VP-proximity violins, the MTL
case viewer and the LTD fit viewer.

## Glossary

Every name in v5 follows this table. `answer_space.GLOSSARY` writes it into
every `meta.json` and manifest.

| term | meaning | names |
|---|---|---|
| target (TG) | one target server behind an IP | `tg_*` |
| grid | one HEALPix pixel at nside 128, NESTED | `grid_*` |
| grid_km | nominal grid distance, √(grid area): 50.9 km | `grid_km` |
| grid offset | grid steps from the TG's grid to the prediction's grid, exact (−1 = no prediction) | `pred_dist_to_tg_grid` |
| ring tier | the bands `summarize` groups that offset into | `ring0` · `ring1` · `ring2` · `beyond` |
| site | a unique location of TGs, keyed `(run_id, tg_lat, tg_lon)` | `site_*` |
| seed | spherical centroid of sites grouped by complete linkage, diameter ≤ `grid_km` | `seed_*` |
| cell | Voronoi cell of a seed, unbounded, i.e. the serving region | `cell_*` |
| answer space | the TG answer space: grid partition and cell partition | |
| `*_dist_to_tg_km` | distance to the raw TG coordinate | |
| `*_dist_to_seed_km` | distance to the TG's seed | |

v4 used `seed` to mean a HEALPix centre and `cell` to mean a HEALPix pixel.
That clash is why v5 is a separate package and not an edit to v4.

## Methods

Methods are named by short terms throughout v5, in figures, manifests and
prose. The lookup table is `methods.METHOD_TERMS`:

| term  | full name             | combo id(s)                |
|-------|-----------------------|----------------------------|
| OCT-H | Octant-Hull CBG       | octant_cbg_hull            |
| OCT-S | Octant-Spline CBG     | octant_cbg_spl, octant_cbg |
| SOI   | Speed-of-Internet CBG | million_scale_cbg          |
| S-P   | Shortest-Ping         | shortest_ping              |
| SPO   | Spotter CBG           | spotter_cbg                |
| VAN   | Vanilla CBG           | vanilla_cbg                |

Each outcome-bar figure prints the terms it uses under its panels, and its
manifest records them.

## The two labels

| label | values | bounded by |
|---|---|---|
| `pred_dist_to_tg_grid` | 0 · 1 · 2 · 3 … (−1 = no prediction) | nothing — it is exact; the tiers are what bound it |
| `cell_label` | `correct` · `wrong` · `unanswered` | nothing — it is unbounded |

Every prediction falls in its nearest seed's cell: `correct` when that seed is
the TG's seed, `wrong` when it isn't. `unanswered` is a row with no prediction.
The three partition `n_tgs`, so both axes share one denominator.

The cell label **is** v4's retired nearest-seed rule, deliberately. v4 retired
it because it credited Seattle with a prediction in the Canadian Arctic
2,360 km away, and fixed that with a landmass polygon and a fourth label,
`outland`. v5 removes both: that prediction is `correct` on the cell axis and
68 grids out on the grid axis (`test_classify.TestTheArcticCase`), and the two
read together are the point. A `cell_label` on its own is not a verdict.

## One tolerance, two uses

`grid_km` is the grid pitch and the complete-linkage diameter for seeds, so
both partitions are built at the same granularity. EWR and JFK (33 km apart)
fall in two grids but share one seed. Seeds are a logical grouping of sites
and don't depend on where grid boundaries fall.

Complete linkage caps the group **diameter**. Single linkage would chain sites
40 km apart into one group of any length (`test_seeds.test_complete_linkage_does_not_chain`).

## Outputs

```
outputs/analysis/v5/<run>/answer-space/healpix-128/{grids,sites,seeds,tgs}.csv, meta.json
outputs/analysis/v5/<run>/classify/healpix-128/accuracy.csv, <method>_tgs.parquet, manifest.json
outputs/analysis/v5/<run>/classify/error_cdf[.sentinel].{png,csv,manifest.json}
outputs/analysis/v5/_cross/classify/<datasets>@<arm>/outcome_bars.*, error_cdf.pooled[.sentinel].*
outputs/analysis/v5/_cross/outcome-map/<datasets>@<arm>/outcome_map.<cohort>.{png,csv,manifest.json}
outputs/analysis/v5/_cross/vp-proximity/<datasets>@<arm>/vp_proximity.<cohort>.{png,csv,manifest.json}
outputs/analysis/v5/_cross/rtt-cdf/<datasets>@<arm>/rtt_cdf.{png,csv,manifest.json}
outputs/analysis/v5/<run>/mtl-map/healpix-128/mtl_map.<method>.html
outputs/analysis/v5/<run>/mtl-map/regions/<method>/<tg>.json          # replay cache, rung-free
outputs/analysis/v5/<run>/ltd-model/ltd_model.<method>.html, manifest.json   # no rung: see below
```

`accuracy.csv` contains:

- the grid axis: `accuracy_ring{0,1,2}` (cumulative) and `n_ring{0,1,2}, n_beyond, n_failed`, which partition `n_tgs`;
- the cell axis: `accuracy_cell_correct` and `n_cell_{correct,wrong,unanswered}`, which sum to `n_tgs`;
- the cross-tab: `n_{ring0,ring1,ring2,beyond}_cell_{correct,wrong}`, exclusive, where each tier's two graded labels sum to that tier's count. `unanswered` has no ring, so it takes no tier. `guard_cross_tab` asserts both, plus `n_cell_unanswered == n_failed` — the same rows counted along the two axes;
- percentiles of `pred_dist_to_tg_km` and `pred_dist_to_seed_km` over solved rows.

The denominator is every evaluated TG, on both axes. FALLBACK and ERROR rows
count as wrong, not excluded. A FALLBACK row still gets both labels, but only
`solved_mask` rows enter `correct`/`wrong`; the rest are `unanswered`.

## Figures

**`plot-answer-space`** draws one panel per run with both partitions on it:
the full HEALPix lattice with the TG grids filled, the cell boundaries, sites
(dots) and seeds (crosses). It's written to
`answer-space/answer_space_map.healpix.png`. The cell boundaries run off every
edge of the frame — that overreach is the figure's argument, not an artifact.

The cells are drawn as a planar Voronoi in EPSG:5070, with edges densified
before converting back to lon/lat, then cut to the drawn frame. That cut is a
rendering bound only: `extend_to` alone is a *lower* bound in GEOS and returns
polygons far outside it, whose corners leave EPSG:5070's usable domain and
come back as `inf` or wrapped past the antimeridian, drawn as lines across the
map. On a point sample the polygons agree with the great-circle nearest-seed
rule `classify` uses on 99.4–99.6% of the frame, and the manifest records it.

**`plot-outcome-bars`** draws one figure per layout × mode. Layouts are
`compare` (one panel per dataset) and `pooled` (a micro-average); modes are
`bounded` (default) and `unbounded`. They are written to
`_cross/classify/<datasets>@<arm>/`, the `unbounded` ones taking a
`.unbounded` filename token so `bounded` keeps the paths it has always had.

Every bar stacks the **cell label** first, bottom-up (`correct`, `wrong`, no
answer). In `bounded` each answered group is broken down by **ring tier**,
asking "right serving region or not, and within that, how far off?". In
`unbounded` the breakdown is dropped and the plain nearest-seed verdict is
drawn alone. Read as a pair: a method can lead on serving region while holding
**no ring0 at all** — Spotter does, at 63.7% against Octant-Hull's 63.1%, with
`accuracy_ring0` of 0.0.

- **Colour**: in `bounded`, the ring tier — green (in the TG grid), light blue
  (1 ring out), light purple (2 rings out), light grey (further out), dark
  grey (no answer). In `unbounded`, the cell label — a deeper green
  (`correct`), red (`wrong`), the same dark grey (no answer).
- **Stripe = cell label**: plain for `correct`, `//` for `wrong`.
- **Borders and separators:** each cell-label group has a thick border, and
  white lines separate the ring tiers inside it.
- **Labels:** every sub-segment wide enough prints its own share.
- **Rail** (`bounded` only; in `unbounded` the bar already *is* the cell
  breakdown): right of each bar, one white striped segment per group, with the
  group total set vertically beside it.

Each panel ranks methods by `true` share, then by how tight their true
predictions are (`true & ring0`, `true & <=ring1`, ...).

**`plot-outcome-map`** places what the bars count. One panel per
`(method, dataset)` — rows in `methods.TERM_ORDER`, columns in the order the
runs were named — drawing one cohort of the cell axis, `correct` or `wrong`,
as a separate figure each. A panel carries the run's Voronoi cells and seeds,
every prediction of the cohort as a marker filled by `pred_dist_to_tg_grid` on
a discrete `turbo` ramp binned at `[0, 1, 2, 3, 5, 8, 12, 20, 40]` (the first
three bins are `classify`'s ring tiers), a thin grey line from each prediction
to its target, and a per-cell count.

The counts key on **`tg_seed_id`, never `pred_seed_id`**, so on the `wrong`
map a cell's number reads "predictions that should have landed here". They are
pushed apart by a spring relaxation and joined back to their seed once
displaced past 1.3°.

`solved_mask` is applied before anything is drawn: a FALLBACK row carries the
S-P baseline's coordinate, and counting it reads as01/VAN at **270 correct
instead of 163**. The 107 rows removed are named in the panel title
(`107 FB excl.`), as is the offset p50/max and the off-map count.

All panels share one frame, `--extent`, defaulting to `(-128, -63, 21, 55)` —
wider than `US_MAINLAND_EXTENT` because SPO answers into Canada and over the
Gulf. A shared frame is the point: a per-panel auto-extent would rescale every
map and a reader comparing down a column would be comparing different
pictures. Predictions outside it are drawn as dark-red carets pinned at the
edge and counted (as01/SPO: 20 of 354 `correct`), never cropped.

The CSV twin carries one row per `(run, method, cohort, seed)` — the drawn
count, its distinct site count and its offset percentiles — plus the panel's
own totals repeated onto every row, so each title is reconstructible without
re-reading a parquet.

**Known limitation:** the relaxation repels labels from labels, not from the
prediction markers, so in the dense north-east cluster (NYC / Philadelphia /
DC / Boston) a count can come to rest on a marker and two leader lines can
cross. The numbers stay right and are in the CSV. Unimplemented candidates:
treat markers as obstacles in the repulsion; allocate labels by angle around a
cluster centroid so leaders cannot cross; a callout column for the densest
cluster.

**`plot-error-cdf`** (ported from v4) draws the empirical CDF of
`pred_dist_to_tg_km`, one curve per method, on a log x axis. S-P is the
dark-grey dashed baseline. There are two layouts: `per-run`, written to
`classify/error_cdf.*`, and `pooled`, written to `_cross/.../error_cdf.pooled.*`.
The pooled layout concatenates the runs' rows and recomputes the percentiles
rather than averaging them. Unanswered rows are excluded (`solved_mask`), so
each curve covers the outcome bars' answered stack. The panel is a 4×3in paper
column and carries curves, a key and two axis names — nothing else. The
percentiles (p5/25/50/75/90/95), the row policy and the method glossary are in
the CSV and manifest written beside it. The legend is in `methods.TERM_ORDER` —
S-P, SOI, VAN, OCT-H, OCT-S, SPO — fixed rather than ranked, so a method holds
the same row in every figure; the CSV rows are still written best-first. The
distance is to the raw TG, never to the seed, so it's the same at every rung. That's why the
filenames carry no `healpix-<n>`. p50/p90 match `accuracy.csv` digit for
digit.

`--unanswered sentinel` draws the same curves over the **whole TG roster**
instead, parking each unanswered row at `--sentinel-km` (default 10,000 km).
Every method then shares a denominator, so the curves are comparable by shape
and each one's height at the sentinel line is its answer rate — on as01-03,
VAN plateaus at 0.78 and only reaches 1 at the sentinel. The right edge widens
to 20,015 km (the antipodal maximum) so the sentinel is not drawn on the
spine. The price is censored percentiles: VAN's p50 moves 199 → 285 km and its
p95 is `10,000` in the CSV, which is a statement about its answer rate, not a
distance. These artifacts take a `.sentinel.` infix and carry
`unanswered_policy`/`sentinel_km` columns, because **only the default files
join to `accuracy.csv`**.

**`plot-mtl-map`** (ported from v4) is the **case viewer**: one self-contained
interactive HTML per method, Plotly from a CDN with the payload inlined, so it
opens over `file://` with no web server. Every other v5 figure shows a
distribution; this shows one TG.

It draws both partitions, because in v5 neither is a verdict alone:

- the **grid axis** — the TG's grid and its ring-1/ring-2 neighbours, plus the
  grid the prediction landed in;
- the **cell axis** — the Voronoi cell of every seed, with the TG's own cell
  filled and the cell the prediction fell in outlined when they differ. This is
  what v4 drew as decoration and labelled "context, not the verdict"; here it
  *is* `cell_label`, so it is keyed by `seed_id` and graded.

Plus each VP's LTD constraint (disks, or annuli when the LTD emits a lower
bound), the MTL feasible region they intersect to, and the error.

The status filter and the cell filter are **independent**, which is the point of
having two axes: "right serving region but 14 grids out" is a different failure
from "wrong serving region, 1 grid out", and neither axis alone separates them.
On as01/`octant_cbg_hull`, 70 of 399 TGs are `beyond` yet `correct`.

The grid axis reads **`pred_dist_to_tg_grid`**, not a banded column, and bands it
into the same five statuses `summarize` counts. The popup and the meta strip
print the **exact offset** — `beyond` spans 3 to 68 grids out on these runs, and
a case viewer should say which. The *drawn* neighbourhood still stops at
`MAX_RING`: the disk at offset 68 is ~15,000 grids, undrawable and meaningless
as a shaded region.

The cells are built once per run and shared by every method's page — they belong
to the answer space, not the method. They are cut to `CELL_FRAME`
`(-136, -56, 14, 66)`, which contains every prediction in every run on disk;
`US_MAINLAND_EXTENT` clips Spotter's 64.57°N one. That frame is a **rendering
bound**, drawn on the page as a dotted rectangle and labelled as one, because
the viewer pans and offers an orthographic projection — without it a reader sees
the cells stop mid-Atlantic and reads the stop as a cell edge. Rings are
simplified at 0.02° (a 27× byte reduction for ~2.2 km of deviation against a
50.9 km grid); the page prints the resulting agreement with the great-circle
rule `classify` scores, ~98.9%.

Cost: the MTL feasible region is never serialized by the benchmark, so each is a
full replay of the planar intersection — ~7 s/TG on the Octant family. Hence
`--no-regions`, `-j`, and an on-disk cache under `mtl-map/regions/`, which is
**rung-free**: no nside enters the replay, so a second rung reuses it (measured:
1.4 s). Driven by its own sweep script, `create_mtl_map.sh`, because every other
v5 command costs seconds.

**`plot-vp-proximity`** (ported from v4) pools the given runs and draws two
violins per method on a log x axis: the distance from the TG to its
**geographically closest** VP (`geo_vp_dist_to_tg_km`, blue) and to its
**smallest-RTT** VP (`sping_vp_dist_to_tg_km`, orange), which is the coordinate
S-P returns. The gap between them is RTT inflation. Both come from the run's
canonical edge CSV (`edges.resolve_source_csv`, which refuses a mesh superset
on a weighted arm). `--cohort` picks each method's own best `p5`/`p25`/`p95` by
`pred_dist_to_tg_km` over `solved_mask` rows (FALLBACK can't enter), or `all`
(unanswered included). Rows are ordered by the bound (the max), tightest
first. The stats CSV carries `max_km` beside `distinct_values` and
`max_tie_share`, because at p5 ~20 replicas per site make the violin mostly
smoothing. S-P's row is `circular`. On as01-03 the CSVs match v4's cell for
cell (`test_figure_vp_proximity.TestRealRuns`).

**`plot-rtt-cdf`** describes the **input** rather than any method: one CDF per
dataset on one linear axis, over every `(vp, tg)` edge at its minimum RTT, read
through the same `edges` resolver as `plot-vp-proximity` (so a weighted arm
still refuses a mesh superset). It needs no answer space and no `classify`, and
is the figure to open before comparing two meshes' results — if their latency
distributions differ, every downstream difference is confounded.

Nothing is pooled. Each dataset keeps its own denominator, so
`guard_disjoint_tgs` deliberately does not apply and a TG in two meshes lands on
both curves.

The axis is where this figure can lie, so both of its knobs are recorded.
`--x-max` (default 200 ms) is a real cut: a truncated *linear* axis looks
identical whether the tail beyond it is empty or holds a third of the data, so
`share_within_xmax_pct` and `observed_max_ms` are in the manifest per dataset
and the panel prints a warning whenever any dataset is clipped. On as01-03 it
never fires — the largest RTT in the three meshes is 92.4 ms, so the default
wastes half the axis and is kept only so two runs of the figure are comparable.

`--x-step` sets the linear tick spacing in ms (unset leaves matplotlib's
automatic locator). It is refused above `--x-max`, which would leave a single
tick at 0, and it is **ignored on `symlog`**, where a constant spacing is
meaningless because the decades are the ticks — a test pins that so it cannot
silently half-apply.

`--x-scale` is `linear` or `symlog`. Plain `log` is **not** offered: it would
work on these meshes, but it is one dataset away from silently dropping rows,
since a source that rounds a sub-millisecond RTT to 0 would vanish from the
curve rather than show up in it. `symlog` stays defined at 0 and keeps a linear
window below `--linthresh` (default 1 ms, just under the 0.58 ms floor
observed). That window is its own failure mode — set it above the bulk of the
data and `symlog` draws a linear axis under a log label — so `validate_axis`
refuses `linthresh >= x_max` and the manifest carries
`share_below_linthresh_pct` per dataset (0.15–0.24% on as01-03). Legend
placement follows the scale: `symlog` drags the curves' rise into the lower
right, where the linear layout puts the legend.

Legend labels and line styles come from `DISPLAY`, keyed on the dataset head,
because neither is derivable: `as7018` is **RIPE MIX-ASN** (RIPE probes across
many ASNs) against `as01-03`'s **PRO AS0X** (one operator's single-AS meshes),
and no part of a run id says so. The operator meshes are solid and the RIPE
mesh is grey and dashed, so the reference curve reads as a reference. Because
the solid trio has no linestyle cue left, its hues are blue/orange/**purple**
rather than blue/orange/green — orange against green is the pair deuteranopes
lose, and a test pins green's absence. An unlisted dataset draws dotted under
its uppercased head rather than raising.

Each dataset also gets a dashed **median dropline** (`--no-medians` to drop
them), running from the axis floor to 0.5 — where the median meets its own
curve by definition — with the value in a column anchored right of the
rightmost dropline. A median past `--x-max` is skipped rather than clamped,
since a label pinned at the edge reads as "the median is x_max". A faint
reference at CDF 0.5 says what the verticals are.

Any dataset running past `--x-max` is named **above the axes**, right-aligned,
in its own colour, with its **p99** — `RIPE MIX-ASN  (p99: 104.9 ms)`. p99
rather than the max on purpose: a max above the axis can be one packet, while a
p99 above it says the truncation is structural. The note goes outside the frame
because the top-right corner only looks free — every curve that reaches 1.0
runs flat along the top edge to `x_max`, so a note placed there lands on the
lines it describes. `--no-annotate-clipping` suppresses them, and
`annotated_on_figure` in the manifest records which way it was drawn, so a
silent panel stays distinguishable from an unclipped one.

The y axis is the CDF itself — a fraction in [0, 1] in steps of 0.25. The stats
CSV and the manifest still speak in **percent** (`share_within_xmax_pct`,
`share_below_linthresh_pct`), because those are numbers a reader quotes in prose
rather than axis coordinates; `TestTheYAxisIsACdf` pins both halves so the two
conventions cannot drift into each other.

The unit is an **edge**, not a TG, so ~20 replicas per site oversample that
site's latency ~20×; `n_tgs` and `n_vps` are in the CSV for exactly that check.

**`plot-ltd-model`** (ported from v3) is the other **case viewer**, and the one
command here that looks at the *latency-to-distance stage* rather than at a
prediction. For each fold and each VP it draws the RTT-vs-distance training
scatter, the band the fitted model actually returns and the 2/3 c baseline — the
figure to open when a run's accuracy moves and the question is whether the fit is
the cause. One self-contained HTML per method, Plotly from a CDN with the payload
inlined, so it opens over `file://`.

**Neither partition enters this page**, and that is why it is here rather than in
`classify`'s tree. An LTD fit maps an RTT to kilometres; no grid and no cell are
involved. So it is the only v5 command that needs no `build-answer-space`, it
takes no `--nside`, and `ltd-model/` carries **no `healpix-<n>`** — a slugged
directory would hold one byte-identical 4–8 MB page per rung.

### The model is named, never inferred from the method

`run.json` records `ltd` and `ltd_kwargs` per combo, and that is what selects the
rendering. The combo id cannot be pattern-matched: OCT-S and OCT-H are **the same
class** (`bounded_spline`), differing only in `fit_spline`. Their `predict` bands
coincide, so the spline centre line is the entire visible difference between the
two pages — which is why a model exposing no centre reports that in the meta line
rather than borrowing a flat one.

SOI writes a `.stateless` marker and no pickle at all, because
`SpeedOfInternetLTD` has no post-fit state. That is not a missing model: the class
plus `run.json`'s kwargs reconstructs an equivalent instance, and the page says so.

### A declined RTT is a gap, never a zero

`predict` legitimately fails on part of an axis: the pooled SPO model declines
every RTT below its fitted minimum — on as01 that is 1–4 grid points on 413 of 670
VP panels. Those points are `null` in the payload and the JS splits the band into
separate filled traces at each one, because Plotly would otherwise bridge straight
across inside a single filled polygon, drawing a claim the model does not make at
exactly the short RTTs the accuracy story turns on.

The grid's endpoints are inset by 1e-4 of the observed span for a related but
distinct reason. The Octant hull is built *through* a VP's extreme observations,
so at exactly its first and last RTT the bounds coincide and `predict` returns
`DEGENERATE_REGION`. Sampling the closed boundary is this module's choice, not the
model's domain; uncorrected it put a spurious null on both ends of all 670 octant
panels. The inset is ~0.006 ms on a 59 ms range, and SPO's genuine leading gaps
survive it unchanged — which is how we know it is not hiding one.

### The fit scatter is not in the output tree

This is the one input the command cannot read from the fold directory. `run.json`
and `fit_checkpoint.pkl` describe the fit; the data it was fit on lives elsewhere.
Two routes, tried in order, and **both are load-bearing**:

1. `inputs/benchmark/v2/<source>/<run>/<setup>/<fold>/fit_samples.parquet` — the
   exact artifact the runner consumed. The three meshes have this, all five folds.
2. The dataset CSV (via `edges.resolve_source_csv`, off the run's own
   `eval_source/`) filtered by its pinned `.stratification.json`: the fit set is
   every row whose TG is in one of the other K−1 folds. `as01-260728-260802` has
   only this. A weighted arm has neither — no file holds its pruned flows — and
   `MeshSupersetError` degrades that to a per-fold skip carrying its own hint.

Route 2 is a reconstruction, so it is **asserted, not trusted**: the row count must
equal `run.json`'s `n_fit_samples` or the build fails. A TG the stratification does
not mention is dropped rather than counted as "some other fold" — unassigned is not
outside the evaluated fold, and counting it in would inflate the scatter with rows
the model may have been scored on. Which route produced a page is printed on the
page, echoed by the CLI and recorded in `manifest.json`, because the scatter is
reconstructed for some runs and a reader has to be able to tell.

### The TG overlay is a multi-select

Off by default; `--tg` only decides what is already ticked on load. Each picked TG
contributes its `(rtt, true distance)` point on the current VP's axes, plus the LTD
bound that VP echoed into the multilateration — the constraint the TG had to
satisfy, against where it actually was.

These are **post-filter** participants, from `targets.parquet`'s
`mtl_participants`: the MTL drops a disk that fully contains another before
intersecting, so a TG lists fewer VPs here than the fit had (27 of 134 on a sampled
as01 row). That is the right set — it is what formed the region — but it is why
this is an overlay on the fit rather than a substitute for it.

Cost: ~670 VP panels per method and 4–8 MB of HTML, about 15 s per run. Driven by
its own sweep script, `create_ltd_modeling_html.sh`, for the reason
`create_mtl_map.sh` has one — except that this one invokes the CLI **once per run**
rather than once per method, because the per-method failure isolation is already
inside `build_for_run` and a run's methods share one per-fold scatter cache.

## Guarantees

- `pred_dist_to_tg_km` matches v4's `error_km` row for row on all three meshes,
  and `min(pred_dist_to_tg_grid, 2)` reconstructs v4's `ring` exactly — the
  proof that dropping the capped column lost nothing (`test_real_runs`).
- The grid axis is untouched by the cell axis: `accuracy_ring{0,1,2}` is never
  conditioned on `cell_label`, so removing the landmass moved none of them.
  Verified bit-for-bit against the pre-removal artifacts at nside 128.
- Every seed gets a cell, and the cells cover the whole drawn frame with no
  gap. An unbounded partition leaves nothing unassigned — the one
  simplification the landmass removal buys (`test_real_runs`).
- Seeds never outnumber sites. They're cuts of one complete-linkage tree.
- The case viewer's two verdicts match `accuracy.csv` bucket for bucket, on
  both axes and on the offset percentiles, for every method
  (`test_map_mtl.TestOnRealRuns`). The cell label is masked by `solved_mask`
  the way `summarize` masks it — without that a FALLBACK row's baseline
  coordinate is credited to the method, worth 107 rows on as01/`vanilla_cbg`.
- Both viewers' JS is executed under `node` against a stubbed DOM and Plotly
  (`test_map_mtl_viewer.js`, `test_ltd_model_viewer.js`), which is the only way
  to catch a `draw()` typo: a blank page still passes every payload assertion.
  The LTD harness additionally fails if any filled trace carries a `null`, which
  is the band-gap invariant above.
- The LTD viewer's payload is identical to v3's, field for field, on every method
  of `as01-260728-260802-mesh` — only `method_label` differs, `Vanilla CBG` →
  `VAN`. The port renamed names, not numbers.

v5 runs **one resolution**, so the cross-rung guarantees v4 asserted —
`accuracy_ring0` monotone as grids grow, seeds only merging — have no ladder
left to hold across and were removed with it. HEALPix is still the right grid
for the equal-area reason (a grid count converts to an area, so the
quantisation floor is the same everywhere); the nesting argument that ruled
out H3's 705 monotonicity violations is no longer exercised by a test.

## Usage

```bash
python -m scripts.analysis.v5.cli build-answer-space --run-id as01-260728-260802-mesh
python -m scripts.analysis.v5.cli classify           --run-id as01-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-answer-space  --run-id as01-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-outcome-bars \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-outcome-map \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-error-cdf --layout per-run --layout pooled \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-error-cdf --layout pooled --unanswered sentinel \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-vp-proximity -c p5 -c p25 -c all \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-rtt-cdf \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m scripts.analysis.v5.cli plot-rtt-cdf --x-scale linear --x-max 100 --x-step 10 \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh \
    --run-id as7018-ripe-mesh
python -m scripts.analysis.v5.cli plot-mtl-map --run-id as01-260728-260802-mesh \
    -m octant_cbg_hull --no-regions
./scripts/analysis/v5/create_mtl_map.sh          # every method, every mesh run

python -m scripts.analysis.v5.cli plot-ltd-model --run-id as01-260728-260802-mesh \
    -m octant_cbg_spl --fold fold_4
./scripts/analysis/v5/create_ltd_modeling_html.sh   # every method, every mesh run
METHODS="vanilla_cbg" FOLDS="fold_4" \
    ./scripts/analysis/v5/create_ltd_modeling_html.sh as01-260728-260802

python -m pytest scripts/analysis/v5/tests -q
```
