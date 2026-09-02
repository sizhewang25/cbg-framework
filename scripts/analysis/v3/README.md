# v3 analysis layer

Post-benchmark analysis over `outputs/benchmark/v2/`, implementing the paper's
answer-space definition (§7.3/§7.4) and classification criterion (§8.1).

**Layout.** Every module — libraries and CLI command bodies alike — lives in
[modules/](modules/). [cli.py](cli.py) contains no command logic; it imports
each module and calls its `register(app)`. Adding a command means adding a
module and one name to `_COMMAND_MODULES`.

| module | role |
| --- | --- |
| [modules/paths.py](modules/paths.py) | lib · run discovery + layout resolution |
| [modules/io.py](modules/io.py) | lib · K-fold merge loader and benchmark IO |
| [modules/grid.py](modules/grid.py) | lib · grid interface + registry |
| [modules/h3grid.py](modules/h3grid.py) | lib · H3 hexagons (**default**) |
| [modules/healpix.py](modules/healpix.py) | lib · HEALPix equal-area quads, nesting |
| [modules/answer_space.py](modules/answer_space.py) | cmd · `build-answer-space` |
| [modules/classify.py](modules/classify.py) | cmd · `classify` |
| [modules/venn.py](modules/venn.py) | cmd · `plot-venn` |
| [modules/map_answer_space.py](modules/map_answer_space.py) | cmd · `plot-answer-space` |

[SCHEMA.md](SCHEMA.md) documents the v2 output schemas this layer reads.

## Pipeline

```bash
# 1. Quantize targets -> seeds (answer space). Defaults to h3 res 4.
python -m scripts.analysis.v3.cli build-answer-space --all-runs

# 2. Score every method against it (distance to ALL seeds)
python -m scripts.analysis.v3.cli classify --all-runs

# 3. Set overlap of correct classifications (repeat per top-N)
python -m scripts.analysis.v3.cli plot-venn --all-runs --top-n 1
python -m scripts.analysis.v3.cli plot-venn --all-runs --top-n 3

# 4. Static map of the answer space
python -m scripts.analysis.v3.cli plot-answer-space --all-runs --us-only

# The paper's original grid, or any other rung:
python -m scripts.analysis.v3.cli build-answer-space --all-runs --grid healpix
python -m scripts.analysis.v3.cli build-answer-space --all-runs -r 3 -r 5
```

Every command takes `--grid [h3|healpix]` (default `h3`), `--resolution`/`-r`
(repeatable) and `--sweep`. Omit the resolution and each grid uses **its own**
default — `h3 res=4`, `healpix nside=128` — so a resolution can never be paired
with the wrong grid. Outputs land under `outputs/analysis/v3/<run_id>/`:

```
target-answer-space/       grid_sweep.<grid>.csv
  <grid>-<resolution>/     seeds.csv  assignments.csv  seed_mesh_km.csv  meta.json
                           answer_space_map.png
target-cls-accuracy/
  <grid>-<resolution>/     <method>_seed_distances.parquet  topn_accuracy.csv  manifest.json
                           overlap_{membership,intersections,pairwise}.top<N>.csv
                           overlap_venn.top<N>.png  overlap_upset.top<N>.png
```

e.g. `h3-4/`, `h3-3/`, `healpix-128/` side by side.

**Two axes, two filename mechanisms.** The answer space parameterizes every
number downstream of it, so both output trees are grouped by
`<grid>-<resolution>`; and every overlap artifact carries its top-N in the
filename. Without either, a sweep would overwrite one `topn_accuracy.csv` once
per rung and leave no record of which grid produced the survivor. The scheme
name is in the slug rather than just the number because `h3` res 4 and HEALPix
nside 4 are different grids that would otherwise collide. `grid_sweep` is
per-grid for the same reason. `classify` reads the grid *and* resolution from
the answer space itself rather than from the CLI, so an explicit
`--answer-space` still lands in the directory matching the grid it was built on.

**Schema is grid-neutral.** `seeds.csv` carries `grid_scheme`,
`grid_resolution` and `cell_id`, so `classify`, `plot-venn` and
`plot-answer-space` never branch on which tessellation was used. `cell_id` is
the only column whose *dtype* is grid-specific — int64 for HEALPix, canonical
hex string for H3 — and `Grid.coerce_cell_ids` is the single place that knows,
applied on load.

## The answer space (§7.3 / §7.4)

A grid quantizes the run's ground-truth targets. The grid is a **quantizer, not
the answer space**: its only job is to merge points close enough to count as one
place. Each occupied cell then contributes one **seed** at the spherical
centroid of the targets inside it, so the answer space is K *real locations*
rather than K grid squares. A coordinate is labelled by its nearest seed.

Steps 2 and 3 are pure spherical geometry, which is why the grid is swappable at
all: [modules/grid.py](modules/grid.py) defines the contract and the two
implementations supply only point→cell, scale, and cell boundaries.
`meta.json` carries the occupied-cell count down the grid's coarsening ladder —
the multi-scale concentration diagnostic §7.3 asks for.

### Choosing a grid

**H3 (`res=4`, default).** Hexagons: uniform neighbour distance, no ambiguous
edge/corner adjacency, the working grid in telecom RF analytics, and native in
ClickHouse, Postgres, BigQuery, Snowflake and Spark. `res=4` is the closest rung
to the paper's `nside=128`, so switching grids does not move the merge scale.

**HEALPix (`nside=128`).** The paper's original setting. Exactly equal-area and
exactly nested, which H3 is neither of.

| grid | cell | pitch | reading |
| --- | --- | --- | --- |
| `h3-5` | 253 km² avg | 17.1 km | Starlink service cell |
| `h3-4` | 1,770 km² avg | 45.2 km | **default** |
| `h3-3` | 12,393 km² avg | 119.5 km | metro |
| `h3-2` | 86,802 km² avg | 316.1 km | macro-region |
| `healpix-128` | 2,594 km² | 50.9 km | paper §7.3 setting |
| `healpix-64` | 10,377 km² | 101.9 km | metro |
| `healpix-32` | 41,509 km² | 203.7 km | region |
| `healpix-16` | 166,037 km² | 407.5 km | macro-region |

Pitch is defined differently per grid, on purpose: HEALPix uses `sqrt(area)`
because its cells are exactly equal-area, H3 uses hexagon centre-to-centre
(`edge × √3`) because `sqrt(area)` understates a hexagon's spacing by ~7%. Both
are the distance the straddle diagnostic compares seed pairs against, so both
have to be real distances.

`--sweep` builds a grid's whole ladder and writes `grid_sweep.<grid>.csv` beside
it, pairing what coarsening buys (fewer classes, fewer straddle candidates)
against what it costs (`intra_seed_spread_km`, the floor under every
error-distance figure).

### What each grid costs, measured

`meta.json` records both grids' costs rather than arguing them away.
`straddle_diagnostic` (both grids) counts seed pairs closer than one cell pitch
and targets whose own cell seed is not their nearest seed.
`grid_diagnostics` is grid-specific and **empty for HEALPix**, which has nothing
to disclose. For H3 it reports:

- `cell_area_km2_min` / `_max` / `_max_over_min` over the *occupied* cells. H3
  is not equal-area; measured on as01 at res 4 the ratio is **1.33**.
- `n_occupied_pentagons`. H3 has 12 pentagons per resolution, which break both
  equal area and the uniform-neighbour claim. They sit over ocean, so this reads
  0 for land targets — reporting it is how we know.
- `parent_lineage_disagreements`, per coarser rung. H3 is aperture-7 and
  hexagons cannot tile hexagons, so a parent's outer children straddle its
  boundary: `cell_to_parent` is exact on the *index* but is not a geometric
  container. Measured on as01, **20 of 399 targets** land in a different cell at
  res 3 by lineage than by re-binning, and 39 at res 2. This is why
  `occupied_cell_hierarchy` re-bins from coordinates on both grids instead of
  coarsening ids, even though HEALPix makes the shift exact and free.

**Neither grid fixes straddling, and resolution is not the lever.** Grid lines
fall where the grid falls, so a facility group spanning one yields two seeds and
two classes, and the Voronoi step cannot undo a split the grid already made.
Coarsening does not help on HEALPix because NESTED cells nest — a boundary at
nside 16 is also a boundary at 128 — so a pair separated at a coarse level can
never be merged by any nside. Measured on `as7018_us_test01`, the NY metro's 10
targets occupy 3 classes at nside 128, 2 at nside 64, and still 2 at nside 16,
where the cell is 407 km wide.

Switching grids does not help either, and the direction is instructive: `h3-4`
is *finer* than `healpix-128` (45 vs 51 km) yet yields **fewer** classes on
as7018 (K=22 vs 27, 6 singletons vs 11). Boundary **alignment**, not resolution,
decides whether a metro is split. If grouping co-located facilities is a
*requirement* rather than a tendency, no grid delivers it at any resolution and
a radius-capped linkage (the benchmark's existing `clusters/`) is the right
instrument. What a grid buys instead is alignment-independence: no clustering
run, no ordering sensitivity, and the same cell ids for any target set.

> This answer space is **not** the same as the benchmark's existing
> `clusters/` directory (radius-capped agglomerative). Numbers from the two are
> not comparable — don't mix them.

## Distance to all seeds, and top-N for free

`classify` emits, per (method, target), the distance to **every** seed. That is
deliberately more than a label: `truth_seed_rank` (how many seeds are strictly
closer than the true one) falls out of the full vector, so top-N accuracy for
*any* N is `(truth_seed_rank < N).mean()` with no recomputation. Rank 0 is
top-1.

The answer space is an **explicit input** (`--answer-space`), so predictions can
be re-scored under a different quantization without re-running anything
upstream.

## Two policies the code enforces

**Fallbacks count as failures (§7.2).** When the CBG intersection comes out
empty the pipeline falls back to the Shortest-Ping VP, and `targets.parquet`
records a coordinate *and* an `error_km` for that row. Crediting it would floor
CBG's measured accuracy at the baseline's — precisely the comparison RQ2 rests
on. So: the per-target parquet stays **neutral** (fallback rows keep their
distances, because a coordinate does exist), and the policy is applied in
`topn_accuracy.csv`, where `accuracy_topN` counts fallbacks as wrong and
`fallback_rate` beside it says how many that was.

`error_km_p50` / `error_km_p90` are computed over solved rows only despite the
unqualified name, for the same reason read the other way: a fallback row's
coordinate is the Shortest-Ping VP's, so pooling its error would describe the
baseline rather than the variant.

Keeping the parquet policy-free is what makes the fallback cost *measurable*.
On `as01`, all 106 of Vanilla CBG's fallbacks have `truth_seed_rank == 0` —
they are not bad answers, they are the baseline's good answers. Vanilla scores
164/399 = 0.411; credited it would score 270/399 = 0.677.

**K-fold test sets must be disjoint.** `load_folds` pools every fold into one
**Accuracy and error distance do not share an answer space.** The seeds and the
grid exist to define the classes, so accuracy is measured against them; error
distance has its own ground truth and is measured to the **raw target**
(`error_to_target_km`). Routing it through the seed would import the grid's
quantization into a number that needs none — with cell-centre seeds that would
have been a systematic ~17-20 km added to every *correct* answer. The parquet
also keeps `error_to_truth_seed_km`, whose difference from
`error_to_target_km` is exactly that offset per row, i.e. the per-row form of
`intra_seed_spread_km`. One consequence is a useful check: re-quantizing changes
accuracy but must leave `error_km_*` bit-identical, and it does across `h3-4`
and `healpix-128` on all four runs.

frame labelled with `fold`, and raises if a `target_id` appears twice — a leak
would double-count targets in every pooled figure downstream.

## Shortest-Ping is a method, not a footnote

The baseline is scored through the same answer space, the same scorer and the
same fold partition as every variant (§5). Its estimate is the lowest-RTT VP's
own coordinate, read from `eval_source`'s `shortest_ping_vp_lat`/`_lon`. That
choice is fold-independent: K-fold splits targets, not VPs, so every VP is
available in every fold.

## Overlap figures

`plot-venn` turns each method into the set of targets it classified correctly,
which exposes the trade an aggregate accuracy number hides: which targets a
variant *wins* over the baseline, and which it *loses* that the baseline already
had.

The headline is `overlap_venn.top<N>.png` — a 2-set Venn of **Shortest-Ping vs
"≥1 CBG works"**. The CBG-only region counts rescues, the Shortest-Ping-only
region counts regressions. Its subtitle names the CBG pool size, which is not
decoration: on `as7018_us_test01` the default method set is 16 combos including
11 ablation arms, and the CBG-only region reads 40 over all of them versus 31
over the five published variants. Use `--method` to restrict the pool.

`overlap_upset.top<N>.png` carries the per-method detail, since at six methods a
Venn would need 63 regions. Both bar charts are suppressed and their magnitudes
written as text instead, because the exact values are what a reader wants here:

* **`Intersections (%)`** above each column — **exact, disjoint** intersections
  ("these methods correct, all others wrong"). They partition the target set and
  sum to 100%.
* **`True (%)`** to the right of each row — that method's success rate. These
  are set totals, they overlap, and they do *not* sum to 100%. The two headers
  exist because the numbers are not the same kind of quantity.

Method rows are pinned to `PREFERRED_ORDER` (`sort_categories_by=None`) so the
dot pattern means the same thing in every run's figure. Columns are ranked by
intersection size, largest first, so the shares read monotonically left to
right; that ordering *is* data-dependent, so compare columns across runs by
their dots rather than by position. Every method except Shortest-Ping is
labelled `... CBG`, including the ablation arms, which otherwise read as bare
combo ids.

`--venn-method` adds an explicit per-method Venn on request.

Venns are drawn **unweighted** — fixed-size circles with counts and percentages
written into the regions. The correctness sets are routinely nested
(`all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` on every run we have), and no
area-proportional layout can render nesting; area-proportional drawing warned on
every run for exactly that reason. Since a Venn cannot show the outside region,
the "none correct" count is annotated below the figure.

Everything is backed by `overlap_membership.top<N>.csv`, so every count is
checkable without reading a figure.

## Answer-space map

`plot-answer-space` draws the occupied cells (filled, one colour per seed), the
nearest-seed class boundaries, their targets, and their seed centroids on one
cartopy panel. It exists to make the §7.3 straddle cost visible: a facility group spanning a grid line is quantized
into two adjacent cells and two classes, which reads instantly as two touching
filled cells and is hard to believe from a scalar. Only occupied cells are
drawn — the full grid (288,122 cells at `h3-4`, 196,608 at `healpix-128`) is
unreadable at continental scale.

Cell rings come from `Grid.cell_boundaries` and are **ragged**: 4 sides × `step`
for a HEALPix quad, 6 for an H3 hexagon, 5 for one of its 12 pentagons. They
arrive already in `(lon, lat)` degrees and already made contiguous across the
antimeridian, so the plot layer neither re-wraps longitudes nor imports astropy.

The caption states that occupied cell and class are in one-to-one
correspondence, because there is no clustering algorithm involved and the
benchmark also ships an older agglomerative `clusters/` space.

Rendering both grids is the quickest way to see the previous section's point: the
`as7018` map shows touching cell pairs at LA, Dallas, New York and DC on either
one.

## Tests

### The class-region overlay

The red dashed boundaries are the **classifier's own top-1 decision
boundary**, not an illustration: `classify` labels a prediction by
`argmin` over great-circle distance to the K seeds, so top-1 classification is
nearest-seed labelling and its boundary is the Voronoi diagram of the seeds.
`margin_km` in `seeds.csv` is the scalar summary of the same geometry. Pass
`--no-voronoi` to drop the layer.

Note the two layers sit at wildly different scales, which is the point: a grid
cell is ~45 km while a class region is hundreds of km across. The answer space
is a coarse nearest-seed partition whose *seeds* are placed by a fine quantizer.

Three implementation facts are load-bearing, each measured rather than assumed:

* **The diagram is computed in an azimuthal-equidistant projection**, not in
  lon/lat. A degree is not a distance — at 38 N one degree of longitude is
  87.6 km against 111.2 km of latitude, a 1.27x anisotropy rising to 1.49x at
  48 N — so a lon/lat Voronoi tilts every bisector and misassigns ~10.5% of the
  frame's area, displacing the drawn line by up to ~210 km. The projection
  brings that to 0.35-0.38%.
* **Edges are pre-split at 200 km** before cartopy reprojects them. Cartopy
  only densifies a long straight edge ~1.2x, which left the inked line up to
  54.8 km (21 px) off the true bisector; pre-splitting brings it to 8.2 km
  (3 px), the projection's own floor. 100 km and 25 km buy nothing more.
* **The clip box is padded 25% beyond the frame.** Clipping exactly at the
  frame makes the outermost cells' outlines ink a line along the map border
  that is an artifact of the crop, not of the classifier.

Geometry is reused from `scripts/visualization/cluster/voronoi.py`
(`clipped_voronoi_cells`), which is CRS-agnostic. The landmass helpers there are
deliberately *not* used: this partition is defined over the whole sphere, and
`build_landmass_voronoi` also drops seeds outside its boundary, which would move
boundaries that are visible — an off-frame seed still shapes an on-frame line.

```bash
python -m pytest scripts/analysis/v3/tests/ -q
```
