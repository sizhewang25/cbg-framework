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
| [modules/config.py](modules/config.py) | lib · unified run config → per-command defaults |
| [modules/grid.py](modules/grid.py) | lib · grid interface + registry |
| [modules/h3grid.py](modules/h3grid.py) | lib · H3 hexagons (**default**) |
| [modules/healpix.py](modules/healpix.py) | lib · HEALPix equal-area quads, nesting |
| [modules/answer_space.py](modules/answer_space.py) | cmd · `build-answer-space` |
| [modules/classify.py](modules/classify.py) | cmd · `classify` |
| [modules/venn.py](modules/venn.py) | cmd · `plot-venn` |
| [modules/diagram/](modules/diagram/) | lib · the overlap figures `plot-venn` assembles |
| [modules/map_answer_space.py](modules/map_answer_space.py) | cmd · `plot-answer-space` |
| [modules/pareto.py](modules/pareto.py) | cmd · `plot-pareto` |

[modules/diagram/](modules/diagram/) is the one package here, split from
`venn.py` when that module passed 2,000 lines. The split follows the bargain
each figure makes rather than the file it grew in:

| module | role |
| --- | --- |
| [diagram/common/labels.py](modules/diagram/common/labels.py) | method names, artifact filenames, region letters |
| [diagram/common/palette.py](modules/diagram/common/palette.py) | the validated variant → hue map |
| [diagram/common/membership.py](modules/diagram/common/membership.py) | the boolean matrix every figure is computed from, and the pooling rules |
| [diagram/common/tables.py](modules/diagram/common/tables.py) | exact intersection counts + the generic Venn-tool spec |
| [diagram/common/draw.py](modules/diagram/common/draw.py) | shared matplotlib primitives; selects the Agg backend |
| [diagram/venn/classic.py](modules/diagram/venn/classic.py) | 2-/3-set Venn + the Shortest-Ping vs ≥1-CBG collapse |
| [diagram/venn/ring.py](modules/diagram/venn/ring.py) | the `n`-way ring template and its coverage table |
| [diagram/venn/upset.py](modules/diagram/venn/upset.py) | the UpSet plot |
| [diagram/euler/layout.py](modules/diagram/euler/layout.py) | area-proportional fitting + its fit table |
| [diagram/euler/plot.py](modules/diagram/euler/plot.py) | drawing a fitted layout, and label placement |

`venn.py` keeps the two `render_*` functions and the CLI, and **re-exports the
whole surface** (`__all__`) so `pareto.py`, the tests and any future caller
still have one name to import. The split is pure code motion: every artifact
`plot-venn` writes, PNGs included, is byte-identical across it.

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
# pooled across the three operator datasets -> _cross/venn-diagram/
python -m scripts.analysis.v3.cli plot-venn \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802

# 4. Static map of the answer space
python -m scripts.analysis.v3.cli plot-answer-space --all-runs --us-only

# 5. Accuracy vs cost across datasets (colour = variant, symbol = dataset)
python -m scripts.analysis.v3.cli plot-pareto \
    --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802 \
    --grid h3 -r 4 --cost runtime --top-n 1

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
                           overlap_venn_spec.top<N>.json
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

## One config per run

The pipeline above repeats `--grid`, `-r` and the top-N on five commands, and
they have to agree or the artifacts land in mismatched directories. A unified
config declares them once, `configs/<run_id>.yaml`:

```yaml
run_id: as7018_us_test01
benchmark: {}                  # reserved; see below
analysis:
  common:            {grid: h3, resolution: [4]}
  build-answer-space: {}
  classify:          {topn: "1,3"}
  plot-venn:         {top_n: 1}
  plot-answer-space: {us_only: true}
  plot-pareto:       {top_n: 1, cost: runtime, cost_stat: p50}
```

```bash
python -m scripts.analysis.v3.cli --config configs/as7018_us_test01.yaml classify
#                                 ^^^^^^^^ before the command, not after
```

`--config` belongs to the *group*, unlike `scripts/analysis/cli.py`'s
per-command `--config`. That is what buys the per-command sub-blocks: the
callback in [cli.py](cli.py) parses the file into click's own
`ctx.default_map`, which is keyed by subcommand name, so **no command signature
knows a config exists** and precedence is click's rather than hand-rolled:

    explicit flag  >  config  >  CLI default

It is a defaults mechanism and nothing more — `topn_accuracy.csv` is
byte-identical between `--config <cfg> classify` and the equivalent explicit
flags.

### Four rules, all enforced

Everything is checked against the *live* CLI:
[config.py](modules/config.py) introspects the registered click commands, so
block names, key names, `run_id` arity and which params are paths are read off
the real signatures and cannot drift from them.

1. **Sub-block names are command names verbatim**, as `--help` prints them
   (`plot-answer-space`, not `plot_answer_space`).
2. **Keys are parameter names verbatim** — the long flag minus `--`, dashes as
   underscores. Including the negative-sense ones (`no_baseline`,
   `no_voronoi`) and the `topn` / `top_n` inconsistency between `classify` and
   `plot-venn`. One mechanical rule, no inversion layer, so the validator is
   exact.
3. **`common:` applies a key only where the param exists.** `grid`,
   `resolution`, `sweep` and the two roots are on all five commands; `top_n`
   and `method` on two. A `common:` key matching *no* command is an error.
4. **Relative paths resolve against the repo root**, matching this layer's own
   `DEFAULT_OUTPUTS_ROOT` / `DEFAULT_ANALYSIS_ROOT`. The wider repo resolves
   config paths against three different bases (cwd, the config's own directory,
   repo root), so v3 picks one.

Rule 2 is the load-bearing one, because **click silently ignores a
`default_map` key it does not recognize**. A typo'd `topN:` would be a no-op
rather than an error, which is the failure mode this module exists to prevent;
`test_config.py` pins it, and also validates every checked-in `configs/*.yaml`
on each run.

### Two couplings the config has to preserve

Both come from the group callback running *before* the subcommand's arguments
are parsed — `ctx.args` is empty there — so both are read from `sys.argv`:

- **`--all-runs` drops the config's `run_id`.** Every command enforces
  `--run-id` XOR `--all-runs`, so without this the documented `--all-runs`
  pipeline would collide with any config naming a run and be rejected.
- **`--grid` alone drops the config's `resolution`.** A resolution is
  meaningful only against its own grid — h3 res 4 is 45 km, HEALPix nside 4 is
  1630 km — which is why the CLI defaults it per grid. Inheriting the other
  grid's number would silently build an answer space nobody asked for. Pass
  `-r` too for a specific rung.

`run_id` may be a list. `plot-pareto` and `plot-venn` take `--run-id` repeatably
because both pool datasets, so `configs/cross-as01-as03.yaml` names three runs
and serves them both; the per-run commands say so by name rather than failing on
a type.

### `benchmark:` is reserved and empty

Nothing reads it yet. `scripts/benchmark/v2/Snakefile` takes flat top-level keys
(`config["source"]`, `config["combos"]`), so nesting them under a section would
break it — the benchmark run is still driven by
`configs/benchmark/<run_id>.yaml`. The section is declared so the file has one
obvious home for it when that is repointed.

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

Keeping the parquet policy-free is what makes the fallback cost *measurable*.
On `as01`, all 106 of Vanilla CBG's fallbacks have `truth_seed_rank == 0` —
they are not bad answers, they are the baseline's good answers. Vanilla scores
164/399 = 0.411; credited it would score 270/399 = 0.677.

**K-fold test sets must be disjoint.** `load_folds` pools every fold into one
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
written into the regions. The *collapsed* sets are nested
(`all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` on every run we have), and no
area-proportional layout can render nesting; area-proportional drawing warned on
every run for exactly that reason. Since a Venn cannot show the outside region,
the "none correct" count is annotated below the figure.

Everything is backed by `overlap_membership.top<N>.csv`, so every count is
checkable without reading a figure.

`overlap_venn_spec.top<N>.json` is the same numbers in the shape a generic Venn
drawing tool asks for — a set count, one entry per set with a letter id, a name
and a size, and one entry per combination of two or more sets:

```json
{
  "number_of_sets": 6,
  "n_targets": 1269,
  "n_none": 189,
  "relation_convention": "cumulative",
  "sets": [{"id": "A", "name": "Shortest-Ping", "method": "shortest_ping", "size": 604}],
  "relations": {"A ^ B": 603, "A ^ B ^ C": 129}
}
```

Relations are **cumulative** — `A ^ B` is the full `|A ∩ B|` and counts targets
that are also in `C`. That is what makes `size` and `relations` satisfy
inclusion-exclusion, so a tool can solve for the regions itself (a test checks
the 63 terms sum back to the 1,080 union). `venn_spec(membership,
exclusive=True)` switches to the disjoint reading `intersection_table` and the
ring figure use, where `A ^ B` means "A and B and nothing else"; the convention
is recorded in the document rather than left to be inferred. Empty combinations
are present with a zero, so a missing key never has to be told apart from an
empty region. Written for up to 8 sets — past that (as7018's 17 methods) it is
skipped, since 131,054 relations describe a figure no tool draws.

### Pooling several runs

Passing `--run-id` more than once pools the runs into one population and writes
to a `_cross/` directory instead of back into any run:

```
_cross/venn-diagram/as01+as02+as03/
  overlap_ring_venn.<grid>-<res>.top<N>.png
  overlap_ring_coverage.<grid>-<res>.top<N>.csv
  overlap_euler.<grid>-<res>.top<N>.png
  overlap_euler_fit.<grid>-<res>.top<N>.csv
  overlap_{venn,upset}.<grid>-<res>.top<N>.png
  overlap_{membership,intersections,pairwise}.<grid>-<res>.top<N>.csv
  overlap_venn_spec.<grid>-<res>.top<N>.json
  manifest.<grid>-<res>.top<N>.json
```

Cross-run filenames carry the grid slug; per-run ones do not, because the per-run
directory is already `target-cls-accuracy/<grid>-<res>/` while this one is keyed
by dataset set alone.

Pooling **stacks** the runs — as01/02/03 have disjoint target sets (all three
pairwise intersections are empty), so a target belongs to exactly one run and
1,269 pooled targets is 399 + 412 + 458. Rows are keyed `<run_id>::<target_id>`
anyway, since disjointness is a property of these datasets rather than of the
schema. Every run must have scored **exactly** the same methods; equality rather
than a subset test, or the result would depend on which run was named first —
as01 + as7018 would either error or silently drop as7018's 10 ablation arms
depending on the order. `--method` pins a shared subset explicitly.

`overlap_ring_venn` is the conventional presentation "6-way Venn": six equal
circles on a ring, one per method, with the **exact target count printed in every
region**. The circles are a **fixed template** — same centres, same radii, every
run — so nothing about a region's size or position carries data and every
quantity on the figure is a label. That is deliberate: three area-encoding forms
were tried first and each either lost the higher-order regions or implied a
containment the sets do not have (a least-squares Euler fit matched the 15
pairwise overlaps to a stress of 0.0034 but still misplaced ~41% of targets
across all regions). Drawn only for 3-8 methods; below that `plot_venn` renders
the sets exactly.

**Not every observed intersection has a region.** `n` circles on a ring realize
`n*(n-1)+1` of the `2**n - 1` combinations — 31 of 63 at six methods, and they
are exactly the subsets contiguous around the ring. A mathematical 6-set Venn
needs all 63 and cannot be drawn with circles at all, so this is a property of
the layout, not of the implementation. On as01+as02+as03 that leaves 9
combinations (195 targets, 18.1% of the 1,080 solved) with nowhere to go. Both
the figure's footnote and `overlap_ring_coverage.csv` name them; the CSV lists
every non-empty intersection with a `drawn` flag, and its drawn rows sum to what
the figure's labels sum to. Targets no method solved are not in it — they sit
outside the union rather than in an undrawable region — and are reported as
`n_targets_none_correct` in the manifest.

### The Euler diagram

`overlap_euler` is the ring's opposite. The ring fixes the geometry and prints
every number; this fits the geometry **to** the numbers and prints none of them:

* a circle's **area** is the share of targets that method gets right;
* the **area two circles share** is the share both get right;
* two methods that are never right about the same target are drawn **apart**;
* a method whose correct set falls inside another's is drawn **inside** it.

Nothing is written in a region, and no number appears anywhere. At six sets
there are thirty-one regions, and a combination is legible from the arcs that
bound it — the relationships are read rather than looked up.

Each method's **name alone** labels its circle, in that method's colour, placed
inside the circle: in the set's own exclusive lobe when the lobe is at least
`LOBE_LABEL_SHARE` of it, otherwise just inside its boundary on the bearing
away from the layout's centre, which is the arc least covered by the others.
Because every label sits on the circle it names, the figure needs no leader
lines. Near-coincident circles — Shortest-Ping and SoI CBG differ by one target
— leave on almost the same bearing, so the angles are spread until the labels
clear `LABEL_MIN_GAP`, each staying on its own circle.

The letters that key `overlap_euler_fit.csv`'s `region` column are **not** drawn
here; that table names the methods in full alongside them, and the ring's own
region key (`overlap_ring_regions.png`) is where the letter legend lives.

Each set's label carries its **share of the population** under its name. The
number is `pi * r**2` — literally the circle's drawn area, not a second
derivation of it — so the printed value and the ink cannot drift apart.
**Intersections are not labelled**: a set's area is exact, but an intersection's
is fitted and disagrees with its observed share by up to the figure's error, so a
number inside one would contradict the area holding it. The caption reports that
error in aggregate instead.

Circles are drawn **largest first**, so the biggest set sits at the bottom of the
stack and the smallest on top. At equal `zorder` matplotlib draws in insertion
order, so this is the whole mechanism; without it Octant-Hull at 60.4% is laid
over Vanilla at 34.0% and buries its outline.

The space outside every circle is labelled **None**, with its share — the targets
no method placed correctly, 14.9% here, the complement of the union's 85.1%. It
is named because blank space otherwise reads as "nothing here". The label sits
centred just below the lowest circle, in data units so it tracks the layout;
`EULER_MARGIN` guarantees that band is clear of every circle whatever the fit
produced. **Its area alone is not to scale** — the space outside the union is
leftover frame rather than a fitted share, making it the one label here whose
number and ink are unrelated, which is why it is drawn muted and outside.

The two figures are complements and neither replaces the other. The ring can
state that the six-way region holds exactly 12.7% and cannot show that
Shortest-Ping's correct set sits inside SoI CBG's; the Euler layout shows that
containment at a glance and cannot quantify anything.

**How faithful it is, measured.** `n` circles have `2n` free parameters — the
radii are spoken for by the set sizes — against `2**n - 1` regions, so past two
sets the system is overdetermined and some combination is always misdrawn. The
figure prints the size of its own error, as the share of targets sitting in a
region the picture assigns to a different combination:

| methods | placed correctly | largest pair error |
|---|---|---|
| 3 (Shortest-Ping, Vanilla, Spotter) | 98.0% | 0.5% |
| 6 (all) | 84.0% | 5.4% |

Three circles are essentially exact; six are a blob, because on this data every
one of the fifteen pairs overlaps and four of the six sets cover more than 40%
of the population each. Use `--method` to cut to the sets a question actually
needs — that is the lever that makes this figure sharp. `overlap_euler_fit.csv`
lists every combination with its observed and drawn share and the difference,
including the regions the layout invents: rows with `n_targets == 0` and a
positive `delta` are area the picture asserts and the data denies (all under
0.25% of the population here).

The caption under the figure states the area encoding and the fit's measured
error. `plot_euler(..., caption=False)` drops it for slide use; it defaults on
because a layout that misplaces 16% of targets and says nothing about it asserts
an exactness it does not have.

The fit is deterministic — the starting layout is classical MDS on the ideal
pairwise separations, the restarts are seeded, and the search is bounded — so
the same membership matrix always produces the same figure. Empty combinations
are weighted `EULER_EMPTY_WEIGHT` times a non-empty one in the objective,
which is what buys the "0% means no overlap" rule at the cost of a little area
accuracy elsewhere.

`--ring-order` sets the clockwise order from 12 o'clock and must be a
*permutation* of the scored methods, never a subset: a method left off the ring
would still be in the membership matrix, so its targets would vanish from the
figure without being counted as undrawn either. The default is display order, so
the figure means the same thing in every run — the rule the UpSet column order
already follows. Which combinations land on a region *does* depend on this order
(display order draws 885 of the 1,080 solved targets; reordering to maximise
coverage reaches 959), which is why it is recorded in the manifest.

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

Rendering both grids is the quickest way to see the previous section's point: the
`as7018` map shows touching cell pairs at LA, Dallas, New York and DC on either
one.

## Accuracy vs cost

`plot-pareto` joins `topn_accuracy.csv` to the per-target cost the benchmark
already records in `targets.parquet`
(`{ltd,mtl,ctr}_{ms,alloc_peak_bytes,rss_peak_bytes}`) and draws the two against
each other. Artifacts land outside any single run's tree, because they are a
function of the dataset *set*:

```
_cross/cost-accuracy/as01+as02+as03/
  pareto_<cost>[-throughput].<grid>-<res>.top<N>[.amortized].{csv,png,json}
```

Every axis that parameterizes the numbers is in the path — dataset set, cost
channel, x-presentation, quantization, top-N, fit policy — or a sweep overwrites
its own output. A single-run figure landing on top of a three-dataset one is
exactly the bug this prevents.

### How to read the figure

Three encodings, each carrying exactly one thing:

* **Colour is the CBG variant.** Every variant sits at one x — its median cost
  across the datasets — with a horizontal bar spanning the cost range it
  actually occupied.
* **Symbol (and linestyle) is the dataset.** One marker per dataset at that
  variant's x, at the accuracy the variant scored *on that dataset*.
* **One grey polyline per dataset** connects that dataset's markers in cost
  order, giving each dataset its own cost/accuracy curve.

**Top-left is the most cost-effective corner**, and the vertical spread inside
one colour is that variant's cross-dataset stability. That spread is the reason
the figure is cross-dataset at all: accuracy moves more between datasets than
between some variants, and by very different amounts. Measured top-1 on `h3-4`
over as01/02/03, `spotter_cbg` varies 6pp (0.386-0.448) while
`million_scale_cbg` swings 30pp (0.367-0.662) and the baseline 27pp. A
single-dataset number hides that completely, so `accuracy_range` is a
first-class CSV column.

A variant missing from one dataset is marked `partial_coverage` and **skipped in
that dataset's polyline only** — interpolating across the gap would draw a
measurement that does not exist.

**Shortest-Ping costs exactly 0.** It runs no distance model, no
multilateration and no centroid step — the method is an `argmin` over RTTs the
measurement already produced. The x axis is `symlog` so that point is
renderable at all. (On `--x throughput` it is unplottable, being infinitely
fast, so it drops out with a note and keeps its exact `inf` in the CSV.)

### The palette is validated, not chosen

Colour carries identity, so the palette is a correctness question. Checked with
the dataviz skill's `validate_palette.js` against the reference 8-hue
categorical theme, `--pairs all` on white — the right check here, since every
variant is visible at once:

* Every 6/7/8-slot prefix of that theme **fails**: green vs orange is ΔE 3.2
  under protanopia. Orange is the hue that had to go.
* An exhaustive search over the theme's hues found exactly **two** passing
  6-subsets; `_VARIANT_HUES` is the better one, worst ΔE 6.9 (deutan) /
  7.6 (tritan).

ΔE 6.9 sits in the 6-8 band that is legal *only* with secondary encoding, which
is satisfied three times over: each variant owns its own x column, the legend
names every one, and the CSV is the table view. Aqua, yellow and magenta are
also below 3:1 on white — a contrast warning that obliges visible labels or a
table view, again the legend and the CSV. No 6-subset of this theme clears 3:1
for all six (only five hues do), so at six variants that is unavoidable rather
than a shortcut.

Two properties the tests pin, because both are silent when broken:

* **Hues follow identity, never rank.** `method_colors` reads a table built once
  from `venn.PREFERRED_ORDER`, so filtering with `--method` cannot repaint the
  survivors and a variant is the same colour in every figure of a sweep.
* **`octant_cbg_spl` and `octant_cbg` share a hue.** They are one paper variant
  whose combo id differs per run, exactly as `venn.LABELS` already encodes.

Past six variants a 9th series is never a generated hue: as7018's 11 ablation
arms fold into one grey `other` series with a warning suggesting `--method`.
Their identities stay in the CSV.
### Four cost-model policies, each measured rather than assumed

* **Cost is over every target** (`--cost-rows all`, default). `ctr_ms` is null
  on exactly the FALLBACK rows, so a solved-only cost reads 22% higher on
  `vanilla_cbg` (64.89 vs 53.11 ms, as02). `accuracy_topN`'s denominator is
  every target, so the cost denominator has to be too.
* **Memory reduces with `max`, not `sum`** (`--memory-reduce`).
  `benchmark/v2/instrument.py` resets tracemalloc *inside* each stage and the
  RSS sampler returns a delta, so the columns are per-stage peaks: `max` is the
  pipeline high-water mark, `sum` the no-release upper bound. Summing also
  triples the ~41.6 KB tracemalloc pedestal, which is bookkeeping rather than
  work. Runtime genuinely sums. (`plot_accuracy_cost_box.py` sums memory; that
  is deliberately not inherited.)
* **`memory_alloc` is the default memory channel.** `*_rss_peak_bytes` is
  floored at one 4096-byte page for every stage faster than the 5 ms sampler, so
  it cannot rank methods at p50 — pass `--cost-stat p95` to use it.
* **Fit cost is excluded** and reported separately (`fit_ms_mean`,
  `fit_ms_per_target`). The fold size is an experimental artifact, not a
  deployment denominator: at 82 targets/fold `vanilla_cbg`'s 2-3 s fit amortizes
  to 38 ms/target, comparable to its own inference cost, while at a million
  targets it vanishes. `--amortize-fit` folds it in and marks the filename.

**On the memory axis, read the horizontal bar, not the ordering.** Memory is
effectively bimodal (~0.06 MB for the cheap CTR, ~24.09 MB for
`monte_carlo_medoid`) and the three heavy variants sit within **160 bytes** of
each other, so their left-to-right order there is measurement noise, not a
result. The cost bar is drawn precisely so that overlap is visible instead of
being implied away by a single point.

`million_scale_cbg`'s stages run in ~0.2-1 ms, the regime where
`tracemalloc.start()`/`stop()` per stage inflates the very timing it measures.
Treat its runtime as an upper bound.

Runs spanning more than one `setup` are refused (`--allow-mixed-setups` to
override): as01/02/03 are `anchors_to_probes` and `as7018_us_test01` is
`probes_to_anchors`, which swap the VP and target roles ([SCHEMA.md](SCHEMA.md)
§7).

## Tests

```bash
python -m pytest scripts/analysis/v3/tests/ -q
```
