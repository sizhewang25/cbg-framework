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
| [modules/io.py](modules/io.py) | lib · K-fold merge loader, benchmark IO, the shared shortest-ping-VP reader |
| [modules/config.py](modules/config.py) | lib · unified run config → per-command defaults |
| [modules/grid.py](modules/grid.py) | lib · grid interface + registry |
| [modules/h3grid.py](modules/h3grid.py) | lib · H3 hexagons (**default**) |
| [modules/healpix.py](modules/healpix.py) | lib · HEALPix equal-area quads, nesting |
| [modules/mapping.py](modules/mapping.py) | lib · shared cartopy layers, the seed Voronoi, great-circle polylines |
| [modules/cross.py](modules/cross.py) | lib · cross-run directories, dataset-set slugs, the setup and disjoint-target guards |
| [modules/answer_space.py](modules/answer_space.py) | cmd · `build-answer-space` |
| [modules/bipartite.py](modules/bipartite.py) | cmd · `build-bipartite-graph` |
| [modules/classify.py](modules/classify.py) | cmd · `classify` |
| [modules/proximity.py](modules/proximity.py) | cmd · `build-proximity` |
| [modules/breakdown.py](modules/breakdown.py) | cmd · `breakdown-accuracy` |
| [modules/confusion.py](modules/confusion.py) | cmd · `confusion-density` |
| [modules/accuracy_table.py](modules/accuracy_table.py) | cmd · `table-accuracy` |
| [modules/headline_table.py](modules/headline_table.py) | cmd · `table-headline` |
| [modules/figure_outcome_bars.py](modules/figure_outcome_bars.py) | cmd · `plot-outcome-bars` |
| [modules/venn.py](modules/venn.py) | cmd · `plot-venn` |
| [modules/diagram/](modules/diagram/) | lib · the overlap figures `plot-venn` assembles |
| [modules/map_answer_space.py](modules/map_answer_space.py) | cmd · `plot-answer-space` |
| [modules/map_bipartite.py](modules/map_bipartite.py) | cmd · `plot-bipartite-graph` |
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

## Column naming

Three conventions, applied to every column this layer emits. They exist because
the layer routinely holds two distances that differ only in *which endpoint*
they measure to, and a name that omits the endpoint is a bug waiting to be
introduced by autocomplete.

**1. A distance names both endpoints: `<from>_to_<to>_km`.** `closest_vp_km` is
ambiguous — closest to the target, or to its seed? — and the two answers select
different VPs. Columns carried in from v2's `eval_source` are renamed on the way
across (`closest_vp_km` → `closest_vp_to_tg_km`) with their source names
recorded in [SCHEMA.md](SCHEMA.md). Exempt: quantities that are not
point-to-point, such as `tg_seed_margin_km` (half a gap, matching `seeds.csv`'s
`margin_km`).

**2. The prefix names the axis, not the aggregation.** `tg_seed_*` for
quantities that are a property of the target's seed, `sping_vp_*` for
quantities that are a property of the selected shortest-ping VP. A `min_vp_*`
prefix was rejected: two different minimizations (over seed distance, over seed
rank) select two different VPs in general, so "min" names the operation while
saying nothing about which question was asked.

**3. `tg` for the target, `sping` for the shortest-ping baseline.** The seed a
target falls in is `tg_seed`, not `truth_seed` — "truth" is already spoken for
by the raw ground-truth coordinate, and the seed is a quantization of it rather
than the thing itself. v2's `truth_centroid_*` and `shortest_ping_*` spellings
are legacy and are **not** a compatibility constraint on v3; the rename stops at
this layer's boundary.

## Pipeline

```bash
# 1. Quantize targets -> seeds (answer space). Defaults to h3 res 4.
python -m scripts.analysis.v3.cli build-answer-space --all-runs

# 2. Read the VP side and the measured edges -> the §7.3 dataset geometry
python -m scripts.analysis.v3.cli build-bipartite-graph --all-runs
python -m scripts.analysis.v3.cli plot-bipartite-graph --all-runs --us-only

# 3. Score every method against the answer space (distance to ALL seeds)
python -m scripts.analysis.v3.cli classify --all-runs

# 3b. Label each target with the VP proximity diamond -> the §8.1 strata
python -m scripts.analysis.v3.cli build-proximity --all-runs

# 3c. Cross correctness with the strata, and look at what the mistakes are
python -m scripts.analysis.v3.cli breakdown-accuracy --all-runs
python -m scripts.analysis.v3.cli confusion-density --all-runs

# 3d. The §8.1 headline. Operator and public runs get separate tables (§7.3)
python -m scripts.analysis.v3.cli table-accuracy \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802
python -m scripts.analysis.v3.cli table-accuracy --run-id as7018_us_test01

# 3e. The same numbers as the paper prints them: dataset *types* down the rows,
#     methods across the columns, best-in-row marked. Each type is led by a row
#     pooling its datasets' targets, with the per-AS breakdown beneath it. The
#     TRAFFIC-WEIGHTED group is reserved (no weighted run exists yet).
python -m scripts.analysis.v3.cli table-headline \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802

# 3e2. The same numbers as bars: each is one method's whole target set split
#      into correct / wrong / fallback / error. Fill is the method, hatch is the
#      dataset type. `compare` breaks the pooled panel out per dataset.
python -m scripts.analysis.v3.cli plot-outcome-bars --layout pooled --layout compare \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802

# 3f. Error-distance CDF per method, log x, fallbacks excluded
python -m scripts.analysis.v3.cli plot-error-cdf \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802

# 3g. Where the two metrics disagree: coordinate error vs class error
python -m scripts.analysis.v3.cli plot-error-vs-cells --all-runs
python -m scripts.analysis.v3.cli plot-error-vs-rank  --all-runs
# two or more --run-id go cross-dataset -> _cross/error-vs-class/
#   --layout pooled  (default) the six-panel grid over the merged 1,269 targets
#   --layout compare           one panel per (method, dataset)
python -m scripts.analysis.v3.cli plot-error-vs-rank \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802 \
  --layout pooled --layout compare

# 4. Set overlap of correct classifications (repeat per top-N)
python -m scripts.analysis.v3.cli plot-venn --all-runs --top-n 1
python -m scripts.analysis.v3.cli plot-venn --all-runs --top-n 3
# pooled across the three operator datasets -> _cross/venn-diagram/
python -m scripts.analysis.v3.cli plot-venn \
  --run-id as01-260728-260802 --run-id as02-260728-260802 --run-id as03-260728-260802

# 5. Static map of the answer space
python -m scripts.analysis.v3.cli plot-answer-space --all-runs --us-only

# 6. Accuracy vs cost across datasets (colour = variant, symbol = dataset)
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
bipartite-graph/
  <grid>-<resolution>/     vp_nodes.csv  target_nodes.csv  edge_segments.csv
                           edge_length_cdf.csv  pairwise_distance_cdf.csv  meta.json
                           bipartite_nodes_map.png  bipartite_flows_map.png
                           distance_cdf.png
target-proximity/
  <grid>-<resolution>/     target_labels.csv  meta.json
target-cls-accuracy/
  <grid>-<resolution>/     <method>_seed_distances.parquet  topn_accuracy.csv  manifest.json
                           accuracy_by_flag.csv  accuracy_by_taxonomy.csv
                           confusion_by_density.csv  confusion_pairs.csv
                           overlap_{membership,intersections,pairwise}.top<N>.csv
                           overlap_venn_spec.top<N>.json
                           overlap_venn.top<N>.png  overlap_upset.top<N>.png
```

e.g. `h3-4/`, `h3-3/`, `healpix-128/` side by side.

Cross-run artifacts land in `_cross/<kind>/<dataset-set>/` instead, since their
numbers are a function of a *set* of runs and belong to none of them:

```
_cross/accuracy-table/<dataset-set>/  accuracy_table.<grid>.{csv,md,manifest.json}
                                      dataset_context.<grid>.csv
                                      headline_top{1,3}.<grid>.{csv,md}
                                      headline.<grid>.manifest.json
                                      outcome_bars[_by_dataset].<grid>.top<N>.{png,csv}
                                      outcome_bars.<grid>.top<N>.manifest.json
_cross/cost-accuracy/<dataset-set>/   pareto_<cost>.<grid>.top<N>.{csv,png,json}
_cross/venn-diagram/<dataset-set>/    overlap_*.<grid>.top<N>.*
_cross/error-vs-class/<dataset-set>/  error_vs_{cells,rank}.<grid>.png
                                      error_vs_{cells,rank}_by_dataset.<grid>.png
                                      <stem>_{points,bands}.<grid>.csv
                                      <stem>.<grid>.manifest.json
```

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

The pipeline above repeats `--grid`, `-r` and the top-N on seven commands, and
they have to agree or the artifacts land in mismatched directories. A unified
config declares them once, `configs/<run_id>.yaml`:

```yaml
run_id: as7018_us_test01
benchmark: {}                  # reserved; see below
analysis:
  common:               {grid: h3, resolution: [4]}
  build-answer-space:    {}
  build-bipartite-graph: {}
  plot-bipartite-graph:  {us_only: true}
  classify:             {topn: "1,3"}
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
   `resolution`, `sweep` and the two roots are on all seven commands; `top_n`
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
place. Each occupied cell then contributes one **seed at the cell's own
centre**: choosing a grid at a resolution *is* the granularity claim, so the
grid decides both which targets are one place and where that place is. A seed
therefore never depends on which targets happened to land in the cell — the same
cell yields the same seed in every run. A coordinate is labelled by its nearest
seed.

Everything past cell membership and cell centre is pure spherical geometry, which
is why the grid is swappable at all: [modules/grid.py](modules/grid.py) defines
the contract and the two implementations supply only point→cell, cell→centre,
scale, and cell boundaries.
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
are compared against seed-to-seed distances, which are now distances between
occupied cell centres, so both have to be real distances.

`--sweep` builds a grid's whole ladder and writes `grid_sweep.<grid>.csv` beside
it, pairing what coarsening buys (fewer classes) against what it costs
(`cell_offset_km`: targets sit further from the centre standing in for them).

### What each grid costs, measured

`meta.json` records both grids' costs rather than arguing them away.
`cell_offset_km` (both grids) is the distribution of target-to-seed distances —
the quantization the grid choice buys, bounded by the cell. Measured on the four
runs: p50 ≈ 16-20 km and max ≈ 26 km at `h3-4`, p50 ≈ 19-25 km and max ≈ 41 km
at `healpix-128`. `grid_diagnostics` is grid-specific and **empty for HEALPix**, which has nothing
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

## The bipartite graph (§7.3)

`build-bipartite-graph` is the second step. `build-answer-space` read only the
target side; this reads the **VPs** and the **measured (VP, target) edges**, and
describes both against the grid answer space every accuracy number is scored on.

**No RTT enters and no variant runs.** RTT is used for one thing — the canonical
CSV's own `rtt_ms > 0` row filter, i.e. "was this pair measured" — and is then
dropped. That is what makes these numbers the fixed reference the RTT-dependent
§8.2 and §8.3 material is read *against* rather than another result competing
with it.

### Latent and observed, reported as a pair

§7.3 asks for every VP-to-target distance twice: the **latent** value over all
VP x target pairs, which describes where the infrastructure sits, and the
**observed** value over measured edges, which describes what the dataset can
actually deliver. An edge set is an artifact of the campaign rather than a
property of the deployment, so reporting one half without the other is what
lets sampling bias be misread as an algorithmic result.

The two are therefore **nested** in `meta.json` rather than sitting as siblings
with different names:

```json
"edges": {
  "edge_density":  0.996184,
  "length_km":     {"observed": {...}, "latent": {...}, "n_latent_pairs": 53466, "note": "..."},
  "nearest_vp_km": {"observed": {...}, "latent": {...}, "note": "..."},
  "measured_nearest_vp_ratio_per_target": {"n": 399, "max": 1.0, ..., "note": "..."}
}
```

Nesting is the point: neither half can be read with the other out of view. The
pairing applies to distances and **not to degree** — a target's latent degree is
the VP count for every target, so it carries nothing, and each degree block says
so in its own `note` instead of emitting a constant. Degree is named for the
side it points at, `degree_to_target` on VPs and `degree_to_vp` on targets,
because there is no VP-to-VP or target-to-target edge and a bare "degree" in a
node block invites reading it as one. **Angular geometry stays observed-only**,
matching §7.3's own scoping of it to measured neighbours: the arrangement term
describes the constraints a variant actually receives, and a bearing set no
method ever sees would not be that.

`measured_nearest_vp_ratio_per_target` is what the pair exists to support:
nearest-measured-VP over nearest-VP, one value per target, `>= 1` by
construction. §7.3 calls it *measurement efficiency* and the code keeps that as
the function name, but it is emitted under a name that says what is divided by
what. It separates **"the VP set is badly placed"** (a large latent nearest-VP
distance) from **"the VP set is fine but the campaign allocated probes badly"**
(a ratio above 1), and only the second is fixable by reallocating measurement.

It is **not** `edge_density`. Density is a *count* ratio over all pairs; this is
a *distance* ratio at the minimum only. A campaign at 50% density still scores
1.0 if it always includes each target's closest VP, and one at 99% density
scores badly if the missing 1% is exactly those. Both are reported because only
the second bounds accuracy.

**Measured on all four runs, it is exactly 1.00.** Every target measured its
nearest VP: 399/399, 412/412, 458/458 and 78/78. The campaigns do miss pairs
(204, 218, 79 and 5 of them), but never a target's closest VP.

That is expected, and the reason is worth stating rather than discovering twice:
**every dataset in the repo today is an active mesh ping**, so the edge set is
close to the full cross product by construction (density 0.996-0.999) and the
campaign has no allocation policy to be biased. The latent and observed
distributions coincide for the same reason — observed edge-length p25 differs
from latent by 3 km on as01 and by nothing at all on as02/as03. The pair is
uninformative on these campaigns *because* of how they were collected, which is
a fact about the collection and not about the metric.

The metric is here for the datasets that are not yet in the repo. §7.3's
proprietary operator setting includes RTT **passively observed at the user
planes on production traffic**, where the measured pairs are whatever traffic
happened to traverse and the edge set is emphatically not the cross product; the
same goes for any traffic-weighted set (`has_weight` is `false` on all four runs
today, so §8.1's weighted views cannot be produced from them at all). On those,
efficiency above 1 is the expected reading and the latent half is what makes it
interpretable. Until then it is a control: reported, flat, and cited once to
rule campaign bias out of §8 rather than argued about.

> Two ways this went wrong before it went right, both recorded because both were
> silent. Edge lengths were rounded to 3 dp before the division while the latent
> denominator was not, which put 160 of as01's 399 targets a few 1e-4 *below*
> 1.0 — a value the ratio cannot take. And `nearest_vp_is_measured` compared VP
> **ids**: co-located VPs are the normal case (as01's VP nearest-neighbour p50
> is 0.0 km), so `argmin` and `eval_source`'s BallTree break ties differently
> and agreed on only 259 of 399 targets while their distances agreed to
> 4e-4 km. The test is on distance now, and the ratio snaps a
> residue measured at 1.24e-9 relative so that `== 1.0` is usable downstream
> with no tolerance repeated.

Out of scope, per the run decision: §7.3's optional appendix (Clark-Evans,
anisotropy, nearest-neighbour CV, degree assortativity) and §8.2's VP coverage
ceiling.

### Extent, then dispersion

Diameter and p95 say how far the set reaches; they cannot say whether it is
spread or stacked inside that reach. The `dispersion` block on each node side
answers that, **at this directory's own resolution and no other**:

* **`effective_count`** — the number of distinct places the set resolves to at
  that scale, i.e. §7.3's occupied-cell count. 134 VPs are 81 *places* at 45 km,
  and that 81 is the honest denominator for any claim resting on independent
  observations.
* **`occupancy_ratio`** = `effective_count / count`. 1.0 means every node is its
  own place; low means many share one. Dividing out the set size is what lets a
  134-VP fleet and a 53-VP fleet be compared directly.

At the default `h3-4`:

| | VPs | occupancy | targets | occupancy |
| --- | --- | --- | --- | --- |
| as01 | 134 → 81 | 0.60 | 399 → 18 | 0.05 |
| as02 | 134 → 81 | 0.60 | 412 → 22 | 0.05 |
| as03 | 134 → 81 | 0.60 | 458 → 22 | 0.05 |
| as7018 | 53 → 41 | 0.77 | 78 → 22 | 0.28 |

Both VP fleets are **dispersed** — most VPs are their own place. Both target
sides are **stacked**, and the operator far more so: 399 IPs at 18 places is a
ratio of 0.05, against as7018's 0.28. That is the same fact the flow-line
collapse ratio reports downstream, and it is why 81 rather than 134 is the
denominator behind any independent-observation claim.

**One rung per directory.** §7.3 also asks for the curve up the hierarchy, and
that is what `--sweep` is for: each rung builds its own directory carrying its
own dispersion. Reporting coarser rungs *inside* a finer rung's file would put
the same number in several directories at once and let two copies disagree, so
the file states only its own scale. Each build re-bins the coordinates rather
than coarsening cell ids, since H3 is aperture-7 and a parent id is an exact
index but not a geometric container.

Read the curve as a concentration diagnostic when you have it: flat means
genuinely distinct metros, a steep climb toward fine cells means the set only
separates intra-metro. Measured across the ladder, the operator VP fleet is
shallow (0.64 / 0.60 / 0.54 / 0.37 at 17/45/120/316 km — still 50 places at
316 km) while as01's targets sit at ~0.05 *at every rung*: coarsening changes
nothing because they were never spread. Note that merging is decided by boundary
**alignment**, not distance — Kansas City and Omaha are 268 km apart and still
occupy two different h3 cells at res 2, whose pitch is 316 km.

`nearest_other_node_km` used to sit here and was removed: it is identically zero
on every operator target side (399 targets at 20 coordinates means every target
has a coincident twin), so it answered nothing `dispersion` does not answer
better and at a stated scale.

### Scale: the latent half is |VP| x |targets|

Which is not materializable at the paper's deployment setting. The split is
therefore deliberate:

* **Per-target latent nearest-VP distance is exact at any target count.**
  `nearest_across_km` chunks the cross matrix and reduces with a per-chunk
  `min`, so only `|VP| x chunk` floats are ever resident. This is the quantity
  `measured_nearest_vp_ratio_per_target` needs, so it never degrades.
* **Only the latent *distribution* falls back**, since a distribution needs the
  values rather than their minima. Past `_MAX_CROSS_PAIRS` the target side is
  subsampled deterministically and `meta.json` records that it was, so a
  percentile is never read as exact when it is not. At 134 VPs that fires around
  37k targets; the four runs here are 4k-61k pairs.

### The node sets are the run's, not the edge list's

VPs are `vps.csv`'s roster and targets are the answer space's `assignments`.
Edges naming anything outside those are dropped **and counted**
(`n_edges_dropped_*` in `meta.json`). That is what makes `density` read
`|E| / (|VP roster| x |answer-space targets|)` — the denominator the benchmark
actually spent measurement against — instead of against whatever the CSV
happened to contain, which would report every campaign as complete. It also
means an unmeasured VP lowers density and moves no observed distance, which is
the observed-only contract in its testable form.

`eval_dataset/<basename>_dataset_stats.json` precomputes much of this, and
SCHEMA.md says to check there first. Its answer-space half is keyed to the
benchmark's older radius-capped `clusters/` space rather than to the grid, so it
is used as a **cross-check** and not as an input: on as01-03 our `n_vps`,
`n_targets`, `n_edges`, `density`, both degree p50s, edge-length p50/max and the
VP nearest-neighbour p50 all reproduce `eval_stats.json` exactly. (as7018 ships
the thinner `eval_stats.json` without `bipartite_coverage`, so only the counts
are checkable there — SCHEMA.md §7.)

### What the four runs measure, and where the differences are

| | as01 | as02 | as03 | as7018 |
| --- | --- | --- | --- | --- |
| VPs / occupied cells at `h3-4` | 134 / 81 | 134 / 81 | 134 / 81 | 53 / 41 |
| targets / occupied cells = K | 399 / 18 | 412 / 22 | 458 / 22 | 78 / 22 |
| edges / latent pairs | 53,262 / 53,466 | 54,990 / 55,208 | 61,293 / 61,372 | 4,129 / 4,134 |
| density | 0.996 | 0.996 | 0.999 | 0.999 |
| distinct flow lines | 1,738 | 1,913 | 2,001 | 4,076 |
| nearest VP p50, observed / latent km | 13.3 / 13.3 | 19.7 / 19.7 | 8.8 / 8.8 | 35.7 / 35.7 |
| nearest VP p90, observed / latent km | 48.2 / 48.2 | 154.2 / 154.2 | 30.4 / 30.4 | 481.9 / 481.9 |
| **measured nearest-VP ratio** (max) | 1.00 | 1.00 | 1.00 | 1.00 |
| targets that measured their nearest VP | 399/399 | 412/412 | 458/458 | 78/78 |
| VP to its nearest target, p50 km (latent) | 102.9 | 166.6 | 69.0 | 45.5 |
| max angular gap, p50 / p90 deg | 113 / 200 | 85 / 176 | 115 / 176 | 140 / 265 |

Five things read straight off that table.

**The latent/observed pair is flat, and that is the result.** Every
observed/latent column above agrees to the decimal, and the measured
nearest-VP ratio is exactly 1.00 on all four runs — as it should be for an active mesh ping. The
edge set is not a biased sample of the deployment, so no §8 finding can be
blamed on probe allocation. Reporting the pair is still what licenses that
sentence: it is a measured absence of bias rather than an assumed one, which is
the distinction §7.3 asks for, and it is the same instrument that will read
non-trivially on a passively-collected set.

**The three operator runs share one VP fleet.** as01, as02 and as03 carry the
same 134 `vp_id`s at the same coordinates, which is why every VP-side row above
is identical across their three columns; they differ only in targets, and even
there the *coordinates* overlap (12 of as01's 20 distinct target locations also
appear in as03). Their target *ids* are disjoint, so pooling them in
`plot-venn` / `plot-pareto` is sound as a target population — but it is one
deployment measured against three overlapping target sets, not three
deployments, and a cross-dataset claim should say which of the two it needs.

**Density does not discriminate, and neither does degree.** All four campaigns
are essentially complete bipartite graphs (0.996-0.999) in one connected
component, and target degree has a single-valued IQR on every run (134/134/134,
53). So neither measurement completeness nor constraint count is an axis on
which these datasets differ, and §8.3's regression cannot identify a degree term
from them — mesh ping is why. The axes that do differ are proximity and
arrangement, which is what makes those two the usable covariates.

**The flow-line ratio is the operator/public structural difference in one
number.** The operator runs collapse ~30x (53,262 edges to 1,738 lines: 399
target IPs sit at 20 distinct coordinates), as7018 collapses 1.01x (its targets
are distinct anchors). §3.2's claim that the two settings cannot be substituted
shows up here as an artifact of the data rather than as an argument.

**VP concentration is real but shallow.** 134 VPs occupy 81 cells at 45 km, and
coarsening the grid barely helps: 81 at res 4, 73 at res 3 (120 km), 50 at res 2
(316 km). §7.3 reads that curve as a multi-scale concentration diagnostic, and a
flat curve means genuinely distinct metros rather than a fleet that only
separates at intra-metro scale. as7018's 53 VPs behave the same way (41 / 32 /
21).

**VP proximity is the operator's advantage, and arrangement is not.** The
operator fleets put a measured VP a median 9-20 km from each target against
as7018's 36 km, and the p90 separates far harder (30-154 km against 482 km) —
consistent with as7018 having 37% of targets with no discriminative VP at all
while as01 has none (SCHEMA.md §5). Arrangement does not follow: a max angular
gap above 180 deg means the target lies **outside** the VP convex hull and is
constrained from one side only, and that is the p90 case on as01 (200 deg) as
well as on as7018 (265 deg). Better proximity does not buy better bracketing.

Grid choice moves the VP side almost not at all (81 cells on `h3-4` against 80
on `healpix-128`) while moving the target side on as7018 (22 against 27) — the
same boundary-alignment effect the answer-space section reports, now measured on
a second node set.

### The three figures

`plot-bipartite-graph` writes all three, because none of them can carry another
one's job.

* **`bipartite_nodes_map.png`** — topology. VPs, targets, both node sets'
  occupied cells, and the class regions, with no edges to occlude them. The
  VP-cell outlines are the point: "134 VPs in 81 occupied cells" is the
  denominator behind any independent-observation claim, and a scalar does not
  show that two dozen of them sit in one metro.
* **`bipartite_flows_map.png`** — every measured edge, so coverage reads
  directly. Lines are the **distinct** `(VP coord, target coord)` pairs with
  width carrying multiplicity, never one line per row: on as01 that would draw
  53,262 lines of which 51,524 are exact overplots, and alpha would report
  duplicate IPs at one facility as heavier traffic. Paths are great circles,
  interpolated by slerp in `mapping.great_circle_segments` rather than handed to
  `ccrs.Geodetic()` (too slow at this count) or drawn as lon/lat chords (a
  Chicago-Seattle chord sits 146 km south of the real path).
* **`distance_cdf.png`** — two panels. **Left**, the curve §7.3 asks for behind
  the diameter and p95 scalars: VP pairwise, target pairwise, and the
  VP-to-target distribution drawn **twice**, over measured edges (solid) and over
  all pairs (dotted). Four curves on three hues, deliberately: latent and
  observed are two samplings of one entity rather than two entities, so they
  share a slot and separate by dash. Giving latent its own hue would assert a
  distinction the data does not have *and* force a 4-hue palette whose best
  red/green pair only reaches dE 7.2 under protanopia. **Right**, the measured
  nearest-VP ratio on its own panel, because it is a ratio and not a distance —
  sharing the left x axis would be the two-scales error. The y axis is shared,
  both panels being cumulative shares. On all four runs the right panel is
  degenerate (a vertical line at 1.0), so it prints the sentence instead: a
  curve there would imply a distribution the data does not have. It switches to
  a log x axis above 4x, since a ratio is bounded below by 1 and unbounded above.

  The three hues come from the same categorical theme as
  [diagram/common/palette.py](modules/diagram/common/palette.py) but are **not**
  that tuple, since colour there is bound to a CBG variant's identity and a
  VP-pairwise curve is not a variant. Validated as their own palette with the
  dataviz skill's `validate_palette.js --pairs all`: all five checks pass, worst
  dE 13.0 (deutan) / 7.6 (tritan). The tritan figure is in the 6-8 band that is
  legal only with secondary encoding, so each curve also carries its own dash
  pattern and is direct-labelled — at its own quantile rather than all four at
  p50, where the medians sit within 200 km of each other and the labels
  overprint, with a white stroke behind the text since four near-coincident
  curves leave nowhere clear of all of them.

Both maps call `mapping.seed_voronoi`, so the boundary they draw is byte-for-byte
the geometry `plot-answer-space` draws and is the classifier's own top-1 decision
boundary. [modules/mapping.py](modules/mapping.py) was extracted from
`map_answer_space.py` for that reason; the extraction is pure code motion, pinned
by `answer_space_map.png` being byte-identical across it on all four runs, both
grids and both extents.

### Finding the edge CSV

The v3 configs deliberately do not name it — `benchmark: {}` on every operator
run, whose canonical CSV was reconstructed from the run outputs. The run itself
records it, in `eval_source/<basename>_eval_stats.json`'s `csv` key relative to
the repo root, so that is read rather than reconstructed from `run_id` (which
does not track the CSV stem). Falls back to globbing
`datasets/**/<eval_basename>.csv`, then to naming `--source-csv`.

Reading goes through `benchmark.v2.eval_source.load_canonical_csv`, which owns
the schema and applies the same NaN and non-positive-RTT filter
`GenericCSVSource` applies at materialize time — so the graph described here is
the graph the benchmark ran on. Deliberately **not** `eval_source.build_pairs`:
its `gc_km` / `inflation` / `rtt_rank_norm` are RTT-derived and belong to §8.

## Distance to all seeds, and top-N for free

`classify` emits, per (method, target), the distance to **every** seed. That is
deliberately more than a label: `tg_seed_rank` (how many seeds are strictly
closer than the true one) falls out of the full vector, so top-N accuracy for
*any* N is `(tg_seed_rank < N).mean()` with no recomputation. Rank 0 is
top-1.

The answer space is an **explicit input** (`--answer-space`), so predictions can
be re-scored under a different quantization without re-running anything
upstream.

## The proximity diamond (§8.1's strata)

`classify` says *whether* a method got a target right. `build-proximity` says
**whether the target was answerable at all**, and by whom — the covariate every
§8.1 table stratifies on.

### Every distance is VP → seed, never VP → target

The single decision the module turns on. Classification labels a coordinate by
its nearest seed, so the guarantee worth having is about the *class*:

    d(VP, S) < margin(S)  ⟹  S is that VP's nearest seed

One line of triangle inequality proves it: for any other seed `S′`,
`d(VP, S′) ≥ d(S, S′) − d(VP, S) > 2·margin − margin = margin > d(VP, S)`.

Measure to the **target** instead and the implication fails by exactly that
target's `cell_offset_km` — p50 16-20 km at `h3-4`, the same order as `margin`
itself. v2's `eval_source` measures to the target, so its `closest_vp_km` and
`has_vp_proximity` are *not* what is recomputed here; [SCHEMA.md](SCHEMA.md)
carries the column-by-column mapping. The VP→target distances that *are* kept
(`closest_vp_to_tg_km`, `sping_vp_to_tg_km`) are context, and are deliberately
not what any flag thresholds.

### Four flags, on two axes

```
                 has_proximate_vp
                /                \
 has_discriminative_vp        has_proximate_sping_vp
                \                /
             has_discriminative_sping_vp
```

The **argmin axis** (`has_proximate_*`) applies the classifier's own rule —
`tg_seed` is this VP's nearest seed. The **half-gap axis**
(`has_discriminative_*`) applies the strict guarantee above, and is tighter: it
implies rank 0 without being implied by it. The **left column** is an existence
claim over every measured VP; the **right column** is about the one VP the
baseline designated.

`4 ⟹ 2 ⟹ 1` and `4 ⟹ 3 ⟹ 1`, but 2 and 3 are **incomparable** — geography-
strength and routing-strength are separate axes meeting at the top. That is why
v2's 3-level `proximity_label` is dropped rather than carried: a total order
cannot express an incomparable pair. `meta.json` re-checks all four implications
on every write and reports violations rather than raising, since a violation
would be a real geometric statement about the answer space. It is empty on all
four runs at both grids.

§8.2's taxonomy is the diamond's argmin chain, cut twice, and gets no column of
its own:

| term | flags | what the target requires |
| --- | --- | --- |
| `geometry_only` | `¬has_proximate_vp` | no measured VP resolves the class; only multilateration can |
| `selection_miss` | `has_proximate_vp ∧ ¬has_proximate_sping_vp` | a VP resolves it, the baseline picked another — the CBG opportunity |
| `selection_hit` | `has_proximate_sping_vp` | the baseline's own VP resolves it |

**`geometry_only` is a ceiling on Shortest-Ping, not on CBG**, and the name says
so deliberately. On `as02` its 40 targets are exactly two seeds — every member
of both — whose nearest measured VP sits 92 km and 153 km out, with
`tg_seed_best_rank == 1` throughout: the true seed is always the runner-up.
Shortest-Ping and Vanilla answer 0 of 40; Octant-Hull answers 39. An earlier
name for this stratum, `structural_failure` (inherited from v2's
`NO_PROXIMITY`), filed the case that most favours CBG under a heading saying CBG
cannot win it.

v2's vocabulary is not reused for the other two terms either. `HAS_USED_PROXIMITY`
and `selection_hit` measure related but different things — cluster-keyed
VP→target versus grid-keyed VP→seed — and a near-identical spelling would invite
mixing the two layers' numbers in one table.

### The tautology is the point

`has_proximate_sping_vp` **is** Shortest-Ping's top-1 correctness, by
construction: the baseline predicts its VP's coordinate and classification is
argmin-over-seeds, so the flag and the score are one computation on one input.
It is kept as the *self-check* — if the two ever disagree, the rank rule here
and `classify`'s have drifted. It holds exactly on all four runs at both grids,
and `sping_vp_to_tg_seed_km` is bit-identical to `shortest_ping`'s
`error_to_tg_seed_km`. Consumers must annotate that cell rather than report it;
at top-3 the same cell is informative, because the flag stays top-1.

Getting that for free required one choice, and then one refactor.

The choice: the shortest-ping VP's identity and **coordinate come from
`eval_source`**, not from re-minimizing RTT. That is the VP `classify` scores,
and a second derivation would break ties differently. Deriving it inside v3 would
*look* like less coupling while being worse — `classify` reads `eval_source`
regardless, so v3's two halves would disagree with each other instead of agreeing
with v2.

The refactor: both sides now go through **`io.load_sping_vp`**, one reader with
the missing-column check in one place, translating v2's `shortest_ping_*`
spelling to v3's `sping_*` on the way across. Two independent readers agreed
today and were free to drift tomorrow; `test_proximity.py` pins that
`score_shortest_ping` still routes through the helper, because a regression that
reintroduced a local read would pass every numeric test and only fail there.

### Constant is not the same as inert

`has_proximate_vp` and `has_discriminative_vp` are **True for all 399 targets on
as01** and all 458 on as03 — the operator VP fleet is dense enough that every
target has a VP inside its seed's half-gap. That is a fact about the deployment,
not an absence of effect, and a bare share of `1.0` reads as the latter. So
`meta.json` ships `n_true`/`n_false` beside every share and names the constant
flags in `zero_variance`, and the CLI prints the warning inline.

as02 and as7018 do vary (`372/412` and `49/78` proximate), so the left column is
not dead — it separates on exactly the runs where VP placement is the binding
constraint.

## Crossing accuracy with the strata

`breakdown-accuracy` multiplies `classify`'s correctness by `build-proximity`'s
labels, which is what turns an aggregate accuracy number into a claim about
*where* a variant earns its result. A variant three points ahead overall may be
ahead only on targets a VP was already sitting on — being credited for the
dataset's geometry — or ahead on `geometry_only` targets, where nothing but its
multilateration could have produced the answer. `topn_accuracy.csv` cannot tell
those apart.

Two files, because the four flags and the three terms are different shapes.
`accuracy_by_flag.csv` gives each flag its own 2×2 (rate difference **and** φ,
since a 40-point gap over six targets is a large difference and a weak
correlation); the diamond is not a chain, so the four may disagree.
`accuracy_by_taxonomy.csv` uses the three terms, which *do* partition the
targets, and its `n_targets` sums to the run's target count for every (method,
N) — the check that makes it readable as a decomposition of the headline rather
than three unrelated rates.

**What it already shows: SoI CBG is the baseline in disguise.** Crossed with
`has_proximate_sping_vp` at top-1, `million_scale_cbg` scores 1.000 / 0.993 /
1.000 where the flag holds and 0.069 / 0.000 / 0.012 where it does not, on
as01 / as02 / as03 — φ of 0.947, 0.995, 0.987 against a *baseline correctness
indicator*. Its aggregate accuracy is within three points of Shortest-Ping's on
all three runs, and this says why: it is not arriving at those answers
independently. The tautological cell sits one row above in the same file, which
is exactly why it is marked `is_tautological` — the reader needs to see that
Shortest-Ping's own 1.000/0.000 is arithmetic while SoI's 0.993/0.000 is a
finding.

## What the mistakes look like

`confusion-density` asks the operator's question rather than the scoreboard's:
"wrong by one cell" and "wrong by a continent" are the same zero in an accuracy
column.

`confusion_pairs.csv` carries **`boundary_margin_km`** — `error_to_tg_seed_km −
error_to_pred_seed_km` on each wrong row. Near zero, the estimate sat almost
equidistant from both classes and the flip was a tie-break; large, it was
confidently inside the wrong cell. The medians separate the variants cleanly:
Vanilla CBG's wrong answers sit at 31 / 70 / 87 km on as01 / as02 / as03 while
Shortest-Ping's sit at 670 / 606 / 281 km. Vanilla misses narrowly and often;
the baseline, when it misses, misses by a region.

### Boundaries crossed, not distance rank

**`seeds_crossed` is the column to read as "a neighbouring cell".**
`seed_crossing_matrix` walks the geodesic from the true seed to the predicted
one and counts Voronoi class boundaries: 1 means the estimate slipped across a
single line, 3 means it landed three cells away.

`pred_seed_neighbour_rank` — the predicted seed's position in the true seed's
*distance* ordering — sits beside it and answers a different question. Rank 1
implies one crossing, but not the reverse: a seed can be fifth-nearest and still
share a boundary. The gap is not a rounding difference. On as01, 67% of wrong
top-1 rows are one boundary away against 42% at rank 1, and it **reorders the
methods** — by rank, Octant-Spline (79%) leads SoI CBG (52%); by crossings SoI
leads at 96% against Octant-Spline's 90%. Distance rank was measuring how
crowded the neighbourhood is, not how near the miss was.

`seeds_crossed == 1` is **the** definition of class adjacency in this layer:
`seed_crossing_matrix` lives in `answer_space.py` and the same walk produces
`seeds.csv`'s `class_adjacency_degree`, so the per-pair column and the per-seed
count cannot disagree.

It is stricter than "shares a Voronoi edge", and that is the point. The pairs it
excludes are the far ones, whose cells touch only a long way from both seeds —
median separation 1410 km against 728 km for the one-crossing pairs on as01. A
target near one of those is not realistically confusable with the other, so
counting the pair would inflate the local density §7.4 asks for. It replaced a
`delaunay_degree` column built from the convex hull of the unit vectors, which
triangulates the *whole sphere* and so joined outer seeds across the empty
hemisphere; see [SCHEMA.md](SCHEMA.md).

`confusion_by_density.csv` bins targets by their true seed's `nearest_seed_km`.
Crowding does cost accuracy on as01 (0.33 → 0.71 → 0.82 for Shortest-Ping across
tertiles) and as02 (0.16 → 0.42 → 0.51), **but not on as03** (0.67 → 0.25 →
0.44). So local density is a real effect and not a universal one, and the
non-monotone run is the one to explain rather than the one to drop. Bins are
quantiles over targets, not fixed widths: seed spacing at `h3-4` is dominated by
the grid pitch, so fixed bins would put most of a run in one bucket. On a run
where every seed shares one `nearest_seed_km` the bins collapse to one, which is
reported rather than raised.

## The §8.1 table

`table-accuracy` assembles the paper's headline from what `classify` and
`build-proximity` already wrote — nothing is recomputed — and emits CSV plus a
markdown render, so no number is hand-transcribed between the re-analysis
artifact and the paper.

**It offers no pooling switch.** `as01`, `as02` and `as03` are one VP fleet
under different peering and routing conditions, and that difference *is* the
comparison §8.1's "clean network topology matters" subsection rests on;
averaging them would delete the finding to produce a tidier number. One row per
(run, method), and the operator/public split is the caller's — invoke it twice.

`plot-pareto`'s mixed-setup refusal is deliberately **not** inherited. That
guard exists because one cost frontier over both setups would compare a 134-VP
fleet against a 53-VP one; here each run keeps its own row and no such average
can form. On as01–03 `setup` is a placeholder with no meaning anyway. Mixed
setups are noted in the manifest instead of refused.

The three §8.2 shares go in a separate `dataset_context.csv` rather than being
repeated down every method row, where they would read as a property of the
variant. The accuracy column means something different when 37% of a run's
targets have no proximate VP (as7018) than when none do (as01, as03).

## The paper's own layout

`table-accuracy` prints one row per (run, method), which is right for
re-analysis and wrong for the paper: §8.1 compares *methods within a dataset*,
so the methods have to be adjacent columns. `table-headline` transposes it and
adds the two things a paper table needs and a data table does not.

**A best-in-row mark with a tie rule.** Bolding the argmax alone asserts a
ranking the sample size does not support — on as03 the gap from Octant-Hull
(0.502) to Spotter (0.474) is 0.028 against a standard error of 0.023 on 458
targets, and on as02 the baseline leads SoI by 0.003. The mark is instead
"within one standard error of the row's best", computed once per row on the
best cell's own rate, so several methods can be marked and a near-tie reads as
a tie. That is what happens on as03 at top-3, where Shortest-Ping and SoI both
reach 0.926 and *beat* Octant-Hull — the ranking flip top-1 alone would hide,
which is why the appendix table exists.

**Dataset types own the rows, with the per-AS breakdown beneath.** §8.1's story
line compares the two campaigns at *dataset-type* granularity, so the rows run
`MESH (3 ASes)`, `· AS01`, `· AS02`, `· AS03`, `TRAFFIC-WEIGHTED (0 of 3 ASes)`,
… An earlier shape interleaved the kinds per dataset (`AS01 MESH`,
`AS01 WEIGHTED`, `AS02 MESH`, …) so a weighted row read as a filter on the row
above it; that optimized for a per-AS comparison the paper does not make, and
left the fleet-wide number for the reader to average in their head. With one
dataset the aggregate is suppressed and the breakdown rows carry the kind
themselves, since an indent needs something to indent under.

**The pooled row is a micro-average, and only it is computed.** It sums the
datasets' targets and scores once — "pick a target at random from the fleet" —
rather than averaging the three rates; on as01/02/03 the two differ by at most
0.0064 and the manifest's `weighting` block reports both. Breakdown rows print
`accuracy_topN` verbatim, because three of the thirty-six cells round
differently the two ways (as01's Spotter reads 0.389 verbatim and 0.388
reconstructed) and agreeing with `table-accuracy` to the printed digit is the
invariant the module exists under. `accuracy_is_reconstructed` marks the split
per cell.

Two things the pooling forces the table to say out loud. A pooled winner can
lead the population while leading one dataset in it — at top-3 Octant-Hull takes
the pooled row at 0.891 having led only as02, with as01 going to Octant-Spline
and as03 to Shortest-Ping and SoI — so that cell gets a `†` and a footnote
naming what it lost. And a method absent from one run is pooled over the runs
that carry it, so its denominator is under the row's `n`; that cell gets a `‡`
and its own count. Because an average now exists, `cross.guard_one_setup` and
`cross.guard_disjoint_targets` both apply here (they are skipped when the
aggregate is suppressed); `table-accuracy`'s exemption rested on there being no
averaging to protect.

**A reserved row per dataset variant the section compares.** Every
TRAFFIC-WEIGHTED row is empty: `has_weight` is false on all three operator runs
and the canonical source CSVs carry no weight column, so that campaign is
**uncollected rather than unanalyzed**. The rows print as `—` rather than being
omitted, because a table missing half its rows reads as though the mesh numbers
were the whole comparison.

The pairing is stated, not inferred: `--weighted-run-id as01=<run_id>`.
`short_dataset` supplies the dataset key everywhere else in this layer, but it
strips only an all-numeric trailing tail, so no plausible weighted run name
reduces to its dataset — `as01-weighted-260728` comes back unchanged and
`as01w-260728-260802` reduces to `as01w`. A fallback would file the row under
the wrong dataset instead of failing.

Fallback rate rides inside the cell (`(fb 0.27)`) and only where non-zero,
because Vanilla CBG is the only variant that ever falls back; a parallel
six-column block would be five-sixths zeros with the one number that matters
hardest to find.

Both tables read `topn_accuracy.csv` through `accuracy_table.accuracy_rows`,
and their 18 shared cells agree exactly at both top-1 and top-3. The pooled
counts are reconstructed as `round(accuracy_topN * n_targets)`, which is exact
rather than nearly so — at four decimals and these denominators exactly one
integer maps to a published rate — and is confirmed against
`overlap_membership.top{1,3}.csv`, a per-target correctness matrix whose column
sums match on all thirty-six cells. That file is a `plot-venn` output, so it
checks this table and never computes it.

### The same numbers as bars

`plot-outcome-bars` draws what the table states, plus the third quantity that
completes it: each bar is one method's whole target set partitioned into
**correct · wrong · failed**, labelled with its own percentage to one decimal, so
top-N accuracy is the bottom segment, the failure rate is the top one, and their
complement is visible instead of implied. `n_fallback` and `n_error` merge into
`failed` — a give-up and a crash are both failures to answer — but they stay
separate in the CSV, and neither may merge into `wrong`, where a crash would
read as a scoring miss. `guard_partition` asserts
`n_correct + n_wrong + n_failed == n_targets` per bar.

**This is the one figure where hue does not mean the variant.** Green right, red
wrong, grey never answered; the method is the x position. The departure is
deliberate — the question here is what happened to the targets, and a reader
holding "green = Octant-Hull" while reading a green segment meaning "correct" is
doing the figure's work for it. Nothing in the bars is variant-hued, so there is
no second meaning inside this figure to collide with, and the tick labels stay
neutral rather than reintroducing one an inch below the bars.

The three fills were picked against checks, not by eye, because red-vs-green is
the classic colour-blind pair: lightness is **monotone** up the stack
(L\* 34 / 54 / 73), the worst pairwise ΔE under simulated deuteranopia and
protanopia is 16.8 (above the 8 that would oblige a secondary encoding), and the
nearest variant hue is ΔE 11.8 away. Label ink is chosen per fill by luminance,
which is what frees the fills to be picked for their own separation instead of
for hosting white text.

Hatch is the **dataset type**, the comparison §8.1 hands off to, so no segment
may carry a texture of its own. Every traffic-weighted bar is a dashed outline
rather than a zero-height bar, for the same reason the table prints `—`, and its
legend swatch is dashed to match — the key shows the mark the reader will meet,
while *why* it is empty is a fact about the data and lives in the caption and the
manifest.

Bars run **best first** by pooled correct rate, ranked on the mesh rows over
every dataset in the figure rather than per panel — in `compare` that is what
keeps an x position meaning the same method in all three panels, so AS03's
degradation is a straight-line scan instead of a search. Ties fall back to
`PUBLISHED_METHODS` order.

The figure carries **no prose and no legend titles**. Two legends share one row
tight under the title — two rather than one because a single key would read as
five alternatives on one scale, when every bar is in fact one outcome stack
*and* one dataset type. Filled swatches are outcomes and outlines are dataset
types, and the gap between the groups is wider than the gap inside either, which
is what tells them apart without a pair of words above the row.

Both are placed by **measurement**: `_place_legend` renders each at x = 0 and the
caller centres the pair once it knows their widths, because the widths depend on
the figure size and this module draws at two of them. The comparison grid keeps
its own band (`COMPARE_LEGEND_Y` / `COMPARE_AXES_TOP`) since its panels carry
`AS01 · n=399` titles above the axes that the pooled panel does not.

### The traffic-weighted placeholder

`headline_table.PROVISIONAL_WEIGHTED` holds **hard-coded top-1 rates from an
earlier run** so §8.1's mesh-vs-traffic-weighted comparison can be laid out
before the weighted campaign is collected. It is gated on the row having no run
of its own, so pairing a real `--weighted-run-id` supersedes it with no flag to
remember, and deleting the constant is then a cleanup.

Two things it does not come with, and which are therefore not invented. It has
**no denominator**, so `n_targets` stays NaN — which propagates honestly, since
`best_in_row` cannot compute a standard error without an `n` and a provisional
row is therefore never marked best. And it has **no top-3**, so the appendix
table's weighted row stays reserved; keying the constant on the top-N is what
stops a top-1 figure printing under a top-3 heading.

**The figure does not mark it.** A bar drawn from a constant and a bar drawn from
the pipeline are identical once rendered, so the caveat lives in the table's `§`
footnote, the `provisional` column of the figure's CSV twin, and the manifest's
`provisional_rows` block — and any caption reusing the PNG has to carry it.

The manifest carries the encoding, the partition and the ordering rule in full.

Bars run **best first** by pooled correct rate, ranked on the mesh rows over
every dataset in the figure rather than per panel — in `compare` that is what
keeps an x position meaning the same method in all three panels, so AS03's
degradation is a straight-line scan instead of a search. Ties fall back to
`PUBLISHED_METHODS` order.

The figure carries **no prose**: two legends instead, one per channel, because a
single combined key would read as six alternatives on one scale when every bar
is in fact one outcome stack *and* one dataset type. What the removed footnote
said now lives where it is actionable — the uncollected campaign names itself in
its own legend entry (`traffic-weighted (not collected)`), and the manifest
carries the encoding, the partition and the ordering rule in full.

Counts and pooling are imported from `headline_table`, not reimplemented, so the
figure's CSV twin and the table's long CSV are the same numbers by construction
rather than by agreement. `--layout compare` breaks the pooled panel out per
dataset, each keeping its own denominator.

## The error half

Accuracy and error distance can disagree, and §2.4(a) treats the disagreement
as a finding, so `plot-error-cdf` reports the other axis: one CDF curve per
method on a log x, over solved rows only.

**The distance is to the raw target.** `error_to_target_km`, not
`error_to_tg_seed_km` — seeds sit at cell centres, so routing through one would
add the row's `cell_offset_km` (p50 16-20 km at `h3-4`) to every answer,
correct ones included. Re-quantizing the answer space changes accuracy and
must leave this figure bit-identical.

**Fallbacks are out, and the baseline is not a fallback.** `io.solved_mask` is
the shared predicate — extracted from `classify.topn_summary` when this figure
became its second caller, so the CDF's per-method `n` equals
`topn_accuracy.csv`'s `n_solved` by construction rather than by coincidence. It
also carries the case a hand-written filter gets wrong: Shortest-Ping's rows
are all `BASELINE`, never `SUCCESS`, so `status == "SUCCESS"` returns an
all-false mask and drops the baseline curve entirely.

**Two numbers that had to be made to agree.** `error_cdf_percentiles.csv` and
`topn_accuracy.csv` both carry `error_km_p50`, and they disagreed by 0.7-1.3 km
until this module dropped `method="nearest"` for numpy's default. The v2
plotter's `nearest` had a real reason — a quoted percentile named an actual
target and matched the MTL world-map viewer's bookmarks — but it is not worth
two definitions of the column the paper's accuracy table reads.

**The log floor is 0.1 km, not the v2 plotter's 1 km.** Sub-kilometre errors
are not rare here: 8 to 20 rows per method across as01/02/03, up to 4.4% of a
run, with an observed minimum of 0.135 km. A 1 km floor flattens the left tail
of exactly the curves that earned it and pins their p5 to the clamp. The
manifest records `n_clamped_to_floor` (empty on all three runs) so a future run
cannot start clipping silently, and the clamp never touches the percentile CSV.

**The baseline is grey and dashed** rather than wearing its variant hue: the
figure shows variants against a reference, not seven peers. The guide verticals
at 100/500/1,000 km are neutral ink, because the v2 plotter's green/orange/red
are Octant-Hull, Vanilla and Spotter in this paper's palette and would read as
series.

## Where the two metrics disagree

Two band figures, one renderer. One band per class-error level, one thin line
per target inside it, x = the coordinate error on a log axis. These are the
only figures that put accuracy and error distance on the same point, which is
what turns §2.4(a)'s claim that they are different metrics into something
readable — and each method turns out to have its own signature in the pair.

**Two y modes, because the two quantities are not the same.** `plot-error-vs-cells`
uses `seeds_crossed` (how many class boundaries lie between the true cell and
the predicted one — the answer space's local density, the snapping story).
`plot-error-vs-rank` uses the true class's **top-cell index** — 1 if it is the
cell closest to the estimate, 2 if one other cell is closer, and so on.
That is `tg_seed_rank + 1`, and the 1-indexing is what makes the axis read
straight off the reported metric: `index <= N` *is* top-N, so band 1 is top-1
accuracy and bands 1-3 sum to top-3. SCHEMA.md warns the two quantities get
confused and that they order the methods differently, so they live in one
module with the distinction documented once and the CLI exposes them
separately. Both figures draw four bands, the top one a bucket.

**Rows are grid cells.** Bands stack contiguously from the x axis with
separators between them, so the panel reads as a stack of tracks rather than
marks floating at ticks. Each row keeps a thin gutter above its band for the
median readout, which would otherwise land in the row above and name the wrong
band; the median rule itself is exactly the band's height, like every other
line in it.

**Density is drawn, not binned and not jittered.** Each point is one line at
alpha 0.15, so coincident values darken by overplotting; there is no bin width
to choose and no random offset, and an x position on the figure is an x
position in the data. The limit is that accumulation saturates near 7
coincident lines, which Shortest-Ping and SoI hit because many targets share a
VP coordinate and their answer *is* that coordinate — so their darkest stripes
stop distinguishing 7 from 30. The band's share label carries the count.

**The denominator is every target, so the bottom band is the accuracy.**
`seeds_crossed == 0` and nearest-cell index 1 are the same event (the crossing
matrix is >= 1 off the diagonal) and both are top-1 correctness. `topn_accuracy.csv` divides
by every target and counts fallbacks as failures, so these figures must too:
on as02 Vanilla, 123 band-0 rows over 412 targets is 0.298, its top-1 accuracy
exactly, while over its 337 solved rows it would read 0.365 and match nothing.
The bands therefore sum to `1 - fallback_rate` and the shortfall is named in
the panel header — Vanilla's four bands total 81.8% and the missing 18.2% is
where its fallbacks went.

Both identities are checked rather than assumed: the bottom band's share
equals `accuracy_top1` and the rank mode's cumulative share through band 3
equals `accuracy_top3`, to 0.000000 on all three operator runs.

**What the pair shows.** The two axes correlate — median error rises
monotonically with class error on every method — so the readable result is the
*shape* of each band stack. Shortest-Ping and SoI put 37% of targets in band 0
as a few tight discrete stripes under 60 km, because the answer is a VP
coordinate. Octant-Hull puts 65% there but smeared from 0.3 km to 600 km.
Spotter puts 41% there at a 195 km median, the worst band-0 of the six. Same
metric, three different mechanisms.

One panel per method rather than six hues in one frame: the bands would overlap
into each other, and the panel already carries identity, so colour never has to
separate six series and the palette's all-pairs margin is not called on. The
error axis, row filter and distance column are shared with `plot-error-cdf`.

**Two or more `--run-id` go cross-dataset**, following `plot-venn`'s
convention, into `_cross/error-vs-class/<dataset-set>/`. `--layout` picks which
question the figure answers; they are different questions, so both are kept and
each writes its own file.

**`--layout pooled` (the default)** merges the runs' targets into one
population and draws the same six-panel grid over it. The operator runs' target
sets are disjoint — 399 + 412 + 458 distinct ids, checked by
`guard_disjoint_targets` rather than assumed — so the union is a 1,269-target
population and not a double count, the same pooling `plot-venn` performs over
these three runs. This is also the form where the density encoding has enough
targets to read as a distribution: Octant-Hull's correct band is a continuous
smear over its 789 targets, 0.135 km to 466 km with a 33 km median and a 4-135
km interquartile range, where a single run showed a few dozen stripes.

A pooled share is a **micro**-average, so it is target-weighted and as03
carries 36% of it. The subtitle says so and the manifest's `weighting` block
reports the macro-average (the mean of the three datasets' shares) beside it:
they agree to within 0.6 pp on every method here, so the weighting is measured
rather than assumed away. Pooled correct-band shares equal the target-weighted
mean of the three runs' `accuracy_top1` to 5e-5 (the CSV's rounding), e.g.
Octant-Hull 0.6217 against 0.729 / 0.650 / 0.502 per dataset.

**`--layout compare`** keeps each run separate: one panel per (method,
dataset), methods down the rows and datasets across the columns. It answers
whether a method's signature survives a change of dataset, and on as01/02/03 it
does not uniformly — Octant-Hull's correct band runs 72.9% / 65.0% / 50.2%
while its one-cell-out band runs 21.6% / 28.6% / 40.2%, so the same method
degrades by sliding one band up rather than by scattering.

Methods own the rows there because that is a within-method read; the transpose
would put the same panels on the page and make the cross-dataset comparison a
vertical scan across two intervening rows. Three columns also keep the panel
width, and therefore the log x axis, identical to the per-run figure — the
whole basis for reading an x position across the two — where six method columns
could not label five decades without collisions. Each panel's numbers are its
own run's, so all 18 correct-band shares match that run's `accuracy_top1` to
0.000000 (and the rank mode's cumulative through band 3 its `accuracy_top3`),
and each panel prints its own `n` because the runs differ in size.

`--all-runs` stays per-run either way: which datasets belong in one figure is a
claim about comparability (§7.3), so it is named rather than discovered, and
mixing the two run families is refused.

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
quantization into a number that needs none — since seeds are cell centres, that
would add a systematic ~17-20 km (`h3-4`) to every *correct* answer. The parquet
also keeps `error_to_tg_seed_km`, whose difference from `error_to_target_km`
is exactly that offset per row, i.e. the row's `cell_offset_km`. One consequence is a useful check: re-quantizing changes
accuracy but must leave `error_km_*` bit-identical, and it does across `h3-4`
and `healpix-128` on all four runs.

Keeping the parquet policy-free is what makes the fallback cost *measurable*.
On `as01`, all 106 of Vanilla CBG's fallbacks have `tg_seed_rank == 0` —
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
nearest-seed class boundaries, their targets, and their seeds — the cell centres
— on one cartopy panel. It exists to make the §7.3 quantization visible: a
facility group spanning a grid line is quantized into two adjacent cells and two
classes, which reads instantly as two touching filled cells and is hard to
believe from a scalar. The seed markers show the other half of the same choice,
sitting at cell centres rather than on their targets. Only occupied cells are
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
