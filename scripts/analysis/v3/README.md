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
| [modules/healpix.py](modules/healpix.py) | lib · equal-area quantizer, nesting |
| [modules/answer_space.py](modules/answer_space.py) | cmd · `build-answer-space` |
| [modules/classify.py](modules/classify.py) | cmd · `classify` |
| [modules/venn.py](modules/venn.py) | cmd · `plot-venn` |
| [modules/map_answer_space.py](modules/map_answer_space.py) | cmd · `plot-answer-space` |

[SCHEMA.md](SCHEMA.md) documents the v2 output schemas this layer reads.

## Pipeline

```bash
# 1. Quantize targets -> seeds (answer space). --sweep does nside 128/64/32/16.
python -m scripts.analysis.v3.cli build-answer-space --all-runs --sweep

# 2. Score every method against it (distance to ALL seeds)
python -m scripts.analysis.v3.cli classify --all-runs --sweep

# 3. Set overlap of correct classifications (repeat per top-N)
python -m scripts.analysis.v3.cli plot-venn --all-runs --sweep --top-n 1
python -m scripts.analysis.v3.cli plot-venn --all-runs --sweep --top-n 3

# 4. Static map of the answer space
python -m scripts.analysis.v3.cli plot-answer-space --all-runs --sweep --us-only
```

Every command takes `--nside` (repeatable) or `--sweep`; omitted, they act on
`nside=128` alone. Outputs land under `outputs/analysis/v3/<run_id>/`:

```
target-answer-space/   nside_sweep.csv
  nside-<x>/           seeds.csv  assignments.csv  seed_mesh_km.csv  meta.json
                       answer_space_map.png
target-cls-accuracy/
  nside-<x>/           <method>_seed_distances.parquet  topn_accuracy.csv  manifest.json
                       overlap_{membership,intersections,pairwise}.top<N>.csv
                       overlap_venn.top<N>.png  overlap_upset.top<N>.png
```

**Two axes, two filename mechanisms.** The answer space parameterizes every
number downstream of it, so both output trees are grouped by `nside-<x>`; and
every overlap artifact carries its top-N in the filename. Without either, a
sweep would overwrite one `topn_accuracy.csv` four times and leave no record of
which grid produced the survivor. `classify` reads the nside from the answer
space itself rather than from `--nside`, so an explicit `--answer-space` still
lands in the directory matching its grid.

## The answer space (§7.3 / §7.4)

An equal-area HEALPix grid at `nside=128` (196,608 cells of 2,594 km², ~51 km)
quantizes the run's ground-truth targets. The grid is a **quantizer, not the
answer space**: its only job is to merge points close enough to count as one
place. Each occupied cell then contributes one **seed** at the spherical
centroid of the targets inside it, so the answer space is K *real locations*
rather than K grid squares. A coordinate is labelled by its nearest seed.

NESTED ordering makes coarsening a bit shift (`pix >> 2k`), so `meta.json`
carries the occupied-cell count at nside 128 / 64 / 32 / 16 in one pass — the
multi-scale concentration diagnostic §7.3 asks for.

### Choosing nside

| nside | cell | pitch | reading |
| --- | --- | --- | --- |
| 128 | 2,594 km² | 50.9 km | paper §7.3 setting |
| 64 | 10,377 km² | 101.9 km | metro |
| 32 | 41,509 km² | 203.7 km | region |
| 16 | 166,037 km² | 407.5 km | macro-region |

`--sweep` builds all four and writes `nside_sweep.csv` beside them, pairing what
coarsening buys (fewer classes, fewer straddle candidates) against what it costs
(`intra_seed_spread_km`, the floor under every error-distance figure).

**Coarsening is not a fix for straddling.** NESTED cells nest, so a boundary at
nside 16 is also a boundary at 32, 64 and 128: coarsening removes only the
*finer* lines. Two targets separated at a coarse level can therefore never be
merged by any nside in the hierarchy. Measured on `as7018_us_test01`: the NY
metro's 10 targets fall in 3 classes at nside 128 and 2 at nside 64 — and still
2 at nside 16, where the cell is 407 km wide. The Bay Area goes 4 → 2 and stops
there too. If grouping co-located facilities is a *requirement* rather than a
tendency, a grid is the wrong instrument and a radius-capped linkage (the
benchmark's existing `clusters/`) is the right one. What the grid buys instead is
alignment-independence: no clustering run, no ordering sensitivity, and the same
cell ids for any target set.

`meta.json` also records the grid's **cost** rather than arguing it away: grid
lines fall where the grid falls, so a facility group straddling one yields two
seeds and two classes, and the Voronoi step cannot undo a split the grid already
made. `straddle_diagnostic` counts seed pairs closer than one cell pitch, and
targets whose own cell seed is not their nearest seed.

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

`plot-answer-space` draws occupied cells (filled, one colour per seed), their
targets, and their seed centroids on one cartopy panel. It exists to make the
§7.3 straddle cost visible: a facility group spanning a grid line is quantized
into two adjacent cells and two classes, which reads instantly as two touching
filled cells and is hard to believe from a scalar. Only occupied cells are
drawn — at `nside=128` the full 196,608-cell grid is unreadable at continental
scale.

The caption states that occupied cell and class are in one-to-one
correspondence, because there is no clustering algorithm involved and the
benchmark also ships an older agglomerative `clusters/` space.

Rendering the sweep is the quickest way to see the previous section's point: at
nside 64 the `as7018` map still shows touching cell pairs at LA, Dallas, New
York and DC.

## Tests

```bash
python -m pytest scripts/analysis/v3/tests/ -q
```
