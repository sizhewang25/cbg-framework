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

[SCHEMA.md](SCHEMA.md) documents the v2 output schemas this layer reads.

## Pipeline

```bash
# 1. Quantize targets -> seeds (answer space)
python -m scripts.analysis.v3.cli build-answer-space --all-runs

# 2. Score every method against it (distance to ALL seeds)
python -m scripts.analysis.v3.cli classify --all-runs

# 3. Set overlap of correct classifications
python -m scripts.analysis.v3.cli plot-venn --run-id <run>
```

Outputs land under `outputs/analysis/v3/<run_id>/`:

```
target-answer-space/          seeds.csv  assignments.csv  seed_mesh_km.csv  meta.json
target-cls-accuracy/          <method>_seed_distances.parquet  topn_accuracy.csv
                              manifest.json  overlap_{membership,intersections,pairwise}.csv
                              overlap_venn.png  overlap_upset.png
```

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
`topn_accuracy.csv`, which reports `accuracy_topN` (fallbacks counted as wrong)
beside `accuracy_topN_success_only` (fallbacks excluded from the denominator).
The gap between the two is the fallback cost.

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

Arity decides the rendering. Up to three methods draw as a classic Venn; beyond
that the region count explodes (6 methods = 63 regions), so an UpSet plot
carries all of them and a companion 3-set Venn covers the headline triple
(baseline plus the two most accurate, or `--venn-method`). Both are backed by
`overlap_membership.csv`, so every count is checkable without reading a figure.

## Tests

```bash
python -m pytest scripts/analysis/v3/tests/ -q
```
