# v4: from classification to the outcome bars

**Date:** 2026-09-23
**Code:** [scripts/analysis/v4/](../scripts/analysis/v4/) —
[`answer_space.py`](../scripts/analysis/v4/modules/answer_space.py),
[`classify.py`](../scripts/analysis/v4/modules/classify.py),
[`healpix.py`](../scripts/analysis/v4/modules/healpix.py),
[`figure_outcome_bars.py`](../scripts/analysis/v4/modules/figure_outcome_bars.py)
**Commits:** `8720ff8` (pooled layout), `d00f574` (uniform segment edge)
**Companion notes:**
[2026-05-26-cbg-data-and-pipeline-walkthrough.md](2026-05-26-cbg-data-and-pipeline-walkthrough.md)
(the v2 benchmark side, which produces v4's inputs),
[2026-06-21-cluster-eval-design-and-usage.md](2026-06-21-cluster-eval-design-and-usage.md)
(the earlier answer-space-as-classification framing).

## Summary

v4 scores a prediction as a **classification over places**, not as a distance,
and grades it by **how many HEALPix rings** separate the prediction's cell from
the truth's. Five mutually exclusive outcomes partition every target, which is
what lets the figure be a stacked bar that provably closes at 100%.

This note traces the whole path — target coordinates in, stacked PNG out — and
records the decisions that are not recoverable from reading the code.

Two things added on 2026-09-23: a **pooled** layout (one panel over every input
run's targets, count-weighted) and a fix for the "no answer" segment rendering
~2 px wider than its neighbours.

---

## The shape of it

```
v2 benchmark tree (read-only)
  <run>/<source>/<setup>/fold_*/<method>/targets.parquet
  <run>/eval_source/*_shortest_ping.csv
        │
        │  ① build-answer-space          answer_space.py
        ▼
  target-answer-space/healpix-<n>/  seeds.csv  assignments.csv
        │
        │  ② classify                    classify.py
        ▼
  target-cls-accuracy/healpix-<n>/  <method>_cells.parquet   ← one row per target
                                    accuracy.csv             ← one row per method
        │
        │  ③ build the table             figure_outcome_bars.py
        ▼                                compare │ pooled
  long frame: one row per (dataset, method), exact share_* columns
        │
        │  ④ rank + render
        ▼
  _cross/cls-accuracy/<set>/  outcome_bars[.pooled].healpix-<n>.{png,csv,manifest.json}
```

`build-bipartite` sits beside this and feeds nothing into it — it co-quantizes
targets and VPs for geometry questions, and is not on the accuracy path.

---

## ① Answer space — what counts as "the same place"

[`build_answer_space`](../scripts/analysis/v4/modules/answer_space.py#L147),
driven per run by
[`build_for_run`](../scripts/analysis/v4/modules/answer_space.py#L311).

`ang2pix(target_lat, target_lon, nside)` puts every target in a cell. The
distinct occupied cells, **sorted ascending by cell id**, become classes
`seed_id = 0..K-1`. Sorting by id rather than by arrival order is what makes
`seed_id` a deterministic function of the occupied cell set.

**A class is seeded at its cell centre, not at the centroid of the targets in
it.** So a seed never moves when the target set changes — the quantizer is a
property of the grid, not of your data. Worth knowing before comparing two runs'
`seeds.csv`.

Output `assignments.csv`: `target_id → cell_id, seed_id, cell_offset_km`.

Built at every rung of `NSIDE_LADDER`
([healpix.py:63](../scripts/analysis/v4/modules/healpix.py#L63), `128 → 16`,
i.e. 50.9 km → 407 km cells), but from **one** `ang2pix` pass: NESTED ids
coarsen by bit shift
([`degrade`](../scripts/analysis/v4/modules/healpix.py#L154)), so nside 64 is
nside 128 `>> 2`. The rungs are therefore *exactly* nested, with no boundary
straddling. That is the property H3 could not give us, and the reason v4 is
HEALPix-only rather than a parameterisation of v3.

---

## ② Classify — where did the prediction land?

### Per target — [`score_method`](../scripts/analysis/v4/modules/classify.py#L173)

| step | column |
|---|---|
| join `assignments` | `tg_cell`, `tg_seed_id` |
| `pred_lat/lon` non-null? | `has_pred` |
| `ang2pix(pred)` | `pred_cell` (`-1` when absent) |
| [`ring_distance`](../scripts/analysis/v4/modules/healpix.py#L196)`(tg_cell, pred_cell, max_ring=2)` | **`ring`** → `0 / 1 / 2 / -1` |
| haversine(truth, pred) | `error_km` |
| v3's rule, recomputed | `nearest_seed_id_retired` |

`ring` is grown breadth-first one shell at a time; `astropy_healpix` has no
k-ring primitive (H3's `grid_disk` takes any k), so this costs `max_ring`
`neighbours` calls. Fine at `MAX_RING = 2`.

`ring == -1` means "further than ring 2" and is deliberately **not** a large
number. Once the prediction has left the neighbourhood the metric asks about,
*how far* is `error_km`'s question, and a big ring count would invite reading it
as a distance.

### Why this replaced nearest-seed

Recorded here because it is the justification for the whole package. v3 scored
by **nearest-seed assignment**, which is a Voronoi partition over K seeds and
therefore labels *every point on Earth* — the rule can never answer "this is in
no class."

Verified consequence, target `tg-e1a1545` on as01: truth Seattle
(47.45, −122.31), Spotter predicted (63.74, −97.47) in the Canadian Arctic —
**2,360 km away, scored CORRECT**, because the nearest of 18 US seeds was
Seattle at 2,380 km and the runner-up Omaha at 2,492 km. A 113 km margin between
two absurd options decided it.

Systematic, not anecdotal: among v3's *correct* classifications on as01, **345
of 354** Spotter predictions sat more than 45 km from the seed they were
credited to (210/210 on as02, 263/268 on as03) — while the answer space's own
metadata said no real target is more than 25.3 km from its seed.

Ring distance is **local by construction**: no arrangement of far-away cells can
make a distant cell adjacent. The Arctic prediction is `-1` at every rung.

`accuracy_nearest_seed_retired` reproduces the old rule alongside the new
columns, for one release. Not for use — the gap between it and `accuracy_ring0`
is the measurement that justifies the change, and it can only be taken while
both are computed on identical inputs.

### Per method — [`summarize`](../scripts/analysis/v4/modules/classify.py#L241)

```python
solved  = status == "SUCCESS"        # an all-BASELINE frame is wholly solved
placed  = ring >= 0

n_ring0  = solved &  placed & (ring == 0)
n_ring1  = solved &  placed & (ring == 1)
n_ring2  = solved &  placed & (ring == 2)
n_beyond = solved & ~placed          # answered, and nowhere near
n_failed = ~solved                   # never answered — FALLBACK or ERROR
```

**This is the load-bearing invariant.** Every target satisfies exactly one, so
the five sum to `n_targets`, and
[`guard_partition`](../scripts/analysis/v4/modules/classify.py#L307) asserts it
rather than trusting it — the failure is invisible in the artifact, because a
stack summing to 0.98 still looks like a stack.

Two denominator decisions:

* **`n_failed` is in the denominator.** A method that declines to answer has not
  earned a smaller denominator than one that answers badly.
* **`error_km` percentiles exclude it.** A row with no prediction has no error.

`solved_mask` ([classify.py:89](../scripts/analysis/v4/modules/classify.py#L89))
has an all-`BASELINE` branch for the Shortest-Ping control, which has no
LTD/MTL/CTR pipeline and so no fallback path. Without it the control's accuracy
would divide by zero answers.

Because the rungs are exactly nested, `accuracy_ring0` is **monotone
non-increasing** as cells grow — a guarantee, not an observation. `classify
--strict` (on by default) fails on a violation, because one means the grid or
the metric is wrong, not the data. The same question on H3 produced 705
violations on this repo's data.

---

## ③ Build the table — the one place the layouts diverge

### `compare` — [`build_table`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L380)

Read each run's `accuracy.csv`, tag `dataset = run_id.split("-")[0]`,
concatenate. Runs stay separate; `render` draws one panel each.

### `pooled` — [`pooled_table`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L463)

Go back to the **per-target** `<method>_cells.parquet`, concatenate across runs,
and hand the result back to `C.summarize`.

Re-scoring rather than adding up the summaries is the whole design, and it buys
three things:

1. counts are **recounted**, so `guard_partition` applies to the pooled row and
   the stack provably closes;
2. `accuracy_ring{0,1,2}` and the retired number are true pooled **rates**;
3. `error_km_p50/p90` are true pooled **quantiles**.

(3) is not cosmetic. Order statistics cannot be averaged, and there is no
weighting of the summaries that recovers them:

| method | pooled p50 | mean of the 3 runs' p50s | Δ |
|---|---:|---:|---:|
| `million_scale_cbg` | **96.1 km** | 192.2 km | −96.1 |
| `shortest_ping` | **115.7 km** | 197.0 km | −81.3 |
| `vanilla_cbg` | **198.6 km** | 171.3 km | +27.3 |

Concatenating the rows is sound because everything `summarize` touches is either
absolute at that rung (`ring`, `tg_cell`, `pred_cell` are HEALPix ids;
`error_km` is grid-free) or compared only **within a row**
(`nearest_seed_id_retired` against `tg_seed_id`) — so the runs' per-run seed
namespaces never meet.

### Micro, not macro

The pooled bar is a **micro-average**: sum the counts, divide once.

* It answers a question about **targets** ("pick one of the 1,269 at random"),
  which is what a bar beside three per-dataset panels reads as. Macro answers
  one about **datasets**, and the three panels already show that.
* Macro silently reweights: it gives each of as01's 399 targets 1.15× the
  influence of one of as03's 458, purely because as01 is smaller.
* Micro keeps the pooled row a **count**, so it carries `n_ring0` like every
  per-run row and `guard_partition` applies unchanged. A macro share has no
  integer behind it, and nothing for the percentiles to be recomputed from.

On these three meshes the two differ by 0.15–0.33 pp and **rank the methods
identically** — which is the argument for taking the defensible one rather than
the flattering one, since neither looks better.

Both close at 100%, so closure does not decide it. The two readings of
"count-weighted average" are in fact the same number:

```
Σ_r c[r,seg] / Σ_r n[r]  ==  Σ_r (n[r]/N) · share[r,seg]
```

and because the weights are per-run rather than per-segment, the right-hand form
closes at 1 too. Summing recounted integers is just the form that cannot
accumulate float error.

**What the pooled bar is not:** a method's accuracy *in general*. It is its
accuracy on this target mix, and as03 is 36% of it. The manifest records
`largest_share` and a `reading_caveat` so the number cannot be over-read.

### Three guards, none of which v4 had

| guard | why |
|---|---|
| [`guard_common_methods`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L403) | a method absent from any run is **refused**, not pooled over the runs that carry it — otherwise bars in one panel rest on different denominators and the title's `n=` is true of some and not others |
| uniform denominator (inside `pooled_table`) | coverage matches method *names*; two methods can share a name set and still have been scored on different target counts, since `score_rung` builds each method's frame from whichever folds carry it |
| [`guard_disjoint_targets`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L427) | one shared `target_id` sits in the pooled denominator twice. v3 guards the same thing in `cross.guard_disjoint_targets`; v4 had nothing |

All three name their remedy (`--method` to narrow, or `--layout compare`).
Verified disjoint today: 0 overlap across the three meshes, unique within each.

Both paths end at the same shape, and the shares stay **exact**:

```python
table[f"share_{seg}"] = table[seg] / table["n_targets"]   # never rounded
```

---

## ④ Rank and render

### Ranking

[`_with_rank_keys`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L561)
accumulates the exclusive **counts** into a tolerance ladder, then divides and
rounds **once**:

```
within_ring0 ⊆ within_ring1 ⊆ within_ring2 ⊆ within_beyond
```

`_sorted_methods` sorts on that tuple, all descending, so a tie cascades
outward: level at 51 km and better at 102 km is genuinely better.

Two traps already paid for, both recorded in the code:

* **Accumulate counts, not rounded shares.** Summing already-rounded shares put
  `within_beyond` at 1.01, which is not a share of anything.
* **Round *before* ranking.** as01 at nside 16 had `million_scale_cbg` in-cell
  on 259 of 399 and `octant_cbg_spl` on 258 — 0.6491 vs 0.6466, both printing as
  65%. Ranking on the exact share showed `65%` ahead of `65%` with the second
  bar visibly stronger one ring out, and no way for a reader to tell the sort
  was right.

Each panel ranks **itself**, so a method does not keep one x slot across panels.
Deliberate: Octant-Spline leads as01 while Octant-Hull leads as02 and as03, and
a single pooled order would hide it.

### Rendering — [`render`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L645)

Stacked bottom-up in `SEGMENTS` order with `bottoms += vals`, on the **exact**
shares. Rounding lives in exactly two places: the drawn label and the sort key.

Colour is a single-hue light→dark ramp for the four *placed* outcomes; "no
answer" is a light grey **outside** that ramp, because it is not a worse
placement, it is the absence of one. Both were measured with the dataviz
skill's `validate_palette.js`, not chosen — a green ramp plus red for "further
out" is dE 1.8 under protanopia, and any *mid* grey collides with the ramp under
deuteranopia at dE 2.6–5.0. See
[`SEGMENT_INK`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L191).

---

## Two rendering gotchas worth keeping

Both were found by looking at the PNG, not by reading the code, and both are now
guarded by tests that assert on rendered output rather than on arguments.

### Stroke overhang makes a segment look wider

matplotlib **centres** a patch edge on the boundary, so half the stroke lies
outside the patch. White-on-white that overhang is invisible and the bar reads
at its true width; grey, it shows. The "no answer" slot carried a grey hairline
for delineation and rendered **523 px against 521** for the four greens.

In a stacked share chart apparent width is a *quantitative* channel, so it must
not vary by outcome. Every segment now takes `edgecolor=_SURFACE`. The grey edge
survives on the **legend swatch**, which is where it was load-bearing anyway — a
~10 px block of a 1.44:1 fill floating alone on white needs it, and the overhang
is harmless on a mark that sits beside no other mark.

`TestSegmentWidth` measures the rendered pixels, because the `edgecolor`
argument was not the thing that was wrong — the rendering was, and a test on the
argument would have passed against the bug.

### Figure width must come from the furniture, not the panel count

Width was `4.6 × n_panels`, which is fine at three panels and clips both ends of
the suptitle and the legend at one. The floor is now measured, not estimated
(`get_window_extent` at 150 dpi):

```
7.44 in  suptitle, pooled
6.07 in  legend, five entries in one row
5.88 in  suptitle, compare
```

`_MIN_FIG_W = 8.0`, with the **axes inset** rather than stretched — stretching
would have widened the bars with them, and the same bar should not change
thickness because a reader asked for one dataset instead of three.

A measured constant goes stale the moment someone edits a string it was measured
against: the 7.4 first tried here sat **0.04 in** under the pooled title.
`TestFigureWidth` re-measures both, so that edit fails a test instead of
silently clipping.

---

## Running it

```bash
export PATH="$PWD/.venv/bin:$PATH"     # rules shell out to bare `python`

python -m scripts.analysis.v4.cli build-answer-space --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli classify           --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli plot-outcome-bars \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
```

`--nside` selects rungs (default: the full ladder). `--layout` selects views
(default: **both**) — they read the same `accuracy.csv` files, so pooled costs
only its parquet reads. There is no `--grid`; see
[`healpix.py`](../scripts/analysis/v4/modules/healpix.py) for why.

`plot-outcome-bars` has no `--all-runs`, deliberately: the figure *is* the
comparison between named datasets, and sweeping every run on disk would silently
mix populations that were never meant to share an axis.

## Artifacts

```
outputs/analysis/v4/_cross/cls-accuracy/as01+as02+as03/
  outcome_bars.healpix-<n>.{png,csv,manifest.json}         ← one panel per dataset
  outcome_bars.pooled.healpix-<n>.{png,csv,manifest.json}  ← all of them, count-weighted
```

The pooled triple takes a `.pooled.` **infix** rather than v3's separate stem
(`outcome_bars` vs `outcome_bars_by_dataset`), so the two layouts sort together
and share one prefix to glob.

Directories are keyed on the **dataset set**, so a two-run comparison cannot
overwrite a three-run one.

## The invariant chain, in one line

> five exclusive masks in `summarize` → `guard_partition` asserts the partition
> → shares divide by the same `n_targets` → geometry is never rounded → the
> stack reaches exactly 1.0 → and every segment now *draws* at the same width

## No traffic-weighted arm

None is drawn, and none should be until a weighted run exists. v3 filled that
half of its figure from a hard-coded dict and rendered 99.3% bars that measured
nothing; the pooled and compare manifests both carry a `weighted_arm` key
saying so explicitly, rather than leaving the absence to be inferred.
[`WEIGHTED_HATCH`](../scripts/analysis/v4/modules/figure_outcome_bars.py#L220)
reserves the hatch channel so a later figure does not spend it on an outcome —
which is also why no outcome uses a hatch today.
