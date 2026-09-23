# analysis/v4 — HEALPix answer space and laddered classification accuracy

An MVP replacement for v3's scoring layer. One grid, a correctness rule that is
bounded, and the figures that read it.

## Why this package exists

v3 scored a prediction by **nearest-seed assignment**: whichever class seed was
closest won. A Voronoi partition over K seeds labels every point on Earth, so
the rule can never answer "this is in no class". On `as01-260728-260802-mesh`,
target `tg-e1a1545` has its truth in Seattle and Spotter predicted the Canadian
Arctic — **2,360 km away, scored CORRECT**, because the nearest of 18 US seeds
was Seattle at 2,380 km and the runner-up Omaha at 2,492 km.

Not anecdotal. Among v3's *correct* classifications on as01, 345 of 354 Spotter
predictions sat more than 45 km from the seed they were credited to (210/210 on
as02, 263/268 on as03), while the answer space's own metadata said no real
target is more than 25.3 km from its seed.

## The replacement

**Containment, graded by ring, over a ladder of cell sizes.**

| ring | meaning |
|---|---|
| 0 | the prediction is in the target's own cell |
| 1 | in one of its 8 neighbours |
| 2 | in the second ring |
| -1 | further out — *unplaced*, deliberately not a number |

Local by construction: no arrangement of far-away cells can make a distant cell
adjacent. The Arctic prediction is `-1` at every rung.

Resolution is the tolerance dial. nside 128 asks "the right 51 km cell?",
nside 16 asks "the right 407 km region?".

## Why HEALPix and not H3

The metric needs `accuracy_ring0` to be **monotone** as cells grow — otherwise
"this method achieves nside-64 accuracy" is not a well-formed claim. That
requires the children of a cell to tile their parent exactly.

H3 is aperture-7 and hexagons cannot tile hexagons: a parent's 6 outer children
straddle its boundary. Measured on this repo's real predictions, H3 produced
**705 monotonicity violations**, visible even in aggregate — `vanilla_cbg` on
as01 scored 0.594 at res 1 and 0.659 at res 2, higher than its own parent.
Re-binning and `cell_to_parent` also disagreed on 552/5,906 targets at res 2.

HEALPix is aperture-4 with exact nesting: **0 violations**, asserted on all
three meshes in `tests/test_real_runs.py`. It is also exactly equal-area (H3
res-4 varies 33%), and in NESTED ordering the parent is `pix >> 2`, so the whole
ladder comes from one `ang2pix` pass. HTM (Spotter's own grid) is also
aperture-4 and also measured 0 violations, but is not equal-area and would need
a new locator.

## What it produced

`accuracy_ring0` at nside 128 against the rule it replaces:

| method | retired nearest-seed (as01/02/03) | ring0 same-cell (as01/02/03) |
|---|---|---|
| octant_cbg_hull | 0.694 / 0.624 / 0.498 | 0.366 / **0.240** / **0.207** |
| octant_cbg_spl | 0.669 / 0.582 / 0.417 | **0.368** / 0.204 / 0.205 |
| shortest_ping | 0.637 / 0.369 / 0.432 | 0.301 / 0.180 / 0.175 |
| million_scale_cbg | 0.662 / 0.366 / 0.397 | 0.301 / 0.148 / 0.170 |
| vanilla_cbg | 0.408 / 0.306 / 0.288 | 0.043 / 0.044 / 0.098 |
| spotter_cbg | **0.862** / 0.510 / 0.493 | **0.000** / 0.000 / 0.000 |

The ranking inverts. Spotter led as01 under nearest-seed and is last under
containment, at exactly zero on all three datasets — its H3-cell-centre
estimator never once lands in the truth's own cell, which the old rule could not
see. `accuracy_nearest_seed_retired` is kept in the output **for this comparison
only**; its name says so and the manifest says not to publish it.

## The figure

`plot-outcome-bars` draws one stacked bar per method, partitioned into **where
the prediction landed**: in the cell · 1 ring out · 2 rings out · further out ·
never answered. One figure per rung, three dataset panels each.

v3's three segments (correct / wrong / failed) collapsed the middle three into
"wrong", which is precisely what the ring metric exists to expose.

`shortest_ping` is included. It has no `targets.parquet`, so `combo_ids`
cannot see it — it is appended explicitly and its estimate read from
`eval_source`, restricted to the evaluated roster so it shares one denominator
with every CBG arm.

Colour was **computed, not chosen**. The four *placed* outcomes take a
single-hue ramp **light to dark** — lighter is better (relative luminance
0.437, 0.253, 0.127, 0.056). "No answer" is a **light grey outside that ramp**:
it is not a worse placement, it is the absence of one, so giving it a rank on
the precision ramp would claim an ordering it does not have.

Rejected by measurement with `validate_palette.js`:

* green ramp + **red** for "further out" — red vs the ramp's mid-green is
  Delta E **1.8 under protanopia**, so a protanope cannot separate "two rings
  out" from a total miss;
* a **mid grey** for "no answer" — mid grey is exactly where green lands under
  deuteranopia; every one tested collided with a ramp step at Delta E 2.6-5.0.

Among light greys, separation from the lightest green and surface contrast pull
opposite ways, so the pick came off the measured curve: `#d8d7cf` holds Delta E
**10.8** at 1.44:1. The weak contrast is covered by a hairline edge on that
slot alone plus the in-place label and the CSV twin.

**No hatch on any outcome** — that channel is reserved for a traffic-weighted
arm drawn beside a mesh bar.

**Each panel ranks itself** by its own in-cell share, descending, so a panel
reads as that dataset's leaderboard. A method therefore does not keep one x
slot across panels — deliberate, because the orders genuinely differ
(Octant-Spline leads as01; Octant-Hull leads as02 and as03).

**The sorting key is `(in-cell, 1-ring-out, 2-rings-out, further-out)`,
descending**, each cumulative and **rounded to 2 decimals** — the same number
the labels show, so the order always follows from the values on the page. A
method identical on all four falls back to the pooled order.

Rounding before ranking matters. as01 at nside 16 has `million_scale_cbg`
in-cell on 259 of 399 targets and `octant_cbg_spl` on 258 (0.6491 vs 0.6466) —
both print as 65%. On the exact share million_scale ranked first, so the figure
showed 65% ahead of 65% with the second bar visibly stronger one ring out. At
the reported precision they tie and the 1-ring rung resolves it, 27% to 16%.

Rounding is confined to the label and the sort key; the drawn shares stay exact
so every stack closes at 100%, and the exact counts (`n_ring0` ...) stay in the
CSV twin.

**No traffic-weighted arm is drawn**: none exists, and v3 filled that half from
a hard-coded dict that rendered 99.3% bars measuring nothing.

### Two layouts

`--layout compare` is the per-dataset view above. `--layout pooled` adds a
second figure per rung, `outcome_bars.pooled.healpix-<n>`, with one panel over
every input run's targets at once — the single "how does each method do
overall?" number the per-dataset panels cannot give. **Both are written by
default**; they read the same `accuracy.csv` files, so the second costs only its
parquet reads.

Pooling is a **micro-average**: the per-target rows are concatenated across runs
and re-scored by `classify.summarize`, so a target counts the same whichever
dataset it came from. A macro-average — the mean of the three panels' shares —
would give each of as01's 399 targets 1.15x the weight of one of as03's 458
purely because as01 is smaller. Micro also keeps the pooled row a *count*, so it
carries `n_ring0` like every per-run row and `guard_partition` applies to it
unchanged. On these three meshes the two differ by 0.15–0.33 pp and rank the
methods identically, which is the argument for taking the defensible one.

Re-scoring rather than adding up the summaries is what makes `error_km_p50/p90`
correct. They are order statistics: the pooled p50 for `million_scale_cbg` is
**96 km** against **192 km** for the mean of the three runs' p50s, and no
weighting of the summaries recovers it.

Coverage is **strict** — a method absent from any input run is refused rather
than pooled over the runs that carry it, so every bar in the panel rests on the
same denominator and the title's `n=` is true of all of them. Overlapping target
ids across runs are refused for the same reason. Both refusals name the remedy
(`--method` to narrow, or `--layout compare`).

What the pooled bar is **not** is a method's accuracy in general. It is its
accuracy on *this* target mix, and as03 is 36% of it; the manifest records the
largest run's share so the number cannot be over-read. The compare layout stays
the place to see per-dataset divergence.

## The second figure: the Euler diagram

`plot-outcome-bars` says *how much* each method places in the cell. It cannot
say **whether those are the same targets** — two methods at 24% might agree on
all of them or on none. `plot-euler` is that question and only that question:
circle area is a method's share, shared area is the share **both** place, a set
inside another is drawn inside it, and two that never agree are drawn apart.

It is a **`_cross` figure at one resolution**. Where the bars sweep the rungs,
this fixes one (`--nside`, default **128**) and sweeps the **tolerance**:

| `--top-n` | correct means | printed as |
|---|---|---|
| 1 | `ring == 0` | in the cell |
| 2 | `ring <= 1` | within 1 ring |
| 3 | `ring <= 2` | within 2 rings |

That is the axis worth sweeping here, because it is the one that changes *which
targets are in which set* and therefore the only one that changes the overlaps.
Sweeping nside instead would produce four fitted layouts whose circles cannot be
compared by eye; the ladder already has `accuracy_by_resolution.csv` and the
bars.

**`top_n` no longer means a seed rank.** v3's ranked class seeds by distance to
the prediction, which is the rule v4 retired — a Voronoi partition labels every
point on Earth, so its rank-1 set contained a prediction 2,360 km from its
truth. Here it is a containment tolerance, cumulative, so each circle can only
grow as `top_n` rises.

Pooled over as01+as02+as03 at nside 128, `placed` 97.0% (worst pair error 0.9%):

| top-N | any method | none | largest set | what the picture shows |
|---|---|---|---|---|
| 1 | 39.6% | 60.4% | Octant-Hull 26.8% | Shortest-Ping ⊂ ~SoI; the two Octants overlap heavily but keep exclusive lobes |
| 2 | 61.5% | 38.5% | SoI / Shortest-Ping 49.1% | Spotter's 8.2% sits almost entirely inside the others |
| 3 | 67.8% | 32.2% | SoI 54.1% | every set overlaps every other; the tolerance has stopped discriminating |

**`spotter_cbg` has no circle at top-1.** It places zero targets in the truth's
own cell on all three meshes, and a zero-radius circle is a dot a reader takes
for "very small" rather than "never". Dropping it is *lossless*, not a
convenience: a set nothing belongs to appears in no region, so every other
region's count is unchanged — asserted in `test_dropping_an_empty_set_changes_no_region`
rather than argued. It stays in the denominator, stays in `intersections.csv`
and `pairwise.csv` with its zeroes, and is named in the figure's footnote and
in the manifest. Only if fewer than two sets survive is the figure skipped.

v3 instead skipped the whole figure whenever any method was empty, which at
these numbers means the default invocation produces nothing.

### What the layout can and cannot do

`n` circles have `2n` degrees of freedom against `2**n - 1` regions, so past two
sets the system is overdetermined and some combination is always misdrawn. The
figure prints its own `placed` share and worst pair error, and
`euler.<slug>.top<N>.fit.csv` is the row-by-row audit — observed against drawn,
with `delta` positive on exactly the regions the picture asserts and the data
denies.

Labels are placed **globally**, which is the one part of the geometry that is
not v3's. Every circle offers the same menu — its exclusive lobe's pole of
inaccessibility plus 36 points around its own boundary — and positions are
taken smallest circle first, each maximizing distance to the names already
placed (capped at one label's width) with a bonus for sitting in its own
exclusive lobe, *scaled by how clear the position already is* so exclusivity
can never buy a collision. v3 chose each name independently and spread only
some of them afterwards, which drew "SoI 49.1%" on top of "Octant-Spline 40.4%".

Every name still sits **inside the circle it names**, which is what lets the
figure drop leader lines entirely.

## The third figure: the answer-space map

`plot-answer-space` draws the grid itself — one panel per rung, the whole ladder
in a 2x2. Target cells filled, VP cells outlined, and **every empty cell of the
frame** inked as a hairline lattice.

The two bar figures answer "how well did a method do". This one answers the
question underneath them: *what was it being asked?*

### What is different from v3's

v3's `answer_space_map.png` drew one resolution with a red dashed **nearest-seed
Voronoi partition** over it. That partition was not decoration — it was the
classifier's own decision boundary, so the map was incomplete without it. Under
containment there is nothing equivalent to draw: the cell edge *is* the boundary.
Two thirds of v3's `mapping.py` (`SeedVoronoi`, the azimuthal-equidistant
reprojection, the 200 km re-segmentization) exists to reconstruct that geometry
and has no counterpart here.

v3 also drew **occupied cells only**, on the grounds that the full 288,122-cell
grid "would be both unreadable and pointless at continental scale". True
globally, false in a US frame: the mainland window holds 5,740 cells at nside 128
and 92 at nside 16, one `PolyCollection` each.

Drawing the empty ones is the point. The claim the package rests on is that the
grid is **fixed, not fitted** — a class boundary falls where HEALPix falls, not
where the targets are sparse — and the empty cells are the only direct evidence
of it. Occupied cells alone look exactly like a clustering, which is the reading
the answer space must not invite (the benchmark also ships an older `clusters/`
answer space built by radius-capped agglomeration, and the two are not the same
object).

### Why the whole ladder and not one rung

Resolution is the tolerance dial, so how the cells merge as it turns is the
figure. On as01 the target cells go 18 -> 18 -> 17 -> 15 and the cells holding
more than one distinct operator site go 2 -> 2 -> 3 -> 5. This is the picture
behind `occupancy_by_resolution.healpix.csv`, and a single rung is a different,
smaller figure — so there is no `--nside`.

### Encoding

VP cells outnumber target cells 4:1 (80 against 18 at nside 128), so two fills of
equal weight put the ink on the wrong side. **Fill for targets, outline for VPs**
instead: the overlap then reads compositionally — orange inside a blue border —
with no third hue and no alpha blend, whose result is neither predictable nor
validatable. A target cell *without* a VP keeps its own darker orange edge, so
`share_of_target_cells_with_a_vp` is legible cell by cell.

`#eb6834` / `#2a78d6` are slots 1 and 2 of the dataviz reference theme, validated
with `validate_palette.js --mode light --pairs all` (all-pairs, because a map is
a choropleth form): all five checks pass, worst CVD Delta E **24.7** under
protanopia. Blue is also v3's `map_bipartite._VP_COLOR` unchanged, so the two
packages do not disagree about which side is which.

**The dots are distinct target coordinates, not targets.** These meshes carry
~20 IP replicas per coordinate — 399 targets over 20 sites on as01 — so a
per-target scatter would put 20 marks on one pixel and say nothing. Deduplicated,
a cell holding two dots *is* two operator sites quantized into one class. Target
site and VP take one marker size between them: they are two categories, not two
magnitudes.

### Known limit

At nside 128 a cell is ~19 px in a continental panel, so the finest rung reads as
graph paper with dots on it and the orange is largely hidden under its own site
marker. Making that rung informative needs a metro inset, not a bigger figure.

## Usage

```bash
python -m scripts.analysis.v4.cli build-answer-space --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli build-bipartite    --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli classify           --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli plot-answer-space  --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli plot-outcome-bars \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m scripts.analysis.v4.cli plot-euler \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
```

`classify` needs the answer space; `build-bipartite` is independent. The two
accuracy figures need `classify` and take a repeatable `--run-id`, because each
*is* the cross-dataset comparison. There is no `--grid`.

`plot-answer-space` is the exception on both counts: it is **per-run**, so it
takes one `--run-id` or `--all-runs`, and it needs only `build-bipartite` —
not `build-answer-space`. One occupied target cell is exactly one class, so the
bipartite artifacts already carry the answer space, and the VP side with it,
co-quantized on the same grid.

`plot-answer-space` frames the continental US by default; `--auto-extent`
derives the frame from the run's own targets and VPs instead, and `--extent`
takes one literally.

`plot-outcome-bars` sweeps rungs: `--nside` (default: the full ladder) and
`--layout` (default: both views). `plot-euler` takes a **single** `--nside`
(default 128) and sweeps `--top-n` instead (default: 1, 2 and 3). Roughly 6s
per fit, so the default pass is ~18s for three figures.

## Layout

```
outputs/analysis/v4/<run_id>/
  target-answer-space/healpix-{128,64,32,16}/   seeds.csv assignments.csv seed_mesh_km.csv meta.json
  target-answer-space/grid_sweep.healpix.csv            <- class-count curve
  bipartite-graph/healpix-<nside>/              target_cells.csv vp_cells.csv cell_occupancy.csv meta.json
  bipartite-graph/occupancy_by_resolution.healpix.csv   <- geometry curve
  bipartite-graph/answer_space_map.healpix.{png,manifest.json}  <- the map, all four rungs
  target-cls-accuracy/healpix-<nside>/          <method>_cells.parquet accuracy.csv manifest.json
  target-cls-accuracy/accuracy_by_resolution.healpix.csv <- accuracy curve

outputs/analysis/v4/_cross/cls-accuracy/<dataset-set>/
  outcome_bars.healpix-<nside>.{png,csv,manifest.json}         <- one panel per dataset
  outcome_bars.pooled.healpix-<nside>.{png,csv,manifest.json}  <- all of them, count-weighted
  euler.healpix-<nside>.top<N>.png                             <- the fitted layout
  euler.healpix-<nside>.top<N>.fit.csv                         <- observed vs drawn, per region
  euler.healpix-<nside>.top<N>.intersections.csv               <- exact counts, per combination
  euler.healpix-<nside>.top<N>.pairwise.csv                    <- both / a-only / b-only / neither
  euler.healpix-<nside>.top<N>.membership.csv                  <- the boolean matrix itself
  euler.healpix-<nside>.top<N>.manifest.json
```

The Euler set carries `.top<N>` because a figure is a function of the
tolerance — without it a top-3 pass would overwrite the top-1 files — and the
grid slug because this directory is keyed by dataset set alone. Both figures
write to the **same** directory: one comparison's artifacts belong together
regardless of which command produced them.

The pooled triple takes a `.pooled.` **infix** rather than v3's separate stem
(`outcome_bars` vs `outcome_bars_by_dataset`) so the two layouts sort together
and share one prefix to glob.

One rung per directory, following v3, so each rung is a complete self-describing
artifact and a sweep cannot overwrite one rung with another. The merged CSVs one
level above join the ladder for plotting — and the map sits beside them, for the
same reason: it spans the ladder, so no single rung owns it. It writes no CSV
twin, because `occupancy_by_resolution.healpix.csv` already holds every number it
draws and a second copy would be a thing to keep in sync for nothing.

## Relationship to v3

Independent: v4 imports nothing from v3 and v3 is untouched, so both sets of
numbers survive on disk. **They are not comparable** — nside 128 is 50.9 km
against h3-4's 45.2 km, and the correctness rule differs. Do not put them in one
table.

v4 reads the **v2 benchmark** tree unchanged, so it inherits that population and
fold contract as-is.

## Not in scope

`GaussianDensityMTL` keeps its H3 hypothesis grid. That is a search space over
candidate locations, independent of the scoring grid, and moving it needs the
`top_k`/`neighbor_ring` basin-miss experiment re-run — commit `9e9df7d`
validated `top_k=8, neighbor_ring=1` for H3 2->4 only, and HEALPix 16->128
starts coarser (407 km vs 316 km) over three levels rather than two.
