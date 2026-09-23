# analysis/v4 — HEALPix answer space and laddered classification accuracy

An MVP replacement for v3's scoring layer. Three modules, one grid, and a
correctness rule that is bounded.

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
| million_scale_cbg | 0.662 / 0.366 / 0.397 | 0.301 / 0.148 / 0.170 |
| vanilla_cbg | 0.408 / 0.306 / 0.288 | 0.043 / 0.044 / 0.098 |
| octant_cbg_hull | 0.694 / 0.624 / 0.498 | 0.366 / 0.240 / 0.207 |
| octant_cbg_spl | 0.669 / 0.582 / 0.417 | **0.368** / 0.204 / 0.205 |
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

Colour was **computed, not chosen** — the four placed segments take a
single-hue ordinal ramp, validated with the dataviz skill's
`validate_palette.js --ordinal` (monotone lightness, adjacent gaps >= 0.06,
light end 2.10:1, hue spread 5 degrees; all pass). Two earlier designs were
rejected by measurement:

* green ramp + **red** for "further out" — red vs the ramp's mid-green is
  Delta E **1.8 under protanopia**, so a protanope cannot separate "two rings
  out" from a total miss;
* a **grey** fifth fill for "never answered" — every grey tested collided with
  some ramp step under deuteranopia (Delta E 1.6-4.5), because greens desaturate
  toward grey exactly there.

So "never answered" carries **no fill** — an outlined, hatched slot. An absence
cannot be confused with a hue, and nothing was produced to colour.

The x order is ranked once at the finest rung and reused for every rung, so a
method keeps its slot across the figure set. **No traffic-weighted arm is
drawn**: none exists, and v3 filled that half from a hard-coded dict that
rendered 99.3% bars measuring nothing.

## Usage

```bash
python -m scripts.analysis.v4.cli build-answer-space --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli build-bipartite    --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli classify           --run-id as01-260728-260802-mesh
python -m scripts.analysis.v4.cli plot-outcome-bars \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
```

`classify` needs the answer space; `build-bipartite` is independent.
`plot-outcome-bars` needs `classify` and takes a repeatable `--run-id` because
the figure *is* the cross-dataset comparison. `--nside` selects rungs (default:
the full ladder). There is no `--grid`.

## Layout

```
outputs/analysis/v4/<run_id>/
  target-answer-space/healpix-{128,64,32,16}/   seeds.csv assignments.csv seed_mesh_km.csv meta.json
  target-answer-space/grid_sweep.healpix.csv            <- class-count curve
  bipartite-graph/healpix-<nside>/              target_cells.csv vp_cells.csv cell_occupancy.csv meta.json
  bipartite-graph/occupancy_by_resolution.healpix.csv   <- geometry curve
  target-cls-accuracy/healpix-<nside>/          <method>_cells.parquet accuracy.csv manifest.json
  target-cls-accuracy/accuracy_by_resolution.healpix.csv <- accuracy curve

outputs/analysis/v4/_cross/cls-accuracy/<dataset-set>/
  outcome_bars.healpix-<nside>.{png,csv,manifest.json}
```

One rung per directory, following v3, so each rung is a complete self-describing
artifact and a sweep cannot overwrite one rung with another. The merged CSVs one
level above join the ladder for plotting.

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
