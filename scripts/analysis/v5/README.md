# analysis/v5: two partitions, two labels

v4 graded a prediction on one axis, **distance**: the HEALPix `ring`. That axis
is bounded, but it has no direction. A miss one ring out can land inside the
TG's own serving region or inside a neighbour's, and an operator needs to know
which. v5 adds a second partition, the **unbounded** Voronoi **cell**, so
every prediction carries `(ring, cell_label)`. One resolution: nside 128.

Scope: the answer space, classify, the answer-space map, the outcome bars,
the error-distance CDF and the VP-proximity violins.

## Glossary

Every name in v5 follows this table. `answer_space.GLOSSARY` writes it into
every `meta.json` and manifest.

| term | meaning | names |
|---|---|---|
| target (TG) | one target server behind an IP | `tg_*` |
| grid | one HEALPix pixel at nside 128, NESTED | `grid_*` |
| grid_km | nominal grid distance, √(grid area): 50.9 km | `grid_km` |
| ring | grid steps from the TG's grid to the prediction's grid (0, 1, 2; −1 = beyond) | `ring` |
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
| `ring` | 0 · 1 · 2 · −1 (beyond) | grid adjacency, local by construction |
| `cell_label` | `correct` · `wrong` · `unanswered` | nothing — it is unbounded |

Every prediction falls in its nearest seed's cell: `correct` when that seed is
the TG's seed, `wrong` when it isn't. `unanswered` is a row with no prediction.
The three partition `n_tgs`, so both axes share one denominator.

The cell label **is** v4's retired nearest-seed rule, deliberately. v4 retired
it because it credited Seattle with a prediction in the Canadian Arctic
2,360 km away, and fixed that with a landmass polygon and a fourth label,
`outland`. v5 removes both: that prediction is `correct` on the cell axis and
`ring == −1` on the grid axis (`test_classify.TestTheArcticCase`), and the two
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
outputs/analysis/v5/_cross/vp-proximity/<datasets>@<arm>/vp_proximity.<cohort>.{png,csv,manifest.json}
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

## Guarantees

- `ring` and `pred_dist_to_tg_km` match v4's `ring` and `error_km` row for row
  on all three meshes (`test_real_runs`).
- The grid axis is untouched by the cell axis: `accuracy_ring{0,1,2}` is never
  conditioned on `cell_label`, so removing the landmass moved none of them.
  Verified bit-for-bit against the pre-removal artifacts at nside 128.
- Every seed gets a cell, and the cells cover the whole drawn frame with no
  gap. An unbounded partition leaves nothing unassigned — the one
  simplification the landmass removal buys (`test_real_runs`).
- Seeds never outnumber sites. They're cuts of one complete-linkage tree.

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
python -m pytest scripts/analysis/v5/tests -q
```
