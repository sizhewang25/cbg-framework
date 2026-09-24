# analysis/v5: two partitions, two labels

v4 graded a prediction on one axis, **distance**: the HEALPix `ring`. That axis
is bounded, but it has no direction. A miss one ring out can land inside the
TG's own serving region or inside a neighbour's, and an operator needs to know
which. v5 adds a second partition, the landmass-bounded Voronoi **cell**, so
every prediction carries `(ring, cell_label)`.

Scope: the answer space, classify, the answer-space map, the outcome bars,
the error-distance CDF and the VP-proximity violins.

## Glossary

Every name in v5 follows this table. `answer_space.GLOSSARY` writes it into
every `meta.json` and manifest.

| term | meaning | names |
|---|---|---|
| target (TG) | one target server behind an IP | `tg_*` |
| grid | one HEALPix pixel at a rung (nside), NESTED | `grid_*` |
| grid_km | nominal grid distance, √(grid area): 50.9 / 101.9 / 203.7 / 407.5 km | `grid_km` |
| ring | grid steps from the TG's grid to the prediction's grid (0, 1, 2; −1 = beyond) | `ring` |
| site | a unique location of TGs, keyed `(run_id, tg_lat, tg_lon)` | `site_*` |
| seed | spherical centroid of sites grouped by complete linkage, diameter ≤ `grid_km` | `seed_*` |
| landmass | US mainland (lower 48; Great Lakes inland) buffered by `grid_km` | `landmass_*` |
| cell | Voronoi cell of a seed bounded by the landmass, i.e. the serving region | `cell_*` |
| answer space | the TG answer space: grid partition and cell partition at one rung | |
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
| `cell_label` | `true` · `wrong` · `outland` (· `none`, no prediction) | the landmass |

`outland` means the prediction lies outside the landmass buffered by `grid_km`.
Otherwise the prediction falls in its nearest seed's cell, which is `true` when
that seed is the TG's seed and `wrong` when it isn't.

The cell label is **not** v4's retired nearest-seed rule. That rule had no
outland, so it credited Seattle with a prediction in the Canadian Arctic
2,360 km away. Here that prediction is `ring == −1` and `outland` at every
rung (`test_classify.TestTheArcticCase`).

## One tolerance, three uses

`grid_km` is the grid pitch, the complete-linkage diameter for seeds, and the
landmass buffer. So both partitions are built at the same granularity. EWR and
JFK (33 km apart) fall in two grids but share one seed at nside-128. Seeds are
a logical grouping of sites and don't depend on where grid boundaries fall.

Complete linkage caps the group **diameter**. Single linkage would chain sites
40 km apart into one group of any length (`test_seeds.test_complete_linkage_does_not_chain`).

The landmass is the Natural Earth 110m country polygon, vendored at
`data/us_mainland.geojson` (rebuild with `python -m scripts.analysis.v5.data.make_us_mainland`),
buffered in EPSG:5070. The whole polygon is buffered, land borders included,
so at nside-16 Toronto and Vancouver count as inland. At that grid size they
can't be told apart from the US side.

## Outputs

```
outputs/analysis/v5/<run>/answer-space/healpix-<n>/{grids,sites,seeds,tgs}.csv, meta.json
outputs/analysis/v5/<run>/answer-space/sweep.csv
outputs/analysis/v5/<run>/classify/healpix-<n>/accuracy.csv, <method>_tgs.parquet, manifest.json
outputs/analysis/v5/<run>/classify/accuracy_by_grid.csv
outputs/analysis/v5/<run>/classify/error_cdf.{png,csv,manifest.json}
outputs/analysis/v5/_cross/classify/<datasets>@<arm>/outcome_bars.*, error_cdf.pooled.*
outputs/analysis/v5/_cross/vp-proximity/<datasets>@<arm>/vp_proximity.<cohort>.{png,csv,manifest.json}
```

`accuracy.csv` contains:

- the grid axis: `accuracy_ring{0,1,2}` (cumulative) and `n_ring{0,1,2}, n_beyond, n_failed`, which partition `n_tgs`;
- the cell axis: `accuracy_cell_true` and `n_cell_{true,wrong,outland}`, which sum to `n_solved`;
- the cross-tab: `n_{ring0,ring1,ring2,beyond}_cell_{true,wrong,outland}`, exclusive, where each tier's three labels sum to that tier's count (`guard_cross_tab`);
- percentiles of `pred_dist_to_tg_km` and `pred_dist_to_seed_km` over solved rows.

The denominator is every evaluated TG. FALLBACK and ERROR rows count as wrong.
A FALLBACK row still gets both labels, but it isn't counted.

## Figures

**`plot-answer-space`** draws one 2x2 per run, one panel per rung, with both
partitions on it: the full HEALPix lattice with the TG grids filled, the cell
boundaries, the buffered landmass (dashed blue), sites (dots) and seeds
(crosses). It's written to `answer-space/answer_space_map.healpix.png`. The
cells are drawn as a planar Voronoi in EPSG:5070, with edges densified before
converting back to lon/lat. On a point sample they agree with the great-circle
nearest-seed rule that `classify` uses on 99.4–99.7% of inland points, and the
manifest records that share per rung.

**`plot-outcome-bars`** draws one figure per rung, in two layouts: `compare`
(one panel per dataset) and `pooled` (a micro-average). They are written to
`_cross/classify/<datasets>@<arm>/`. Each bar stacks the **cell label**
first, bottom-up (`true`, `wrong`, `outland`, no answer), and breaks each
group down by **ring tier**. That asks "right serving region or not, and
within that, how far off?". The encoding is as follows:

- **Colour = ring tier**: green (in the TG grid), light blue (1 ring out),
  light purple (2 rings out), light grey (further out), dark grey (no answer).
- **Stripe = cell label**: plain for `true`, `//` for `wrong`, `\\` for
  `outland`.
- **Borders and separators:** each cell-label group has a thick border, and
  white lines separate the ring tiers inside it.
- **Labels:** every sub-segment wide enough prints its own share.
- **Rail:** right of each bar, one white striped segment per group, with the
  group total set vertically beside it.

Each panel ranks methods by `true` share, then by how tight their true
predictions are (`true & ring0`, `true & <=ring1`, ...).

**`plot-error-cdf`** (ported from v4) draws the empirical CDF of
`pred_dist_to_tg_km`, one curve per method, on a log x axis. S-P is the
dark-grey dashed baseline. There are two layouts: `per-run`, written to
`classify/error_cdf.*`, and `pooled`, written to `_cross/.../error_cdf.pooled.*`.
The pooled layout concatenates the runs' rows and recomputes the percentiles
rather than averaging them. Unanswered rows are excluded (`solved_mask`), so
each curve covers the outcome bars' answered stack. A percentile box under the
legend gives p5/25/50/90/95 and `plotted/total`. The distance is to the raw
TG, never to the seed, so it's the same at every rung. That's why the
filenames carry no `healpix-<n>`. p50/p90 match `accuracy.csv` digit for
digit.

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
  on all three meshes at every rung (`test_real_runs`).
- `accuracy_ring0` is monotone across the ladder. The cell axis has no such
  guarantee, because seeds are regrouped at every rung, so it's reported but
  not asserted.
- Seeds only merge as grids grow. They're cuts of one complete-linkage tree.

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
python -m scripts.analysis.v5.cli plot-vp-proximity -c p5 -c p25 -c all \
    --run-id as01-260728-260802-mesh \
    --run-id as02-260728-260802-mesh \
    --run-id as03-260728-260802-mesh
python -m pytest scripts/analysis/v5/tests -q
```
