# analysis/v5: two partitions, two labels

v4 graded a prediction on one axis, **distance**: the HEALPix `ring`. That axis
is bounded, but it has no direction. A miss one ring out can land inside the
TG's own serving region or inside a neighbour's, and an operator needs to know
which. v5 adds a second partition, the landmass-bounded Voronoi **cell**, so
every prediction carries `(ring, cell_label)`.

Scope so far: the answer space and classify. Figures come later.

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
```

`accuracy.csv` contains:

- the grid axis: `accuracy_ring{0,1,2}` (cumulative) and `n_ring{0,1,2}, n_beyond, n_failed`, which partition `n_tgs`;
- the cell axis: `accuracy_cell_true` and `n_cell_{true,wrong,outland}`, which sum to `n_solved`;
- the cross-tab: `n_{ring0,ring1,ring2,beyond}_cell_{true,wrong,outland}`, exclusive, where each tier's three labels sum to that tier's count (`guard_cross_tab`);
- percentiles of `pred_dist_to_tg_km` and `pred_dist_to_seed_km` over solved rows.

The denominator is every evaluated TG. FALLBACK and ERROR rows count as wrong.
A FALLBACK row still gets both labels, but it isn't counted.

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
python -m pytest scripts/analysis/v5/tests -q
```
