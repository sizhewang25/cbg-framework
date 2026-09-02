# v3 Pluggable Answer-Space Grid (HEALPix + H3) — Report

**Status**: Complete
**Created**: 2026-09-02
**Last Updated**: 2026-09-02

## Summary

`scripts/analysis/v3/` now quantizes the answer space through a `Grid` interface with
two implementations. H3 is the default at `res=4`; HEALPix remains fully supported and
produces **byte-identical** output to before the refactor. Resolutions 2-5 are supported
for H3; only res 4 was generated, as scoped.

**Delivered**

1. `modules/grid.py` — `Grid` ABC, lazily-importing registry (`get_grid`), the shared
   `resolve_cli_grid` CLI front door, the `ring_lonlat` dateline helper, and two
   concrete methods that carry most of the value: `coarsening_ladder` and
   `occupied_cell_hierarchy`.
2. `modules/h3grid.py` — `H3Grid`, verified against h3-py 4.5.0.
3. `modules/healpix.py` — kept, existing functions untouched, `HealpixGrid` appended.
   `spherical_centroid` moved out to `answer_space.py` (it was never grid math).
4. Grid-neutral schema: `grid_scheme`, `grid_resolution`, `cell_id`. No `healpix_*`
   column survives anywhere in the outputs.
5. `--grid` / `--resolution` / `--sweep` on all four commands, `--nside` removed.
   Directories are `<scheme>-<resolution>/`; sweeps are `grid_sweep.<grid>.csv`.
6. 155 tests pass (60 before). Both grids are held to one contract by parametrized
   fixtures in `test_grid.py`, `test_answer_space.py` and `test_map_answer_space.py`.

## Findings

**The refactor is a strict no-op for HEALPix.** `topn_accuracy.csv` at `healpix-128` is
byte-identical to the pre-refactor `nside-128` baseline on all four runs. as01
`vanilla_cbg` still reads `accuracy_top1=0.411`, `fallback_rate=0.2657`, `n_solved=293`,
`n_fallback=106`.

**H3 res 4 vs HEALPix nside 128** (K = classes, best = top-1 over all methods):

| run | grid | pitch | K | singleton | best top1 | vanilla top1 |
| --- | --- | --- | --- | --- | --- | --- |
| as01 | h3-4 | 45.2 km | 18 | 0 | 0.687 | 0.411 |
| as01 | healpix-128 | 50.9 km | 18 | 0 | 0.687 | 0.411 |
| as02 | h3-4 | 45.2 km | 22 | 0 | 0.634 | 0.298 |
| as02 | healpix-128 | 50.9 km | 22 | 0 | 0.634 | 0.298 |
| as03 | h3-4 | 45.2 km | 22 | 0 | 0.504 | 0.317 |
| as03 | healpix-128 | 50.9 km | 23 | 0 | 0.491 | 0.286 |
| as7018 | h3-4 | 45.2 km | 22 | 6 | 0.500 | 0.359 |
| as7018 | healpix-128 | 50.9 km | 27 | 11 | 0.397 | 0.308 |

**My pre-implementation prediction was wrong, and the way it was wrong is the finding.**
The plan predicted that because res 4 is *finer* than nside 128 (45 vs 51 km), K would be
equal or slightly higher and accuracy equal or slightly lower. On as7018 the opposite
happened: the finer grid produced **fewer** classes (22 vs 27) and **higher** accuracy
(0.500 vs 0.397), with singletons dropping 11 -> 6. Resolution is not what decides whether
a metro gets split at these scales — boundary **alignment** is. This is the same lesson as
the earlier EWR/JFK finding, now confirmed from the opposite direction: that pair is
closer than either grid's pitch and still splits on both. as01/as02 are unaffected
(K identical, accuracy identical) because their 399/412 targets sit at only ~20 distinct
coordinates, far from any boundary.

The sharpest single case, found while verifying the claim for the paper: the 10
as7018 targets in the New York metro (44 km extent) occupy **3 HEALPix classes at 51 km,
2 at 102 km, 2 at 204 km and still 2 at 407 km** — a cell nine times the group's extent
— whereas **H3 res 4, which is finer at 45 km, puts all 10 in a single class**. A finer
grid producing coarser classes is the finding in its cleanest form. It also happens to
resolve the EWR/JFK split that prompted this whole line of work, though by alignment
luck rather than by design: H3 res 5 splits the same group back into 3.

**Practical consequences**: an accuracy number must name its grid *and* resolution (two
grids at nominally the same scale differ by 10 points of top-1 on as7018); and a grid
cannot be chosen by the class counts it produces on one dataset, because that is a fact
about alignment and will not transfer. Hence fixing the grid in advance and reporting
both, rather than selecting on the outcome.

**H3's costs are now measured, not argued.** In `meta.json` under `grid_diagnostics`
(empty for HEALPix, which has nothing to disclose):

- **Not equal-area**: occupied-cell area ratio max/min is **1.33** on as01 at res 4
  (1,573-2,085 km² against a 1,770 km² average). HEALPix is exactly 1.00.
- **Pentagons**: 0 occupied on every run, as expected — H3's 12 pentagons per resolution
  sit over ocean. Reported so that is a measurement rather than an assumption.
- **Non-nesting is real and material**: `parent_lineage_disagreements` on as01 is
  **20 of 399 targets** at res 3 and **39 of 399** at res 2 — 5% and 10% of the target
  set land in a different coarse cell by `cell_to_parent` lineage than by re-binning
  from coordinates. as7018 shows 1 and 2. This is why `occupied_cell_hierarchy` re-bins
  on both grids rather than coarsening ids, even though the HEALPix bit shift is exact
  and free.

**Two pre-existing bugs fixed.** `occupied_cells_by_resolution` (was
`occupied_cells_by_nside`) now starts at the resolution actually built: `{4: 18, 3: 18,
2: 16}` for h3-4 rather than a fixed ladder that at the coarsest rung reported counts for
grids finer than the answer space. And the dead `spread_by_pix` dict is gone.

## Conclusions

All six goals met. The interface is thin enough to be worth having: `venn.py` never
touched the answer space at all, `classify.py` reads only `seed_id` plus centroids, and
10 of 12 `test_answer_space.py` invariants generalized to H3 verbatim — confirming the
seeds/Voronoi layer was already grid-agnostic in substance and only two column names were
holding it to HEALPix.

**Three problems found and fixed during implementation**

- `Grid.cell_boundaries` initially inferred HEALPix's resolution from the cell ids. That
  is unsound — cell 5 exists at every nside — so a run whose occupied ids happened to be
  small would have drawn the wrong cells with no error. Resolution is now a required
  parameter, which H3 accepts and ignores since its ids self-describe.
- Dateline-straddling cells return mixed-sign longitudes from *both* libraries (verified:
  H3 gives `[179.85, 179.75, 179.85, -179.96, ...]`). The pre-existing HEALPix code's
  `np.where(lon > 180, lon - 360, lon)` did not handle this either — such a ring renders
  as a band across the whole map. `ring_lonlat` now places every vertex within half a turn
  of the first, and a test asserts the ring spans < 10°.
- A bad `--resolution` raised a raw `ValueError` with a traceback. Consolidated the
  `--grid` whitelist and resolution validation into one `resolve_cli_grid` helper so all
  four commands give the same clean CLI error and the whitelist cannot drift.

**Verified empirically before use, not from signatures**: every h3 v4 call, following the
`venn2_unweighted` lesson. Three behaviours would have caused silent bugs —
`cell_to_boundary` returns `(lat, lng)` not `(lng, lat)`; pentagons return 5 vertices
where hexagons return 6, so rings are genuinely ragged; and `latlng_to_cell` returns a
hex `str` by default rather than an int.

## Paper draft

`papers/cbg-benchmark-as-network-operator/paper-flow-draft-v2.md` updated in six places.
The load-bearing one is the metric list, which asserted the multi-scale diagnostic is
computed "up the hierarchy ... which nesting makes a single pass". That is true of HEALPix
and **false of H3**, so it was replaced with the re-binning rule and the measured
disagreement rate (1-8% of targets at resolution 3, 3-10% at resolution 2, across the four
RIPE-derived sets). Also: the §7.3 grid definition swapped to H3 res 4 with HEALPix named
as the exactly-equal-area/exactly-nested alternative; the operator rationale and both
disclosed costs added; the straddle probability restated at w=45 km (21%/39% at 5/10 km,
was 19%/35%); the alignment-not-resolution finding added with its three consequences; and
the §1 contribution line corrected from "equal-area 51 km grid" to "45 km hexagonal grid".

Two corrections I made to my own paper text while checking it:

- I had attributed both the class-count and the lineage-disagreement numbers to "our RIPE
  data" as if one dataset. They come from different setups — the NY-metro and K figures
  from `as7018_us_test01` (probes-to-anchors, which is what §7.2 describes), the lineage
  figures from the three `anchors_to_probes` runs, whose target/VP roles are reversed.
  Now attributed separately, and the lineage claim stated as a range across all four.
- I first wrote "switching tessellation does not help either", which the measurement
  contradicts: H3 res 4 merges the New York group that HEALPix cannot merge at any
  resolution. The accurate claim is that changing tessellation moves the answer
  unpredictably because the mechanism is alignment, not scale — which is a stronger point
  and is what the passage now makes.

**Not done / deliberately out of scope**: resolutions 2, 3 and 5 are supported and tested
but no artifacts were generated for them.
