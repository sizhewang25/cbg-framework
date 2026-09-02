# v3 Pluggable Answer-Space Grid (HEALPix + H3) — Plan

## Background

`scripts/analysis/v3/` quantizes ground-truth targets with an equal-area HEALPix grid at
`nside=128` to build the paper's answer space (§7.3/§7.4). HEALPix was chosen for two
exact properties: equal area, and NESTED coarsening as a bit shift (`pix >> 2k`), which
makes the multi-scale concentration diagnostic a single pass.

We now want **H3 (Uber) as an alternative and as the default**. The motivation is
operator-facing rather than mathematical: hexagons give uniform neighbour distance with
no ambiguous edge/corner adjacency, H3 is the standard in telecom RF analytics, Starlink
partitions service cells on it, and it ships natively in ClickHouse, Postgres, BigQuery,
Snowflake and Spark — so quantization could eventually move into SQL. HEALPix's marquee
property here (iso-latitude rings for fast spherical harmonic transforms) does nothing
for us; we are using it as a generic equal-area quad grid.

HEALPix stays fully supported and must produce byte-identical numbers to today.

Full design detail lives in the working plan at
`/home/nuwinslab/.claude/plans/silly-wandering-planet.md`. This file is the task-local
summary.

## Context

**Coupling surface** (mapped, complete). HEALPix does not leak outside
`scripts/analysis/v3/` — only `pyproject.toml:51` (`astropy-healpix`) and docs reference
it. No Snakefile, shell script, or module outside v3 calls these commands or reads their
output. `outputs/analysis/v3/` is gitignored and fully regenerable.

- **Irreducibly HEALPix**: power-of-two validation, `12 * nside**2`, exact-equal-area
  `pixel_area_km2`, `pix >> 2k` coarsening, `lonlat_to_healpix`, `boundaries_lonlat` with
  its uniform 4-sided rings and astropy `Quantity` return, `NSIDE_HIERARCHY`, and the
  `nominal_cell_km = sqrt(area)` identity.
- **HEALPix only in naming** (rename, semantics carry over): `seeds.csv` columns
  `healpix_nside` / `healpix_pix`; `assignments.csv` column `healpix_pix`;
  `meta["grid"]` keys; `meta["occupied_cells_by_nside"]`; `paths.nside_slug`;
  `nside_sweep.csv`; `sweep_row`'s `nside` key; `manifest.json`'s `nside`; the
  `--nside` / `--sweep` options in all four command modules; every `nside=` echo string.
- **Already grid-agnostic** (leave alone): `pairwise_km`, `_unit_vectors`, `_describe`,
  `_delaunay_degree`, the whole `AnswerSpace` dataclass and its round-trip,
  `seed_mesh_km.csv`, all of `classify.py`'s scoring path, all of `venn.py`'s set logic
  (it never reads `seeds.csv`), `paths.analysis_dir`, and 10 of 12 `test_answer_space.py`
  tests.
- **Two capability gaps** the interface must close: cell-boundary rings for the map (today
  `map_answer_space.py:127` reaches around `healpix.py` straight into `astropy_healpix`),
  and grid self-description in the written artifact so `classify` can re-derive its output
  slug.

**Dependency.** h3 4.5.0 ships a `cp312-manylinux_2_17_x86_64` wheel and the venv is
Python 3.12.12, so no compilation. API names verified against h3-py v4.5.0 source:
`latlng_to_cell`, `cell_to_latlng`, `cell_to_boundary` (returns **(lat, lng)** pairs),
`cell_to_parent`, `average_hexagon_area`, `average_hexagon_edge_length`, `cell_area`,
`get_num_cells`, `int_to_str` / `str_to_int`, `is_pentagon`.

**Resolution ladder.** 2–5 supported, **4 the default**, and **only res 4 generated**.

| grid | sqrt(area) | centre-to-centre |
| --- | --- | --- |
| h3 res 5 | 15.9 km | 17.1 km |
| h3 res 4 **(default, only one generated)** | 42.1 km | 45.2 km |
| h3 res 3 | 111.3 km | 119.5 km |
| h3 res 2 | 294.6 km | 316.1 km |
| healpix nside 128 | 50.9 km | — |
| healpix nside 64 / 32 / 16 | 101.9 / 203.7 / 407.5 km | — |

res 4 is the closest analogue to `nside=128`, so switching the default grid does not
silently change the merge scale. res 5 is Starlink's service-cell resolution; res 3 is the
metro rung.

## Goals

1. One abstract `Grid` interface in `modules/grid.py`, with `HealpixGrid` and `H3Grid`
   implementations held to the same contract by parametrized tests.
2. `--grid [h3|healpix]` on all four commands, defaulting to `h3`; `--resolution`
   (repeatable) defaulting to the grid's own default; `--sweep` resolving against the
   grid's own hierarchy. `--nside` removed.
3. Grid-neutral on-disk schema — `grid_scheme`, `grid_resolution`, `cell_id` — so no
   consumer branches on grid type.
4. Both grids coexist on disk under `<scheme>-<resolution>/` with no collisions.
5. **HEALPix numbers unchanged.** `as01-260728-260802` `vanilla_cbg` at
   `--grid healpix --resolution 128` still reads `accuracy_top1 = 0.411`,
   `fallback_rate = 0.2657`, `n_solved = 293`, `n_fallback = 106`.
6. H3 res 4 results generated for all four runs, with K and the accuracy table recorded
   against HEALPix nside 128.

## Approach

**Interface** (`modules/grid.py`) — ABC plus a lazily-importing registry
(`get_grid(name)` imports the implementation inside the function body, avoiding a cycle
and keeping both astropy and h3 lazy, matching the convention at `healpix.py:34-39`).

Two concrete methods on the base class carry most of the value:

- `occupied_cell_hierarchy` re-bins from coordinates at each rung. For HEALPix this is
  provably identical to the bit shift; for H3 it is the *only* correct route, because
  hexagons cannot tile hexagons and a parent's 6 outer children straddle its boundary.
- `hierarchy_at_or_below(resolution)` fixes a live bug (see Caveats).

`cell_boundaries` returns a ragged list of `(V, 2)` lon/lat degree arrays — H3 rings are
5–6 sided, so the current uniform 4-sided assumption cannot survive. `step` is honoured by
HEALPix (edge curvature) and ignored by H3, which returns exact vertices.

**Implementations** — `modules/healpix.py` is kept and its existing functions untouched;
it gains `HealpixGrid` at the bottom. New `modules/h3grid.py` holds `H3Grid` and its math.
`spherical_centroid` moves out of `healpix.py` into `answer_space.py` beside
`_unit_vectors`/`pairwise_km` (it is the same math and is not grid-specific), so
`h3grid.py` never imports `healpix.py`.

`H3Grid.nominal_cell_km` = `average_hexagon_edge_length(res) * sqrt(3)`, the
centre-to-centre distance — the honest merge scale for a hexagon, and the right thing to
compare a seed-pair distance against. Not `sqrt(area)`: H3 cells are not equal-area, so
`sqrt(mean area)` understates the pitch by ~7%.

Cell ids are stored as H3's canonical hex string (`int_to_str`) — what ClickHouse,
BigQuery and the H3 docs all show, and always containing `f` padding so pandas never
coerces it to an integer on read. HEALPix stays int64; `coerce_cell_ids` reconciles them
on load.

**Schema** — `seeds.csv` gains `grid_scheme` and renames `healpix_nside` ->
`grid_resolution`, `healpix_pix` -> `cell_id`; `assignments.csv` renames `healpix_pix` ->
`cell_id`; `meta["grid"]` becomes `{scheme, resolution, n_cells, cell_area_km2,
nominal_cell_km, ...scheme extras}`; `occupied_cells_by_nside` ->
`occupied_cells_by_resolution`. `nside_sweep.csv` -> `grid_sweep.<grid>.csv` (the per-grid
suffix is load-bearing: with the flat layout both grids' sweeps land in
`target-answer-space/`, and one fixed name would let an h3 sweep clobber a healpix one —
same bug class and same fix as `venn.artifact_name`'s `.top<N>`).

**Paths** — `nside_slug` -> `grid_slug(grid, resolution)` yielding `h3-4` /
`healpix-128`; `answer_space_dir` and `cls_accuracy_dir` take `grid` and `resolution` as
required keyword-only args so they cannot silently drift.

**Honest H3 diagnostics** in `H3Grid.describe()`, all cheap: `avg_edge_km`,
`cell_area_km2_min`/`_max` over occupied cells (the real area variation HEALPix does not
have), `n_occupied_pentagons`, and `parent_lineage_disagreements: {coarse_res: n_targets}`
— the count of targets for which `cell_to_parent` lineage and re-binning disagree. That
turns the non-nesting caveat from an argument into a measured number.

## Caveats

- **H3's hierarchy is not exactly nested.** H3 is aperture-7 and hexagons cannot tile
  hexagons, so `cell_to_parent` is exact on the *index* but is not a geometric container:
  `cell_to_parent(latlng_to_cell(p, fine), coarse) != latlng_to_cell(p, coarse)` for
  points near boundaries. HEALPix's `pix >> 2k` has no such gap (pinned by
  `test_degrade_matches_direct_ang2pix`). This is why the multi-scale diagnostic must
  re-bin rather than coarsen ids.
- **The paper draft asserts the HEALPix property.**
  `papers/cbg-benchmark-as-network-operator/paper-flow-draft-v2.md` line 275 says "which
  nesting makes a single pass" — true for HEALPix, not for H3. Line 258 pins
  `nside=128`. Flagged only; **no paper edits in this task.**
- **H3 cells are not equal-area.** Areas vary within a resolution (roughly up to ~2x),
  plus 12 pentagons per resolution (placed over ocean, so a non-issue for land targets but
  they break both equal area and the uniform-neighbour claim locally). Every
  straddle-probability statement stated against a single pitch acquires an error bar.
- **Neither grid fixes straddling.** Hexagons have boundaries too; a facility group
  spanning one is still quantized into two cells and two classes. Hexagons help only
  marginally at corners (three cells meet at a hex vertex versus four at a quad corner, so
  worst-case fragmentation of a tight cluster goes 4 -> 3). The prior finding stands: if
  grouping co-located facilities is a *requirement*, a grid is the wrong instrument and
  the radius-capped `clusters/` space is the right one.
- **Two live bugs to fix in passing.** `answer_space.py:275` calls
  `occupied_cell_hierarchy(...)` with no rung argument, so
  `meta["occupied_cells_by_nside"]` is always keyed 128/64/32/16 regardless of the grid
  built — at `nside=16` it reports counts for grids *finer* than the one used.
  And `spread_by_pix` (lines 176/184) is written but never read.
- **`venn.py` re-guesses its directory from the CLI** while `classify.py` derives it from
  the artifact. Not resolved here, just carried forward; `venn.py` reads the
  classification directory rather than the answer space, so it has nothing to read the
  grid from.
- **Removing `--nside` is a breaking CLI change.** Justified by there being no external
  caller (verified) and gitignored, regenerable outputs — but any local shell history or
  notebook invoking `--nside` will break.
- **`requirements.txt` is already a drifted partial mirror** of `[project].dependencies`
  (it lists neither `astropy-healpix`, `matplotlib-venn` nor `upsetplot`). Leaving it
  alone rather than partially fixing it.
