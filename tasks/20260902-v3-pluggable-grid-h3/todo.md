# v3 Pluggable Answer-Space Grid (HEALPix + H3) — Todo

## Phase 0: Setup & Baseline
- [x] Record the pre-change invariant from `outputs/analysis/v3/as01-260728-260802/target-cls-accuracy/nside-128/topn_accuracy.csv`: `vanilla_cbg` `accuracy_top1=0.411`, `fallback_rate=0.2657`, `n_solved=293`, `n_fallback=106`; `octant_cbg_hull` `accuracy_top1=0.687`
- [x] Baseline `python -m pytest scripts/analysis/v3/tests/ -q` (60 passed as of 2026-09-02)
- [x] Add `"h3 (>=4.5.0,<5.0.0)"` to `[project].dependencies` in `pyproject.toml` beside `astropy-healpix`; `poetry lock`; install into `.venv`
- [x] Smoke-verify every h3 v4 call before relying on it — `latlng_to_cell`, `cell_to_boundary` (confirm it returns `(lat, lng)`), `cell_to_parent`, `average_hexagon_area`, `average_hexagon_edge_length`, `cell_area`, `get_num_cells`, `int_to_str`/`str_to_int`, `is_pentagon`

## Phase 1: The interface
- [x] New `modules/grid.py`: `Grid` ABC (`name`, `resolution_arg`, `DEFAULT_RESOLUTION`, `HIERARCHY`, `validate_resolution`, `cell_ids`, `cell_area_km2`, `nominal_cell_km`, `n_cells`, `cell_boundaries`, `coerce_cell_ids`, `describe`)
- [x] Concrete `Grid.coarsening_ladder(resolution)` — never returns a rung finer than the one requested
- [x] Concrete `Grid.occupied_cell_hierarchy(lat, lon, resolutions)` that **re-bins from coordinates** at each rung
- [x] `DEFAULT_GRID = "h3"`, `GRID_NAMES`, and `get_grid(name)` importing the implementation lazily inside the function body (avoids the `grid` <-> `healpix` cycle)
- [x] Move `spherical_centroid` from `healpix.py` into `answer_space.py` beside `_unit_vectors`; update the 2 call sites (`answer_space.py:178`, `tests/test_answer_space.py:46`)

## Phase 2: The two implementations
- [x] `HealpixGrid(Grid)` at the bottom of `modules/healpix.py`, delegating to the existing functions; leave those functions untouched
- [x] `HealpixGrid.cell_boundaries` wrapping `boundaries_lonlat` — unwrap `.to_value("deg")`, wrap lon into [-180, 180], split into per-cell rings; keep `degrade` as a HEALPix-only extra documenting the nesting property
- [x] New `modules/h3grid.py` with `H3Grid(Grid)`, `import h3` lazily inside methods; `DEFAULT_RESOLUTION = 4`, `HIERARCHY = (5, 4, 3, 2)`, `SUPPORTED_RESOLUTIONS = 2..5`, `resolution_arg = "res"`
- [x] `H3Grid.nominal_cell_km = average_hexagon_edge_length(res) * sqrt(3)` (centre-to-centre, not `sqrt(area)`) — document why
- [x] `H3Grid.cell_boundaries` — swap `cell_to_boundary`'s `(lat, lng)` to `(lon, lat)`, handle the antimeridian, return ragged rings
- [x] `H3Grid.coerce_cell_ids` / cell ids stored as canonical hex strings via `int_to_str`
- [x] `H3Grid.describe()` extras: `avg_edge_km`, `cell_area_km2_min`/`_max` over occupied cells, `n_occupied_pentagons`, `parent_lineage_disagreements`

## Phase 3: Schema and paths
- [x] `paths.py`: replace `nside_slug` with `grid_slug(grid, resolution)`; make `grid` and `resolution` required keyword-only on `answer_space_dir` and `cls_accuracy_dir`
- [x] `answer_space.py`: `build_answer_space(targets, *, grid, resolution, source_label)`; write `grid_scheme` / `grid_resolution` / `cell_id` into `seeds.csv` and `cell_id` into `assignments.csv`
- [x] `answer_space.py`: `meta["grid"]` from `grid.describe(resolution)`; `occupied_cells_by_nside` -> `occupied_cells_by_resolution`, fed by `coarsening_ladder`
- [x] `sweep_row`: leading `nside` key -> `grid` + `resolution`; `SWEEP_CSV` -> `grid_sweep.<grid>.csv`
- [x] Delete the dead `spread_by_pix` dict (`answer_space.py` lines 176/184)
- [x] `load_answer_space`: coerce `cell_id` via the scheme named in `grid_scheme`

## Phase 4: Consumers and CLI
- [x] Swap `--nside` for `--grid` / `--resolution` / `--sweep` in all four command modules; help text and echoes use `grid.resolution_arg` so HEALPix reads `nside=128` and H3 reads `res=4`
- [x] `classify.py`: read `grid_scheme` + `grid_resolution` from the loaded `seeds.csv` (not from the CLI) to derive `cls_accuracy_dir`; `manifest.json` `nside` key -> `grid` + `resolution`
- [x] `map_answer_space.py`: drop the direct `from astropy_healpix import HEALPix` and the astropy unit unwrapping; draw ragged rings from `grid.cell_boundaries`; grid from `get_grid(seeds["grid_scheme"])`; title/caption from `grid.name` / `resolution_arg` / `nominal_cell_km`
- [x] Confirm `cli.py` needs no change (`grid.py`, `healpix.py`, `h3grid.py` expose no `register`)

## Phase 5: Tests
- [x] New `tests/test_grid.py` parametrized over both grids: registry resolution + unknown-name rejection; `DEFAULT_GRID == "h3"`; `validate_resolution` range; occupied-cell counts non-increasing down `HIERARCHY`; `coarsening_ladder` excludes finer rungs; `cell_ids` deterministic and order-independent; `coerce_cell_ids` round-trips through CSV; `cell_boundaries` yields one closed ring per cell; `nominal_cell_km`/`cell_area_km2` match the plan's scale table
- [x] Test H3 `parent_lineage_disagreements` on a deliberately boundary-straddling fixture
- [x] `tests/test_healpix.py`: keep the HEALPix-specific tests; move the 2 `spherical_centroid` tests to `test_answer_space.py`
- [x] `tests/test_answer_space.py`: parametrize the 10 grid-agnostic tests over both grids; update the 2 coupled ones (`nside=` kwarg, `meta["grid"]["nside"]`)
- [x] `tests/test_io_and_venn.py`: rewrite the 2 path tests for `grid_slug`, asserting `h3-4` / `healpix-128` and that the grids cannot collide
- [x] `tests/test_map_answer_space.py`: parametrize the 2 render tests over both grids

## Phase 6: Verification
- [x] `python -m pytest scripts/analysis/v3/tests/ -q` green
- [x] `rm -rf outputs/analysis/v3`; regenerate **h3 res 4 only** (build-answer-space, classify, plot-venn at top-1 and top-3, plot-answer-space) for `--all-runs`
- [x] Regenerate **healpix nside 128 only** (build-answer-space, classify, plot-answer-space) as the no-op regression guard
- [x] **Invariant**: as01 `vanilla_cbg` at `healpix-128` still `accuracy_top1=0.411`, `fallback_rate=0.2657`, `n_solved=293`, `n_fallback=106`; `octant_cbg_hull` still `0.687`
- [x] `--help` on all four commands shows `--grid` defaulting to `h3`, `--resolution` to 4, and no `--nside`
- [x] `h3-4/` and `healpix-128/` coexist under both `target-answer-space/` and `target-cls-accuracy/`; `grep -rl healpix_ outputs/analysis/v3/` returns nothing
- [x] Record K and the accuracy table for as01 and as7018 at `h3-4` vs `healpix-128` (prediction was wrong — see report: as7018 gave FEWER classes and HIGHER accuracy)
- [x] `meta["occupied_cells_by_resolution"]` contains only rungs at or coarser than the one built
- [x] Record H3 `parent_lineage_disagreements` (zero is legitimate on a small target set; the point is that it is measured)
- [x] Eyeball the H3 maps for hexagons with no gaps or self-intersections (confirms the ragged-ring path and the (lat,lng) -> (lon,lat) swap)
- [x] Update `scripts/analysis/v3/README.md` (module table, pipeline commands, output tree, a "Choosing a grid" section with the scale table and the non-nesting caveat) and `SCHEMA.md:122`

## Phase 7: Paper draft
- [x] `paper-flow-draft-v2.md` §7.3: swap the grid definition to H3 res 4 (288,122 cells, 1,770 km² avg, 45 km centre-to-centre), keeping HEALPix `nside=128` named as the exactly-equal-area/exactly-nested alternative
- [x] Add the operator rationale (hexagon adjacency, RF-planning convention, native in ClickHouse/BigQuery/Snowflake/Spark) and disclose both costs given up: 1.33x occupied-cell area spread, and pitch = edge x sqrt(3) rather than sqrt(area)
- [x] Restate the straddle probability at w=45 km (21% at 5 km, 39% at 10 km; was 19%/35% at w=51)
- [x] **Remove the false claim** at the metric list: "which nesting makes a single pass" is true of HEALPix and not of H3; replace with the re-binning rule plus the measured lineage-vs-rebin disagreement rate
- [x] Add the alignment-not-resolution finding with its three consequences
- [x] Fix the §1 contribution line ("equal-area 51 km grid" -> "45 km hexagonal grid")
- [x] Verify every claim against the runs before writing it; correct the dataset attributions
