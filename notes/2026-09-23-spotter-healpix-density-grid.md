# Spotter's density grid: H3 → HEALPix

`GaussianDensityMTL` evaluates Spotter Eq. (2) over a discretised globe and
`density_argmax` takes the MAP cell. The paper discretises with HTM (§V-A-2); we
substituted H3 because the v3 analysis tree was keyed on it. That reason expired
when v4 moved its scoring grid to HEALPix, and the H3 grid turned out to carry a
defect the substitution never accounted for.

**Bottom line: the grid was wrong in a way that mattered structurally and almost
not at all numerically.** The descent is now correct by construction and 2.6×
faster, the estimator is 6–15 km worse because the cells are 13% larger, and the
classification outcome is unchanged. Spotter's accuracy problem is not the grid.

## Why H3 was wrong

H3 is aperture-7. Hexagons cannot tile hexagons, so a parent's six outer
children straddle its boundary and `cell_to_children` is exact on the *index*
without being a geometric container. Measured over 400,000 random points:

> **7.12% of points land in an H3 res-3 cell that is not a child of their own
> res-2 cell.**

The coarse-to-fine descent therefore had holes at every level, and
`neighbor_ring=1` was *patching* that rather than providing the safety margin it
is documented as. HEALPix is aperture-4 with the four children exactly tiling
the parent, so the candidate set is now exactly the refinement of what was
carried.

HEALPix over HTM — also aperture-4, and the paper's own choice — because HEALPix
cells are exactly equal-area where HTM's vary by **110%** at every level
(measured: max/min 2.068 at L3, 2.096 at L4, 2.105 at L6, against H3 res-4's
1.326 and HEALPix's 1.000). Equal area is what lets a cell count convert to an
area without qualification, and what makes the argmax's quantisation floor the
same everywhere instead of position-dependent. HTM would also have meant a new
dependency; `astropy_healpix` was already one.

## The basin-miss experiment, this time committed

`top_k=8, neighbor_ring=1` was chosen in `9e9df7d` on exactly this measurement,
but the script was never committed — only its result table, in the commit
message. So when the grid changed the numbers could not be re-derived, which is
why v4's README had to list this migration as blocked on an experiment nobody
could run. It is now `scripts/benchmark/v2/cli.py mtl-basin-miss`.

It compares each pruned setting's argmax against `GaussianDensityMTL` **itself**
in its documented no-pruning mode (`coarse_resolution == resolution`), so both
sides exercise the shipped path and differ only in the pruning. Constraints are
synthetic — real µ/σ are not recoverable from the benchmark's output, see below
— but built on the run's real VP geometry and real target positions, which is
what decides whether the surface is multi-modal. The inflation is Pareto, not
Gaussian: RTT error is one-sided, and it is the long tail that creates the far
spurious modes pruning can lose.

50 targets per dataset, misses counted at > 50 km of argmax displacement:

| top_k | ring | H3 2→4 as01/02/03 | HEALPix 16→128 as01/02/03 |
|---|---|---|---|
| 1 | 0 | 0/50, 0/50, **1/50** | **8/50, 5/50, 7/50** |
| 8 | 0 | 0/50, 0/50, 0/50 | 0/50, 0/50, 0/50 |
| 1 | 1 | 0/50, 0/50, 0/50 | 0/50, 0/50, 0/50 |
| **8** | **1** | 0/50, 0/50, 0/50 | **0/50, 0/50, 0/50** ← shipped |
| 64 | 1 | 0/50, 0/50, 0/50 | 0/50, 0/50, 0/50 |

Worst displacement at the starved setting: 46–85 km under H3, 55–131 km under
HEALPix. So the concern that made re-running this a precondition was real — the
third descent level does cost pruning headroom — and the shipped setting
survives it with margin on every dataset.

## What it cost and what it bought

Three mesh datasets, five folds each, `spotter_cbg` only (the rename invalidated
exactly that target, so Snakemake rebuilt 5 jobs per config rather than 30):

| | as01 | as02 | as03 |
|---|---|---|---|
| `error_km` p50, H3 res-4 → nside 128 | 198.1 → 213.5 (**+15.4**) | 338.0 → 344.3 (**+6.3**) | 235.2 → 249.9 (**+14.7**) |
| `error_km` p90 | 2355.7 → 2363.6 | 994.0 → 1003.3 | 823.7 → 791.1 |
| `mtl_ms` p50 | 117.9 → 46.9 (**2.5×**) | 126.9 → 46.6 (**2.7×**) | 129.0 → 50.1 (**2.6×**) |
| `rss_after_fit` MB | 228.5 → 248.8 (**+20.3**) | 230.2 → 250.9 (**+20.7**) | 235.2 → 255.2 (**+20.0**) |

The error regression is the grid pitch: nside 128 is 50.9 km against H3 res-4's
45.2 km, 13% coarser, and the argmax cannot beat its own quantisation. nside 256
(25.4 km) is the lever if it ever matters, at 4× the global pass. **It is not
evidence against the migration**, and nside 256 would not fix the real problem —
see the next section.

The speedup is `astropy_healpix` being vectorised where `h3` needs a Python
loop: all 196,608 nside-128 cell centres build in 27 ms against h3's 346 ms for
288,122 res-4 cells, and the coarse pass is 3,072 cells against H3 res-2's
5,882. Pruning remains mandatory regardless — a full global nside-128 pass is
1,085 ms per target against 27.5 ms pruned.

The **+20 MB is the `astropy_healpix` import**, not a leak. It lands between
`rss_after_inputs` and `measure_block("fit")`, so every per-stage column is
unaffected (they are deltas against a baseline sampled inside the block); only
the absolute marks `rss_after_fit_bytes` and `run_peak_rss_bytes` carry it, and
only for Spotter. Do not "fix" it by importing astropy at module scope —
`framework/v2/__init__.py` imports the MTL, so every combo would pay it and no
previously collected run would stay comparable.

## The classification outcome did not move

Scored under v4's laddered containment metric at nside 128, with the H3 arm
preserved as `spotter_h3_cbg` and scored alongside:

| dataset | arm | ring0 | ring1 | ring2 | retired nearest-seed |
|---|---|---|---|---|---|
| as01 | HEALPix | 0.000 | 0.193 | 0.341 | 0.887 |
| as01 | H3 | 0.000 | 0.163 | 0.296 | 0.862 |
| as02 | HEALPix | 0.000 | 0.049 | 0.109 | 0.510 |
| as02 | H3 | 0.000 | 0.049 | 0.175 | 0.510 |
| as03 | HEALPix | 0.000 | 0.092 | 0.323 | 0.491 |
| as03 | H3 | 0.000 | 0.042 | 0.312 | 0.493 |

**ring0 is exactly 0.000 for both arms on all three datasets.** Spotter's
cell-centre estimate never once lands in the truth's own cell, and that is
invariant to which grid quantises it. The ring1/ring2 differences go in both
directions across datasets, so they are noise at this sample size, not a grid
effect. Anyone reading the `Spotter (H3)` bar in the outcome-bar figures should
read it as confirmation that the grid was not the problem.

## Two things that were removed, and why

**`credible_mass` is gone**, along with `_credible_region`. It could not mean
what it said: the mass is normalised over the retained cells rather than the
globe, so pruning turned "the 95% region" into a statement about a
neighbourhood. Worse, `credible_mass = 1.0` did not even mean "all of them" —
`probabilities()` max-shifts before exponentiating, so with a sharp posterior
(the realistic case; summed z² runs to thousands over ~130 VPs) the partial sum
reaches 1.0 in float64 after a handful of cells. Measured on a 104-cell field: a
wide posterior kept 104, a sharp one kept **1**. And nothing read it —
`density_argmax` takes the maximum of `log_density` directly, and it is the only
CTR composed with this MTL. `intersection` is now the retained field itself,
whose area is exactly `len(cells) × pixel_area_km2`.

**H3 is gone from the framework entirely.** No grid adapter and no `Grid` ABC —
that polymorphism is what broke `main` at `d0cc6f6`. `h3` remains a project
dependency because `scripts/analysis/v3/modules/h3grid.py` still uses it.

`grid` is a **required** kwarg with no default. The H3-era combos stored
`resolution: 4` and no `grid`, and 4 is a legal nside, so a default would let a
stale payload replay as a 192-cell globe instead of failing.

## Two pre-existing holes this surfaced

**`eval-bench-results` cannot evaluate a density combo, and never could.** It
replays each combo's MTL from `mtl_participants`, which stores
`echoed_upper_km`/`echoed_lower_km` and **no `mu_km`/`sigma_km`** — verified:
`mu_km|sigma_km` appears 0 times in `benchmark/v2/schema.py`. So the
reconstructed `Distance` has `has_distribution == False` and the density MTL
returns `INSUFFICIENT_DATA` for every target: `recompute_matches` comes back
**0.0** over 399 targets and `truth_in_region` is null. This is the same schema
limit `map_mtl.py` documents and predates the grid change entirely. Fixing it
needs two nullable float64 columns on `_MTL_PARTICIPANT_FIELD`; until then, treat
that command's MTL-derived columns as inapplicable to Spotter rather than as
measurements.

That command also used to **lose the whole run** over one such combo: the MTL
was constructed inside the per-combo loop with no guard, and combos are
discovered by globbing, so a single stale `mtl_kwargs` payload raised and
nothing was produced — including every combo sorting after it. Now fixed; the
backed-up `spotter_h3_cbg` (whose `run.json` still carries `credible_mass`) is
warned about and skipped, and `vanilla_cbg` still produces its row.

**`spotter_cbg` now names two different compositions across configs.** The as0*
configs have the HEALPix density arm; `as7018-ripe-mesh{,-reciprocal}.yaml` still
have `normal_dist → planar_annulus_weighted → monte_carlo_medoid`, i.e. the
Octant-geometry hybrid that the as0* configs renamed to `spotter_hybrid_cbg`.
That is pre-existing and was left alone, but it means 30 of the 45 on-disk
`spotter_cbg` directories are the hybrid, not the density arm — which is why
`rename-combo` gates on `run.json["mtl"]` rather than on the path, and why its
derived-artifact sweep is scoped to the runs whose combo directory actually
moved. Unscoped it would have relabelled the hybrid's scorings as the density
arm's: 27 artifacts instead of the correct 15.

## Where the pieces are

| | |
|---|---|
| grid primitives | `scripts/libs/healpix/grid.py` (shared; the framework may not import `scripts.analysis`) |
| the MTL | `scripts/framework/v2/mtl/gaussian_density.py` |
| pruning validation | `scripts/benchmark/v2/cli.py mtl-basin-miss` |
| preserving results | `scripts/benchmark/v2/cli.py rename-combo` |
| the map's region | `map_mtl.argmax_cell_regions` — re-bins the stored prediction, no replay |
