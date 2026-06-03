# Four-CBG-Variant Analysis Across VP Setups — Plan

## Background

We want a finalizable analysis of the **4 fundamental CBG variants** over a
controlled, deterministic benchmark: same global (Europe-dominant) target set,
deterministic VPs/targets per fold, deterministic seed (`42`) across all phases,
combos run over identical fold inputs. The point is to characterize how each
variant's error behaves across **6 vantage-point setups** and to attribute the
differences to VP geometry vs. LTD calibration.

The four variants are **two clean pairs** that each share a full geometry stack;
they differ only in the LTD (RTT→distance) model within a pair:

| combo_id            | label (display) | LTD              | shape   | MTL                     | CTR                  |
|---------------------|-----------------|------------------|---------|-------------------------|----------------------|
| `vanilla_cbg`       | Vanilla CBG     | `low_envelope`   | disk    | `spherical_circle`      | `boundary_vertex_mean` |
| `million_scale_cbg` | SOI CBG         | `speed_of_internet` | disk | `spherical_circle`      | `boundary_vertex_mean` |
| `octant_cbg_nofil`  | Octant CBG      | `bounded_spline` | annulus | `planar_annulus` (unweighted) | `monte_carlo_medoid` |
| `spotter_cbg_nofil` | Spotter CBG     | `normal_dist`    | annulus | `planar_annulus` (unweighted) | `monte_carlo_medoid` |

- **vanilla ↔ million_scale**: isolates LTD calibration (per-VP best-line vs. fixed 2/3·c). Same MTL+CTR.
- **octant ↔ spotter**: isolates LTD model (per-VP spline band vs. pooled global normal). Same MTL+CTR.
- **disk pair ↔ annulus pair**: confounded by LTD shape + MTL + CTR all at once — a *stack* contrast, not a single-component one.

## Context

**Six setups (configs / run stems):**

| slug    | config stem                     | region        |
|---------|---------------------------------|---------------|
| as7018  | `north_america_as7018_final`    | US            |
| as7922  | `north_america_as7922_final`    | US            |
| as3209  | `europe_as3209_final`           | Europe        |
| as3215  | `europe_as3215_final`           | Europe        |
| as16509 | `global_as16509_final`          | Global        |
| as31898 | `global_as31898_final`          | Global        |

**Data locations:**
- Per-fold combo outputs: `scripts/analysis/outputs/<stem>/ripe_atlas_asn_corpora/probes_to_anchors/fold_*/<combo_id>/targets.parquet`
  (TARGETS_SCHEMA — has `error_km`, `status`, `target_id`, `target_lat/lon`, `pred_lat/lon`, `ltd_predictions`, `seed`).
- Merged plots already land in `.../probes_to_anchors/merged/`.
- Materialized eval inputs: `scripts/benchmark/v2/inputs/ripe_atlas/probes_to_anchors/fold_*/eval_observations.parquet`
  (EVAL_OBSERVATIONS_SCHEMA — `target_id, target_lat, target_lon, vp_id, vp_lat, vp_lon, latency_ms`).

**Closest-VP distance** (the proposed x-axis / color variable): per target, the
minimum great-circle distance from the target to any of its VPs —
`min_vp haversine(target_lat/lon, vp_lat/lon)` over that target's rows in
`eval_observations.parquet`. This is *geometric* closest VP (the cross-continent
vs. home-continent signal), independent of RTT. Use
`scripts.libs.cbg.rtt_model.haversine_distance`.

**Reuse helpers:** `scripts/analysis/_v2_io.py` (`discover_combos`,
`group_combos_by_id`, `load_targets`, `palette`). Reference plot for deliverable 3:
`scripts/analysis/plot_per_target_sorted.py`.

**New workspace:** all code for this task lives under `scripts/paper/cbg_bench/`.
All artifacts (figures + their `.json` data dumps) are written under
`scripts/paper/cbg_bench/<run_id>/`. Every figure script is **config-driven** via
a typer CLI (matching the existing `scripts/analysis/plot_*.py` style): it takes
a `--config <file>` pointing at a YAML that declares the 6 setups (run stems /
input paths), the 4 variants, and the `run_id`; output paths are derived as
`scripts/paper/cbg_bench/<run_id>/...`.

## Goals

Finalizable deliverables (the 4 the user signed off on):

1. **6 per-setup error CDFs, SUCCESS-only**, one figure per setup, 4 variant
   curves, fallback-rate shown in the legend per variant. A **fixed, reusable
   color + label mapping** (a shared style module). Dump each figure's plotted
   data to a sibling `.json` with the same basename.
2. **Per-setup summary table**: variant × {p5, p25, p50, p75, p95, fallback%},
   rows ranked by **p50 ASC** by default. Dump the table data to a `.json` with
   the same basename.
3. **error_km vs. closest-VP-distance, faceted by variant** — adapt
   `plot_per_target_sorted.py` so the x-axis becomes closest-VP-distance ASC
   (instead of target_id / per-combo rank).
4. **Two paired scatters** for the clean within-pair contrasts — **one separate
   figure per pair** — points colored by closest-VP distance, with a y=x line:
   - Figure A: `error(vanilla_cbg)` vs `error(million_scale_cbg)`
   - Figure B: `error(octant_cbg_nofil)` vs `error(spotter_cbg_nofil)`

All four deliverables are produced by config-driven typer CLI scripts under
`scripts/paper/cbg_bench/`, writing to `scripts/paper/cbg_bench/<run_id>/`.

## Approach

- All code under **`scripts/paper/cbg_bench/`**. Each figure is a config-driven
  typer CLI (`--config <yaml>`) mirroring the `scripts/analysis/plot_*.py` style,
  writing figure + `.json` to `scripts/paper/cbg_bench/<run_id>/`. The config
  declares the 6 setups (run stems / input roots), the 4 variants, and `run_id`.
- Build a small **shared style module** (e.g. `scripts/paper/cbg_bench/_variant_style.py`)
  holding `VARIANT_LABELS` (the mapping below) and `VARIANT_COLORS`, so all four
  deliverables render consistently. Label mapping is fixed:
  `vanilla_cbg → "Vanilla CBG"`, `million_scale_cbg → "SOI CBG"`,
  `octant_cbg_nofil → "Octant CBG"`, `spotter_cbg_nofil → "Spotter CBG"`.
- Merge folds per (setup, combo) with `group_combos_by_id`, concatenating
  `targets.parquet` across `fold_*`. SUCCESS-only views filter `status == "SUCCESS"`;
  fallback% is computed over all rows (`FALLBACK / total`).
- Compute closest-VP distance once per (setup, target) from the fold's
  `eval_observations.parquet` and join onto the per-target table by `target_id`.
  (Folds partition targets, so the union over folds covers the target set.)
- Always dump the underlying plotted data to JSON alongside each artifact for
  reproducibility and downstream re-plotting.

## Caveats

- **Not a clean single-factor sweep.** Only within-pair contrasts isolate the
  LTD. The disk-vs-annulus headline is a 3-change stack effect — don't attribute
  it to one component. The `*_geo` / `*_hull` / weighted siblings in the configs
  exist to decompose it if needed (out of scope here, noted for discussion).
- **Fallback ≠ bad geolocation.** The two `_nofil` annulus variants can return
  empty intersections → `FALLBACK` (the unweighted `planar_annulus` ANDs every
  constraint; one over-tight annulus empties the region — cf. logged octant
  collapse on AS7018 NA). SUCCESS-only CDFs *plus* a visible fallback% keep the
  two failure modes separate. Decide whether to also dump an all-rows CDF.
- **Europe-dominant targets.** All-targets aggregates are biased toward the easy
  EU cluster; closest-VP-distance binning / faceting is the antidote.
- **closest-VP distance is geometric, not RTT-closest.** It is the intended
  cross-continent signal; do not confuse with the shortest-ping VP.
- Determinism is already established (seed `42`, SeedSequence per target) — see
  `notes/2026-06-02-randomness-determinism-assessment.md`. No re-derivation needed.

## To discuss (not in the finalizable scope)

1. **"Close but all on one side" vs "surrounded":** large residual error even
   with a near VP. Quantify via VP **azimuthal coverage / largest azimuth gap**
   around the target, and relate it to error after controlling for closest-VP
   distance.
2. **Constraints that determine the final intersection:** for the VPs whose
   circles/annuli bound the chosen region, what are the target→VP RTTs within a
   τ-ms band (the "tightest few constraints")? Needs surfacing which LTD
   predictions actually touch the intersection boundary.
