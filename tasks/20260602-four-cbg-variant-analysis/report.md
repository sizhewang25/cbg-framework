# Four-CBG-Variant Analysis Across VP Setups — Report

**Status**: Complete (finalizable deliverables); two analysis questions parked for discussion
**Created**: 2026-06-02
**Last Updated**: 2026-06-02

## Summary

Built a config-driven figure suite under `scripts/paper/cbg_bench/` comparing the
4 fundamental CBG variants (Vanilla CBG, SOI CBG, Octant CBG, Spotter CBG) over
the 6 `*_final` VP setups (2 US, 2 Europe, 2 Global), against the fixed
Europe-dominant target set with deterministic folds/seed. All four deliverables
are implemented, run end-to-end from one config, and emit a figure **plus a
sibling `.json`** for every artifact.

Implementation was parallelized: a shared foundation (`_variant_style.py`,
`_io.py`, `config/four_variants.yaml`) was committed first, then the four figure
scripts were built concurrently by subagents in isolated git worktrees under
`trees/` and merged back conflict-free (disjoint files).

## Deliverables (all in `scripts/paper/cbg_bench/`)

| # | Script | Output (under `four_variants/`) |
|---|---|---|
| 1 | `plot_error_cdf.py` | `cdf_<slug>.png/.json` ×6 — SUCCESS-only error CDF, fallback% in legend |
| 2 | `summary_table.py` | `table_<slug>.png/.json` ×6 + `summary_table.md` — {p5,p25,p50,p75,p95,fallback%}, ranked p50 ASC |
| 3 | `plot_error_vs_vp_dist.py` | `error_vs_vpdist_<slug>.png/.json` ×6 — error vs closest-VP-distance, faceted by variant |
| 4 | `plot_paired_scatter.py` | `paired_<vx>__<vy>.png/.json` ×2 — within-pair error scatters, colored by closest-VP distance |

Totals: 20 PNG + 20 JSON + 1 MD. Every PNG has a sibling JSON.

## Findings

The numbers already corroborate the design hypotheses (geometry sets the floor;
LTD calibration sets how close you get; `_nofil` annulus variants are
collapse-prone):

- **Closest-VP distance tracks the setup.** US setups have large median
  closest-VP distance (~5.8–6.3k km — cross-continent to the EU-dominant
  targets); Global setups are small (~0.36–0.39k km). EU setups in between
  (~1.0–1.3k km).
- **SOI CBG never collapses but is loosest at the median.** `million_scale_cbg`
  fallback ≈ 0% everywhere; its p50 is the largest on the cross-continent US
  setups (as7018 p50 6073 km vs Octant/Spotter ~3070 km).
- **`_nofil` annulus variants collapse on Global.** Spotter CBG fallback hits
  74–81% on the two Global setups (as16509: 808/1000-scale → 137/713 SUCCESS);
  Octant CBG 30–34%. On US/EU they are tighter at the median than the disk pair
  but carry 5–17% fallback.
- **Pairing exposes the collapse directly.** The vanilla/SOI paired scatter keeps
  4135/4278 targets (143 dropped); the octant/spotter scatter keeps only 2712
  (1566 dropped where one `_nofil` variant fell back).

## Verification

- **Cross-deliverable consistency:** SUCCESS counts per (setup × variant) agree
  exactly across the CDF, table, and facet JSONs — 0 mismatches (shared `_io`).
- **Fallback% vs `run.json`:** as7018/octant_cbg_nofil = 83/713 = 11.64%, matches
  the table to 2 dp.
- **Closest-VP by hand:** as16509 target 101.53.31.6 → 1097.00 km by direct
  haversine == `_io` value.
- **JSON round-trip:** CDF `error_km_sorted` arrays reproduce the table's
  p5/p25/p50/p75/p95 exactly for all 4 variants.
- **Visual:** CDF and paired-scatter figures render with the fixed color/label
  mapping (Vanilla=blue, SOI=orange, Octant=green, Spotter=red).

## Study 1 — closest-VP distance impact (2026-06-03)

`scripts/paper/cbg_bench/plot_study1_distance.py` — error_km vs closest-VP
distance on **log–log, equal-scale axes** (`y=x` at a true 45°), threshold lines
on both axes, SOI as the calibration-free / zero-fallback geometry baseline. Per
setup: `study1_soi_<slug>.png` (hero) + `study1_facet_<slug>.png` (2×2) +
`study1_<slug>.json` (SUCCESS arrays + binned p50/p90, computed not plotted).
Shared axes `[1, 20000]` km computed once across all setups. New reusable helper
`_io.binned_percentiles`.

**Findings:**
- **Ramp with saturation, not a cliff.** SOI binned p50 on as7018 (US):
  144→334→925→1380→1568 km across 0–1150 km, a sparse middle, then a high plateau
  (~5400→6800 km) for the cross-Atlantic EU cluster sitting at ~5800–6600 km from
  the US VPs. Error saturates near the closest-VP distance once the target is far.
- **SOI rides the `y=x` diagonal.** ~40% of SOI targets fall within 2% of
  `error == closest-VP distance` (vanilla 9%, octant ~1% — SOI-specific). Its
  loose `2/3·c` radii mean that for a large fraction of targets it cannot localize
  any tighter than "near your closest VP." Another ~34% beat that (error <
  closest-VP distance, below the diagonal). This is the geometry-baseline limit
  the other variants' calibration is measured against.

**Verification:** `n_success` per (setup × variant) matches the deliverable-3
JSON exactly (shared `_io`); SOI fallback as7018 = 0/713, as16509 = 3/713 (the
3/713 reconciles the original screenshot — it was the Global as16509 setup).

**Caveat (2026-06-04):** the macro distance story (ramp + saturation; far targets
unrecoverable) is robust to the geometry stack, but the `diagonal_split` /
win-rate is **strongly MTL+CTR-dependent** — swapping only the stack (`_geo`)
moves p50 20–55% and flips the win-rate (e.g. vanilla as7018 wins 67% → 3%), and
the best stack flips by regime (boundary_vertex_mean far/one-sided, geometric_
centroid near/surrounded). The diagonal split must therefore be read per-stack,
and the MTL+CTR impact study (parked in todo) gates Study 2.

## Conclusions

The four finalizable deliverables plus Study 1 are done, verified, and
reproducible from a single config
(`scripts/paper/cbg_bench/config/four_variants.yaml`). Re-running any figure
regenerates both the PNG and its JSON. Remaining parked questions: Study 2
(azimuthal coverage "one side vs surrounded" — buildable from `eval_observations`
coords, planned) and the Global-act tightest-constraint RTTs (needs MTL internals
not yet surfaced in the benchmark outputs).
