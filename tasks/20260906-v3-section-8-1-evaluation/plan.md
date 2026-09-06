# §8.1 Evaluation Scripts (v3) — Plan

## Background

`papers/cbg-benchmark-as-network-operator/paper-flow-draft-v2.md` §8 is the
paper's evaluation section and is currently prose plus figure placeholders. §8.1
(RQ1 — *what accuracy is achievable under operator-realistic conditions*) is the
largest unbuilt block, and §8.2/§8.3 both read against it.

The v3 analysis layer already produces the answer space (`build-answer-space`),
the dataset geometry (`build-bipartite-graph`), the per-method classification
(`classify`), the set-overlap figures (`plot-venn`) and the cost/accuracy Pareto
(`plot-pareto`). What is missing is the **stratification layer**: the per-target
labels that say *why* a target was or was not geolocatable, and the tables that
cross those labels against every method's correctness.

The design was settled in conversation on 2026-09-06. This plan records the
decisions so they are not re-litigated.

## Context

### Inputs

| input | path | role |
| --- | --- | --- |
| answer space | `outputs/analysis/v3/<run>/target-answer-space/<grid>-<res>/seeds.csv` | seeds, `margin_km`, `nearest_seed_km` |
| | ` … /assignments.csv` | target → seed |
| | ` … /seed_mesh_km.csv` | K×K seed distances |
| classification | `outputs/analysis/v3/<run>/target-cls-accuracy/<grid>-<res>/<method>_seed_distances.parquet` | per-target correctness + drift |
| | ` … /topn_accuracy.csv` | per-method rollup |
| VP roster | `outputs/benchmark/v2/<run>/generic_csv/<setup>/vps.csv` | VP coordinates |
| edges | canonical CSV named in `eval_source/<basename>_eval_stats.json` `csv` key | measured (VP, target, rtt) |
| carried columns | `eval_source/<basename>_eval_per_target.csv` | `closest_vp_km`, `shortest_ping_vp_*`, `min_inflation` |

### Runs in scope

`as01-260728-260802`, `as02-260728-260802`, `as03-260728-260802` (operator,
134 VPs each, 399/412/458 targets) and `as7018_us_test01` (public RIPE, 53 VPs,
78 targets). Grid: **h3 res 4** (`healpix-128` stays supported, not reported).

### Methods in scope

The six shared by every run: `shortest_ping`, `million_scale_cbg`,
`vanilla_cbg`, `octant_cbg_hull`, `octant_cbg_spl` (`octant_cbg` on as7018),
`spotter_cbg`. as7018's 11 ablation arms are RQ3 material, excluded here.

## Goals

1. **`tg_seed` rename across the whole v3 layer** — `truth_seed_id` →
   `tg_seed_id`, `truth_seed_rank` → `tg_seed_rank`, `error_to_truth_seed_km` →
   `error_to_tg_seed_km`. Stops at v3's boundary; v2's `truth_centroid_*` and
   `truth_in_region` are left alone.
2. **`build-proximity`** → `target-proximity/<grid>-<res>/target_labels.csv`,
   the per-target proximity ladder every §8.1 table strata against.
3. **`breakdown-accuracy`** — per-method accuracy crossed with each proximity
   flag, at top-1 and top-3, with cell counts and separation scalars.
4. **`confusion-density`** — misclassification against answer-space local
   density (the truth seed's `nearest_seed_km`), plus where wrong answers land
   in the seed-to-seed ranking.
5. **`table-accuracy`** — the §8.1 headline: per-run × per-method top-1/top-3,
   fallback rate, error p50/p90. Operator runs in one table, public in its own.

## Approach

### The proximity diamond

Four boolean flags, **top-1 context**, over **measured VPs only**:

```
        has_proximate_vp                  (∃ measured VP whose argmin seed is tg_seed)
       /                \
has_discriminative_vp    has_proximate_sping_vp
       \                /
        has_discriminative_sping_vp
```

- **argmin axis** (`has_proximate_*`): a VP is proximate when `tg_seed` is its
  nearest seed — the classifier's own rule. `has_proximate_sping_vp` is
  *identically* "Shortest-Ping correct at top-1", by construction.
- **half-gap axis** (`has_discriminative_*`): a VP is discriminative when
  `d(VP, tg_seed) < tg_seed_margin_km`. This is the grid analogue of
  `eval_source`'s `cell_gap/2` rule, measured **to the seed** — which is what
  makes it a strict guarantee (`d(VP,S) < margin ⟹ VP's argmin seed is S`) and
  what makes the diamond's implications hold.

It is a **diamond, not a chain**: `4 ⟹ 2 ⟹ 1` and `4 ⟹ 3 ⟹ 1`, but 2 and 3 are
incomparable. Geography-strength and routing-strength are two axes meeting at
the top.

§8.2's three-way taxonomy is the diamond's **argmin chain** and needs no
separate column:

| §8.2 term | flags |
| --- | --- |
| structural failure | `¬has_proximate_vp` |
| selection failure (the CBG opportunity) | `has_proximate_vp ∧ ¬has_proximate_sping_vp` |
| baseline already correct | `has_proximate_sping_vp` |

### `target_labels.csv` schema

One row per target. Observed-only, method-free, fold-free, N-free filename.

| group | columns |
| --- | --- |
| identity | `target_id`, `tg_seed_id`, `tg_seed_margin_km` |
| left chain (existence over measured VPs) | `tg_seed_best_rank`, `tg_seed_best_rank_vp_id`, `tg_seed_nearest_vp_km`, `tg_seed_nearest_vp_id` |
| right chain (the one designated VP) | `sping_vp_id`, `sping_vp_tg_seed_rank`, `sping_vp_to_tg_seed_km` |
| flags (top-1) | `has_proximate_vp`, `has_discriminative_vp`, `has_proximate_sping_vp`, `has_discriminative_sping_vp` |
| context, VP→target, not used by the flags | `n_measured_vps`, `closest_vp_to_tg_km`, `sping_vp_to_tg_km`, `min_inflation` |

`meta.json`: per-flag base rates, zero-variance warnings, and the four-flag
cross-tab so the diamond's occupancy is visible in one place.

### Naming conventions (adopted, apply to all future v3 columns)

1. **Every distance names both endpoints**: `<from>_to_<to>_km`. Carried
   `eval_source` columns are renamed on the way in (`closest_vp_km` →
   `closest_vp_to_tg_km`) and their source names recorded in SCHEMA.md.
   `tg_seed_margin_km` is exempt — it is half a distance, not a point-to-point
   one, and matches `seeds.csv`'s `margin_km`.
2. **Prefix by the diamond's chain, not by "min"**: `tg_seed_*` for the
   existence axis (a property of the seed), `sping_vp_*` for the routing axis
   (a property of the selected VP). The two axes minimize *different* things and
   in general select *different VPs*, which is why `min_vp_*` was rejected.
3. **`sping_` throughout v3**, not `shortest_ping_`. v2's `eval_source` spelling
   is legacy and is not a compatibility constraint.

### Why `target-proximity/` is its own directory

Not `target-answer-space/`: that tree is a pure function of (target
coordinates, grid), which is what lets `classify --answer-space` re-score under
a different quantization. The labels depend on the VP roster **and on RTT**
(identifying the shortest-ping VP is a min over RTT), so writing them there
would make the answer space campaign-dependent and leave stale labels behind a
re-score.

Not `bipartite-graph/`: that module's contract is explicitly RTT-free.

The labels are a join of three inputs — answer space × VP roster × RTT edges —
and no existing directory owns all three.

## Caveats

- **`has_proximate_sping_vp` × `shortest_ping` × top-1 is tautological.**
  Perfect separation there is a pipeline self-check, not a finding, and must be
  annotated as such in the output. The same cell at top-3 is informative.
- **Top-1 flags stratify top-3 accuracy, deliberately.** The flags describe the
  dataset's geometry; top-N tolerance is a property of the scoring. Crossing
  them is the point of the table, not an inconsistency.
- **Flags may be zero-variance on the operator runs.** `no_proximity_share` is
  **0.0 on as01** in the existing cluster-keyed data — every target has a
  discriminative VP. Whether that survives under the grid partition is untested.
  A constant flag must report as "no variance" and never as "no effect", so
  cell counts ship with every rate.
- **Latent (whole-roster) columns are deliberately out.** The latent/observed
  pairing already has an owner in `bipartite.py`
  (`measured_nearest_vp_ratio_per_target`); duplicating it here would put the
  same fact in two directories free to disagree. When the passively-collected
  weighted set lands, "resolvable but unmeasured" will need answering and that
  ratio is only a **proxy** — it is keyed to nearest-VP-by-target-distance, not
  to seed distance or rank. Add the seed-keyed latent column *then*, do not
  stretch the ratio.
- **Two VPs, not one, on the left chain.** `tg_seed_best_rank_vp_id` and
  `tg_seed_nearest_vp_id` are different VPs in general: a distant VP in an empty
  region can rank `tg_seed` first while a nearer VP in a dense region does not.
  The diamond survives regardless (`d < margin ⟹ rank 0` for that same VP).
- **`closest_vp_id` never joins across layers.** `eval_source` uses a BallTree
  and v3 uses `argmin`; co-located VPs are the normal case (as01 VP
  nearest-neighbour p50 is 0.0 km) so they break ties differently — 259 of 399
  agree on as01 while every *distance* matches. Compare distances only.
- **`setup` gates nothing.** `anchors_to_probes` on as01–03 is a placeholder
  with no meaning; do not inherit `plot-pareto`'s mixed-setup refusal here.
  Operator and public runs are kept in separate tables for a *substantive*
  reason (§7.3 declines the head-to-head), not a schema one.
- **The rename invalidates existing parquet.** `classify` must be re-run on all
  4 runs × both grids after the rename. Cheap, but the ordering matters: rename
  first, re-run, then build on top.

## Out of scope (recorded, not forgotten)

- **Traffic-weighted views.** `has_weight: false` on all four runs, so §8.1's
  mesh-vs-traffic-weighted comparison — the second and third subsections,
  including the "Shortest-Ping wins under good peering" story — has no data
  behind it. Build weighted-ready; produce nothing weighted.
- **The VP&TG topology-paired dataset** (§8.1's fourth table row, §7.3's open
  TODO). Subsampling operator VPs to match as7018 changes the LTD fit, so it is
  a `scripts/benchmark/v2/` re-run, not an analysis-layer script. Needs VP and
  target sets co-curated. **TODO, separate task.**
- **§8.2 and §8.3** — `build-proximity` is their input, but their own commands
  (baseline decomposition, resolvability profile, cost tables, region forensics
  re-run on as01–03) are not in this task.
- The nearest-seed-argmin cross-check of the half-gap rule is a diagnostic in
  `meta.json`, not a second labelling scheme.
