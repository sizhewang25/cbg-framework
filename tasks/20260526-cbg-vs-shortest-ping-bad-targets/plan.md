# CBG vs Shortest-Ping — Identify Bad-Prediction Targets — Plan

## Background

On the all-6-ASN error CDF we observed that **none of the CBG variants beat the
shortest-ping baseline** (the fallback that picks the lowest-RTT VP's coord).
This task drills into one ASN — `north_america_as7018` (AT&T) — to enumerate
**which specific target anchors CBG predicts worse than shortest ping**, so we
can characterize the failure mode (geometry, RTT distribution, fit quality, …)
case by case instead of via aggregate statistics.

## Context

### Inputs

- **Benchmark outputs** (5 folds × 4 combos):
  `scripts/benchmark/v2/outputs/north_america_as7018/ripe_atlas_asn_corpora/probes_to_anchors/fold_{0..4}/{vanilla_cbg,million_scale_cbg,octant_cbg,spotter_cbg}/`
  Each combo dir holds `targets.parquet` (per-target prediction + error_km),
  `fit_checkpoint.pkl` (LTD snapshot), `run.json` (runtime metadata).
- **Materialized inputs** (eval observations per target):
  `scripts/benchmark/v2/inputs/ripe_atlas_asn_corpora/north_america_as7018/probes_to_anchors/fold_{0..4}/eval_observations.parquet`
  One row per `(target, vp, latency_ms)` — needed to compute the shortest-ping
  baseline (the runner doesn't persist per-target nearest-VP error directly).
- **Source corpora**:
  - Probes: `datasets/ripe_atlas/asn_corpora/probes/north_america/probes_of_as_7018.json`
  - Anchors (folded): `datasets/ripe_atlas/asn_corpora/anchors/kfolds/anchor_fold_{0..4}.json`
- **Existing plotter**:
  [scripts/analysis/plot_error_diff_cdf.py](../../scripts/analysis/plot_error_diff_cdf.py)
  already does combo_A − combo_B error-diff CDFs; we'll reuse its loaders
  and possibly extend it to accept a `shortest_ping` pseudo-combo.

### Definitions

- **Shortest-ping error per target**: pick the VP with the lowest measured RTT
  for that target; predict that VP's coordinate; error = haversine to the
  target's true coord. Computable from `eval_observations.parquet`.
- **CBG error per target**: `error_km` column in `<combo>/targets.parquet`
  (already populated by the runner, includes both SUCCESS and FALLBACK rows;
  FALLBACK rows are *literally* shortest-ping predictions so their delta = 0).
- **Bad target** (working definition): a target where `error_CBG > error_shortest_ping`,
  i.e. `delta_km = error_CBG − error_shortest_ping > 0`. Severity = magnitude of delta.

### Joinability

- The fold prefix in [plot_error_diff_cdf.py:117-121](../../scripts/analysis/plot_error_diff_cdf.py#L117-L121)
  (`bucket[f"{fold}/{tid}"] = …`) already keeps the K folds disjoint in the
  combo-error dict, so the (fold, target_id) tuple is the natural join key
  for the baseline as well.
- The runner's fallback path means FALLBACK-status rows in targets.parquet
  share the *same* prediction as the shortest-ping baseline → those targets
  contribute `delta = 0` (small numerical noise from haversine recomputation,
  but otherwise indistinguishable). Useful sanity check.

## Goals

1. Compute the per-target shortest-ping error baseline for every fold of
   `north_america_as7018`.
2. For each of the 4 CBG combos, produce a per-target table of
   `(fold, target_id, error_CBG, error_shortest_ping, delta_km, status)`.
3. Identify the **worst-N targets per combo** (largest positive delta), and
   the **always-bad set** (targets where every combo loses to shortest ping).
4. Emit a CDF plot of `error_CBG − error_shortest_ping` for the 4 combos,
   mirroring `plot_error_diff_cdf.py`'s style (one curve per combo, vertical
   line at 0, "A better"/"B better" annotations).
5. For the worst-N targets, capture enough metadata to investigate them
   downstream: anchor lat/lon/ASN/country, n_obs, n_ltd_success, MTL
   intersection_kind, CTR status, and the RTT distribution from
   `eval_observations.parquet`.

## Approach

### Phase 0 — Compute the baseline

Add a helper that loads `eval_observations.parquet` per fold and emits
`{(fold, target_id): (nearest_vp_id, error_km)}`:

```python
# For each (fold, target):
#   pick the row with the smallest latency_ms
#   error_km = haversine(target_lat/lon, vp_lat/lon)
```

The `haversine_distance` helper is already in
`scripts/libs/cbg/rtt_model.py:32`.

### Phase 1 — Join + rank

A small script (`scripts/analysis/inspect_cbg_vs_shortest_ping.py`) that:

1. Walks the `outputs/north_america_as7018/` tree using `_v2_io.discover_combos`.
2. Loads the baseline (Phase 0) and per-combo `targets.parquet`.
3. Inner-joins on `(fold, target_id)`, computes `delta_km`.
4. Writes one row per `(combo_id, fold, target_id)` to a parquet/CSV with the
   forensic columns (`status`, `n_obs`, `n_ltd_success`, `mtl_intersection_kind`,
   `error_CBG`, `error_shortest_ping`, `delta_km`).
5. Prints the worst-N per combo and the intersection set (targets bad across
   all 4 combos).

### Phase 2 — Plotting (re-use existing diff plotter)

Two options — start with option A:

- **A. Synthesize a `shortest_ping` pseudo-combo dict** and feed it through
  `plot_error_diff_cdf.plot_error_diff_cdf` with
  `pairs=[(vanilla_cbg, shortest_ping), …]`. No edit to the plotter needed.
- **B.** If we want this Snakemake-integrated, add a `--baseline shortest_ping`
  flag (or a sibling rule) that bypasses `--pair` and auto-builds the baseline.
  Defer to Phase 3 if Phase 1 reveals we need it routinely.

### Phase 3 — Drill into worst targets (manual, post-Phase 1)

For each of the top ~10 worst targets:
- Look up the anchor row in `anchor_fold_<N>.json` → coords, ASN, country.
- Reload the LTD checkpoint (`fit_checkpoint.pkl`) and inspect the per-VP
  fitted slope/intercept for the VPs in that target's obs list.
- Plot the VP geometry (probes around the anchor) with the predicted radii
  to see whether the feasible region contains the truth.

The exact deliverable shape for this phase depends on what the Phase 1 list
looks like — we'll branch the plan then.

## Caveats

- **FALLBACK-status rows have delta = 0** by construction (the runner used the
  shortest-ping coord as the prediction). Filter to `status == "SUCCESS"` when
  asking "where does CBG *succeed* but still lose to the baseline".
- **`load_targets` doesn't carry VP-level RTTs**. For RTT diagnostics on a
  worst target we must reload from `eval_observations.parquet`.
- **K-fold join hygiene**: don't drop the fold prefix in the join key —
  target_ids (anchor IPs) are globally unique across folds, but stratification
  is fold-disjoint, so collapsing would still be safe; keeping the fold
  prefix makes provenance explicit and matches the existing diff-CDF
  convention.
- **Haversine vs runner's error_km**: the runner uses the same
  `haversine_distance` helper, so deltas should be numerically exact up to
  float roundoff. If not, suspect a different ground-truth source.
- **Datasets directory name**: the user wrote "ans_corpora" in the prompt —
  actual path is `asn_corpora` (already corrected throughout this doc).
