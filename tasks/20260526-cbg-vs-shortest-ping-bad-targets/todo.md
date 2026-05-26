# CBG vs Shortest-Ping — Identify Bad-Prediction Targets — Todo

## Phase 0: Setup & Discovery
- [ ] Verify all 5 folds × 4 combos exist under `outputs/north_america_as7018/`
- [ ] Verify `inputs/ripe_atlas_asn_corpora/north_america_as7018/.../fold_*/eval_observations.parquet` is present (re-materialize if missing)
- [ ] Read one `eval_observations.parquet` to confirm `(target_id, target_lat, target_lon, vp_id, vp_lat, vp_lon, latency_ms)` schema matches the writer in `scripts/benchmark/v2/inputs.py`
- [ ] Read one `targets.parquet` to confirm the columns we need: `target_id, target_lat, target_lon, pred_lat, pred_lon, error_km, status, n_obs, n_ltd_success, mtl_intersection_kind`

## Phase 1: Baseline + Join + Worst-N
- [ ] Write `scripts/analysis/inspect_cbg_vs_shortest_ping.py` skeleton with arg parsing (`--run-dir`, `--source`, `--top-n`, `--out-csv`, `--out-plot`)
- [ ] Add `load_shortest_ping_baseline(inputs_dir) -> dict[(fold, target_id), {nearest_vp, error_km}]` that scans `eval_observations.parquet` per fold and picks the lowest-latency row per target
- [ ] Sanity check: cross-reference shortest-ping `error_km` against FALLBACK rows in `targets.parquet` — should agree to ~1e-6 km
- [ ] Build the per-combo join: emit a dataframe `(combo_id, fold, target_id, error_CBG, error_baseline, delta_km, status, n_obs, n_ltd_success, mtl_intersection_kind)`
- [ ] Restrict to `status == "SUCCESS"` for the "CBG-loses-when-CBG-succeeds" set; keep the FALLBACK rows in a separate "trivially equal" bucket
- [ ] Print the worst-N (default N=20) per combo with their forensic columns
- [ ] Compute the intersection set: targets where all 4 combos have `delta_km > 0`
- [ ] Write the full per-target table to a CSV under `tasks/20260526-.../outputs/`

## Phase 2: Diff-CDF Plot
- [ ] Synthesize a `shortest_ping` pseudo-combo dict (`{fold/target_id: error_km}`) shaped like `plot_error_diff_cdf`'s `target_errors_by_combo` input
- [ ] Call `plot_error_diff_cdf.plot_error_diff_cdf` with the 4 CBG combos paired against `shortest_ping`
- [ ] Save the resulting PNG under `tasks/20260526-.../outputs/`
- [ ] Eyeball: confirm the legends show "{combo} better: <50%" if the bulk-CDF pathology really holds

## Phase 3: Drill into worst targets (post Phase 1)
- [ ] Pick top ~5 from the always-bad intersection set
- [ ] For each: look up anchor metadata in `datasets/ripe_atlas/asn_corpora/anchors/kfolds/anchor_fold_<N>.json` (coords, ASN, country, FQDN)
- [ ] Reload `vanilla_cbg/fit_checkpoint.pkl` for the matching fold, inspect per-VP `slope`/`intercept` for the VPs that observed this target
- [ ] Compute per-VP `(rtt, predicted_radius, true_distance)` triples for the target — flag any VP where `predicted_radius < true_distance` (constraint violation)
- [ ] Decide whether to take this further with a per-target geometry plot (deferred until we know what the rows look like)

## Phase 4: Wrap-up
- [ ] Summarize findings in `report.md`: how many targets lose to baseline per combo, the always-bad set size, recurring failure patterns
- [ ] Capture any anomalies in `lesson.md` for future combos / future ASN drilldowns
