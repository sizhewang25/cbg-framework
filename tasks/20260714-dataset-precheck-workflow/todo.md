# Dataset Precheck Workflow — Todo

## Phase 0: Metric definitions & discovery
- [x] Settle inflation-at-proximity: `closest_vp_rtt_rank` + VP mesh distance map (excess-radius-km dropped); Spearman stays a coherence descriptor, not an inflation detector (plan.md)
- [x] Settle anycast/spread metric: `vp_pair_disk_overlap_km` + iGreedy `n_disjoint_sites` over low-RTT set `rtt ≤ min_rtt + 10ms` (plan.md)
- [x] Verify `cluster_ground_truth` behavior at R=0 (all singletons; duplicate-coord edge documented in plan.md caveats)
- [x] Fix `cluster_ground_truth` O(n²) memory blowup at large target counts — cluster over unique `(lat, lon)` rows only, broadcast labels back (2026-07-15)
- [x] Decide output layout for inspect_dataset.smk (per-config out dir vs alongside CSV) — resolved 2026-07-15: alongside the CSV, matching eval_source.py/inspect_source.py's existing `csv.parent` defaults (plan.md)

## Phase 1: eval_source.py extensions (metrics only, canonical CSV in) — DONE 2026-07-14
- [x] Add `cell_gap_km` per cluster/target (BallTree self-query k=top_n+1, one shot with neighbors)
- [x] Add top-N neighbor clusters; write per-cluster CSV `<stem>_eval_clusters.csv` (centroid, members, radius, gap, neighbor{i}_cluster_id/_km)
- [x] Add centroid-based proximity: `closest_vp_to_centroid_km`, `shortest_ping_vp_to_centroid_km` (+ truth centroid coords on per-target rows)
- [x] Add discriminative-set metrics: `n_discriminative_vps`, `has_vp_proximity`, `shortest_ping_vp_is_discriminative`, `best_discriminative_rtt_rank`, `target_distinguishable_vp_dist_km` (= gap/2)
- [x] Add `cbg_opportunity_share` + `opportunity_baseline_lucky_share` to stats JSON `proximity` block
- [x] Emit distance mesh artifacts: `<stem>_vp_mesh_km.csv` + `<stem>_cluster_mesh_km.csv` (id-indexed, `mesh_max_n` cap, 0 disables; answer-space mesh over cluster centroids, not raw targets — target×target explodes at million scale)
- [x] Add RTT-regime metrics: `rtt_dist_spearman`, `closest_is_shortest_ping` + `closest_to_shortest_ping_km`, `closest_vp_rtt_rank`, `soi_violation_share` (per-target) + `pair_soi_violation_share` (dataset), `vp_pair_disk_overlap_km` + `n_disjoint_sites` + `anycast_suspect`
- [x] Derive per-target `proximity_label` ladder (NO_PROXIMITY/HAS_NOT_USED_PROXIMITY/HAS_USED_PROXIMITY; no SPARSE label — folded into NO_PROXIMITY when D empty; Voronoi diagnostic-only); label shares in stats JSON
- [x] Extend stats JSON: `target_clustering` (+n_singletons), `proximity`, `rtt_quality` blocks; new metrics in percentile table
- [x] Wire params through `eval-source` CLI: --top-n-neighbors, --anycast-delta-ms, --spearman-min-pairs, --mesh-max-n (--sparse-floor removed with the SPARSE label)
- [x] Unit tests: 22 new tests (rank norm, Spearman guards, SOI, anycast disjointness, cluster frame, ladder incl. single-VP D-only labeling + Voronoi-luck-not-label, mesh artifacts + cap); full v2 suite green (194 tests)
- [x] Smoke run on as7018-us-test01.csv — see report.md findings
- [x] Post-review trim (2026-07-14): drop `best_radius_km` + the threshold `resolvability` table (+ `--thresholds` flag) — LTD-flavored, belongs to the LTD-specific study; `cbg_opportunity_share` finalized = NO_PROXIMITY + HAS_NOT_USED_PROXIMITY (baseline-beatable), lucky share over the combined opportunity population

## Phase 2: inspect_source.py visualization
- [x] Consume eval_source per-target/per-cluster CSVs (no metric recomputation) — 2026-07-15: `eval_source.py`'s `cluster_targets` now also writes the canonical `cluster_ground_truth` triplet (clusters.csv/assignments.csv/meta.json) to `<stem>_clusters/`; inspect_source reads it, computes nothing
- [x] Add cluster map layer: centroids, cluster membership, Voronoi cells clipped to geometry constraint (mainland US) — 2026-07-15: reused `scripts/visualization/cluster/plot_ground_truth_clusters.py` + `voronoi.py` wholesale (already had landmass-clipped Voronoi via Natural Earth/cartopy, no new dependency) instead of building a new interactive layer; `inspect_source.py` gained `_build_cluster_map` + `--landmass`, writes `<stem>_cluster_map.png`. Verified on as7018-us-test01 (17 regions, US-clipped cells render sanely — see report.md)
- [ ] Color targets by proximity margin / rtt_regime; show per-target metrics in popups — **descoped to a static color-by (no popups)** per 2026-07-15 decision to reuse the static matplotlib/cartopy plotter rather than build a new interactive map; needs a small opt-in extension to `plot_ground_truth_clusters._plot_map`/`plot_clusters` (color-by `proximity_label` instead of only `cluster_id`) — not yet done
- [x] Keep existing occurrence CDFs + flow map intact — additive change, both untouched and still produced

## Phase 3: inspect_dataset.smk orchestration
- [ ] Create `inspect_dataset.smk` reading benchmark config yaml(s) for csv_path + precheck params (style: cluster_world_map.smk)
- [ ] Rules: eval_source metrics → inspect_source visuals; document run command in header

## Phase 4: Verification
- [ ] Run end-to-end on `datasets/ripe_as7018/as7018-us-test01.csv` with `as7018_us_test01.yaml` (via the Phase-3 smk, once it exists — ad hoc CLI run already done, see below)
- [ ] Sanity-check metrics against known findings (53 VPs × 78 targets; AS7018 US proximity profile)
- [x] Eyeball the Voronoi map layer; confirm clipped cells render sanely — done ad hoc 2026-07-15 (`eval-source` → `inspect-source --landmass US` on as7018-us-test01.csv): 17 regions, US-clipped Voronoi cells, singleton/clustered coloring all render correctly; not yet run through the Phase-3 smk since that doesn't exist yet
