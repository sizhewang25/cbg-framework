# Dataset Precheck Workflow — Report

**Status**: In Progress (Phase 1 done; Phases 2–4 pending)
**Created**: 2026-07-14
**Last Updated**: 2026-07-15

## Summary

Design settled and Phase 1 (eval_source metric engine) implemented and
verified: discriminative-set proximity ladder, cbg_opportunity headline,
RTT-regime diagnostics, anycast disk-disjointness, per-cluster topology CSV,
and VP/cluster-centroid distance meshes — all from the canonical CSV alone.
Full benchmark v2 suite green (194 tests). CLI `eval-source` exposes every
knob. Same-day trims after user review: `best_radius_km` and the threshold
"resolvability" table removed (LTD-flavored, belongs to the LTD-specific
study); `cbg_opportunity_share` finalized as NO_PROXIMITY +
HAS_NOT_USED_PROXIMITY (baseline-beatable share). Remaining:
inspect_source visuals (Phase 2), inspect_dataset.smk (Phase 3).

## Findings

Smoke run on `datasets/ripe_as7018/as7018-us-test01.csv` (53 VPs × 78 US
anchors, 4129 pairs, R=50 km → 17 clusters / 4 singletons):

- **Proximity ladder** (final names: HAS_USED / HAS_NOT_USED / NO, no
  SPARSE label): 48.7% HAS_USED_PROXIMITY, 14.1% HAS_NOT_USED_PROXIMITY,
  37.2% NO_PROXIMITY → **cbg_opportunity_share = 51.3%** (NO +
  HAS_NOT_USED: everywhere the baseline carries no correctness
  guarantee). Within it, the 37.2% NO_PROXIMITY slice is where only
  multilateration can help; the 14.1% HAS_NOT_USED slice has a
  discriminative VP that is RTT-buried, so better VP selection can also
  fix it.
- **opportunity_baseline_lucky_share = 0.0** — none of the 40 opportunity
  targets (29 NO + 11 HAS_NOT_USED) get rescued by directional luck;
  consistent with shortest_ping_vp_in_same_cluster_share (48.7%) exactly
  equaling the HAS_USED share.
- **RTT quality is clean**: zero SOI violations, zero anycast suspects
  (vp_pair_disk_overlap_km p5 = +657 km — nowhere near disjoint), Spearman
  ρ median 0.86. The RIPE anchor mesh is a well-behaved unicast dataset.
- **closest_is_shortest_ping = 50%**, but closest_vp_rtt_rank p50 = 0.01 —
  when the closest VP isn't the fastest, it's usually a near-tie, not
  buried (p95 rank 0.30).
- cell_gap_km p50 = 377 km at R=50 over the US answer space; 49/78 targets
  (62.8%) have a non-empty discriminative set.

## 2026-07-15 update

Fixed an O(n²) memory blowup in `cluster_ground_truth` (used by both
`eval-source`'s clustering block and the `cluster-eval` CLI command):
the complete-linkage step built a full haversine distance matrix over every
target, which becomes prohibitive on sources with large target counts. Since
duplicate `(lat, lon)` coordinates (common for sources where many targets
share one data-center location) sit at distance 0 and can never change a
merge decision, clustering now runs over unique coordinate rows only and
broadcasts labels back to every input point, with `member_counts` /
`is_singleton` / centroid / diameter recomputed against the true (weighted)
membership so nothing downstream regresses. Verified: 20,000 targets over
40 unique locations peaks at ~24 MB instead of the ~3.2 GB a dense 20,000×
20,000 matrix would need; full v2 suite green (198 tests, incl. 4 new
duplicate-coordinate tests). This is orthogonal to the R=0 duplicate-coord
semantics caveat above (that one's about singleton behavior, not memory).

## Conclusions

<Final assessment when task completes.>
