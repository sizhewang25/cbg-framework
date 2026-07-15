# Dataset Precheck Workflow — Report

**Status**: Complete
**Created**: 2026-07-14
**Last Updated**: 2026-07-15
**Completed**: 2026-07-15

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

Also closed Phase 0's last open item: `inspect_dataset.smk`'s output layout
will be **alongside the CSV**, matching `eval_source.py`'s and
`inspect_source.py`'s existing `out_dir` defaults (both already `csv.parent`,
already realized on disk under `datasets/ripe_as7018/`). Phase 0 is now
fully done.

Started Phase 2 (inspect_source.py visualization). Rather than build a new
interactive cluster/Voronoi layer, reused the existing
`scripts/visualization/cluster/plot_ground_truth_clusters.py` +
`voronoi.py` (already implements landmass-clipped Voronoi over the
`cluster_ground_truth` answer space via Natural Earth/cartopy — no new
dependency). To feed it without recomputing anything, `eval_source.py`'s
`cluster_targets` now also writes the canonical
clusters.csv/assignments.csv/meta.json triplet to `<stem>_clusters/`;
`inspect_source.py` gained `_build_cluster_map` (skips gracefully if that
dir doesn't exist yet) and a `--landmass` flag, producing
`<stem>_cluster_map.png`. Verified end-to-end on
`datasets/ripe_as7018/as7018-us-test01.csv`: `eval-source` →
`inspect-source --landmass US` produces a correctly US-clipped 17-region
Voronoi partition with sane singleton/clustered-member coloring (see PNG
rendered during this session). Not done: per-target proximity-label/margin
coloring — this was descoped from the original "popups" plan (a static
figure can't do interactive popups) to a plain color-by extension, which
still needs a small opt-in addition to `plot_ground_truth_clusters`'s
plotting functions. Dropped for good at task close (see Conclusions).

## Phase 3 + 4 (2026-07-15)

Created `scripts/benchmark/v2/inspect_dataset.smk`: two rules
(`eval_source`, `inspect_source`) plus `rule all`, reading
`source_kwargs.csv_path` and an optional `precheck:` block from the same
benchmark config yaml the main `Snakefile` uses (cluster_radius_km,
top_n_neighbors, anycast_delta_ms, spearman_min_pairs, mesh_max_n, landmass —
each defaults to `eval_source.py`'s own `DEFAULT_*` constant). Added
`precheck: {landmass: US}` to `as7018_us_test01.yaml` and ran it for real:

```
snakemake -s scripts/benchmark/v2/inspect_dataset.smk \
    --configfile scripts/benchmark/v2/config/as7018_us_test01.yaml -j 1
```

Dry-run DAG matched expectations (2 jobs + `all`), the real run completed
3/3 jobs, and a second dry-run confirmed idempotence ("Nothing to be done").
Regenerated metrics match the recorded findings exactly (4129 pairs, 53
VPs, 78 targets, R=50 → 17 clusters/4 singletons, cbg_opportunity_share =
51.3%), and the smk-driven cluster map is byte-identical to the earlier ad
hoc render (US-clipped Voronoi, 17 cells, correct singleton/clustered
coloring). The two already-tracked artifacts (`_flow_map.html`,
`_occurrence_cdf.png`) regenerated byte-identical to what's committed, so
nothing needed re-staging there. Phases 0, 1, 3, and 4 are now fully done;
Phase 2's one remaining item (proximity-label/margin coloring) is the only
thing left on this task.

## Conclusions

The dataset precheck exists and works end-to-end, driven entirely by a
benchmark config yaml: `snakemake -s scripts/benchmark/v2/inspect_dataset.smk
--configfile <yaml> -j 1` takes a canonical CSV to per-target/per-cluster
metrics (`eval_source.py`), a proximity ladder headline
(`cbg_opportunity_share`), RTT-quality diagnostics, and a rendered
cluster/Voronoi map (`inspect_source.py`), all without touching the
benchmark's own combo/slice grid. Verified on
`datasets/ripe_as7018/as7018-us-test01.csv`: metrics and map match
hand-checked expectations exactly, and a second dry-run confirmed the
pipeline is idempotent.

Scope was deliberately kept to what the precheck actually needs: metric
logic stayed in `eval_source.py` (never touching benchmark outputs),
visualization reused two already-mature, already-tested modules
(`plot_ground_truth_clusters.py` + `voronoi.py`) instead of building a new
interactive map, and the smk reads the same yaml the benchmark run itself
uses rather than inventing a parallel config path. One originally-planned
item — coloring the cluster map by per-target `proximity_label`/margin,
with interactive popups — was dropped: the popups half became infeasible
once the static plotter was reused (a flat PNG can't be interactive), and
the remaining color-by polish wasn't judged worth building once the core
deliverable (metrics + map + orchestration) was already working. It's
recorded as skipped, not silently absorbed, so it can be picked up later if
a specific need for it arises (`plot_ground_truth_clusters._plot_map` would
need a `color_by` option keyed off `<stem>_eval_per_target.csv`'s
`proximity_label` column).

A useful side effect of this task: fixing `cluster_ground_truth`'s O(n²)
memory blowup (Phase 0) benefits every consumer of that function, not just
this precheck — including the `cluster-eval` CLI command and
`cluster.smk`'s answer-space materialization.
