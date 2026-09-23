# Integrate `eval_source` into the analysis/v3 CLI — Report

**Status**: In Progress — Phases 0-1 complete (Stages 0-3, three commits); Phase 2 next; Phase 5 scoped and agreed
**Created**: 2026-09-23
**Last Updated**: 2026-09-23 (Phase 5 agreed: eval-source becomes RTT-only)

## Summary

The refactor was scoped after establishing that only one of the three
"eval source" modules is live, that `analysis/v3` consumes a five-column
slice of a 44-column artifact, and that the layer it reaches into costs
~2 s and arrives with the CBG solver stack. Scope chosen: move the *entry
point* and the *data contract*, leave the artifacts and the analysis half
in place.

Stages 0-3 are committed. Net **+458 / −608 lines**, 1294 tests passing,
and as01's per-target CSV and stats JSON reproduce **byte-identically**
at every stage.

| Stage | Commit | Result |
|---|---|---|
| 1 — delete dead evaluator | `e90c608` | `eval_source_v2.py` gone (−398) |
| 2 — extract the contract | `4392dfc` | `scripts/libs/canonical/` owns schema + pairs + eval filters |
| 3 — top-level imports | `47338a7` | 5 modules rewired; `test_layering.py` added |

## Findings

### F1 — `eval_source_v2.py` was dead, and unsafe to revive

One repo-wide reference: its own `def eval_source_v2(` at line 294. No
import, CLI command, Snakemake rule, test or shell caller. Arrived dead in
merge `316c06a`. Three independent reasons not to salvage it:

- it writes the **same six filenames** as v1, so pointing it at a run's
  `eval_source/` silently overwrites the live artifacts;
- `_classification_easiness_v2` reuses v1 stats keys for different
  quantities — `geography_edge_distance_km_distribution` is VP→target in
  v1 and VP→centroid there;
- it would `KeyError` on any CSV without a `weight` column (unguarded
  city-level groupby at `:229-234`, where v1 guards at `eval_source.py:942`);
- the design note its docstring cites,
  `notes/2026-07-20-eval-source-classification-easiness-design.md`, is not
  in the tree.

### F2 — The layering defect, measured

`import scripts.benchmark.v2.eval_source` = **1.96 s**, and pulls
`scripts.framework.v2` (every MTL solver) via
`eval_source.py:144 → sources/generic_csv._raw_str → sources/__init__`.
Five `analysis/v3` modules imported through that chain, inside function
bodies, to read a CSV. After Stage 2, `import scripts.libs.canonical`
pulls neither `scripts.benchmark` nor `scripts.framework`; after Stage 3
neither do the five modules, nor the v3 CLI.

### F3 — v3's read surface is 5 columns of 44

`target_id`, `shortest_ping_vp_{id,lat,lon,km}` (`io.load_sping_vp`) and
`min_inflation` (`proximity._carry`), plus `eval_stats.json["csv"]`
(`bipartite.resolve_source_csv`). Nothing in v3 reads `_eval_clusters.csv`,
`_vp_mesh_km.csv`, `_cluster_mesh_km.csv`, `_clusters/`, or any
proximity/topology column — `io.py:296`'s docstring advertising the "§8.2
proximity decomposition" as consumed is inaccurate.

### F4 — The committed `eval_source/` artifacts are not reproducible

Re-running the precheck on as01 with today's CSV yields the same 399
targets **in a different row order**, and
`classification_easiness.vertex_props.targets.n_unique_target_asns` goes
`1 → null` — the reconstructed CSV has no `target_asn` column, consistent
with `*.reconstruction.json`'s `columns_unrecoverable`. The committed
artifacts predate the reconstruction. The golden baseline was therefore
switched to a self-generated pre-change run, which is the correct baseline
for a refactor regardless. **Those artifacts are due a regeneration**,
tracked separately.

### F5 — Centroid divergence between the two clusterings is real but tiny

`materialize-target-space` deduplicates coordinates before clustering
(`cli.py:791-804`) while `eval_source` passes one row per target, so
duplicates act as weights on the spherical centroid. Measured:

| run | clusters with >1 coordinate | replicas | centroid shift |
|---|---|---|---|
| as01-260728-260802 | cluster 0 | [20, 20] | 0 |
| as01-260728-260802 | cluster 2 | **[19, 20]** | **73 m** |
| as03-260728-260802-mesh | cluster 0 | [20, 20] | 0 |

Predictive rule: equal replica counts → identical centroids; unequal →
a shift. Against `cell_gap_km` in the hundreds of km this is inert, which
is why the correction was ruled out of scope.

### F6 — **Unplanned**: `n_members` / `is_singleton` mean two different things

Same root cause as F5, much larger effect. For
`as03-260728-260802-mesh`, comparing `eval_source/<stem>_clusters/` against
`generic_csv/anchors_to_probes/clusters/`:

- **identical**: target set (458), cluster count (22), target→`cluster_id`
  mapping (100%), the partition itself, `dist_to_centroid_km` on all 458,
  `centroid_lat/lon` (to 0.000000000 km), `radius_km`, `diameter_km`,
  column list;
- **different**: `n_members` sums to **458** (targets) vs **23** (unique
  coordinates); `is_singleton` is true for **0 of 22** vs **21 of 22**.

458 targets sit at 23 distinct coordinates (~20 IP replicas per location),
so the two columns differ by ~20×, and a cluster holding 20 targets is
reported `is_singleton=True` by the benchmark side.

Three consumers read those columns: `_cluster_data.py:94,308` sums
`n_members` **as a target count** (would read 23, not 458),
`plot_ground_truth_clusters.py:86` sizes markers by it, and
`voronoi.py:226` re-derives `is_singleton = n_members <= 1`.

Neither definition is wrong in isolation; the same column name in an
identically-shaped file meaning two different things is the defect.

### F7 — Producer and naming hazards in `eval_source/`

- Two producers write the same six files with **different knob coverage**:
  `materialize-target-space --with-eval-source` forwards the radius and
  the two weight filters only, while the smk rule forwards four more from
  the config's `precheck:` block.
- `eval-source`'s `--out-dir` defaults to the CSV's own parent, "where
  nothing downstream looks" (`benchmark/v2/README.md:124`).
- `<stem>_stats.json` (written by `visualization/.../inspect_source.py`)
  sits beside `<stem>_eval_stats.json` (written by `eval_source`), with
  disjoint key sets.

### F8 — Every dropped column already has a grid-native counterpart

The evidence behind the Phase 5 decision. Nothing in the drop list is lost:

| dropped (complete-linkage) | already on disk (grid) |
|---|---|
| `cell_gap_km` | `seeds.csv → nearest_seed_km` (as01 p50 301.0 km) |
| `target_distinguishable_vp_dist_km` | `seeds.csv → margin_km` (p50 150.5 km) |
| `knn_*`, `n_competitors_1_5x` | `meta.json → seed_pairwise_km`, `adjacency_edge_km`, `class_adjacency_degree` |
| `closest_vp_to_centroid_km`, `shortest_ping_vp_to_centroid_km` | proximity's `tg_seed` / `sping.to_seed_km` |
| `n_discriminative_vps`, `proximity_label`, `*_in_same_cluster` | proximity's `has_{proximate,discriminative}_*` |
| `n_members` / `is_singleton` | `seeds.csv → n_targets` / `meta → n_singleton_seeds` |
| `n_avail_vps`, `closest_vp_km`/`_id` | `target_nodes.csv → degree_to_vp`, `nearest_measured_vp_km`/`_id` |
| `bipartite_coverage_summary`, VP mesh | `bipartite meta.json → edges`, `pairwise_distance_cdf.csv` |

`vp_nodes.csv` already carries `cell_id`, so VPs are grid-clustered today.

### F9 — The paper table moves by exactly one cell

`build_dataset_properties_csv.py` re-pointed at the grid-native source:

| run | `# Target Cluster` | `% Colocated VPs` |
|---|---|---|
| as01-mesh | 18 → 18 | 95.0% → 95.0% |
| as02-mesh | **20 → 22** | 66.0% → 66.0% |
| as03-mesh | 22 → 22 | 95.6% → 95.6% |

`% Colocated VPs` is unchanged because `closest_vp_km` (BallTree, eval_source)
and `nearest_measured_vp_km` (argmin, bipartite) agree to **0.0005 km** — the
documented tie-break difference affects which VP is named, not the distance.

### F10 — bipartite cannot absorb the remainder

Its `meta.json` declares `scope.rtt = "not used beyond the canonical CSV's own
rtt_ms > 0 row filter"`. §7.3 is the measurement graph's geometry independent
of latency; merging RTT in would dissolve a stated invariant.
`target-answer-space` has no VPs in it. So eval-source survives as a command
at ~250 lines (from 1011), RTT-only — small and coherent rather than trivial.

## Conclusions

*(To be filled when the task completes.)*

Held so far: every stage is byte-identical on real data, the layering
property is pinned by a test with a verified negative control, and the two
hazards the user originally flagged — the dead `eval_source_v2` and the
confusable overlap with v3 — are half-closed, with the remaining half
(entry point, `eval_dataset/` fallback, `--out-dir` footgun) scoped into
Stages 4-6.
