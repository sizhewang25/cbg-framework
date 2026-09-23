# Integrate `eval_source` into the analysis/v3 CLI — Plan

## Background

Three things named "eval source" overlap, and the overlap is a live hazard
rather than untidiness:

- **`scripts/benchmark/v2/eval_source.py`** (1189 lines) is the real dataset
  precheck. It never touches benchmark outputs — its own docstring says so —
  yet it lives in the benchmark layer, so six `analysis/v3` modules reached
  *down* into it, with deferred inside-function imports, for the canonical-CSV
  contract they need to parse their own input.
- **`scripts/benchmark/v2/eval_source_v2.py`** (398 lines) was **dead**: one
  repo-wide reference, its own `def` at line 294. It wrote the *same six
  filenames* as v1, so running it against a run dir would have silently
  overwritten the live artifacts, and `_classification_easiness_v2` reused v1
  stats keys for different quantities.
- **`outputs/benchmark/v2/<run>/eval_dataset/`** has no producer in HEAD. Four
  runs still carry one holding an *older* eval_source output (missing the 7
  `knn_*` columns and later stats blocks), and `paths.eval_file` silently falls
  back to it.

Measured layering defect: `import scripts.benchmark.v2.eval_source` took
**1.96 s** and pulled **`scripts.framework.v2`** — every CBG solver the
analysis layer exists to evaluate — because `load_canonical_csv` borrowed
`_raw_str` from `sources/generic_csv.py`.

Prior related tasks: `20260714-dataset-precheck-workflow`,
`20260715-eval-bench-results`.

## Context

**Approved plan**: `~/.claude/plans/according-to-the-hidden-humming-hamming.md`
(7 stages; scope chosen as "entry point + shared library", artifacts stay put).

**Producers of `eval_source/`** — three, with different knob coverage:

| Producer | Knobs forwarded |
|---|---|
| `benchmark.v2.cli eval-source` | all 7 |
| `materialize-target-space --with-eval-source` (`cli.py:1044-1058`) | radius + 2 weight filters only |
| `inspect_dataset.smk` `rule eval_source` (line 324) | all 7, from the config's `precheck:` block |

A fourth tool, `visualization/benchmark/inspect_source.py`, writes
`<stem>_stats.json` + two PNGs into the same directory — note that filename
against `<stem>_eval_stats.json` beside it.

**What v3 actually consumes**: 5 of the 44 columns —
`target_id`, `shortest_ping_vp_{id,lat,lon,km}` (via `io.load_sping_vp`) and
`min_inflation` (via `proximity._carry`) — plus `eval_stats.json["csv"]` for
provenance (`bipartite.resolve_source_csv`). Nothing in v3 reads
`_eval_clusters.csv`, `_vp_mesh_km.csv`, `_cluster_mesh_km.csv`, `_clusters/`,
or any proximity/topology column.

**Five non-v3 readers** glob `<run>/eval_source/*` directly and pin the
artifacts in place: `dataset_props/topology/` (×3),
`correlation/classification_category_association.py`, and `inspect_source.py`
(which also writes back into that directory). `paper/datasets/
build_dataset_properties_csv.py` takes explicit paths and is path-agnostic.

**Three answer spaces over the same targets**: `eval_source/<stem>_clusters/`
(agglomerative over target rows), `<source>/<setup>/clusters/` (same algorithm
over unique coordinates), `target-answer-space/<grid>-<res>/` (grid seeds).

## Goals

1. One entry point for the precheck: `analysis.v3.cli eval-source`.
2. The canonical-CSV contract owned by a library both layers import, with the
   v3 import graph free of `scripts.benchmark` / `scripts.framework`.
3. Dead evaluator gone; `eval_dataset/` silent fallback turned into a loud
   error; the `--out-dir defaults to the CSV's parent` footgun closed.
4. `adjacency_metrics.py` promoted from orphan to a wired-in metric.
5. **Byte-identical output through Stage 6.** Stage 7 is additive-only.

## Approach

Seven stages, one commit each, independently revertible, with a golden diff
after every one.

| Stage | Content | State |
|---|---|---|
| 0 | Golden harness: self-generated baseline + column/JSON-leaf diff | done |
| 1 | Delete `eval_source_v2.py` (keep `adjacency_metrics.py`) | done `e90c608` |
| 2 | Extract `scripts/libs/canonical/{schema,pairs,eval_filters}.py`; flip the `raw_str` edge | done `4392dfc` |
| 3 | Six deferred imports → top-level; `test_layering.py` | done `47338a7` |
| 4 | `resolve_source_csv_detail` split + the v3 `eval-source` command | pending |
| 5 | Point `inspect_dataset.smk` at V3_CLI; deprecate the v2 command | pending |
| 6 | Retire `eval_dataset/` reader code | pending |
| 7 | Wire in the Voronoi adjacency (additive) | pending |

Stage 4's command takes `--csv/--out-dir` (both required together) **or**
`--run-id/--all-runs`; no `--grid/--resolution/--sweep` (no seed enters the
computation) and no `--analysis-root` (nothing is written under it). Its CSV
resolution uses `prefer_manifest=True` so it reads `target_space.json["csv"]`
rather than the `eval_stats.json` it is about to overwrite, and it **scores
rather than raises** on `csv_is_mesh_superset`, mirroring `cli.py:1051-1058`.

## Phase 5 (agreed 2026-09-23): one answer space, and eval-source becomes RTT-only

Decided after establishing that every agglomerative-derived column already has
a grid-native counterpart on disk. The end state is three commands with
**disjoint scopes, all keyed on the same seeds**:

| command | owns |
|---|---|
| `build-answer-space` | targets x grid: cells, seeds, `nearest_seed_km`, `margin_km`, adjacency |
| `build-bipartite-graph` | VP<->target geometry: degrees, nearest-VP, edge lengths, angular — **RTT-free by declared scope** |
| `eval-source` | **RTT only**: the shortest-ping VP, inflation, RTT/distance regime, anycast |

**Dropped from eval_source** (obsolete — complete-linkage, superseded by the
grid; duplicated; no v3 consumer):
`cell_gap_km`, `target_distinguishable_vp_dist_km`, `closest_vp_to_centroid_km`,
`shortest_ping_vp_to_centroid_km`, `n_discriminative_vps`,
`best_discriminative_rtt_rank`, `proximity_label`, the seven `knn_*`,
`n_competitors_1_5x`, `*_in_same_cluster` — with their producers
`cluster_targets`, `_knn_centroid_metrics`, `proximity_metrics`,
`proximity_summary`, `_answer_space_graph_metrics`, and
**`classification_easiness_summary`** (agreed 2026-09-23).

**Also dropped** (duplicated in `bipartite-graph/` today):
`n_avail_vps` (= `degree_to_vp`), `closest_vp_km` / `closest_vp_id`
(= `nearest_measured_vp_km` / `_id`), `bipartite_coverage_summary`
(= `meta.json.edges`), `_nearest_neighbor_stats`, the VP mesh CSV.

**Kept** — everything RTT, ~250 lines: `shortest_ping_vp_{id,lat,lon,km}`
(load-bearing, see caveats), `min_rtt_ms`, `min_inflation`,
`rtt_weighted_dist_km`, `rtt_dist_spearman`, `closest_vp_rtt_rank`,
`closest_is_shortest_ping`, `closest_to_shortest_ping_km`,
`soi_violation_share`, the anycast block, `rtt_quality_summary`, and the
percentile rollup trimmed to the survivors.

**Not merged into bipartite**, despite the shrinkage: its `meta.json` declares
`scope.rtt = "not used beyond the canonical CSV's own rtt_ms > 0 row filter"`.
§7.3 is the measurement graph's geometry independent of latency, and pushing
RTT in dissolves a stated, testable invariant. `target-answer-space` has no VPs
in it at all. So eval-source stays a command; it just stops pretending to be
three things.

**`build_dataset_properties_csv.py` moves to the grid-native source**
(agreed 2026-09-23):

| column | was | becomes | effect |
|---|---|---|---|
| `# Target Cluster` | `target_clustering.n_clusters` | `target-answer-space/<grid>-<res>/meta.json -> n_seeds` | as01 18->18, **as02 20->22**, as03 22->22 |
| `% Colocated VPs` | `target_clustering.closest_vp_within_radius_share` | `(target_nodes.nearest_measured_vp_km <= 50).mean()` | **unchanged** (95.0 / 66.0 / 95.6; the two distances agree to 0.0005 km) |

Open sub-decision: the 50 km threshold was `radius_km`, the clustering
parameter, so the old metric was self-consistent. Grid-native it is a bare
constant. Keeping 50 km preserves the published number exactly; switching to
per-target `margin_km` would ask a different question and move the number.
Recommendation: keep 50 km, named, and say so in the column's docstring.

**Stage 7 is re-homed by this.** The Voronoi adjacency was to be wired into
`classification_easiness_summary`, which is now deleted. Cell adjacency is
answer-space territory: `seeds.csv` already carries `class_adjacency_degree`
and `meta.json` an `adjacency_edge_km`, which SCHEMA.md describes as doing
"§7.4's Delaunay-neighbour role, done as k-nearest instead". So
`voronoi_adjacency_distances` goes into `answer_space.py`, replacing a
documented approximation with the real thing — a better home than the one it
was rescued into.

**Deliberately left broken**: `plot_category_filtered_topology.py` and
`classification_category_association.py` read `proximity_label`. Never used;
an import error beats them reporting labels keyed to a space that no longer
exists.

## Caveats

- **Do not unify the two proximity taxonomies.** v2's `proximity_label`
  (agglomerative centroids) and v3's `TAXONOMY` (grid seeds) apply the same
  `d(VP, centre) < d(centre, nearest other)/2` inequality to different class
  centres and cut different axes of the diamond. `proximity.py:24-28,48-50,
  176-179` documents the non-reuse as deliberate; merging by name swaps a
  guarantee for a rank test.
- **The committed `eval_source/` artifacts are stale** relative to today's
  CSVs: re-running as01 gives the same 399 targets in a different row order,
  and `n_unique_target_asns` goes `1 → null` because the reconstructed CSV has
  no `target_asn`. The golden baseline is therefore self-generated, not the
  on-disk artifacts. Those artifacts are due a regeneration, separately.
- **This tree has concurrent edits** (`figure_error_cdf.py`,
  `figure_spotter_normality.py`, `run_finals.sh`, several notes). Stage
  commits must stage only their own files; test counts move underneath.
- **Newly found, not yet scoped** — see report.md: `clusters.csv`'s
  `n_members` / `is_singleton` mean different things in the two triplets.
- Out of scope: relocating artifacts to the analysis root, a `radius_slug`,
  the `<stem>_` prefix, `eval_bench_results.py` / `bench_eval/`, deleting the
  v2 command or the four stale `eval_dataset/` trees from disk.
