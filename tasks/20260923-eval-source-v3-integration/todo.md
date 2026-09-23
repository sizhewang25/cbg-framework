# Integrate `eval_source` into the analysis/v3 CLI — Todo

## Phase 0: Survey & golden harness — DONE 2026-09-23
- [x] Map the three modules, their artifacts, and every caller; confirm `eval_source_v2` is dead (one self-reference, `def` at line 294) and its cited design note absent from the tree (2026-09-23)
- [x] Map what v3 consumes: 5 of 44 columns + `eval_stats.json["csv"]`; nothing reads the cluster/mesh artifacts (2026-09-23)
- [x] Confirm `eval_dataset/` has no producer; 4 runs carry a stale copy behind `prefer_source`'s fallback (2026-09-23)
- [x] Measure the layering defect: `import eval_source` = 1.96 s, pulls `scripts.framework.v2` via `sources/generic_csv._raw_str` (2026-09-23)
- [x] Measure the centroid divergence between the two clusterings — max 73 m on as01 (1 of 18), 0 on as02/as03 — and rule the correction out of scope (2026-09-23)
- [x] Golden harness `scratchpad/golden_diff.py`: column-by-column + JSON-leaf diff, 1e-9 float tolerance, path keys excluded, target_id-aligned (2026-09-23)
- [x] Discover the committed artifacts are not reproducible from today's CSVs; switch the baseline to a self-generated pre-change run (2026-09-23)

## Phase 1: Dead code & the shared library — DONE 2026-09-23
- [x] Stage 1 — delete `eval_source_v2.py`; keep `adjacency_metrics.py` for Stage 7 (`e90c608`) (2026-09-23)
- [x] Stage 2 — create `scripts/libs/canonical/{schema,pairs,eval_filters,__init__}.py`; move the contract; flip the `raw_str` edge in `sources/generic_csv.py`; re-export the historical private names from `eval_source.py` (`4392dfc`) (2026-09-23)
- [x] Stage 3 — five modules + `test_pni.py` to top-level `scripts.libs.canonical` imports (`47338a7`) (2026-09-23)
- [x] `test_layering.py`: subprocess import-graph checks + a source-level check that catches a *deferred* reintroduction; verified by negative control (one reverted import → 6 failures) (2026-09-23)

## Phase 2: The CLI entry point
- [ ] Stage 4a — split `bipartite.resolve_source_csv` into `resolve_source_csv_detail(run, override, *, prefer_manifest) -> SourceCsv(path, origin, is_mesh_superset)`; wrapper keeps the existing raise verbatim. `test_bipartite.py` must pass **unmodified**
- [ ] Stage 4b — `scripts/analysis/v3/modules/eval_source.py` with `register(app)` / `@app.command("eval-source")`; register first in `_COMMAND_MODULES`
- [ ] Stage 4c — `test_eval_source_cli.py`: the 7 tests in the approved plan (byte-identity vs the v2 CLI, option defaults, run-id mode, pre-benchmark resolution, `prefer_manifest`, mesh-superset scored not rejected, `--csv` without `--out-dir` exits 2)
- [ ] Stage 5 — `inspect_dataset.smk` line 346 `BENCH_CLI` → `V3_CLI`; update the rule comment; deprecation echo on `benchmark/v2/cli.py:384`
- [ ] Stage 6 — delete `io.load_dataset_stats`, drop `prefer_source`, narrow `eval_file`/`eval_basename` to `eval_source/` with a loud error naming the stale dir; keep `eval_dataset_dir` and its `_NON_SOURCE_DIRS` entry

## Phase 3: Metrics
- [x] ~~Stage 7 — wire the Voronoi adjacency into `classification_easiness_summary`~~ **superseded 2026-09-23**: that function is dropped in Phase 5, and cell adjacency belongs to the answer space. Re-homed below (2026-09-23)
- [x] ~~Decide the `n_members` / `is_singleton` divergence~~ **dissolved 2026-09-23**: Phase 5 leaves one answer space, so the column has one meaning. `seeds.csv → n_targets` replaces it (2026-09-23)
- [ ] Wire `voronoi_adjacency_distances` / `adjacency_concentration_from_representative` into `answer_space.py`, replacing the k-nearest proxy SCHEMA.md:446 describes as standing in for "§7.4's Delaunay-neighbour role". Additive keys beside `class_adjacency_degree` / `adjacency_edge_km`

## Phase 5: eval-source becomes RTT-only (agreed 2026-09-23)
Ordering: **after Phase 2**, so Stages 4-6 keep their byte-identical contract and the golden baseline is regenerated once, not twice.
- [ ] Drop the agglomerative half: `cluster_targets`, `_knn_centroid_metrics`, `proximity_metrics`, `proximity_summary`, `_answer_space_graph_metrics`, `classification_easiness_summary`, and their ~13 per-target columns; trim `PER_TARGET_METRICS` to the survivors
- [ ] Drop the geometry half duplicated by `bipartite-graph/`: `n_avail_vps`, `closest_vp_km`/`_id`, `bipartite_coverage_summary`, `_nearest_neighbor_stats`, the VP mesh CSV
- [ ] Verify `io.load_sping_vp`'s four `shortest_ping_vp_*` columns survive untouched, and that `classify` + `build-proximity` reproduce their artifacts byte-for-byte
- [ ] Re-point `build_dataset_properties_csv.py`: `# Target Cluster` ← `target-answer-space/meta.json["n_seeds"]`; `% Colocated VPs` ← `(target_nodes.nearest_measured_vp_km <= 50).mean()`. Decide the 50 km constant (recommend: keep, named). Expect exactly one cell to move — **as02 20 → 22**
- [ ] Regenerate the golden baseline; record the new eval_source line count and the surviving stats-JSON key set
- [ ] Leave `plot_category_filtered_topology.py` / `classification_category_association.py` broken by design; note it in `SCHEMA.md` and the task report
- [ ] Update `SCHEMA.md` §5 + `scripts/analysis/v3/README.md`: three commands, disjoint scopes, one set of seeds

## Phase 4: Verification & close-out
- [ ] Golden diff identical after Stages 4-6; added-keys-only after Stage 7
- [ ] `snakemake -s inspect_dataset.smk --configfile configs/as01-260728-260802-mesh.yaml -n -j1` resolves; real run into a scratch outputs-root reproduces the baseline
- [ ] `eval_file("eval_per_target.csv")` resolves under `/eval_source/` for every run in `discover_runs()`, with no run newly erroring
- [ ] Full suite green (`scripts/benchmark/v2/tests scripts/analysis/v3/tests`)
- [ ] Docs: `scripts/analysis/v3/README.md`, `SCHEMA.md` §5 (line 4 attributes `eval_source*.py` to the v2 layer), `benchmark/v2/README.md:123`, the smk header block
