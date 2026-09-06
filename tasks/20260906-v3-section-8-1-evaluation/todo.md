# §8.1 Evaluation Scripts (v3) — Todo

## Phase 0: The `tg_seed` rename

- [x] 2026-09-06 Rename `truth_seed_id` → `tg_seed_id`, `truth_seed_rank` →
      `tg_seed_rank`, `error_to_truth_seed_km` → `error_to_tg_seed_km` in
      `modules/classify.py` and every v3 consumer (`venn.py`,
      `diagram/common/membership.py`; `pareto.py` had no reference).
      Locals renamed too (`truth`/`truth_pos`/`err_to_truth`)
- [x] 2026-09-06 Added README `## Column naming` (3 conventions) and a v2→v3
      rename map in SCHEMA.md §5 recording that `proximity_label` is dropped
- [x] 2026-09-06 Tests updated (401 passed); v2's `truth_centroid_*` /
      `truth_in_region` confirmed untouched
- [x] 2026-09-06 Re-ran `classify` on 4 runs × {h3-4, healpix-128}; all 8
      `topn_accuracy.csv` byte-identical (`cmp` clean); `plot-venn` and
      `plot-pareto` smoke-tested green

## Phase 1: `build-proximity`

- [x] 2026-09-06 `modules/proximity.py` with `register(app)`; added to
      `_COMMAND_MODULES`, to the README module table, the pipeline block and the
      outputs tree. New `RunPaths.proximity_dir`
- [x] 2026-09-06 Inputs resolved via `bipartite.resolve_source_csv` +
      `eval_source.load_canonical_csv` (RTT/NaN filter reused, not re-implemented);
      node sets pinned to `vps.csv` + the answer space exactly as
      `build_bipartite` pins them, with dropped edges counted
- [x] 2026-09-06 Left chain: per-target min over measured VPs of
      `d(VP, tg_seed)` and of `rank(tg_seed | VP)`, both VP ids kept separately
- [x] 2026-09-06 Right chain taken from `eval_source`'s `shortest_ping_vp_*`
      rather than re-minimized over RTT — **design change**, see report.md
- [x] 2026-09-06 Four top-1 flags; context columns `closest_vp_to_tg_km`
      (recomputed observed), `sping_vp_to_tg_km`, `min_inflation`,
      `n_measured_vps`
- [x] 2026-09-06 `target-proximity/<grid>-<res>/target_labels.csv` + `meta.json`
      (base rates with cell counts, `zero_variance`, four-flag cross-tab,
      argmin-vs-half-gap counts, §8.2 taxonomy block, implication violations)
- [x] 2026-09-06 `build-proximity: {}` added to all four run configs and
      `template.yaml`; `test_config.py`'s live-CLI validation passes (27)

## Phase 2: The three consumers

- [x] 2026-09-06 **`io.load_sping_vp`** — one resolution of the baseline's VP,
      read by both `classify` and `proximity`; v2's `shortest_ping_*` translated
      to `sping_*` once, on the way across. `topn_accuracy.csv` re-verified
      byte-identical after the switch
- [x] 2026-09-06 **Taxonomy renamed**: `structural_failure` → `geometry_only`,
      `selection_failure` → `selection_miss`, `baseline_already_correct` →
      `selection_hit`. New `proximity.TAXONOMY` constant; README + SCHEMA record
      why v2's `NO_PROXIMITY` vocabulary is not reused
- [x] 2026-09-06 `modules/breakdown.py` → `breakdown-accuracy`:
      `accuracy_by_flag.csv` (per-method 2×2 per flag, cell counts, rate
      difference, φ, `zero_variance`) + `accuracy_by_taxonomy.csv` (the
      partition), at top-1 and top-3
- [x] 2026-09-06 Tautological cell marked `is_tautological` in the CSV and named
      in `breakdown_manifest.json`; test pins that it is *not* marked at top-3
- [x] 2026-09-06 `modules/confusion.py` → `confusion-density`:
      `confusion_by_density.csv` (quantile bins on the tg_seed's
      `nearest_seed_km`) + `confusion_pairs.csv` (`pred_seed_neighbour_rank`
      from `seed_mesh_km.csv`, `boundary_margin_km`, all three error columns)
- [x] 2026-09-06 `modules/accuracy_table.py` → `table-accuracy`: CSV + markdown
      render + per-run `dataset_context.csv`; no pooling switch, no inherited
      mixed-setup refusal. Operator table and as7018 table both produced
- [x] 2026-09-06 Config blocks for `breakdown-accuracy` and `confusion-density`
      in all four run configs + `template.yaml`; `test_config.py` passes (27)

## Phase 3: Verification

- [x] 2026-09-06 Unit tests per module: `test_proximity.py` (14),
      `test_breakdown.py` (8), `test_confusion.py` (10),
      `test_accuracy_table.py` (9). Diamond implications pinned on random
      geometry; the half-gap-tighter-than-argmin cell and the two-different-VPs
      case each pinned on constructed geometry
- [x] 2026-09-06 Tautology pinned twice: numerically in `test_proximity.py`
      (against `_seed_distance_frame` on the same input) and structurally
      (`score_shortest_ping` must route through `io.load_sping_vp`)
- [x] 2026-09-06 argmin-vs-half-gap disagreement recorded as a count in
      `meta.json`, never asserted equal
- [x] 2026-09-06 Cross-checked `closest_vp_to_tg_km` against
      `bipartite-graph/target_nodes.csv`'s `nearest_measured_vp_km` on
      **distance** (max diff 0.0000 km) and `n_measured_vps` against
      `degree_to_vp`; no VP id was joined
- [x] 2026-09-06 Full pipeline run on all 4 runs; base rates and zero-variance
      flags recorded in `report.md`
- [x] 2026-09-06 `pytest scripts/analysis/v3/tests/ -q` → **442 passed**

## Deferred (unchanged)

- [ ] Traffic-weighted views — blocked on data (`has_weight: false` everywhere)
- [ ] The VP&TG topology-paired dataset — a `scripts/benchmark/v2/` re-run
- [ ] §8.2 and §8.3 commands
