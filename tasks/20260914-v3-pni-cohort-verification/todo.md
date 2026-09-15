# §8.1 PNI Cohort Verification (v3) — Todo

## Phase 0: Region-level statistics (no run) — DONE 2026-09-14
- [x] New `scripts/analysis/v3/modules/places.py` — `assign_region_ids()`, `region_diagnostics()`, `region_level_accuracy()` (clustered bootstrap), `region_rate()`, `fold_region_overlap()` (2026-09-14)
- [x] Emit `region_id` from `build-proximity`; `regions` block added to its `meta.json`; all 11 runs rebuilt (2026-09-14)
- [x] Reproduce F1 — **20 / 22 / 23**, 19.9 / 18.7 / 19.9 rows per region. Count identical at 2–6 dp, so region identity is *exact* coordinate equality, not clustering (2026-09-14)
- [x] Reproduce F2 — **worse than recorded: all three runs, not just as01.** `share_regions_in_all_folds` = 1.0 and min = max = 5 on as01/02/03 (2026-09-14)
- [x] Reproduce F3 — 18 / 22 / 22 occupied cells, margin_km p50 150.5 / 172.5 / 159.7, nearest_seed_km p50 301.0 / 345.0 / 319.4 (2026-09-14)
- [x] Region-level rows: `accuracy_by_flag.csv` (+4 cols), `accuracy_by_taxonomy.csv` (+2), `dataset_context.csv` (+`n_regions`), headline markdown (+`regions` column) (2026-09-14)
- [x] `test_places.py` (8 tests) + 2 region tests in `test_breakdown.py`; full suite **897 passed** (2026-09-14)
- [x] README section "The independent unit is the site, not the IP"; SCHEMA `region_id` entry (2026-09-14)
- [x] Extend to as02 / as03: `breakdown-accuracy` re-run, F1/F2/F3 confirmed, full 36-cell clustered matrix, all 15 pairwise paired bootstraps per run (2026-09-15)
- [x] Document the in-stratum weighting caveat in `region_rate`'s docstring (2026-09-15)
- [ ] **New, unplanned:** chase `region_homogeneity` on the fitted variants — Octant-Spline splits 15 of 23 as03 sites. Same coordinate, same seed, same VP distances, different class. Only the per-address RTT vector varies. Needs its own investigation and probably its own paper subsection
- [ ] Decide the tie convention for Shortest-Ping vs SoI (interval touching exactly 0.000 on all three runs) and state it wherever the comparison is printed

## Phase 1: PNI stack on as01 with the real list (no new code) — DONE 2026-09-14
- [x] `detect-pni-strategy` — verdict `mixed` (target-side plurality `direct_no_pni` 32.6%), but **VP-side `tg_nearest` 88.2%** (2026-09-14)
- [x] `build-pni-graph --strategy argmin --split-csv …` — `sel_pni_is_tg_nearest_share` **59.3%** vs as02's 45.2% (2026-09-14)
- [x] `build-pni-feasibility --decoy-trials 200` (2026-09-14)
- [x] `compare-pni-linearity --holdout-only` — `routing_tg_nearest` r² **0.704** > `routing_selected` 0.686 > air 0.655 (2026-09-14)
- [x] `breakdown-sping-pni --grid h3 -r 4` (2026-09-14)
- [x] Record whether `singleton_share` fell as predicted — **it did not; the prediction was wrong.** Sping-only rose 0.633 → **0.837** (2026-09-14)
- [x] Add `pni_csv` / `peer_asn` to `configs/as01-260728-260802.yaml`; all five commands verified to reproduce byte-equal numbers from `--config` alone (2026-09-14)

## Phase 2: Accuracy split by A
- [ ] New `scripts/analysis/v3/modules/pni_mediation.py`, command `breakdown-pni-mediation`, grid-keyed
- [ ] Condition **A** `pni_in_tg_cell` = `nearest_seed(tg_nearest_pni) == tg_seed` — argmin rule only, no half-gap variant, no distance threshold
- [ ] Report accuracy split by A at target and region level; A=true row's accuracy *is* B's rate (given A, `correct ⟺ B` exactly), so B gets no column
- [ ] Emit `B` as an optional diagnostic column only, for the A=false cancellation cases
- [ ] Reuse `pni.load_pni_sites`, seeds.csv `seed_lat/seed_lon`, `io.load_sping_vp`, and `diagram.common.membership.build_membership` as the only correctness source
- [ ] Self-check, reported not raised: zero `A ∧ B ∧ ¬correct` counterexamples; `has_proximate_sping_vp` ≡ `shortest_ping` top-1
- [ ] Report whether the A=true row reads 1.000 — that is the draft's identity, measured rather than assumed
- [ ] Register in `cli.py` `_COMMAND_MODULES`

## Phase 3: The PNI-co-located cohort (best-case surrogate)
- [ ] Define the cohort as condition A — threshold-free, no km cutoff to defend
- [ ] Score all six variants split by A over the same strata (a trend in all of them is target difficulty, not a sping mechanism)
- [ ] Label every artifact "PNI-co-located cohort"; never "traffic-weighted"
- [ ] Record `flow_filtered: false` in `meta.json` so dev-fixture artifacts self-label
- [ ] Document the best-case framing and the one-directional bias (cohort keeps all flows, so CBG is at its best)
- [ ] Treat all cohort numbers as pipeline validation, not findings — 20 regions, fake cohort
- [ ] Verify the stack would accept a real weighted run unchanged if weights ever land

## Phase 4: Draft figures 1a / 1b — build now, interpret later
- [ ] **1a**: scatter of d(VP,TG) vs RTT per target, weighted stacked on mesh, overall and per ASN
- [ ] **1b**: Pearson r of minRTT with d(VP,TG), by dataset
- [ ] Emit **x-range** and **residual RMSE** beside r so range-restriction attenuation is diagnosable when real weights land
- [ ] Reuse `bipartite.ols` so the tabulated fit is the drawn one
- [ ] Validate the code path against the cohort as a **dev fixture** — its 1a/1b numbers are not interpretable (cohort prunes targets, real weighting prunes flows)
- [ ] **No metric substitution**: `min_inflation` is not a stand-in for 1b at this stage

## Phase 5: State the limits
- [ ] Short subsection: the surrogate tests the consequent (targets at interconnects have these properties), never the antecedent (weighting selects them)
- [ ] Record effective n — 20 regions for as01, 8 in the cohort
- [ ] Delete `PROVISIONAL_WEIGHTED` or keep it explicitly flagged

## Phase 6: Leakage-free folds — DEFERRED (needs a benchmark run)
- [ ] Gate on Phase 3: does F2 move any number?
- [ ] Re-cut stratification by region so no coordinate spans train and test
- [ ] Re-run as01 under a new `run_id` (published tree's folds are pinned by stored stratification)
- [ ] Compare; if accuracy barely moves F2 is a footnote, if it moves it is a §7 protocol finding

## Verification
- [ ] `.venv/bin/python -m pytest scripts/analysis/v3/tests/ -q` green, including `test_places.py`, `test_pni_mediation.py`
- [ ] `.venv` on PATH — Snakemake rules shell out to bare `python`
- [ ] Region-level sanity: region-level accuracy × regions ≈ target-level × targets, within the 20-replica quantization
- [ ] Golden path: `plot-proximity-inflation` and `table-headline` on as01+as02+as03 byte-identical except added region-level rows
- [ ] Update `scripts/analysis/v3/README.md` pipeline steps and output tree; update `SCHEMA.md` with new columns
