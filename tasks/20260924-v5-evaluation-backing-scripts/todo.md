# v5 Evaluation Backing Scripts — Todo

## Phase 0: Port to v5
- [ ] Re-run v4 `plot-vp-proximity` for p5/p25/all and confirm the committed CSVs still match the numbers quoted in §1b (`plan.md` Context).
- [ ] Port `v4/modules/vp_proximity.py` to `v5/modules/figure_vp_proximity.py` against v5's `classify`/`cross`/`paths`/`methods`.
- [ ] Carry all 15 tests from `v4/tests/test_vp_proximity.py` across; `TestCohortExcludesUnanswered` is the load-bearing one.
- [ ] Diff the ported module's output against the v4 CSVs cell for cell before building anything on top.
- [ ] Register `plot-vp-proximity` on the v5 CLI and decide `_cross/<combo>/<kind>/` naming for the two new kinds.

## Phase 1: `cohort_overlap`
- [ ] `v5/modules/cohort_overlap.py` reusing the ported `load` / `cohort_frame` unchanged.
- [ ] `set_structure`: union size, degree histogram, pairwise overlap matrix.
- [ ] `against_reference`: shared / answered-not-best / refused over the reference cohort, plus error percentiles on solved rows only.
- [ ] `baseline_margin` with a required `margin_km` (default 1.0) and a docstring saying why zero is wrong.
- [ ] `report-cohort-overlap` CLI command; write tidy CSVs, no figure.

## Phase 2: `vp_ranking`
- [ ] `v5/modules/vp_ranking.py`: implied SoI constraint radius per VP RTT.
- [ ] `far_tail` at configurable thresholds, emitting a distinct-value count beside every raw count.
- [ ] `mechanism_rows`: both VPs, both RTTs, radius, `n_vp`, method error, one row per target.
- [ ] Classify each far-tail row `short-range-unrankable` vs `indirect-routing` from the closest VP's unexplained delay; do not hard-code one mechanism.
- [ ] `report-vp-ranking` CLI command.

## Phase 3: Verification
- [ ] Test: cohorts are never rebuilt locally — a FALLBACK row with a small `error_km` cannot enter any cohort or any reference decomposition.
- [ ] Test: `baseline_margin` at margin 0 scores Shortest-Ping ~100% against itself and at margin 1.0 scores it 0% — pin the floating-point degeneracy.
- [ ] Test: far-tail counts ship distinct-value counts, and a fully-tied cluster reports distinct=1.
- [ ] Test: no output column contains a latitude, longitude or place name.
- [ ] Reproduce all §1b numbers from the committed CSVs; diff against the values recorded in `plan.md` Context.
- [ ] Full v4 and v5 suites green (v4 baseline was 400 passed).

## Phase 4: Paper
- [ ] Delete the `[PENDING MODULE]` block in `sections/evaluation.md` §1b and add real provenance lines.
- [ ] Rewrite `sections/evaluation.tex` line ~34 against the new framing: best cases are different targets, only the baselines' are proximity, the gain is the discarded radius. The old "97 VP-adjacent targets" wording is superseded.
- [ ] Relabel figure legends to the paper's abbreviations (figures print `SoI`/`Octant-Hull`; text uses `SOI`/`OCT-H`).
