# v4 Evaluation Backing Scripts — Todo

## Phase 0: Setup & Discovery
- [x] Confirm the canonical site key (2026-09-23). No region id exists —
      `*.mainland.sanitized.csv` has only vp/target coords, `vp_asn`,
      `vp_country`, `rtt_ms`; `target_city`/`target_asn` are unrecoverable.
      Key is `(run_id, target_lat, target_lon)` @ 6 dp, stable 2–6 dp.
      Verified 20/22/23 per run = 65 — but **43 distinct coordinates** pooled
      (7 in all three runs, 8 in two). Run id is in the key because operators
      geolocate ASN by ASN.
- [ ] Decide the fate of `nearest_seed_id_retired` — promote to
      `nearest_seed_id` and rewrite the retirement note to scope it to the
      *unbounded rule*, not the column.
- [x] Vocabulary test landed for the T1 surface (2026-09-23):
      `TestNeverSaysVoronoi` scans emitted columns, manifest values and both
      module docstrings. Extend it to each new module as it lands.

## Phase 1: Implementation
- [x] `modules/sites.py` (2026-09-23) — `site_key`, `site_ids`,
      `site_diagnostics`, `solved_mask` re-export. `map_answer_space`'s
      duplicate spelling now calls into it, so the drift is closed rather than
      widened.
- [ ] `modules/recovery.py` + `@app.command("recovery")` — per-method ring0,
      +ring1, +ring2 recovery; per-ring agreement; **measured ring bound**
      (max/p95 `error_km` per ring) into the manifest.
- [ ] `modules/significance.py` + `@app.command("significance")` — ICC, design
      effect, effective n, naive vs clustered CI, paired site-level tests for
      all method pairs at all rungs; clustered bootstrap for the CIs.
      **Scope widened 2026-09-23: must also cover the §1 error percentiles**,
      not just ring-0 rates — see report.md Finding B.
- [ ] `modules/site_stability.py` + `@app.command("site-stability")` — per-site
      all/some/none profile, micro vs macro rate, within-site unanimity.
- [x] Class-collapse table (2026-09-23) — `modules/class_collapse.py` +
      `class-collapse` command, wired into `create_analysis_artifacts.sh`.
      Reproduces §3 exactly: 65 sites, 52/60/63/63 classes, merges 25→12 (one
      holds 3) / 10→5 / 4→2 / 4→2. Manifest publishes the unioned count
      (26/33/38/40) and the per-run quantizer floor as measurements.
- [ ] Figures: recovery stacked bars and site-stability profile, reusing the
      validated palette and the manifest conventions in
      `modules/figure_outcome_bars.py`.
- [ ] Wire all four commands into `create_analysis_artifacts.sh` (mesh arm).

## Phase 2: Verification
- [ ] Unit tests mirroring `tests/test_classify.py`: a FALLBACK row must not
      reach a numerator; the Arctic-style row must be `-1` and excluded from
      recovery at every rung.
- [ ] Regression test on real runs pinning the session's numbers: ring-1
      agreement 36.4/57.4/69.2/89.9 across rungs; Spotter nside-128 ring0 0.0
      with 100%/95% ring-1/ring-2 agreement; micro-macro gap <= 0.3 pp.
- [ ] Confirm the published ring bound holds on all three meshes and both arms
      (ring-1 <= ~2.9x nominal cell, ring-2 <= ~5.3x).
- [ ] Verify significance output reproduces the paired p-values
      (Octant-Hull vs Shortest-Ping n.s. at 16/32/128; vs Vanilla/Spotter ***).

## Phase 3: Paper wiring
- [ ] Regenerate every table marked **[SCRIPT NEEDED]** in
      `cbg-benchmark-paper/sections/evaluation.md` and strike the markers.
      (§3 struck 2026-09-23; §4, §5, §6 remain.)
- [ ] **§1/§2b repair (Finding B).** Restore the six suppressed cells; delete
      the false "sub-2 km at p5 for every method"; re-ground §2b's crossing
      argument on the tail, which survives clustering, not the median, which
      does not.
- [ ] Replace the stale figures listed in §9 of the companion
      (`cls-acc-top1*.png`, `target-resolvability-top1.png`).
- [ ] Add the two new figures (ring-bounded recovery, per-site stability) to
      `figs/` and cite them from `evaluation.tex`.
- [ ] Cut or re-run the TF column and the MP-vs-TF spine (companion §8 item 2).
