# CBG Benchmark — Discussion & Working Notes

Scratch/working doc for analyses not yet locked into `paper-flow.md`. Everything here is
candidate material: mechanism story, failure modes, the confidence-tier characterization, the
the tolerance-dividend numbers (computed; promoted to paper-flow), and metrics still under consideration.

Holds everything **from the "Mechanism (the bridge)" point onward** in our 2026-06 discussion;
`paper-flow.md` §6.1 holds the two clean-regime rankings up to that point.

---

## 1. Mechanism — the bridge between the two regimes

The two regimes in §6.1 (in-distribution global; matched-regional) are connected by a single
explanatory axis: **min(VP–TG) distance** (closest-VP distance).

- The in-distribution global run (n=713) already spans the full distance range — some targets
  sit near a VP, many are far. Re-read **sliced by min(VP–TG) distance**, it reproduces the
  whole colocation story (accuracy decays with distance) **without** needing a separate
  mismatched (regional-VP → global-TG) regime, which we dropped for being too noisy to rank.
- The matched-regional regime is then the "what good colocation buys you" endpoint.

So the global run is read **two ways**: (a) as a ranking at full scale (→ §6.1 Regime 1), and
(b) distance-sliced to expose the mechanism (this doc).

**Planned figure:** classification accuracy + confidence-tier breakdown vs. binned closest-VP
distance, per textbook variant. This single curve also delivers:
- *Spotter's effective range* — where it dies relative to distance.
- *Octant's long-range limit* — where bounded-spline extrapolation breaks down.

**TODO:** lock the distance-bin edges for this figure.

---

## 2. Failure analysis & named failure modes

Both modes are surfaced by the answer-space metric (FALLBACK / failures count as inaccurate).

- **Spotter collapse.** The k·σ bands produce constraints that frequently empty the inside-all
  intersection → fallback → counted inaccurate; worst when targets are far (7% on the global
  run). Cross-ref `[[finding_spherical_circle_brittle]]`.
- **Octant long-range limit.** The bounded spline cannot extrapolate beyond the VP cloud →
  degrades on far targets. This is why a regional single-ASN VP fleet fails on out-of-region
  targets.

(Both feed `paper-flow.md` §6.5 once stabilized.)

---

## 3. Characterizing precisely-geolocated targets (three confidence tiers)

Run **primarily on the n=713 in-distribution set** (large enough to regress; sidesteps the
regional small-n problem). Matched-regional used only as a confirmatory overlay.

Per-target tier labels:
- **Tier 1 — accurate:** prediction within R of the truth's centroid (the within-R / point-estimate rule).
- **Tier 2 — correct-but-imprecise:** snaps to the right centroid but is >R away (the "tolerance dividend" targets — right answer, wrong precision).
- **Tier 3 — low-confidence:** snaps to the wrong centroid (or falls back / fails).

**Question to answer:** which target features predict Tier-1 membership?

**Feasibility resolved (2026-06-22): chose the instrument-and-rerun path.** The MTL stage now
persists `participating_vp_ids` — the constraints surviving the redundant-disk filter that
actually decide the region — and the runner joins them back to each VP's RTT and echoed distance
band into a nested `mtl_participants` column of `targets.parquet` (+ `n_mtl_participants`). So the
"participating VPs that decide the intersection" set is now exact, not the LTD-success proxy.
The 6 characterization configs (global ×2, EU ×2, US ×2) were re-run on this instrumentation.

Implemented per-target features (`scripts/analysis/partvp/extract_features.py`):
- **Available geometry (combo-independent):** `avail_min_vp_km` (closest VP over all observed
  VPs), `avail_min_rtt_ms` (shortest-ping signal), `n_obs`.
- **Participating-VP (combo-specific, the deciding set):** `n_part`; `part_{min,mean,med}_dist_km`;
  `part_{min,mean,med}_rtt_ms`; **`part_max_gap_deg`** (max angular gap between consecutive
  participants as seen from the target — large ⇒ one-sided) and **`part_circ_var`** (circular
  variance of bearings — high ⇒ surrounded); **`part_{mean,min}_infl`** (RTT inflation =
  measured / (slope·dist), decoupling congestion from raw distance).
- **Answer-space (target-level):** `truth_centroid_km` (the floor) and
  **`nearest_other_centroid_km`** (truth-centroid isolation — small ⇒ crowded ⇒ easy to
  misclassify; the Tier-1-vs-Tier-2 lever from §3 above).

Analysis (`scripts/analysis/partvp/analyze_tiers.py`) answers two questions per run × family:
Q1 "geolocatable?" (Tier-3 vs Tier-1∪2) and Q2 "precise?" (Tier-1 vs Tier-2 among matched), via
single-feature AUC, per-tier box plots, and depth-3 decision-tree thresholds. Full write-up in
`participating_vp_findings.md`.

**Result (confirmed — see `participating_vp_findings.md`).** A **three-lever model**, by priority:
1. **Proximity to the nearest VP is the universal primary driver** of *both* geolocatability and
   precision (Tier-1), in every regime. Operating point: nearest-VP **RTT ≲ 5–7 ms / distance ≲
   10–40 km** ⇒ Tier-1 (decision-tree primary split in global *and* regional runs; pooled
   single-feature AUC ranks min-RTT/min-dist #1 for both questions). Globally CBG reaches Tier-1
   only by collapsing onto a single near VP (`n_part`≈1 ≈ shortest-ping); regionally the in-country
   fleet gives 7–20 participating VPs *and* a close one (genuine multilateration).
2. **Answer-space isolation owns Tier-2 vs Tier-3 — but only at scale.** `nearest_other_centroid_km`
   AUC **0.64–0.68** globally (isolated truth-centroid ⇒ a coarse estimate still snaps right); weak
   in the small in-country answer spaces.
3. **Angular surround is a regime-gated secondary lever.** `part_circ_var` (surrounded) reaches AUC
   **0.53–0.82** for "geolocatable?" in the matched-regional runs (DE highest), invisible globally
   (degenerate single-VP Tier-1). Confirms "surrounded helps" but always behind proximity.

**Inward/outward natural experiment (EU fleets → all-EU, `participating_vp_findings.md` §4.6).**
Single-region fleets (AS3209 DE-central, AS3215 FR-western) geolocating all 415 EU anchors give a
*dramatic* raw split — octant Tier-1 **31–32% inward** (target inside the VP hull, `avail_max_gap_deg
< 180°`) vs **2–5% outward** — but inward targets are also ~70× closer, so the split is largely a
proximity proxy. **Distance-controlled** (matched closest-VP band 20–80 km), the gap shrinks to
**14% vs 8%** (DE) / 12% vs 0% (FR): a real but **secondary ~1.5–2× angular residual**. Confirms
lever 3 is genuine yet subordinate to proximity — there is essentially no Tier-1 beyond ~50 km
regardless of surround.

Matched-regional fleets roughly **double** the precise-Tier-1 share and **halve** Tier-3 vs global.
Spotter collapses everywhere (Tier-1 ≈ 0–3%).

**Refined deliverable sentence:** *"A target is geolocated precisely iff a VP is close to it (RTT ≲
5–7 ms); among far targets, it still lands on the right answer iff its candidate site is isolated;
being angularly surrounded helps only as a second-order tiebreaker."*

---

## 4. The tolerance dividend (DONE — promoted to paper-flow §1.5/§5.4/§6.1)

Definition: the gap between **same-centroid accuracy** (classification) and the **within-R rate**
(point-estimate scoring) — the share of targets that land on the right answer-space cell despite
being >R off (the Tier-2 band). Measures how much the bounded answer space forgives bounded
coordinate error — the §1.2 thesis / contribution #3, made into a number.
Computed by `scripts/analysis/partvp/tolerance_dividend.py` → `analysis/tolerance_dividend.csv`.

**It is a headline-sized, 25–31-point story in the operator regime — not a footnote.** Per-variant
dividend (absolute pp / relative = share of *correct* answers that are tolerance wins):

| regime | vanilla | million-scale | octant | spotter |
| --- | --- | --- | --- | --- |
| Global (as16509) | +6.3 pp / 26% | +1.8 pp / 8% | **+10.8 pp / 43%** | +6.6 pp / 94%* |
| Matched US (as7018) | **+31.3 pp / 68%** | +19.8 pp / 45% | +27.1 pp / 53% | +6.3 pp / 100%* |
| Matched US (as7922) | +25.0 pp / 71% | +24.0 pp / 61% | +29.2 pp / 53% | +12.5 pp / 80%* |

(*spotter relative is ~1 on a collapsed base — not meaningful.)

Two findings worth carrying:
1. **Biggest exactly where the paper lives.** In the matched-regional operator regime, **53–71% of
   all correct answers are >R-off-but-right-cell** — invisible to a coordinate metric. Strongest
   justification for the dual-evaluation contribution.
2. **The metric reorders the variants.** Global within-R: million > vanilla > **octant (last)**;
   same-centroid: **octant (first)** > vanilla > million. Octant relies most on the dividend,
   million-scale least (nails-or-misses-the-cell). The dividend is a per-variant fingerprint.

**Caveat (carry wherever headlined):** the dividend is a property of (variant × answer-space
granularity) — it grows with the cluster radius R (fixed at 50 km here), so it is not a property of
the variant alone.

---

## 5. Metrics still under consideration

*(Scratch — user is weighing additional metrics here. Add candidates below.)*

- [ ] (placeholder)

---

## 6. Operator-facing confidence model — what's knowable at inference (2026-06-22 discussion)

Follow-up to §3. The three-lever model in `participating_vp_findings.md` characterizes tiers
using **truth-anchored** features (distance/bearing computed from the target's real lat/lon). An
operator geolocating an *unknown* target does not have the truth, so the open question is: **from
the RTT measurements alone (plus standard priors), can the operator predict the confidence tier —
in particular flag "no CBG variant will get this right"?** Reframed: a per-target confidence model
yielding the three-tier label from inference-observable features.

### 6.1 The observability split
Partition the §3 features by what's actually available at inference:
- **Observable pre-prediction (RTT only):** `avail_min_rtt_ms` (shortest ping), `n_obs`. This is
  almost the entire raw-measurement signal. Good news: `avail_min_rtt_ms` is the **#1 Q1 driver**
  (pooled mean |AUC−0.5| = 0.26), so **min-RTT alone is a deployable Tier-1 gate** (RTT ≲ 5–7 ms
  ⇒ high confidence). This is the operator's answer to "which metric flags a confident prediction."
- **Observable post-prediction (anchor features on the *predicted* location):** once CBG emits a
  point, recompute `part_min_dist_km`, `n_part`, `part_circ_var`, RTT inflation, and the predicted
  centroid's `nearest_other_centroid_km` — the answer-space / cluster map is a **known fixed input**,
  not truth-dependent, so isolation of the *predicted* centroid is computable.
- **Not observable (truth-anchored):** the report's distance/bearing-to-truth features — including
  the **inward/outward** label of §4.6 (`avail_max_gap_deg` is computed from the target's true
  location). Its inference-time **proxy** is the angular spread of VPs around the *predicted* point
  (or whether the predicted point / constraint region sits inside the VP convex hull) — computable
  without truth and worth adding to the post-prediction feature set, though §4.6 warns its
  independent (distance-controlled) signal is modest.

### 6.2 The residual blind spot
Per `participating_vp_findings.md` §3.2, the Tier-2-vs-Tier-3 split is driven **only** by
answer-space isolation; VP distance/RTT carry *zero* signal there (both tiers are far-from-VP and
look identical in the measurements). So from RTT alone the operator can confidently flag Tier-1 and
confidently flag "no near VP", but **cannot** separate recoverable-far (Tier-2) from hopeless-far
(Tier-3) — that's a property of the candidate-site layout, not the measurement.

### 6.3 The observable lever that *does* work: MTL-region vs. answer-space overlap (measured 2026-06-22)
The landmass-filter idea (earlier draft) was **dropped** — the answer space already encodes the
spatial constraint, so rather than bolt on an external plausibility gate we ask a question that
reuses CBG's *own* geometry and is fully observable at inference: **how many answer-space cluster
cells does the MTL feasible region land in?** A region inside exactly one cell is unambiguous; one
straddling many cells, or touching none, is not — no ground truth needed. This replaces the
truth-anchored isolation lever (§6.2) with an observable one.

**Method** (`scripts/analysis/partvp/region_confidence.py`, results
`analysis/region_confidence.csv`). The MTL region is reconstructed offline from the persisted
`mtl_participants` annuli via `compute_feasible_region_unweighted`; `n_hit` = distinct cluster disks
(uniform **R=50 km**) the region overlaps (sampled-point + haversine test); `d_hub` = point-estimate
→ nearest centroid. Confidence levels (priority order, all observable):
- **L1** highest: `n_hit == 1` (region in exactly one cell, regardless of `d_hub`).
- **L2** high: `n_hit > 1` and `d_hub < R`.
- **L3** mid: `n_hit > 1` and `d_hub ≥ R`.
- **L0** low/fail: `n_hit == 0` (empty / no overlap) or FALLBACK/ERROR. (Truth splits this only in
  validation: "snaps right anyway" vs fail — the user's cases 4/5; observably one bucket.)

**Result — L1 is a real, deployable high-confidence flag; the rest are not reliably high.**
P(correct centroid | level), weighted by n (global / matched-regional):

| combo | L1 | L2 | L3 | L0 | L1 coverage (share of all correct) |
| --- | --- | --- | --- | --- | --- |
| vanilla_cbg | **0.85 / 0.91** | 0.20 / 0.21 | 0.09 / 0.31 | 0.26 / 0.10 | 0.41 / 0.56 |
| million_scale_cbg | **0.70 / 0.83** | 0.28 / 0.63 | 0.08 / 0.28 | 0.00 / 0.00 | 0.28 / 0.06 |
| octant_cbg | **0.71 / 0.66** | 0.22 / 0.15 | 0.08 / 0.14 | 0.29 / 0.51 | 0.33 / 0.28 |
| spotter_cbg | 0.16 / 0.20 | 0.07 / – | 0.05 / 0.00 | 0.07 / 0.16 | 0.08 / 0.03 |

Takeaways, in order of importance:
1. **L1 (single-cell region) is the operator's "trust this" signal** — precision **0.66–0.91** for the
   three real CBG variants, capturing **~30–56%** of all correct answers (vanilla). Fully observable.
   Spotter is the structural exception (0.16–0.20): its wide bands never isolate a single cell.
2. **The user's case-2 (L2) intuition only half-holds.** Multi-cell region + point-in-hub is **not**
   high-confidence globally (~0.20–0.28): region ambiguity dominates, and the point snaps to the wrong
   hub ~80% of the time. It is moderate only for `million_scale` regionally (0.63), whose larger
   regions rarely reach L1. So the clean operating point is **L1, not L1∪L2**.
3. **L3 (region present but point far from any hub) ≈ low** everywhere (0.08–0.31, mostly Tier-3) —
   confirms "far point ⇒ unreliable."
4. **L0 ("no intersection") is *not* uniformly failure — it depends on the variant's fallback.**
   Spotter L0 is pure collapse (~0.07–0.16); octant's centroid fallback still snaps right **0.29
   (global) / 0.51 (regional)** of the time — the user's "case 4" (no intersection yet correct) is
   real and common for octant. vanilla sits between.

**Synthesis.** Region-overlap *is* the inference-observable confidence lever §6.2 said was missing —
but only at the top: **L1 is reliably high; L2 is not; L0's value is variant-dependent.** The
far-target ambiguity §6.2 flagged remains genuinely hard (L2/L3 ≈ 0.1–0.3). Stability: level
assignment is 94% stable between n=300 and n=1200 samples; since `n_hit` only grows with sampling,
reported L1 precision is a mild **lower bound**.

### 6.4 Model feasibility
A cross-validated 3-tier confidence classifier is feasible from the existing labeled rows
(713 × combos global + regional), now with a **proven observable feature**: the §6.3 region-overlap
level (`n_hit`, `d_hub`), alongside `avail_min_rtt_ms` and the §6.1 prediction-anchored geometry.
Expected shape: strong, calibrated **L1/Tier-1 detection** (region-overlap + min-RTT); far-target
Tier-2/3 separation remains the hard part (L2/L3 precision ~0.1–0.3). Current §3 decision trees are
**in-sample** — a real model must use held-out folds. Scope decision: per-variant vs. an ensemble
"will *any* variant be correct?" model (closer to the "none will be correct" framing). **Deferred**
for now; §6.3 produces and validates the core feature it would consume.

**Cross-refs:** classification-over-known-answer-space framing `[[project_answer_space_clustering]]`;
operator airport metric `[[project_airport_eval_metric]]`; empty-region collapse
`[[finding_spherical_circle_brittle]]`.

---

## 7. §7 "when-to-use-what" — assertions to quantify

Two existing §7 bullets can be upgraded from assertion to quantified once the above lands:
- "Loses to shortest-ping when VPs are globally dispersed" → backed by 23.4% baseline vs.
  23–25% CBG (global run).
- "Loses when the target is far from every VP" → backed by the distance-sliced curve (§1 here).

---

## Open items carried from the discussion

- [x] Tolerance-dividend numbers per variant — DONE (§4), promoted to paper-flow §1.5/§5.4/§6.1.
- [ ] Small-n regional fix — pooling ASNs vs. bootstrap CIs (deferred decision).
- [ ] Lock distance-bin edges for the §6.1 mechanism figure (§1).
- [x] ~~Measure geographic-feasibility-filter yield~~ — **dropped** (landmass idea superseded by the
      region-overlap lever, §6.3). Done 2026-06-22: L1 = region-in-one-cell is the observable
      high-confidence flag (precision 0.66–0.91 ex-spotter); L2 not reliably high; L0 variant-dependent.
- [ ] Build cross-validated 3-tier confidence model on observable features incl. the §6.3
      region-overlap level (`n_hit`, `d_hub`) + min-RTT (§6.4); decide per-variant vs. ensemble scope.
