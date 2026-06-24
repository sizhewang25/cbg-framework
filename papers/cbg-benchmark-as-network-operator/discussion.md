# CBG Benchmark — Discussion & Working Notes

Scratch/working doc for analyses not yet locked into `paper-flow.md`. Everything here is
candidate material: mechanism story, failure modes, the confidence-tier characterization, the
tolerance-dividend numbers (computed; promoted to paper-flow), and metrics still under consideration.

Holds everything **from the "Mechanism (the bridge)" point onward** in our 2026-06 discussion;
`paper-flow.md` §6.1 now holds the VP/target span-mismatch stress test, and §6.2 holds the
setup-local classification rankings.

---

## 1. Mechanism — the bridge between the two regimes

The in-distribution global and matched-regional regimes in `paper-flow.md` are connected by a small
fleet-geometry primitive set. The old single axis, **min(VP-TG) distance**, was directionally
right but too coarse because it ignored the local density of the answer space. The revised bridge is:

- **Raw VP proximity:** `fleet_abs_km = d(V*, C)`, where `C` is the truth centroid and `V*` is the
  closest available VP to that centroid.
- **Target-distinguishable VP margin:** `target_distinguishable_vp_margin_km = d(C, N)/2 -
  fleet_abs_km`, where `N` is the nearest competing centroid. If the margin is positive, at least
  one VP is inside the loose `d(C,N)/2` bound and is therefore guaranteed, by triangle inequality,
  to favor the truth centroid over its nearest answer-space competitor.

This combo separates two questions that were previously blurred:
1. *How far is the fleet from the target?* (`fleet_abs_km`)
2. *Is that close enough for this target's local answer-space geometry?*
   (`target_distinguishable_vp_margin_km`)

- The in-distribution global run (n=713) already spans the full distance range — some targets
  sit near a VP, many are far. Re-read **sliced by min(VP–TG) distance**, it reproduces the
  whole colocation story (accuracy decays with distance). The separate mismatched
  regional-VP → global-TG runs are therefore not used to rank variants, but they are useful as a
  coordinate-error stress test: they show that none of the textbook variants works well when the VP
  fleet span is narrower than the target span.
- The matched-regional regime is then the "what good colocation buys you" endpoint.

So the global run is read **two ways**: (a) as a ranking at full scale (→ `paper-flow.md` §6.2),
and (b) distance-sliced to expose the mechanism (this doc). The regional→global runs are read only
as the span-mismatch failure case now placed in `paper-flow.md` §6.1.

**Result from the focused failure reassessment** (`scripts/analysis/partvp/assess_vp_proximity_failures.py`,
`analysis_fleet/VP_PROXIMITY_FAILURE_ASSESSMENT.md`): across the 4 textbook variants and 5 VP-target
setups (5,540 rows), missing a target-distinguishing VP (`margin <= 0`) covers **84.4% of all
failures**; among rows missing such a VP, **92.6% fail**. But it is not a complete model: when a
target-distinguishing VP exists, the residual failure rate is still **49.3%**.

The primitive is most explanatory for the variants that actually behave like CBG estimators:
- **Million-scale:** missing target-distinguishing VP covers **91.8%** of failures; residual failure
  with such a VP present falls to **24.9%**. The cleanest setup is **global-global / million-scale**:
  **97.8%** of failures covered, residual failure only **7.3%**.
- **Vanilla and Octant:** broadly consistent with the same story, but with more residual failures.
- **Spotter:** structural exception. Missing VP still covers many failures because the sparse-fleet
  setups are hard, but even when a target-distinguishing VP exists Spotter fails **91.9%** of the
  time. Its failure mode is therefore not primarily fleet proximity.

**Planned figure:** classification accuracy + confidence-tier breakdown vs. binned `fleet_abs_km`,
with an overlay/split for `target_distinguishable_vp_margin_km > 0`, per textbook variant. This
single curve also delivers:
- *Spotter's effective range* — where it dies relative to distance.
- *Octant's long-range limit* — where bounded-spline extrapolation breaks down.
- *The answer-space-normalized VP-proximity gate* — where raw distance is not enough because a
  target in a dense answer-space neighborhood needs a closer VP than an isolated target.

**TODO:** lock the `fleet_abs_km` bin edges and margin split for this figure.

---

## 2. Failure taxonomy (canonical, 2026-06-23)

A prediction is scored against the bounded answer space: **success iff the predicted cell equals the
truth's cell**. We separate **what failed (region geometry)** from **why it failed (root cause)** —
two layers that earlier notes conflated ("containment" was an effect, "RTT inflation" a cause; listing
them side by side was the source of the mess).

### 2.1 Layer A — geometric partition of the MTL feasible region

Every prediction falls into exactly one class, defined **purely by the geometry of the MTL feasible
region `R` relative to the truth's answer-space cell `D_truth`**. `R` is reconstructed offline from
the persisted `mtl_participants` constraints. The discriminating predicate is the **polygon–disk
intersection `R ∩ D_truth`**, not a point-in-region test — because the answer space is not a point:
`D_truth` is the cluster disk of radius `r = 50 km` centred on the truth centroid, the same geometry
used by the evaluation metric.

| Class | Geometric condition | Outcome |
| --- | --- | --- |
| **EMPTY_REGION** | `R = ∅` — constraints unsatisfiable → fallback | always failure |
| **EXCLUSIVE_REGION** | `R ≠ ∅` and `R ∩ D_truth = ∅` — region has no overlap with the truth's answer cell | always failure; no centroid rule can recover |
| **INCLUSIVE_REGION** | `R ≠ ∅` and `R ∩ D_truth ≠ ∅` — region overlaps the truth's answer cell | success iff the centroid lands inside `D_truth`; else misclassification |

These three are a **mutually exclusive, exhaustive geometric partition**. The key distinction:

- **EXCLUSIVE_REGION** is a Phase 1/2 failure: the variant's LTD model or multilateration placed the
  *region itself* outside the right answer cell. No centroid rule or post-processing can fix it.
- **INCLUSIVE_REGION** (misclassified) is a Phase 2/3 failure: the region was geometrically compatible
  with the correct answer (it overlapped `D_truth`), but the centroid selection — shaped by region
  one-sidedness, answer-space density, or centroid rule — landed outside it. The algorithm had the
  information to succeed but failed to resolve it.

Using `R ∩ D_truth` (not `y ∈ R`) keeps the taxonomy **consistent with the evaluation criterion**:
the same cluster radius `r` is used in both the scoring rule and the geometric discriminator.

### 2.2 Layer B — root-cause attribution across the 3 CBG phases

The geometric class is the *effect*; the cause is attributed across the pipeline. The per-VP **signed
distance residual `r_v = d̂_v − d_true(v, target)` (km)** is the primary lever, but it is itself an
output of the variant's LTD model and is surrounded by Phase-2/3 causes.

**Phase 1 — Latency-to-distance (per participating VP).**
- **Band-validity:** `1[d_true ∈ [lo_v, hi_v]]` — does this VP's constraint admit the truth? (disks:
  `lo_v = 0`; Octant/Spotter annuli: both edges bite). The *fraction of participants whose band
  excludes truth* is the direct driver of EMPTY/EXCLUSIVE.
- **Signed residual to nearest band edge (km)** — generalizes `r_v` to annuli; negative = truth excluded.
- **Residual decomposition (measurement vs. model):** split `r_v` into (i) an *inflation* term —
  excess RTT over the propagation floor at `d_true` — and (ii) a *model* term — how the variant's
  envelope/slope/σ maps even a clean RTT to its band. This makes "RTT-distance model per variant" a
  *measured* cause, not an assertion: lets us say a band excluded truth because the low-envelope is
  too tight (model) vs. because RTT was inflated and pushed the annulus outward (measurement). The
  **variant LTD identity** enters here as a categorical cause.

**Phase 2 — Multilateration.**
- **Participant selection:** `n_part`, and *which* VPs the redundant-disk / inside-all filter dropped
  (dropping the wrong constraint is a cause; `[[finding_spherical_circle_brittle]]` = one retained
  tight disk empties the region → EMPTY_REGION).
- **Constraint weights** (Octant/Spotter weighted annulus): an over-weighted biased constraint drags
  the region off the truth → EXCLUSIVE_REGION.
- **Participant geometry:** `part_circ_var`, `part_max_gap_deg`, `part_min_dist` — one-sidedness and
  proximity set region shape/size → the main INCLUSIVE-misclassification lever.
- **Region area** (resolution proxy).

**Phase 3 — Centroid selection.**
- The centroid rule (boundary-vertex-mean / geometric / Monte-Carlo) can resolve the *same* region to
  different cells — a pure INCLUSIVE-misclassification sub-cause. Measure: centroid→truth distance vs.
  region extent.

### 2.3 Causal hypothesis (falsifiable)

| Class | Dominant phase | Primary cause signal |
| --- | --- | --- |
| **EMPTY_REGION** | P1 (+ P2 filter) | ≥1 band excludes `D_truth` (model-tight or inflation-shifted annulus) → no common intersection |
| **EXCLUSIVE_REGION** | P1 + P2 weights | collective band bias / mis-weighting places `R` outside `D_truth`; `R ∩ D_truth = ∅` even though `R ≠ ∅` |
| **INCLUSIVE_REGION** (misclassified) | P2 geometry + P3 centroid | `R ∩ D_truth ≠ ∅` but centroid falls outside `D_truth`: one-sided geometry, answer-space crowding, or centroid rule failure |

This restores the precise role of inflation: it is a Phase-1 *measurement* term that for upper-bound
disks (Vanilla, Million-scale) only loosens constraints → feeds INCLUSIVE-misclassification; for
annuli (Octant, Spotter) it can also shift the band outward and exclude `D_truth` entirely →
additionally feeds EMPTY/EXCLUSIVE. That asymmetry is why the variants fail differently.

**Implementation note (`R ∩ D_truth` test):** reconstruct `R` from the persisted `mtl_participants`
annuli (reuse `compute_feasible_region_unweighted` from `region_confidence.py`); test overlap against
the cluster disk `D_truth = disk(truth_centroid, r=50 km)` via sampled interior points + haversine.
This is the same sampling approach used in `region_confidence.py`; the §8 planar-frame caveat applies
equally, and L1 precision there is already a mild lower bound for the same reason.

### 2.4 Named failure modes restated as root-cause instances

The earlier ad-hoc modes are now instances of the table above:
- **Missing target-distinguishing VP** — a Phase-2 geometry/proximity cause; in proximity-limited
  regimes it drives EXCLUSIVE_REGION and INCLUSIVE-misclassification (the fleet-geometry primitive of
  §1, `[[finding_when_cbg_fails]]`).
- **Spotter collapse** — Phase-1 model cause: wide k·σ bands → EMPTY_REGION
  (`[[finding_spherical_circle_brittle]]`).
- **Octant long-range limit** — Phase-1 model cause: bounded spline cannot extrapolate beyond the VP
  cloud → EXCLUSIVE_REGION / EMPTY_REGION on far targets.

(Feeds `paper-flow.md` §6.3/§6.5 once the per-class counts and cause attribution are computed.)

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

- [x] **Fleet-geometry primitive combo locked for failure explanation:** `fleet_abs_km` +
      `target_distinguishable_vp_margin_km`. Keep fixed-km thresholds descriptive only; use the
      target-specific margin as the primary VP-proximity rule.

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
- "Loses when the operator fleet cannot distinguish the target from its nearest answer-space
  neighbor" → backed by the `fleet_abs_km` + target-distinguishable-margin assessment (§1 here).

---

## Open items carried from the discussion

- [x] Tolerance-dividend numbers per variant — DONE (§4), promoted to paper-flow §1.5/§5.4/§6.1.
- [ ] Small-n regional fix — pooling ASNs vs. bootstrap CIs (deferred decision).
- [ ] Lock `fleet_abs_km` bin edges and the target-distinguishable-margin split for the §6.1
      mechanism figure (§1).
- [x] ~~Measure geographic-feasibility-filter yield~~ — **dropped** (landmass idea superseded by the
      region-overlap lever, §6.3). Done 2026-06-22: L1 = region-in-one-cell is the observable
      high-confidence flag (precision 0.66–0.91 ex-spotter); L2 not reliably high; L0 variant-dependent.
- [ ] Build cross-validated 3-tier confidence model on observable features incl. the §6.3
      region-overlap level (`n_hit`, `d_hub`) + min-RTT (§6.4); decide per-variant vs. ensemble scope.
