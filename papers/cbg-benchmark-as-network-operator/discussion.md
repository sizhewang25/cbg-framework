# CBG Benchmark — Discussion & Working Notes

Scratch/working doc for analyses not yet locked into `paper-flow.md`. Everything here is
candidate material: mechanism story, failure modes, the confidence-tier characterization, the
tolerance-dividend numbers (pending data extraction), and metrics still under consideration.

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

## 4. The tolerance dividend (pending CSV extraction)

Definition: the gap between **same-centroid accuracy** (classification, the blue bars) and the
**within-R rate** (point-estimate scoring, the red diamonds). It quantifies how much the bounded
answer space forgives bounded coordinate error — the §1.2 thesis / contribution #3, made into a
number.

- **Status:** numbers being extracted to CSV (agent in progress). Once available, report the
  per-variant gap across the clean runs and decide whether it is headline-worthy (5-point vs.
  20-point story) before committing §1.2 / contribution #3 framing.
- **`paper-flow.md` §5.4** should then name "classification accuracy (same-centroid)" and
  "within-R rate" as the two scoring rules and define their gap as the *tolerance dividend*.

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

### 6.3 Does the operator's standard prior rescue this? (one-ASN-at-a-time + known sites + landmass)
Operators typically target **one ASN at a time, know the full candidate location set, and can apply
geographic feasibility** (e.g. US targets must fall on US landmass; ocean/out-of-country predictions
are obviously failures). This splits the blind spot into three pieces — and the report has **already
half-tested** this via the matched-regional runs:

| residual Tier-3 case | detectable at inference? |
| --- | --- |
| FALLBACK / ERROR | yes (already) |
| SUCCESS, prediction off-landmass / out-of-region | **yes — NEW, via a geographic-feasibility filter (untested)** |
| SUCCESS, prediction on a plausible *in-region but wrong* candidate | **no — irreducible** (needs truth) |

Two **opposite-signed** mechanisms:
1. **Geographic-feasibility filter — genuinely new, not in any run.** A SUCCESS prediction landing
   off-landmass or out-of-country is a guaranteed failure detectable without truth; our pipeline
   currently counts these as valid SUCCESS. Softer version: if CBG's constraint-intersection region
   doesn't overlap any in-region candidate (or the landmass) → flag low-confidence. **Yield is
   measurable:** of today's ~75% global Tier-3, what fraction of the SUCCESS-but-wrong-centroid ones
   are geographically implausible? That number = the value of the prior.
2. **Isolation lever gets *weaker*, not stronger, when constrained.** The "one ASN + known small
   candidate set" scenario *is* the matched-regional regime. `participating_vp_findings.md` §4.4
   found isolation degrades from global AUC 0.64–0.68 to **~0.3–0.5** in the small in-country answer
   space (US 32 centroids / FR 12) — isolation is a *large*-answer-space phenomenon. So shrinking the
   answer space does **not** rescue far-target Tier-2/3 confidence; per §4.1 the regional benefit
   showed up instead as **more near-VPs** (Tier-1 doubles, Tier-3 halves) — i.e. the proximity lever
   again, which was already observable.

**Honest revised caveat:** the landmass/region prior catches the geographically-implausible failures
for free, but the residual — a confident-looking snap onto the wrong *in-region* candidate — stays
invisible, and shrinking the answer space makes the only lever that addressed it (isolation) noisier.

### 6.4 Model feasibility
A cross-validated 3-tier confidence classifier is feasible from the existing labeled rows
(713 × combos global + regional), restricted to the §6.1 observable / prediction-anchored features
plus a hard geographic-feasibility pre-gate. Expected shape: strong, calibrated **Tier-1 detection**
(min-RTT + pred-anchored proximity); **Tier-2/3 separation only as good as isolation** (global
~0.65, weak regionally). Current §3 decision trees are **in-sample** (train acc only) — the model
must use held-out folds. Scope decision: per-variant confidence vs. an ensemble "will *any* variant
reach Tier-1/2" model (closer to the "none will be correct" framing).

**Cross-refs:** classification-over-known-answer-space framing `[[project_answer_space_clustering]]`;
operator airport metric `[[project_airport_eval_metric]]`.

---

## 7. §7 "when-to-use-what" — assertions to quantify

Two existing §7 bullets can be upgraded from assertion to quantified once the above lands:
- "Loses to shortest-ping when VPs are globally dispersed" → backed by 23.4% baseline vs.
  23–25% CBG (global run).
- "Loses when the target is far from every VP" → backed by the distance-sliced curve (§1 here).

---

## Open items carried from the discussion

- [ ] Tolerance-dividend numbers per variant once CSVs land (§4).
- [ ] Small-n regional fix — pooling ASNs vs. bootstrap CIs (deferred decision).
- [ ] Lock distance-bin edges for the §6.1 mechanism figure (§1).
- [ ] **Measure geographic-feasibility-filter yield** — fraction of Tier-3 (SUCCESS-but-wrong) that
      is off-landmass / out-of-region, i.e. detectable without truth (§6.3). Decides headline vs.
      footnote *before* building the model.
- [ ] Build cross-validated 3-tier confidence model on observable / prediction-anchored features +
      feasibility pre-gate (§6.4); decide per-variant vs. ensemble scope.
