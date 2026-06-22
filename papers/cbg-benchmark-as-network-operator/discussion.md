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

**Target deliverable sentence:** *"Precisely-geolocated targets are the ones close to ≥1 VP
**and** angularly surrounded by VPs."* — to be confirmed/refined against the AUC + tree results.

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

## 6. §7 "when-to-use-what" — assertions to quantify

Two existing §7 bullets can be upgraded from assertion to quantified once the above lands:
- "Loses to shortest-ping when VPs are globally dispersed" → backed by 23.4% baseline vs.
  23–25% CBG (global run).
- "Loses when the target is far from every VP" → backed by the distance-sliced curve (§1 here).

---

## Open items carried from the discussion

- [ ] Tolerance-dividend numbers per variant once CSVs land (§4).
- [ ] Small-n regional fix — pooling ASNs vs. bootstrap CIs (deferred decision).
- [ ] Lock distance-bin edges for the §6.1 mechanism figure (§1).
