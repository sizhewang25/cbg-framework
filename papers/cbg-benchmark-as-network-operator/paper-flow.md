# CBG Benchmark — Paper Flow

**Working title:** Constraint-Based Geolocation: Accuracy × Practicality from Network Operator Perspectives

---

## 1. Introduction

### 1.1 Motivation & the operator's problem

IP geolocation underpins a broad set of network operator decisions: networks steer traffic toward nearby peers, CDNs select edge servers, regulators enforce jurisdictional content rules, and capacity planners forecast regional demands. Internet Service Providers (ISPs), CDN operators, and researchers therefore routinely need to know where arbitrary Internet hosts sit on the globe.

Network operators in particular must geolocate millions of Internet hosts on a recurring basis. Active measurement at that scale is prohibitively expensive, and commercial geolocation services are black-box, costly, and raise privacy concerns. Operators need a measurement-light, in-house alternative they can actually trust.

### 1.2 Definition — operator-scoped IP geolocation is *not* the general problem

The operator setting differs from general-purpose IP geolocation in three structural ways:

- **Free vantage points.** Operators already collect *passive RTT* at core infrastructure across many physical sites — essentially a no-cost fleet of probes.
- **Bounded answer space for high-value targets.** For hypergiants, CDNs, and peering partners, operators *already know the set of possible locations* — because that is where they built the interconnection. An Akamai IP observed at a mobile operator's core doesn't live somewhere arbitrary in continuous lat/long space; it lives at one of N known data-center sites.
- **Evaluation as classification, not coordinate regression.** Because the answer space is finite, success is no longer absolute error distance from a continuous lat/long — it is *classification accuracy over the known candidate set*. A method emits a lat/long estimate; we snap it to the closest known site (e.g., the airport associated with an Akamai POP); the prediction is correct iff that site matches the ground-truth site. The task is *"can the method pick the right site from a known list?"*, not *"how close can it get?"* — easier (finite, tolerant of bounded error) and more useful (matches the operator's actual decision).

### 1.3 Challenges — why existing methods are not enough

A natural reaction: *"If operators already know so much about their own peers, why not just use Geofeed or rDNS lookups for the canonical IPs and call it a day?"* In practice, those canonical signals leave a large, structurally biased residual.

- **Geofeed ingestion is fragmented.** Some operators register Geofeed in the WHOIS database (RIPE/ARIN-style remarks); others self-host the feed at idiosyncratic URLs. There is no single ingestion endpoint — building an in-house pipeline that *finds* the data is itself a substantial engineering effort.
- **Staleness.** Existing studies show that both rDNS hints and Geofeed records drift out of date — operators do not re-publish on every relocation or capacity expansion, so even when the data is reachable it can be wrong.
- **Manipulation.** Both signals are self-reported and can be purposefully misreported to advantage one party at the expense of another — they cannot be relied on as authoritative without external validation.
- **Coverage gaps.** Adoption is uneven. Our measurements show that for entire ASNs — including ones whose traffic operators care about — *no* rDNS or Geofeed signal is available at all. *[TODO: insert headline coverage numbers — % of traffic resolvable by Geofeed/rDNS vs. residual requiring latency-based methods.]*

The coverage gap is the decisive point: when canonical signals are missing, the operator still needs a location estimate. **Latency measurement is the only remaining lever that runs entirely on the operator's own infrastructure and requires no cooperation from the target organization.** Within that lever we draw a distinction prior work conflates:

- **The physical bound** — speed-of-internet (≈⅔ c) per VP — is *not* a statistical estimator. With operator-owned VPs whose locations are ground truth, a claimed location is **physics-validated** iff it lies in the intersection of (≥ k of N) VPs' speed-of-internet circles. This is a calibration-free, physics-grounded validation primitive (see §5.3).
- **Calibrated CBG estimators** — Vanilla, Octant, Spotter — fit per-VP latency-to-distance models from labeled RTT–distance pairs and emit point estimates for IPs without canonical records.

This separation yields **two operator workflows**:

- **Anchored track** — any RDNS or Geofeed coverage exists for the ASN. Speed-of-internet validates the claims; the validated subset becomes labels; calibrated CBG variants train on those labels **per-ASN** (see §1.5 and §6.3); the operator gets *verification* for covered IPs and *prediction* for the uncovered residual.
- **Unanchored track** — zero canonical coverage. Speed-of-internet prediction only, evaluated against the bounded answer space.

> **Tradeoff — physics-validation cannot uniquely identify a site** when the SoI intersection contains multiple bounded-answer-space candidates. Label quality in the Anchored track is bounded by intersection tightness; the empirical intersection-cardinality distribution is reported in §5.3 / §6.4.

Yet existing CBG literature is framed around the general IP-geolocation problem and rarely speaks to what operators actually need to decide — which variant to deploy under which VP placement, at what runtime cost, against what evaluation criterion, and at what training granularity.

### 1.4 Thesis

We present the first cross-variant CBG benchmark from the network-operator perspective, run on **two different datasets**:

- A **proprietary operator dataset** curated from one network operator's user-plane / mobile-core (UP) RTT measurements — VPs are the operator's own UPs, and targets are restricted to a **single ASN** (e.g., a hypergiant the operator peers with), so calibration is naturally per-ASN with no inter-ASN mixing.
- A **public RIPE-based dataset** with RTT across many ASNs — the standard data available to academic researchers, with mixed-ASN measurements.

To keep the comparison fair, we match **VP topology** between the two datasets on a best-effort basis (VP count, geographic spread, pairwise distances) so any accuracy difference reflects *data composition* rather than VP placement.

We decompose existing CBG variants into a unified, composable 3-phase framework; evaluate them under both continuous (error-distance) and discrete (bounded-candidate classification) criteria across both datasets; identify when CBG works and when it doesn't; In addition, we evaluate the runtime and memory consumption of CBG variants to further benchmark their practicality; Lastly, we release an open-source implementation.

### 1.5 Contributions (preview)

1. First CBG phase-level interpretation backed by statistics and case studies, evaluated across **two different datasets** — a proprietary single-ASN operator dataset and a public mixed-ASN RIPE dataset with matched VP topology.
2. Unified, composable CBG framework spanning Vanilla / Million-scale / Octant / Spotter, deployed under a **two-track operator taxonomy** (Anchored / Unanchored) that separates physics-grounded validation from statistical estimation.
3. Dual evaluation — error-distance *and* bounded-candidate classification — exposing scenarios where CBG is "off on lat/long but right on the answer that matters." We quantify this as the **tolerance dividend** (classification accuracy − within-R rate): in the matched-regional operator regime it reaches **25–31 points — i.e. 53–71% of all correct answers are won only by the answer-space tolerance**, invisible to a coordinate-error metric.
4. **Per-ASN calibration regime:** empirical evidence that pooled-ASN training degrades accuracy because calibration captures per-ASN propagation topology; we recommend per-ASN training as the methodological default.
5. New CBG combinations that outperform the originals in specific operator scenarios.
6. Open-source framework so downstream work can be explicit about *which* CBG it uses.

---

## 2. Background & Related Work

### 2.1 The IP geolocation toolkit for operators

Compare foundation techniques against operator constraints (scale, privacy, cost, freshness):


| Method            | Pros                                                                   | Cons                                                                                      |
| ----------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **DNS-based**     | Canonical                                                              | Low coverage for some ASNs · Prone to staleness (needs latency validation)                |
| **Geofeed**       | Canonical                                                              | No adoption by some ASNs · Staleness · Malicious modifications (needs latency validation) |
| **Traceroute**    | Fine-grain                                                             | Heavy measurement overhead · Not all routers/hosts traceable                              |
| **Latency / CBG** | Low overhead · Passive-capable · Physics-bounded · Locates unknown IPs | Mixes propagation + last-mile + congestion · Noisy inflation                              |


> Latency methods double as *validation* for the other three — another reason CBG matters even when other signals are present.

### 2.2 CBG at a glance

Three-phase pipeline:

1. **Latency-to-distance modeling** per VP (train on known TG–VP RTT/distance pairs).
2. **Multilateration** across VPs (build a geometric constraint per VP, intersect).
3. **Centroid selection** within the intersection region.

Inputs/outputs and ground-truth flow follow the diagrams on [[CBG-Benchmark-From-Network-Operators.excalidraw]].

### 2.3 The four foundational variants


| Variant           | LTD (P1)                   | Multilateration (P2)         | Centroid (P3)        |
| ----------------- | -------------------------- | ---------------------------- | -------------------- |
| **Vanilla**       | Low Envelope               | Spherical circle             | Boundary-Vertex Mean |
| **Million-scale** | Speed-of-internet (no fit) | Spherical circle             | Boundary-Vertex Mean |
| **Octant**        | Bounded Hull (or Spline)   | Planar annulus (un/weighted) | Monte Carlo          |
| **Spotter**       | Normal Distribution        | Planar annulus (un/weighted) | Monte Carlo          |


**Speed-of-Internet (used in Million-scale CBG) is the only calibration-free variant** — Vanilla, Octant, and Spotter all require labeled RTT–distance pairs in their original forms. This asymmetry drives the two-track taxonomy in §1.3: the calibration-free variant serves as the validation primitive in both tracks, while the calibration-required variants act as estimators only when labeled data (from Geofeed/rDNS-validated IPs) is available.

---

## 3. Gap Analysis & Research Questions

### 3.1 Gaps in the literature

- **Gap 1:** VP setups in existing works do not reflect operator reality.
- **Gap 2:** Few cross-variant CBG evaluations exist under a *controlled* dataset.
- **Gap 3:** Accuracy results lack interpretation — readers cannot tell *why* a variant wins or loses.
- **Gap 4:** Accuracy alone is insufficient for deployment decisions (runtime, memory, robustness are missing).

### 3.2 Research questions

- **RQ1 — Accuracy under operator-realistic VPs:** Which CBG variants yield the best accuracy at each precision tier (city / state / country / continent) given limited VP deployment?
- **RQ2 — Phase ablation:** How do modifications to each algorithm phase affect accuracy compared to the original variant?
- **RQ3 — Practicality:** Which variants are viable at production scale in terms of runtime and memory?

---

## 4. A Unified Composable CBG Framework

The contribution that makes everything else measurable.

### 4.1 Framework anatomy

- **Phase 1 — Latency-to-distance:** Speed-of-internet · Low Envelope · Bounded Hull · Bounded Spline · Normal Distribution.
- **Phase 2 — Multilateration constraint shape:** Spherical circle (disk) · Planar annulus (unweighted) · Planar annulus (weighted).
- **Phase 3 — Centroid selection:** Boundary-Vertex Mean · Geometric (area-weighted) Centroid · Monte Carlo.

### 4.2 Constraint preprocessing (subroutines that affect P2)

- Constraint filtering · Constraint weighting · Overlapped circle filtering.
- Octant-style weight adjustment on empty intersection.
- Inside-all filter for intersecting constraints.

### 4.3 Fallback paths

- Empty intersection → loosen constraints or fall back to **Shortest Ping** VP.
- Single tighter constraint can null the whole intersection — a documented failure mode worth quantifying.

---

## 5. Dataset & Experimental Setup

### 5.1 Datasets

We benchmark on two different datasets, matched on **VP topology** (VP count, geographic spread, pairwise distance distribution) at best effort so that cross-dataset comparisons isolate the effect of data composition rather than VP placement.

- **Public RIPE-based dataset:** VPs = RIPE Probes (within a single ASN); TGs = RIPE Anchors (across ASNs, excluding the probe ASN). **6 VP setups** for operator realism: 1 US · 1 Europe · 1 Global, each paired with an additional validation set.
- **Proprietary operator dataset:** VPs = the network operator's user planes / mobile cores (UPs); TGs restricted to a **single target ASN** (e.g., a hypergiant the operator peers with); RTT is passively collected at the UPs.

### 5.2 Protocol

- K-fold split per VP: (K-1)/K folds → train (RTT–distance pairs of non-target-fold Anchors); 1/K fold → test (target-fold Anchors).
- Output: predicted location vs. ground truth + intersection geometry artifacts.

### 5.3 Validation primitive — Speed-of-internet intersection

For any claimed target location (from Geofeed / rDNS / operator record), the **physics-validation rule** is:

1. For each VP `v`, compute the speed-of-internet circle: `C_v = { x : dist(v, x) ≤ c · 2/3 · RTT(v, T) }`.
2. The claim is **physics-consistent** iff it lies inside at least `k` of `N` VPs' SoI circles. Default `k = N` (unanimous); relax to majority to tolerate congested outlier VPs.
3. **Tradeoff — physics-consistent ≠ uniquely identified.** Multiple bounded-answer-space candidates can fall inside the intersection; the Geofeed/rDNS claim is consistent with all of them. We report the empirical distribution of intersection cardinality across the bounded answer space — tighter intersections give more discriminative validation.

The primitive is calibration-free, runs entirely on operator-owned VPs, and bootstraps the Anchored-track training pipeline.

### 5.4 Metrics

- **Accuracy:** p5 / p25 / p50 / p75 / p95 error percentiles · classification accuracy at each precision tier · diff from shortest-ping baseline.
- **Dual scoring rules + tolerance dividend.** Two ways to score the same prediction: the *coordinate* rule **within-R** (predicted point ≤ R km of truth) and the *classification* rule **same-centroid accuracy** (prediction snaps to the truth's cell in the bounded answer space). Their gap is the **tolerance dividend** = `same_centroid_acc − within_r` — the share of targets that land on the *right* candidate site despite being >R off (the Tier-2 band; see §6.5b). It measures how much the finite answer space forgives bounded coordinate error. *Note: the dividend is a property of (variant × answer-space granularity) — it grows with cluster radius R, fixed at 50 km here.*
- **Practicality:** Runtime · Memory.
- **Diagnostic:** Intersection Success Rate · **Intersection Cardinality Distribution** · Failure Analysis · Outlier VP Resistance · Intersection Agreement · Latency-to-distance Agreement.

---

## 6. Results

Each RQ is evaluated across **both datasets** (proprietary single-ASN, public mixed-ASN) with VP topology matched at best effort. Cross-dataset agreement points to an effect driven by CBG variant choice; cross-dataset divergence points to an effect driven by data composition.

### 6.1 RQ1 — Variant performance under operator-realistic VPs

We read RQ1 through the bounded-answer-space (classification) lens of §1.2 / §5.4, over the
**four textbook variants** (§2.3). Two clean VP→target regimes anchor the comparison.

**Regime 1 — in-distribution (global VP → global TG).** On the global VP corpus (AS16509,
n=713 targets, 257 centroids at R=50 km), classification accuracy over the four textbook
variants ranks:

| Variant | Same-centroid accuracy |
| ------------- | ---------------------- |
| Octant | 25.2% |
| Vanilla | 24.6% |
| Million-scale | 23.1% |
| Spotter | 7.0% |

against a **shortest-ping-VP baseline of 23.4%** (same-centroid). AS31898 corroborates the
ordering.

> **Key finding.** Under globally dispersed VPs, only Octant and Vanilla clear the trivial
> shortest-ping baseline — and only by ~1–2 points; Million-scale essentially ties it and
> Spotter falls far below. *CBG barely earns its keep when VPs are dispersed.*

**Regime 2 — matched-regional (regional VP → in-region TG).** With a regional single-ASN VP
fleet geolocating in-region targets (e.g. AS7018→US, AS3215→FR), the same four variants jump
to ~40–60% and beat the (higher, ~40%) regional shortest-ping baseline by 12–15 points.
*CBG earns its keep when VPs and targets are colocated.* This regional regime is the public-data
stand-in for the proprietary single-ASN operator setting (§1.4).

> *Note: per-ASN regional target sets are currently small (US n=96, FR n=39); the regional
> family **ranking** is not yet stable and is treated as deferred (see `discussion.md`).*

**The metric choice reorders the variants (tolerance dividend, §5.4).** On the global run, ranking
by the *coordinate* rule (within-R) gives Million-scale (21.2%) > Vanilla (18.0%) > **Octant
(14.5%, last)**; ranking by the *classification* rule (same-centroid) **flips** it to **Octant
(25.3%, first)** > Vanilla (24.3%) > Million-scale (23.0%). Octant is worst on coordinate error but
best on the answer that matters — it relies most on the tolerance dividend (+10.8 pp globally,
+27–29 pp regionally), while Million-scale relies on it least (+1.8 pp globally: its predictions
either nail the cell or miss it entirely). So *which variant wins is metric-dependent*, and the
dividend is itself a per-variant geometric fingerprint.

### 6.2 RQ2 — Phase ablation

- **Disk-based CBG:** Best LTD + Centroid combination over spherical-circle constraints.
- **Annulus-based CBG:** Best LTD + Centroid combination over planar-annulus constraints.
- Per-phase contribution to accuracy.

### 6.3 Training-granularity ablation — per-ASN vs. pooled

- **Per-ASN training:** fit each VP's latency-to-distance model using only the target ASN's RTT–distance pairs.
- **Pooled training:** fit a single per-VP model across all ASNs.
- **Hypothesis:** per-ASN training outperforms pooled because calibration captures ASN-specific propagation topology (peering geometry, congestion patterns, last-mile characteristics).
- **Outcome:** report accuracy gap per CBG variant; recommend per-ASN training as the methodological default. Quantify the minimum #anchors-per-ASN required to make per-ASN training viable.

### 6.4 RQ3 — Practicality at production scale

- Runtime per pipeline phase.
- Memory footprint.
- Trade-off plots: accuracy vs. runtime; accuracy vs. memory.

### 6.5 Failure analysis & root cause

- Empty intersection cases · uninformative low-envelope blow-ups · single bad probe contamination.
- **Multi-candidate intersection cases** — physics-validated but ambiguous (§5.3 tradeoff): how often, and how does the variant choice resolve it?
- Quantitative correlation metrics: VP–TG distance vs. outer radius vs. intersection area.

### 6.5b What characterizes a precisely-geolocated target? (three-lever model)

Per-target characterization over the three confidence tiers (Tier-1 within R of the truth's
centroid; Tier-2 right centroid but >R; Tier-3 wrong/fallback), driven by the new participating-VP
instrumentation (the VPs that survive the MTL filter and decide the region, with their RTT and
echoed distance). Full study: `participating_vp_findings.md`. Headline:

- **Lever 1 — proximity to the nearest VP (universal, primary).** Min RTT / min distance to the
  closest VP is the #1 driver of *both* geolocatability and precision in every regime (pooled
  single-feature AUC #1; decision-tree primary split in global *and* regional runs). Operating
  point: nearest-VP **RTT ≲ 5–7 ms / distance ≲ 10–40 km ⇒ Tier-1**. Globally CBG reaches Tier-1
  only by collapsing onto a single near VP (≈ shortest-ping); regionally it becomes genuine
  multilateration (7–20 participating VPs).
- **Lever 2 — answer-space isolation (Tier-2 vs Tier-3, at scale).** When no VP is near, a coarse
  estimate still snaps to the right candidate iff the truth's centroid is isolated
  (`nearest_other_centroid_km`, global AUC 0.64–0.68); weak in small in-country answer spaces.
- **Lever 3 — angular surround (regime-gated, secondary).** Being ringed by VPs helps only once
  proximity is widely available (regional `part_circ_var` AUC up to 0.63); invisible globally.

Matched-regional fleets **double** the precise-Tier-1 share and **halve** Tier-3 vs global; Spotter
collapses everywhere (Tier-1 ≈ 0–3%). This separates a **measurement-geometry** lever (proximity)
from an **answer-space-geometry** lever (isolation) — a framing prior CBG work lacks.

---

## 7. Practical Insights — When to Use What

The "actionable" section that distinguishes this paper from prior CBG comparisons.

- **CBG is not a silver bullet.**
- **Wins:** when VPs and targets are densely colocated.
- **Loses to shortest-ping** when VPs are globally dispersed (RIPE IPMAP's choice of single-radius reflects this).
- **Loses** when the target is "far away" from every VP.
- Anycast targets → distinct failure mode, worth a sub-section.
- Recommendations per operator scenario (US-only ops, EU-only ops, global ops).

---

## 8. Limitations & Open Questions

- Quantifying the *effective minimum-VP distance range* more rigorously.
- Error-distance inference directly from intersection geometry (status, area, shape).
- **Ladder CBG:** yield a precision tier (city/state/country/continent) alongside the point estimate.
- Anycast geolocation as a first-class problem.
- IP relocation detection (VPN / ISP maintenance / history diff).
- Octant/Spotter denoising under richer noise models.

---

## 9. Conclusion

Restate: composable framework + operator-realistic benchmark + actionable when-to-use-what guidance + open-source release. Position the framework as the substrate future CBG work should plug into so accuracy claims become directly comparable.

---

## Open TODOs (tracked here, not yet placed in a section)

- [ ] Quantify the effective distance range numerically (referenced in §7).
- [ ] Decide whether anycast geolocation is a §7 sub-section or a §8 open problem.
- [ ] Confirm exact VP counts per setup for §5.1 once datasets are finalized.
- [ ] Lock the proprietary-dataset description that legal/operator partner will approve.
- [ ] Identify which "new CBG combination" claim in §1 contribution #5 is strongest — pick the headline.
- [ ] Fill in Geofeed/rDNS coverage numbers in §1.3 (% of traffic covered vs. residual requiring latency-based methods).
- [ ] Operationalize **VP-topology matching** between proprietary and public datasets (criteria: VP count, geographic spread, pairwise distance distribution).
- [ ] Decide whether to **name the target ASN** in the proprietary dataset (§1.4, §5.1) or anonymize for publication — likely contingent on operator/legal approval.
- [ ] Pin down **k/N threshold** for SoI validation (§5.3) — hard cutoff vs. weighted by per-VP RTT confidence vs. intersection-area shrinkage?
- [ ] Define **"traffic-weighted"** operationally (bytes / flows / IPs / RTT samples / BGP-prefix size) for the Anchored-track cutoff in §1.3.
- [ ] Confirm minimum-anchors-per-ASN feasibility for the per-ASN ablation (§6.3) — which ASNs in the dataset have enough Anchors to train independently?