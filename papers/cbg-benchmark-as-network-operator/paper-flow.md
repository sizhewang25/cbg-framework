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

## 5. Dataset Prior & Experimental Setup

The paper's evaluation starts with a data-composition prior, before any CBG result: not every
RTT corpus is equally suitable for operator-scoped, per-ASN CBG.

### 5.1 Dataset characteristics — public RIPE vs. proprietary operator data

We use two datasets for different roles.

- **Public RIPE-based dataset:** VPs are RIPE Atlas probes grouped into operator-like ASN fleets;
  targets are RIPE Atlas anchors. This is public and reproducible, but noisy for the operator
  question because target anchors are sparse once we condition on both region and ASN. In the US,
  for example, no target ASN contributes more than 7 anchors, so RIPE cannot cleanly emulate a
  single target ASN with many regional sites.
- **Proprietary operator dataset:** VPs are one operator's user-plane / mobile-core sites, and
  targets are 200+ IPs from the same target ASN across 20+ US clusters. This is the cleaner
  operator regime: RTTs from one operator ASN to one target ASN tend to share the same peering,
  backbone, and interconnection topology, which is exactly the structure CBG's latency-to-distance
  calibration needs.

This distinction is a prior for interpreting results. RIPE is the public stress test and
reproducibility substrate; the proprietary dataset is the deployment-shaped target regime. Where
the two disagree, the first explanation to check is data composition, not only algorithm choice.

### 5.2 Data processing and setup construction

The benchmark turns both datasets into the same bounded-answer-space task.

1. **Target clustering.** Ground-truth target coordinates are clustered into answer cells with
   radius `R = 50 km`. Each cell has a centroid. A CBG prediction is scored by snapping it to the
   nearest centroid and checking whether it matches the truth's centroid.
2. **VP grouping.** VPs are grouped by ASN, then by geographic deployment scope. The public RIPE
   setups currently cover global, US, and Europe views; the proprietary setup is a US single-target-ASN
   operator view.
3. **Geometry visualization.** Each setup gets a map showing VP locations, target centroids/cells,
   and VP-target geometry. These maps establish whether the setup is global, colocated regional, or
   sparse before the accuracy table is interpreted.
4. **K-fold evaluation.** For RIPE, anchors are split by fold: `(K-1)/K` folds provide labeled
   RTT-distance pairs for CBG calibration, and the held-out fold provides targets. The VP fleet is
   shared across folds; target leakage is avoided on the anchor side.

### 5.3 Validation primitive — Speed-of-internet intersection

For any claimed target location (from Geofeed / rDNS / operator record), the **physics-validation
rule** is:

1. For each VP `v`, compute the speed-of-internet circle:
   `C_v = { x : dist(v, x) ≤ c · 2/3 · RTT(v, T) }`.
2. The claim is **physics-consistent** iff it lies inside at least `k` of `N` VPs' SoI circles.
   Default `k = N` (unanimous); relax to majority to tolerate congested outlier VPs.
3. **Tradeoff — physics-consistent ≠ uniquely identified.** Multiple bounded-answer-space candidates
   can fall inside the intersection; the Geofeed/rDNS claim is consistent with all of them.

The primitive is calibration-free, runs entirely on operator-owned VPs, and bootstraps the
Anchored-track training pipeline.

### 5.4 Metrics

- **Classification accuracy:** same-centroid accuracy over the bounded answer space, always reported
  against a shortest-ping VP baseline.
- **Coordinate error:** p5 / p25 / p50 / p75 / p95 error-to-truth-centroid percentiles, separately
  for correct and incorrect matches where useful.
- **Dual scoring rules + tolerance dividend.** The coordinate rule is **within-R** (prediction within
  `R` km of the truth centroid); the classification rule is **same-centroid accuracy**. Their gap,
  `same_centroid_acc − within_r`, measures how often the finite answer space accepts a bounded
  coordinate error because the prediction still lands in the right answer cell.
- **Failure/success stats:** failed intersections, fallbacks/give-ups, wrong-cell matches, and the
  number/geometry of participating VPs that survive multilateration.
- **Fleet-geometry diagnostics:** raw VP proximity (`fleet_abs_km`) and the
  target-distinguishable VP margin (`d(C,N)/2 − fleet_abs_km`).
- **Practicality:** runtime and memory by phase.

---

## 6. Results

The first results section is deliberately narrow: four textbook CBG variants, a coordinate-error
stress test for VP/target span mismatch, setup-local classification accuracy where the answer spaces
are comparable within setup, and the two fleet-geometry primitives that explain where CBG fails.
Phase ablations and production-cost results come after this core accuracy story.

### 6.1 VP span must match target span

Before classification, we use coordinate error to compare the same global RIPE target set
(713 targets pooled across the held-out folds) under three VP spans: global VPs, US-only VPs, and
Europe-only VPs. This is not a variant-ranking table; it is the sanity check that CBG cannot solve
long-distance target prediction when the VP fleet is regional.

| VP span → target span | Fleets | Best textbook p50 error | Best textbook p75 error | Textbook p50 range | Main read |
| --- | --- | ---: | ---: | ---: | --- |
| Global → global | AS16509, AS31898 | 322–417 km | 918–1082 km | 322–965 km | Global VPs give the only viable global-target regime. |
| US → global | AS7018, AS7922 | 2760–2820 km | 4085–4154 km | 2760–6882 km | A US fleet cannot geolocate global targets well. |
| Europe → global | AS3209, AS3215 | 1214–1232 km | 5684–5835 km | 1214–2158 km | Europe looks less bad at p50 because many targets are in/near Europe; p75 exposes the global spillover failure. |

**Read:** the VP deployment has to cover the span of the target population. A regional VP fleet can
be useful for regional targets, but using it on global targets turns CBG into long-distance
extrapolation. No textbook variant fixes that geometry.

### 6.2 Four textbook CBG variants on setup-local answer spaces

After establishing the span-mismatch failure, we switch back to classification. All rows use
`R = 50 km` answer-space clustering. The percentages are setup-local because the answer space
differs by row: global uses 713 targets / 257 clusters, US uses 96 targets / 32 clusters, and Europe
uses 415 targets / 120 clusters.

| Setup | Targets / clusters | Shortest ping | Vanilla | Million-scale | Octant | Spotter | Main read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Global AS16509 | 713 / 257 | 23.4% | 24.3% | 23.0% | **25.2%** | 7.0% | Octant/Vanilla only barely beat baseline; Spotter collapses. |
| Global AS31898 | 713 / 257 | 20.6% | **25.0%** | 22.4% | **25.0%** | 7.0% | Vanilla/Octant lead; Million-scale is modest. |
| US AS7018 | 96 / 32 | 39.6% | 45.8% | 43.8% | **51.0%** | 6.3% | Regional colocation lets CBG beat baseline. |
| US AS7922 | 96 / 32 | 54.2% | 35.4% | 39.6% | **55.2%** | 15.6% | Baseline is already strong; only Octant clears it. |
| Europe AS3209 | 415 / 120 | 10.8% | 5.5% | 10.4% | **12.0%** | 2.4% | Octant gives a small setup-local lift. |
| Europe AS3215 | 415 / 120 | 5.8% | 4.6% | 6.5% | **12.0%** | 1.4% | Octant is the only clear winner. |

**Read:** CBG's rank is not global. Octant is the most consistent textbook variant in this bounded
answer-space view, but the margin over shortest ping depends strongly on VP-target geometry. Spotter
is a structural outlier across these setups.

**The metric choice reorders the variants (tolerance dividend, §5.4).** On Global AS16509, ranking
by the coordinate rule (within-R) gives Million-scale (21.2%) > Vanilla (18.0%) > Octant (14.5%);
ranking by same-centroid accuracy flips the top to Octant (25.2%) > Vanilla (24.3%) > Million-scale
(23.0%). Octant is worse as a point estimator but better at choosing the right bounded answer cell.

### 6.3 Success and failure profiles

For each setup × variant, the evaluation reports:

- same-centroid accuracy, within-R rate, and accuracy delta from shortest ping;
- error-to-centroid CDFs for all rows, matched rows, and mismatched rows;
- `SUCCESS`, `FALLBACK` / give-up, and wrong-cell counts;
- participating VP count and geometry for the constraints that actually survive multilateration.

This is the bridge from rank tables to mechanism: a variant that wins by many wrong-but-near
coordinates is different from a variant that wins by isolating the correct answer cell, and a variant
that fails by empty intersection is different from one that confidently selects the wrong cell.

### 6.4 Feature correlation — VP proximity and target-distinguishable margin

We keep two complementary fleet-geometry primitives:

- **Raw VP proximity:** `fleet_abs_km = d(V*, C)`, where `C` is the truth centroid and `V*` is the
  closest available VP to that centroid.
- **Target-distinguishable VP margin:** `target_distinguishable_vp_margin_km = d(C,N)/2 -
  fleet_abs_km`, where `N` is the nearest competing centroid. Positive margin means at least one VP
  is inside the loose half-gap bound and is guaranteed, by triangle inequality, to favor the truth
  centroid over the nearest competing answer cell.

The pair separates two questions: *how far is the fleet from the target?* and *is that close enough
for this target's local answer-space density?* We do not learn a universal km threshold across
setups; fixed km cuts are descriptive only.

| Setup | Median `fleet_abs_km` | Median target-distinguishable distance | Median margin | Missing target-distinguishing VP |
| --- | ---: | ---: | ---: | ---: |
| Global | 348.2 km | 71.0 km | -313.3 km | 77.0% |
| US | 35.1 km | 130.9 km | 62.0 km | 43.8% |
| Europe | 259.2 km | 45.3 km | -214.6 km | 88.2% |

Across all four textbook variants and five VP-target setups (5,540 rows), `margin <= 0` covers
**84.4%** of all failures; among rows missing such a VP, **92.6%** fail. It is strongest for
Million-scale: missing a target-distinguishing VP covers **91.8%** of its failures, and the residual
failure rate when such a VP exists is **24.9%**. Spotter remains the exception: even when a
target-distinguishing VP exists, it still fails **91.9%** of the time.

### 6.5 Phase ablation and production cost

This section follows the core accuracy/mechanism story and compares non-textbook phase combinations.
The factual result already measured for practicality is that cost lives in the centroid phase:
`monte_carlo_medoid` takes **190–390 ms** per target versus `geometric_centroid` at **~0.25 ms** and
`boundary_vertex_mean` at **~0.04 ms**. In the global AS16509 run, `octant_cbg_hull_geo` reaches
**30.0%** accuracy at **65 targets/s**, while default `octant_cbg` reaches **25.2%** at **3.2
targets/s**.

The paper should introduce this only after the four foundational variants are understood, because
the early story is about when standard CBG earns its keep, not yet about optimizing the framework.

---

## 7. Discussion Material Deferred From Main Flow

The confidence-tier characterization, inference-time confidence model, anycast handling, ladder-CBG
precision labels, relocation detection, and richer noise models remain in `discussion.md` until the
core dataset/evaluation/proximity story is stable.

---

## 8. Conclusion

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
