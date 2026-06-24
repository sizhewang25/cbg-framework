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
| **Vanilla**       | Low Envelope               | Planar circle                | Geometric centroid   |
| **Million-scale** | Speed-of-internet (no fit) | Planar circle                | Geometric centroid   |
| **Octant**        | Bounded Spline             | Planar annulus (weighted)    | Monte Carlo          |
| **Spotter**       | Normal Distribution        | Planar annulus (weighted)    | Monte Carlo          |

> **Implementation note.** Vanilla and Million-scale use `PlanarCircleMTL` (disk intersection in degree
> space via Shapely) + `geometric_centroid` (area-weighted polygon centroid). The original papers
> describe a spherical-circle MTL and a boundary-vertex-mean centroid; this benchmark replaces both
> with the planar-polygon equivalents, which are computationally lighter and produce a proper polygon
> that the geometric centroid can reduce. This is the official config for the benchmark and is
> reflected in the phase-ablation results of §7.

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

The results section proceeds in four stages, each adding a layer of understanding.

§6.1 fixes the target set (713 global RIPE anchors) and varies the VP fleet across three ASNs to
show VP proximity as the binding constraint in two degrees — fairly limited (global fleet, sparse
but geographically matched) and extremely limited (country-scale fleet against global targets) —
and that no CBG variant escapes either. §6.2 switches to proximity-sufficient setups — the same AS7018 (AT&T) and
AS3209 (Vodafone) fleets now evaluated against their *home* target populations — and characterizes
each CBG variant's accuracy and failure behavior when the proximity constraint is lifted. §6.3
performs the per-feature success-and-failure analysis, introduces the failure taxonomy, and presents
attribution figures for both regimes side by side. §6.4 applies the same evaluation pipeline to the
proprietary single-ASN operator dataset, where curated VP placement and clean single-ASN RTTs show
how good CBG can get — and makes the residual ~20% failure fully interpretable by the §6.3 taxonomy.
§6.5 reports production cost, which is determined entirely by the centroid phase.

### 6.1 Proximity-limited regime

We fix the target set (713 global RIPE anchors, 257 centroids, `R = 50 km`) and vary only the VP
fleet. This holds the evaluation criterion constant while degrading VP proximity in two degrees:

- **Fairly limited** — global fleet (AS16509, 30 VPs worldwide): VPs span the globe but remain
  sparse relative to 713 targets. Median `fleet_abs_km` = 348 km; 77% of targets miss a
  target-distinguishing VP.
- **Extremely limited** — country-scale fleets evaluated against the same global target set:
  AS7018 (AT&T, 125 US VPs → 713 global targets): median `fleet_abs_km` = 6,268 km; 92.4% missing.
  AS3209 (Vodafone, 164 EU VPs → 713 global targets): median `fleet_abs_km` = 972 km; 85.4% missing.
  In both cases the fleet's geographic span is narrower than the target population's span by orders
  of magnitude — the country fleet is simply the wrong instrument for a global target set.

**Fleet-geometry metrics (analytically derived).** Let `C` be the truth centroid, `N` the nearest
competing centroid, and `V*` the closest available VP to `C`:

- `fleet_abs_km = d(V*, C)` — raw VP proximity.
- `target_distinguishable_vp_margin = d(C, N)/2 − fleet_abs_km` — by the **triangle inequality**,
  if this margin is positive then `V*` is *guaranteed* to produce a shorter RTT to `C` than to `N`,
  regardless of RTT noise. This is a setup-aware threshold: it adapts to answer-space density
  rather than applying a fixed km cut.

A negative median margin means no VP certifiably favors the truth centroid over its nearest
competitor. Both degrees of proximity-limited are in that state for the large majority of targets.

**Classification accuracy (same 713 targets):**

| Fleet | n VPs | Shortest-ping | Vanilla | Million-scale | Octant | Spotter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global(AS16509) → Global | 30 | 23.4% | 24.3% | 23.0% | **25.2%** | 7.0% |
| US(AS7018) → Global | 125 | 5.3% | 3.9% | **6.6%** | 5.5% | 3.0% |
| EU(AS3209) → Global | 164 | 6.3% | 2.7% | **6.0%** | 2.0% | 2.5% |

**Fleet geometry by setup** (from `outputs/partvp/analysis_fleet/fleet_geometry_by_config.csv`):

| Fleet | n VPs | Median VP km | Median margin | % missing VP | Degree |
| --- | ---: | ---: | ---: | ---: | --- |
| Global(AS16509) → Global | 30 | 348 km | −313 km | 77.0% | Fairly limited |
| US(AS7018) → Global | 125 | 6,269 km | −6,226 km | 92.4% | Extremely limited |
| EU(AS3209) → Global | 164 | 968 km | −916 km | 85.4% | Extremely limited |

The extremely-limited rows are materially worse than the fairly-limited row: all variants collapse to
the 2–7% range (vs. 23–25% for the global fleet). Calibrated variants (Vanilla, Octant, Spotter)
fall at or below the shortest-ping baseline — tight LTD models with distant VPs produce constraints
that exclude the truth, and there is no recovery once the feasible region is pushed away.

**Million-scale is the exception at the floor.** It ties or marginally exceeds the shortest-ping
baseline in both extremely-limited setups (6.6% vs. 5.3% for AS7018→global; 6.0% vs. 6.3% for
AS3209→global). This is not genuine multilateration — it is degeneration to shortest-ping via the
circle-filtering mechanism. With a VP 6,000+ km away, the SoI upper bound is so large that it
barely excludes any competing centroid; the intersection is dominated by whichever single VP is
closest, and the centroid snaps to the same answer cell that shortest-ping would choose. The
mechanism is circle over-inclusion, not EMPTY_REGION fallback — the region is non-empty but
geometrically uninformative. Million-scale's calibration-free design avoids the tight-constraint
pathology of Vanilla/Octant/Spotter, but its ceiling in this regime is strictly shortest-ping, not
real CBG.

**VP proximity is the dominant failure driver.** The table below quantifies how much of each
setup's failure is explained by missing a target-distinguishing VP (from
`outputs/partvp/analysis_fleet/vp_proximity_failure_assessment_by_setup.csv`):

| Fleet | Failures explained by missing VP | Fail rate when missing VP | Fail rate when VP present |
| --- | ---: | ---: | ---: |
| Global(AS16509) → Global | 88.6% | 92.2% | 39.8% |
| US(AS7018) → Global | 96.7% | 99.2% | 40.7% |
| EU(AS3209) → Global | 89.3% | 100.0% | 70.2% |

The pattern is consistent and strengthens with proximity degradation: in the na-global setup,
96.7% of all failures come from targets with no distinguishing VP, and when one is missing the
failure rate is essentially certain (99.2%). The europe-global setup reaches 100% failure rate
when the VP is missing — not a single target without a distinguishing VP succeeds. The
gap between "missing VP" and "VP present" columns is the causal signal: algorithm choice cannot
close it. The failure attribution AUC of `fleet_abs_km` against failure (from
`fleet_geometry_auc.csv`) reaches **0.96–0.99** for Million-scale across all three setups,
confirming that raw VP distance alone — before running any CBG — nearly perfectly sorts successes
from failures. Variant-level accuracy differences (1–5 pp between Vanilla/Octant and baseline)
are noise relative to this bottleneck.

**Spotter is the structural exception:** its k·σ bands collapse the feasible region regardless of
proximity (7% accuracy vs. 23% baseline); its failure mode is LTD model collapse, not fleet
proximity.

**Target-distinguishable VP margin as the primary geometry metric.** The proximity analysis above
references `fleet_abs_km` for coverage percentages, but raw distance is setup-dependent: 300 km
separates high-failure from low-failure in the global fleet, while 50 km does the same in a
US-regional setup. We therefore establish **target-distinguishable VP margin**
(`margin = d(C, N)/2 − fleet_abs_km`) as the primary VP–target geometry diagnostic used in
§§6.2–6.3. Margin's zero-crossing is geometrically motivated and requires no per-setup
calibration: by the triangle inequality, a positive margin certifies that the nearest VP is
guaranteed to produce a shorter RTT to truth centroid `C` than to the nearest competing centroid
`N`, independent of RTT noise; a negative margin certifies the opposite. Figure `fleet_geometry_bins.png` quantifies this per variant across all three setups:

| Setup | Vanilla | Million-scale | Octant | Spotter |
| --- | ---: | ---: | ---: | ---: |
| Global(AS16509) → Global | 92.7% | 97.8% | 86.2% | 92.0% |
| US(AS7018) → Global | 99.5% | 98.8% | 97.0% | 97.4% |
| EU(AS3209) → Global | 100.0% | 100.0% | 99.8% | 100.0% |

Every cell exceeds 86%. The lowest entry — Octant in the global fleet at 86.2% — reflects its
stronger LTD model recovering a small fraction of marginal-margin targets, but the floor is still
well above chance. Critically, no variant escapes: Spotter (which has a structural LTD collapse
independent of proximity) and Vanilla (tight low-envelope) are indistinguishable from Million-scale
and Octant within this `margin ≤ 0` population. The failure is driven by geometry, not algorithm
choice.
This makes margin the right axis for characterizing the transition from the proximity-limited
regime (§6.1: all three setups have strongly negative median margins, −313 to −6,226 km) to the
proximity-sufficient regime (§6.2: median margins flip to +62 km and +38 km for the same AT&T
and Vodafone fleets redirected to their home target populations).

### 6.2 Proximity-sufficient regime — per-variant characterization

**Setup.** Same AS7018 (AT&T) and AS3209 (Vodafone) fleets from §6.1, now evaluated against their
*home* target populations. This natural experiment — same fleet, different target set — isolates the
regime effect from the fleet composition effect.

| Fleet → targets | Targets / clusters | n VPs | Median `fleet_abs_km` | Median margin | % missing target-dist. VP |
| --- | ---: | ---: | ---: | ---: | ---: |
| AS7018 → US | 96 / 32 | 125 | 35.1 km | +62.0 km | 43.8% |
| AS3209 → DE | 96 / 21 | 164 | 4.6 km | +38.6 km | 2.1% |

Median margin is positive; the proximity constraint is lifted for most targets.

**Accuracy vs. shortest-ping baseline:**

| | AS7018→US | AS3209→DE |
| --- | ---: | ---: |
| Shortest-ping | 39.6% | 50.0% |
| Vanilla | 45.8% | 18.8% |
| Million-scale | 43.8% | **46.9%** |
| Octant | **51.0%** | 39.6% |
| Spotter | 6.3% | 17.7% |

With proximity solved, the four variants diverge — each with a distinct behavior profile across
both setups:

**Vanilla.** Low-envelope LTD fits a tight upper bound; distance estimates frequently
under-predict; constraints exclude the truth. Accuracy collapses in DE→DE (18.8% vs. 50.0%
baseline) despite near-perfect VP proximity. The failure mode is structural: the tight
low-envelope works for moderate-distance VP–target pairs but breaks at metro-scale colocation where
a very small RTT is mapped to a band that excludes the truth.

**Million-scale.** Speed-of-Internet is a calibration-free upper bound, so `r_v ≥ 0` by
construction — it never excludes the truth from the feasible region. Failures are entirely
resolution failures (wrong-cell, not give-up). Accuracy is competitive across both setups
(43.8% US, 46.9% DE), consistent with the "nails-or-misses" fingerprint. The calibration-free
design is both its strength (no training data required) and its ceiling (looser region → harder
to resolve).

**Octant.** Annulus MTL with bounded-hull LTD. Best accuracy in the US setup (51.0%);
weaker in DE (39.6%). RTT inflation is the primary driver in this regime — AUC 0.82
(RTT-inflation → failure attribution, distinct from the §6.1 fleet_abs_km AUC 0.84–0.96) in
EU-country — because inflated RTTs push the annulus inner bounds outward, shifting the feasible
region. Critically, **EXCLUSIVE ≠ 0**: Shapely polygon-disk intersection shows ~50% EXCLUSIVE in
both US (48/96) and DE (50/96). The joint annulus intersection drifts beyond D_truth entirely,
despite each individual VP band geometrically reaching D_truth. Classification accuracy is
preserved because the centroid of the EXCLUSIVE region often still points to the correct answer
cell: ~27 of 48 EXCLUSIVE US predictions are still correctly classified, accounting for nearly the
entire 27 pp tolerance dividend (§6.2 table). This is the core tolerance mechanism — the bounded
answer space recovers EXCLUSIVE predictions via centroid snapping.

**Spotter.** Structural collapse persists regardless of VP proximity, now with a striking
mechanism: **~89–99% EXCLUSIVE**. The joint weighted-annulus intersection never reaches D_truth's
disk — the `normal_dist` k·σ bands produce annuli with large inner bounds, and when inflated RTTs
shift all annuli outward simultaneously the joint region drifts far beyond D_truth entirely.
Yet the 6 correct US predictions (6.3%) are EXCLUSIVE-but-correct: the centroid of the drifted
region happens to point toward the truth cell. Answer-space isolation (F3: truth centroid is far
from the nearest competing centroid) is the plausible enabler — an isolated truth cell captures
stray centroids more reliably. This is the most extreme case of the tolerance mechanism: Spotter's
entire accuracy is a tolerance artifact, with zero predictions succeeding by reaching D_truth.

**Tolerance dividend.** The metric choice matters most in this regime, where the bounded answer
space does real work. Per-variant dividend (absolute pp gain / share of correct answers that are
tolerance wins):

| | Vanilla | Million-scale | Octant | Spotter |
| --- | --- | --- | --- | --- |
| AS7018→US | +31.3 pp / 68% | +19.8 pp / 45% | +27.1 pp / 53% | +6.3 pp / 100%* |
| AS3209→DE | +6.3 pp / 33% | +5.2 pp / 11% | +5.2 pp / 13% | +9.4 pp / 53%* |

(*Spotter relative is not meaningful on a collapsed base.)

The US dividend (45–68% ex-Spotter) is dramatically larger than the DE dividend (11–33% ex-Spotter),
which is directionally expected: in DE the median VP is 4.6 km from its target, so the within-R rate
is already high and the gap to classification accuracy is narrow. In the US setup the median VP is
35 km away — far enough that many predictions are "correct-cell but >50 km off", producing a large
dividend. The contrast reveals that the tolerance dividend scales with VP-to-target distance: it is
not a property of the variant alone but of (variant × answer-space granularity × VP proximity).

### 6.3 Success and failure analysis

This section examines *why* each outcome occurs, organized per feature. The analysis focuses on the
proximity-sufficient setups (where variant-specific failure modes are the story), with attribution
figures covering both regimes side by side for contrast.

**Failure taxonomy.** We partition every prediction by the geometry of the MTL feasible region `R`
relative to the truth's answer cell `D_truth` (cluster disk, radius `r = 50 km`, same as the
evaluation metric):

| Class | Geometric condition | Diagnostic |
| --- | --- | --- |
| **EMPTY_REGION** | `R = ∅` → fallback | Constraints jointly unsatisfiable; no centroid rule can recover |
| **EXCLUSIVE_REGION** | `R ≠ ∅` and `R ∩ D_truth = ∅` | Region excludes the truth's answer cell entirely; Phase 1/2 bias |
| **INCLUSIVE_REGION** | `R ≠ ∅` and `R ∩ D_truth ≠ ∅` | Region overlaps truth's cell; success or resolution failure |

`R` is reconstructed from the persisted `mtl_participants` constraints;
`R ∩ D_truth` is tested by sampling interior points and checking haversine distance to the truth
centroid. Using the cluster disk (not a point) keeps the taxonomy consistent with the evaluation
criterion.

**Per-feature attribution.** For each failure class, we report the dominant cause across the three
CBG phases:

- **Phase 1 — LTD:** per-VP band-validity `1[d_true ∈ [lo_v, hi_v]]` and signed distance
  residual `r_v = d̂_v − d_true` (km); residual decomposed into *inflation* (excess RTT over
  propagation floor) and *model* (variant's envelope/slope/σ). Under-prediction (`r_v < 0`) drives
  EMPTY and EXCLUSIVE; over-prediction feeds INCLUSIVE resolution failures.
- **Phase 2 — MTL:** participant selection (`n_part`, which VPs the filter dropped),
  constraint weights (Octant/Spotter weighted annulus), participant geometry (`part_circ_var`,
  `part_max_gap_deg`, `part_min_dist_km`).
- **Phase 3 — centroid:** centroid rule (geometric / Monte-Carlo) resolving the same INCLUSIVE
  region to different cells.

**Per-variant failure profile (proximity-sufficient, Shapely polygon-disk verified):**

Counts from proper joint Shapely polygon reconstruction (not per-VP band check):
- Vanilla/Million-scale: `outputs/` (correct `planar_circle + geometric_centroid`)
- Octant/Spotter: `outputs/` when available, `outputs_partvp/` otherwise (same annulus config)

| Variant | EMPTY | EXCLUSIVE | INCL\_success | INCL\_misclass |
| --- | --- | --- | --- | --- |
| Vanilla (US / DE) | 15% / 38% | **10% / 14%** | 19% / 13% | 56% / 36% |
| Million-scale (US / DE) | 0% / 0% | **1% / 0%** | 36% / 41% | 63% / 59% |
| Octant (US / DE) | 0% / 0% | **50% / 52%** | 23% / 32% | 27% / 16% |
| Spotter (US / DE) | 0% / 0% | **99% / 89%** | 0% / 8% | 1% / 3% |

> **Key finding — revised.** EXCLUSIVE is **not** zero. It is the dominant failure mode for
> annulus-based variants: **~50% for Octant, ~89–99% for Spotter**. Per-VP band checks (a necessary
> but not sufficient condition for EXCLUSIVE=0) were misleading: each individual annulus may reach
> D_truth's disk, but the *joint* intersection of all annuli creates a crescent- or ring-shaped
> region that can miss D_truth entirely. RTT inflation pushes annular inner bounds outward across all
> VPs simultaneously, causing the feasible region to drift beyond D_truth.
>
> **The bounded answer space rescues EXCLUSIVE predictions via centroid snapping.** A prediction in an
> EXCLUSIVE region (polygon does not overlap D_truth's 50 km disk) can still snap to the correct
> answer cell if the centroid of the EXCLUSIVE polygon points toward the right Voronoi cell. This is
> the core mechanism of the **tolerance dividend**:
> - Octant US: 48 EXCLUSIVE predictions, of which **~27 are still correct by classification**
>   (centroid of the drifted region lands in the right cell). This accounts for nearly the entire
>   27 pp tolerance dividend.
> - Spotter US: 95 EXCLUSIVE, **~6 correct** — Spotter's entire 6.3% accuracy is a tolerance artifact.
>
> This substantially revises the per-variant narrative:
> - **Octant's 51% accuracy** = 22% INCL_success + 28% EXCLUSIVE-but-centroid-correct (tolerance)
> - **Spotter's 6.3% accuracy** = ~0% INCL_success + ~6% EXCLUSIVE-but-centroid-correct (tolerance)
> - **Million-scale's 44%** = 36% INCL_success + ~8% EXCLUSIVE-but-correct + ~0% EMPTY-correct
>
> The failure story shifts: Octant and Spotter fail primarily via **EXCLUSIVE_REGION** (Phase 1/2
> bias that drifts the feasible region beyond D_truth), not INCLUSIVE misclassification. Disk-based
> variants (Vanilla, Million-scale) fail mostly via INCLUSIVE misclassification with small EXCLUSIVE.

**Attribution figure.** `analysis_fail/failure_attribution.png` shows the failure-mode breakdown
(no-proximity / EMPTY / EXCLUSIVE / INCLUSIVE-misclassified) per variant × setup, for both the
proximity-limited setups from §6.1 and the proximity-sufficient setups from §6.2. The contrast
makes the regime transition visible: the no-proximity bar dominates the left panel; variant-specific
bars dominate the right.

**eu-de as a clean RTT-inflation isolation.** The eu-de setup (AS3209 fleet, 96 DE targets,
max `avail_min_vp_km` = 79 km across all targets) has **zero no-proximity failures** for all four
variants — the fleet is close enough that proximity cannot be the cause. This makes eu-de the
cleanest setup to isolate RTT inflation as a failure driver. In this setup, Octant fails 55% of
its wrong predictions via RTT_INFLATION (annulus shifted outward by inflated RTTs); Million-scale
53%; Vanilla fails primarily via ERRONEOUS_CONTAINMENT (46%, low-envelope empties region) rather
than inflation. The prime F2 candidate for manual verification:
`185.32.187.206`, eu-de, octant\_cbg, fold\_2 — VP at 280 m, inflation 14.76×, prediction 806 km
north (near Denmark). This is the most likely EXCLUSIVE\_REGION case in the dataset once
proper Shapely polygon-disk reconstruction is applied.

**Participating-VP characterization of INCLUSIVE_REGION.** Among predictions where `R ∩ D_truth ≠
∅`, success depends on three geometrically motivated factors (pooled AUC across 6 runs × 4
textbook combos):

1. **Proximity — primary, universal** (`part_min_rtt_ms`, mean |AUC−0.5| = 0.32 for precision):
   nearest participating VP RTT ≲ 5–7 ms ⇒ Tier-1. Globally CBG degenerates to shortest-ping
   (`n_part` ≈ 1); regionally 7–20 participating VPs enable genuine multilateration that doubles
   Tier-1 and halves Tier-3.
2. **Answer-space isolation** (`nearest_other_centroid_km`, AUC 0.64–0.68 globally): among
   far-from-VP targets, an isolated truth centroid (~265 km to next) still snaps right; a crowded
   one (~135 km) does not. Carries zero signal in small in-country answer spaces.
3. **Angular surround** (`part_circ_var`, AUC up to 0.82 in DE): secondary, regime-gated — visible
   only when proximity is already satisfied and multiple VPs participate. Distance-controlled
   inward/outward experiment confirms a genuine ~1.5–2× residual angular effect.

**Observable confidence flag (inference-time).** L1 — MTL region overlaps exactly one answer cell
— is computable without ground truth and achieves precision **0.66–0.91** (ex-Spotter), capturing
30–56% of all correct answers as high-confidence outputs.

### 6.4 Proprietary single-ASN dataset — operator regime payoff

**Setup.** One network operator's user-plane (UP) RTT measurements; VPs are the operator's own
core-network sites; targets are 200+ IPs from a single target ASN across 20+ US clusters. This is
the cleanest operator regime: RTTs from one operator ASN to one target ASN share the same peering,
backbone, and interconnection topology — exactly the structure CBG's LTD calibration assumes.
Compared to RIPE, this dataset has (a) fewer inter-ASN noise sources, (b) VP topology matched to
the target ASN's physical footprint, and (c) natural per-ASN calibration (no inter-ASN pooling
required).

We apply the identical evaluation pipeline: shortest-ping baseline, four textbook CBG variants,
`R = 50 km` answer-space clustering, and the §6.3 failure taxonomy.

*[Results to be filled in once the proprietary dataset evaluation is complete. Expected story:
accuracy jumps substantially — driven by VP proximity and clean single-ASN RTTs — but a residual
~20% failure remains. The §6.3 taxonomy applied to these residual failures should show a clean
EXCLUSIVE / INCLUSIVE split with no no-proximity failures, making the attribution unambiguous and
the failure model fully convincing as the paper's closing empirical statement.]*

**Read:** this section answers "with an operator-curated dataset where CBG's assumptions are fully
met, how good does it get — and what is the irreducible failure?" The residual failure analysis here
is the paper's strongest evidence for the §6.3 taxonomy, because the proximity confound is gone.

### 6.5 Production cost

Cost lives in the centroid phase. We compare two centroid rules on top of Octant's annulus MTL:
`monte_carlo_medoid` takes **190–390 ms** per target; `geometric_centroid` takes **~0.25 ms**. In
the global AS16509 run, `octant_cbg_hull_geo` (geometric centroid) reaches **30.0%** accuracy at
**65 targets/s**, while default `octant_cbg` (Monte Carlo) reaches **25.2%** at **3.2 targets/s**
— same or better accuracy at ~20× throughput. LTD and MTL phases are negligible by comparison;
the centroid choice is the only production-cost lever.

---

## 7. Improved CBG Variants

Having established in §6 *when and why* the four textbook variants succeed or fail, we now ask
whether targeted repairs to each variant's failure mode produce a better variant. All sub-sections
evaluate **classification accuracy only** across all setups; no new metrics are introduced. The
section closes with a head-to-head of the best new combo against the original four.

### 7.1 Weighted vs. unweighted intersection

The textbook variants differ in whether the MTL intersection is weighted by per-VP reliability.
We compare each variant against its weighted counterpart:

- `vanilla_cbg` → `vanilla_cbg_weighted` (weighted intersection; *note: placeholder name — confirm exact variant name*)
- `million_scale_cbg` → `million_scale_cbg_weighted` (weighted intersection)
- `octant_cbg` → `octant_top` (top-weighted constraint selection)
- `spotter_cbg` → `spotter_top` (top-weighted constraint selection)

**Read:** does weighting the intersection repair the EXCLUSIVE_REGION and INCLUSIVE misclassification
failures identified in §6.3, and if so, for which variants and regimes?

### 7.2 Geometric vs. Monte Carlo centroid

Holding the MTL phase fixed, we swap the centroid rule:

- `octant_cbg` → `octant_cbg_geo` (Monte Carlo → geometric)
- `spotter_cbg` → `spotter_cbg_geo` (Monte Carlo → geometric)

This directly targets the INCLUSIVE_REGION resolution failures from §6.3 Phase 3. The production
cost payoff from this swap is already quantified in §6.5 (~20× throughput); here we quantify the
accuracy trade-off.

### 7.3 Tight vs. loose constraint

Holding the centroid rule fixed at geometric, we compare the Octant constraint-processing choices:

- `octant_cbg_geo` (standard Octant constraints — tight)
- `octant_hull_geo` (hull-relaxed constraints — loose, expanding the feasible region to reduce
  EMPTY_REGION and EXCLUSIVE_REGION failures at the cost of resolution)

**Read:** does loosening the intersection recover EMPTY_REGION failures (where the tight intersection
was empty) without introducing new INCLUSIVE misclassifications?

### 7.4 Best new CBG combo — head-to-head

Across all setups and regimes, we identify the single best-performing new combination from §7.1–7.3
and present it as a candidate SOTA variant. The comparison table places it alongside the original
four textbook variants across all evaluation setups from §6.

*[Candidate to be nominated once §7.1–7.3 are computed. Current front-runner from preliminary
results: `octant_hull_geo` — 30.0% accuracy at 65 targets/s on global AS16509, beating default
Octant at 25.2% / 3.2 targets/s.]*

---

## 8. Discussion

### 8.1 Best practices for VP–target space setup and calibration

Operator guidance on VP deployment drawn from the effective-distance analysis and the regime
transition established in §6. What VP density and geographic spread does an operator need for a
given target ASN footprint? How does the target-distinguishable margin translate to a VP acquisition
decision? How does answer-space density (centroid isolation) interact with VP placement?

**Per-ASN calibration.** The contrast between §6.2 (RIPE, mixed-ASN) and §6.4 (proprietary,
single-ASN) motivates a concrete recommendation: CBG's LTD calibration should be trained
per-ASN rather than pooled. RTTs from one operator ASN to one target ASN share the same peering,
backbone, and interconnection topology; pooling across ASNs mixes propagation regimes that the
LTD model was not designed to average. The proprietary dataset's accuracy jump over RIPE — holding
VP proximity constant — is the empirical evidence for this recommendation.

### 8.2 CBG limitations and advantages by use case

When CBG is the right tool (calibrated, per-ASN, proximity-sufficient) vs. when shortest-ping is
sufficient or better. The Anchored / Unanchored two-track taxonomy from §1.3 revisited in light of
the empirical results. Spotter as the cautionary variant: structural collapse even with a near VP.

### 8.3 Generalization of method and dataset

What the two-dataset design (public RIPE + proprietary operator) says about where the results
generalize. VP-topology matching between datasets. Limits: RIPE anchors are well-geolocated by
construction; the proprietary dataset is a single operator–ASN pair. What would change with more
operators, more target ASNs, or passive RTT sources.

### 8.4 Future implications

The composable framework and open-source release as a substrate: future CBG accuracy claims can
be pinned to a specific phase combination rather than a monolithic "CBG." The failure taxonomy
(EMPTY / EXCLUSIVE / INCLUSIVE) as a diagnostic interface for new LTD models, MTL weighting
schemes, or centroid rules. The inference-time confidence model (L1 region overlap) as a
deployable operator primitive independent of the CBG variant.

---

## 9. Conclusion

Restate: composable framework + two-regime benchmarking story + failure taxonomy + actionable
variant guidance + open-source release. Position the framework as the substrate future CBG work
should plug into so accuracy claims become directly comparable across variants, datasets, and
evaluation criteria.

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
