**Working title:** Does Latency-Based Geolocation Work for Network Operators? A Benchmark of Shortest-Ping and CBG Variants

**Status:** narrative spine (§1 to §6) tightened · §7 onward pending
**Supersedes:** [[CBG-Benchmark-Paper-Flow-v1]]
**Related boards:** [[CBG-Benchmark-From-Network-Operators.excalidraw]] · [[CBG Variant Benchmark.excalidraw]] · [[Paper Agenda]]

**Scope decisions locked for v2:**
- The CBG + XGBoost candidate classifier ([[CBG + XGBoost]]) is out of scope. It is the follow-up paper, built on v2's answer-space definition and fold protocol. v2 mentions it only in Limitations.
- Phase-level ablation is not its own RQ. The composable 3-phase framework is the instrument through which RQ3 answers root cause.

---

## 1. Motivation

Network operators deliver content that originates from hypergiants and CDNs. Running that delivery (traffic monitoring, traffic engineering, troubleshooting, capacity planning, content-steering verification) repeatedly requires answering one question: **where is this hypergiant/CDN IP physically served from?**

This is not a one-off lookup. It is a continuous, million-IP-scale obligation against a substrate that keeps changing: peering points move, CDNs re-home prefixes, capacity shifts between metros. An operator that cannot answer the question cheaply and repeatedly cannot tell whether traffic it hands to a peer is being served locally or hauled across a continent.

**Framing device.** The operator's question is defined by three variables:

| Variable               | Meaning                                                                          | Example / sub-dimensions                                                                                                                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intent**             | what the operator is trying to achieve by geolocating, and how it weights errors | continuous content-steering monitoring (prioritizing traffic-weighted flows) · long-distance routing detection (equal weight to target sites)                                                                                                                               |
| **Dataset**            | the measurement setting the operator already owns                                | **VP topology**: count, placement, geographic spread, device type · **Target peering**: which ASNs, on-net embedded cache vs. off-net via IXP/transit · **RTT collection**: passively collected at UPs vs. active mesh ping · public RIPE Atlas as the academic counterpart |
| **Geolocation method** | the technique applied to that dataset                                            | Shortest-Ping · SoI CBG · Vanilla · Octant-Hull · Octant-Spline · Spotter                                                                                                                                                                                                          |

Every claim in this paper is indexed to an (intent, dataset, method) setting. "Which method is best" is not well-posed without the first two.

---

## 2. Problem Statement

Operator-scoped IP geolocation is a narrower and differently shaped problem than general-purpose IP geolocation. Four structural differences: two about the data the operator has (the vantage-point side, then the target side), and two about the answer it needs.

**2.1 The vantage-point side is free, ground-truthed, and traffic-representative.**
The operator gets three properties here. The third is the one prior work cannot obtain.

- **Free.** Operators already collect RTT at core infrastructure, so no probe fleet has to be rented or deployed.
- **Ground-truthed.** Those VPs sit at physical sites whose coordinates the operator owns exactly. In the academic setting, VPs are scarce, rented, and approximately located.
- **Traffic-representative.** The measured paths *are* the production paths. RTT is observed at the user planes where subscriber traffic is actually served, and the interconnect it traverses to reach a hypergiant or CDN (embedded cache on-net, or IXP/transit off-net) is the same interconnect that carries the real content flows. The RTT distribution therefore reflects the operator's actual traffic mix, weighted by where demand is.

The third property is what makes the setting worth studying separately. A public platform can be free and reasonably well-located, but its VP-to-target pairs are chosen by probe availability rather than by traffic: paths get measured that carry no production load, and the interconnects that dominate the operator's traffic are largely unmeasured. Accuracy computed over such pairs answers *"how well does this method locate a reachable target?"*, whereas the operator needs *"how well does it locate the targets my traffic actually goes to, over the paths it actually takes?"* Those are different questions, and only the second one supports using these VPs to monitor one's own network. (The reverse-direction consequence, that neither setting can be substituted for the other, is §3.2.)

**2.2 The target side is uneven, in peering regime and in label coverage.**
The VP side is uniform and well-controlled; the target side is neither. It is heterogeneous along two axes, and the first one is what makes the second one bite.

***(a) Peering regime differs per ASN, so calibration is per-ASN rather than global.***
A calibrated CBG model is a latency-to-distance function fitted to observed pairs of RTT and distance. That function is not fitting the speed of light alone. It also absorbs everything the peering arrangement adds: queueing at the interconnect, the detour through an IXP or transit hop, and whether the target is an on-net embedded cache one hop from the user plane or an off-net server reached through a third-party colo. Those additive terms are properties of the operator's relationship with that specific ASN, and they differ across peers by more than they differ within a peer.

Consequences that carry into the methodology:
- **The calibration unit is (VP, peer ASN), not VP alone.** A bestline or spline fitted over all targets pooled together averages an on-net cache regime against an off-net transit regime and describes neither.
- **RTT statistics are not portable across peers.** The same great-circle distance yields a different RTT distribution depending on the interconnect it crosses, so a model fitted on one peer is mis-specified on another. 【TODO: NEED AN EXAMPLE FIGURE TO SHOWCASE】
- **This is testable, and worth testing.** Per-ASN vs. pooled calibration becomes an ablation rather than an assumption (§7, RQ3). If pooled calibration turns out to be adequate, that is itself a useful negative result for operators who lack the per-peer label volume to fit separate models.

***(b) Label coverage is never uniform, and where it is absent, accuracy becomes unverifiable.***
For any operator, some part of the target population carries a canonical location claim (rDNS naming, self-published Geofeed, a known interconnect site) and the rest carries none at all. That split is not an artifact of a particular dataset. It is the standing condition of the problem, and it partitions the work into two tracks:

- **Anchored track: labels exist for the peer in question.** Speed-of-internet (≈ ⅔ c) is calibration-free, so with ground-truthed VPs a *claimed* location is physics-validated if and only if it lies inside the intersection of ≥ k of N VPs' SoI circles. Validated claims become that peer's training labels, and calibrated CBG (Vanilla, Octant-Hull, Octant-Spline, Spotter) then predicts the uncovered residual. This is label propagation from an anchored seed, within one peering regime.
- **Unanchored track: no labels for that peer.** The methods still run. SoI bounds where a target can be, and a calibrated model borrowed from elsewhere will emit coordinates. What is missing is any basis for trusting the output. By (a), labels imported from another ASN carry that ASN's topology and RTT statistics, so the borrowed model is mis-calibrated by an unknown amount, and with no local labels that amount cannot be measured. The method still produces an answer; the problem is that its accuracy is unverifiable in the one setting where you would need the guarantee. 【FUTURE WORK: BORROW MODELS BASED ON RTT STATS (implicitly meaning similar peering): ONLY IF RTT STATS W.R.T VPs ARE SIMILAR FROM A UNKNOWN TARGET】

So the unit of work is the peer. Own data per ASN, meaning labels for calibration and held-out labels for verification drawn from the same peering regime, is what makes an accuracy claim meaningful at all. The anchored track is this paper's target because it is the setting where a benchmark result can actually be validated.

The unanchored track is the harder case, and it depends on this paper's results rather than competing with them. Making it usable requires a confidence model that can flag when a borrowed or bootstrapped estimate is untrustworthy, and that model cannot be built before SoI CBG's accuracy and failure behaviors are characterized per peering regime, which is what this benchmark produces. It is out of scope here and named as the designated follow-up.

> **Known limit on the anchored track.** Physics-consistent does not mean uniquely identified. When several answer-space candidates fall inside the SoI intersection, the claim is consistent with all of them. Label quality is therefore bounded by intersection tightness, which we quantify rather than assume away. Combined with (a), this sets a floor on the label volume needed per peer, not only in total.

**2.3 The answer space is bounded and metro-granular.**
For hypergiants, CDNs, and peering partners, the operator already knows the set of *possible* serving locations, because that is where the interconnection was built or where a self-published geofeed says it is. An Akamai IP seen at a mobile core does not live at an arbitrary point in continuous lat/lon space; it lives at one of N known sites. The operator's decisions are also made at metro granularity rather than city or street granularity, because infrastructure is deployed to serve regions. Sub-city precision is not a requirement the operator is willing to pay for.

**2.4 Success has two parts: the right answer, and the ability to produce it at scale.**
A method is deployable only if it satisfies both. Existing literature reports the first and is largely silent on the second.

***(a) Correctness is classification, not coordinate regression.***

Because the answer space is finite and metro-granular, the question to ask is *"can the method pick the right metro from the known list?"* rather than *"how many kilometres off is it?"* A method emits a lat/lon, we resolve it against the bounded answer space, and the prediction is correct if it lands on the ground-truth region. That is both easier (the space is finite and tolerant of bounded error) and more useful (it is the operator's actual decision) than error-distance minimization.

> **Consequence for evaluation.** Error distance and classification accuracy can disagree. A variant can be off on lat/lon but right on the answer that matters, and the converse. v2 reports both and treats the disagreement itself as a finding.

***(b) Practicality at million-IP scale is a first-class metric.***

Operator geolocation is a routine rather than a one-shot study over a few hundred targets: a recurring job over the operator's full address space, on the order of millions of IPs, re-run as prefixes move and interconnects change. Two costs follow from that.

- **Runtime** must fit the routine's cadence. A variant that takes hours per run cannot support continuous content-steering monitoring, whatever its accuracy.
- **Memory** must stay bounded as target count grows. Variants differ sharply here: region-intersection geometry, Monte Carlo sampling over constraint sets, and per-(VP, ASN) calibration models all scale differently in the number of VPs, targets, and peers.

Prior work rarely reports either. Published CBG results are stated as accuracy on modest target sets, so an operator has no way to know whether a variant that wins on accuracy is affordable to run, or whether the cheapest variant, Shortest-Ping, is being beaten by an amount worth paying for. Measuring runtime and memory alongside accuracy for every variant is part of the benchmark's contribution, not instrumentation around it.

> **Consequence for the paper's claims.** Every result is reported as an (accuracy, runtime, memory) triple rather than accuracy alone, and "best variant" is stated relative to a stated budget. A variant that wins on accuracy while losing on practicality is reported that way rather than silently ranked first, which is what makes §9's when-to-use-what guidance possible.

---

## 3. Challenges

### 3.1 Every available signal fails structurally

If operators know so much about their own peers, why not just read the canonical records? Each signal fails in a way that is structural rather than incidental.

| Signal                                        | Why it does not suffice for the operator                                                                                                                                                                                                                                                                                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Large-scale active probing / traceroute**   | Highest-accuracy prior work depends on it. At millions of IPs, continuously, the measurement cost is prohibitive; not all routers/hosts are traceable.                                                                                                                                                                                   |
| **Commercial geolocation DBs / services**     | Black-box: when a result is wrong the operator cannot troubleshoot *why*, which is disqualifying for a signal feeding traffic-engineering decisions. Also cost and privacy exposure of shipping query streams to a third party.                                                                                                          |
| **Self-published data (rDNS PTR, Geofeed)**   | Coverage holes (entire ASNs publish nothing) · fragmented ingestion (WHOIS remarks vs. idiosyncratic self-hosted URLs, so *finding* the data is itself an engineering project) · staleness (not re-published on relocation) · inconsistent location naming conventions · self-reported, therefore manipulable in the publisher's favour. |
| **Public measurement platforms (RIPE Atlas)** | Coverage does not reach the ASNs and target sets the operator cares about; VP device/network types are heterogeneous and uncontrolled.                                                                                                                                                                                                   |

That residual is what the paper addresses. When canonical signals are absent, stale, or suspect, the operator still needs an estimate, and latency measurement is the only lever that runs entirely on the operator's own infrastructure and requires no cooperation from the target organization. (Assumption: connectivity between ISP and hypergiant/CDN networks is well connected by cables)

### 3.2 The data-access asymmetry cuts both ways

The operator holds two assets no academic study can reproduce: topology knowledge (its own network graph and the geographic connectivity to peering hypergiants/CDNs) and private passive data (large-volume in-network RTT at ground-truthed vantage points). That asymmetry is itself the challenge, and it breaks *both* research directions.

- **Academia cannot reproduce the operator setting.** Public platforms offer no way to recreate an operator's traffic pattern: VP sets are self-selected and heterogeneous in device and network type rather than sited at user planes, target sets are anchors rather than the hypergiant prefixes the operator actually carries, and the peering relationship that shapes the path (on-net embedded cache vs. off-net via IXP/transit) cannot be replicated at all. Published accuracy numbers are therefore measured on a topology no operator has, and whether they transfer is unknown.
- **Operators cannot substitute public infrastructure.** An operator wanting to monitor its own network from RIPE Atlas finds that coverage does not reach its ASNs or its target sets, that probe placement is not controllable, and that probes sit at heterogeneous positions rather than where its traffic is actually observed.

Neither side can validate the other's results, and no dataset exists on which the two settings can be compared apples-to-apples. This is the concrete reason the benchmark must be run on both a proprietary operator dataset and a public dataset, and the reason the paper's claim is *generalization across differing topologies* rather than a controlled head-to-head (§5, §7).

---

## 4. Gap Analysis & Research Questions

### 4.1 Gaps

- **G1 · VP realism.** VP setups in existing CBG work do not reflect operator deployments (count, placement, on-net position, passive collection).
- **G2 · No cross-variant comparison on operator-realistic data.** Variants are published against different datasets, VPs, and metrics; nobody has run them side by side under operator conditions.
- **G3 · Results without mechanism.** Accuracy tables do not tell the reader *why* a variant wins or loses, so the result does not transfer to a new setting.
- **G4 · Accuracy is not a deployment criterion.** Runtime, memory, and robustness decide what actually ships at million-IP scale, and they are largely unreported.
- **G5 · Inappropriate evaluation criterion.** Continuous error distance is scored against a problem the operator does not have; bounded metro-level classification is the operator's criterion and is not used.

### 4.2 Research questions

> The three variables of §1 (intent × dataset × geolocation method) define the question space. The RQs walk it in order: does anything work, does the sophisticated thing beat the simple thing, and why.

- **RQ1 · Assessment.** *What geolocation accuracy is achievable using Shortest-Ping or CBG variants under operator-realistic deployment conditions?* → addresses **G1, G2, G5**.
- **RQ2 · Characterization.** *Does CBG deliver a meaningful gain over Shortest-Ping under operator criteria (accuracy, runtime, memory), and if so, which variant, under which (intent, dataset) conditions?* → addresses **G2, G4**.
- **RQ3 · Root cause.** *What factors determine the accuracy and failure behaviour of CBG variants?* Answered by decomposing every variant into the composable 3-phase framework and attributing outcomes to phase choice, VP-to-target geometry, and dataset composition. → addresses **G3**.

**Baseline discipline:** Shortest-Ping is the reference point throughout. A CBG variant that does not beat it on the operator's criterion is reported as not worth deploying.

---

## 5. Contributions

1. **The first operator-perspective benchmark of latency-based geolocation.** Three things make it operator-perspective rather than another accuracy table. (i) *Data:* two datasets with structurally different composition, a proprietary single-target-ASN operator dataset (VPs = user planes / mobile cores) and a public mixed-ASN RIPE-based dataset. We do *not* claim an apples-to-apples match; we claim, and test, generalization across differing topologies (see §7). (ii) *Criterion:* a bounded, metro-granular answer space whose class count is fixed by an equal-area 51 km grid shared by both datasets, each class seeded at the centroid of the targets it merges, scored by classification accuracy alongside conventional error distance, including region-vs-region scoring so that CBG's native output (an intersection region) is judged as a region. (iii) *Cost:* runtime and memory reported per pipeline phase with accuracy-vs-cost trade-off curves, so every result is an (accuracy, runtime, memory) triple rather than a single number.
2. **A unified, composable CBG framework with an open-source implementation.** The framework (latency-to-distance × multilateration constraint shape × centroid selection) spans SoI CBG / Vanilla / Octant-Hull / Octant-Spline / Spotter, which turns variant comparison into phase attribution and makes RQ3's root-cause analysis possible. The implementation lets downstream work state *which* CBG it used.
3. **Actionable characterization and when-to-use-what guidance.** Where each variant and each phase wins or loses, tied to the peering regime and VP geometry that produced the result, and stated as a deployment recommendation per operator scenario relative to a stated budget rather than as a leaderboard.

*[Pending: pick the single headline claim among these (likely #1's criterion or the strongest RQ2 finding) once results land.]*

---

## 6. Related Work

Prior latency-based geolocation work contains two neighboring but distinct problem families. **Single-location host geolocation** assumes that every VP measuring a target IP is constraining the same latent coordinate. This is the setting of CBG: an RTT is mapped to a distance constraint, constraints from multiple VPs are intersected, and the resulting region is reduced to a point when necessary. Vanilla CBG, Octant-Hull, Octant-Spline, Spotter, and the calibration-free speed-of-internet construction differ in how they implement those phases, but they share the one-target/one-location assumption. **Anycast geolocation** begins after that assumption has failed: one IP is served by multiple replicas, different VPs may reach different physical sites, and the first task is therefore to detect and enumerate those sites before assigning locations to them.

### 6.1 Anycast geolocation uses CBG geometry, but is not another CBG variant

The foundational anycast-geolocation system is **iGreedy**, introduced in *[A Fistful of Pings](https://doi.org/10.1109/INFOCOM.2015.7218670)* and expanded in *[Latency-Based Anycast Geolocation: Algorithms, Software, and Data Sets](https://doi.org/10.1109/JSAC.2016.2558898)*. iGreedy shares one primitive with calibration-free CBG: it converts each VP's RTT into a speed-of-internet disk that must contain the replica reached by that VP. Its use of those disks is different. If two disks cannot overlap, the observations cannot have come from one physical host, which proves that the address is anycast under the model. iGreedy then solves a Maximum Independent Set over the disks to obtain a conservative lower bound on the number of replicas. Finally, rather than intersecting all constraints and taking a geometric centroid, it chooses a city within each inferred replica region using a likelihood dominated by city population, collapses the region around that city, and iterates.

This makes iGreedy **CBG-adjacent rather than a sixth CBG variant**. Its latency-to-distance step is the same fixed-speed choice as SoI CBG, but its geometry answers a different question: non-overlap separates observations that must belong to different replicas, whereas conventional CBG intersects observations assumed to belong to the same target. Its final estimator is also a discrete city classifier using an auxiliary population prior, not boundary-vertex mean, Monte Carlo, or polygon centroid. Calling iGreedy “SoI CBG” would therefore hide the detection and enumeration stages that define the method; calling it unrelated would hide the common constraint geometry.

| Work / family | What is inferred | Latency-derived geometry | How location is selected | Relationship to this benchmark |
| --- | --- | --- | --- | --- |
| **CBG variants in this paper** | One location per target | Intersect constraints from VPs assumed to reach the same host | Boundary mean, Monte Carlo, or polygon centroid; then map to the bounded answer space | Methods under test |
| **iGreedy / GCD** | Whether an IP is anycast, a lower bound on replica count, and a city per replica | Fixed-speed disks; non-overlap detects distinct replicas; Maximum Independent Set enumerates them | Population- and distance-biased city classification, followed by iteration | Shares the SoI constraint primitive, but is not a conventional CBG pipeline |
| **MAnycast2** | Whether a prefix is anycast | Responses to probes sourced from distributed anycast VPs reveal multiple return sites | Does not geolocate the detected sites | Non-latency candidate detector |
| **LACeS** | Daily anycast census, replica count, and expected site locations | MAnycast2-style candidate detection followed by iGreedy/GCD latency validation | Reuses iGreedy's city-population geodetection | Scalable hybrid of candidate detection and the same GCD method, not a comparison of CBG variants |
| **Traceroute approach** | Replica enumeration and location from paths | Near-PoP hops are geolocated and grouped | Location evidence from the path rather than RTT disks | Slight precision gain over iGreedy at roughly 4× probing cost |
| **HTTP-response approach** | CDN edge-city location | No latency geometry | Geographic hints exposed in HTTP responses | Independent label or validation signal, not CBG |

Later work extends the measurement system rather than the CBG model. *[Characterizing IPv4 Anycast Adoption and Deployment](https://doi.org/10.1145/2716281.2836101)* and *[A Longitudinal Study of IP Anycast](https://doi.org/10.1145/3211852.3211855)* apply iGreedy to Internet-wide and repeated censuses. MAnycast2 replaces latency-based detection with probes sourced from distributed anycast VPs, gaining speed but not site geolocation. *[LACeS](https://doi.org/10.1145/3730567.3764484)* combines that fast candidate stage with an optimized iGreedy/GCD follow-up and still emits locations using iGreedy's population-based city classifier. *[Locating and Enumerating Anycast: a Comparison of Two Approaches](https://doi.org/10.1145/3744200.3744783)* compares iGreedy with traceroute-derived location evidence; traceroute slightly improves precision but overestimates PoP count and costs roughly four times as many probes. An orthogonal HTTP-based method extracts geographic hints from CDN responses, reporting city-level results without multilateration (*[Locating CDN Edge Servers with HTTP Responses](https://doi.org/10.1145/3546037.3546051)*).

### 6.2 What anycast changes for a CBG benchmark

The literature establishes four constraints on our evaluation. First, an anycast address has no single context-free ground-truth coordinate: the appropriate label is indexed by target IP, VP or catchment, and measurement time. Second, an empty intersection across VPs can be evidence that those VPs reached different replicas rather than a failure of the latency-to-distance model. Third, anycast enumeration is conservative: geographically close replicas can remain hidden because their disks overlap, so inferred site count is a lower bound. Fourth, routing policy can defeat geographic locality. Selective announcements and remote peering can direct traffic to a distant replica and inflate latency, as quantified by *[Too Remote to Be Local](https://doi.org/10.23919/CNSM67658.2025.11297415)*; “BGP-nearest” must not be read as “geographically nearest.”

These constraints define a dataset rule rather than another method in the leaderboard. The benchmark must either **exclude addresses whose measurements can terminate at multiple concurrent replicas**, or key their ground truth by `(target IP, VP/catchment, time)` and evaluate each replica group separately. Pooling RTTs from multiple replicas under one target would violate the shared assumption of every CBG variant and turn valid observations into apparent empty-intersection failures. Anycast detection, operator PoP inventories, rDNS/CHAOS identifiers, HTTP hints, or traceroute-near-PoP evidence can supply this curation signal; they are not silently folded into the latency-only methods being compared.

No related work we found performs the missing composition: enumerate or cluster an anycast IP's replica-specific observations, then compare SoI CBG, Vanilla, Octant-Hull, Octant-Spline, and Spotter within each replica group under one answer space and metric suite. That is a distinct **anycast-aware CBG benchmark** and a natural extension of this work, not a result claimed here. This paper's contribution remains the controlled comparison of the CBG variants after the single-location evaluation unit has been made valid.

**Research notes:** [[Anycast IP Geolocation Literature]]

---

## 7. Methodology

### 7.1 The latency-based geolocation benchmarking framework
![[Pasted image 20260825141829.png]]
No two published latency-based geolocation results are comparable, because each paper brings its own dataset, its own answer granularity, and its own definition of a correct answer. Our first methodological artifact is therefore a harness rather than a method: one engine that any latency-based geolocation method plugs into, reported on the same axes throughout.

**The goal is an input, not a background assumption.** The deployment goal of §2 enters the framework explicitly, and it fixes two things before any method runs. It selects which dataset is admissible, since a goal stated over an operator's own peers cannot be tested on targets the operator never carries traffic for. And it sets how fine the answer space is allowed to be, since a metro-level routing decision does not need a street-level answer and should not be scored as though it did. Making the goal an input is what keeps the rest of the framework from silently optimizing for a target the operator does not have.

**The engine holds fixed what must be shared.** Three components sit inside it and do not vary across the methods under test. The *method runtime* executes any method against a dataset, baseline included. The *target answer space* discretizes ground truth into the finite region set every method must choose from. The *evaluation methods* score that choice. Holding all three fixed is the whole point: once they are shared, a difference between two reported numbers is a difference between two methods, which is exactly what the published literature cannot currently claim.

**Methods are plug-ins against one interface.** Shortest-Ping, the five CBG variants, and later candidates such as the learned classifier of the follow-up work all enter the same way: dataset and VP set in, region choice plus diagnostics out. This is what makes the baseline discipline of §5 enforceable rather than aspirational, since Shortest-Ping runs through the same runtime, the same folds, and the same scorer as every variant it is compared against.

Shortest-Ping is worth being explicit about, because it is trivial enough to be tempting to compute on the side. It is an argmin over the RTT vector followed by the VP's own coordinate, so it has no fit stage and no geometry, and its plug-in is a few lines. We still run it inside the method runtime, for three reasons. Its runtime and memory have to be measured the same way as everything else, since §5 states every CBG result relative to it and that comparison is an (accuracy, runtime, memory) triple rather than an accuracy gap; its cost is also the floor that tells a reader what CBG's accuracy gain is being bought with. Its output is a coordinate, so it needs the same answer-space mapping, the same folds, and the same region-vs-region scoring as any other method anyway. And CBG's empty-intersection fallback is Shortest-Ping (§7.2), so the rule already lives inside the runtime; a second copy outside it could drift, and it would drift precisely on the fallback cases we count as failures.

**Every run exits with three metrics, not one.** Accuracy, speed, and memory come out together for every method, which is the mechanism behind treating practicality as first-class (§2.4(b)) rather than as a closing remark. A method that wins on accuracy and loses on the operator's runtime budget is reported as exactly that, because the harness has no way to emit the first number without the other two.

**What the framework is not.** It is not a measurement platform. It does not collect RTT, choose VP placements, or curate ground truth; it consumes datasets in which those decisions have already been made, and §7.3 states them for the two we use. It also does not rank methods on its own: the ranking is a function of the goal fed into it, and a different goal can reorder the same results without any of them changing.

All CBG methods and the runtime that executes them are implemented through a single composable framework rather than variant by variant, described next. That choice is what makes the plug-in interface real, since it means the five variants share one code path and differ only in the configuration they declare.

### 7.2 A unified, composable CBG framework

![[Pasted image 20260826130139.png]]

![[Pasted image 20260826130307.png]]

Each CBG variant in the literature is presented as a monolithic method, which is part of why their published numbers cannot be compared. Read side by side, they are the same pipeline making different choices at three points:

1. **Latency-to-distance.** Two stages, and separating them matters for both labels and cost. *Fit:* estimate the function from labeled pairs of RTT and distance, per (VP, peer ASN) for the reasons in §2.2(a). *Apply:* map one observed RTT to a distance bound for a target whose location is unknown.
2. **Multilateration.** Turn each VP's bound into a geometric constraint, then intersect the constraints. Three sub-steps, each an independent choice: (2a) *constraint shape*, the geometry a bound becomes; (2b) *constraint filtering*, which constraints are admitted to the intersection; (2c) *intersection weighting*, how the admitted constraints are weighted when the final region is selected.
3. **Centroid selection.** Reduce the intersection region to a point estimate, when the downstream decision needs a point.

Phases 2 and 3 have no fit stage. The framework's entire label dependency sits in Phase 1, and speed-of-internet is the only Phase 1 option whose fit stage is empty as well, which is why it is the one option that stays usable when a peer has no labels at all (§2.2(b)).

The foundational variants are cells of that cross product, with the calibration-free one first:

| Variant               | Phase 1 · latency-to-distance | Phase 2a · constraint shape | Phase 2b · constraint filtering | Phase 2c · intersection weighting | Phase 3 · centroid selection |
| --------------------- | ----------------------------- | --------------------------- | ------------------------------- | --------------------------------- | ---------------------------- |
| **SoI CBG**           | Speed-of-internet (no fit)    | Spherical circle            | none † inclusion filtering      | unweighted                        | Boundary-vertex mean         |
| **Vanilla CBG**       | Low envelope                  | Spherical circle            | inclusion filtering             | unweighted                        | Boundary-vertex mean         |
| **Octant-Spline CBG** | Bounded spline                | Planar annulus              | inclusion filtering             | unweighted † inverse-RTT          | Monte Carlo                  |
| **Octant-Hull CBG**   | Bounded hull                  | Planar annulus              | inclusion filtering             | inverse-RTT                       | Monte Carlo                  |
| **Spotter CBG**       | Normal distribution           | Planar annulus              | inclusion filtering             | inverse-RTT                       | Monte Carlo                  |

† Where the original paper leaves the switch open, we implement both settings and report both rather than picking one. Every other cell is the configuration its paper specifies, so each row is one variant as published, not a variant of our own choosing. For Octant we implement its latency-derived constraints only, and exclude its auxiliary geographic priors (population density, undersea-cable geography), which are outside this paper's latency-only scope.

We call the calibration-free variant **SoI CBG** rather than by its original title, Million-scale CBG, because within the framework it is the speed-of-internet Phase 1 choice that defines it, and that same choice is available to any other Phase 2 and Phase 3 pairing. We also treat **Octant-Hull** and **Octant-Spline** as separate variants rather than as one Octant, because their Phase 1 models are different (a bounded convex hull over the RTT-to-distance point cloud vs. a bounded spline fitted through it) and there is no reason to expect them to fail the same way. The full option set per phase is larger than what these five use:

- **Phase 1 · latency-to-distance:** speed-of-internet · low envelope · bounded hull · bounded spline · normal distribution.
- **Phase 2 · multilateration:**
	- **2a · constraint shape:** spherical circle (disk) · planar annulus.
	- **2b · constraint filtering:** none · inclusion filtering.
	- **2c · intersection weighting:** unweighted · inverse-RTT weighted.
- **Phase 3 · centroid selection:** boundary-vertex mean · Monte Carlo · polygon (area-weighted) centroid.

Two things follow from the decomposition. Variant comparison becomes phase attribution: hold everything else fixed, sweep one phase or one multilateration sub-step, and the accuracy difference is attributable to that choice, which is how RQ3 is answered rather than speculated about. The sub-steps matter here, since 2b and 2c are where two variants that look identical in shape can still diverge. And the cross product contains cells no published variant occupies, so new combinations can be found by construction rather than by luck.

**The switches are reported, not buried.** Filtering and weighting are normally decided inside a variant's implementation and left unstated, yet they change the intersection materially: a single tight constraint can null an otherwise healthy intersection, and a weight relaxation can revive it. We treat them as explicit switches and name the setting behind every reported result. 【TODO: one-line operational definition per switch, including inclusion filtering, inverse-RTT weighting, and the Octant-style weight relaxation applied when the intersection comes out empty.】

**Fallbacks count as failures.** When the intersection is empty and no relaxation recovers it, the pipeline falls back to the Shortest-Ping VP. Scoring that fallback as a CBG success would floor CBG's measured accuracy at Shortest-Ping's, which is precisely the comparison RQ2 rests on (§5), so fallbacks are counted as failures. Fallback rate is reported as a first-class diagnostic, and accuracy is given both with and without fallback cases included.

**Cost is measured per phase and per stage.** Fit cost and apply cost are reported separately, because they scale differently and are paid on different schedules. Fit cost grows with the number of (VP, peer ASN) pairs and the labels available per pair, and is paid once per recalibration. Apply cost grows with the number of targets, and at million-IP scale it is paid on every run. Separating them is also what lets the per-ASN vs. pooled ablation report a trade-off rather than a winner, since per-ASN calibration multiplies fit cost by the peer count.

### 7.3 Datasets and experimental setup
- **Proprietary Mesh Unicast IP dataset** 
	- VPs: Mobile cores of AS7018
	- Targets: Top-ASN servers (3 ASNs)
	- Measurements: Mesh-ping RTT from VPs to Targets
	- Properties: 
		- In-network active measurement: mobile cores/targets are used in real traffic.
		- Known peering locations for each ASN (support RTT root cause analysis)
		- VP-Target bipartite graph of 3 ASNs are geographically & topologically different
		- Clean ASN separation, allow data training and testing on each ASN individually, thus cleaner RTT-distance modeling for routing topology
		- p5 RTTs from weekly tests that avoid congestion impact
- **Proprietary Traffic-Weighted Unicast IP dataset**
	- VPs: Mobile cores of AS7018
	- Targets: Top-ASN servers that serve top 95% of Unicast traffic (3 ASNs)
	- Measurements: Passive RTT of real traffic flows from VPs to Targets
	- Properties:
		- In-network passive measurement, reflecting real-world traffic patterns and important target locations
		- Fewer VPs have RTT measurements to targets compared to mesh-ping.
- **Public Mesh Unicast IP dataset**
	- VPs: RIPE probes of AS7018 (need reciprocal filtering)
	- Targets: RIPE anchors of mixed ASNs
	- Measurements: Mesh-ping RTT from VPs to Targets
	- Properties:
		- Best-effort active measurement that relies on crowd-source infrastructure 
		- Vantage points' network types are heterogeneous and locations are not always correct.
		- Mixture of ASNs in training, which could impact accuracy of RTT-distance modeling
		- Unknown peering locations for the ASNs
- **Proprietary Traffic-Weighted Anycast IP dataset**
	- VPs: Mobile cores of AS7018
	- Targets: Top-ASN servers that serve top 95% of Anycast traffic
	- Measurements: Passive RTT of real traffic flows from VPs to Targets
	- Properties:
		- In-network passive measurement
		- Multiple location labels per IP, where exact location depends on input of regional VPs


【Discuss datasets】
- Unique Dataset properties:
	- VPs 
		- in-network measurement:
			- VPs all belong to the same ASN of the operator networks -> PNI locations
			- Mesh measurement to targets flow via the same in-network routes, enabling traffic weighting
	- Targets
		- ASN separation with sufficient samples and known PNI locations, removing noise and enabling deep reasoning
	- PNI
		-  At every PNI location, there will be at least one VP (operator's infra) and one TG deployed (physical direct-connect) 
		- Expectation: PNI enables packets flow from VP to TG mainly through TG PNI location in normal conditions (no failure, congestion)
			- min-RTTs will have linear relationship to d(VP, PNI) + d(PNI, TG). Hence **min-RTT is a good proxy for routing distance**. 【 Show scatter of min-RTT vs propagation delay over d(VP, PNI) + d(PNI, TG) 】
			- If there is hairpin routing, d(VP, PNI) + d(PNI, TG) / d(VP, TG) will be > 1 -> **minRTT inflation** (minRTT / propagation RTT between VP and TG) will be > 1; otherwise if it is direct routing, minRTT inflation will be close to 1.
		- **Good peering** guarantees TG will choose the nearest PNI location for majority of the traffic, hence making d(PNI, TG) deterministic for measurements from any closeby VP to the target
			- d(VP, PNI) becomes a deterministic factor for small min-RTT's magnitude. i.e., smaller min-RTT -> VP closer to PNI. 【Show d(sping VP, PNI) vs d(other VP, PNI) per target 】
	- Mesh dataset
		- importance:【Weight design can be used for different weight metrics】capacity/traffic-weight/... create any sub dataset
	- Traffic-weight filtering (95%) on top of mesh 
		- show how good content routing is.
		- **Good content routing** enables traffic weight filtering to preserve TGs mainly at peering locations【Show TG-PNI distance of TW/MESH datasets side-by-side box plots】

**The datasets are described as a bipartite graph.** VPs and individual targets are the two measured node sets, and an edge exists wherever a (VP, target) pair carries a measurement. The same grid quantizes both node sets, and the Voronoi partition it seeds is the answer space. Both are defined immediately below. Everything in this subsection is geometry and structure only, with no RTT entering and no variant running, which is what makes these numbers the fixed reference that the RTT-dependent characterization of §8.2 and the per-variant results of §8.3 are read against.

**Latent and observed geometry are reported as a pair.** Because an edge records a measurement, the edge set is an artifact of the campaign rather than a property of the deployment, and every distance quantity therefore has two values: the latent one over all VP × target pairs, which describes where the infrastructure sits, and the observed one over measured edges only, which describes what the dataset can actually deliver. The pairing applies to distances and not to degree, whose latent value is the VP count for every target and so carries nothing. The gap between the two is a property of the campaign, and reporting only one of them is what allows sampling bias to be misread as an algorithmic result.

**Degree is a precision covariate, and no feasibility gate arises.** Every CBG constraint is a distance upper bound, so the feasible region is an intersection of disks: bounded, convex, and nonempty with a well-defined centroid at any degree of one or more. There is no threshold at three constraints. A target also enters either dataset only by having been measured, so degree is at least one everywhere by construction and degree zero is not a case that occurs. That guarantee is a property of how the datasets are assembled rather than of the method, and it does not extend to deployment, where an unmeasured target is an ordinary case; §10 takes that up. Precision degrades continuously as degree falls, and it degrades far more sharply under a poor angular arrangement than under a low count, since two landmarks on opposite sides of a target constrain it better than five clustered in one metro. Degree is therefore carried forward purely as a covariate that §8.3 regresses error and region area against, jointly with the angular statistics, and the expectation to be tested is that the angular term dominates the count term.

**Best-effort VP-topology matching.** Where the public dataset allows it, we select probes to approximate the operator's VP count, geographic spread, and pairwise distance distribution, so that a cross-dataset accuracy difference is less likely to be an artifact of VP placement alone. This is a mitigation rather than a control, and we report the residual mismatch instead of claiming it away. 【TODO: operationalize the matching criteria.】
#### Metric List:
*Node-set geometry, computed identically for VPs and for targets, printed side by side.*
- Count, plus ASN count on the target side only, since each VP set is single-ASN by construction as described above.
- **Geographic diameter** (max pairwise great-circle distance), reported with **p95 pairwise distance** alongside it, since a diameter is a maximum and one near-antipodal node sets it single-handedly. Diameter 19,400 km with p95 8,100 km is a regional cloud plus an outlier, not a global deployment. The full pairwise distance CDF is the figure behind those two scalars.
- **Occupied cell count**, on the resolution-4 grid defined above, so "location" here is exactly the merge scale that generates the classes. This is what is left after near-coincident points collapse: on the VP side, how many distinct constraint disks the set can produce, and on the target side, $K$ itself. "60 VPs in 31 occupied cells" is the honest denominator for any claim resting on independent observations.
- Reported up the hierarchy at H3 resolutions 5, 4, 3 and 2 (17, 45, 120 and 316 km). The shape of that curve is a multi-scale concentration diagnostic: a steep climb toward fine cells means the set only separates at intra-metro scales, while a flat curve means genuinely distinct metros.
- Each rung is computed by re-binning the coordinates, not by coarsening the cell identifiers, because H3's hierarchy is aperture-7 and hexagons cannot tile hexagons: a parent's six outer children each straddle its boundary, so the parent identifier is exact as an index but is not a geometric container. The two routes genuinely disagree, on 1% to 8% of targets at resolution 3 and 3% to 10% at resolution 2 across the four RIPE-derived sets we have, so the cheaper route would report a partition that no resolution actually produces. On HEALPix the coarsening is a bit shift and the two routes coincide exactly; we re-bin on both grids so the diagnostic means the same thing regardless of which is in use.
- **Degree distributions, both sides**, as median/IQR/p90 rather than bare means, since both are heavily skewed. VP degree is measurement effort spent; target degree is constraints available.
- 【Optional appendix: Clark-Evans index · anisotropy ratio and major-axis bearing · CV of nearest-neighbour distances.】

*Edge-set geometry.*
- **Density** $|E| / (|VP| \times |\text{targets}|)$, stated as measurement completeness.
- **Connected Components**: as traffic could concentrated regionally, edges might only appear as separated components.
- **Observed edge-length distribution**: great-circle distance over measured pairs (median, IQR, p90, CDF).
- **Nearest measured VP distance per target**: distance to the closest VP that actually carries an edge. The dominant scalar predictor of region size, since the smallest disk does most of the constraining.
- **Measurement efficiency**: nearest-measured-VP distance over nearest-VP distance, median across targets. This separates "the VP set is badly placed" from "the VP set is fine but the campaign allocated probes badly", two findings that a latent-only or observed-only description conflates, and only the second is actionable by reallocating measurement. It is also the readable form of campaign bias, which is a live risk here since probe failure is plausibly distance-correlated, and it is why the all-pairs latent distance distribution enters as a denominator rather than as a result of its own.
- 【Optional: degree assortativity, to show whether the best-observed VPs and best-observed targets coincide.】

*Angular geometry, over measured neighbours only. This is the arrangement term.*
- **Max angular gap per target**: the largest empty wedge among bearings from the target to its measured VPs, as median and p90. VPs at bearings 10°, 40°, 75° leave a 295° gap and barely constrain the target; VPs at 20°, 140°, 260° leave 120° and bracket it. Identical degree, different precision, which is the whole point.
- **Circular variance of bearings**: near 0 means every landmark lies in one direction, near 1 means well surrounded. Smoother than max gap, so this is the form used in the §8.3 regressions.

*[Remaining §7 subsections: leakage-free K-fold protocol · the SoI validation primitive and its k-of-N threshold · metric definitions (error distance, classification accuracy, classification confidence, practicality, diagnostics) · whether per-VP calibration geometry earns a place · the runtime and memory measurement protocol (hardware, whether the million-IP figure is measured or extrapolated, calibration time counted separately from inference). Material to migrate and tighten from [[CBG-Benchmark-Paper-Flow-v1]] §5.]*

---

### 7.4 Answer space construction
We transform the geolocation task into Voronoi cell classification problem.
#### 7.4.1 Why?
Error distance as a metric is useful to reflect the capability of a geolocation method, but not necessarily reflecting the effective geolocation accuracy for operators' needs based on different target answer space.
Example: two methods who vary in geolocation error distance in continuous space might point to the same coarse region where only one possible location candidate reside. In this case, they both correctly identify the target region. So for benchmarking the effectiveness of a geolocation method, the goal is classification accuracy, annotated with error distance.
Cicalese et al. firstly point out geolocation can be viewed as a classification problem about city selection in an estimated region. 
Operators care about regional classification as the way of their infrastructure deployment and service operation.
Instead of focusing on city-level granularity, we group target cities by H3 cell geospatial partition with size 4 (~50 km granularity) to find effective regions, then construct geolocation answer space by Voronoi tessellation with candidate regions as seeds.
#### 7.4.2 Method
Voronoi tessellation based on occupied cell centroids as seeds, as classification is evaluated based on nearest-site snapping for the estimated coordinates.

**The grid is a quantizer, not the answer space.** An H3 hexagonal grid at resolution 4 gives 288,122 cells averaging 1,770 km², 45 km centre to centre, as prior work shows metro level granularity is around 40 km. Its only job is to merge points that sit close enough to count as one place. Each occupied cell yields one class, so the grid fixes how many classes exist and roughly where.
> H3 is the grid an operator already has: hexagons give a uniform neighbour distance with no ambiguous edge-versus-corner adjacency, which is why the tessellation is standard in RF planning.

**The seeds and the partition they induce.** Each occupied target cell contributes one seed, the centroid of the targets inside it, so the answer space is $K$ real locations rather than $K$ grid squares. A coordinate is labelled by its nearest seed, and that one rule scores ground truth. Error distance is measured to the seed, which is where targets actually are.

The cost of the grid is stated rather than claimed away. Grid lines fall where the grid falls, not where targets are sparse, so a facility group straddling a line yields two seeds and two classes. The Voronoi step then places the boundary between those two seeds in the gap between them, which is sensible, but it cannot undo a split the grid already made.

*Answer-space geometry, over the $K$ seeds.*
- Seed count $K$, which equals the occupied target cell count · targets per seed · **intra-seed target spread**, the max pairwise distance among targets sharing a seed. This says how well one centroid represents what it stands for, and it is the floor under any error-distance figure.
- **Delaunay degree per seed**, computed on the sphere as the convex hull of the unit vectors so that no projection enters. A coordinate can only be misassigned to a class whose region it can cross into, so this is how many classes each answer is confusable with, and it is the scale-free form of local density in the answer space.
- **Delaunay edge lengths**, as a distribution with the per-seed minimum. The boundary between two adjacent classes bisects the geodesic joining their seeds, so half the per-seed minimum is the margin at the seed, which is the scale at which coordinate error becomes a wrong label. Intra-seed spread enters here as well, since a target sits away from its seed rather than on it, so its own margin is that half-distance give or take the spread.
- The same extent and footprint statistics as the node-set block, since the coverage ceiling in §8.2 is the number of classes reachable from VP coordinates over $K$, and that ceiling is only readable against how the seeds are spread.

## 8. Evaluations

### 8.1 RQ1: What accuracy achievable by latency-based geolocation with different datasets?

![[Pasted image 20260903160833.png]]
Show weighted-average accuracy/failure rate over dataset types (mesh, weighted, ripe) in the appendix. All results with Top-1 accuracy & failure rate across all datasets【NEED table】
- Proprietary Mesh Unicast IP dataset
	- Proprietary Traffic-weighted Unicast IP dataset
- Public Mesh Unicast IP dataset
	- VP&TG topology-paired proprietary Mesh Unicast IP dataset
##### Traffic-weighted targets vs MESH targets: where good peering and optimal content routing matter
Show stack bars per method with dataset types as hatch types (mesh - solid, weighted - hatched).
> Traffic-weighted dataset is a subset of mesh dataset where only VP-target flows and target locations that contribute to the major 95% traffic get preserved. Therefore we are comparing datasets that **differ in closeness of targets to peering locations** where VPs co-locate with targets.
![[Pasted image 20260908172516.png]]
![[Pasted image 20260908172500.png]]

First readout weighted-average accuracy of methods over MESH and TRAFFIC-WEIGHTED datasets, showcasing the overall achievable accuracy per method.

Observations:
1. Achievable accuracy
2. Top-4 rank candidates do not change: Octant family, SoI and shortest ping.
3. Shortest-ping and SoI near perfect classification.

**【Why shortest ping achieves near-perfect classification accuracy】**
- **if good? (technical) interconnect design and good content routing exist simultaneously**, traffic weight filtering mostly preserves TGs mainly at PNI locations, and RTT measurements from nearby VPs 
- Shortest-ping picks VP that are closest to PNI (min RTT -> VP closest to PNI), where TG colocate【Show scatter of SVP to PNI distance vs TG to PNI distance per TG, or spearsman】
- Nearest-answer snapping makes shortest ping VP location correct

Explain traffic-weighted dataset formation: traffic-weighting preserves heavy flows and target location where peering resides. Mention properties of geometry and routing proximity, and define VP proximity of target, VP discrimination power of target, shortest-ping VP of target AND min-RTT inflation. 

Then characterize "targets that are close to peering locations": 
	scatter plot of the【closest vp distance to seed】vs【min-RTT inflation of of closest vp】 among targets for two datasets, with mesh being gray dots and weighted being red dots. Try to see target clustering and overlaps.

![[Pasted image 20260907215821.png]]

![[Pasted image 20260908174023.png]]

![[Pasted image 20260908174528.png]]
![[Pasted image 20260908174548.png]]
【NEED WEIGHTED DATA】Try to see if traffic-weighted data will gather around left-bottom.
Expect to see traffic-weighted points 
【Also consider PNI locations】consider

【Show why good content routing is needed, NEED DATA】What if there are targets carrying non-trivial traffic flows (hence survive traffic weighting) that are far away from any peering location? (caused by bad content routing or peering insufficiency)

【conclude】 scenarios where shortest ping works perfectly: targets with good content routing at peering locations, which is represented by traffic-weighted dataset

【Why some CBG works well】
- SoI CBG: 
	- When has proximate sping vp, it becomes back shortest ping
	- what fails?
- Octant family:
	- Robust RTT-distance modeling on low RTT
	- what fails?

【Show why some CBG not working perfectly in traffic-weighted dataset】
- Vanilla CBG: 
	- fail exactly when there is a proximate sping vp: small circle causes failure of intersection due to RTT-distance modeling too tight
	- Anything else?
	- what fails?
- Spotter CBG: 
	- RTT-distance modeling on low RTT got skewed: excluding the truth and push answer far away.
	- Anything else?
	- what fails?

【Implication】For traffic monitoring, shortest ping/SoI CBG can achieve near perfect classification accuracy on datasets of ASNs with good peering and content routing. Octant CBG is more robust against quality of content routing.

![[Pasted image 20260908174803.png]]

![[Pasted image 20260908174754.png]]

![[Pasted image 20260908174736.png]]


##### **Targets without proximate shortest-ping VP are the difficulties (CBG Opportunities)** 
> Traffic-weighted datasets with good peering and content routing happen to include targets that are benefiting from both geography and routing wellness. We need to prove that only with one wellness is not enough to boost all latency-geolocation methods.

【Venn diagram】For all those targets without proximate shortest-ping VP, how are CBG solving them?
#### Target Resolvability (Mesh datasets only)
![[Pasted image 20260903164111.png]]
![[Pasted image 20260903171829.png]]

**Story line**:
- Solution Diversity: No single method resolves all targets that are geolocated
	- Only X% of target can be geolocated by all targets
	- Together they resolve 85.1% targets, 25% higher than the top single method.
	- Each CBG method has its own geolocateable targets
- Fully resolvable target distribution & Unresolvable target distribution【Add figure】
	- Characterizing dataset-specific properties: 
		- Geography: VP proximity
		- Routing: min-RTT inflation
		- Voronoi partition density (Confusion, use top-3 classification to resolve)
	- Error distance characterization
	- Success & Failure patterns (category with examples)
		- Shortest Ping 
		- SoI CBG
		- Vanilla CBG
		- Octant family
		- Spotter CBG
- Extrapolation of Success/failure mode on variant-specific scenarios (how many targets are explainable per CBG?)
#### Where all CBG can resolve:


#### Where all CBG fail:
dense answer region?

#### Where specific CBG succeed:


##### Targets that reside in dense answer regions are prone to misclassification by CBG
> This is an artificial error due to nearest-site snapping. Showing the percentage of targets confused by this factor by defining region with close nearest-site distance.

Show wrong target distribution in crossed voronoi boundary count. (1=adjencent, n=n cells away)

##### **Datasets with clean network topology also matters**
Then show curated similar topology of proprietary vs public RIPE dataset consisting of the same VP ASN. (with sparser VP footprint and different ASN mixture)

**Story line:**
- We show that CBG RTT-latency modeling could be skewed by fitting with mixed-ASN targets, hence degrading the accuracy.

### 8.2 RQ2: What method outperform another under what scenarios? Why?

### Geolocation granularity  
【Error distance distribution of top-1 cell】

Robustness
【Show top-cell x error distance distribution】


#### With shortest-ping VPs proximate to peering, which one succeed or fail? Why?

#### Without shortest-ping VPs proximate to peering, which one succeed or fail? Why?


#### **Takeaways:**
1. Octant family are the most robust and accurate across datasets
2. Traffic-weighted dataset: shortest-ping is the most cost-effective option, SoI CBG add more robustness without training;
3. Mesh dataset: Octant family & SoI CBG


**Why the baseline gets its own section.** Every CBG result in §8.1 is stated relative to Shortest-Ping, so the baseline's number is the denominator of every claim in the paper. Shortest-Ping is correct for a target exactly when the lowest-RTT VP is discriminative for that target, and that sentence carries two independent conditions. Some VP has to be discriminative for the target at all, which is a question about geometry. The RTT ranking then has to select that VP rather than another, which is a question about latency. Every statistic below answers one of the two. Where both hold is the accuracy already reported in §8.1, and what is left over is the room the variants of §8.3 had to win.

**Can any VP be right at all?** A VP is *discriminative* for the class whose seed is nearest to it, which is the same rule that labels any coordinate, and a target is *VP-proximate* when some VP is discriminative for its class. Two consequences follow from the definition alone. Coverage cannot exceed the number of distinct classes reachable from VP coordinates divided by $K$, and since VPs sharing a grid cell almost always share a nearest seed, that numerator is the count of VP-occupied cells rather than the raw VP count, so VP concentration fixes the ceiling before any measurement is taken. Coverage also upper-bounds Shortest-Ping, because the baseline is correct for a target exactly when the VP it selects is discriminative for that target. We report coverage per dataset as the fraction of classes with at least one discriminative VP, unweighted, and traffic-weighted on the operator side under the weighting of §7.1.

**Does the RTT ranking pick it?** Where it does not, the target is reachable but unreached, and that gap is the CBG opportunity. We report the Shortest-Ping failure rate decomposed into two classes, alongside the RTT distribution behind it, because the two classes are two different rooms:
- *Selection failures.* The target is VP-proximate, but the lowest-RTT VP sits in a neighboring cell. One VP could have pinned the target, so nothing here requires multilateration. Three causes separate cleanly, and they do not point at the same fix:
- *Answer-space geometry.* Even the geometrically nearest VP has a different nearest seed, because some other seed sits closer to it. RTT is not implicated, and no latency model repairs it, because Shortest-Ping can only ever return a class that some VP is discriminative for. CBG is not restricted to VP coordinates, so this is recoverable, and recovery is attributable to multilateration itself (Phases 1 and 2a) rather than to constraint handling. It is also the clearest deployment signal in the section, since a VP in that metro would remove the failure outright.
- *Indirect peering.* The winning VP wins because the shorter path is routed the long way around, so RTT ranks VPs in an order distance does not. This is stable per (VP, peer ASN) pair, and absorbing it is what per-ASN calibration in Phase 1 exists for, so its rate is the concrete evidence behind §2.2(a).
- *Congestion and measurement noise.* The RTT ordering flips between measurements instead of staying wrong. This is what constraint filtering and weighting (Phases 2b and 2c) act on. However, the impact of this is negligible in our dataset as we measure RTT over weeks and take P5 RTTs among them for each (VP, target) pair.
- *Structural failures.* The target has no discriminative VP at all, so no VP coordinate is the right answer no matter which VP wins the RTT ranking. Multilateration is strictly necessary here, and a gain is attributable to latency-to-distance and constraint shape (Phases 1 and 2a).

Both classes are fixed before any method runs, so every variant in §8.3 is reported against them separately. That is what turns "CBG beats the baseline" into a claim about which phase produced the gain. It also exposes regression, since the targets the baseline already solves are targets a variant can lose, and an aggregate accuracy number hides that trade.

The coverage ceiling and the two failure rates are settled by geometry and RTT rather than by any modeling choice, which is what makes this section an explanation of §8.1's baseline number instead of a rationalization of it. It is also what makes a cross-dataset difference attributable: where the operator and RIPE datasets report different baseline accuracies, these three numbers say which structural property is responsible, and that is the generalization claim of §7.3 restated as something testable.

### 8.3 RQ3:  Feasibility of  Latency-based methods?

##### Accuracy wise
- Shortest Ping: excel when good peering exists between network operators and hypergiants/CDN providers.
- SoI CBG: similar to shortest ping, adding more robustness to non-VP proximity. Shortest-ping VP contributes to the final accuracy.
- Vanilla CBG: oversimplified linear RTT-distance model, which cause tight constraints and thus inaccurate prediction or failures.
- Octant Hull: Loose Piece-wise Convex Hull modeling. Loose constraint, generating areas with good balance between area size and accuracy. 
- Octant Spline: Tighter Piece-wise spline modeling, could over constraints such that negative excludes truth. But still accurate and robust in general. Couldn't compete with shortest-ping/SoI when RTT is small.
- Spotter CBG: Oversimplified assumption on RTT-distance modeling using normal distribution, could create tight/inaccurate constraint for close/far-range targets, but good at mid-range targets.
##### Resource Consumption wise
Memory

Runtime

Data requirement: Shortest ping / SoI CBG do not need training data

TODO: error distance
## 9. Practical Insights · 10. Limitations · 11. Conclusion

*[Pending. v1 §7 to §9 hold the current material; migrate after §8 is settled. Limitations must name the XGBoost candidate classifier as the designated follow-up.]*
