# CBG Benchmark Paper — Related Work

## 1. Foundational CBG Papers

### Gueye et al. — Original CBG
**"Constraint-based geolocation of internet hosts"**
IMC 2004 / IEEE/ACM Transactions on Networking 2006
[ACM IMC 2004](https://dl.acm.org/doi/10.1145/1028788.1028828) · [IEEE/ACM ToN 2006](https://dl.acm.org/doi/10.1109/TNET.2006.886332)

The original CBG paper. Fits a Delay-Distance Relation (DDR) per landmark using linear regression over PlanetLab RTT measurements, then constrains the target location to the intersection of per-landmark distance circles. Multilateration via spherical circle intersection; final estimate is the centroid of intersection vertices. Sets the template for the three-phase pipeline this paper benchmarks.

### Wong et al. — Octant
**"Octant: A Comprehensive Framework for the Geolocalization of Internet Hosts"**
NSDI 2007
[USENIX NSDI 2007](https://www.usenix.org/conference/nsdi-07/octant-comprehensive-framework-geolocalization-internet-hosts)

Replaces CBG's linear DDR with a convex-hull fit (bounding spline), producing annular constraints (inner + outer radius) rather than simple disks. Also incorporates negative constraints (oceans, uninhabitable areas) to tighten feasible regions. Reports 22-mile median error vs. CBG's 89-mile and GeoPing's 68-mile on the same dataset. Our Octant spline distance model and unweighted annulus multilateration are direct implementations of this work.

### Hu et al. — Million-Scale
**"Towards geolocation of millions of IP addresses"**
IMC 2012
[ACM IMC 2012](https://dl.acm.org/doi/10.1145/2398776.2398790)

Uses a simplified 2/3c (two-thirds speed of light) RTT-to-distance model, removing the need for per-landmark calibration. Introduces a greedy VP selection algorithm that prioritizes proximity to the target, scaling CBG to geolocate ~35% of the allocated IPv4 address space. Our 2/3c distance model baseline is a direct implementation.

### Wang et al. — Street-Level
**"Towards Street-Level Client-Independent IP Geolocation"**
NSDI 2011
[ACM/USENIX NSDI 2011](https://dl.acm.org/doi/10.5555/1972457.1972494)

Three-tier algorithm refining CBG estimates to street level using landmark discovery and traceroute path analysis. Implemented in this repo as `scripts/street_level/`; not the focus of this benchmark paper.

### Darwich et al. — IMC 2023 Replication
**"Replication: Towards a Publicly Available Internet Scale IP Geolocation Dataset"**
IMC 2023
[ACM IMC 2023](https://dl.acm.org/doi/10.1145/3618257.3624801)

The only publicly available implementation of CBG algorithms (Million-Scale + Street-Level). Provides reproducible datasets via RIPE Atlas. This repo is the direct codebase; our benchmark framework (`scripts/framework/`) extends it with a modular multi-variant pipeline.

---

## 2. RTT-Distance Modeling

### "Modelling of IP Geolocation by use of Latency Measurements"
IEEE 2015 / arXiv 2020
[arXiv 2004.07836](https://arxiv.org/pdf/2004.07836)

Analyzes the correlation between network latency and geographic distance, proposes improved DDR models. Focused on Phase 1 in isolation; no cross-phase benchmark.

### "Dragoon: Advanced Modelling of IP Geolocation by use of Latency Measurements"
arXiv 2020
[arXiv 2006.16895](https://arxiv.org/abs/2006.16895)

Introduces optimized landmark placement via a greedy "Dragoon" algorithm and an advanced RTT-distance modulation approach. Evaluated in a European context. Improves Phase 1 accuracy but does not benchmark the downstream multilateration or centroid phases.

### "Delay-Distance Correlation Study for IP Geolocation"
arXiv 2019
[arXiv 1909.02439](https://arxiv.org/pdf/1909.02439)

Systematic empirical study of factors perturbing the RTT-to-distance relationship (queueing delays, non-great-circle paths). Provides theoretical grounding for why a spline model (Octant) outperforms a fixed linear model (2/3c).

**Gap:** All three papers propose Phase 1 improvements independently with no cross-variant or cross-phase evaluation. Confirms the benchmark gap this paper addresses.

---

## 3. Official / Declarative Methods (GeoFeed, rDNS)

### "Geofeeds: Revolutionizing IP Geolocation or Illusionary Promises?"
ACM Proceedings on Networking 2024
[ACM 2024](https://dl.acm.org/doi/10.1145/3676869)

Critical large-scale assessment of GeoFeed (RFC 8805/9092) accuracy and deployment promises. Concludes coverage is still limited and accuracy varies significantly by operator.

### "Geofeed Adoption and Authentication"
IEEE / arXiv 2025
[arXiv 2502.08849](https://arxiv.org/abs/2502.08849)

Surveys GeoFeed adoption at RIR and AS level. Finds ~7.76% of GeoFeed URLs inaccessible; RFC 9092 authentication mechanism lacks key security properties. Coverage gaps confirm CBG is needed as a fallback.

### "IP Geolocation through Reverse DNS"
ACM Transactions on Internet Technology 2021
[ACM TOIT 2021](https://dl.acm.org/doi/10.1145/3457611)

Parses rDNS hostnames to extract location hints; places ~54% of hostnames within 20 km of ground truth. Open-source (Microsoft). Effective for well-named infrastructure but silent on cloud/anycast IPs with opaque hostnames.

**Role in our work:** GeoFeed and rDNS form Tier 1 of the multi-tier pipeline. Their coverage and accuracy limitations (especially for anycast IPs) motivate CBG as the necessary empirical fallback tier.

---

## 4. Anycast Geolocation

### Cai et al. — iGreedy
**"Latency-Based Anycast Geolocation: Algorithms, Software, and Data Sets"**
IEEE JSAC 2016
[IEEE JSAC 2016](https://ieeexplore.ieee.org/document/7470242/)

iGreedy: iterative algorithm combining latency-based enumeration (maximizing non-overlapping latency disks across VPs) with city-level geolocation via population-weighted classification. SOTA for anycast geolocation but (a) requires prior knowledge that the IP is anycast, and (b) is adversely affected by network processing delay noise. Does not benchmark CBG-based approaches on anycast targets.

### "A Fistful of Pings: Accurate and Lightweight Anycast Enumeration and Geolocation"
IEEE INFOCOM 2015
[IEEE INFOCOM 2015](https://ieeexplore.ieee.org/document/7218670/)

Lightweight baseline for anycast site enumeration using ping latency. Establishes that simple ping-based approaches can enumerate anycast replicas at low probing cost.

### "LACeS: An Open, Fast, Responsible, and Efficient Longitudinal Anycast Census System"
arXiv 2025
[arXiv 2503.20554](https://arxiv.org/pdf/2503.20554)

Large-scale daily anycast census using iGreedy as the geolocation backend. Reduces probing cost by ~100× by pre-filtering with BGP anycast detection before running iGreedy. Confirms iGreedy remains SOTA but highlights its noise sensitivity at scale.

### "Locating and Enumerating Anycast: a Comparison of Two Approaches"
ACM ANRW 2025
[ACM ANRW 2025](https://dl.acm.org/doi/10.1145/3744200.3744783)

Head-to-head comparison of traceroute-based vs latency-based (iGreedy) anycast localization. Traceroute achieves marginal precision gain but at 4× the probing cost — not viable for mobile-operator-scale deployment.

### "Regional IP Anycast: Deployments, Performance, and Potentials"
ACM SIGCOMM 2023
[ACM SIGCOMM 2023](https://dl.acm.org/doi/10.1145/3603269.3604846)

Characterizes regional anycast deployment patterns and performance implications. Provides context for why anycast geolocation is non-trivial and why commercial services fail on anycast IPs.

**Gap:** All existing anycast geolocation methods assume the IP is known to be anycast and use purpose-built algorithms (iGreedy). CBG-based multilateration has never been evaluated on anycast targets. Our hypothesis: CBG degrades gracefully on anycast (per-VP latency constraints remain valid) whereas commercial services fail entirely.

---

## 5. Commercial Geolocation Accuracy / Criticism

### "Accuracy and Coverage Analysis of IP Geolocation Databases"
IEEE 2023
[IEEE 2023](https://ieeexplore.ieee.org/document/10167899/)

Evaluates MaxMind, DBIP, IP2Location, and IPGeolocationIO accuracy across the full IPv4 address space. Accuracy varies significantly by region and IP type; mobile and anycast IPs are systematically under-served.

### "IP geolocation databases: unreliable?"
ACM SIGCOMM CCR 2011
[ACM CCR 2011](https://dl.acm.org/doi/10.1145/1971162.1971171)

Early influential critique of commercial DB accuracy showing large median errors and frequent gross mislocations. Established the community's skepticism of commercial services as ground truth.

### "GPS-Based Geolocation of Consumer IP Addresses"
PAM 2022
[PAM 2022](https://dl.acm.org/doi/10.1007/978-3-030-98785-5_6)

Uses GPS-tagged user requests as ground truth to evaluate commercial geolocation services. Finds significant errors, especially for mobile and residential IPs.

**Role in our work:** These papers collectively motivate our position that commercial services are insufficient as the sole geolocation layer for mobile operators — they are opaque, inaccurate on mobile/anycast IPs, and unauditable.

---

## 6. Recent Adjacent Work

### "Leveraging Traceroute Inconsistencies to Improve IP Geolocation"
arXiv 2025
[arXiv 2501.15064](https://arxiv.org/html/2501.15064v1)

Improves geolocation by detecting and exploiting topological inconsistencies in traceroute paths. Topology-based approach — not scalable to Internet scale (requires traceroute per target) but citable as a complementary direction for high-value targets.

### "GeoFINDR: Practical Approach to Verify Cloud Instances Geolocation in Multicloud"
arXiv April 2025
[arXiv 2504.18685](https://arxiv.org/abs/2504.18685)

RIPE Atlas-based delay approach for VM-scale cloud instance localization in multicloud environments. Achieves 22.6 km median accuracy. Close in motivation (cloud provider localization via latency) but focuses on compliance verification rather than CBG pipeline benchmarking. Differentiate by: (a) we benchmark CBG variants rather than propose a single method, (b) we focus on unicast IP geolocation at scale rather than VM compliance auditing.

---

## 7. IP Geolocation Method Landscape

This section covers the broader landscape of unicast IP geolocation methods — necessary context for justifying why CBG is the right approach and what alternatives fail to address.

### GeoPing — Closest-VP Heuristic
No standalone paper; compared directly against CBG and Octant in Wong et al. (NSDI 2007).

Picks the vantage point with the lowest RTT to the target and reports that VP's location as the estimate. Single-VP, no multilateration, no geometric constraint. On the Octant benchmark dataset: GeoPing achieves **68-mile median error** — worse than Octant (22 miles) but better than original CBG (89 miles, due to its loose LP distance model). The VP-proximity heuristic works as a rough approximation but is systematically inaccurate when the nearest VP is not co-located with the target.

### Padmanabhan & Subramanian — GeoCluster / Prefix-Based
**"An Investigation of Geographic Mapping Techniques for Internet Hosts"**
IMC/WWW 2001
[ACM WWW 2001](https://dl.acm.org/doi/10.1145/383059.383073)

Propagates known location labels through BGP prefixes: if any IP in a /24 is known, all IPs in that prefix are assigned the same location. Works for coarse-grained geolocation when prefixes are small and belong to a single operator. Fails for large ISP prefixes spanning multiple cities, produces silent errors (returns a location without indicating confidence), and is inapplicable to anycast IPs. Foundational work demonstrating prefix clustering as a practical fallback.

### Li et al. — Graph Neural Network (Street-Level ML)
**"Connecting the Hosts: Street-Level IP Geolocation with Graph Neural Networks"**
KDD 2022
[ACM KDD 2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539049)

Reframes unicast IP geolocation as node regression on attribute graphs combining network topology and RTT measurements. State-of-the-art for supervised ML approaches; achieves street-level accuracy on the evaluated datasets. **Limitation for our use case:** Requires large labeled training datasets (known ground-truth IP locations) to train the model. Not auditable (inference is a black box). Does not generalize to IPs outside the training distribution. Cannot be applied to the long tail of unlabeled IPs at Internet scale without substantial labeled data collection.

### Jiang — Neural Network with Stable Landmarks
**"IP Geolocation Estimation using Neural Networks with Stable Landmarks"**
SIGCOMM GI Workshop 2016
[IEEE 2016](https://ieeexplore.ieee.org/document/7562066/)

Neural network classifier on RTT feature vectors collected from stable landmark nodes. Achieves **4.1 km median error** on a US dataset with 1547 landmarks. Demonstrates the accuracy ceiling achievable by supervised ML when training data is available. Same limitations as above: depends on curated labeled training set and does not scale to unlabeled IPs.

### Topology-Based: Intermediate Routers as Landmarks
**"Towards IP Geolocation with Intermediate Routers Based on Topology Discovery"**
Cybersecurity (SpringerOpen) 2019
[Springer 2019](https://cybersecurity.springeropen.com/articles/10.1186/s42400-019-0030-2)

Uses traceroutes to discover intermediate routers as secondary landmarks, which are then geolocated and used to tighten CBG constraints. Improves accuracy in landmark-sparse regions. **Scalability limitation:** Requires a traceroute (10–20 probes) per target IP. At 10M+ IPs, this represents 100M–200M probes per measurement cycle — prohibitive for operational use at ISP scale. Active measurement budgets (RIPE Atlas credits, probe bandwidth) further constrain this approach.

### Efficient Landmark Selection for Active Geolocation
**"Selection of Landmarks for Efficient Active Geolocation"**
TMA 2024
[IEEE TMA 2024](https://ieeexplore.ieee.org/document/10559002/)

Proposes optimal landmark selection strategies for active geolocation (minimizing probing while maximizing coverage). Shows that even geographic distribution of landmarks significantly improves CBG precision, especially in underserved regions (Africa, South America). Relevant to our VP selection evaluation in the benchmark; confirms that landmark placement is a first-order factor in CBG accuracy.

**Gap:** None of these alternatives — GeoPing, GeoCluster, ML/GNN, topology-based — is suitable as a general-purpose unicast geolocation fallback at Internet scale. GeoPing and GeoCluster lack accuracy guarantees; ML/GNN requires labeled training data; topology-based methods are too expensive to probe. CBG occupies the unique position of being physics-grounded, label-free, and scalable to millions of IPs with a fixed measurement infrastructure.

---

## Summary Table

| Paper | Year | Phase(s) | Venue | Relation |
|-------|------|----------|-------|----------|
| Gueye et al. — CBG | 2004/2006 | 1+2+3 | IMC/ToN | Baseline: LP model + spherical intersection |
| Wong et al. — Octant | 2007 | 1+2+3 | NSDI | Baseline: spline model + annulus multilateration |
| Hu et al. — Million-Scale | 2012 | 1+3 | IMC | Baseline: 2/3c model + VP selection |
| Wang et al. — Street-Level | 2011 | — | NSDI | In-repo; out of scope |
| Darwich et al. — IMC 2023 | 2023 | 1+2+3 | IMC | Direct codebase; only public CBG impl |
| Modelling of IP Geolocation | 2015/2020 | 1 | IEEE/arXiv | Phase 1 improvement only |
| Dragoon | 2020 | 1 | arXiv | Phase 1 improvement only |
| Delay-Distance Correlation | 2019 | 1 | arXiv | Phase 1 analysis |
| Geofeeds | 2024 | Tier 1 | ACM Networking | Motivates CBG as fallback |
| Geofeed Adoption | 2025 | Tier 1 | IEEE/arXiv | Motivates CBG as fallback |
| rDNS Geolocation | 2021 | Tier 1 | ACM TOIT | Motivates CBG as fallback |
| iGreedy | 2016 | Anycast | IEEE JSAC | SOTA anycast geoloc; doesn't benchmark CBG |
| Fistful of pings | 2015 | Anycast | IEEE INFOCOM | Anycast enumeration baseline |
| LACeS | 2025 | Anycast | arXiv | Large-scale anycast census using iGreedy |
| Anycast comparison | 2025 | Anycast | ACM ANRW | Traceroute vs latency comparison |
| Regional Anycast | 2023 | Anycast | SIGCOMM | Anycast deployment context |
| DB accuracy | 2023 | Eval | IEEE | Motivates open alternatives |
| DB unreliable | 2011 | Eval | ACM CCR | Motivates open alternatives |
| GPS-based | 2022 | Eval | PAM | Motivates open alternatives |
| Traceroute inconsistencies | 2025 | Topology | arXiv | Adjacent; out of scope (not scalable) |
| GeoFINDR | 2025 | CBG-like | arXiv | Adjacent; different goal (VM compliance) |
| Padmanabhan & Subramanian — GeoCluster | 2001 | Tier 1 alt | WWW | Prefix-based fallback; fails on large ISP prefixes |
| Li et al. — GNN Street-Level | 2022 | ML | KDD | SOTA supervised ML; requires labeled training data |
| Jiang — NN + Stable Landmarks | 2016 | ML | GI/SIGCOMM | Neural network baseline; labeled data required |
| Topology + Intermediate Routers | 2019 | Topology | Cybersecurity | Traceroute-based; infeasible probing cost at scale |
| Landmark Selection (TMA 2024) | 2024 | Phase 1 | TMA | Landmark placement is first-order accuracy factor |
