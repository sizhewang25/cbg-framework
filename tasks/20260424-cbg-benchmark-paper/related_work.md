# CBG Benchmark Paper — Related Work

## 1. Foundational CBG Papers

### Gueye et al. — Original CBG
**"Constraint-based geolocation of internet hosts"**
IMC 2004 / IEEE/ACM Transactions on Networking 2006
[ACM IMC 2004](https://dl.acm.org/doi/10.1145/1028788.1028828) · [IEEE/ACM ToN 2006](https://dl.acm.org/doi/10.1109/TNET.2006.886332)

The original CBG paper. Fits a Delay-Distance Relation (DDR) per landmark using linear regression over PlanetLab RTT measurements, then constrains the target location to the intersection of per-landmark distance circles. Multilateration via `spherical_circle`; final estimate is the centroid of intersection vertices. Sets the template for the three-phase pipeline this paper benchmarks.

### Wong et al. — Octant
**"Octant: A Comprehensive Framework for the Geolocalization of Internet Hosts"**
NSDI 2007
[USENIX NSDI 2007](https://www.usenix.org/conference/nsdi-07/octant-comprehensive-framework-geolocalization-internet-hosts)

Replaces CBG's linear DDR with a convex-hull fit (bounding spline), producing annular constraints (inner + outer radius) rather than simple disks. Also incorporates negative constraints (oceans, uninhabitable areas) to tighten feasible regions. Reports 22-mile median error vs. CBG's 89-mile and GeoPing's 68-mile on the same dataset. Our Octant spline distance model and `planar_annulus` multilateration are direct implementations of this work.

### Hu et al. — Million-Scale
**"Towards geolocation of millions of IP addresses"**
IMC 2012
[ACM IMC 2012](https://dl.acm.org/doi/10.1145/2398776.2398790)

Uses a simplified 2/3c (two-thirds speed of light) RTT-to-distance model, removing the need for per-landmark calibration. Introduces a greedy VP selection algorithm that prioritizes proximity to the target, scaling CBG to geolocate ~35% of the allocated IPv4 address space. Our 2/3c distance model baseline is a direct implementation.

### Chandrasekaran et al. — Alidade
**"Alidade: IP Geolocation without Active Probing"**
Duke University Technical Report CS-TR-2015-001, January 2015; revised April 2015
[PDF](https://users.cs.duke.edu/~bmm/assets/pubs/alidade--cs-tr-2015-001.pdf)

Strong end-to-end system work from Akamai and coauthors including one of the Octant authors. Alidade tries to unify passive measurement data, non-measurement geolocation datasets, and auxiliary sources to geolocate the full IP address space. It keeps CBG's constraint-region view but changes the deployment model: rather than issuing active probes at query time, it fuses already-available constraints, resolves conflicts among them, and precomputes a joint solution for IP space. Each prediction includes both a representative point and a geographic region, making it closer to CBG/Octant than to opaque commercial databases.

Alidade is built on top of the Octant line of work, including its traceroute-extension direction, but simplifies the CBG integration. Specifically, it adopts the simplest 2/3c speed-of-Internet RTT-to-distance mapping, polygon-based multilateration and intersection-region optimization, and does not specify a distinct centroid-selection technique. While Alidade focuses on end-to-end system design and whole-Internet precomputation, we argue that the individual CBG variants still deserve careful understanding and benchmarking. Alidade does not compare against other CBG variants or isolate phase-level contributions, leaving the gap this paper addresses.

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

### Arif et al. — GeoWeight
**"GeoWeight: Internet Host Geolocation Based on a Probability Model for Latency Measurements"**
ACSC 2010
[ACM DL](https://dl.acm.org/doi/10.5555/1862219.1862232)

Models RTT-distance behavior probabilistically instead of using a single deterministic delay-distance curve. GeoWeight is relevant because it is one of the Alidade-cited latency-modeling alternatives to CBG: it improves the handling of noisy latency measurements, but it is not organized as a modular CBG pipeline and does not isolate the downstream multilateration or centroid effects.

### Laki et al. — Spotter
**"Spotter: A Model Based Active Geolocation Service"**
IEEE INFOCOM 2011
[IEEE INFOCOM 2011](https://ieeexplore.ieee.org/document/5935225)

Builds a model-based active geolocation service that estimates target locations from delay measurements against a calibrated landmark set. Spotter is another Alidade-cited active geolocation system and belongs with RTT-distance modeling work because its main contribution is a learned/model-based delay-to-location service rather than a new CBG multilateration variant. It reinforces the point that many papers improve the modeling stage without comparing CBG phase combinations.

**Gap:** These papers propose Phase 1 or model-level improvements independently with no cross-variant or cross-phase evaluation. Confirms the benchmark gap this paper addresses.

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

### Guo et al. — Web-Mined Geolocation
**"Mining the Web and the Internet for Accurate IP Address Geolocations"**
IEEE INFOCOM 2009
[IEEE INFOCOM 2009](https://doi.org/10.1109/INFCOM.2009.5062197)

Mines web pages and network data to extract IP-to-location evidence, then combines noisy hints to improve geolocation accuracy. This is an important Alidade antecedent because Alidade similarly unifies measurement and non-measurement sources. It is complementary to our benchmark: web-mined hints can feed an end-to-end geolocation system, but they are not a CBG phase variant.

### Scheitle et al. — HLOC
**"HLOC: Hints-Based Geolocation Leveraging Multiple Measurement Frameworks"**
TMA 2017
[IEEE TMA 2017](https://doi.org/10.23919/TMA.2017.8002903)

Extracts location hints from DNS names and validates them using latency measurements from public measurement platforms such as RIPE Atlas. HLOC is relevant because it bridges declarative rDNS evidence and measurement-based validation: unlike pure rDNS parsing, it can reject implausible hostname hints. It is complementary to our pipeline's Tier 1 sources; our CBG benchmark focuses on the empirical fallback when such hints are unavailable, ambiguous, or insufficiently trustworthy.

### Izhikevich et al. — Operator-Reported Geolocation
**"Trust, But Verify, Operator-Reported Geolocation"**
arXiv 2024
[arXiv 2409.19109](https://arxiv.org/abs/2409.19109)

Audits operator-reported geolocation for RIPE Atlas vantage points and shows that misreported VP locations, while rare overall, can disproportionately affect coverage in underrepresented regions. This matters for our benchmark because RIPE Atlas probes and anchors are both measurement infrastructure and ground-truth/reference points; we should validate or filter operator-reported coordinates before treating them as reliable landmarks.

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

### Gouel et al. — Database Stability
**"IP Geolocation Database Stability and Implications for Network Research"**
TMA 2021
[PDF](https://dl.ifip.org/db/conf/tma/tma2021/tma2021-paper2.pdf)

Studies longitudinal changes in MaxMind snapshots and shows that database version choice can materially change research results. This is directly relevant to our evaluation protocol: whenever IPInfo or MaxMind is used as supporting evidence rather than authoritative ground truth, the paper should report the database provider, edition, and snapshot date.

**Role in our work:** These papers collectively motivate our position that commercial services are insufficient as the sole geolocation layer for mobile operators — they are opaque, inaccurate on mobile/anycast IPs, and unauditable.

---

## 6. Recent Adjacent Work

### Rimlinger et al. — GeoResolver
**"GeoResolver: An Accurate, Scalable, and Explainable Geolocation Technique Using DNS Redirection"**
Proceedings of the ACM on Networking / CoNEXT 2025
[ACM 2025](https://doi.org/10.1145/3749219)

Uses DNS redirection behavior, including EDNS Client Subnet, to infer which RIPE Atlas vantage points are likely to be informative for a target IP, reducing the measurement cost of active geolocation while keeping estimates explainable. GeoResolver is adjacent rather than a CBG phase variant: it targets scalable VP selection and DNS-redirection-derived locality, whereas our benchmark asks which CBG distance model, multilateration method, and point estimator should be used once RTT constraints are available.

### Du et al. — RIPE IPmap
**"RIPE IPmap Active Geolocation: Mechanism and Performance Evaluation"**
ACM SIGCOMM CCR 2020
[ACM CCR 2020](https://doi.org/10.1145/3402413.3402415)

Introduces and evaluates RIPE IPmap's single-radius active geolocation engine, including accuracy, coverage, and consistency against ground truth and commercial databases. This is important operational context for our RIPE Atlas evaluation: IPmap demonstrates that public measurement infrastructure can support active geolocation, while our work decomposes the CBG algorithmic choices that such systems can use internally.

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

### Freedman et al. — Prefix Locality
**"Geographic Locality of IP Prefixes"**
IMC 2005
[ACM IMC 2005](https://doi.org/10.5555/1251086.1251099)

Studies whether IP prefixes exhibit geographic locality and how reliably prefix structure can be used for geolocation. This is relevant to Alidade's full-IP-space goal because prefix-level aggregation is a natural way to propagate sparse geolocation evidence. For our work, prefix locality is an auxiliary source rather than a CBG variant, and it does not answer which latency-constraint algorithm should be used when RTT observations are available.

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

### Youn et al. — Statistical Geolocation
**"Statistical Geolocation of Internet Hosts"**
ICCCN 2009
[IEEE ICCCN 2009](https://doi.org/10.1109/ICCCN.2009.5235373)

Replaces deterministic delay-to-distance constraints with a statistical model: kernel density estimation over landmark delay measurements followed by maximum-likelihood location estimation. Relevant as an early probabilistic alternative to CBG. It improves over deterministic bounds in the evaluated PlanetLab setting, but it is not a modular CBG variant and does not address the phase-level accuracy/runtime tradeoffs we benchmark.

### Eriksson et al. — Learning-Based IP Geolocation
**"A Learning-Based Approach for IP Geolocation"**
PAM 2010
[Springer PAM 2010](https://doi.org/10.1007/978-3-642-12334-4_18)

Frames IP geolocation as a Naive Bayes classification problem using lightweight measurements from monitors to targets. It reports improvements over prior CBG-style baselines, but depends on learned probability densities from training data. It belongs with the supervised/statistical alternatives rather than the CBG variant set because it does not produce explicit physical feasible regions.

### Eriksson et al. — Posit
**"Posit: A Lightweight Approach for IP Geolocation"**
ACM SIGMETRICS Performance Evaluation Review 2012
[ACM SIGMETRICS PER 2012](https://doi.org/10.1145/2381056.2381058)

Proposes a lightweight geolocation method based on a small number of measurements and probabilistic inference. Posit is useful to cite because Alidade includes it among the active/statistical geolocation lineage. Like the learning-based and statistical approaches above, it optimizes a different point in the design space: fewer measurements and probabilistic location inference rather than explicit CBG feasible-region construction and phase-level benchmarking.

### Katz-Bassett et al. — Topology-Based Geolocation
**"Towards IP Geolocation Using Delay and Topology Measurements"**
IMC 2006
[ACM IMC 2006](https://doi.org/10.1145/1177080.1177090)

Introduces TBG, which combines delay constraints with traceroute-derived topology constraints and jointly solves for router and host locations. TBG is foundational for topology-assisted geolocation and should be cited before later intermediate-router approaches. Its cost profile differs from CBG because it requires topology discovery and alias/router reasoning, making it more expensive than fixed-VP RTT multilateration for large target sets.

### Topology-Based: Intermediate Routers as Landmarks
**"Towards IP Geolocation with Intermediate Routers Based on Topology Discovery"**
Cybersecurity (SpringerOpen) 2019
[Springer 2019](https://cybersecurity.springeropen.com/articles/10.1186/s42400-019-0030-2)

Uses traceroutes to discover intermediate routers as secondary landmarks, which are then geolocated and used to tighten CBG constraints. Improves accuracy in landmark-sparse regions. **Scalability limitation:** Requires a traceroute (10–20 probes) per target IP. At 10M+ IPs, this represents 100M–200M probes per measurement cycle — prohibitive for operational use at ISP scale. Active measurement budgets (RIPE Atlas credits, probe bandwidth) further constrain this approach.

### Gill et al. — Circumventing Measurement-Based Geolocation
**"Dude, Where's That IP? Circumventing Measurement-Based IP Geolocation"**
USENIX Security 2010
[USENIX Security 2010](https://www.usenix.org/conference/19th-usenix-security-symposium/dude-wheres-ip-circumventing-measurement-based-ip-geolocation)

Shows that measurement-based geolocation systems can be intentionally misled by adversarial delay manipulation. This work is not a geolocation algorithm variant, but it is an important caution for our threat model and result interpretation: CBG's physical constraints are auditable, yet RTT-derived constraints assume that targets and paths are not actively manipulating delay to appear elsewhere.

### Efficient Landmark Selection for Active Geolocation
**"Selection of Landmarks for Efficient Active Geolocation"**
TMA 2024
[IEEE TMA 2024](https://ieeexplore.ieee.org/document/10559002/)

Proposes optimal landmark selection strategies for active geolocation (minimizing probing while maximizing coverage). Shows that even geographic distribution of landmarks significantly improves CBG precision, especially in underserved regions (Africa, South America). Relevant to our VP selection evaluation in the benchmark; confirms that landmark placement is a first-order factor in CBG accuracy.

### Li et al. — GeoCAM
**"GeoCAM: An IP-Based Geolocation Service Through Fine-Grained and Stable Webcam Landmarks"**
IEEE/ACM Transactions on Networking 2021
[IEEE/ACM ToN 2021](https://doi.org/10.1109/TNET.2021.3073926)

Builds a large landmark set by discovering stable live webcams with extractable geographic coordinates, then uses latency and topology constraints to geolocate targets. GeoCAM is relevant to the benchmark's VP/landmark-sensitivity analysis: it shows that landmark density and quality can dominate geolocation accuracy. It is not a CBG phase variant, but it motivates evaluating how phase rankings change with VP count and geographic coverage.

**Gap:** None of these alternatives — GeoPing, GeoCluster, statistical/ML/GNN, topology-based — is suitable as a general-purpose unicast geolocation fallback at Internet scale under our deployment assumptions. GeoPing and GeoCluster lack accuracy guarantees; statistical and ML/GNN methods require labeled training data or learned regional densities; topology-based methods are too expensive to probe. CBG occupies the unique position of being physics-grounded, label-free, and scalable to millions of IPs with a fixed measurement infrastructure.

---

## Summary Table

| Paper | Year | Phase(s) | Venue | Relation |
|-------|------|----------|-------|----------|
| Gueye et al. — CBG | 2004/2006 | 1+2+3 | IMC/ToN | Baseline: LP model + spherical_circle intersection |
| Wong et al. — Octant | 2007 | 1+2+3 | NSDI | Baseline: spline model + `planar_annulus` multilateration |
| Hu et al. — Million-Scale | 2012 | 1+3 | IMC | Baseline: 2/3c model + VP selection |
| Chandrasekaran et al. — Alidade | 2015 | CBG-like | Tech report | Most recent CBG-family system; offline constraints, no query-time probes |
| Wang et al. — Street-Level | 2011 | — | NSDI | In-repo; out of scope |
| Darwich et al. — IMC 2023 | 2023 | 1+2+3 | IMC | Direct codebase; only public CBG impl |
| Modelling of IP Geolocation | 2015/2020 | 1 | IEEE/arXiv | Phase 1 improvement only |
| Dragoon | 2020 | 1 | arXiv | Phase 1 improvement only |
| Delay-Distance Correlation | 2019 | 1 | arXiv | Phase 1 analysis |
| GeoWeight | 2010 | Statistical | ACSC | Probabilistic latency model; Alidade-cited |
| Spotter | 2011 | Active/model | INFOCOM | Model-based active geolocation service |
| Geofeeds | 2024 | Tier 1 | ACM Networking | Motivates CBG as fallback |
| Geofeed Adoption | 2025 | Tier 1 | IEEE/arXiv | Motivates CBG as fallback |
| rDNS Geolocation | 2021 | Tier 1 | ACM TOIT | Motivates CBG as fallback |
| Web-mined geolocation | 2009 | Tier 1 alt | INFOCOM | Mines web/network hints; Alidade antecedent |
| HLOC | 2017 | Tier 1 + validation | TMA | rDNS hints validated by latency measurements |
| Trust, But Verify | 2024 | Ground truth | arXiv | Validates operator-reported VP locations |
| iGreedy | 2016 | Anycast | IEEE JSAC | SOTA anycast geoloc; doesn't benchmark CBG |
| Fistful of pings | 2015 | Anycast | IEEE INFOCOM | Anycast enumeration baseline |
| LACeS | 2025 | Anycast | arXiv | Large-scale anycast census using iGreedy |
| Anycast comparison | 2025 | Anycast | ACM ANRW | Traceroute vs latency comparison |
| Regional Anycast | 2023 | Anycast | SIGCOMM | Anycast deployment context |
| DB accuracy | 2023 | Eval | IEEE | Motivates open alternatives |
| DB unreliable | 2011 | Eval | ACM CCR | Motivates open alternatives |
| GPS-based | 2022 | Eval | PAM | Motivates open alternatives |
| DB stability | 2021 | Eval | TMA | Snapshot date affects reproducibility |
| GeoResolver | 2025 | VP selection | PACM Networking | Scalable explainable geolocation using DNS redirection |
| RIPE IPmap | 2020 | Active geoloc | ACM CCR | Operational active geolocation with RIPE Atlas |
| Traceroute inconsistencies | 2025 | Topology | arXiv | Adjacent; out of scope (not scalable) |
| GeoFINDR | 2025 | CBG-like | arXiv | Adjacent; different goal (VM compliance) |
| Padmanabhan & Subramanian — GeoCluster | 2001 | Tier 1 alt | WWW | Prefix-based fallback; fails on large ISP prefixes |
| Freedman et al. — Prefix Locality | 2005 | Prefix/locality | IMC | Prefix-level propagation evidence |
| Li et al. — GNN Street-Level | 2022 | ML | KDD | SOTA supervised ML; requires labeled training data |
| Jiang — NN + Stable Landmarks | 2016 | ML | GI/SIGCOMM | Neural network baseline; labeled data required |
| Youn et al. — Statistical Geolocation | 2009 | Statistical | ICCCN | Probabilistic delay model; not modular CBG |
| Eriksson et al. — Learning-Based | 2010 | ML/statistical | PAM | Naive Bayes geolocation from monitor measurements |
| Eriksson et al. — Posit | 2012 | Statistical | SIGMETRICS PER | Lightweight probabilistic geolocation |
| Katz-Bassett et al. — TBG | 2006 | Topology | IMC | Foundational delay+topology geolocation |
| Topology + Intermediate Routers | 2019 | Topology | Cybersecurity | Traceroute-based; infeasible probing cost at scale |
| Gill et al. — Circumvention | 2010 | Security | USENIX Security | Adversarial limits of measurement-based geolocation |
| Landmark Selection (TMA 2024) | 2024 | Phase 1 | TMA | Landmark placement is first-order accuracy factor |
| GeoCAM | 2021 | Landmarks | ToN | Large webcam landmark set; motivates VP/landmark analysis |
