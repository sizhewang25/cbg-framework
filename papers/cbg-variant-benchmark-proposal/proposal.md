# Constraint-Based Geolocation for Internet Hosts at Scale: Which Variant Is Production-Ready?

---

## 1. Project Objective

An operator who wants to deploy Constraint-Based Geolocation (CBG) today faces a practical problem: three landmark papers have been proposed for decades, none has been benchmarked against the others in a controlled setting, and no paper characterizes compute cost at operational scale. There is no evidence-based guide for choosing a configuration.

We close this gap with the **first systematic, cross-variant CBG benchmark** for unicast IP geolocation. We decompose CBG into three independent phases — RTT-to-distance modeling, multilateration, and single-point estimation — and evaluate the accuracy/runtime tradeoff of each valid configuration family using curated RTT datasets from operational vantage points and RIPE Atlas.

> **Comment:** The first full-dataset experiment now supports the production-readiness argument, but it covers the default 8-combination suite over all Vultr US targets, not yet the full 15 active valid combinations. Keep this distinction explicit until the remaining combinations and RIPE cross-validation are complete.

Our contributions:

1. A three-phase CBG taxonomy that unifies disparate prior implementations
2. A modular open-source framework implementing all known CBG variants (original CBG and Octant have no public code)
3. Controlled cross-variant benchmark results isolating the per-phase contribution to accuracy and runtime
4. A Pareto frontier of median error vs. runtime per IP — the first evidence-based answer to "which CBG variant is production-ready?"

---

## 2. Motivation

### 2.1 Why Latency-Based Geolocation?

ISPs, CDN operators, network researchers, and security analysts need to know where Internet hosts are physically located — for traffic engineering, CDN server selection, regulatory compliance, anomaly detection, and capacity planning. No single method covers all unicast IPs with high accuracy.

Declarative sources (GeoFeed, rDNS) have limited coverage [[ACM 2024](https://dl.acm.org/doi/10.1145/3676869), [ACM TOIT 2021](https://dl.acm.org/doi/10.1145/3457611)]. Other approaches each have a fundamental deployment barrier:


| Method                                                                                                                                           | Key Limitation                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| **GeoCluster** [[WWW 2001](https://dl.acm.org/doi/10.1145/383059.383073)]                                                                        | Silent failure for large ISP prefixes spanning multiple cities                          |
| **Commercial DBs** [[IEEE 2023](https://ieeexplore.ieee.org/document/10167899/), [ACM CCR 2011](https://dl.acm.org/doi/10.1145/1971162.1971171)] | Opaque, inaccurate on cloud/mobile IPs; not auditable                                   |
| **ML/GNN** [[KDD 2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539049)]                                                                      | Requires large labeled training datasets; black-box; cannot generalize to unlabeled IPs |
| **Topology-based** [[Cybersecurity 2019](https://cybersecurity.springeropen.com/articles/10.1186/s42400-019-0030-2)]                             | 10–20 probes per IP × 10M targets = 100–200M probes per cycle; infeasible at ISP scale  |


Latency-based geolocation avoids all of these barriers. It requires no labeled training data and no per-IP active probing beyond the RTT data ISPs already observe passively — PoPs and mobile core networks see round-trip times to communicating hosts as a byproduct of normal traffic. Furthermore, **any geolocation estimate can be validated against observed RTTs**: `distance ≤ RTT/2 × propagation speed` is a hard physical bound — any estimate violating it is provably wrong.

### 2.2 Why CBG Among Latency-Based Methods?

The simplest latency-based approach, **GeoPing**, reports the nearest VP's location (lowest RTT) — a single-VP heuristic with no geometric constraint, producing a **68-mile median error** [[NSDI 2007](https://www.usenix.org/conference/nsdi-07/octant-comprehensive-framework-geolocalization-internet-hosts)] that degrades whenever no VP is co-located with the target.

**CBG** converts RTTs from multiple VPs into geographic constraints (circles or annuli) and intersects them to find the feasible region where the target must reside — aggregating independent physical evidence rather than trusting a single VP:

- **More accurate** — Octant (best CBG variant) achieves **22-mile median error** [[NSDI 2007](https://www.usenix.org/conference/nsdi-07/octant-comprehensive-framework-geolocalization-internet-hosts)], 3× better than GeoPing; on our full Vultr US cloud dataset, the best practical CBG variant reaches **359.7 km median error** vs. **522.7 km** for the speed-of-Internet spherical baseline and **559.2 km** for the LP spherical baseline
- **Auditable** — raw RTTs → explicit circles → intersection; every step is inspectable and reproducible
- **Label-free** — works for any IP reachable from the VP set, no training data needed
- **Tunable** — accuracy improves directly as more or better-distributed VPs are added

CBG is therefore the practical choice for operators who need high-accuracy, auditable geolocation at scale. **But which CBG variant?** That question has never been answered.

### 2.3 Why Practitioners Cannot Choose a Variant Today

**No controlled cross-variant evaluation exists.** Each CBG paper (Gueye 2004, Hu 2012, Wong 2007) evaluates only its own full pipeline on its own proprietary dataset. No paper isolates phase contributions, so practitioners cannot determine: Is it worth calibrating a per-VP spline, or does 2/3c suffice? Does `planar_annulus` multilateration justify its added complexity over `spherical_circle`? Does Monte Carlo sampled-medoid selection deliver meaningful accuracy gains over a geometric centroid? Does weighted-annulus geometry justify its pathological tail latency? Our first full-dataset run answers the latter two questions negatively, but a full phase-isolation grid is still required for the final paper.

**Only one public CBG implementation exists.** The IMC 2023 replication codebase [[Darwich et al.](https://dl.acm.org/doi/10.1145/3618257.3624801)] covers only Million-Scale and Street-Level CBG. Original CBG and Octant have **no public code** — an operator who wants to evaluate Octant must reimplement it from scratch.

### 2.4 What the Benchmark Delivers

Accuracy benchmarks alone are insufficient for deployment decisions. Despite IP geolocation being studied for over two decades, none of the foundational CBG papers report per-IP runtime or memory figures — the field has never evaluated algorithms against a deployment budget. At operational scale (10M–50M IPs), this matters: in our full Vultr run, the practical `planar_annulus + geometric_centroid` setting geolocates a target in **8.7 ms on average**, while the unweighted Monte Carlo sampled-medoid variant takes **160.1 ms on average** and is less accurate. The weighted-annulus path is more severe: it improves median error by only 2.1 km over the practical setting but spends ~40.7 minutes on one pathological target.

The final benchmark will provide the first Pareto frontier of median error vs. runtime per IP across all valid CBG phase combinations — a concrete, evidence-based answer to which configuration maximizes accuracy within a given compute budget, and where future phase-level investment will have the most impact.

---

## 3. The CBG Pipeline Abstraction

A key insight of this work is that **every published CBG variant can be decomposed into three independent phases**, each with interchangeable implementations. This abstraction enables systematic cross-variant benchmarking for the first time.

The implementation also supports optional constraint filtering between Phase 1 and Phase 2. We treat this as Phase 1.5 preprocessing rather than a fourth conceptual CBG phase because it does not create a new geolocation representation; it only decides which Phase 1 constraints proceed to multilateration.

```
Input: RTT measurements from N vantage points (VPs) to target IP
          ↓
Phase 1: RTT-to-Distance Modeling
         Convert per-VP RTT → distance constraint (radius or annulus)
          ↓
Optional preprocessing: Constraint Filtering
         Remove redundant or invalid constraints when enabled
          ↓
Phase 2: Multilateration
         Intersect per-VP constraints → feasible region (geometry)
          ↓
Phase 3: Single-Point Estimation
         Collapse feasible region → estimated (lat, lon)
          ↓
Output: Geolocation estimate
```

### Phase 1 — RTT-to-Distance Modeling


| Variant                      | Source                   | Method                                                                  | Output                         |
| ---------------------------- | ------------------------ | ----------------------------------------------------------------------- | ------------------------------ |
| **2/3c (Speed-of-Internet)** | Million-Scale (IMC 2012) | Fixed constant: `radius = RTT/2 × 2c/3`                                 | Disk (outer radius only)       |
| **LP Low-Envelope**          | Original CBG (IMC 2004)  | Per-VP linear regression fitted to RTT-distance scatter via LP bestline | Disk (outer radius only)       |
| **Bounded Spline**           | Octant (NSDI 2007)       | Per-VP spline fit + shared delta band calibrated to coverage target     | Annulus (inner + outer radius) |


The spline model produces **annuli** rather than disks, encoding both a maximum and minimum distance from each VP — a fundamentally tighter constraint.

### Phase 2 — Multilateration


| Variant                             | Source                       | Method                                                                     | Output          |
| ----------------------------------- | ---------------------------- | -------------------------------------------------------------------------- | --------------- |
| **`spherical_circle`**         | Original CBG / Million-Scale | Pairwise great-circle crossings on the Earth sphere; keeps points inside all circles | Vertex list     |
| **`planar_circle`**            | —                            | Approximate circles as 100-point polygons in `(lon, lat)` degree space; sequential Shapely intersection | Shapely polygon |
| **`planar_annulus`**           | Octant                       | `∩(outer disks) − ∪(inner disks)` in planar `(lon, lat)` degree space      | Shapely polygon |
| **`planar_annulus_weighted`**  | Octant                       | Grid-based weight accumulation over planar annuli; fused Phase 2+3         | Shapely polygon |


`planar_annulus` is qualitatively different from `planar_circle`: by subtracting the inner exclusion zones, it removes near-VP regions that are geometrically inconsistent with the RTT constraint — producing a tighter, more accurate feasible region.

### Phase 3 — Single-Point Estimation


| Variant                 | Source       | Method                                                           | Complexity   |
| ----------------------- | ------------ | ---------------------------------------------------------------- | ------------ |
| **`boundary_vertex_mean`** | Original CBG | Average of boundary vertex coordinates; includes polygon holes for annuli | O(1)         |
| **Geometric Centroid**  | —            | Area-weighted centroid of feasible polygon (Shapely `.centroid`) | O(1)         |
| **MC Sampled Medoid**   | Octant       | 1000-point Sobol QMC sampling + sampled point with minimum total pairwise distance | O(n_samples²) |


The MC sampled medoid minimizes sum of distances to all sampled feasible points and returns one of those points, making it robust to irregular polygon shapes while preserving feasibility. However, it incurs a large runtime penalty versus the geometric centroid.

### Valid Phase Combinations

Not all combinations are valid due to type constraints:

- `planar_annulus` and `planar_annulus_weighted` require the spline distance model (annuli as input)
- `boundary_vertex_mean` works on `spherical_circle` vertex lists and on planar polygon boundary rings
- `geometric_centroid` requires a Shapely polygon; it is not valid for unordered `spherical_circle` crossing vertices
- MC sampled medoid can operate on planar polygon samples or directly on `spherical_circle` vertex point sets

The framework defines **15 active valid combinations** across these constraints. The first full Vultr all-US experiment evaluates the default 8-combination suite (see Section 6); the final benchmark should run the remaining valid combinations before claiming a complete 15-way Pareto frontier. The historical 18-combination run included `spherical_circle + geometric_centroid`, which is now excluded because unordered spherical crossing vertices cannot reliably form a polygon.

---

## 4. Challenges

### 4.1 RTT-Distance Modeling Accuracy

The fundamental challenge of CBG is that the RTT-to-distance mapping is noisy. For an RTT of 20ms, the actual distance can range from 250 to 1100 km. Simple linear models (2/3c) are fast but systematically over-estimate distances in high-latency tails. Per-VP fitted models (LP, spline) are more accurate but require calibration data. The spline model's annulus output captures this uncertainty explicitly — but calibration requires a sufficiently dense set of anchor RTT measurements near the target.

### 4.2 Multilateration Failure (Zero Intersection)

When RTT constraints are noisy or the VP set is poorly chosen, the intersection of distance circles may be empty. All existing methods handle this differently (fallback to nearest VP, centroid of closest circles, barycenter). Consistent failure handling is critical for a fair benchmark and for production use. Our implementation records fallback usage separately from successful multilateration so availability, intersection rate, and accuracy metrics are not conflated.

### 4.3 Scalability vs. Accuracy Tradeoff

The full Vultr all-US run changes the scalability story. The most accurate evaluated setting is weighted annulus + geometric centroid (`B3`) at 357.6 km median error, but it has pathological geometry tail latency: one target spent ~40.7 minutes in multilateration. The practical setting is unweighted annulus + geometric centroid (`B4`) at 359.7 km median error and 8.7 ms mean geolocation time per target. The Monte Carlo sampled-medoid variants are dominated in this run: they are slower and less accurate than geometric centroid for both weighted and unweighted annulus paths.

### 4.4 VP Selection and Coverage

CBG accuracy depends heavily on which vantage points are selected and how many. At mobile operator scale, the VP set is fixed (operator's own measurement infrastructure). Unlike academic settings where any RIPE Atlas probe can be used, we must evaluate CBG under realistic VP count and distribution constraints.

### 4.5 Ground Truth Availability

Authoritative ground truth for unicast IPs is difficult to obtain at scale. RIPE Atlas anchors provide verified coordinates for ~500 well-geolocated reference points. For broader evaluation, we rely on IPInfo and MaxMind cross-reference — both of which have known inaccuracies that must be accounted for in result interpretation.

---

## 5. Related Work

### 5.1 Foundational CBG Papers

**Gueye et al., "Constraint-based geolocation of internet hosts"** (IMC 2004 / IEEE/ACM ToN 2006) [[ACM](https://dl.acm.org/doi/10.1145/1028788.1028828)]
Original CBG. Fits a DDR per landmark via linear regression (PlanetLab). Introduces the three-phase structure (modeling → intersection → centroid) that all subsequent CBG variants follow. Defines the LP low-envelope Phase 1 variant and `spherical_circle` Phase 2 variant used as our baselines.

**Wong et al., "Octant: A Comprehensive Framework for the Geolocalization of Internet Hosts"** (NSDI 2007) [[USENIX](https://www.usenix.org/conference/nsdi-07/octant-comprehensive-framework-geolocalization-internet-hosts)]
Replaces the linear DDR with a convex-hull spline model producing annular constraints (inner + outer radius). Adds negative constraints (oceans, uninhabitable areas). Reports 22-mile median error vs. CBG's 89-mile — a 4× improvement. Introduces MC sampled point selection as the centroid estimator. The source of our bounded spline Phase 1 model and `planar_annulus` Phase 2 multilateration.

**Hu et al., "Towards geolocation of millions of IP addresses"** (IMC 2012) [[ACM](https://dl.acm.org/doi/10.1145/2398776.2398790)]
Simplifies to the 2/3c (two-thirds speed of light) model, eliminating the need for per-landmark calibration. Introduces greedy VP selection prioritizing proximity to the target. Scales to geolocate ~35% of the IPv4 address space. The source of our 2/3c Phase 1 baseline.

**Wang et al., "Towards Street-Level Client-Independent IP Geolocation"** (NSDI 2011) [[ACM](https://dl.acm.org/doi/10.5555/1972457.1972494)]
Three-tier refinement from CBG to street-level using landmark discovery and traceroute path analysis. Implemented in this repo; out of scope for the current benchmark (unicast CBG focus).

**Darwich et al., "Replication: Towards a Publicly Available Internet Scale IP Geolocation Dataset"** (IMC 2023) [[ACM](https://dl.acm.org/doi/10.1145/3618257.3624801)]
The only publicly available CBG implementation. Replicates Million-Scale and Street-Level algorithms with RIPE Atlas. Shows that neither technique achieves previously claimed accuracy on today's Internet using public infrastructure. Our benchmark framework extends this codebase.

### 5.2 RTT-Distance Modeling (Phase 1)

**"Modelling of IP Geolocation by use of Latency Measurements"** (IEEE 2015 / arXiv 2020) [[arXiv](https://arxiv.org/pdf/2004.07836)]
Analyzes the correlation between network latency and geographic distance; proposes improved DDR models. Focused on Phase 1 in isolation — no cross-phase benchmark.

**"Dragoon: Advanced Modelling of IP Geolocation by use of Latency Measurements"** (arXiv 2020) [[arXiv](https://arxiv.org/abs/2006.16895)]
Optimized landmark placement via greedy diversification + advanced RTT-distance modulation for European networks. Another Phase 1 improvement without cross-phase evaluation.

**"Delay-Distance Correlation Study for IP Geolocation"** (arXiv 2019) [[arXiv](https://arxiv.org/pdf/1909.02439)]
Systematic empirical study of RTT-to-distance perturbing factors (queueing delay, non-great-circle routing). Provides theoretical grounding for why bounded spline outperforms 2/3c.

All three papers propose Phase 1 improvements in isolation; none benchmarks across variants or phases, confirming the gap this paper addresses.

### 5.3 Official / Declarative Methods (Tier 1 Context)

**"Geofeeds: Revolutionizing IP Geolocation or Illusionary Promises?"** (ACM Networking 2024) [[ACM](https://dl.acm.org/doi/10.1145/3676869)]
Critical large-scale assessment of GeoFeed (RFC 8805/9092) accuracy and adoption promises. Coverage is still limited and accuracy varies significantly by operator.

**"Geofeed Adoption and Authentication"** (IEEE / arXiv 2025) [[arXiv](https://arxiv.org/abs/2502.08849)]
Surveys GeoFeed adoption at RIR and AS level; finds ~7.76% of GeoFeed URLs inaccessible and RFC 9092 authentication lacking. Coverage gaps confirm CBG is needed as a fallback.

**"IP Geolocation through Reverse DNS"** (ACM TOIT 2021) [[ACM](https://dl.acm.org/doi/10.1145/3457611)]
Parses rDNS hostnames to extract location hints; places ~54% of hostnames within 20 km of ground truth. Open-source (Microsoft). Effective for named infrastructure but silent on cloud IPs with opaque hostnames.

These papers motivate the multi-tier pipeline: GeoFeed and rDNS coverage failures make CBG necessary as an empirical fallback.

### 5.4 Commercial Geolocation Accuracy / Criticism

**"Accuracy and Coverage Analysis of IP Geolocation Databases"** (IEEE 2023) [[IEEE](https://ieeexplore.ieee.org/document/10167899/)]
Cross-database accuracy study (MaxMind, DBIP, IP2Location, IPGeolocationIO) over the full IPv4 space. Mobile and cloud IPs are systematically under-served.

**"IP geolocation databases: unreliable?"** (ACM CCR 2011) [[ACM](https://dl.acm.org/doi/10.1145/1971162.1971171)]
Early influential demonstration of large median errors and frequent gross mislocations in commercial databases.

**"GPS-Based Geolocation of Consumer IP Addresses"** (PAM 2022) [[ACM](https://dl.acm.org/doi/10.1007/978-3-030-98785-5_6)]
Uses GPS-tagged user requests as ground truth to evaluate commercial services. Finds significant errors for mobile and residential IPs.

Commercial services are opaque, inaccurate on mobile IPs, and unauditable — motivating an open, measurement-based alternative.

### 5.5 Broader Geolocation Method Landscape

**Padmanabhan & Subramanian, "An Investigation of Geographic Mapping Techniques for Internet Hosts"** (WWW 2001) [[ACM](https://dl.acm.org/doi/10.1145/383059.383073)]
Introduced GeoCluster: propagate known location labels through BGP prefixes. Works for coarse-grained geolocation in small operator prefixes; produces silent errors for large ISP prefixes spanning multiple cities.

**Li et al., "Connecting the Hosts: Street-Level IP Geolocation with Graph Neural Networks"** (KDD 2022) [[ACM](https://dl.acm.org/doi/abs/10.1145/3534678.3539049)]
GNN-based street-level geolocation treating IP location as node regression on attribute graphs. State-of-the-art for supervised ML approaches. Requires large labeled training datasets; not auditable; cannot generalize to unlabeled IPs outside the training distribution.

**"Towards IP Geolocation with Intermediate Routers Based on Topology Discovery"** (Cybersecurity 2019) [[Springer](https://cybersecurity.springeropen.com/articles/10.1186/s42400-019-0030-2)]
Uses traceroutes to discover intermediate routers as secondary landmarks. Improves accuracy in landmark-sparse regions but requires 10–20 probes per target — prohibitive at ISP scale.

**"Selection of Landmarks for Efficient Active Geolocation"** (TMA 2024) [[IEEE](https://ieeexplore.ieee.org/document/10559002/)]
Demonstrates that even geographic distribution of landmarks significantly improves CBG precision. Confirms landmark placement is a first-order factor in CBG accuracy — directly relevant to our VP selection evaluation.

**"Leveraging Traceroute Inconsistencies to Improve IP Geolocation"** (arXiv 2025) [[arXiv](https://arxiv.org/html/2501.15064v1)]
Improves geolocation by detecting topological inconsistencies in traceroute paths. Topology-based — not a CBG variant and not scalable to millions of IPs; complementary direction for high-value targets.

**"GeoFINDR: Practical Approach to Verify Cloud Instances Geolocation in Multicloud"** (arXiv April 2025) [[arXiv](https://arxiv.org/abs/2504.18685)]
RIPE Atlas delay-based VM-scale cloud localization using DDR sectorization and barycenter estimation; achieves 22.6 km average accuracy. Shares the RIPE Atlas landmark infrastructure but differs in goal (CSP compliance verification vs. unicast IP geolocation at scale), algorithm (DDR sectorization, not CBG multilateration), and VP model (internal audit from within the VM).

### 5.6 Positioning Summary


| Gap in Existing Literature                                             | Our Contribution                                                          |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| No systematic cross-phase CBG benchmark                                | First to decompose CBG into 3 phases and benchmark the valid combination families |
| Phase 1 improvements proposed in isolation                             | Controlled evaluation holding other phases fixed                          |
| Only one public CBG implementation (IMC 2023)                          | Open-source framework covering all known variants                         |
| No accuracy-vs-scalability characterization                            | Pareto frontier of median error vs. runtime, starting with the full Vultr all-US default suite and extending to the remaining valid combinations |
| Simpler alternatives (GeoPing, GeoCluster, ML) each have critical gaps | CBG: physics-grounded, label-free, auditable, scalable to millions of IPs |
| Commercial services opaque and inaccurate on cloud IPs                 | Auditable, reproducible CBG alternative                                   |


---

## 6. First Full-Dataset Experiment

### Dataset

- **RTT measurements:** US source probes pinging the fixed seven-anchor Vultr set
- **Input:** `datasets/cbg_test/vultr_pings_us_only.csv`
- **Scale:** 9,866 RTT rows, 1,423 target probes, 7 anchors
- **Vantage points:** 7 US Vultr anchors used as CBG landmarks
- **Combinations evaluated:** 8 default full-run settings (`S1,S2,L1,L2,B1,B2,B3,B4`)
- **Artifacts:** `scripts/analysis/benchmark/outputs/vultr7/all_us_first_run/`

> **Comment:** This is now strong enough to replace the old 266-target historical table as the headline result. The proposal should still label it as the default-suite experiment until the remaining 7 valid combinations run on the same dataset.

### Results Table


| ID | Distance Model | Multilateration | Centroid | Median Error | P90 Error | Within 500km | Within 1000km | Intersection | Fallback | Runtime |
|----|----------------|-----------------|----------|:------------:|:---------:|:------------:|:-------------:|:------------:|:--------:|:-------:|
| S1 | Speed-of-Internet | `spherical_circle` + redundant filtering | `boundary_vertex_mean` | 522.7 km | 1659.4 km | 48.8% | 70.6% | 76.7% | 23.3% | 3.4s |
| S2 | Speed-of-Internet | `spherical_circle` | `boundary_vertex_mean` | 522.7 km | 1666.0 km | 48.8% | 70.6% | 22.2% | 77.8% | 9.1s |
| L1 | LP low-envelope | `spherical_circle` + redundant filtering | `boundary_vertex_mean` | 559.2 km | 1798.7 km | 45.7% | 69.1% | 70.0% | 30.0% | 3.3s |
| L2 | LP low-envelope | `spherical_circle` | `boundary_vertex_mean` | 559.2 km | 1798.7 km | 45.7% | 69.1% | 30.1% | 69.9% | 10.0s |
| B1 | Bounded spline | weighted `planar_annulus@0.9` | `monte_carlo_median` | 387.0 km | 1272.7 km | 60.5% | 84.7% | 92.6% | 7.4% | 48.0m |
| B2 | Bounded spline | `planar_annulus` | `monte_carlo_median` | 380.3 km | 1283.1 km | 60.5% | 84.6% | 93.5% | 6.5% | 3.8m |
| B3 | Bounded spline | weighted `planar_annulus@0.9` | `geometric_centroid` | **357.6 km** | **1157.0 km** | **64.3%** | **87.0%** | 92.6% | 7.4% | 45.2m |
| B4 | Bounded spline | `planar_annulus` | `geometric_centroid` | 359.7 km | 1171.7 km | 63.5% | 86.6% | **93.5%** | **6.5%** | 14.8s |

All settings return estimates for 100% of targets because failed multilateration falls back to a non-null estimate. We therefore report both intersection rate and fallback rate: they expose whether the result came from real geometric overlap or from failure handling.

### Error CDF

Generated figures from the full run:

- `error_cdf_all.png`
- `error_diff_cdf.png`
- `rtt_error_scatter.png`
- `benchmark_phase_latency_memory.png`

*Figure caption draft: Cumulative distribution of geolocation error for the full Vultr all-US default suite. Bounded-spline annulus configurations dominate the spherical baselines above the 500 km threshold; the practical unweighted-annulus geometric-centroid setting (`B4`) is visually close to the slightly more accurate but much slower weighted-annulus setting (`B3`).*

> **Comment:** Embed the CDF once the paper figure style is settled. The CDF should include a clear note that `B3` is not the production recommendation despite its small accuracy edge.

### Key Findings

**Finding 1 — Bounded-spline annulus variants dominate the evaluated spherical baselines.**
The best annulus result is 357.6 km median error (`B3`) and the practical annulus result is 359.7 km (`B4`), compared with 522.7 km for the speed-of-Internet spherical baseline (`S1`) and 559.2 km for the LP spherical baseline (`L1`). This is a 31-36% median-error reduction on the full dataset.

**Finding 2 — `B4` is the current production point.**
`B3` has the best raw accuracy, but only by 2.1 km at the median and 0.4 percentage points within 1000 km. `B4` finishes the full 1,423-target dataset in 14.8s, while `B3` takes 45.2 minutes because weighted-annulus multilateration has pathological tail latency. One target (`182.54.147.130`) spent ~40.7 minutes in multilateration.

**Finding 3 — MC sampled medoid is dominated in this run.**
For unweighted annulus, MC sampled medoid (`B2`) is slower and less accurate than geometric centroid (`B4`): 380.3 km vs. 359.7 km median error, 84.6% vs. 86.6% within 1000 km, and 3.8m vs. 14.8s full-run runtime. The same pattern holds for weighted annulus (`B1` vs. `B3`). The historical claim that MC median buys accuracy no longer holds after the sampled-medoid semantics cleanup and full-dataset rerun.

**Finding 4 — Redundant filtering improves spherical feasibility and runtime.**
Filtering improves spherical-circle intersection rates without materially changing median/threshold accuracy: `S1` vs. `S2` has 76.7% vs. 22.2% intersections and 3.4s vs. 9.1s runtime; `L1` vs. `L2` has 70.0% vs. 30.1% intersections and 3.3s vs. 10.0s runtime.

**Finding 5 — Availability and intersection rate must be reported separately.**
All 8 settings produce estimates for 100% of targets because fallback is always available, but fallback usage varies sharply. Annulus variants fall back on only 6.5-7.4% of probes; unfiltered spherical baselines fall back on 69.9-77.8%. Availability alone would hide this difference.

### Scalability Comparison


| Configuration | Median Error | Mean geolocate / target | Full-run runtime | 50M-target single-core estimate | Status |
| ------------- | ------------ | ----------------------- | ---------------- | ------------------------------- | ------ |
| **B4: spline + `planar_annulus` + `geometric_centroid`** | **359.7 km** | **8.7 ms** | **14.8s** | **~5 CPU-days** | Recommended production point |
| B3: spline + weighted `planar_annulus@0.9` + `geometric_centroid` | 357.6 km | 1904.9 ms | 45.2m | ~3.0 CPU-years | Accuracy leader, tail-latency blocker |
| B2: spline + `planar_annulus` + `monte_carlo_median` | 380.3 km | 160.1 ms | 3.8m | ~93 CPU-days | Dominated by B4 |
| S1: speed-of-Internet + filtered `spherical_circle` | 522.7 km | 1.0 ms | 3.4s | ~13 CPU-hours | Fast baseline |
| L1: LP low-envelope + filtered `spherical_circle` | 559.2 km | 1.3 ms | 3.3s | ~18 CPU-hours | Fast baseline |


*Runtime measured on the full Vultr all-US dataset, 1,423 targets, single-threaded. The 50M-target estimates use mean `total_geolocate` time and exclude one-time dataset loading and model fitting.*

> **Comment:** The old proposal framed MC median as an accuracy-vs-cost tradeoff. The new result is cleaner: MC sampled medoid is not on the Pareto frontier. The riskier but interesting frontier question is now `B3` vs. `B4`: can weighted-annulus geometry be made robust enough to justify its tiny accuracy edge?

---

## 7. Proposed Work Plan

**Scope: Unicast IP geolocation with CBG** — anycast and topology-based methods are deferred to future work.

### Phase 1: Finalize Unicast Benchmark (Vultr all-US dataset)

- Complete the remaining valid combinations on the full Vultr all-US dataset, especially `geometric_median` and any `planar_circle` variants not covered by the default suite
- Add timeout or geometry simplification controls for weighted-annulus pathological cases before using `planar_annulus_weighted` in larger experiments
- Summarize memory profiling already captured in `benchmark_phase_summary.json`
- Promote generated full-run plots (`error_cdf_all.png`, latency/memory plots, scatter plots) into paper-ready figures
- Preserve availability, intersection rate, and fallback rate as separate metrics

### Phase 2: RIPE Atlas Cross-Validation

- Run all 15 active valid combinations on RIPE Atlas anchor meshed pings (US subset)
- Run all 15 active valid combinations on RIPE Atlas EU subset
- Test whether phase rankings hold across datasets and geographies

### Phase 3: VP Count Sensitivity Analysis

- Vary number of VPs (1, 3, 5, 10, 20) for top-3 configurations
- Quantify accuracy degradation with fewer VPs — relevant for mobile operator deployments where VP count is fixed and potentially small

### Phase 4: Paper Writing and Open-Source Release

- Clean and document `scripts/framework/` as a standalone open-source CBG library
- Write paper targeting **IMC 2026** (deadline TBD)
- Release curated datasets alongside the paper

> **Comment:** The work plan should now emphasize validation and generalization rather than proving that a full-size run is possible. The full Vultr run exists; the open questions are whether `B4` stays best across the remaining combinations and whether RIPE Atlas reproduces the ranking.

---

## 8. Expected Contributions

1. **CBG pipeline taxonomy**: First formal decomposition of CBG into three independent, interchangeable phases — enabling reproducible cross-variant comparison
2. **Open-source framework**: Faithful implementations of all CBG variants (2/3c, LP, Octant spline; `spherical_circle`, `planar_circle`, `planar_annulus` multilateration; arithmetic, geometric, MC median centroid)
3. **Benchmark dataset**: Curated RTT measurements from mobile VPs + RIPE Atlas, with verified ground truth
4. **Scalability analysis**: First accuracy-vs-runtime Pareto characterization of CBG variants, directly applicable to production deployment decisions
5. **Practical guidance**: Bounded spline + unweighted `planar_annulus` + `geometric_centroid` is the current recommended configuration on the full Vultr all-US run — 359.7 km median error, 86.6% within 1000 km, 8.7 ms mean geolocation time per target
