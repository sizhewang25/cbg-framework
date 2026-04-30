# CBG Benchmark Paper — Report

**Status**: In Progress
**Created**: 2026-04-24
**Last Updated**: 2026-04-24 (related work survey added)

## Summary

Planning stage. Scope defined based on findings from task `20260415-cbg-combination-evaluation`.

## Findings

### 2026-04-24 — Baseline results from CBG combination evaluation

From 18-combination benchmark on AS7922 Vultr mobile VP dataset (266 probes, 7 anchors):

| Combination | Distance Model | Multilateration | Centroid | Median Error | Within 1000km | Runtime |
|-------------|---------------|-----------------|----------|:------------:|:-------------:|:-------:|
| G3 (best) | Octant spline | `planar_annulus` | MC median | **312 km** | 94.0% | ~27s |
| F3 | Octant spline | `planar_annulus` | Geometric | 328 km | 94.0% | ~0.21s |
| A3 | Octant spline | `spherical_circle` | Arith. mean | 337 km | 86.8% | ~0.18s |

Key patterns established:
- Distance model dominates: Octant spline consistently outperforms 2/3c and LP low-envelope
- `planar_annulus` multilateration outperforms `spherical_circle` and `planar_circle` at the 500km+ thresholds
- MC median adds ~5% accuracy over geometric centroid at ~130x the compute cost
- Geometric centroid is the best accuracy/speed trade-off for production use

### 2026-04-24 — Related Work Survey

#### 1. Foundational CBG Papers (must cite)

| Paper | Venue | Key Contribution | Relation to This Work |
|-------|-------|------------------|-----------------------|
| Gueye et al., "Constraint-based geolocation of internet hosts" | IMC 2004 / IEEE/ACM ToN 2006 | Original CBG: linear regression DDR, PlanetLab landmarks, multilateration via circle intersection | Defines LP low-envelope (Phase 1) and spherical_circle multilateration (Phase 3) — one of our baselines |
| Wong et al., "Octant: A Comprehensive Framework for the Geolocalization of Internet Hosts" | NSDI 2007 | Convex-hull RTT-distance model; positive + negative constraints; 22-mile median error vs CBG 89-mile | Source of our Octant spline distance model and `planar_annulus` multilateration |
| Hu et al., "Towards geolocation of millions of IP addresses" | IMC 2012 | 2/3c RTT-distance model; VP selection greedy algorithm; ~35% of IPv4 space geolocated | Source of our 2/3c distance model baseline |
| Wang et al., "Towards Street-Level Client-Independent IP Geolocation" | NSDI 2011 | Three-tier street-level algorithm | Implemented in this repo; not the focus of this benchmark paper |
| Darwich et al., "Replication: Towards a Publicly Available Internet Scale IP Geolocation Dataset" | IMC 2023 | Only publicly available CBG implementation; replicates both IMC 2012 and NSDI 2011 | This repo — our benchmark builds directly on it |

Sources: [Gueye IMC 2004](https://dl.acm.org/doi/10.1145/1028788.1028828) · [Gueye ToN 2006](https://dl.acm.org/doi/10.1109/TNET.2006.886332) · [Octant NSDI 2007](https://www.usenix.org/conference/nsdi-07/octant-comprehensive-framework-geolocalization-internet-hosts) · [Million-Scale IMC 2012](https://dl.acm.org/doi/10.1145/2398776.2398790) · [Street-Level NSDI 2011](https://dl.acm.org/doi/10.5555/1972457.1972494) · [IMC 2023 Replication](https://dl.acm.org/doi/10.1145/3618257.3624801)

#### 2. RTT-Distance Modeling (Phase 1 related)

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| "Modelling of IP Geolocation by use of Latency Measurements" | IEEE 2015 / arXiv 2020 | Improved DDR modeling; RTT-distance correlation analysis |
| "Dragoon: Advanced Modelling of IP Geolocation by use of Latency Measurements" | arXiv 2020 | Optimized landmark selection + advanced RTT-distance modulation; European focus |
| "Delay-Distance Correlation Study for IP Geolocation" | arXiv 2019 | Systematic study of RTT-distance correlation factors |

These papers each propose Phase 1 improvements in isolation; none benchmarks across variants or phases. Confirms our benchmark gap.

Sources: [Modelling arXiv](https://arxiv.org/pdf/2004.07836) · [Dragoon arXiv 2020](https://arxiv.org/abs/2006.16895) · [Delay-Distance arXiv 2019](https://arxiv.org/pdf/1909.02439)

#### 3. Official / Declarative Methods (GeoFeed, rDNS — Tier 1 in our pipeline)

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| "Geofeeds: Revolutionizing IP Geolocation or Illusionary Promises?" | ACM Networking 2024 | Critical assessment of GeoFeed accuracy and adoption promises |
| "Geofeed Adoption and Authentication" | IEEE / arXiv 2025 | Deployment at RIR/AS level; RFC 8805/9092 adherence; ~7.76% of GeoFeed URLs inaccessible |
| "IP Geolocation through Reverse DNS" | ACM TOIT 2021 | rDNS method placing ~54% of hostnames within 20 km; open-source |

GeoFeed coverage gaps and rDNS failures motivate CBG as a necessary fallback tier.

Sources: [Geofeeds ACM 2024](https://dl.acm.org/doi/10.1145/3676869) · [Geofeed Adoption arXiv 2025](https://arxiv.org/abs/2502.08849) · [rDNS ACM TOIT 2021](https://dl.acm.org/doi/10.1145/3457611)

#### 4. Anycast Geolocation

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Cai et al., "Latency-Based Anycast Geolocation: Algorithms, Software, and Data Sets" (iGreedy) | IEEE JSAC 2016 | iGreedy: latency-based anycast site enumeration + city-level geolocation via population classification; adversely affected by processing delay noise |
| "A fistful of pings: Accurate and lightweight anycast enumeration and geolocation" | IEEE INFOCOM 2015 | Lightweight ping-based anycast enumeration baseline |
| "LACeS: An Open, Fast, Responsible, and Efficient Longitudinal Anycast Census System" | arXiv 2025 | Large-scale daily anycast census; reduces probing cost by two orders of magnitude using anycast pre-filtering |
| "Locating and Enumerating Anycast: a Comparison of Two Approaches" | ACM ANRW 2025 | Traceroute vs latency: traceroute 4× costlier for marginal precision gain |
| "Regional IP Anycast: Deployments, Performance, and Potentials" | ACM SIGCOMM 2023 | Regional anycast deployment characterization |

Key gap: iGreedy is SOTA for anycast geolocation but requires prior knowledge that the IP is anycast. Commercial services (IPInfo, MaxMind) fail silently on anycast. CBG-based approaches have never been benchmarked on anycast IPs in existing literature — this is a direct contribution.

Sources: [iGreedy IEEE JSAC 2016](https://ieeexplore.ieee.org/document/7470242/) · [Fistful of pings IEEE 2015](https://ieeexplore.ieee.org/document/7218670/) · [LACeS arXiv 2025](https://arxiv.org/pdf/2503.20554) · [Anycast comparison ACM ANRW 2025](https://dl.acm.org/doi/10.1145/3744200.3744783) · [Regional Anycast SIGCOMM 2023](https://dl.acm.org/doi/10.1145/3603269.3604846)

#### 5. Commercial Geolocation Accuracy / Criticism

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| "Accuracy and Coverage Analysis of IP Geolocation Databases" | IEEE 2023 | Accuracy study of MaxMind, DBIP, IP2Location, IPGeolocationIO across full IPv4 space |
| "IP geolocation databases: unreliable?" | ACM CCR 2011 | Early critique of commercial DB accuracy |
| "GPS-Based Geolocation of Consumer IP Addresses" | PAM 2022 | GPS ground truth comparison vs commercial services |

Motivates our position: commercial services are opaque, inaccurate on mobile/anycast IPs, and unauditable.

Sources: [DB accuracy IEEE 2023](https://ieeexplore.ieee.org/document/10167899/) · [DB unreliable ACM CCR 2011](https://dl.acm.org/doi/10.1145/1971162.1971171) · [GPS-based PAM 2022](https://dl.acm.org/doi/10.1007/978-3-030-98785-5_6)

#### 6. Recent Adjacent Work

| Paper | Venue | Notes |
|-------|-------|-------|
| "Leveraging Traceroute Inconsistencies to Improve IP Geolocation" | arXiv 2025 | Topology-based refinement — out of scope (not CBG) but citable as complementary direction |
| "GeoFINDR: Practical Approach to Verify Cloud Instances Geolocation in Multicloud" | arXiv Apr 2025 | RIPE Atlas delay-based VM-scale cloud localization; 22.6 km accuracy — close in motivation; differentiate by our focus on CBG pipeline benchmarking rather than verification |

Sources: [Traceroute Inconsistencies arXiv 2025](https://arxiv.org/html/2501.15064v1) · [GeoFINDR arXiv 2025](https://arxiv.org/abs/2504.18685)

#### 7. Positioning Summary

| Gap in Existing Literature | Our Contribution |
|---------------------------|-----------------|
| No systematic cross-phase CBG benchmark | First to decompose CBG into 3 phases and benchmark all valid phase combinations |
| Only one public CBG implementation (IMC 2023 repo) | Open-source framework for all CBG variants |
| No CBG evaluation on anycast IPs | Characterize CBG accuracy on anycast vs unicast cloud IPs |
| No CBG benchmark from mobile operator VP perspective | Curated RTT dataset from mobile VPs + RIPE Atlas cross-validation |
| Commercial services opaque, fail on anycast | Provide auditable, reproducible CBG alternative |

## Conclusions

*To be filled as paper progresses.*
