# Du et al. 2020 — RIPE IPmap Active Geolocation: Mechanism and Performance Evaluation

**Venue:** ACM SIGCOMM Computer Communication Review, Vol. 50 Issue 2, April 2020
**Authors:** Ben Du, Massimo Candela, Bradley Huffaker, Alex C. Snoeren, kc claffy

## Overview

The paper introduces and evaluates the RIPE NCC's IPmap *single-radius* engine, an active geolocation service designed for **core Internet infrastructure** (routers, servers) rather than end hosts. Single-radius issues pings from RIPE Atlas probes, picks the single probe with the lowest RTT, converts that RTT into a radius around the probe, and selects the most plausible city inside the resulting disk. The paper measures accuracy, coverage, and consistency against commercial databases (NetAcuity, MaxMind GeoLite2) using a ground-truth dataset drawn from NLNOG Ring, M-Lab, and CAIDA Ark.

## Core Algorithm (Single-Radius)

Four steps per target IP:

1. Map target IP → originating AS via RIPE RIS BGP. Build a candidate probe-list (up to 500 probes) topologically near the target: probes in AS(t), in BGP neighbors of AS(t), and in ASes co-located at IXPs/facilities with AS(t) (PeeringDB).
2. Ping target from probe-list. Discard RTTs > 10 ms; convert remaining RTTs to one-way latency (RTT/2).
3. Pick probe *p* with minimum latency π. Convert to distance d using distance-delay coefficient **2/3 · c** (speed of light through fiber). Draw circle C of radius d around *p*.
4. From a list of ~100 closest cities to *p* (RIPE Worlds DB), keep those inside C. Rank by weighted score (population, IXP/facility count, and distance-from-probe weight = 10 − π). Return top-ranked city.

**Relation to CBG:** Single-radius is essentially a **degenerate single-VP CBG**. Standard CBG (Gueye et al. 2006) multilaterates by intersecting disks from many vantage points; single-radius uses one disk (the closest probe) and resolves ambiguity through population/topology priors rather than geometric intersection. The authors explicitly note this and recommend (§7) that a multilateration engine "would work better with tighter geographical constraints."

## Evaluation Setup

- **Ground truth (§4.2):** 968 IPs / 651 ASes / 84 countries combining NLNOG Ring (500 nodes), M-Lab (148 pods), Ark monitors (123 reachable), and *ark-proximity* hosts within 26.5–33.3 km of an Ark monitor (206 hosts). Heavily biased toward Western Europe (509 IPs) and North America (242 IPs).
- **Vantage points:** RIPE Atlas (>10K probes across 179 countries) — same platform our CBG project uses.
- **Coverage dataset:** CAIDA MANIC, 26,559 interconnection (core-infrastructure) IP addresses.
- **Consistency dataset:** CAIDA ITDK *ip_alias* — 540 routers, up to 4 interfaces each.
- **Commercial baselines:** NetAcuity (Digital Element) and MaxMind GeoLite2.
- **Accuracy threshold:** 40 km (city-level / metropolitan).

## Key Results

- **Accuracy (Fig. 1, §6.1):** 870 results out of 968 queries. Single-radius median / 75th / 95th percentile error = **6 / 26 / 344 km**, vs. NetAcuity **10 / 80 / 2867 km** and MaxMind **17 / 278 / 2886 km**. **80.3%** of single-radius inferences within 40 km.
- **Regional skew (Table 2):** AS 97.6%, AF 96.0%, SA 94.7%, OC 84.0%, NA 78.3%, EU 73.3% within 40 km — accuracy is higher where ground-truth IPs concentrate in few cities.
- **Coverage (§6.3):** geolocated 12,319 of 15,694 reachable MANIC IPs = **78.5%** at 10 ms RTT cutoff; drops to 51.1% at 5 ms, 24.1% at 2 ms.
- **AS-size effect (§6.2):** median probe-proximity distance 0.36 km (EC/CAHP), 0.27 km (STP), **17.63 km** (LTP). Within-40 km accuracy = 80.6% / 76.1% / **55.8%** — degrades sharply on large transit ASes.
- **Router consistency (§6.4):** 87.0% of 540 routers had all interfaces geolocated city-consistently (median disagreement 0 km, 75th pct 16 km); 61.3% to exactly identical coordinates.
- **RTT–error relationship (Fig. 6):** a 2 ms RTT threshold keeps 95th-percentile error < 40 km, but costs ~3/4 of coverage.

## Strengths

- Fast, real-time, no traceroutes or alias resolution required.
- Outperforms commercial databases on core infrastructure by a wide margin.
- Topologically-aware probe selection (BGP + IXP + facility) is more principled than "use all probes."
- Public API: `ipmap.ripe.net/api/v1/single-radius/[IP]`.

## Limitations (acknowledged by authors)

- **Single-VP only** — no multilateration; cannot exclude the wrong side of a large circle.
- **Fixed 2/3·c coefficient** ignores regional RTT inflation (Candela et al. 2019 show this varies by region).
- **Last-mile RTT inflation** not subtracted (DSL/satellite).
- **100-closest-cities cap** can omit the true city in dense metro areas (their Budapest→Vienna case study, §6.5).
- **Ground truth bias** — Western Europe + US dominate; uses RIPE Atlas probes themselves, a circular dependency.
- **AS-size sensitivity** — performs poorly on large transit ASes due to wide geographic footprint of candidate probes.

## Relevance to CBG Variant Benchmarking

This paper is a direct neighbor of our work and yields concrete benchmark-design implications:

1. **Shared platform**: Both use RIPE Atlas probes (>10K VPs, 179 countries) as the measurement substrate. Performance numbers are directly comparable in spirit.
2. **Single-VP baseline**: Single-radius can be implemented as a trivial CBG-variant baseline — picks the closest probe, no intersection. Useful lower-bound for "geometric" CBG variants.
3. **Threshold conventions**: 40 km city-level threshold and the [0, 40, 100, 500, 1000] km style CDF reporting align with this project's `THRESHOLD_DISTANCES`.
4. **Distance-delay coefficient debate**: Du et al. tested replacing 2/3·c with Candela et al.'s regional coefficients and report it changed only **0.96%** of cases in their setup. A CBG-variant benchmark should still treat the coefficient as a tunable knob, but expect modest sensitivity for single-disk methods — multilateration variants may be more affected.
5. **Target population matters**: Single-radius targets **core infrastructure**; our project's primary GT (IMC 2023 RIPE anchors) is also core infrastructure. This validates anchor-based GT as the comparable population. Commercial DBs are tuned for end-hosts and will look bad here — frame those as cross-DB references, not baselines (matches the project memory note on cross-DB usage).
6. **Avoid the RIPE-Atlas-as-GT trap**: Du et al. explicitly flag using Atlas probes as ground truth as circular. Our SWAP pressure-test (anchors→probes, 12,129 hard-GT targets) inherits the same caveat and should be reported as such.
7. **Probe-selection effectiveness as a sub-metric**: Their "probe proximity distance" (Eq. 2) — gap between selected probe and *closest* probe — is a useful diagnostic for evaluating any CBG variant's VP-selection sub-algorithm independently of the geolocation algorithm.
8. **Coverage–accuracy tradeoff curve**: Their Fig. 5/6 (RTT-threshold sweep) is a benchmark format worth replicating: plot accuracy CDF and coverage as functions of max-RTT cutoff.
9. **AS-size stratification**: Reporting accuracy split by EC/CAHP / STP / LTP (customer-degree bins) exposes where probe-selection breaks down. Recommended stratification for any CBG variant comparison.
