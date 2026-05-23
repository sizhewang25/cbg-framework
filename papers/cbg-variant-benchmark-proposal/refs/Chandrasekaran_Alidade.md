# Alidade: IP Geolocation without Active Probing

Chandrasekaran et al., Duke / Akamai / Cornell / Waterloo. Tech Report CS-TR-2015.001, Jan 2015 (rev. Apr 2015).

## Overview

Alidade is a *passive* constraint-based geolocation system that precomputes a location estimate for every routable IPv4 address by fusing third-party measurement traces (iPlane, CAIDA Ark, a Tier-1 CDN's traceroute/ping logs) with non-measurement hints (Internet registries, hostname parsing via a tool like `Undns`/HostParser, AS hierarchy, city/country shape files). Unlike commercial geolocation databases, every prediction is a *polygon* (the intersection region of constraints) plus a representative point chosen inside it; unlike active CBG systems, Alidade never issues probes — neither at ingest nor at query time. The system is built as a Hadoop/HBase pipeline (Preprocessor, Iterative Solver, Extrapolator, Preloader, Aggregator, Query Engine) and was run on a 40-node cluster.

## Core Algorithm vs. Classical CBG

Classical CBG (Gueye et al. 2006) takes RTTs from landmarks to a target on-demand, converts each RTT to a max-distance circle using a speed-of-Internet constant, intersects circles, and returns the centroid. Alidade keeps the CBG geometry but changes both the data source and the constraint structure:

- **Latency from logs, not probes.** RTT samples come from precomputed third-party datasets. The Preprocessor summarizes the distribution per (landmark, target) pair, typically using the *median* (or mean/min depending on shape) rather than the minimum.
- **Direct vs. indirect observations.** A direct observation is a landmark→target latency; constraints are polygons (N=32 sides) centered on the landmark. An indirect observation is a latency between two non-landmark hops on a traceroute, computed as a difference of direct observations; its constraint is the source's current polygonal estimate *dilated* by the latency-derived distance. This lets Alidade reuse traceroutes that don't originate at a landmark — an idea borrowed from Octant.
- **Speed-of-light constant.** Alidade uses 2/3 c (≈ speed of light in fiber) rather than the looser 4/9 c often used in CBG. Tighter constraints risk empty intersections but better match observed iPlane data (only 4,031 violations in 2011).
- **Iterative solver.** Three iterations are typical: each pass re-derives indirect-observation constraints using the previous round's target polygons.
- **Non-measurement fusion.** Registry/HostParser city hints are intersected with measurement polygons; AS-rank and prefix-size filters down-weight unreliable registry entries. The Aggregator propagates estimates within a BGP prefix to addresses with no direct measurements (≈68% of MLAB targets).
- **Polygonal answers.** Each prediction is a region (often simplified via α-shape), with a point answer derived by snapping to a contained city centroid.

## Evaluation Setup

Compared head-to-head against six commercial databases (anonymized DB1–DB6): EdgeScape, MaxMind GeoCity, MaxMind GeoLite2, DB-IP, IP2Location, IPligence. Six ground-truth sets (Table 1):

| Dataset | #IPs | #Locations |
|---|---|---|
| PLAB (PlanetLab) | 835 | 331 |
| Ark | 66 | 61 |
| MLAB | 882 | 36 |
| GPS | 152 | 139 |
| NTP | 99 | 77 |
| EuroGT (Tier-1 ISP) | 23,737,281 | 73 |

EuroGT evaluation samples 100,000 targets. Input data: ~700M HostParser answers (211M city-level), 3-month CDN traceroutes, 1 week of CDN→router pings, 1 month of CDN↔end-user TCP RTTs, plus iPlane + Ark.

## Key Results

- On EuroGT (100k sample, ground truth is city-level so ECDFs start at 10 km): Alidade reaches **~79% of targets within 10 km error**, beating all six commercial DBs.
- On **MLAB**: median error **16 km**; all targets within **370 km** (DB1's max error is ~6× larger), despite Alidade having no direct measurements to any MLAB target (predictions come entirely from registry/HostParser/aggregates).
- On **Ark**: ~80% of targets under **14 km**; max error ~3,200 km vs. >100 km median for other DBs at the tail. Comparable to DB1 in this range.
- On **PLAB**: only marginally better than the registry-only baseline (PLAB locations are well-predicted from hostnames alone, and measurements cover only 34.61% of targets — see Table 2).
- On **GPS / NTP**: Alidade is competitive but not dominant; some commercial DBs do better, especially on the 7–8% of NTP targets where Alidade has measurements.
- **Measurement-data ablation** (Figure 18, "WITH-MEAS" vs. "WITHOUT-MEAS" vs. "SKIPPED-AGG"): adding measurements improves accuracy beyond registry+HostParser alone, and aggregates further help targets *with* measurements by filtering inconsistent hints.
- **Staleness matters**: 2014 input data (1+ quarter after GT collection) had ~20% fewer targets ≤10 km than 2013 input (Figure 17).
- **Feasibility-area check**: Alidade's polygons can flag commercial-DB point answers that fall outside the feasible region (e.g., an Apple IP that registry-based DBs place in Cupertino but measurements place in Asia).

## Strengths

- No query-time probing → fast, unobtrusive, doesn't tip off the target.
- Predictions carry a *region* with a correctness guarantee under the speed-of-fiber assumption; useful as a sanity filter on other DBs' point answers.
- Fuses many evidence sources rather than relying on any one; resilient when registry/hostname hints are absent.
- Covers the entire IPv4 space (~900M targets in the reported run).
- Indirect-observation constraints exploit transit-segment latencies, not just landmark→target pairs.

## Limitations

- Quality is bounded by the freshness, density, and geographic diversity of *whatever* logs are ingested — Alidade itself produces none. Targets without nearby latency data fall back to registry/aggregate guesses.
- Requires substantial infrastructure (40×8-core × 32GB Hadoop cluster; preprocessing alone ~8 h, Extrapolator 1.5–2 h, Preloader ~45–60 min).
- Speed-of-fiber (2/3 c) assumption is tighter than classical CBG; intersection can become empty if even one measurement is corrupt. No queuing/path-inflation model.
- Polygon → point reduction uses ad-hoc heuristics (centroid of a contained city) rather than population-weighted scoring.
- HostParser/registry hints inherit any errors in those sources; PLAB ground truth itself contained errors (Peking U. 2010, USC ISI 2011).
- 2015 tech report, never published in a refereed venue; code/datasets not released.

## Relevance to a CBG Variant Benchmark

Alidade is the natural representative of the **"CBG over passive/offline RTTs + multi-hint fusion"** branch of the CBG family. Three implications for our benchmark design:

1. **Variant to replicate.** A faithful Alidade-style variant would (a) use median (not min) RTT per (landmark, target) pair from logs, (b) build 32-sided polygon constraints with 2/3 c, (c) add indirect-observation constraints derived from traceroute hop-pair latencies, (d) iterate ≥3 times, (e) intersect with hostname/registry/AS-rank hints. Comparison points: classical CBG (Gueye, min RTT, 4/9 c), Octant (positive+negative constraints, weight-based), Spotter (statistical landmark model), Posit (precomputed embedding).
2. **Active-probe-free angle changes the harness.** Our benchmark's IMC-2023 RIPE Atlas dataset *is* the kind of pre-collected measurement corpus Alidade was designed for. For an Alidade-style variant, we can fix the measurement corpus once and evaluate "no further probing allowed", which aligns cleanly with our leakage-free protocol — no per-query VP selection, no adaptive rounds. This also means a fair comparison with active-CBG variants (which would normally probe on demand) requires capping their VP budget to the same fixed corpus or reporting both regimes.
3. **Evaluation conventions to borrow.** Region-based answers + point-answer headline metric (error CDF in km), ablations isolating measurement vs. non-measurement contribution, and a staleness/timeline-alignment experiment. Alidade also motivates including a "feasibility-area" diagnostic: measure how often each variant's point answer falls inside its own (or a reference) constraint polygon — a quality signal independent of ground truth.
4. **What not to copy.** Alidade's heavy reliance on a private Tier-1 CDN traceroute feed and a 24M-IP private ground-truth set (EuroGT) means its headline 79%-under-10-km number is not directly reproducible. The PLAB/Ark/MLAB curves are closer to what is achievable with public data, and confirm Hu et al. (Million-Scale)'s point that public-only inputs hit a much lower ceiling.
