# CBG Benchmark Paper — Report

**Status**: In Progress
**Created**: 2026-04-24
**Last Updated**: 2026-05-01 (benchmark memory metric definitions)

## Summary

The first full-dataset Vultr all-US benchmark run is available for the default 8-combination suite (`S1,S2,L1,L2,B1,B2,B3,B4`): 9,866 RTT rows, 1,423 probes, and 7 anchors. Current best practical setting is bounded spline + unweighted `planar_annulus` + `geometric_centroid` (`B4`): 359.7 km median error, 86.6% within 1000 km, and 14.8s end-to-end for the full dataset.

## Findings

### 2026-04-24 — Baseline results from CBG combination evaluation

From the historical 18-combination benchmark on AS7922 Vultr mobile VP dataset before the centroid semantics cleanup (266 probes, 7 anchors):

| Combination | Distance Model | Multilateration | Centroid | Median Error | Within 1000km | Runtime |
|-------------|---------------|-----------------|----------|:------------:|:-------------:|:-------:|
| G3 (best) | Octant spline | `planar_annulus` | `monte_carlo_median` | **312 km** | 94.0% | ~27s |
| F3 | Octant spline | `planar_annulus` | `geometric_centroid` | 328 km | 94.0% | ~0.21s |
| A3 | Octant spline | `spherical_circle` | `boundary_vertex_mean` | 337 km | 86.8% | ~0.18s |

Key patterns established:
- Distance model dominates: Octant spline consistently outperforms 2/3c and LP low-envelope
- `planar_annulus` multilateration outperforms `spherical_circle` and `planar_circle` at the 500km+ thresholds
- MC median adds ~5% accuracy over geometric centroid at ~130x the compute cost
- Geometric centroid is the best accuracy/speed trade-off for production use

The MC-median accuracy advantage above is historical. It no longer holds after the 2026-04-30 sampled-medoid semantics cleanup and the 2026-05-01 full-dataset rerun.

### 2026-04-30 — Centroid semantics cleanup

Active framework combinations now exclude `spherical_circle + geometric_centroid`: `spherical_circle` returns unordered crossing vertices, while `geometric_centroid` requires a Shapely polygon region. The old vertex-average centroid is now named `boundary_vertex_mean`, and for planar annuli it includes both exterior and interior-ring vertices so annulus holes are represented in the boundary mean.

This changes the active benchmark set from 18 historical combinations to 15 valid combinations. The previous accuracy and runtime tables should be treated as historical. The 2026-05-01 full-dataset run below supersedes them for the default 8-combination benchmark suite; a full 15-combination grid is still pending.

### 2026-05-01 — Full Vultr all-US first run

New results from `scripts/analysis/benchmark/outputs/vultr7/all_us_first_run/` cover the full Vultr all-US dataset (`datasets/cbg_test/vultr_pings_us_only.csv`): 9,866 RTT rows, 1,423 probes, 7 anchors. This run evaluates the default 8-combination suite, not the full 15 active valid combinations.

| ID | Combination | Median Error | Mean Error | P90 Error | Within 500km | Within 1000km | Intersection | Fallback | Runtime |
|----|-------------|:------------:|:----------:|:---------:|:------------:|:-------------:|:------------:|:--------:|:-------:|
| S1 | SoI + `redundant_circle` + `spherical_circle` + `boundary_vertex_mean` | 522.7 km | 800.2 km | 1659.4 km | 48.8% | 70.6% | 76.7% | 23.3% | 3.4s |
| S2 | SoI + no filtering + `spherical_circle` + `boundary_vertex_mean` | 522.7 km | 809.4 km | 1666.0 km | 48.8% | 70.6% | 22.2% | 77.8% | 9.1s |
| L1 | LP low-envelope + `redundant_circle` + `spherical_circle` + `boundary_vertex_mean` | 559.2 km | 828.4 km | 1798.7 km | 45.7% | 69.1% | 70.0% | 30.0% | 3.3s |
| L2 | LP low-envelope + no filtering + `spherical_circle` + `boundary_vertex_mean` | 559.2 km | 827.8 km | 1798.7 km | 45.7% | 69.1% | 30.1% | 69.9% | 10.0s |
| B1 | Bounded spline + weighted `planar_annulus@0.9` + `monte_carlo_median` | 387.0 km | 590.0 km | 1272.7 km | 60.5% | 84.7% | 92.6% | 7.4% | 48.0m |
| B2 | Bounded spline + `planar_annulus` + `monte_carlo_median` | 380.3 km | 579.8 km | 1283.1 km | 60.5% | 84.6% | 93.5% | 6.5% | 3.8m |
| B3 | Bounded spline + weighted `planar_annulus@0.9` + `geometric_centroid` | **357.6 km** | **527.6 km** | **1157.0 km** | **64.3%** | **87.0%** | 92.6% | 7.4% | 45.2m |
| B4 | Bounded spline + `planar_annulus` + `geometric_centroid` | 359.7 km | 528.7 km | 1171.7 km | 63.5% | 86.6% | **93.5%** | **6.5%** | 14.8s |

Runtime detail from `benchmark_phase_summary.json`:

| ID | Median geolocate / probe | P95 geolocate / probe | Max geolocate / probe | Main runtime driver |
|----|:------------------------:|:---------------------:|:--------------------:|---------------------|
| S1 | 0.747 ms | 2.025 ms | 9.511 ms | spherical-circle intersection after redundant filtering |
| S2 | 5.653 ms | 7.205 ms | 9.613 ms | spherical-circle intersection without filtering |
| L1 | 0.980 ms | 3.174 ms | 9.918 ms | spherical-circle intersection after redundant filtering |
| L2 | 5.963 ms | 7.666 ms | 11.188 ms | spherical-circle intersection without filtering |
| B1 | 195.721 ms | 741.485 ms | 2406.001s | weighted annulus multilateration plus MC centroid |
| B2 | 130.439 ms | 392.601 ms | 987.968 ms | MC centroid |
| B3 | 74.743 ms | 359.454 ms | 2441.132s | weighted annulus multilateration |
| B4 | 8.947 ms | 9.064 ms | 10.043 ms | unweighted annulus multilateration |

Key findings:
- Bounded spline + annulus variants dominate the evaluated spherical baselines: the best annulus median is 357.6 km vs 522.7 km for SoI spherical and 559.2 km for LP spherical.
- `B3` has the best raw accuracy, but its weighted annulus geometry has severe tail latency: one probe (`182.54.147.130`) spent ~40.7 minutes in multilateration. This makes `B3` impractical without geometry simplification or timeout controls.
- `B4` is the current production trade-off: it is only 2.1 km worse than `B3` at the median and 0.4 percentage points worse within 1000 km, but it finishes the full dataset in 14.8s instead of 45.2m.
- The sampled-medoid `monte_carlo_median` is no longer justified in this run. It is slower and less accurate than `geometric_centroid` for both unweighted annulus (`B2` vs `B4`) and weighted annulus (`B1` vs `B3`).
- Redundant-circle filtering materially improves spherical-circle intersection rates and runtime without materially changing median/threshold accuracy: `S1` vs `S2` has 76.7% vs 22.2% intersections and 3.4s vs 9.1s runtime; `L1` vs `L2` has 70.0% vs 30.1% intersections and 3.3s vs 10.0s runtime.
- Availability is 100% for all 8 settings because failed multilateration falls back to a non-null estimate. The fallback rate remains an important quality signal: annulus variants fall back on only 6.5-7.4% of probes, while unfiltered spherical baselines fall back on 69.9-77.8%.

Immediate next steps:
- Treat `B4` as the main mobile-VP result in the paper draft unless a later full 15-combination run changes the ranking.
- Investigate or cap weighted-annulus pathological cases before using `planar_annulus_weighted` in larger benchmarks.
- Run the remaining valid combinations, especially `geometric_median` variants, on `all_us` before calling the mobile benchmark complete.

### 2026-05-01 — Benchmark memory metric interpretation

`tracemalloc` measures Python-level allocations tracked by the Python interpreter. In the benchmark outputs:
- `tracemalloc_peak_bytes` is the absolute Python heap peak seen during a phase after `tracemalloc.reset_peak()`.
- `tracemalloc_peak_delta_bytes` is the phase-local increase over the Python heap size at phase start.
- This is useful for attributing Python object allocation pressure to CBG phases.
- It does not capture all native memory, such as GEOS/Shapely internals, all NumPy native buffers, OS page cache, or allocator-retained memory.

RSS means resident set size: the process memory physically resident in RAM as reported by the OS. It includes Python heap, native-library allocations, NumPy/Shapely/GEOS memory, interpreter/runtime overhead, and memory retained by allocators after Python objects are freed. In the benchmark outputs:
- `rss_before_bytes` and `rss_after_bytes` are process-level snapshots around a phase.
- `rss_delta_bytes = rss_after_bytes - rss_before_bytes`.
- `max_rss_after_mb` is the largest RSS-after value observed for a setting.
- `rss_high_water_delta_bytes` records how much a phase raised the process-level RSS high-water mark.

Interpretation rule: use `tracemalloc_peak_delta_bytes` for phase-local Python allocation pressure, and use `max_rss_after_mb` for total process memory footprint. Do not interpret stacked `tracemalloc` bars as total RAM usage; they are attribution aids. Do not interpret RSS delta as exact phase allocation, because RSS can stay high after memory is freed.

### 2026-04-30 — Why Shapely is used and possible spherical alternatives

Current `planar_circle` and `planar_annulus` variants use Shapely because they need filled-region geometry and Boolean set operations: intersect all outer disks, union inner disks, subtract inner exclusions, compute polygon centroids, and sample points inside feasible regions. The legacy `spherical_circle` helper does not build a filled region; it computes pairwise great-circle crossing points, filters those points against all circles, and returns a sparse vertex list. That is sufficient for disk-only CBG plus `boundary_vertex_mean`, but it does not naturally represent annulus holes, polygon/MultiPolygon regions, Boolean difference, area-weighted centroids, or region sampling.

This does not mean spherical geometry cannot support annuli. A future `spherical_annulus` variant is possible, but it would need a spherical polygon/Boolean geometry backend. Candidate libraries:
- Google S2 Geometry: strongest general-purpose option for robust spherical polygons, containment, centroids, and Boolean operations. Main implementation is C++; Python bindings exist but are less mature than the core library.
- `spherical-geometry`: easiest Python prototype path. It supports `SphericalPolygon.from_cone(...)`, `intersection`, `multi_intersection`, `union`, and `invert_polygon`. An annulus could be represented as `outer_cone.intersection(inner_cone.invert_polygon())`. Caveat: small circles are approximated as spherical polygons with configurable steps, so this is still a polygonal approximation on the sphere.
- PostGIS `geography`: useful in database workflows, but `ST_Intersection` on geography is documented as using a best-fit planar projection internally, so it is not a pure exact spherical-annulus backend.
- GeographicLib: excellent for geodesic polygon area/perimeter, but not a Boolean overlay engine, so it cannot directly replace Shapely for intersection/difference.

Recommended future work: prototype `spherical_annulus` with `spherical-geometry` first, then evaluate S2 if robustness or performance matters. This would directly quantify the approximation gap between current planar `(lon, lat)` Shapely annuli and true spherical polygon operations.

Reminder: the current framework uses its own sampled medoid for `monte_carlo_median` and a framework-owned Weiszfeld-style solver for `geometric_median`. If the snapped `geometric_median` path becomes a runtime bottleneck, re-evaluate integrating the `geom-median` package as an optimized continuous median backend before snapping to the nearest sampled feasible point.

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
