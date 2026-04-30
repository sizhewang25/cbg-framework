# CBG Benchmark Paper — Plan

## Background

No up-to-date, systematic benchmark of Constraint-Based Geolocation (CBG) variants exists. Academia defaults to commercial geolocation services (IPInfo, MaxMind) whose accuracy is opaque and which fail entirely on anycast IPs. The only publicly available CBG implementation is the IMC 2023 replication codebase, leaving researchers without a reproducible baseline.

This paper fills that gap: a systematic benchmark of CBG variants across the three pipeline phases (RTT-distance modeling, multilateration, single-point estimation), evaluated over curated RTT measurements from mobile vantage points and cross-validated against RIPE Atlas data.

The work is motivated by a larger goal of building a multi-tier geolocation framework for mobile operators that needs to handle both unicast and anycast IPs at scale (tens of millions of IPs). The multi-tier pipeline orders methods from easy to hard — GeoFeed → rDNS → CBG — where CBG outperforms simpler latency heuristics (GeoPing, GeoCluster) and serves as both the last prediction layer and a validation layer via latency bounds.

## Context

### The CBG pipeline abstraction

Every CBG variant can be decomposed into three phases:

| Phase | Description | Known variants |
|-------|-------------|---------------|
| **1. RTT-distance modeling** | Converts measured RTT to a distance bound per VP | 2/3c (Million-Scale), low-envelope LP fit (Vanilla CBG), bounded spline + delta band (Octant) |
| **2. Multilateration** | Intersects per-VP constraints to form a feasible region | `spherical_circle`, `planar_circle`, `planar_annulus`, `planar_annulus_weighted` |
| **3. Single-point estimation** | Collapses feasible region to one (lat, lon) | Arithmetic mean of vertices, geometric centroid (Shapely .centroid), Monte Carlo sampled medoid |

The current repo (`geoloc-imc-2023`) already implements and benchmarks 18 combinations of these phases (task `20260415-cbg-combination-evaluation`). Key finding: Octant spline + `planar_annulus` + geometric centroid achieves 328 km median error at near-zero overhead; replacing geometric centroid with MC median reduces this to 312 km but at ~130x the compute cost.

### Multi-tier geolocation pipeline (broader context)

The CBG layer sits at the bottom of a tiered pipeline:

```
Tier 1 — Official:     GeoFeed, rDNS parsing
Tier 2 — CBG:          RTT multilateration (this paper) — outperforms GeoPing / GeoCluster
Validation:            Speed-of-internet violation checks using observed RTT
```

Methods not in scope: topology-based (AS path analysis), active probing campaigns — both too slow for tens-of-millions-of-IPs scale.

### Datasets

- **Primary**: Curated RTT pings from AT&T mobile VPs to cloud provider services in the US (Vultr AS7922 subset already in `datasets/cbg_test/vultr_pings_us_only.csv`)
- **Cross-validation**: RIPE Atlas anchor meshed pings (US and EU) already in ClickHouse (`anchors_meshed_pings`, `probes_to_prefix_pings`)
- **Ground truth**: IPInfo per-anchor coordinates (`datasets/static_datasets/ip_info_geo_anchors.json`), MaxMind (`maxmind_free_geo_anchors.json`)

### Existing implementation

All 18 CBG combinations are implemented in `scripts/framework/` and evaluated by `scripts/analysis/cbg_evaluation/`. Evaluation harness, plots, and JSON results already exist for the mobile VP dataset.

## Goals

1. **Position** the paper: first systematic CBG benchmark with open-source implementations and datasets
2. **Benchmark** all three CBG phases across the known variants, reporting accuracy (median error, CDF at 40/100/500/1000 km) and compute cost
3. **Cross-validate** findings on RIPE Atlas US and EU datasets to test generalizability
4. **Characterize** CBG behavior on anycast IPs (where commercial services fail) using cloud provider anycast endpoints
5. **Publish** open-source code (`scripts/framework/`) and curated datasets for reproducibility
6. **Synthesize** practical guidance: which CBG configuration is SOTA, and when is the cost of MC median justified

## Approach

### Phase 1: Finalize benchmark (mobile VP dataset)
- Confirm all 18 combinations produce stable results on `vultr_pings_us_only.csv`
- Add availability metric: fraction of targets where CBG produces a non-null estimate
- Add per-combination compute cost (already partially logged)

### Phase 2: RIPE Atlas cross-validation
- Run the same 18 combinations on `anchors_meshed_pings` (US subset) and EU subset
- Compare accuracy distributions: do rankings hold across datasets?
- Identify any dataset-specific failure modes

### Phase 3: Anycast characterization
- Identify anycast IP targets in the dataset (BGP anycast detection via `bgp_prefixes.json`)
- Report CBG accuracy on anycast vs unicast — hypothesis: CBG degrades gracefully (latency constraint is still valid per-VP) vs commercial services (fail entirely)

### Phase 4: Paper writing
- Venue target: IMC 2026 or similar (deadline TBD)
- Sections: Introduction, Related Work, CBG Pipeline Taxonomy, Benchmark Setup, Results (mobile + RIPE), Anycast Analysis, Practical Guidance, Conclusion
- Open-source artifact: cleaned `scripts/framework/` + dataset pointers

## Caveats

- RIPE Atlas cross-validation requires ClickHouse to be running with the reproducibility data loaded
- Anycast detection is approximate (BGP-based); some anycast IPs may be misclassified as unicast
- Mobile VP dataset coverage is US-centric; EU generalizability depends on RIPE Atlas cross-validation
- MC median runtime (~27s for 266 probes) may be prohibitive for full-scale evaluation; may need to subsample or parallelize
- Ground truth quality: IPInfo and MaxMind have known inaccuracies, especially for mobile/anycast IPs — need to document ground truth limitations
