# CBG Benchmark Paper — Todo

## Phase 0: Scope & Setup
- [ ] Confirm venue target and submission deadline (IMC 2026 or alternative)
- [ ] Decide on paper length (short/long) and section structure
- [ ] Verify ClickHouse has RIPE Atlas reproducibility data loaded for cross-validation
- [ ] Identify anycast IPs in the Vultr dataset via BGP prefix analysis

## Phase 1: Finalize Mobile VP Benchmark
- [ ] Confirm all 18 combinations produce stable, reproducible results on `vultr_pings_us_only.csv`
- [x] Add availability metric (fraction of targets with non-null CBG estimate) per combination
- [ ] Tabulate per-combination compute cost from evaluation logs
- [ ] Produce final accuracy table: median error + CDF at 40/100/500/1000 km thresholds
- [ ] Quantify Shapely polygon approximation error against spherical disk intersection on synthetic disk cases
- [ ] Audit Shapely-based annulus results as lon/lat planar approximations, not exact spherical geometry

## Phase 2: RIPE Atlas Cross-Validation
- [ ] Extract US anchor subset from `anchors_meshed_pings` for US-only comparison
- [ ] Run all 18 combinations on RIPE Atlas US dataset
- [ ] Run all 18 combinations on RIPE Atlas EU dataset
- [ ] Compare accuracy rankings across mobile VP, RIPE US, and RIPE EU datasets
- [ ] Document any combination whose ranking shifts significantly across datasets

## Phase 3: Anycast Characterization
- [ ] Identify anycast vs unicast targets using `bgp_prefixes.json` BGP anycast detection
- [ ] Run top-3 CBG combinations on anycast targets
- [ ] Compare CBG accuracy on anycast vs unicast
- [ ] Benchmark commercial services (IPInfo, MaxMind) on the same anycast targets for contrast

## Phase 4: Paper Writing
- [ ] Draft Introduction + Related Work
- [x] Draft CBG Pipeline Taxonomy section (3-phase abstraction + variant table)
- [ ] Clarify GNP vs CBG geometry: GNP artificial network coordinates are not physical Earth coordinates; Shapely CBG uses real lat/lon but is still a planar approximation over geographic coordinates
- [ ] Label methods precisely: spherical intersection is the exact spherical disk baseline; Shapely disk/annulus methods are polygonal approximations in `(lon, lat)` degree space
- [ ] Draft Benchmark Setup section (datasets, ground truth, metrics)
- [ ] Draft Results: mobile VP benchmark (tables + CDF figures)
- [ ] Draft Results: RIPE Atlas cross-validation
- [ ] Draft Anycast Analysis section
- [ ] Draft Practical Guidance / Discussion
- [ ] Draft Conclusion
- [ ] Internal review pass
- [ ] Prepare open-source artifact (cleaned `scripts/framework/` + README)
- [ ] Submit
