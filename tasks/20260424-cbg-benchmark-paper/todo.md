# CBG Benchmark Paper — Todo

## Phase 0: Scope & Setup
- [x] Confirm venue target and submission deadline (IMC 2026 or alternative)
- [x] Decide on paper length (short/long) and section structure
- [x] Verify ClickHouse has RIPE Atlas reproducibility data loaded for cross-validation
- [ ] Identify anycast IPs in the Vultr dataset via BGP prefix analysis

## Phase 1: Finalize Mobile VP Benchmark
- [ ] Confirm all 15 active valid combinations produce stable, reproducible results on `vultr_pings_us_only.csv`
- [x] Add availability metric (fraction of targets with non-null CBG estimate) per combination
- [ ] Tabulate per-combination compute cost from evaluation logs
- [ ] Produce final accuracy table: median error + CDF at 40/100/500/1000 km thresholds
- [ ] Compare CBG variants against closest-VP / RIPE IPmap Single Radius performance; find an existing implementation if available, otherwise implement the baseline ourselves
- [ ] Rerun MC centroid combinations after `monte_carlo_median` change from continuous `geom_median` to sampled medoid; previous G/H-path accuracy and runtime numbers may shift
- [ ] Rerun the full benchmark after centroid semantics cleanup: vertex-mean centroid is now `boundary_vertex_mean`, annulus holes contribute interior vertices, and invalid `spherical_circle + geometric_centroid` combinations are excluded
- [ ] Quantify planar_circle approximation error against spherical_circle on synthetic disk cases
- [ ] Audit planar_annulus results as lon/lat planar approximations, not exact spherical geometry

## Phase 1.5: Centroid Semantics Cleanup
- [x] Rename the vertex-mean centroid to `boundary_vertex_mean` to make the method semantics explicit
- [x] Update `boundary_vertex_mean` to include interior ring vertices for `planar_annulus` / polygon holes, not only exterior vertices
- [x] Restrict `geometric_centroid` to polygon-region inputs only (`planar_circle`, `planar_annulus`, `planar_annulus_weighted`); exclude `spherical_circle` vertex-list inputs because unordered crossing vertices cannot reliably form a valid polygon
- [x] Reconcile `monte_carlo_median` with Octant's original point-selection semantics: Octant selects one sampled point inside the estimated region, while the previous code used `geom_median.numpy.compute_geometric_median(...)`, which may return an unconstrained point outside the region
- [x] Implement Octant-faithful `monte_carlo_median` as sampled medoid via minimum total pairwise distance, ensuring the final estimate is a sampled feasible point
- [x] Add `geometric_median` centroid as the faster continuous geometric median snapped to the nearest sampled feasible point

## Phase 2: RIPE Atlas Cross-Validation
- [ ] Extract US anchor subset from `anchors_meshed_pings` for US-only comparison
- [ ] Run all 15 active valid combinations on RIPE Atlas US dataset
- [ ] Run all 15 active valid combinations on RIPE Atlas EU dataset
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
- [ ] Clarify GNP vs CBG geometry: GNP artificial network coordinates are not physical Earth coordinates; planar_circle and planar_annulus use real lat/lon but are still planar approximations over geographic coordinates
- [ ] Label methods precisely: spherical_circle is the exact spherical disk baseline; planar_circle/planar_annulus are polygonal approximations in `(lon, lat)` degree space
- [ ] Draft Benchmark Setup section (datasets, ground truth, metrics)
- [ ] Draft Results: mobile VP benchmark (tables + CDF figures)
- [ ] Draft Results: RIPE Atlas cross-validation
- [ ] Draft Anycast Analysis section
- [ ] Draft Practical Guidance / Discussion
- [ ] Draft Conclusion
- [ ] Internal review pass
- [ ] Prepare open-source artifact (cleaned `scripts/framework/` + README)
- [ ] Submit
