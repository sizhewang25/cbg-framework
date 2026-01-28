# Todos

## In Progress
- [ ] Understand current CBG implementation gaps

## Pending

### Phase 1: Data Preparation
- [ ] Reverse data perspective: Vultr anchors as VPs, probes as targets
- [ ] Group target probes by ASN
- [ ] Prepare calibration dataset (known-location pairs)

### Phase 2: Bestline Calibration
- [ ] Implement lower envelope fitting (quantile regression)
- [ ] Compute per-anchor calibration parameters (slope, intercept)
- [ ] Visualize calibration: scatter plot with bestline for each anchor
- [ ] Validate calibration makes physical sense

### Phase 3: CBG Implementation
- [ ] Modify `rtt_to_km()` to use calibrated parameters
- [ ] Update CBG pipeline for reversed VP/target roles
- [ ] Test on single ASN first

### Phase 4: Evaluation
- [ ] Run CBG on all target ASNs
- [ ] Compare accuracy: calibrated vs. fixed 2/3
- [ ] Analyze errors by ASN, distance, VP count
- [ ] Generate maps and visualizations

### Phase 5: Documentation
- [ ] Document methodology
- [ ] Create summary figures
- [ ] Write findings report

## Completed
- [x] Created CBG tutorial notebook (`analysis/cbg_tutorial.ipynb`)
- [x] Identified 3 key gaps between implementation and original CBG
- [x] Documented initial analysis in notes
- [x] Located ASN analysis script with map visualization
