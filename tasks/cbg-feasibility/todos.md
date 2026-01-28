# Todos

## In Progress
- [ ] Create rtt_model.py with core functions (haversine, fit_bestline, RTTDistanceModel)

## Pending

### Phase 1: RTT Modeling Module
- [ ] Write unit tests in test_rtt_model.py
- [ ] Validate haversine distance calculation
- [ ] Validate bestline fitting with synthetic data

### Phase 2: Per-Anchor-ASN Model Fitting (AS7922 Comcast)
- [ ] Create fit_models.py script
- [ ] Fit models for all 7 Vultr anchors
- [ ] Generate RTT-distance scatter plots with bestline overlay
- [ ] Save model parameters as pkl files

### Phase 3: CBG Multilateration Visualization
- [ ] Create visualize_cbg.py script
- [ ] Test CBG on random Comcast probes
- [ ] Generate interactive Folium maps with circles
- [ ] Verify circle intersection contains true location

### Phase 4: Evaluation & Documentation
- [ ] Compare calibrated vs. fixed 2/3 accuracy
- [ ] Document methodology and findings

## Completed
- [x] Created CBG tutorial notebook (`analysis/cbg_tutorial.ipynb`)
- [x] Identified 3 key gaps between implementation and original CBG
- [x] Documented initial analysis in notes
- [x] Located ASN analysis script with map visualization
- [x] Created implementation plan with design decisions
