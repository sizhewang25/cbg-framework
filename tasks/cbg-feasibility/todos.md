# Todos

## In Progress
(none)

## Pending
(none - all phases complete)

## Completed

### Phase 1: RTT Modeling Module
- [x] Create rtt_model.py with core functions (haversine, fit_bestline, RTTDistanceModel)
- [x] Write unit tests in test_rtt_model.py (29 tests, all passing)
- [x] Validate haversine distance calculation
- [x] Validate bestline fitting with synthetic data

### Phase 2: Per-Anchor-ASN Model Fitting (AS7922 Comcast)
- [x] Create fit_models.py script
- [x] Fit models for all 7 Vultr anchors
- [x] Generate RTT-distance scatter plots with bestline overlay
- [x] Save model parameters as pkl files

### Phase 3: CBG Multilateration Visualization
- [x] Create visualize_cbg.py script
- [x] Test CBG on random Comcast probes (10 probes tested)
- [x] Generate interactive Folium maps with circles
- [x] Implement Shapely circle intersection for proper geometry

### Phase 4: Evaluation & Documentation
- [x] Document methodology and findings in progress.md
- [x] Record CBG accuracy metrics (mean error: 784 km, 70% within 1000 km)

### Earlier Completed
- [x] Created CBG tutorial notebook (`analysis/cbg_tutorial.ipynb`)
- [x] Identified 3 key gaps between implementation and original CBG
- [x] Documented initial analysis in notes
- [x] Located ASN analysis script with map visualization
- [x] Created implementation plan with design decisions
