# RTT Filtering Task TODOs

## Completed
- [x] Research statistical approaches for lower bound estimation
- [x] Evaluate assumption: "VP to ASN probes have similar routes"
- [x] Evaluate linear vs non-linear fitting (Bezier curves)
- [x] Research quantile regression theory
- [x] Create quantile regression demonstration
- [x] Compare QR vs LP+filter on synthetic data
- [x] Compare QR vs LP+filter on real AS7922 data
- [x] Document findings

## In Progress
- [ ] Integrate quantile regression into `rtt_model.py`
  - Add `fit_bestline_quantile()` function
  - Add `method='quantile'` option to `RTTDistanceModel.fit()`

## Pending
- [ ] Run comparative evaluation on all anchors
- [ ] Compare CBG geolocation error: QR vs LP methods
- [ ] Decide on recommended default method
- [ ] Update filter_demonstration.ipynb with QR comparison
- [ ] Consider hybrid approach (basic filter + QR)

## Questions to Resolve
- Should we enforce speed-of-light constraint in QR? (post-hoc vs pre-filter)
- What τ value is optimal? (0.05 vs 0.10)
- Do we need different τ for different distance ranges?
