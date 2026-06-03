# Four-CBG-Variant Analysis Across VP Setups — Lessons

## 2026-06-02

- The four variants are two clean LTD pairs sharing a geometry stack
  (vanilla/SOI = spherical_circle + boundary_vertex_mean; octant/spotter =
  planar_annulus + monte_carlo_medoid). Only within-pair contrasts isolate the
  LTD; disk-vs-annulus is a 3-change stack effect.
- `_nofil` = unweighted `planar_annulus` (every constraint ANDed equally), which
  is the collapse-prone configuration — fallback% must be reported separately
  from error.
