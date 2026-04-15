# Octant Geolocation Visualization — Todo

## Phase 0: Setup & Core Script
- [x] Create `visualize_octant_geolocation.py` alongside `octant_spline_visualization.py`
- [x] Data loading: load Vultr CSV, filter by ASN, compute distances, extract anchor_coords
- [x] Model fitting: call `fit_octant_models()` with configurable cutoff variant
- [x] Per-target geolocation with retained intermediates (constraints, region, sampled points)

## Phase 1: Map Visualization
- [x] Convert from plain matplotlib to Cartopy GeoAxes (LambertConformal projection)
- [x] Add map features (land, ocean, state borders, coastlines)
- [x] Draw annulus rings per landmark with colorblind-safe palette
- [x] Plot weighted feasible region with yellow fill / orange outline
- [x] Plot geometric median (blue diamond) and true location (red star)
- [x] Fixed full-US extent (`[-130, -64, 24, 55]`) for all maps
- [x] Error/method annotation text box

## Phase 2: Output & Polish
- [x] Individual per-target PDF figures
- [x] Grid summary figure (2xN layout)
- [x] CLI with argparse (--asn, --n-targets, --seed, --cutoff-variant, --method, etc.)
- [ ] Re-enable MC scatter overlay (currently commented out)
- [ ] Consider adding weight heatmap overlay to show the grid-based weighting
