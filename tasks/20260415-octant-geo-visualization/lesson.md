# Octant Geolocation Visualization — Lessons

## 2026-04-15

### Weighted feasible region != strict annulus intersection

When visualizing the Octant geolocation, the yellow "feasible region" polygon appeared to extend beyond the intersection of all annulus rings. Initial assumption was a bug, but investigation revealed this is the intended behavior of `compute_feasible_region_weighted()`:

- The algorithm uses a **grid-based weighted voting** scheme, not strict geometric intersection
- Each grid cell accumulates constraint weights (`exp(-rtt/tau)`) from annuli that contain it
- A cell qualifies if its weight >= `weight_threshold * max_possible_weight` (default 50%)
- Far-away landmarks have low weight due to high RTT, so they don't veto cells
- The jagged/pixelated boundaries are artifacts of the grid resolution (0.25° cells)

**Lesson**: When building a visualization, always show exactly what the algorithm uses — not an idealized version. The weighted region is the actual sampling domain for MC points and geometric median. Showing the strict intersection instead would be misleading about what the algorithm is actually doing.

### Cartopy transform requirement

All matplotlib `plot`/`scatter`/`fill` calls on a Cartopy GeoAxes must include `transform=ccrs.PlateCarree()` when passing geographic coordinates (lat/lon). Without it, coordinates are interpreted in the projected CRS (LambertConformal) and shapes render in wrong locations. The existing `_draw_annulus` and `_plot_region` helpers from `visualize_monte_carlo.py` don't pass `transform=` since they work on plain axes — separate Cartopy-aware versions were needed.
