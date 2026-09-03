# 6-Way Ring Venn for Pooled Cross-Run `plot-venn` — Plan

## Background

`plot-venn` scored one run at a time, so the set-overlap picture for the three
operator datasets (as01/as02/as03) was split across three figures that could not
be compared. The task was to pool them and add a circle figure where each of the
six geolocation methods is a circle.

The circle figure went through three rejected forms before the reference landed:

1. **Concentric bullseye** (built, shipped, now being replaced) — chosen when
   area-proportionality was the goal, but it does not read as a Venn diagram and
   its nesting falsely implies containment.
2. **Least-squares Euler fit** — circles, area-proportional, pairwise overlaps
   fitted to stress 0.0034. Rejected: higher-order regions unreadable.
3. **True 63-region Venn** (`venn==0.1.3`, triangles) — mathematically exact but
   visually unusable, and 30 of 63 regions are empty on this data.

The user then supplied a reference image — the standard "6-way Venn" template of
six equal circles on a ring — with the instruction: *"keep the circle position
and size intact, but just put data labels on the intersection."*

That resolves the tension that drove all three rejections: **the geometry is a
fixed template carrying no data, and every number is a region label.** There is
no area encoding.

## Context

- **Figure target**: `outputs/analysis/v3/_cross/venn-diagram/as01+as02+as03/`
- **Code**: [scripts/analysis/v3/modules/venn.py](../../scripts/analysis/v3/modules/venn.py)
- **Pooled population**: 1,269 targets (399 + 412 + 458), disjoint across runs
- **Six methods**: `shortest_ping`, `million_scale_cbg`, `vanilla_cbg`,
  `octant_cbg_hull`, `octant_cbg_spl`, `spotter_cbg`
- **Dependencies**: `shapely` 2.1.2 and `scipy` 1.17.1 already installed. No new
  dependency. `venn==0.1.3` was fetched to a scratch dir while evaluating form 3
  and is deliberately **not** in the project venv.

### Measured geometry facts

- Six equal circles on a ring produce exactly **31 regions** — 6 singles, 6
  adjacent pairs, 6 triples, 6 quads, 6 quints, 1 centre. This matches the
  reference's label set exactly and is stable for radius/ring ratio 1.05–1.55.
  `r/d = 1.2` matches the reference visually.
- **Labels fit.** At 9 in / 200 dpi the union spans ~1700 px; the smallest region
  (5-set sliver, 0.108% of union) is ~56 px across; median region ~287 px.
- **Not every region has a home.** The data has 33 non-empty regions; the ring
  draws 31 patterns. Which of the 33 land on one depends on ring order:
  - `PREFERRED_ORDER`: 24/33 regions, **885 of 1,080** solved targets (81.9%)
  - target-maximising order: 23/33 regions, **959 of 1,080** (88.8%)

## Goals

1. `overlap_ring_venn.<grid>.top<N>.png` matching the reference: six equal
   circles on a ring, per-method hue, every drawable region labelled with its
   exact target count.
2. Geometry provably independent of the data — same centres and radii in every
   run.
3. The regions with no home are disclosed on the figure and enumerated in
   `overlap_ring_coverage.<grid>.top<N>.csv`.
4. Retire the bullseye (`plot_nested_circles`, `set_size_table`) — its only
   reason to exist was the area encoding, now explicitly out of scope.
5. Cross-run pooling behaviour and the single-run path stay unchanged.

## Approach

`plot_ring_venn(membership, out_path, *, title, subtitle, ring_order=None)`:

- Fixed geometry: six equal circles, centres on a ring of radius `d = r / 1.2`,
  first method at 12 o'clock, clockwise.
- Region polygons via `shapely` — intersect the member circles, difference the
  rest.
- Fill each circle at `alpha≈0.30` in its `method_colors` hue so overlaps blend
  like the reference; outlines at full hue.
- Label each drawable region with its count at `representative_point()`, falling
  back to `shapely.ops.polylabel` for concave slivers. Font tiers by region area
  (~7 pt for 5-set slivers, ~13 pt for singles). Method names outside the ring.
- Ring order defaults to `PREFERRED_ORDER` so the figure means the same thing in
  every run — the rule `plot_upset` already follows via `sort_categories_by=None`.
  `--ring-order` accepts an explicit list for the target-maximising arrangement.

## Caveats

- **~11–18% of solved targets have no region.** Inherent to the template, not a
  bug. Must be stated on the figure, or a reader concludes those targets do not
  exist. This is the single biggest correctness risk in the task.
- **Data-dependent ring ordering is not the default.** It would silently change
  the figure's meaning between datasets — the same failure mode the codebase
  already warns about for UpSet column order.
- Counts, not percentages — the denominator lives in the title.
- The reference is a *presentation* 6-way Venn (31 regions), not a mathematical
  6-set Venn (63 regions). Do not describe the output as showing every possible
  intersection.
