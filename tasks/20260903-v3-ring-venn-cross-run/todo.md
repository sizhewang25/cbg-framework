# 6-Way Ring Venn for Pooled Cross-Run `plot-venn` — Todo

## Phase 0: Setup & Discovery
- [x] Confirm cross-run pooling, CLI arity, config and docs are green (258 tests)
- [x] Establish that 6 equal circles on a ring yield exactly 31 regions
- [x] Verify the 31 patterns match the reference image's label set
- [x] Confirm region labels physically fit (smallest ~56 px at 9 in / 200 dpi)
- [x] Quantify undrawable regions under both ring orders

## Phase 1: Implementation
- [x] Add ring geometry + region enumeration helper (shapely) to `venn.py`
- [x] Implement `plot_ring_venn` — fixed circles, blended fills, method labels
- [x] Label each drawable region with its count; font tiers by region area
- [x] Build the coverage table (every non-empty data region + `drawn` flag)
- [x] Add the footnote naming undrawn region and target counts
- [x] Wire into `render_cross_overlap`; emit ring PNG + coverage CSV
- [x] Record `ring_order` and undrawn counts in `manifest.json`
- [x] Delete `plot_nested_circles` and `set_size_table`
- [x] Add `--ring-order` to the CLI

## Phase 2: Verification
- [x] Test: ring yields exactly 31 patterns, and they are the expected ones
- [x] Test: geometry identical across two different membership frames
- [x] Test: each region label equals the count from `intersection_table`
- [x] Test: coverage CSV drawn/undrawn rows partition the non-empty regions
- [x] Swap out the three `plot_nested_circles` tests
- [x] Full suite green; single-run artifact names unchanged (no grid slug)
- [x] Visually compare the rendered figure against the reference
- [x] Confirm centre region reads 161 and drawn labels sum to 885
- [x] Update README (replace the nested-circle paragraph)

All phases complete 2026-09-03. 264 tests passing (was 258: +9 ring, -3 nested).
