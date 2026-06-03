"""Paper figures for the four fundamental CBG variants across VP setups.

Shared foundation:
  - `_variant_style`: fixed combo→label / combo→color mapping (reused by every figure).
  - `_io`: config loader + data assembly (fold-merge, closest-VP distance, JSON dumps).

Figure scripts (config-driven typer CLIs, write to scripts/paper/cbg_bench/<run_id>/):
  - `plot_error_cdf`        : deliverable 1 — 6 per-setup SUCCESS-only error CDFs.
  - `summary_table`         : deliverable 2 — per-setup percentile + fallback% table.
  - `plot_error_vs_vp_dist` : deliverable 3 — error_km vs closest-VP-distance, faceted by variant.
  - `plot_paired_scatter`   : deliverable 4 — within-pair paired error scatters.
"""
