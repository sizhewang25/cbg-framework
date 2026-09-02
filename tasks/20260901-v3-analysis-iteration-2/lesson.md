# v3 Analysis Iteration 2 — Lessons

## 2026-09-01

- A rendering warning can be a data finding. matplotlib-venn's "Bad circle
  positioning" was not a styling nuisance — it was the library correctly
  reporting that the correctness sets are nested and no area-proportional
  layout exists. Chasing the warning surfaced
  `all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` on all four runs.
- Keeping the raw artifact policy-free paid off. Because
  `<method>_seed_distances.parquet` stores distances for FALLBACK rows rather
  than nulling them, it was possible to *measure* that all 106 of Vanilla's
  as01 fallbacks would have been scored correct. Had the artifact applied the
  §7.2 policy at write time, the fallback cost would have been unmeasurable.

- Check that a library function actually works before naming it in a plan.
  `venn2_unweighted` / `venn3_unweighted` exist in matplotlib-venn 1.1.2 and
  have the right signature — verifying that much passed — but they are
  deprecated *and* broken: they forward `normalize_to` into a layout algorithm
  that rejects it, so every call raises `ValueError`. Signature inspection is
  not the same as a smoke call. The working route is
  `venn2(..., layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1)))`.

- Cartopy silently escalates to 10m Natural Earth data — and downloads it —
  when the map extent is small. A single-point auto-extent in a unit test turned
  a hermetic suite into a networked one and pushed runtime from 2.9s to 13s.
  Any test that renders a map should pass an explicit, wide extent.

- A Venn cannot show the region outside its circles, so the "no method correct"
  count is invisible unless annotated. It is 70/399 on as01 and 30/78 on
  as7018 — large enough that omitting it misleads.
