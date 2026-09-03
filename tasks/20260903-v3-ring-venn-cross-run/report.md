# 6-Way Ring Venn for Pooled Cross-Run `plot-venn` — Report

**Status**: Completed
**Created**: 2026-09-03
**Last Updated**: 2026-09-03

## Summary

Done. `plot-venn` pools runs and the pooled directory now carries
`overlap_ring_venn` — six equal circles on a ring, one per method, with the exact
target count printed in every region. The circles are a fixed template carrying
no data; the bullseye (`plot_nested_circles`, `set_size_table`) is gone.

## Findings

### Pooling (complete)

- as01/as02/as03 have **fully disjoint** `target_id` sets (all three pairwise
  intersections empty), so pooling is a concat: 399 + 412 + 458 = **1,269**.
- Per-method correct counts: Octant-Hull 766 (60.4%), Octant-Spline 688 (54.2%),
  SoI 616 (48.5%), Shortest-Ping 604 (47.6%), Spotter 528 (41.6%), Vanilla 432
  (34.0%).
- **Shortest-Ping and SoI CBG are nearly the same set** — 603 of their 604/616
  targets are shared. Any layout giving them separate prominent regions looks
  degenerate.
- 33 of the 63 possible regions are non-empty; 30 are empty. The top 10 regions
  hold 62% of targets. The all-six centre region holds **161**.

### Bug caught by tests

`pooled_membership` first checked each run's methods as a *subset* of the first
run's. That made the result depend on argument order: `as01 + as7018` errored,
while `as7018 + as01` silently discarded as7018's 10 ablation arms. Now requires
exact set equality, with `--method` as the explicit opt-out.

### Figure form (three rejected, one accepted)

| Form | Circles | Area-proportional | All regions | Verdict |
|---|---|---|---|---|
| Concentric bullseye | yes | yes | no | Built; not a Venn, implies false containment |
| Euler least-squares fit | yes | yes | pairwise only | Stress 0.0034; higher-order unreadable |
| 63-region Venn (triangles) | no | no | yes | Exact but unusable; 30/63 empty |
| **6-way ring (reference)** | **yes** | **no** | **31 patterns** | **Accepted** |

Geometry: 31 regions, stable for r/d 1.05–1.55; `r/d = 1.2` matches the
reference. Labels fit — smallest region ~56 px at 9 in / 200 dpi.

Coverage under the two candidate ring orders:

| Ring order | Regions shown | Targets shown |
|---|---|---|
| `PREFERRED_ORDER` (default) | 24/33 | 885 / 1,080 (81.9%) |
| target-maximising | 23/33 | 959 / 1,080 (88.8%) |

## Shipped

`plot_ring_venn` + `ring_regions` / `ring_centres` / `ring_order_for` /
`exact_combination_counts` / `ring_coverage_table` in
[venn.py](../../scripts/analysis/v3/modules/venn.py). `render_cross_overlap`
emits `overlap_ring_venn.<grid>.top<N>.png` and
`overlap_ring_coverage.<grid>.top<N>.csv`; the manifest gained `ring_order`,
`ring_drawn`, `n_regions_drawn/undrawn`, `n_targets_undrawn` and
`n_targets_none_correct`. CLI gained `--ring-order` (cross-run only, must be a
permutation). The single-run path is untouched — its artifacts still carry no
grid slug.

### Generalized beyond six

The ring is drawn for **3-8 methods**, not just six. Measured: `n` circles at
`r/d = 1.2` yield exactly `n*(n-1)+1` regions for every `n` in that range, and
those regions are precisely the subsets **contiguous around the ring**. Two
tests pin this. Below 3 methods the ring is skipped (`plot_venn` already renders
2-3 sets exactly); the arity floor also keeps the 2-method test fixture honest.

### Verified on the real data

| quantity | value |
|---|---|
| pooled targets | 1,269 |
| solved by >= 1 method | 1,080 (189 none-correct) |
| regions drawn / observed | 24 / 33 |
| targets drawn | **885** |
| targets with no region | 195 (18.1% of solved) |
| all-six centre region | **161** |

`--ring-order shortest_ping million_scale_cbg vanilla_cbg spotter_cbg
octant_cbg_hull octant_cbg_spl` reaches 959 drawn / 121 undrawn across 23
regions, matching the pre-build measurement exactly. Both orders were rendered;
the default ships.

## Conclusions

The reference layout was the right call. Every number a reader can take off the
figure is printed, so there is no encoding to misread — which is what the three
rejected forms each failed at in a different way. The cost is real and bounded:
18.1% of solved targets have no region under the default order, disclosed in the
footnote and enumerated in `overlap_ring_coverage.csv`, whose drawn rows sum to
the figure's labels.

The default stays `PREFERRED_ORDER` despite the target-maximising order showing
74 more targets. Coverage is a property of the *data*, so optimising for it would
make the figure's meaning drift between datasets — the same failure the UpSet
column order is already documented against. `--ring-order` is there for anyone
who wants the trade.
