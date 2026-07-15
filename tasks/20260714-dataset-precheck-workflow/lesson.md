# Dataset Precheck Workflow — Lessons

## 2026-07-15

- When a clustering/distance step is O(n²) in a quantity meant to scale to
  millions, check whether the *inputs* actually have that much entropy before
  reaching for a fundamentally different algorithm. `cluster_ground_truth`'s
  distance matrix was sized by raw target count, but real sources (cloud VM
  targets pinned to shared data-center coordinates) often have far fewer
  *unique* coordinates than targets — clustering on the dedup'd set and
  broadcasting labels back is free correctness-wise (duplicates are 0 apart,
  so they never change a merge) and cuts memory to O(u²). It's a targeted fix,
  not a general one: sources with u ≈ n (all-distinct coordinates) get no
  benefit and would need a real algorithmic change (sparse connectivity graph
  instead of a dense precomputed matrix).

## 2026-07-14

- `eval_source.py` already covered ~60% of the envisioned precheck (clustering,
  closest/shortest-ping VP, within-R shares) — extending it beat adding
  topology logic to `inspect_source.py`, which stays visualization-only.
- `_fleet_geometry.py` metrics are the right *definitions* but the wrong
  *dependency*: it consumes materialized benchmark inputs. Port the math, not
  the function, to keep the precheck benchmark-independent.
- Raw RTT inflation (ratio to 2/3c floor) is misleading at short distances —
  last-mile constant milliseconds blow up the ratio for nearby VPs. Any
  proximity-conditioned inflation metric needs an absolute (km-slack) or
  near-VP-min formulation.
- "Closest VP" is not the unit of proximity agreement — the *discriminative
  set* (all VPs with margin > 0) is. The closest VP is merely its deepest
  member; testing shortest-ping VP against the closest VP alone mislabels
  targets whose baseline snaps to a different-but-still-discriminative VP
  as disagreement.
- Don't invent a distance heuristic when an exact test exists: cell_gap/2 is
  a direction-free *sufficient* bound (∈ D ⇒ Voronoi-assigns to truth), but
  ∉ D does not imply a wrong snap — the competitor may lie the other way.
  The parameter-free Voronoi assignment (`shortest_ping_vp_in_same_cluster`)
  is the correct disagreement split; the proposed
  `shortest_ping_to_discriminative_km` cutoff was redundant and needed an
  arbitrary threshold.
- The `cbg_opportunity_share` definition flipped twice (NO+HAS → NO-only →
  final NO+HAS_NOT_USED): both readings are coherent, and the flip-flop
  happened because the name doesn't pin one down. "Opportunity" as
  *only-CBG-can-help* points to NO_PROXIMITY alone; as *the baseline is
  beatable* it includes HAS_NOT_USED_PROXIMITY. Final call: baseline-
  beatable (NO + HAS_NOT_USED), with the individual label shares kept in
  the JSON so the finer only-CBG reading is still one subtraction away.
  When a headline aggregates labels, state the semantics in the docstring
  next to the formula — the formula alone re-opens the debate.
- Keep the precheck free of method-flavored axes: `best_radius_km`
  (rtt × 100 km) and the threshold "resolvability" table baked one LTD's
  slope into a dataset-level report and were removed — constraint-radius
  analysis belongs to the LTD-specific study, not the source precheck.
- Size artifacts by what stays bounded, not what's convenient: the
  target×target mesh was O(n²) in a quantity meant to reach millions —
  the cluster×cluster centroid mesh carries the same answer-space
  geometry (gaps, neighbors) at a size bounded by the clustering, and
  the mesh_max_n cap stops being the thing that silently drops the
  artifact on real datasets.
- Label names should carry the set structure: the final ladder
  (NO_PROXIMITY / HAS_NOT_USED_PROXIMITY / HAS_USED_PROXIMITY) makes
  "used ⊂ has" explicit, which the earlier HAS/USE sibling naming hid. A
  SPARSE gate on top of a D-based ladder was ambiguous (a sparse target
  could sit anywhere in D-space) — dropped; n_avail_vps stays a plain
  metric.
