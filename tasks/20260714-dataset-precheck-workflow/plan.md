# Dataset Precheck Workflow — Plan

## Background

Before running a CBG benchmark (e.g. `scripts/benchmark/v2/config/as7018_us_test01.yaml`),
we want a precheck that characterizes the *dataset's* topological and RTT
properties — independent of any benchmark results. The intuition: how well
could shortest-ping / CBG work if the **whole canonical CSV** were used as the
evaluation set, with all properties made explicit. Today
`scripts/visualization/benchmark/inspect_source.py` reports only flow/count
statistics, and `scripts/benchmark/v2/eval_source.py` covers part of the
geography/RTT axes — the topology block (clustering geometry, VP-proximity
margins, RTT regimes) is missing.

## Context

- **Input**: one canonical-schema CSV (one row per `(vp, target, rtt_ms)`
observation), e.g. `datasets/ripe_as7018/as7018-us-test01.csv`. Full CSV,
**not per-fold** — in `probes_to_anchors` folds split targets, so per-target
VP availability from the full CSV is exact anyway.
- **Existing building blocks**:
  - `scripts/benchmark/v2/eval_source.py` — pairs frame, per-target
  closest/shortest-ping VP, min_inflation, rtt_weighted_dist_km,
  `cluster_targets` (answer-space clustering + within-R / same-cluster shares).
  - `scripts/benchmark/v2/sources/cluster_ground_truth.py` — R-coherent
  complete-linkage clustering (centroid-radius capped).
  - `scripts/benchmark/v2/cluster_topn.py` — `build_truth_neighbor_index`
  (top-N nearest centroids per centroid).
  - `scripts/analysis/_fleet_geometry.py` — metric *definitions* to port
  (cell_gap_km, target_distinguishable_vp_dist/margin_km,
  has_target_distinguishing_vp); do NOT call it (it consumes materialized
  parquet + a centroid index, i.e. benchmark-side artifacts).
  - `scripts/visualization/benchmark/inspect_source.py` — occurrence CDFs +
  Leaflet flow map (templates in `scripts/visualization/benchmark/templates/`).
  - `.smk` style reference: `scripts/visualization/benchmark/v2/cluster_world_map.smk`
  (reads benchmark yaml configs to derive params).

## Architecture decisions (settled)

1. **All precheck metric logic lives in `eval_source.py`**, consuming only the
  canonical CSV — never benchmark outputs.
2. **All visualization lives in `inspect_source.py`** — keeps its flow
  CDFs/map, gains the cluster/Voronoi map layer; consumes eval_source's
   per-target CSV / stats JSON rather than recomputing metrics.
3. **New `inspect_dataset.smk`** orchestrates: canonical CSV → eval_source
  metrics → inspect_source visuals. **The benchmark config yaml is the
   parameter source** (csv_path from `source_kwargs`, cluster radius R;
   later LTD kwargs) so the precheck describes *this* run's setup.
4. Clipped/constrained Voronoi (mainland-US etc.) is a **map layer only**
  (inspect_source); metrics use centroid distances (gap + top-N), no
   landmass-polygon dependency in eval_source.
5. Per-VP training coverage (LTD fittability) is a **separate script, later
  task** — it must be k-fold aware, unlike this full-CSV precheck.

## Metric design (settled)

### Target clustering topology (extends `cluster_targets`)

- `cell_gap_km` — distance to nearest *other* centroid (port
`_nearest_other_centroid_km` from `_fleet_geometry.py`).
- Top-N neighbor clusters per cluster — reuse `build_truth_neighbor_index`;
record neighbor ids + distances in a per-cluster output.

### VP proximity per target — **cluster-centroid based**

- All VP-proximity distances measured to the target's **truth cluster
centroid**, with R an explicit required parameter. R=0 degenerates to
target-based distances (every target a singleton, centroid = target).
- `closest_vp_to_centroid_km`, `shortest_ping_vp_to_centroid_km`.
- `target_distinguishable_vp_dist_km` = cell_gap_km / 2.
- **Discriminative VP set** D per target = VPs whose distance to the truth
centroid < cell_gap/2 (per-VP margin > 0) — the VPs guaranteed to favor the
truth centroid over the nearest competitor. The closest VP is just D's
deepest member; agreement is tested against the whole set:
  - `n_discriminative_vps` = |D|; `has_vp_proximity` = |D| > 0
  (equivalent to closest-VP margin > 0) → "fleet **has** proximity".
  - `shortest_ping_vp_is_discriminative` = shortest-ping VP ∈ D → "fleet
  **uses** proximity" (shortest-ping is determined once the topology is
  fixed — it's where the baseline snaps).
  - `best_discriminative_rtt_rank` — min normalized RTT rank over D: how
  buried the discriminative set is in the RTT ordering (generalizes
  `closest_vp_rtt_rank`, which stays as the simple special case).
  - Proximity labels are **purely D-based** (a ladder; renamed 2026-07-14
    so the HAS_USED ⊂ has-proximity nesting is explicit, and the SPARSE
    label dropped — it was ambiguous against the D-based definitions, and
    a sparse target with D empty is simply NO_PROXIMITY):
    NO_PROXIMITY (D empty, no VP meets the cell_gap/2 requirement) →
    HAS_NOT_USED_PROXIMITY (proximate VPs exist, shortest-ping VP ∉ D) →
    HAS_USED_PROXIMITY (shortest-ping VP ∈ D). The two HAS_* shares sum to
    the has-proximity superset (geometric ceiling for any proximity
    method). Voronoi assignment does NOT drive the labels — it is a
    diagnostic *within* NO_PROXIMITY: ∈ D is sufficient for
    Voronoi-assigning to the truth cluster (d < gap/2 ⇒ every competitor
    is farther) but not necessary (direction-free bound), so
    `shortest_ping_vp_in_same_cluster` splits the opportunity population
    into "baseline still lands correctly by directional luck" vs "baseline
    provably snaps to a non-truth cluster".
- **CBG opportunity** (headline metric, final definition 2026-07-14):
`cbg_opportunity_share` = NO_PROXIMITY + HAS_NOT_USED_PROXIMITY — every
target where the shortest-ping baseline carries no correctness guarantee.
User settled on this after two same-day flips (NO+HAS → NO-only →
NO+HAS): "opportunity" reads as *the baseline is beatable here*, whether
by multilateration (NO_PROXIMITY: no discriminative VP exists) or by
better VP selection / multilateration (HAS_NOT_USED: a discriminative VP
exists but the baseline misses it). The individual label shares keep the
finer split available, so "only-CBG-can-help" is still readable as
no_proximity_share alone. The Voronoi diagnostic
(`opportunity_baseline_lucky_share`, computed over the combined
opportunity population) reports what fraction the baseline nevertheless
converts by directional luck.

### RTT stats per target

- `rtt_dist_spearman` — Spearman ρ(gc_km, rtt_ms) over the target's pairs
(guard: needs a minimum pair count; NaN below it).
- Closest vs shortest-ping disagreement: boolean + km gap between the two VPs.
- SOI violations: share of pairs with rtt < THEORETICAL_SLOPE × gc (faster
than 2/3c ⇒ bad ground truth or anycast) — reported per pair (dataset level)
and per target.
- Anycast / infeasibility test over the low-RTT VP set: does any pair have
`radius_i + radius_j < d(vp_i, vp_j)` (disjoint constraint disks)? Also
predicts empty-intersection MTL failures. Formulation settled — see the
"Anycast / infeasibility" section below.

### Inflation-at-proximity (settled 2026-07-14, revised same day)

Raw inflation ratio rejected (explodes at short distances from last-mile
constants). `closest_vp_excess_radius_km` (magnitude-of-inflation in km) was
proposed then **dropped** — keep the diagnostic rank-and-geometry based only:

- `closest_vp_rtt_rank` — normalized rank of the geographically closest VP in
the target's RTT ordering; 0 = proximity is used, high = the fleet's best
geographic asset is RTT-poisoned relative to the fleet.
- **Distance mesh artifacts** — two great-circle distance matrices emitted
once per dataset:
  - VP×VP mesh — feeds the anycast pair-slack test and the per-target
  closest↔shortest-ping VP separation. Diagnostic rule: if closest VP ≠
  min-RTT VP **and** d(closest_vp, shortest_ping_vp) is large, then either
  RTT inflation (congestion) or indirect routing is happening at the close
  VP. The disagreement columns (bool + km gap) carry this per target.
  - Cluster×Cluster centroid mesh (revised 2026-07-14; was Target×Target) —
  underlies the cell gaps / top-N neighbors; emitted over cluster centroids
  instead of raw targets because a target×target matrix explodes at
  million-scale target counts while the centroid count stays bounded by the
  clustering. Exposed for neighbor inspection and the viz layer.
- Per-target Spearman ρ stays as the regime/coherence descriptor, not an
inflation detector (global rank coherence; one inflated closest VP barely
moves ρ yet flips the CBG answer — head-of-ordering diagnostics above cover
that case).

### Anycast / infeasibility (settled 2026-07-14)

- Low-RTT VP set: RTT ceiling relative to the target's best —
`rtt ≤ min_rtt + δ`, δ = 10 ms default (not fixed-k, which would pull in
far VPs whose big disks never conflict).
- `vp_pair_disk_overlap_km` = min over low-RTT VP pairs of
(r_i + r_j − d(vp_i, vp_j)) — negative ⇒ infeasible (disjoint-disk) pair
exists; magnitude = separation strength. Doubles as empty-feasible-region
predictor for the MTL stage.
- `n_disjoint_sites` — iGreedy-style greedy count of mutually disjoint
low-RTT disks; ≥ 2 ⇒ anycast-suspect, count ≈ visible sites. Encodes the
VP-geo-spread requirement natively (disjointness ⇒ spread > r_i + r_j).

## Output artifacts (eval_source, per canonical CSV)

- `<stem>_eval_per_target.csv` — extended per-target table (proximity margins,
RTT-regime columns, cluster_id, regime label)
- `<stem>_eval_clusters.csv` — per-cluster: centroid, n_members, radius_km,
cell_gap_km, top-N neighbor cluster ids + distances
- `<stem>_vp_mesh_km.csv` / `<stem>_cluster_mesh_km.csv` — distance matrices
(id-indexed; skipped above the `mesh_max_n` cap — full float matrix is O(n²);
answer-space mesh is over cluster centroids, not raw targets)
- `<stem>_eval_stats.json` — dataset summary blocks (existing + topology +
proximity margins + regime shares)

Visualization (inspect_source): existing occurrence CDF png + flow map html,
plus the cluster/Voronoi map layer (clipped to geometry constraint).

## Interpretation flow

**Dataset-level reading order** (how the stats JSON is meant to be read):

1. *Scale & coverage* — n_vps, n_targets, flows per endpoint (inspect_source
  CDFs): is the mesh dense enough to say anything?
2. *Answer space* — n_clusters at R, targets/cluster, cell_gap distribution,
  top-N neighbor distances: how separable is the classification problem?
   Small gaps ⇒ even perfect proximity cannot distinguish neighbors.
3. *Fleet HAS proximity* — share with |D| > 0 (discriminative set non-empty):
  the geometric ceiling for any RTT method on this dataset.
4. *Fleet USES proximity* — share with shortest-ping VP ∈ D: what the
  baseline is guaranteed. `cbg_opportunity_share` = NO_PROXIMITY +
   HAS_NOT_USED_PROXIMITY — the dataset's headline: everywhere the
   baseline is beatable. NO_PROXIMITY is the only-multilateration slice;
   HAS_NOT_USED_PROXIMITY has a discriminative VP that is RTT-buried, so
   better VP selection can also fix it.
5. *RTT quality regimes* — Spearman ρ distribution, SOI shares, anycast
  suspects: which targets break for non-geometric reasons, and why.

**Per-target decision path** (columns chained into a regime label):

```
D = discriminative VP set = {VPs with dist-to-truth-centroid < cell_gap/2}

Proximity label (D-based ladder, mutually exclusive; no SPARSE label —
n_avail_vps stays a plain per-target metric, a sparse target with D empty
is just NO_PROXIMITY):
D empty ──────────────────────────────────────────→ NO_PROXIMITY
   (proximity cannot work, only multilateration can; expect
    neighbor-cell errors ~ cell_gap)
   diagnostic within: shortest_ping_vp_in_same_cluster?
     true  → baseline still lands correctly (directional luck)
     false → baseline provably snaps to a non-truth cluster
└─ D non-empty, shortest-ping VP ∉ D ─────────────→ HAS_NOT_USED_PROXIMITY
   (congestion or indirect routing buried the discriminative set —
    best_discriminative_rtt_rank says how deep; better VP selection or
    multilateration can fix it)
Opportunity = NO_PROXIMITY + HAS_NOT_USED_PROXIMITY (baseline beatable).
└─ shortest-ping VP ∈ D ──────────────────────────→ HAS_USED_PROXIMITY
   (guaranteed by the bound; expect baseline & CBG to win; ρ high)

Parallel flags (orthogonal to the ladder):
├─ vp_pair_disk_overlap_km < 0 / n_disjoint_sites ≥ 2 → ANYCAST_SUSPECT
│    (also predicts empty-feasible-region MTL failures)
└─ SOI violation share > 0 → ground-truth suspect (VP or target coords)
```

Label shares roll up into the dataset summary: "X% has-used-proximity, Y%
has-not-used-proximity, W% no-proximity (cbg_opportunity_share = Y + W,
Z% of it baseline-lucky), plus anycast/SOI flag shares" as the headline.

## Open questions

1. Output naming/layout for the smk (per-config out dir vs alongside CSV).
2. ~~SPARSE floor for n_avail_vps~~ — resolved 2026-07-14: no SPARSE label
  (ambiguous against the D-based definitions; folded into the ladder,
  `n_avail_vps` stays a plain metric). The Spearman guard keeps its own
  `spearman_min_pairs` (8). The disagreement split needs no threshold —
  the exact Voronoi test replaced the distance-cutoff heuristic.

## Goals

- `eval_source.py` emits per-target CSV + per-cluster CSV + stats JSON with
the full topology/proximity/RTT-regime blocks, from the canonical CSV alone.
- `inspect_source.py` renders the cluster/Voronoi map layer (clipped to a
geometry constraint like mainland US) + proximity/margin coloring on top of
its existing outputs.
- `inspect_dataset.smk` runs the whole precheck from a benchmark config yaml.
- Verified end-to-end on `datasets/ripe_as7018/as7018-us-test01.csv` with
`as7018_us_test01.yaml`.

## Caveats

- Precheck is full-CSV by design; per-fold caveat only matters for setups
where the VP side gets folded (note it in the docstring).
- `cluster_ground_truth` at R=0: verified 2026-07-14 — all singletons, no
crash. Edge: exact-duplicate coordinates become two singleton clusters at
the same spot, so their cell_gap_km = 0 and margins are always negative
(honest — two indistinguishable answers — but document it).
- `cluster_ground_truth` builds a precomputed haversine distance matrix for
complete-linkage — O(n²) memory in raw target count. This precheck is meant
to scale to million-target sources, so it's fixed (2026-07-15) to cluster
over *unique* `(lat, lon)` rows only (duplicate coords sit at distance 0 and
can never change a merge decision) and broadcast labels back — O(u²) for u
unique coordinates instead of O(n²). Only helps when the source has real
coordinate duplication (e.g. many targets pinned to one data-center
location); a source with u ≈ n (all-distinct coordinates, e.g. RIPE anchors)
gets no benefit and would need a sparse/connectivity-graph approach instead
if it ever blows up memory on its own.
- The `eval-source` CLI command in `scripts/benchmark/v2/cli.py` wraps
`eval_source` — keep it working (add the new params there too).
- Landmass polygon for the clipped Voronoi layer is a new dependency
(geopandas is already in the project; a Natural-Earth-style polygon needed) —
visualization-only, must not leak into eval_source.

