# RIPE Atlas Source — Data Characteristics

Reference doc for `RipeAtlasSource` ([ripe_atlas.py](ripe_atlas.py)), the primary
eval path in the v2 benchmark. Numbers below are from the IMC 2023
reproducibility dataset bundled with the repo, regenerated via
[characterize_ripe_atlas.py](../../../analysis/characterize_ripe_atlas.py) on
2026-05-23. To refresh:

```
python -m scripts.analysis.characterize_ripe_atlas \
    --output scripts/analysis/outputs/ripe_atlas_characterization.json
```

## Source class

- **name**: `"ripe_atlas"`
- **Dataset**: IMC 2023 reproducibility
  (`reproducibility_probes_and_anchors.json`)
- **RTTs**: ClickHouse table `ping_10k_to_anchors` via
  `scripts.analysis.analysis.compute_rtts_per_dst_src` (threshold=70, matches
  v1/`million_scale.py`)
- **Sanitization**: on by default. Re-runs the paper's SOI-violation removal at
  load time (see [Sanitization](#sanitization) below).

### Setups (`DataSource.ALLOWED_SETUPS`)

| Setup | VPs | Eval targets | Notes |
|---|---|---|---|
| `PROBES_TO_ANCHORS` (default) | 9,683 probes | 752 anchors | Hard ground truth on the target side |
| `ANCHORS_TO_PROBES` | 752 anchors | 9,683 probes | Roles swapped; harder pressure test |

(Counts after sanitization and after dropping IPs that don't appear in
`ping_10k_to_anchors` or are missing geometry; matches what the iterators yield.)

### Slices (`_apply_slice`)

- `"all_anchors"` — every anchor with ≥1 valid (probe, RTT)
- `"n<K>"` — top-K anchors by VP count; tiebreak by `anchor_ip` ascending
  (deterministic)

### Constructor knobs ([ripe_atlas.py:43-91](ripe_atlas.py#L43-L91))

| Param | Default | Purpose |
|---|---|---|
| `slice` | `"all_anchors"` | `"all_anchors"` or `"n<K>"` (top-K by VP count) |
| `setup` | `PROBES_TO_ANCHORS` | which side is the VP — see Setups |
| `threshold` | `70` | min-RTT filter for the main query (matches v1) |
| `sanitize` | `True` | re-run paper's SOI removal at load time |
| `sanitize_threshold` | `300` | threshold for the sanitization queries |
| `anchor_mesh_table` | `default.ANCHORS_MESHED_PING_TABLE` | source for sanitization phase 1 |
| `ping_table` | `default.PROBES_TO_ANCHORS_PING_TABLE` | main RTT table |
| `rtt_query` | `None` | test injection seam; production lazy-imports `compute_rtts_per_dst_src` |
| `holdout` | `None` | optional `HoldoutPolicy` (Sechidis) or `DistGeoKFoldPolicy` (greedy-Prim, see [Holdout](#holdout-leakage-free-fit--eval-split)) — when set, anchors are split into K folds; fit/eval iterators emit disjoint subsets |

### Load pipeline

`_ensure_loaded()` runs once on first `iter_*()` call:

1. **`_load_coords()`** ([:196-215](ripe_atlas.py#L196-L215)) — reads
   `reproducibility_probes_and_anchors.json` directly. Builds `{ip: Coord}`,
   `{ip: asn}`, `{ip: country}`, and an `anchor_ips` set from `is_anchor`.
   Skips entries missing `address_v4` or `geometry.coordinates`. Avoids
   `analysis.compute_geo_info` because that helper also pulls a 567MB
   pairwise-distance file we don't need.
2. **`_load_rtts()`** ([:217-254](ripe_atlas.py#L217-L254)) — calls
   `compute_rtts_per_dst_src(ping_table, "", threshold=70)`, returns
   `{dst: {src: [min_rtt]}}`. If sanitize on, calls `_compute_soi_removed_ips`
   and filters both keys and values. Then collapses `[rtts]` → `min(rtts)` and
   drops RTT ≤ 0.
3. **`_apply_slice()`** ([:311-329](ripe_atlas.py#L311-L329)) — for `n<K>`,
   sorts by `(-len(measurements), anchor_ip)` and keeps top K. Deterministic
   tiebreak by IP.

### Internal cache shape

After loading, `self._rtts_by_anchor` is `{anchor_ip: {probe_ip: min_rtt_ms}}`
regardless of setup. The setup flag only controls which side the three
iterators label as the "VP".

### Iterators ([ripe_atlas.py:101-185](ripe_atlas.py#L101-L185))

- **`iter_vp_configs`** — one `VpConfig` per IP active on the VP side
  (probes for `PROBES_TO_ANCHORS`; anchors for the swap).
- **`iter_fit_samples`** — yields `(vp_id, vp_coord, target_coord, rtt)`
  training tuples.
- **`iter_eval_targets`** — for `PROBES_TO_ANCHORS`, iterates
  `_rtts_by_anchor` directly. For `ANCHORS_TO_PROBES`, transposes to
  `{probe_ip: [(anchor_ip, anchor_coord, rtt)...]}` first, then yields one
  `EvalTarget` per probe.

### Loading behavior

- Lazy: `_ensure_loaded()` fires on first `iter_*()` call, then caches in memory.
- RTT load is the ClickHouse hit — cache parquets when iterating repeatedly;
  prefer `VultrCSVSource` for unit tests.

### Filtering invariants

- Drops IPs missing coord or geometry (`_load_coords`).
- Drops RTT ≤ 0 (`_load_rtts` and the iterators).
- Drops IPs flagged by live SOI sanitization (`_load_rtts`, see below).
- Collapses `{dst: {src: [rtts]}}` → single `min(rtt)` per (dst, src) pair
  (`_load_rtts`).

## Sanitization

The source re-runs the IMC 2023 sanitization at load time (constructor flag
`sanitize=True`, default). Replicates `datasets/create_datasets.ipynb` cells 29
& 32 exactly. Implementation: `_compute_soi_removed_ips`
([ripe_atlas.py:256-309](ripe_atlas.py#L256-L309)).

1. **Phase 1** — query `anchors_meshed_pings` (threshold=300), call
   `compute_remove_wrongly_geolocated_probes` to iteratively remove the anchor
   with the most speed-of-Internet (2c/3) violations until none remain.
2. **Phase 2** — same procedure on `ping_10k_to_anchors`, with phase-1 anchors
   already excluded.

Distances are computed on the fly via haversine over the loaded coords
([:289-301](ripe_atlas.py#L289-L301)) — only for (dst, src) pairs that
actually appear in the RTT data. No pre-computed pairwise-distance file needed,
so the sanitizer works on any dataset shape.

If the anchor-mesh query fails (e.g. a test fake that only knows the probe
table), phase 1 silently skips and phase 2 still runs.

Live run on the bundled data: drops **105 IPs** total
(9 anchors + 96 probes), which matches the paper's removal counts and matches
`reproducibility_filtered_probes.json` on 104/105 entries (one greedy-tiebreak
swap between two equivalent IPs in AS16276/OVH). Validation script:
[validate_sanitization.py](validate_sanitization.py).

Pass `sanitize=False` to disable.

## Holdout (leakage-free fit / eval split)

Implemented in [holdout.py](holdout.py). Passing a holdout policy
(`HoldoutPolicy` or `DistGeoKFoldPolicy`) to the constructor partitions the
anchor corpus into K folds; for the configured `fold_index`,
`iter_eval_targets()` emits **only** that fold's anchors, and
`iter_fit_samples()` emits **only** rows whose target anchor is in one of the
other K−1 folds. This eliminates the partial answer-memorization that
otherwise affects every fitted LTD (Octant spline, bounded_spline, NormalDist,
Spotter) when fit_samples and eval_observations are drawn from the same
anchor set.

Both policies are anchor-level K-fold (the leakage unit is the anchor — see
the task's `report.md` for why per-pair K-fold leaks via multi-VP centroid
aggregation). They differ in how anchors are distributed into folds:

| Policy | Algorithm | Strength |
|---|---|---|
| `HoldoutPolicy` | Sechidis-style iterative multi-label stratification on country + ASN-bucket (optional spatial k-means pre-clustering) | best country balance; spatial-block mode available |
| `DistGeoKFoldPolicy` | Per-ASN-bucket greedy-Prim distance ordering + balanced round-robin | explicit intra-fold spatial spread within each ASN bucket |

Dispatch is method-based — `policy.compute_fold_assignments(anchors) → {ip: fold_index}`
and `policy.slice_suffix() → "..."` are the only entry points the source
calls, so adding a third policy doesn't require any change to
`RipeAtlasSource`.

### `HoldoutPolicy` parameters (Sechidis)

| Param | Default | Purpose |
|---|---|---|
| `kind` | `"sechidis_kfold"` | algorithm selector (only this value implemented for HoldoutPolicy) |
| `k` | `5` | number of folds |
| `fold_index` | `0` | which fold (0..k-1) is held out as the eval set |
| `seed` | `42` | determinism seed for tiebreaks and k-means init |
| `labels` | `("country", "asn_bucket")` | balance axes |
| `asn_bucket_top_n` | `20` | top-N ASNs each get their own bucket; rest collapse to `"other_AS"` |
| `spatial_clusters` | `30` | k-means cluster count (3D unit-vector projection); `None` disables spatial blocking |

### `DistGeoKFoldPolicy` parameters (greedy-Prim + ASN bucketing)

| Param | Default | Purpose |
|---|---|---|
| `kind` | `"dist_geo_kfold"` | algorithm selector (only this value implemented) |
| `k` | `5` | number of folds |
| `fold_index` | `0` | which fold (0..k-1) is held out as the eval set |
| `seed` | `42` | seed for the greedy-Prim start-edge tiebreak (passed into `select_vps`) |
| `asn_bucket_top_n` | `20` | same bucketing as HoldoutPolicy; rest → `"other_AS"` |

Algorithm: for each ASN bucket, run `select_vps(strategy="dist_geo")`
(reused from [scripts/vp_selection/strategies.py](../../../vp_selection/strategies.py))
to order the bucket's anchors by greedy-Prim maximum pairwise spread, then
round-robin into K folds with a smallest-fold tiebreak that absorbs
singleton-bucket placements across the corpus. Result: ~1/K of each ASN
bucket per fold, with maximal spatial spread within each fold-bucket slice.

### Effect on `slice_id()`

When `holdout` is set, `slice_id()` returns `"<base_slice>__<policy_suffix>"`.
The suffix format is policy-specific:

- `HoldoutPolicy` (Sechidis) → `fold{i}of{k}_seed{seed}`
- `DistGeoKFoldPolicy` → `distgeo_fold{i}of{k}_seed{seed}`

Each fold materializes into its own directory under
`<inputs_root>/ripe_atlas/<setup>/`:

- `all_anchors/` — paper-faithful (no split)
- `all_anchors__fold0of5_seed42/` — Sechidis fold 0 held out
- `all_anchors__fold1of5_seed42/` — Sechidis fold 1 held out
- `all_anchors__distgeo_fold0of5_seed42/` — DistGeo fold 0 held out
- ... etc.

The runner reads each as an independent slice; no changes to
[inputs.py](../inputs.py) or the materialize CLI are needed to support
cross-validation. To run K=5 CV under one policy, materialize K times with
`fold_index=0..4` and aggregate downstream. To compare two policies, run
both — they produce parallel directory trees.

### Per-setup behavior

| Setup | Holdout applied? |
|---|---|
| `PROBES_TO_ANCHORS` | yes — eval anchors held out from fit samples |
| `ANCHORS_TO_ANCHORS` | yes — same axis (anchors); fit-side VP-reuse of train-fold anchors is intentional and not leakage |
| `ANCHORS_TO_PROBES` | **no** — eval targets are probes (noisy GT, secondary setup). Holdout is stripped at construction with a logged warning; `slice_id()` does not carry the suffix. |

### Spatial blocking vs label balance — a real tradeoff

`spatial_clusters` is intentionally on by default (=30) because the
underlying scientific question is "how well does the model generalize to
*spatially novel* anchors" — and naive label stratification puts adjacent
anchors on opposite sides of the train/test boundary, leaking
spatial-autocorrelation. Spatial atomicity comes at a cost: a few large
spatial clusters (e.g. a dense metro) get assigned atomically to one fold,
which can produce per-country imbalance of up to ~10–15% in the most
clustered countries (US, DE).

If you want maximum label balance instead (e.g. to isolate the
stratification effect from the spatial effect), set `spatial_clusters=None`.
Empirically that gives per-country max-min ≤ 2 across folds on the
752-anchor corpus, at the cost of ignoring spatial autocorrelation.
Reporting both side-by-side is the Roberts et al. recommendation —
their suggested workflow is "estimate the spread between random/stratified
CV and spatial-block CV; the gap quantifies the autocorrelation premium."

## Dataset span

Source: `datasets/reproducibility_datasets/atlas/reproducibility_probes_and_anchors.json`.

| Metric | Probes raw | Anchors raw | Probes eval | Anchors eval |
|---|---:|---:|---:|---:|
| Count | 10,134 | 785 | 9,683 | 752 |
| With `country_code` | 100% | 100% | 100% | 100% |
| With `asn_v4` | 100% | 100% | 100% | 100% |
| **Unique countries** | **171** | **96** | **159** | **96** |
| **Unique ASNs** | **3,173** | **561** | **3,270** | **543** |

"Raw" = JSON entries with v4 IP + geometry (what `_load_coords` keeps).
"Eval" = what the iterators actually yield (post-sanitize ∩ has-measurements in
`ping_10k_to_anchors` ∩ has-coord ∩ rtt > 0).

### Top 10 countries (eval set)

- **Probes**: US 1485, DE 1107, FR 796, GB 523, NL 507, RU 424, CH 291,
  IT 260, CA 247, PT 241
- **Anchors**: US 109, DE 102, NL 43, FR 41, GB 36, CH 28, SG 22, IT 19,
  RU 19, AT 17

### Top 10 ASNs (eval set)

- **Probes**: AS7922 (Comcast) 263, AS3320 (DTAG) 218, AS12322 (Free) 209,
  AS3209 (Vodafone) 202, AS3215 (Orange) 200, AS2860 170, AS7018 149,
  AS47583 130, AS701 121, AS1136 112 — heavy eyeball-ISP skew
- **Anchors**: AS396982 (Google) 18, AS20473 (Choopa/Vultr) 17, AS202422 15,
  AS12008 14, AS48503 12, AS15133 (Edgio) 9, AS14061 7, AS208722 7, AS42473 7,
  AS680 7 — cloud/CDN-heavy

### Overlap (eval set)

| | Shared | Probe-only | Anchor-only |
|---|---:|---:|---:|
| Countries | 96 | 63 | 0 |
| ASNs | 535 | 2,735 | 8 |

~83.6% of probe ASNs have zero anchors.

## Paper-claim cross-check

The IMC 2023 paper states:

> Probes: more than 10K vantage points in **172 countries** and **3,494 ASes**.
>
> Anchors: **723** RIPE Atlas anchors, located in **441 cities**,
> **96 countries**, and **561 ASes** — 133 Asia, 16 Africa, 18 Oceania,
> 125 North America, 399 Europe, 27 South America (sum = 718).

Verified against the bundled data:

| Claim | Paper | Verified | Status |
|---|---:|---:|:---:|
| Anchor count | 723 | 752 (post-sanitize, has-measurements) | +29 |
| Anchor countries | 96 | **96** | ✅ |
| Anchor ASes | 561 | 561 raw / 543 eval | matches **raw** |
| Anchor "cities" | 441 | ~477 (0.1° lat/lon clusters) | approx |
| Anchor continent sum | 718 | 752 | (paper itself doesn't sum to 723) |
| Probe count | >10K | **10,134 raw / 9,683 eval** | ✅ |
| Probe countries | 172 | 171 raw | -1 |
| Probe ASes | 3,494 | 3,173 raw / 3,270 eval | -224 (raw) |

Notes on the gaps:

- **Anchor count 723 vs 752**: `create_datasets.ipynb` cell 37 adds a filter
  requiring ≥100 self-respond rows in `street_lvl_traceroutes`. That table has
  no `resp_addr = dst_addr` rows in the bundled DB, so the filter can't be
  reproduced. The ~29-anchor gap is consistent with this missing step.
- **Anchor ASes 561**: paper number matches our **pre-sanitize** count exactly.
  Paper likely quoted the input anchor set, not the post-filter set.
- **Anchor cities 441**: the `city` field is `null` for every entry in the
  bundled JSON, so this requires external reverse-geocoding. A 0.1° grid
  approximation (~11 km cells) gives 478, same order of magnitude.
- **Probe ASes 3,494 vs 3,173**: ~9% gap, most likely ASN-to-IP mapping drift
  between the paper snapshot and the bundled JSON's static `asn_v4` tags.
- **Eval probe ASN count > raw**: `iter_vp_configs` yields any IP active in
  the ping table that has a coord — it does not exclude `is_anchor=True`
  entries. So when an anchor's IP also responds as a measurement source, its
  ASN is counted under "eval probes". That bumps the eval ASN count
  (3,270 > raw 3,173) without changing the country count materially.

## Takeaway

Probes span ~5.5× more ASNs and ~1.7× more countries than anchors. The anchor
set is a thin, cloud/CDN-biased slice of the probe footprint. That asymmetry is
why the `ANCHORS_TO_PROBES` swap (9,198 targets across 2,978 ASNs) is a
meaningfully harder pressure test than the default `PROBES_TO_ANCHORS` setup
(754 anchor targets, mostly in datacenters).
