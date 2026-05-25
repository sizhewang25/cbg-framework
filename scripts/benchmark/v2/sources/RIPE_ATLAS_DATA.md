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

### Slices

The slice IS the fold for the anchor-targeted setups (P2A / A2A):

- `"fold_N"` (where N is `0` … `K-1`) — eval set is the Nth fold of the
  stratification at `stratification_path`; fit set is the union of the
  other K-1 folds.

For `ANCHORS_TO_PROBES`, the slice is just a label (fold semantics don't
apply — eval targets are probes). Use any string, e.g. `"all"`.

### Constructor knobs

| Param | Default | Purpose |
|---|---|---|
| `slice` | (required) | `"fold_N"` for P2A / A2A; any string for A2P |
| `setup` | `PROBES_TO_ANCHORS` | which side is the VP — see Setups |
| `stratification_path` | (required for P2A/A2A) | JSON written by `stratify.py`; ignored with warning for A2P |
| `threshold` | `70` | min-RTT filter for the main query (matches v1) |
| `sanitize` | `True` | re-run paper's SOI removal at load time |
| `sanitize_threshold` | `300` | threshold for the sanitization queries |
| `anchor_mesh_table` | `default.ANCHORS_MESHED_PING_TABLE` | source for sanitization phase 1 |
| `ping_table` | `default.PROBES_TO_ANCHORS_PING_TABLE` | main RTT table |
| `rtt_query` | `None` | test injection seam; production lazy-imports `compute_rtts_per_dst_src` |

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
3. **`_apply_holdout()`** — for P2A / A2A: builds a `LoadedStratification`
   from `stratification_path` + the fold index parsed from `slice`,
   intersects its assignments with the active anchor set, and populates
   `_train_anchors` / `_test_anchors`. No-op for A2P.

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

## Stratification (leakage-free fit / eval split)

Two-step workflow:

1. A **stratification** (deterministic anchor-level K-fold) is computed
   once by [`scripts/processing/ripe_atlas/stratify.py`](../../../processing/ripe_atlas/stratify.py)
   and written to `datasets/ripe_atlas/stratifications/<algo>/<tag>.json`.
2. The source consumes the JSON via `LoadedStratification`: when
   `slice="fold_N"`, `iter_eval_targets()` emits **only** that fold's
   anchors and `iter_fit_samples()` emits **only** rows whose target anchor
   is in one of the other K−1 folds. This eliminates the partial
   answer-memorization that otherwise affects every fitted LTD (Octant
   spline, bounded_spline, NormalDist, Spotter) when fit_samples and
   eval_observations are drawn from the same anchor set.

The split is anchor-level (the leakage unit is the anchor — see the task's
`report.md` for why per-pair K-fold leaks via multi-VP centroid aggregation).

### Algorithms

Live in [stratification.py](../../../processing/ripe_atlas/stratification.py).
Used by `stratify.py` to produce the JSON; never instantiated by the source.

| Algorithm class | Behavior | Strength |
|---|---|---|
| `SechidisStratification` | Iterative multi-label stratification on country + ASN-bucket (optional spatial k-means pre-clustering) | best country balance; spatial-block mode available |
| `DistGeoStratification` | Per-ASN-bucket greedy-Prim distance ordering + balanced round-robin | explicit intra-fold spatial spread within each ASN bucket |

To compare algorithms, run `stratify.py --algo sechidis ...` and
`--algo distgeo ...` to produce two JSONs; point `stratification_path` at
whichever you want to evaluate.

### `LoadedStratification` (what the source uses)

| Param | Default | Purpose |
|---|---|---|
| `path` | (required) | path to a stratification JSON written by `stratify.py` |
| `fold_index` | `0` | which fold (0..k-1) is held out as the eval set; `k` is read from the file |

The source builds it internally from `stratification_path` + the fold
parsed from `slice`. Mismatch handling: `compute_fold_assignments` intersects
the loaded assignments with the source's active corpus (post-sanitization,
post-RTT filter). Anchors in the active corpus but missing from the
stratification are dropped from both fit and eval (logged WARNING). Anchors
in the stratification but absent from the active corpus are ignored (logged
WARNING). Raises `ValueError` if the target fold or its complement ends up
empty after intersection.

### Producing a stratification

```
python -m scripts.processing.ripe_atlas.stratify --algo distgeo \
    --k 5 --seed 42 --asn-bucket-top-n 20
# → datasets/ripe_atlas/stratifications/distgeo/k5_seed42_top20.json

python -m scripts.processing.ripe_atlas.stratify --algo sechidis \
    --k 5 --seed 42 --spatial-clusters 30
# → datasets/ripe_atlas/stratifications/sechidis/k5_seed42_spatial30_top20.json
```

The CLI reads the canonical 723-anchor file (`reproducibility_anchors.json`);
see [stratify.py](../../../processing/ripe_atlas/stratify.py) for arguments.
It has no ClickHouse dependency. To use a sanitized corpus, first run
`sanitize_anchors.py` (which queries ClickHouse) and pass the output to
`stratify.py --anchors-file ...`.

### Yaml driving the materialize + run grid

```yaml
source: ripe_atlas
setup: probes_to_anchors
slices: [fold_0, fold_1, fold_2, fold_3, fold_4]

source_kwargs:
  stratification_path: datasets/ripe_atlas/stratifications/distgeo/k5_seed42_top20.json
```

`source_kwargs` is forwarded verbatim through the CLI's `--source-kwargs`
JSON option to the source constructor as `**kwargs`. See
[config/ripe-smoke.yaml](../config/ripe-smoke.yaml) for a working example.

### Effect on `slice_id()`

`slice_id()` returns the slice verbatim — `"fold_0"`, `"fold_1"`, etc. The
stratification file fingerprint (algorithm, k, seed) is **not** encoded in
the directory name; that lives in `stratification_path` and is the user's
responsibility to track across runs. Each fold materializes into its own
directory under `<inputs_root>/ripe_atlas/<setup>/`:

- `fold_0/` — eval set = fold 0 anchors
- `fold_1/` — eval set = fold 1 anchors
- ... etc.

The runner reads each as an independent slice; no changes to
[inputs.py](../inputs.py) or the materialize CLI are needed to support
cross-validation. To run K=5 CV, materialize K times with `slice=fold_0..fold_4`
against the same `stratification_path` and aggregate downstream.

### Per-setup behavior

| Setup | Stratification applied? |
|---|---|
| `PROBES_TO_ANCHORS` | yes — eval anchors held out from fit samples; `slice` must be `fold_N` and `stratification_path` is required |
| `ANCHORS_TO_ANCHORS` | yes — same axis (anchors); fit-side VP-reuse of train-fold anchors is intentional and not leakage |
| `ANCHORS_TO_PROBES` | **no** — eval targets are probes (noisy GT, secondary setup). `stratification_path` is dropped at construction with a logged warning; `slice` is used as an opaque label. |

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
