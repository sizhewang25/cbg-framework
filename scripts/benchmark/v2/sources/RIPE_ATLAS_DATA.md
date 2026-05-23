# RIPE Atlas Source — Data Characteristics

Reference doc for `RipeAtlasSource` ([ripe_atlas.py](ripe_atlas.py)), the primary
eval path in the v2 benchmark. Numbers below are from the IMC 2023
reproducibility dataset bundled with the repo, verified against ClickHouse on
2026-05-22.

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
| `PROBES_TO_ANCHORS` (default) | 9,198 probes | 754 anchors | Hard ground truth on the target side |
| `ANCHORS_TO_PROBES` | 754 anchors | 9,198 probes | Roles swapped; harder pressure test |

(Counts after sanitization and after dropping IPs that don't appear in
`ping_10k_to_anchors`.)

### Slices (`_apply_slice`)

- `"all_anchors"` — every anchor with ≥1 valid (probe, RTT)
- `"n<K>"` — top-K anchors by VP count; tiebreak by `anchor_ip` ascending
  (deterministic)

### Loading behavior

- Lazy: `_ensure_loaded()` fires on first `iter_*()` call, then caches in memory.
- Coords loaded locally from JSON (not via `analysis.compute_geo_info`, which
  also pulls an unrelated pairwise-distance file).
- RTT load is the ClickHouse hit — cache parquets when iterating repeatedly;
  prefer `VultrCSVSource` for unit tests.

### Filtering invariants

- Drops IPs missing coord or geometry.
- Drops RTT ≤ 0.
- Drops IPs flagged by live SOI sanitization (see below).
- Collapses `{dst: {src: [rtts]}}` → single `min(rtt)` per (dst, src) pair.

## Sanitization

The source re-runs the IMC 2023 sanitization at load time (constructor flag
`sanitize=True`, default). Replicates `datasets/create_datasets.ipynb` cells 29
& 32 exactly:

1. **Phase 1** — query `anchors_meshed_pings` (threshold=300), call
   `compute_remove_wrongly_geolocated_probes` to iteratively remove the anchor
   with the most speed-of-Internet (2c/3) violations until none remain.
2. **Phase 2** — same procedure on `ping_10k_to_anchors`, with phase-1 anchors
   already excluded.

Distances are computed on the fly via haversine over the loaded coords, so the
sanitizer works on any dataset shape — no pre-computed pairwise file required.

Live run on the bundled data: drops **105 IPs** total
(9 anchors + 96 probes), which matches the paper's removal counts and matches
`reproducibility_filtered_probes.json` on 104/105 entries (one greedy-tiebreak
swap between two equivalent IPs in AS16276/OVH). Validation script:
[validate_sanitization.py](validate_sanitization.py).

Pass `sanitize=False` to disable.

## Dataset span

Source: `datasets/reproducibility_datasets/atlas/reproducibility_probes_and_anchors.json`
(12,914 total entries; 11,185 with v4 IPs; 1,729 v6-only/no-IP entries dropped).

| Metric | Probes raw | Anchors raw | Probes eval | Anchors eval |
|---|---:|---:|---:|---:|
| With v4 address | 10,400 | 785 | 9,198 | 754 |
| With `country_code` | 100% | 100% | 100% | 100% |
| With `asn_v4` | 100% | 100% | 100% | 100% |
| **Unique countries** | **171** | **96** | **167** | **96** |
| **Unique ASNs** | **3,173** | **561** | **2,978** | **543** |

"Raw" = entries with v4 IPs. "Eval" = post-sanitize ∩ has measurements in
`ping_10k_to_anchors` — what `RipeAtlasSource` actually yields.

### Top 10 countries (eval set)

- **Probes**: US 1425, DE 1034, FR 777, GB 496, NL 486, RU 412, CH 272,
  PT 245, IT 244, CA 237
- **Anchors**: US 109, DE 102, NL 43, FR 42, GB 36, CH 28, SG 22, IT 19,
  RU 19, AT 17

### Top 10 ASNs (eval set)

- **Probes**: AS7922 (Comcast) 270, AS3320 (DTAG) 220, AS12322 (Free) 215,
  AS3215 (Orange) 205, AS3209 (Vodafone) 205, AS2860 176, AS7018 153,
  AS47583 130, AS701 123, AS1136 116 — heavy eyeball-ISP skew
- **Anchors**: AS396982 (Google) 18, AS20473 (Choopa/Vultr) 17, AS202422 15,
  AS12008 14, AS48503 12, AS15133 (Edgio) 9, AS31713 8, AS42473 7, AS680 7,
  AS36236 7 — cloud/CDN-heavy

### Overlap (eval set)

| | Shared | Probe-only | Anchor-only |
|---|---:|---:|---:|
| Countries | 95 | 72 | 1 |
| ASNs | 221 | 2,757 | 322 |

~92.6% of probe ASNs have zero anchors.

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
| Anchor count | 723 | 754 (post-sanitize, has-measurements) | +31 |
| Anchor countries | 96 | **96** | ✅ |
| Anchor ASes | 561 | 561 raw / 543 eval | matches **raw** |
| Anchor "cities" | 441 | ~478 (0.1° lat/lon clusters) | approx |
| Anchor continent sum | 718 | 754 | (paper itself doesn't sum to 723) |
| Probe count | >10K | **10,400 raw / 9,198 eval** | ✅ |
| Probe countries | 172 | 171 raw | -1 |
| Probe ASes | 3,494 | 3,173 raw / 2,978 eval | -321 |

Notes on the gaps:

- **Anchor count 723 vs 754**: `create_datasets.ipynb` cell 37 adds a filter
  requiring ≥100 self-respond rows in `street_lvl_traceroutes`. That table has
  no `resp_addr = dst_addr` rows in the bundled DB, so the filter can't be
  reproduced. The ~31-anchor gap is consistent with this missing step.
- **Anchor ASes 561**: paper number matches our **pre-sanitize** count exactly.
  Paper likely quoted the input anchor set, not the post-filter set.
- **Anchor cities 441**: the `city` field is `null` for every entry in the
  bundled JSON, so this requires external reverse-geocoding. A 0.1° grid
  approximation (~11 km cells) gives 478, same order of magnitude.
- **Probe ASes 3,494 vs 3,173**: ~10% gap, most likely ASN-to-IP mapping drift
  between the paper snapshot and the bundled JSON's static `asn_v4` tags.

## Takeaway

Probes span ~5.5× more ASNs and ~1.7× more countries than anchors. The anchor
set is a thin, cloud/CDN-biased slice of the probe footprint. That asymmetry is
why the `ANCHORS_TO_PROBES` swap (9,198 targets across 2,978 ASNs) is a
meaningfully harder pressure test than the default `PROBES_TO_ANCHORS` setup
(754 anchor targets, mostly in datacenters).
