# RIPE Atlas Source — Data Characteristics

Reference doc for `RipeAtlasSource` ([ripe_atlas.py](ripe_atlas.py)), the primary
eval path in the v2 benchmark. Numbers below are from the IMC 2023
reproducibility dataset bundled with the repo.

## Source class

- **name**: `"ripe_atlas"`
- **Dataset**: IMC 2023 reproducibility (`reproducibility_probes.json` +
  `reproducibility_anchors.json`)
- **RTTs**: ClickHouse table `ping_10k_to_anchors` via
  `scripts.analysis.analysis.compute_rtts_per_dst_src` (threshold=70, matches
  v1/`million_scale.py`)

### Setups (`DataSource.ALLOWED_SETUPS`)

| Setup | VPs | Eval targets | Notes |
|---|---|---|---|
| `PROBES_TO_ANCHORS` (default) | ~12K probes | 723 anchors | Hard ground truth on the target side |
| `ANCHORS_TO_PROBES` | 785 anchors | 12,129 probes | Roles swapped; harder pressure test |

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
- Collapses `{dst: {src: [rtts]}}` → single `min(rtt)` per (dst, src) pair.

## Dataset span

Source: `datasets/reproducibility_datasets/atlas/reproducibility_probes_and_anchors.json`

| Metric | Probes (VPs) | Anchors (targets) |
|---|---:|---:|
| Total | 12,129 | 785 |
| With country | 12,129 (100%) | 785 (100%) |
| With ASN (v4) | 11,975 (98.7%) | 785 (100%) |
| **Unique countries** | **174** | **96** |
| **Unique ASNs** | **3,463** | **561** |

### Top 10 countries

- **Probes**: US 1744, DE 1692, FR 1005, GB 617, NL 594, RU 533, IT 344,
  CH 340, CZ 311, CA 290
- **Anchors**: US 115, DE 104, FR 45, NL 44, GB 37, CH 28, SG 23, RU 21,
  IT 20, AT 18

### Top 10 ASNs

- **Probes**: AS3320 (DTAG) 386, AS7922 (Comcast) 337, AS12322 (Free) 276,
  AS3215 (Orange) 271, AS3209 (Vodafone) 268, AS2860 181, AS7018 180,
  *None* 154, AS701 150, AS1136 139 — heavy eyeball-ISP skew
- **Anchors**: AS396982 (Google) 20, AS20473 (Choopa/Vultr) 18, AS202422 15,
  AS12008 14, AS48503 12, AS15133 (Edgio) 9, AS36236 9, AS16276 (OVH) 9,
  AS31713 8, AS14061 (DigitalOcean) 8 — cloud/CDN-heavy

### Overlap

| | Shared | Probe-only | Anchor-only |
|---|---:|---:|---:|
| Countries | 95 | 79 | 1 |
| ASNs | 253 | 3,210 | 308 |

~93% of probe ASNs have zero anchors.

## Takeaway

Probes span ~5.5× more ASNs and ~1.8× more countries than anchors. The anchor
set is a thin, cloud/CDN-biased slice of the probe footprint. That asymmetry is
why the `ANCHORS_TO_PROBES` swap (12,129 targets across 3,463 ASNs) is a
meaningfully harder pressure test than the default `PROBES_TO_ANCHORS` setup
(723 anchor targets, mostly in datacenters).
