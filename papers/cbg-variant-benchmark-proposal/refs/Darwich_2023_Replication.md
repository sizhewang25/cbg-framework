# Darwich et al. (IMC 2023) — Replication: Towards a Publicly Available Internet Scale IP Geolocation Dataset

**Citation:** Darwich, Rimlinger, Dreyfus, Gouel, Vermeulen. IMC 2023. LAAS-CNRS / Sorbonne. Code+data: https://github.com/dioptra-io/geoloc-imc-2023

This is the source paper for the dataset our CBG-variant benchmark is built on. Below is what it actually delivers and the caveats that matter when using it as a benchmark substrate.

## What the paper does

Replicates two influential geolocation papers using only publicly available infrastructure (RIPE Atlas + open services) and re-evaluates their core insights:

1. **Million-scale / Hu et al., IMC 2012** [32]: CBG + a vantage-point (VP) selection algorithm that probes representative IPs in the target's /24 from all VPs, then uses the 10 lowest-RTT VPs to geolocate the target. Original claim: geolocate 35% of routable IPv4 in a few months.
2. **Street-level / Wang et al., NSDI 2011** [46]: Three-tier system (CBG → locally hosted website landmarks → traceroute-based delay refinement). Original claim: 690 m median error.

Scope is IPv4 only; data collected March–May 2023.

## The released dataset

**Targets:** 723 RIPE Atlas anchors (after sanitization, see below). 441 cities, 96 countries, 561 ASes. Geographic skew: 399 EU, 133 AS, 125 NA, 27 SA, 18 OC, 16 AF. AS-type mix (CAIDA): 31.7% content, 29.2% access, 27.2% transit/access, 7.6% enterprise, 0.8% Tier-1, 3.5% unknown.

**Vantage points (million-scale):** ~10K RIPE Atlas probes (after removing 96 mis-geolocated ones). Probe AS mix is access-network dominated (75.2%).

**Vantage points (street-level):** Restricted to the 723 anchors only (probes too rate-limited and too expensive in credits for the per-landmark traceroute fan-out).

**Ground-truth quality:** Anchor coords are RIPE's reported geolocation, sanitized by iteratively removing anchors with speed-of-Internet (SOI = 2/3·c) violations in the anchor-meshed pings (9 anchors removed). Probes sanitized similarly against the cleaned anchors (96 removed). The paper is explicit: "this dataset is best effort... there exists no publicly available ground truth dataset for IP geolocation."

**Released measurement tables (ClickHouse, in `datasets/clickhouse_data/`):**
- `anchors_meshed_pings` — all-anchor-to-all-anchor pings. Drives SOI sanitization and is the cleanest table (good GT on both ends).
- `anchors_meshed_traceroutes` — same pairs, traceroutes.
- `probes_to_prefix_pings` (~1 GB compressed, largest) — all probes pinging /24-prefix representatives selected via ISI hitlist; backbone of the million-scale VP-selection replication.
- `anchors_to_prefix_pings` — anchors pinging /24 representatives.
- `ping_10k_to_anchors` — full 10K-probes → 723-anchors meshed ping; this is the primary CBG eval substrate.
- `targets_to_landmarks_pings`, `street_lvl_traceroutes` — street-level tier-2/3 measurements.

Total compressed measurement data ~2 GB; uncompressed 10–15 GB; billions of pings.

## Replication methodology and deviations from the originals

**Million-scale:** Methodology replicated without change. Targets and VPs are different (RIPE Atlas anchors+probes vs PlanetLab nodes+ping servers). Hilbert-curve IPv4-coverage visualization not replicated — they cannot probe all /24s on RIPE Atlas (overhead).

**Street-level:** Speed-of-internet kept at 4/9·c per the original. Two deliberate deviations:
1. Traceroutes from only the 10 closest VPs per landmark, not all VPs (shown not to lose accuracy; needed to fit RIPE Atlas budget).
2. Reverse-geocoding/landmark discovery switched from Geonames (rate-limited) to a local Nominatim + Overpass API; landmark candidates pulled from "all amenities with a website" rather than only business/university/government keywords. The D1/D2 latency-between-landmark-and-target formula is reconstructed (paper §3.2.2 + Appendix B); the original was under-specified.
ISP-vs-accuracy result not replicated.

## Key replicated results

**Million-scale:**
- Median CBG error with all ~10K VPs: **8 km**, much better than the original's hundreds of km. **73% of targets geolocated at city level (≤40 km)**, 11% at ≤1 km.
- Hypothesis 1 ("few VPs as good as many") is misleading: error keeps decreasing past thousands of VPs, just with diminishing returns. What actually matters is having *some* close VP — removing VPs within 40 km of the target pushes median error from 8 km → 120 km; the fraction ≤40 km drops from 73% → 6%.
- A single well-chosen VP outperforms larger VP sets for sub-40 km errors (62% ≤10 km with the closest VP vs 52% with all).
- Original VP-selection algorithm cannot be deployed on RIPE Atlas at the required pps. Their two-step modification uses **2.88M pings (13.2% of the 21.7M baseline)** at no accuracy cost.

**Street-level:**
- Median error **28 km** for the three-tier technique vs 29 km for plain CBG — i.e., it is *not* street-level on this dataset (original claimed 690 m).
- Only 28% of targets have any locally-hosted landmark within 1 km (19% with latency double-check); 76% within 40 km.
- Pearson correlation between measured (latency-derived) and geographic landmark-to-target distance: median **0.08** — the "order preservation" insight does not hold.
- D1+D2 is negative (unusable) for ≥28% of landmarks for half the targets.
- Median wall-clock to geolocate one target end-to-end: **1238 s (20 min)**, vs the original's claimed 1–2 s.

**Comparisons:** IPinfo (free) beats CBG-all-VPs on the 723-anchor set: **89% ≤40 km** for IPinfo vs 73% for CBG-all-VPs vs 55% for MaxMind free.

## Strengths and limitations as a CBG-variant benchmark substrate

**Strengths**
- Fully public: code, datasets, dump files, and reproducibility instructions all on GitHub.
- Largest publicly-released RIPE Atlas measurement corpus for geolocation (10K VPs × 723 anchors meshed; plus probes-to-prefix and street-level tables).
- Targets are AS-type diverse compared to PlanetLab-era datasets (only 31.7% content, real spread across access/transit/enterprise) — better than prior baselines for evaluating CBG behavior across network types.
- Anchor geolocation has been SOI-sanitized; remaining 723 anchors are the highest-confidence GT subset RIPE Atlas can offer.
- Provides explicit baseline numbers (CBG median 29 km, 73% city-level, 11% sub-km) that a new CBG variant must beat to be a contribution.

**Limitations / caveats**
- **No true ground truth.** "Best effort" GT comes from RIPE-reported anchor coordinates; the authors themselves caveat this. Probes used as VPs are even less reliable — 96 were removed and others may still be mis-located (e.g., the 26 EU targets with >300 km error are suspected to have probes suffering from last-mile delay *or* bad geolocation).
- **Geographic bias:** 55% of targets in Europe, only 16 in Africa and 18 in Oceania. Aggregate numbers are EU-dominated; per-continent CDFs (Fig. 4) show non-trivial variance.
- **AS-type bias on the VP side:** 72.4% of VPs are in access networks, with last-mile delay risk.
- **Anchors are well-connected servers** — they avoid last-mile delay themselves, so geolocation results on this dataset are *optimistic* compared to geolocating arbitrary client IPs in access networks.
- IPv4 only.
- The street-level tables reflect deviations from the Wang et al. methodology (landmark sourcing, 10-VP traceroute fan-out, Nominatim/Overpass) — be careful if comparing landmark-augmented CBG variants directly to the NSDI 2011 numbers.

## Relevance to CBG-variant benchmarking

This dataset *is* the benchmark substrate for our work, and the most useful tables for CBG variants are:

- **`ping_10k_to_anchors`** — the cleanest eval table: best-GT targets, full VP coverage, all-pairs pings. Primary surface for comparing CBG variants on the 723-target hard-GT set.
- **`anchors_meshed_pings`** — smaller, both endpoints are anchors; useful as a "swap roles" pressure test where anchors act as both VPs and targets (12,129-pair hard-GT, per the project's existing benchmark-dataset decision).
- **`probes_to_prefix_pings`** — only needed if a variant explicitly uses /24-representative pre-probing (Hu et al. VP-selection style); otherwise mostly irrelevant to a vanilla CBG comparison.
- **`anchors_to_prefix_pings`** — useful if exploring anchor-as-VP variants.

Reference baselines on the 723-anchor set to beat: median 29 km, 73% ≤40 km, 11% ≤1 km (CBG with all RIPE Atlas VPs). For cross-DB context: MaxMind free 55% ≤40 km, IPinfo free 89% ≤40 km.

**Key caveats when interpreting CBG-variant results on this dataset:**
1. RIPE Atlas probe coordinates are noisy; some "bad" CBG outputs may reflect VP-side error rather than variant weakness.
2. Anchors are atypically well-connected — sub-km errors here likely won't generalize to consumer IPs.
3. Speed-of-internet constant choice (2/3·c vs 4/9·c) materially affects intersection feasibility — the authors had to fall back from 4/9·c to 2/3·c on 5 targets in the street-level replication.
4. The accuracy ceiling is geographic VP coverage, not algorithm: "the closest VPs generally maximize accuracy" is the one original hypothesis that fully survived. Any CBG variant claiming improvement should be evaluated under controlled VP-availability conditions (e.g., the "remove VPs within X km" ablation in Fig. 2c).
