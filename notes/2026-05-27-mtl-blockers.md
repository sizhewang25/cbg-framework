# MTL blockers — probes that cause `EMPTY_REGION` / `NO_INTERSECTION` failures

**Date:** 2026-05-27
**Inputs:** all FALLBACK targets across 6 ASN configs (`europe_as3209`, `europe_as3215`, `global_as16509`, `global_as31898`, `north_america_as7018`, `north_america_as7922`), 5 folds each, for the unweighted intersection combos (`vanilla_cbg`, `vanilla_cbg_planar_centroid`, `million_scale_cbg`, `million_scale_cbg_planar_centroid`, `octant_cbg`, `octant_hull_cbg`, `spotter_cbg`).
**Reproducible source:** [scripts/analysis/inspect_mtl_blockers.py](../scripts/analysis/inspect_mtl_blockers.py), wired into the analysis Snakefile. Per-config artifacts at `scripts/analysis/outputs/<run_id>/ripe_atlas_asn_corpora/probes_to_anchors/merged/inspect_mtl_blockers.{json,md}`. This note captures the first ad-hoc cross-ASN scan that motivated the script.

For every FALLBACK target with error `EMPTY_REGION` or `NO_INTERSECTION`, each VP's per-target LTD constraint was re-evaluated against the eval anchor's known coord. A VP is a "blocker" on target T iff its predicted `[inner, outer]` band does *not* contain the true VP→target distance — a necessary condition for that VP to have been the empty-region culprit.

Weighted MTLs are out of scope: they tolerate disagreement by construction.

---

## Hall of fame — single probes causing the most failures

Cross-combo aggregate: number of (combo, target) pairs where the probe was a non-bracketing constraint.

| Rank | Probe IP | Total target-failures | AS / Run |
|---:|:---|---:|:---|
| 1 | **132.145.108.63** | **1109** | Oracle Cloud — `global_as31898` |
| 2 | 13.236.249.185 | 483 | AWS — `global_as16509` |
| 3 | 16.50.94.65 | 443 | AWS — `global_as16509` |
| 4 | 35.180.40.172 | 405 | AWS — `global_as16509` |
| 5 | 152.67.105.67 | 401 | Oracle — `global_as31898` |
| 6 | 140.238.207.82 | 399 | Oracle — `global_as31898` |
| 7 | 18.198.7.141 | 397 | AWS — `global_as16509` |
| 8 | 18.170.54.122 | 390 | AWS — `global_as16509` |
| 9 | 16.62.90.90 | 381 | AWS — `global_as16509` |
| 10 | 15.161.57.223 | 372 | AWS — `global_as16509` |

A single Oracle Cloud probe `132.145.108.63` causes more downstream failures than the next 5 worst probes combined. This is overwhelmingly an AS31898 / AS16509 (cloud-fleet) phenomenon — cloud anchors with route asymmetry or NAT-induced RTT inflation appear to break CBG constraints regardless of LTD.

---

## Worst probe in each ASN run

Per-combo top blocker for each ASN; "X / Y" = blocks X of Y FALLBACK targets in that combo.

| Run | million_scale_cbg | vanilla_cbg | octant_cbg |
|---|---|---|---|
| europe_as3209 | — (no FALLBACK) | 109.90.40.189 (3/31) | **88.152.186.207** (31/58, 53%) |
| europe_as3215 | — (no FALLBACK) | **86.245.44.36** (12/20, 60%) | 86.246.184.50 (28/37, 76%) |
| global_as16509 | — (no FALLBACK) | 44.233.178.121 (2/9) | 3.99.186.207 (54/217, 25%) |
| global_as31898 | **132.145.108.63** (106/121, 88%) | **132.145.108.63** (107/127, 84%) | **132.145.108.63** (126/272, 46%) |
| north_america_as7018 | **107.203.252.210** (14/15, 93%) | **107.203.252.210** (17/28, 61%) | **107.203.252.210** (60/88, 68%) |
| north_america_as7922 | **24.18.184.15** (12/17, 71%) | **24.18.184.15** (18/39, 46%) | **24.18.184.15** (62/97, 64%) |

Four ASN runs have a single dominant blocker that survives across multiple LTD families — strong evidence the probe itself (location metadata or routing path) is the problem, not a particular model's calibration.

The user-reported `107.203.252.210` in AS7018 is confirmed: it blocks 14/15 (93%) of million_scale_cbg fallbacks and 14/14 (100%) of the planar-centroid variant. Visualization screenshot showed the disk of this VP not containing the truth.

---

## Reading the numbers

- **High percentage + repeats across combos** → the probe's stored location is likely wrong, or routing inflation makes its RTTs unrepresentative of geographic distance. Worth a closer look at MaxMind / RIPE Atlas metadata for these IPs.
- **High count in one combo only** → likely an LTD-calibration artifact, not a metadata issue. e.g., the `spotter_cbg` blocker list is much longer than other combos because the pooled-normal `[μ ± kσ]` band is narrower; probes that are fine for `bounded_spline` are blockers for Spotter.
- The "blocker" metric is necessary-but-not-sufficient: a VP not bracketing the truth on a FALLBACK target *could* still not be the proximate cause (other VPs may also have failed). To strictly confirm, do a leave-one-out replay of the MTL — left for follow-up.

## Per-ASN top-5 blocker lists

Detail in JSON; abbreviated below for `octant_cbg` (representative for cross-LTD agreement).

```
europe_as3209  (FALLBACK=58)
  88.152.186.207     31 (53%)
  130.180.35.89      30 (52%)
  5.147.69.76        29 (50%)
  37.201.95.176      29 (50%)
  95.223.227.167     29 (50%)

europe_as3215  (FALLBACK=37)
  86.246.184.50      28 (76%)
  90.63.249.134      28 (76%)
  81.51.191.194      25 (68%)
  92.169.150.221     24 (65%)
  83.114.116.175     23 (62%)

global_as16509 (FALLBACK=217)
  3.99.186.207       54 (25%)
  3.15.133.134       51 (24%)
  204.236.148.197    50 (23%)
  44.233.178.121     49 (23%)
  177.71.185.54      44 (20%)

global_as31898 (FALLBACK=272)
  132.145.108.63    126 (46%)
  129.146.131.8      63 (23%)
  129.146.130.16     63 (23%)
  158.101.13.142     61 (22%)
  132.145.165.186    59 (22%)

north_america_as7018 (FALLBACK=88)
  107.203.252.210    60 (68%)
  104.179.46.88      46 (52%)
  99.22.6.116        43 (49%)
  217.28.163.28      41 (47%)
  172.127.119.32     40 (45%)

north_america_as7922 (FALLBACK=97)
  24.18.184.15       62 (64%)
  76.142.123.119     49 (51%)
  73.42.144.51       48 (49%)
  69.255.45.153      46 (47%)
  73.251.198.1       46 (47%)
```

---

## Suggested follow-up

1. Validate the top per-ASN suspects against RIPE Atlas probe metadata (declared lat/lon vs. observed RTT minima to nearby anchors).
2. For confirmed mislocated probes, decide whether to remove them upstream (probes filter) or down-weight them in the LTD fit.
3. Repeat with leave-one-out replay to confirm the blockers are actually *causally* responsible (not just non-bracketing).
