# ripe-smoke-01 findings + what needs to change for a paper-grade benchmark

First end-to-end K=5 benchmark of the v2 CBG variants on the RIPE Atlas
723-anchor corpus, run under the leakage-free stratification protocol shipped
in commit [`a1f00aa`](https://github.com/dioptra-io/geoloc-imc-2023). The
numbers and the methodology gaps they expose are captured here.

Config: [scripts/benchmark/v2/config/ripe-smoke.yaml](../scripts/benchmark/v2/config/ripe-smoke.yaml).
Outputs: `scripts/benchmark/v2/outputs/ripe-smoke-01/ripe_atlas/probes_to_anchors/{fold_0..fold_4, all_folds}/`.

---

## Part 1 — The numbers (all_folds = K=5 CV, n=718)

| combo | LTD | MTL | n_success | n_fallback | err_p50 km | err_p95 km |
|---|---|---|---|---|---|---|
| `vanilla_cbg` | low_envelope (half-disk) | spherical_circle | 135 | 583 | 5.59 | 370.66 |
| `million_scale_cbg` | speed_of_internet (⅔c disk) | spherical_circle | 592 | 126 | 6.46 | 352.74 |
| `octant_cbg` | bounded_spline (±90% annulus) | planar_annulus | 5 | 713 | 4.46 | 245.02 |
| `spotter_cbg` | normal_dist (pooled μ±σ annulus) | planar_annulus | 4 | 714 | 4.46 | 258.03 |
| **baseline `shortest_ping`** | — | — | 718 | — | **1** (single-digit) | **16** |

The shortest_ping CDF (dashed line in
[plot_error_cdf.py](../scripts/analysis/plot_error_cdf.py#L248-L271)) **beats
every method on this corpus**. That is the methodological alarm bell.

Per-fold breakdown of `status_counts` (consistent across folds; no fold is an
outlier):

| combo | per-fold SUCCESS range (n=144 each) |
|---|---|
| vanilla_cbg | 24–29 |
| million_scale_cbg | 110–124 |
| octant_cbg | 0–2 |
| spotter_cbg | 0–2 |

---

## Part 2 — Three diagnoses

### 2a. shortest_ping wins because the corpus is co-located, not because of any method choice

Eval corpus stats on fold_0 (144 anchors, 517,788 observations):

| Metric | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|
| **nearest-VP-to-anchor distance (km)** | 0.16 | 0.44 | **3.9** | 19.93 | 329.22 | 2527 |
| **min-RTT to that VP (ms)** | 0.29 | — | **0.73** | — | 8.36 | — |
| VPs observing each anchor | 32 | 345 | **5,289** | 6,062 | 6,173 | 6,210 |

12,129 probes vs 723 anchors that share the same population — RIPE Atlas
literally co-locates probes and anchors in many datacenters. Half of all
anchors have a sub-millisecond RTT to *some* probe; that probe sits ≤4 km
away. There is no method that can beat "the nearest VP" when the nearest VP
is in the same building.

`shortest_ping` is not a CBG variant. It's
`df.groupby("target_id").latency_ms.idxmin()` post-hoc on the eval
observations. With this density it IS the geolocation.

### 2b. The annulus methods fail because there is no per-target VP selection

The v2 framework currently has **no top-K VP-pruning step** between
`iter_eval_targets` and `model.geolocate`. Each target with median **5,289**
observations sees every single one piped into
`ltd.predict_all(obs) → mtl.multilaterate(ok)`.

Original literature ([scripts/analysis/analysis.py:35-70](../scripts/analysis/analysis.py#L35-L70)
in this very repo) prunes to `n_shortest=10` at predict time. The IMC 2012,
Octant, and Laki/Spotter papers all worked with ≤40 VPs per target. The
intersection topology under 5K VPs is brutal:

| combo | LTD shape | per-VP tightness | success/fold | why |
|---|---|---|---|---|
| million_scale_cbg | disk, radius=(RTT/2)·⅔c | very loose (5 ms ≈ 500 km) | 110–124/144 | huge disks almost always overlap |
| vanilla_cbg | half-disk via low_envelope linear fit | upper-only, tighter than ⅔c | 24–29/144 | only ~17% survive ∩ of 5K disks |
| octant_cbg | annulus, per-VP, ±~10% band | tight | 0–2/144 | one bad annulus among 5K → ∅ |
| spotter_cbg | annulus, pooled μ(rtt)±σ(rtt) | tight (well-fit σ on 2.1M pairs is small) | 0–2/144 | same — fail-fast on first bad band |

`enable_circle_filter` (the IMC 2012 redundancy trick) drops dominated
circles. It cannot rescue a single literally-empty annulus that one of 5,000
VPs forced.

### 2c. Spotter at 4 successes total is *not* a pooling-vs-per-VP issue

The instinct "pooled normal_dist should be looser" is backwards in this
corpus. With 2.1M (RTT, distance) pairs across 5K VPs, the global σ(rtt) is
*small* — exactly because pooling drains the noise. From
[scripts/framework/v2/ltd/normal_dist.py:6-7](../scripts/framework/v2/ltd/normal_dist.py#L6-L7):

```
lower_km = max(0, μ(rtt) − σ(rtt))
upper_km = min(max(0, μ(rtt) + σ(rtt)), rtt / THEORETICAL_SLOPE)
```

Per-VP that's a narrow band; intersect 5,000 such bands and you get an empty
feasible region for almost every anchor. 714/718 fallbacks means the Spotter
CDF is visually identical to shortest_ping — fallback in
[model.py:153-162](../scripts/framework/v2/model.py#L153-L162) returns the
nearest-VP coordinate, which IS shortest_ping.

The 4 successes are anchors where the highest-RTT VPs happened to widen the
band enough to admit a planar intersection. No methodology signal.

---

## Part 3 — Why this is unpublishable as-is

A paper claiming `vanilla_cbg p50 = 5.59 km` while shortest_ping shows
p50 ≈ 1 km is mostly reporting **probe density on RIPE Atlas**, not method
quality. Three independent problems with the current setup:

1. **Unrealistic VP fleet.** No real deployment has 10K geographically-spread
   probes. ISPs have 20–300 internal nodes; CDNs have 200–1000 edge POPs;
   academic measurement platforms have ~100. The 10K corpus is a *candidate
   pool*, not a *deployment*.
2. **No per-query VP selection.** The literature's standard `n_shortest`
   prune is missing from the framework. Annulus methods cannot survive this.
3. **Anchor-only stratification.** The K-fold splits *anchors* across folds;
   VPs are not held out. Eval anchor X retains access to VPs in its own /24,
   its own ASN, and its own metro — all of which trivialize the prediction.

---

## Part 4 — What a paper-grade benchmark looks like

### 4a. Make VP fleet size a 2-D experimental variable

```
deployment scale  ×  per-query scale
n_deploy_vps ∈ {20, 50, 100, 200, 500, 1000}   ×   k_per_target ∈ {5, 10, 20, 40, all}
```

With deployment-time VP selection respecting realistic operator topologies:

| Corpus scenario | What it represents |
|---|---|
| single-ASN | one ISP geolocating hosts in other networks |
| multi-ASN federation (N=5, 10, 20) | RIPE/CAIDA-style consortium |
| anchor-as-VP (the 723 anchors themselves) | production-grade well-geolocated fleet |
| random RIPE Atlas downsample to N | "I bought N credits on a public platform" |

And per-query top-K by RTT (port `compute_closest_rtt_probes` from
[scripts/analysis/analysis.py:35-70](../scripts/analysis/analysis.py#L35-L70)
into the v2 pipeline as a stage between `iter_eval_targets` and
`model.geolocate`).

This separates **deployment scale** from **method scale** and makes the
variants actually comparable. With k=10 the annulus methods will recover
their literature accuracy; with k=all (today's setting) they cannot.

### 4b. Block VPs at the ASN level too, not just stratify anchors spatially

Today's stratification (Sechidis or DistGeo) splits **anchors** across folds.
VPs are global. The right realistic split:

```
fold k: choose held-out ASN set A_k
  eval anchors = anchors whose ASN ∈ A_k
  excluded VPs = VPs whose ASN ∈ A_k          ← missing today
  deployment VPs = VPs whose ASN ∉ A_k
  fit pairs = (VP, anchor) with both ASNs ∉ A_k
```

This maps onto "operator with VPs in some ASNs wants to geolocate hosts in
*other* ASNs." It's the institutional-exclusion idea — which lesson
2026-05-23 dismissed as PlanetLab-specific — but at the **ASN level**, which
is the right axis for the production framing. Building-level (PlanetLab) is
not the same as ASN-level (here).

Combined with the spatial blocking already in
`SechidisStratification(spatial_clusters=30)`, you get a 2×2 leakage matrix
worth reporting:

| Split | Trivial-VP leakage | Geographic autocorrelation leakage |
|---|---|---|
| anchor-only (current) | **yes** | yes |
| anchor + spatial block | yes | partially mitigated |
| anchor + ASN block | mitigated | yes |
| anchor + ASN + spatial | mitigated | mitigated — paper-defensible |

### 4c. The leakage proof (runnable on existing outputs)

For each eval anchor *X* in each fold:

- `d_nearest_VP(X)` = haversine to the nearest non-held-out VP
- `err_method(X)` = method's reported error on *X*

Plot `err_method(X)` vs `d_nearest_VP(X)`. The smoking gun:

- low `d_nearest_VP` → low err regardless of method ⇒ scoring co-location
- slope ≈ 1 in the small-`d_nearest_VP` regime ⇒ eval bottlenecked by VP topology

The summary stats above are already half the proof: p50 nearest-VP-to-anchor
= 3.9 km, while `vanilla_cbg` reports p50 = 5.59 km. The two numbers are
within the noise of "method ≈ VP proximity."

---

## Part 5 — Concrete follow-up task shape

One bundled task with three deliverables:

1. **VP-fleet design module** — `scripts/processing/ripe_atlas/vp_corpus.py`
   + CLI. Emits a `vp_corpus.json` per scenario (single-ASN, multi-ASN,
   anchor-as-VP, plus N-VP downsamples). `RipeAtlasSource` consumes it
   analogously to `stratification.json`.
2. **Predict-time top-K VP selection** — new stage between
   `iter_eval_targets` and `model.geolocate` in
   [scripts/benchmark/v2/runner.py](../scripts/benchmark/v2/runner.py), with
   `k_per_target` as a yaml knob (or `source_kwargs:`).
3. **ASN-blocked stratification** — `AsnBlockedStratification` in
   [scripts/processing/ripe_atlas/stratification.py](../scripts/processing/ripe_atlas/stratification.py),
   composable with `spatial_clusters` blocking. Loads via the existing
   `LoadedStratification` path.

Plus a one-off leakage-quantification notebook on the existing 5-fold outputs
to publish the `err_method` vs `d_nearest_VP` scatter as Figure 1 of the
methodology section.

---

**Status:** findings captured 2026-05-25, no implementation kicked off.
Followup task pending user direction.
