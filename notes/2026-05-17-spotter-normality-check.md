# Spotter normality check on ping_10k_to_anchors

**Date:** 2026-05-17
**Source paper:** Laki et al., *Spotter: A Model Based Active Geolocation Service*, 2011.
**Script:** [scripts/libs/cbg_feasibility/spotter_normality_check.py](../scripts/libs/cbg_feasibility/spotter_normality_check.py)
**Figures:** [scripts/libs/cbg_feasibility/outputs/spotter_normality/](../scripts/libs/cbg_feasibility/outputs/spotter_normality/)

## What Spotter claims

Spotter's central methodological move (Section IV.A–IV.B, Fig. 3) is that the
delay-distance distribution `f_d(s)` — distance given a fixed RTT bin — is

1. **Normal** when standardized, with `μ ≈ 0`, `σ ≈ 1`.
2. **Landmark-independent**, so a single pooled fit (one `μ(d)`, one `σ(d)`)
   describes every landmark's measurements.

On PlanetLab (~40 000 (delay, distance) pairs across ~100 nodes) they reported
`μ = −0.078, σ = 1.035`, and verified landmark-independence with a Q-Q plot
showing per-landmark quantiles lying on the diagonal.

## Setup on our dataset

- **Table:** `geolocation_replication.ping_10k_to_anchors` (8.3M rows).
- **Aggregation:** `min(min RTT)` per `(src, dst)` pair, RTT filtered to
  `0 < min < 200 ms`.
- **Geo lookup:** [reproducibility_probes_and_anchors.json](../datasets/reproducibility_datasets/atlas/reproducibility_probes_and_anchors.json)
  (10 919 IPs with coordinates).
- **Distance:** vectorized haversine (R = 6 367 km) matching
  [scripts/utils/helpers.py](../scripts/utils/helpers.py).
- **Fit:** binned mean / std on 40 RTT bins, polynomial fits — deg 3 for
  `μ(d)`, deg 2 for `σ(d)`.

| | Spotter (PlanetLab) | Ours (ping_10k_to_anchors) |
|---|---|---|
| Pairs | ~40 k | 5 855 219 |
| Landmarks / anchors | ~100 | 783 |
| Pooled standardized μ | −0.078 | **+0.027** |
| Pooled standardized σ | 1.035 | **0.894** |

## Methodology notes

### Why we bin

Spotter's model is `f_d(s) = N(μ(d), σ(d)²)` — at each fixed delay `d`, the
distribution of distances is normal with delay-dependent mean *and* spread.
Fitting it means estimating two functions of `d`. Estimating `μ(d)` as a
smooth function of a covariate is standard regression; estimating `σ(d)` is
the awkward part — heteroskedastic regression that needs either joint MLE
under a parametric form, or a moment-summary step.

We take the moment-summary route:

1. Slice the RTT range into 40 bins, drop bins with < 30 points.
2. Within each bin compute empirical `mean` and `std` of distance.
3. Fit a degree-3 polynomial to the binned means and degree-2 to the binned
   stds.

This **decouples mean and variance estimation** — within a bin both are just
summary statistics of a sample (essentially noise-free at our scale, ~10⁵
points/bin) — and lets you visually sanity-check the polynomial against the
bin dots in panel (a). Without binning you'd jump straight to joint MLE,
which requires committing to functional forms up front and can diverge if
the `σ(d)` polynomial dips negative.

### Differences from Spotter

Spotter (Sec IV.A) says only *"we fitted the μ(d) and σ(d) polynomials to
the data set"* — they don't describe binning explicitly, but Fig. 3a clearly
shows polynomial curves of mean and std as functions of delay, which is
exactly what the two-step binning approach produces. **Methodologically we
are doing the same thing.**

| Aspect | Spotter (2011) | Ours |
|---|---|---|
| Calibration source | ~100 PlanetLab nodes | ~800 anchors / ~12 k probes (Atlas) |
| Points fit | ~40 000 | 5.85M (probes→anchors), 465k (meshed) |
| RTT range plotted | 0–80 ms | 0–200 ms |
| `μ(d)` shape | Near-linear (intra-continental) | Linear ≤ 125 ms, **saturates** near ~10 000 km |
| Polynomial degrees | Not specified | deg 3 for μ, deg 2 for σ (by inspection) |
| Bin count | Not specified | 40 |
| Pre-aggregation | "minimal RTT for each landmark" | identical: `arrayMin(groupArray(min))` per `(src, dst)` |
| Standardization | `z = (s − μ(d)) / σ(d)` | identical |
| Q-Q anchor selection | "five selected landmarks…representative" (editorial) | top-5 by sample count (auto, reproducible) |

Two consequential differences:

1. **RTT range and saturation.** Spotter never modeled distances above
   ~5000 km. We do, and curvature there forces a higher-degree `μ(d)`. A
   linear `μ(d)` (the cleanest "speed-of-internet ⇒ distance" model) would
   systematically over-estimate distance at high RTT in our data — a real
   modeling gap their paper sidestepped by sticking to PlanetLab geography.
2. **Anchor selection for Q-Q.** Their hand-picked "representative
   landmarks" could be cherry-picked to agree with the diagonal. We pick
   top-N by sample count, which is reproducible but not adversarial — could
   be tightened by raising `--n-anchors` to 50 or random-sampling.

## Findings

### Panel (a) — delay-distance scatter
Qualitatively matches Spotter: `μ(d)` rises near-linearly until ~125 ms then
saturates as great-circle distance approaches ~20 000 km. `σ(d)` widens with
delay. No surprises.

### Panel (b) — standardized histogram, partial agreement
- Mean is essentially zero (`+0.027`), so the pooled fit is centered.
- σ is **0.894**, ~12 % narrower than a true standard normal. The distribution
  is visibly **leptokurtic**: taller and narrower than N(0, 1).
- Likely cause: a heavy upper tail (long indirect / submarine paths) inflates
  the binned `σ(d)` used for standardization, while the bulk clusters tighter
  than that inflated σ. So when normalized by σ(d), the central mass collapses
  inside |z| < 1.

### Panel (c) — Q-Q plot, **landmark-independence fails**
This is the load-bearing claim. On our data it does **not** hold:

- All five top-volume anchors trace clear **S-shapes off the diagonal**.
- Slopes through the center are < 1 → per-anchor σ is smaller than the pooled
  σ for each of them.
- Curves are **horizontally offset from each other** → each anchor has its own
  characteristic mean (e.g. `173.248.145.27` runs to the right of the bunch;
  `91.132.8.99` is closest to the diagonal).
- Tails diverge in different directions per anchor.

A single `μ(d), σ(d)` therefore cannot describe all anchors uniformly on this
dataset.

## Why the assumption breaks here: PlanetLab → Atlas network shift

The shift from 2010 PlanetLab to modern RIPE Atlas probes isn't cosmetic —
it changes what the variable `d` (RTT) actually measures, and Spotter's
model has no slack for that.

### Mechanism: last-mile offset as a confounding term

Spotter's model implicitly assumes `d` is a proxy for great-circle distance
plus a small *universal* additive overhead. That assumption is what licenses
a single landmark-independent `f_d(s)`.

On PlanetLab (2010) the assumption was approximately true:
- All endpoints sat behind university LAN → NREN (Internet2 / GÉANT) →
  IXP → NREN. Last-mile delay was a few-ms constant, near-identical across
  nodes.
- Routing through NRENs was direct, low-jitter, well-peered.
- So `d ≈ k·s + ε` with `ε` small and roughly the same distribution
  everywhere → standardize once, fit one normal, done.

On modern RIPE Atlas probes the RTT decomposes very differently:

```
RTT = propagation(distance) + last_mile(probe) + transit(probe → target)
```

The `last_mile(probe)` term is the troublemaker:
- Fiber/cable: ~5–10 ms baseline
- DSL with interleaving: ~20–40 ms
- 4G/5G mobile: ~30–50 ms with high jitter
- Starlink / GEO satellite: distinctive bimodal patterns
- Rural copper: 100 ms+

And critically: **`last_mile(probe)` is essentially uncorrelated with
distance.** A 30 ms DSL probe pinging a target 100 km away gets the same
access overhead as one pinging across an ocean. So when you condition on
`d`, the "distance given delay" distribution becomes a *mixture* over the
access-technology population that produced that `d` bin — not a clean
propagation-only distribution.

### Panel-by-panel mapping

Each of the three failures follows mechanically from this mixture structure,
not from any vague "noisier data" argument:

1. **σ_z = 0.894 < 1 in panel (b).** Pooled `σ(d)` is the std of a mixture
   of last-mile populations; the mixture variance is larger than any
   sub-population's variance. Standardize sub-populations by the pooled
   over-wide σ and the bulk compresses inside `|z| < 1`, giving a
   leptokurtic histogram. **σ_z < 1 isn't a failure of normality per se —
   it's a signature of unmodeled heterogeneity.**

2. **Horizontal anchor offsets in panel (c).** Each anchor's per-bin
   distance distribution depends on *which probes* land in that bin. An
   anchor in a fiber-rich region (Western Europe, NE US) sees its `d` bins
   dominated by low-last-mile probes → distance-given-delay shifts higher.
   An anchor served by DSL-heavy probe populations shifts the other way.
   This is exactly the per-anchor mean offset observed (`173.248.145.27`
   rightward, `91.132.8.99` near diagonal).

3. **Per-anchor S-shape with slope < 1 in panel (c).** An anchor whose
   probe population is *homogeneous* (e.g., served mostly by one large
   fiber ISP) has narrower own-`σ(d)` than the pooled mixture σ.
   Standardize by the wider pooled σ → quantiles compress toward zero →
   shallow Q-Q slope. Mixed-probe-population anchors compress less.

### Anchors-meshed is the controlled experiment

The `anchors_meshed_pings` run isolates exactly this hypothesis. Anchors
are hosted at IXPs, ISP cores, datacenters — uniformly well-connected with
negligible `last_mile(anchor)` variance. Both endpoints belong to the
"PlanetLab-like" tier.

When restricted to that subset:
- σ_z: **0.894 → 0.964** (mostly closes the gap to Spotter's 1.035).
- Per-anchor Q-Q curves collapse onto the diagonal.

If the failure had been driven by geographic spread, routing indirection,
or sample size, anchors-meshed would have failed too — it has the same
geography, the same routing, and 465k pairs (still 10× Spotter). It didn't.
The only variable that changes between the two runs is access-network
heterogeneity on the probe side.

### Alternatives ruled out as primary causes

- **Geographic coverage** (Atlas reaches APAC/Africa, PlanetLab didn't).
  Would also affect anchors-meshed, but that data recovers Spotter's
  result. Not the dominant driver.
- **Sample size.** Larger samples make deviations easier to detect, but the
  Q-Q signature is qualitatively S-shaped with horizontal offsets — a
  model-mismatch signature, not a finite-sample one.
- **Backbone improvements 2010 → 2026.** Modern backbones are *more*
  uniform and lower-latency than 2010, which should help Spotter's model,
  not hurt it.

### Deeper takeaway

Spotter's "landmark-independent normal" was never really a claim about
Internet physics — it was a claim about the **uniformity of their
measurement infrastructure**. The model worked on PlanetLab because the
calibration set was implicitly preselected to one connectivity class.
Push it into a representative cross-section of real Internet endpoints
(what Atlas captures), and the calibration assumption is the first thing
that breaks. **Any approach that pools per-VP behavior into a single
model is implicitly assuming the same uniformity Spotter did.** That's
the caution the CBG-variant benchmark should inherit.

## Implications for the CBG-variant benchmark

1. A **pooled normal model is a reasonable rough summary** (μ ≈ 0, σ ≈ 0.9) —
   but it understates per-anchor variance heterogeneity.
2. **Per-landmark calibration likely matters more on Atlas than Spotter argued
   for PlanetLab.** Octant / CBG-style per-VP fits could outperform a single
   pooled fit on this dataset.
3. If we adopt Spotter's model in the benchmark, we should **flag the
   assumption violation** rather than inherit the original paper's confidence
   in landmark-independence.
4. Worth re-running with a stricter RTT cap (e.g. 80 ms to match Spotter's
   plotted range) and on `anchors_meshed_pings` (anchors → anchors only, more
   like Spotter's homogeneous setup) to see whether the failure is driven by
   probe-side access-network noise or by anchor-side heterogeneity.

## Follow-up: anchors → anchors (anchors_meshed_pings)

Re-ran with `--table anchors_meshed_pings` (PlanetLab-analogue: both endpoints
are RIPE Atlas anchors, no consumer probes involved).

| | Spotter (PlanetLab) | probes → anchors | **anchors → anchors** |
|---|---|---|---|
| Pairs | ~40 k | 5 855 219 | 464 770 |
| Landmarks / anchors | ~100 | 783 | 780 |
| Pooled standardized μ | −0.078 | +0.027 | **+0.071** |
| Pooled standardized σ | 1.035 | 0.894 | **0.964** |

- Panel (b) histogram now nearly indistinguishable from N(0, 1); only a faint
  leptokurtic tilt remains.
- Panel (c) Q-Q plot per-anchor curves hug the diagonal much more tightly. A
  few anchors still drift (e.g. `91.132.8.99` shifted right), but the
  dramatic S-shapes from `ping_10k_to_anchors` are gone.

**Conclusion:** the failure on `ping_10k_to_anchors` is driven by
**probe-side heterogeneity** (consumer DSL/fiber/mobile last-mile delay and
queueing), not by anchor-side heterogeneity. On the PlanetLab-analogue setup
(anchor → anchor) Spotter's landmark-independence claim approximately holds
on RIPE Atlas too. Outputs:
[scripts/libs/cbg_feasibility/outputs/spotter_normality/anchors_meshed_pings/](../scripts/libs/cbg_feasibility/outputs/spotter_normality/anchors_meshed_pings/).

Practical takeaway sharpens: **Spotter is a fine model when calibrating
between high-quality vantage points, but using a single pooled normal to
geolocate *consumer endpoints* on Atlas will systematically underweight
per-source variance differences.**

## Reproduce

```bash
# Default: probes -> anchors
python -m scripts.libs.cbg_feasibility.spotter_normality_check

# PlanetLab-analogue: anchors -> anchors
python -m scripts.libs.cbg_feasibility.spotter_normality_check --table anchors_meshed_pings

# Optional flags: --max-rtt 80 --n-bins 40 --n-anchors 5
```
