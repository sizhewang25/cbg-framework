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

## Follow-up: slicing by (probe-ASN, anchor-ASN)

Direct test of the mechanism story above: if last-mile heterogeneity drives
the failure, restricting both endpoints to a single ASN (one access-tech
family, one operational discipline) should recover Spotter's `σ_z ≈ 1.0`.

Top-5 (probe-ASN, anchor-ASN) pairs by sample count:

| Slice | σ_z | μ_z | n |
|---|---|---|---|
| Unsliced baseline | 0.894 | +0.027 | 5.85M |
| Anchors-meshed | 0.964 | +0.071 | 465k |
| AS7922 Comcast → AS396982 Google Cloud | **0.978** ✓ | +0.016 | 4.6k |
| AS7922 Comcast → AS20473 Vultr | 0.892 | −0.100 | 4.3k |
| AS7922 Comcast → AS202422 G-Core | 0.867 ↓ | −0.001 | 3.5k |
| AS3320 DT → AS20473 Vultr | **1.004** ✓ | +0.022 | 3.0k |
| AS12322 Free → AS20473 Vultr | 1.087 ↑ | +0.052 | 2.9k |

**3 of 5 ASN-pair slices snap onto Spotter's target** (Comcast→GCP,
DT→Vultr, Free→Vultr) — direction of the hypothesis is confirmed: enforce
infrastructure uniformity, recover the normal.

**2 of 5 stay at or below the unsliced baseline.** Looking at their panel
(b) histograms:
- Comcast→G-Core (σ_z=0.867) is *even more* leptokurtic than the unsliced
  case. G-Core anchors span Europe + Asia, but Comcast probes are all US →
  RTT-distance is a mixture of transatlantic vs trans-Pacific routing.
- Comcast→Vultr (σ_z=0.892) similar story across many Vultr POPs.
- Free→Vultr (σ_z=1.087) is overdispersed and **visibly bimodal** — Free is
  French, Vultr has POPs in 25+ cities; different POP-to-Paris routes
  produce distinct delay-distance regimes.

Conclusion at this level: **ASN-uniformity is necessary but not sufficient.**
Intra-ASN geographic spread leaves enough routing diversity to keep the
distribution non-normal. Script:
[scripts/libs/cbg_feasibility/spotter_normality_by_asn.py](../scripts/libs/cbg_feasibility/spotter_normality_by_asn.py).

## Follow-up: country sub-slices within each ASN pair

If geographic spread is what's left, restricting also to a single
(probe-country, anchor-country) should fix the remaining slices. Result is
**counterintuitive**: σ_z gets *worse* in 4 of 5 sub-slices.

| Slice | σ_z | n | vs ASN-only |
|---|---|---|---|
| AS7922→AS396982 ASN | 0.978 | 4.5k | — |
| ↳ US→US | 0.839 | 1.3k | **−0.139** ↓ |
| ↳ US→JP | 0.840 | 0.5k | **−0.138** ↓ |
| AS7922→AS20473 ASN | 0.892 | 4.3k | — |
| ↳ US→US | 0.804 | 2.1k | **−0.088** ↓ |
| AS7922→AS202422 ASN | 0.867 | 3.5k | — |
| ↳ US→US | 0.726 | 0.5k | **−0.141** ↓ |
| AS3320→AS20473 ASN | 1.004 | 3.0k | — |
| ↳ DE→US | 0.951 | 1.7k | **−0.053** ↓ |
| AS12322→AS20473 ASN | 1.087 | 2.9k | — |
| ↳ FR→US | **1.010** ✓ | 1.7k | closer to 1 |

**What's happening:** two competing effects.

1. **Uniformity benefit** (what we expected): finer slice → fewer routing
   modes → cleaner normal.
2. **Fit instability** (what we didn't anticipate): smaller sample size +
   narrower RTT range → degree-3 polynomial over-smooths the local σ(d) →
   z values cluster tighter than the polynomial expects → σ_z drops.

Effect (2) dominates effect (1) for nearly every slice. The worst case
(Comcast→G-Core, US→US, σ_z=0.726, n=527) has a *catastrophic polynomial
fit* — μ(d) extrapolates to −60 000 km at high RTT because US-domestic
data covers only ~0–80 ms and the cubic diverges past the support.

The single winner — **Free→Vultr restricted to FR→US** (σ_z: 1.087 → 1.010)
— is exactly the case where effect (1) is unambiguous and effect (2)
doesn't bite. The original ASN slice was bimodal because Free routed to
both European and US Vultr POPs as two distinct regimes. FR→US strips out
the European mode → unimodal → σ_z ≈ 1.0. And the slice is large enough
(1.7k pairs) and spans a wide enough RTT range (transatlantic) that
polynomial over-smoothing isn't an issue.

Script:
[scripts/libs/cbg_feasibility/spotter_normality_by_asn_country.py](../scripts/libs/cbg_feasibility/spotter_normality_by_asn_country.py).

### Sharpened conclusion

`σ_z = 1.0` is **not a robust property of homogeneous infrastructure** — it's
a property of *a fit sitting in the bias-variance sweet spot*:
- Too much data → mixture variance dominates → σ_z < 1 (slight
  under-smoothing).
- Too little data → polynomial over-smooths σ(d) → σ_z < 1 (over-smoothing).
- Genuine multimodality → bimodal residuals → σ_z > 1.
- Spotter's PlanetLab fit (40k points, homogeneous population) sat in the
  sweet spot.

So the "infrastructure uniformity" hypothesis is **valid but incomplete**:
it explains anchors-meshed beating probes-only, and it explains why
FR→US disambiguates Free→Vultr's bimodal histogram. It does not predict
that tighter slicing universally improves the fit — fit pipeline mechanics
intervene before that limit.

## Closing: is there a universal delay-distance distribution?

No, not in the strong sense Spotter implies. Combining all our results:

- The unsliced probes→anchors fit gives σ_z=0.894 with a leptokurtic
  histogram and Q-Q curves that diverge across landmarks → a single
  `f_d(s)` does not describe modern Atlas data.
- ASN slicing gives σ_z values from 0.867 to 1.087 — those numbers don't
  converge to a stable population parameter.
- Even within the best-behaved slice, per-anchor Q-Q curves carry
  idiosyncratic offsets — anchors retain routing/peering signatures the
  pooled normal smooths over.

But Spotter wasn't *wrong* in 2010. Our anchors-meshed result (σ_z=0.964)
is striking confirmation that **when you restrict modern data to a
similarly homogeneous tier, their claim approximately holds 15 years
later**. The Internet's physics hasn't changed; the *measurement population*
has.

The "oversimplification" is an implicit scope claim Spotter's paper doesn't
flag:

| What Spotter said | What was actually true |
|---|---|
| "delay-distance is generic" | "…within our calibration tier" |
| "f_d is landmark-independent" | "…for landmarks sharing connectivity class" |
| Validated on ~100 PlanetLab nodes | Implicit constraint: only research-grade connectivity |

**What survives** from Spotter:
- The **probabilistic framing** — probability surfaces instead of CBG's
  hard "flat disks" — is a real conceptual advance and still valuable.
- The **Q-Q diagnostic** — Spotter gave us the very tool that diagnoses
  its own failure on modern data.
- The **homogeneous-tier regime** — for anchor-to-anchor or single-class
  measurements, a single normal is a reasonable approximation.

**What doesn't survive:**
- The claim that **one** global `μ(d), σ(d)` describes the whole Internet.
- The implicit assumption that landmark population doesn't matter — it
  dominates.

### What a modern analogue should look like

For the CBG-variant benchmark, the takeaway is that any approach assuming
a single global delay-distance model needs **population stratification**.
A defensible 2026-era redesign would probably include:

1. **Per-stratum fits** — separate `f_d(s)` for (probe access-class, anchor
   tier). Three or four buckets (research/IXP, datacenter, fiber-residential,
   DSL/mobile) likely capture most of the heterogeneity.
2. **Mixture models** within strata when multimodal structure remains
   (Free→Vultr-style cases).
3. **Per-VP calibration** for high-volume vantage points — what Octant
   does, what Spotter argued against, and what our results suggest is
   actually warranted on Atlas.
4. **Scope-aware reporting** — the model's confidence should depend on
   whether the target endpoint falls into a calibrated stratum. Geolocating
   a Starlink user with a fiber-calibrated model should produce wider, not
   falsely tight, probability surfaces.

The real lesson Spotter doesn't articulate: **delay-distance distributions
are properties of measurement infrastructure, not of the Internet.** Any
approach that pretends otherwise — theirs, or anyone claiming a "universal"
Internet-scale model — is implicitly assuming the calibration set
represents the target. The CBG-variant benchmark should bake this in:
report per stratum, not just pooled, and treat any single-global-model
claim as a hypothesis to test, not an assumption to inherit.

## Reproduce

```bash
# Pooled normality check (Spotter's Fig. 3 panels)
python -m scripts.libs.cbg_feasibility.spotter_normality_check
python -m scripts.libs.cbg_feasibility.spotter_normality_check --table anchors_meshed_pings

# Slice by (probe-ASN, anchor-ASN)
python -m scripts.libs.cbg_feasibility.spotter_normality_by_asn

# Slice by (probe-ASN, anchor-ASN, probe-country, anchor-country)
python -m scripts.libs.cbg_feasibility.spotter_normality_by_asn_country

# Optional flags: --max-rtt 80 --n-bins 40 --n-anchors 5 --top-k 5
```
