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

## Why the assumption breaks here

Spotter validated on ~100 research-grade PlanetLab nodes hosted at universities
with relatively homogeneous connectivity. `ping_10k_to_anchors` pulls min RTTs
from ~11 k RIPE Atlas probes — consumer DSL/fiber/mobile/etc — to 783 anchors
spread globally. Per-anchor probe populations differ systematically:

- An anchor in a well-peered IXP city sees a fundamentally different probe
  set than one served mostly via long-haul transit.
- Access-network heterogeneity (last-mile delay, queueing) is far larger here
  than on PlanetLab.
- Both effects show up exactly where Spotter's Q-Q test was designed to catch
  them.

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
