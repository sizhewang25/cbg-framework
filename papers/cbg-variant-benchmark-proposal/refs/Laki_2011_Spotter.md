# Spotter: A Model Based Active Geolocation Service (Laki et al., 2011)

## Overview

Spotter is a probabilistic, measurement-based IP geolocation method. Instead of producing a hard feasible region from RTT-derived distance constraints (as CBG and Octant do), it converts each landmark's RTT into a ring-shaped spatial probability density over the globe, multiplies the per-landmark densities, and reports either the maximum-probability location or a region at a chosen confidence level. Its central empirical claim is that the delay-distance relationship is *landmark-independent* — a single common distribution can be calibrated across all landmarks rather than one per landmark.

## Core Algorithm / Model

- For each landmark `L` with measured propagation delay `d` to target `T`, Spotter defines a spatial probability density `g_L^d(τ)` over candidate target locations `τ`. The density is isotropic around `L`, so it is fully specified by a radial profile `f_d(s)` where `s` is great-circle distance.
- Calibration data (~40,000 PlanetLab inter-node pairs of (RTT, great-circle distance)) shows that, after standardizing per delay bin, the distance distribution is approximately standard normal (fit: µ = −0.078, σ = 1.035). Spotter therefore models `f_d` as a normal distribution with mean `µ(d)` and stddev `σ(d)` learned as polynomial fits over delay.
- Per landmark: `g_L^d(τ) = A_d · f_d(S(L, τ))`, where `S` is great-circle distance.
- Multi-landmark fusion: assuming independence, the joint posterior over a region `H` is the product of per-landmark integrals (Eq. 2 in paper). Uniform prior on target location.
- Discretization uses Hierarchical Triangular Mesh (HTM) cells; probability is computed per cell, and the best-guess point is the maximum/mean cell or the center of the confidence region.

### What makes Spotter different from vanilla CBG
- CBG: hard *flat disk* feasibility region per landmark (tightest line above delay-distance pairs); intersect disks; centroid of intersection.
- Octant: hard *flat ring* (positive + negative constraints).
- Spotter: *soft probabilistic ring* per landmark; multiply densities. No hard infeasibility, so it degrades gracefully under inflated RTTs / indirect routing.
- Calibration is *global*, not landmark-specific — argued to be more stable because per-landmark calibration sets contain only a few hundred points each.

## Evaluation Setup

Two ground-truth datasets:
1. **PLANETLAB**: every PlanetLab node geolocated using all others as landmarks (leave-one-out, same environment used for calibration — authors acknowledge this is a limited test).
2. **COGENT**: CAIDA-curated set of >23,000 Cogent (Tier-1 ISP) router interfaces with verified locations across North America and Europe. Independent of calibration data.

- Landmarks: PlanetLab nodes (10 RTT probes per landmark per target, min RTT used to suppress queuing).
- Baselines: reimplemented CBG (Gueye et al.) and Octant (Wong et al.) delay models only — not the full Octant framework.
- Metric: great-circle error between estimated and true coordinates; reported as CDFs and median.

## Key Results

Median errors (km):

| Dataset | Spotter | Octant | CBG |
|---|---|---|---|
| PLANETLAB (all) | 75 | 125 | 175 |
| COGENT (all) | **30** | 120 | 100 |

Fraction of targets within error thresholds:

- PLANETLAB ≤10 km: Spotter 13%, Octant 10%, CBG 3%. ≤50 km: 35% / 26% / 23%.
- COGENT North America ≤10 km: Spotter 35%, Octant 2%, CBG 9%. ≤50 km: ~70% / 27% / 40%.
- COGENT Europe ≤10 km: Spotter 19%, Octant 1%, CBG 1%. ≤50 km: 40% / 4% / 8%.

Scalability: linear in number of targets; ~25 min measurement + 7 min evaluation for 10,000 targets on a 2×Xeon 2.5 GHz / 32 GB server.

## Strengths

- Significantly tighter median error than CBG/Octant on an independent, large-scale router dataset (COGENT).
- Robust to indirect routing / RTT inflation — soft rings don't catastrophically over-constrain.
- Calibration scales: one global model trained on the full pooled set rather than ~hundreds of points per landmark.
- Produces a full posterior surface (useful for confidence regions, aggregating over IP sets, fusing with priors like population density).

## Limitations

- PlanetLab calibration and PlanetLab landmarks — calibration set is geographically biased toward research/academic networks.
- Assumes per-landmark RTT measurements are statistically independent (paper notes this is not strictly true).
- Uniform prior on target location; no use of geographic/population priors in evaluation (though framework supports them).
- COGENT contains only router interfaces (well-connected, Tier-1), not residential or mobile hosts; generalization to arbitrary endpoints is untested.
- Baselines are reimplementations of *delay models only*, not the full Octant pipeline (no DNS/Whois/topology constraints) — comparison favors Spotter's strengths.

## Relevance to CBG Variant Benchmarking

Spotter is one of the canonical "soft" CBG variants and should be included as a probabilistic baseline alongside vanilla CBG and Octant. For our benchmark:

- **Replicate**: the Gaussian radial profile `f_d ~ N(µ(d), σ(d))` calibrated on RIPE Atlas anchor-mesh ping data (analogous to the 40k PlanetLab calibration pairs). The `µ(d)`, `σ(d)` polynomial fit is straightforward.
- **Compare against**: vanilla CBG (hard disk) on identical VPs/targets — Spotter's headline claim is the median-error gap (30 vs 100 km on COGENT). We should test whether this gap reproduces on the IMC-2023 RIPE Atlas anchor set.
- **Key knob to study**: landmark-pooled vs landmark-specific calibration. Spotter's pooled calibration is its main novelty; with thousands of RIPE anchors we have far more per-landmark data than PlanetLab did, so the pooled-vs-specific tradeoff may look different.
- **Output format**: Spotter produces a probability surface, not a point — useful if our benchmark wants to score posterior quality (e.g., calibration, region coverage at a confidence level) rather than just point error.
- **Caveat for fair comparison**: Spotter's reported gains came partly from a hand-picked router dataset. We should evaluate on the IMC-2023 hard-GT anchor set (and the SWAP pressure test) to avoid replicating the original's evaluation bias.
