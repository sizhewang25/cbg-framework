# Gueye et al. 2006 — Constraint-Based Geolocation of Internet Hosts

**Citation:** Gueye, Ziviani, Crovella, Fdida. *IEEE/ACM Transactions on Networking*, Vol. 14, No. 6, December 2006. (Extended journal version of the IMC 2004 paper.)

## Overview

CBG is the first measurement-based geolocation method to use **multilateration with geographic distance constraints**, producing a **continuous** confidence region instead of the discrete-answer space of prior landmark-equals-answer methods (GeoPing, GeoTrack, GeoCluster). Each landmark `L_i` converts its measured RTT to a target `τ` into an overestimated upper bound on the great-circle distance via a per-landmark **"bestline"** calibration, draws a circle of that radius around itself, and CBG intersects the K circles to obtain a feasible region `R`. The centroid of `R` is the point estimate; the area of `R` is a per-target confidence region — a key novelty enabling applications to decide whether the estimate is "good enough."

## Core Algorithm

### 1. Bestline calibration (RTT → distance)

For each landmark `L_i`, gather (delay `d_ij`, known great-circle distance `g_ij`) pairs to every *other* landmark `L_j`. Compute the **tightest lower linear bound** with non-negative intercept:

- Feasibility: `y − m_i·x − b_i ≥ 0` for all data points (so the line lies below all measurements)
- Constraint: `b_i ≥ 0`, `m_i ≥ m` where `m` is the slope of the theoretical **baseline** (2/3 c in fiber → ~1 ms RTT per 100 km of cable)
- Objective: `min Σ (y − m_i·x − b_i)` — minimize total distance from line to all points

This is solved as a linear program per landmark. The bestline `y = m_i·x + b_i` captures the *least-distorted* observed relationship from `L_i`'s viewpoint. The non-zero intercept `b_i` absorbs fixed/localized delay.

### 2. RTT → distance constraint

Given measured delay `d_iτ` from `L_i` to target `τ`:

$$\hat{g}_{i\tau} = \frac{d_{i\tau} - b_i}{m_i}$$

The constraint `ĝ_iτ` is an **overestimate** of the true great-circle distance because delay distortion is purely *additive* (queueing, circuitous routing, lack of great-circle paths all add delay, never subtract).

### 3. Feasible region (intersection)

Each landmark defines a disk `C_iτ` of radius `ĝ_iτ` centered at `L_i`. The feasible region is:

$$\mathcal{R} = \bigcap_{i=1}^{K} \mathcal{C}_{i\tau}$$

`R` is convex (intersection of convex disks). Because all constraints are overestimates, `R` is non-empty and contains `τ` in practice — verified empirically across all target hosts in both datasets. (Self-calibration via bestline prevents the "mismatch" failure mode where some constraints underestimate.)

### 4. Point estimate (centroid)

`R` is approximated by a polygon whose vertices are intersection points of the boundary circles that lie inside all other disks. Point estimate = polygon centroid via the standard shoelace-style formula (Eq. 5–7). Confidence region = polygon area in km².

## Evaluation Setup

- **Dataset 1 (W.E.):** 42 RIPE TTM hosts in Western Europe with GPS-known coordinates, one-way delays sampled over 10 weeks (Dec 2002 – Feb 2003). 2.5th-percentile RTT used as min RTT (filters queueing/local delay).
- **Dataset 2 (U.S.):** 95 NLANR AMP hosts in continental U.S., RTT data from Jan 30, 2003.
- **Leave-one-out:** Each host is geolocated using the other K−1 hosts as landmarks.
- **PlanetLab deployment:** 57 landmarks (24 US, 24 EU, 5 Asia, 3 SA, 1 Oceania) geolocating 42 US + 43 EU PlanetLab nodes (May 2005).
- **Baselines compared:** DNS-based (SarangWorld Traceroute) and GeoPing-like (discrete-answer measurement-based).

## Key Results

**Dataset experiments (Section IV-C):**

| Dataset | Median error | Mean error | 80th-percentile error |
|---|---|---|---|
| Western Europe (42 landmarks) | **22 km** | 78 km | 134 km |
| U.S. (95 landmarks) | **95 km** | 182 km | 277 km |

**Confidence regions (Section IV-D):**
- U.S.: 80% of estimates have confidence area ≤ 10⁵ km² (≈ size of Portugal / Indiana). 25% achieve ≤ 10³ km² (metropolitan scale).
- W.E.: 80% ≤ 10⁴ km²; 65% ≤ 10³ km².

**PlanetLab deployment (Section V):** median error 42 km (Europe targets), 130 km (US targets); 80th-percentile 218 km / 411 km — degraded vs. the controlled dataset due to access-link diversity (modems, wireless, ADSL).

**Scaling with landmark count (Fig. 10):** Mean error levels off at roughly **K = 30 landmarks**; beyond that the marginal benefit is small.

**vs. baselines:** Median error 22/95 km vs. ~100/150 km for GeoPing-like on W.E./U.S. respectively. CBG dominates the CDF on both datasets.

## Strengths

- **Continuous answer space** (vs. discrete-equals-landmark methods) — accuracy not capped at landmark density.
- **Per-target confidence region** — applications can self-assess whether the estimate is usable.
- **Self-calibrating bestlines** — adapts to current network conditions per landmark; absorbs asymmetric/localized delay through non-zero intercept.
- **Always-feasible region** (R non-empty) by construction in practice, since overestimation is conservative.
- **Privacy lever** — confidence-region area can be deliberately enlarged to give coarser answers to unprivileged users.

## Limitations (as identified)

- **Circuitous routing, localized delay, shared paths** inflate the confidence region — Section IV-F analyses these explicitly (e.g., Lisbon/Porto landmarks hiding each other → ~57,000 km² region for the Porto target; Pullman/Bozeman hidden behind Seattle in the US dataset).
- **Localized delay** appears as large intercept `b_i`; bestlines with large `b_i` always yield large confidence regions when used as targets (Fig. 11).
- **Geographic scope:** dataset hosts are confined to U.S. + Western Europe — well-connected, dense regions. Authors caution against extrapolating to sparser regions.
- **Landmark quality:** PlanetLab degradation shows network-diversity assumptions matter; landmarks behind access links bias bestlines.
- **No mismatch handling** — if some constraints underestimate (theoretically possible, not observed), `R` could exclude the true location; the paper leaves this for future work.
- **Proxies / firewalls / non-responsive targets** are unresolved (general to all active geolocation).

## Relevance to CBG-Variant Benchmarking

This paper **is the baseline** every later CBG variant modifies. Parameters and design choices subsequent work tweaks:

| CBG knob | Gueye 2006 setting | Variants that change it |
|---|---|---|
| **RTT → distance model** | Per-landmark bestline LP (non-negative intercept, slope ≥ baseline) | Octant uses *positive* (lower) + *negative* (upper) constraints; Spotter fits statistical delay-distance model; Alidade uses passive features |
| **Speed-of-Internet assumption** | Baseline slope = 2/3 c in fiber (1 ms RTT / 100 km cable) | Million-Scale (Hu 2012) uses 4/9 c, 3/9 c, 1/6 c tiers depending on RTT range |
| **RTT statistic** | 2.5th-percentile (W.E.) or minimum RTT (U.S.) | Many variants use plain min RTT; some use distributions/percentiles |
| **Region from constraints** | Disk intersection only | Octant adds negative constraints; round-based methods iterate; ML methods skip geometry entirely |
| **Point estimate** | Polygon centroid of `R` | Variants use weighted centroid, closest-landmark, ML regression, etc. |
| **Landmark count needed** | ~30 to plateau | Million-Scale extends to thousands of VPs (RIPE Atlas); landmark-selection algorithms (Cho 2024) study which subset to use |
| **Confidence output** | Polygon area in km² | Most successors retain or extend the region concept |

Specific numbers to anchor benchmarks against: **median 22 km (W.E., 42 landmarks)** and **median 95 km (U.S., 95 landmarks)** in the controlled-landmark setting; **median 42–130 km** on PlanetLab. Any modern CBG variant evaluated on RIPE Atlas anchors should be expected to match or improve these, since today's RIPE measurement coverage and routing are substantially better than 2003.
