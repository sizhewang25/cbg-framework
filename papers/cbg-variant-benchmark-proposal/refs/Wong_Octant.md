# Octant: A Comprehensive Framework for the Geolocalization of Internet Hosts

**Authors:** Bernard Wong, Ivan Stoyanov, Emin Gün Sirer (Cornell University)
**Venue:** NSDI 2007
**Relevance:** Primary CBG-family variant; reference for the `scripts/libs/octant` module in this repo.

## Overview

Octant is a constraint-based geolocation framework that generalizes vanilla CBG along three axes: (1) it uses **both positive and negative constraints** (where a node *can* and *cannot* be), (2) it represents arbitrary, possibly non-convex and disjoint regions as **Bézier-bounded** areas with efficient union/intersection/subtraction, and (3) it **calibrates per-landmark latency-to-distance functions** ($R_L$, $r_L$) from the convex hull of inter-landmark measurements rather than assuming a single global $2/3 c$ bound. It further introduces a "height" term per landmark to absorb last-hop delays, weights to handle noisy constraints, secondary landmarks via piecewise router localization, and optional geographic/demographic clipping. On PlanetLab the authors report a **median single-point error of 22 miles** — roughly a factor-of-three improvement over GeoLim/GeoPing/GeoTrack.

## Core Algorithm: Constraint Framework

**Constraint formulation.** Given a set $\Omega$ of positive constraints (regions the target *must* lie in) and $\Phi$ of negative constraints (regions it *cannot* lie in), the estimated region for target $i$ is:
$$\beta_i = \bigcap_{X_i \in \Omega} X_i \ \setminus\ \bigcup_{X_i \in \Phi} X_i.$$

**Positive vs. negative constraints (vs. vanilla CBG).** Vanilla CBG (Gueye et al. 2006) only uses positive constraints — disks of radius $d = \frac{2}{3}c \cdot \text{RTT}/2$ centered on each landmark — and intersects them. Octant additionally extracts a **negative constraint**: a high latency means the target cannot be within some minimum distance $r$ of the landmark. Combined, each primary landmark contributes an **annulus** $[r_L(d), R_L(d)]$ rather than a disk, dramatically tightening the feasible region. For secondary (uncertain) landmarks the positive constraint becomes a union of disks ($\bigcup_{(x,y) \in \beta_k} c(x,y,d)$) and the negative an intersection — both still computable in closed form on Bézier regions.

**Region representation.** Regions are bounded by piecewise Bézier curves (cubic, 4 control points each). A circle uses four Bézier segments / twelve control points exactly. Intersection, union, and subtraction reduce to transformations on segment endpoints, enabling non-convex and disjoint outputs efficiently. For scalability, complex secondary-landmark regions may be approximated by bounding circles.

**Latency-to-distance calibration ($R_L$, $r_L$).** Instead of a single $2/3 c$ bound, each landmark $L$ periodically pings the others and builds a latency-vs-distance scatter plot. The **upper convex hull** gives $R_L(d)$ (outer radius) and the **lower convex hull** gives $r_L(d)$ (inner radius), guaranteeing all empirical inter-landmark measurements are envelope-consistent. To prevent statistical brittleness at high latency, a percentile cutoff $\rho$ (50th or 75th) is used; beyond $\rho$, the bound smoothly transitions toward the speed-of-light line.

**Height function (last-hop delay).** Octant attributes inelastic last-hop delays to a per-landmark scalar "height." For three primary landmarks $a,b,c$ with measured inter-latencies $[a,b], [a,c], [b,c]$ and known great-circle distances, heights $a', b', c'$ are solved from a 3×3 system that distributes residual delay across endpoints. Each landmark then shifts $R_L$ up (or $r_L$ down) by the appropriate height before constraint extraction — preventing under-provisioned last-mile links from creating empty solutions.

**Handling uncertainty (weights).** Each constraint carries a weight that decays exponentially with latency (high-latency = lower confidence). Overlapping regions sum weights; the final estimate is the union of sub-regions whose weight exceeds a threshold. This lets Octant tolerate a few erroneous constraints without collapsing to the empty set — a known failure mode of GeoLim.

**Indirect routes (secondary landmarks).** Routers on the path from primaries to the target are localized first (piecewise) using Octant itself, optionally refined via `undns` reverse-DNS city extraction, and then used as secondary landmarks. Closer secondary landmarks dramatically shrink the region.

**Iterative refinement and geographic clipping.** A second optional phase tightens the spline-derived constraints with a scale parameter $\delta$ until the region is below a target size (or empty). Population density and ocean masks can be applied as extra positive/negative Bézier regions.

## Evaluation

- **Vantage points / landmarks:** 51 PlanetLab nodes in North America; separately, 53 public traceroute servers.
- **Targets:** 104 total (mix of PlanetLab + traceroute servers); no target and landmark co-located at the same institution.
- **Measurements:** 10 ICMP pings per pair with kernel timestamps; full traceroutes for secondary-landmark experiments. Collected Feb 1, 2006 (PlanetLab) and Sept 18, 2006 (traceroute servers).
- **Ground truth:** Self-reported coordinates for PlanetLab and traceroute server operators (acknowledged as imperfect).
- **Baselines:** GeoPing, GeoTrack, GeoLim.

## Key Results

| Method  | Median error (PlanetLab) | Worst case | Median error (traceroute servers) |
|---------|-------------------------|------------|-----------------------------------|
| Octant  | **22 mi**               | 173 mi     | **25 mi**                         |
| GeoLim  | 89 mi                   | 385 mi     | 56 mi                             |
| GeoPing | 68 mi                   | 1,071 mi   | 155 mi                            |
| GeoTrack| 97 mi                   | 2,709 mi   | 50 mi                             |

- GeoLim returned the empty set for ~30% of targets (50 landmarks); Octant returned a non-empty region for all.
- Octant's region area is ~½ of GeoLim's at all landmark counts; with as few as **15 landmarks**, accuracy is nearly identical to using all 50.
- **>90%** of targets fall inside Octant's estimated region (vs. ~80% for GeoLim at 10 landmarks, dropping sharply).
- **Ablations** (Fig. 12–13): intermediate (secondary) routers give the largest accuracy boost; weights contribute ~33%; height and exponential weights are smaller but meaningful. Geographic clipping ("cities" / oceans) shrinks region area substantially (Fig. 14).
- Implementation: ~9,800 LOC; localization runs in seconds on a 2 GHz machine once landmarks are calibrated.

## Strengths

- First framework to systematically combine **positive + negative** latency constraints.
- Per-landmark empirical calibration ($R_L$, $r_L$) is tighter and more principled than global $2/3 c$.
- Weighted, soft-constraint solver gracefully handles erroneous measurements (no empty-set failure).
- Bézier regions admit arbitrary geographic/demographic priors uniformly.
- Secondary landmarks via piecewise router localization extend the effective VP set without new infrastructure.

## Limitations

- 2007-era PlanetLab evaluation: only 104 targets, North America only, self-reported ground truth.
- The `undns` router-name heuristic can be wrong ("misnamed" routers hundreds of miles from where their DNS implies) — handled by weights, but adds variance.
- Bézier curve count grows with each intersection/union; scalability requires bounding-circle approximations.
- Calibration requires a meshed landmark fabric — not a black-box drop-in.
- Negative constraints depend on the heuristic latency-distance lower hull; aggressive cutoff $\rho$ trades soundness for precision.

## Relevance to CBG-Variant Benchmarking

Octant is arguably the most cited CBG generalization and a natural anchor in any CBG-family benchmark. The design points that the benchmark should be able to **isolate and toggle independently** are exactly Octant's ablation axes:

1. **Negative constraints** (annulus vs. disk-only). Does adding $r_L$ help on RIPE-Atlas-scale data, or has VP density made it redundant?
2. **Per-landmark calibration** ($R_L$, $r_L$ via convex hull) vs. fixed $2/3 c$ — the IMC 2023 baseline uses the latter.
3. **Height function** for last-hop delay — likely valuable for residential targets, less so for anchor-to-anchor RIPE Atlas meshes.
4. **Constraint weighting** (exponential-in-latency) and the union-by-weight-threshold solver vs. hard intersection. This is where Octant avoids GeoLim/CBG's empty-set failure mode.
5. **Secondary landmarks** via piecewise router localization — directly comparable to RIPE IPmap's `single-radius` and to traceroute-based VP augmentation.
6. **Geographic/demographic priors** (oceans, population density) — a "free" precision boost that any CBG variant can adopt.
7. **Region representation** (Bézier vs. discrete grid vs. polygon-intersection) — affects both accuracy and runtime.

A clean benchmark should expose each toggle separately so that comparisons against vanilla CBG (Gueye 2006), CBG-with-empty-set-fallback (IMC 2023), and full Octant attribute gains to the right mechanism rather than to "Octant" monolithically.

## Citation

```
Wong, B., Stoyanov, I., and Sirer, E. G. Octant: A Comprehensive
Framework for the Geolocalization of Internet Hosts. In NSDI '07.
```
