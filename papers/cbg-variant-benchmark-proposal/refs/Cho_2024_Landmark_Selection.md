# Cho et al. 2024 — Selection of Landmarks for Efficient Active Geolocation

**Citation:** Cho, Weinberg, Bhattacharya, Dai, Rauf. *Selection of Landmarks for Efficient Active Geolocation.* TMA 2024 (IFIP). Code: https://github.com/grace71/tma24-vp-ls

## Overview

This paper addresses an **efficiency** (not accuracy) question for active geolocation: given a pool of landmarks and a group of targets distributed worldwide, what is the smallest subset of landmarks that produces the **same** geolocation verdicts as the full pool? They evaluate several landmark-selection strategies — random, clustering-based (by AS / city / country / continent), greatest-distance (geodesic and RTT-based), and hybrids — against the RIPE Atlas anchor pool (780 anchors) using a simplified CBG-style accept/reject geolocation algorithm. Best result: a hybrid of random initialization + AS-cluster + geodesic-distance maximization reproduces full-pool verdicts using only **32% (280/780) of landmarks**.

## Core Algorithm

**Geolocation algorithm under test** is ICLab's CBG-style verifier: given a target's *claimed country*, convert each landmark→target RTT to an implied minimum speed (RTT vs. great-circle distance to the nearest border of the claimed country); reject if any landmark implies a speed above the calibrated limit. Speed limit = **153 km/ms (0.51 c)**, calibrated via pairwise anchor distances (an improvement over Katz-Bassett's 133 km/ms / 0.44 c). Reducing landmarks only ever converts rejects into accepts — so disagreements with the full pool are one-directional.

**Selection strategies tested:**
- **Random** — uniform sampling without replacement.
- **Clustering** — partition landmarks by AS / city / country / continent (binary metric $d_C(a,b)=0$ if same cluster else 1); subset must draw evenly across clusters.
- **Greatest-distance (diversity)** — greedy Prim-style maximization of $D(\mathbb{S}) = \sum_{s,t \in \mathbb{S}} d(s,t)$ over (i) geodesic distance $d_g$ or (ii) measured min-RTT $d_r$. Optional "start in claimed country" variant.
- **Hybrid 1** — clustering + geodesic distance combined with weights $d_h = W_C d_C + W_d d_d$ ($W_d=1$, $W_C \gg \max d_d$).
- **Hybrid 2** — first 100 landmarks chosen at random, then extend via Hybrid 1.

**Metric:** *Agreement* with the full-pool verdict (% of 559 targets where subset and full pool produce identical accept/reject), not raw accuracy.

> We test our landmark selection techniques using a simple active geolocation algorithm developed for internal use by ICLab [25, 35]. Given a set of landmarks, a target, and a “claimed location”—a country where the target supposedly isICLab’s algorithm produces a yes-or-no judgment of whether the claimed location is accurate. It works by assuming each measurement packet traveled the shortest possible great-circle distance from its source landmark to the nearest border of the claimed country, and no further. Each measured RTT is converted to the minimum speed that the measurement packet would have had to travel to cover that distance. If any of these minimum speed estimates is greater than a calibrated speed limit, the claimed location is rejected, otherwise it is accepted. The speed limit is calibrated using a simplified version of the calibration procedure for CBG [14]: For all pairs of landmarks, divide the distance between that pair by the minimum RTT measured between that pair, producing a travel speed estimate. Take the fastest of all such speed estimates as the speed limit. For this study, we used a calibrated speed limit of 153 km/ms (0.51 c). This is a little higher than Katz-Bassett et al. [19]’s estimated “speed of Internet” (133 km/ms, 0.44 c), but well below the theoretical limit of 200 km/ms (0.67 c), the speed of light in long-distance optical fiber.1 We presume the difference from Katz-Bassett et al.’s estimate reflects improvements to global network latency since 2006. In most cases, measurement packets will have to travel much farther than just to the nearest border of the claimed country. Dividing an RTT measurement by a shorter distance than the packet actually would have had to travel produces a speed estimate lower than the packet’s true speed. Therefore, ICLab’s algorithm errs systematically on the side of acceptance. Reducing the number of landmarks used for a measurement can only increase this systematic error, because removing data points from a measurement that rejected a claim can convert it into a measurement that accepts a claim, but not vice versa. In the rest of the paper, whenever we discuss “agreement” between results from the full landmark pool and results from a reduced set of landmarks, keep in mind that each disagreement means the data from the full landmark pool rejected a claim and the data from the reduced set did not.

## Evaluation Setup

- **Landmarks:** 780 RIPE Atlas **anchors** (Dec 2022). Probes excluded — bandwidth-limited, volunteer-run, potentially mislocated. Geographic distribution skewed: 438 in Europe vs. 18 in Africa, 25 Oceania.
- **Targets:** 559 commercial VPN endpoints; claimed country from operator. Distribution: 176 N. America, 120 Europe, 110 Asia, 103 Africa, 41 Oceania, 27 S. America.
- **Measurements:** ~2M ICMP Echo pairs over 5 days; each VPN server received ≤2400 packets.
- **Calibration validity:** min-RTT between anchors is ≥99.5% correlated hour-to-hour — calibration safe within a day.

## Key Results

| Selection method | Landmarks for 100% agreement | % of pool |
|---|---|---|
| **H2-AS** (random-100 + Hybrid1 with AS clusters) | **280** | **36%** |
| H1-COUNTRY (clustering + geodesic, country) | 384 | 49% |
| H2-CONTINENT | 610 | — |
| H1-AS | 410 | — |
| DIST-GEO (geodesic diversity only) | 590 | 76% |
| DIST-RTT (RTT diversity only) | 547 | 70% |
| Random (pure) | never — caps at 90% with ~14 landmarks, 95% with 50, 97.5% with 180 | — |

- Random gets 90% agreement at only 14 landmarks but **plateaus** — never reaches 100%.
- City- and AS-clustering slightly beat random; country/continent clustering do not.
- Geodesic diversity beats RTT diversity for most subset sizes — RTTs are confounded by queueing / detour effects, so "RTT-far" landmarks can be RTT-far from *everything* and contribute no constraint.
- **Best overall:** H2-AS — total ICMP request load drops from 1,308,060 to **469,560 packets** (~3× reduction) for the same verdicts.
- H1-AS reaches **99.46% agreement at 213 landmarks (27% of pool)** if minor instability is acceptable.

## Strengths

- Clean, reproducible methodology with public code.
- Honest framing: optimizing efficiency *at fixed accuracy*, not accuracy itself.
- Quantifies a useful negative result: **RTT-distance maximization is worse than geodesic-distance maximization**, contradicting an intuition from prior work (Dang, Xie).
- Demonstrates fine-grained diversity (city/AS) matters; coarse (continent/country) does not.
- Calibration-stability analysis (Fig. 2) is reusable.

## Limitations

- Geolocation algorithm is **country-level accept/reject only** — not point-estimate CBG. "Agreement" ≠ "accuracy"; any systematic error in the full-pool verdict is preserved.
- **Anchors only** — probes excluded. Authors acknowledge this hurts coverage in Africa / S. America and is a priority for future work.
- Targets are commercial VPN endpoints, which have known location-claim issues; ground truth is the VPN operator's *claim*, not verified physical location.
- Non-incremental: selection is computed offline against a fixed target set. Does not adapt per-target (unlike Jiang/Du/Hu).
- Speed limit 0.51 c is conservative — works for accept/reject, would inflate radii in a true CBG centroid.

## Relevance to CBG Variant Benchmarking

Landmark/VP selection is **one of the core axes** of CBG variant design (alongside speed model, circle aggregation, and round structure). This paper is the cleanest recent reference for that axis. Concretely for our benchmark:

1. **Variants to include:** random baseline, AS-cluster, city-cluster, geodesic-diversity, H1-AS, H2-AS. These span the design space the paper validates.
2. **Replicate the negative result:** confirm RTT-diversity ≤ geodesic-diversity on our IMC 2023 RIPE setup — if it flips, that's a finding.
3. **Different evaluation target:** the paper uses *agreement with full-pool verdict*; our setup has hard ground truth (RIPE anchor coordinates, MaxMind/IPInfo), so we can score **selection strategies on actual point-estimate error**, not just agreement. This is a strict generalization and likely shifts the ranking — e.g., RTT-diversity may matter more when error magnitude (not just accept/reject) is the metric.
4. **Probe inclusion:** the paper's biggest acknowledged gap. Our IMC 2023 dataset includes ~10K probes — running their selection strategies over probes+anchors lets us answer the open question they punt on.
5. **Scale comparison:** their pool is 780 anchors / 559 targets; ours is ~10K probes / 723 hard-GT targets (probes→anchors). Larger pool means the 32% number is unlikely to transfer directly — worth re-measuring.
6. **Concrete reuse:** their calibrated speed limit (0.51 c) and the calibration-stability finding (intra-day min-RTT correlation ≥99.5%) directly inform our experimental protocol.
