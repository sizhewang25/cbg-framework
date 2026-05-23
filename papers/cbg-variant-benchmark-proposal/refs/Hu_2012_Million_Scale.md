# Hu et al. 2012 — Towards Geolocation of Millions of IP Addresses

**Citation:** Zi Hu, John Heidemann, Yuri Pradkin. *Towards Geolocation of Millions of IP Addresses.* IMC 2012 (USC/ISI).

## Overview

This paper addresses the scalability problem in measurement-based IP geolocation. Prior CBG and Shortest-Ping work used hundreds of VPs probing each target many times — fine for hundreds of targets, but generating 1.8 × 10^13 records and overwhelming target traffic at Internet scale. Hu et al.'s contribution is not a new geolocation algorithm but a **VP-selection wrapper around existing CBG / Shortest-Ping that picks the few closest VPs per /24 block**, cutting probe traffic ~50× while preserving accuracy. With this approach they geolocated ~35% of allocated unicast IPv4 (~85% of directly-probeable addresses) as of August 2012.

## Core Algorithm

### CBG variant used
The paper uses **standard Gueye-2006 CBG as a black box**: each VP draws a circle centered at its location with radius derived from RTT via a "bestline" latency-to-distance estimate; the target is placed in the intersection (multilateration over all circles). They explicitly contrast this with Octant (positive + negative constraints) and TBG (topology-aware) and do not adopt those refinements. The paper's contribution sits *upstream* of CBG: which VPs feed it.

### Three conjectures justifying VP reduction
1. *A few VPs can be as accurate as many.*
2. *Certain small subsets have good accuracy.*
3. *The closest VPs generally maximize accuracy.*

Validated on 400 PlanetLab VPs × 25 university targets, with 10,000 random subset trials. CBG with 10 VPs gave median error 231 km vs. 208 km with all 400 VPs (≈11% worse). Shortest Ping with 10 VPs gave median 105 km — indistinguishable from all 400.

### VP-selection algorithm (4 steps)
Operates per /24 block:
1. **Find representatives:** Use IPv4 census + hitlist prediction (Fan & Heidemann) to pick 3 responsive IPs per /24.
2. **Select nearby VPs:** All 500 VPs probe the 3 representatives 10× (~16 hours total at 200 probes/s) — uses *second-to-minimum RTT* (drops the absolute min as outlier). Per block, retain only the closest ~10 VPs.
3. **Probe block:** Selected nearby VPs probe every address in the /24 at 500 probes/s.
4. **Centralize and geolocate:** Run Shortest Ping or standard CBG on the collected RTTs.

This is *not* a round-based refinement in the iterative-CBG sense; it is a **two-stage filter** (cheap fan-out → narrow probe). Round-based iteration is later work (cf. the IMC 2023 replication).

### Relation to Gueye 2006
CBG itself is unchanged. Gueye 2006's bottleneck was per-target VP count; Hu 2012's contribution is showing the *closest few* VPs preserve CBG accuracy, enabling Internet-scale rollout. No changes to the speed-of-internet model or bestline calibration are proposed.

## Evaluation Setup

- **VPs:** ~500 PlanetLab nodes (well-connected, university-biased — the authors flag this as a possible optimism).
- **Ground-truth targets for accuracy:** 25 universities worldwide (12 of which do *not* host PlanetLab nodes, mitigating co-location bias).
- **Block sample for VP-selection validation:** 18 /24 blocks from a CAIDA ground-truth dataset, ~100 responsive IPs each.
- **Production scale:** 393k /24 blocks per run (≈6 /8s); ~150k–217k /24s processed per 8-second window; final coverage 78 /8s ≈ 35% of allocated unicast IPv4.
- **Probe pacing:** Cap 200 probes/s per VP during VP-selection; 500 probes/s during block probing. Per-/24 incoming rate ≤ 0.5–0.9 packets/s (below Internet-background-radiation noise).

## Key Results

- **CBG, 10 closest VPs vs. all 400:** median error 231 km vs. 208 km (≈11% worse). 90th-percentile errors also close (Fig. 5).
- **Shortest Ping, 10 closest VPs vs. all 400:** median 105 km, "basically indistinguishable" (Fig. 4).
- **Selected VPs are ≈2% of the full set** → ~50× traffic reduction per target.
- **Linear RTT-vs-error correlation:** Shortest Ping r = 0.88; CBG r = 0.71 (noisier when min-RTT > 25 ms). Confirms that closest-RTT VPs really are most informative.
- **Coverage milestone:** 35% of allocated unicast IPv4 (~85% of directly-probeable) geolocated as of Aug 2012.

## Strengths

- First demonstration that CBG / Shortest-Ping scale to the full IPv4 address space without accuracy collapse.
- Practical operational guidance: probe pacing, representative discovery, second-to-min RTT to suppress outliers.
- Generates an artifact (Hilbert-curve visualization, public dataset) usable by downstream researchers.

## Limitations

- **VP and target bias:** 500 PlanetLab + 25 universities — well-connected academic networks; authors acknowledge accuracy is likely optimistic for the general Internet.
- **No new accuracy on CBG itself** — median CBG error stays in the 200–230 km range. Street-level claims are out of scope.
- **No negative constraints** (Octant) or topology constraints (TBG); leaves accuracy gains on the table.
- **Latency-to-distance via bestline only** — no per-region recalibration.
- **Assumes target stability** (≥2 days route stability; static physical location).
- **Ground truth is small** (25 universities) — statistical confidence on tail behaviour is limited.

## Relevance to CBG-Variant Benchmarking

This is the **direct ancestor of the CBG variant implemented in `scripts/analysis/analysis.py`** in our repo. Specific algorithmic choices that the benchmark should treat as configurable knobs and explicitly evaluate:

- **`n_shortest` VP-selection count** (paper uses 10 closest; code default = 10 via `compute_closest_rtt_probes`). Sweep 1, 3, 5, 10, 20, 50 to reproduce Hu's 11% claim and locate the knee.
- **Speed-of-internet constant** (`SPEED_OF_LIGHT * 2/3` in `default.py`). Paper uses bestline; our repo uses a fixed fraction. Benchmark should quantify the gap on the IMC 2023 anchor set.
- **Threshold distances** `[0, 40, 100, 500, 1000]` km (in `default.py`) — these directly continue Hu's median-error framing; report CDFs at these breakpoints.
- **Second-to-minimum RTT vs. minimum RTT** for representative probing — Hu prefers second-to-min to drop outliers; worth A/B testing.
- **Closest-VP heuristic vs. greedy geographic diversity** — Hu's "closest few" is a strong baseline; the round-based refinement and greedy VP-selection layered on top in later work (and in `analysis.py`) should be ablated *against* plain closest-N to isolate the gain.
- **Block-level vs. per-IP geolocation** — Hu geolocates whole /24s from a few representatives. Benchmark should distinguish per-target accuracy from per-block accuracy.
- **Bestline vs. plain speed-of-internet bound** for CBG radius — Hu's choice; not currently in the repo. Candidate variant to add.

Hu 2012 sets the *floor* the benchmark must clear: a CBG variant adding cost (round-based iteration, topology, negative constraints) should beat closest-10-VP CBG by a margin justifying the extra measurements.
