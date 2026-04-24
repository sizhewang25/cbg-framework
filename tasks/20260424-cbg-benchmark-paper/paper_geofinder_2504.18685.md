# Paper Summary: GeoFINDR (arXiv 2504.18685)

**Title:** GeoFINDR: Practical Approach to Verify Cloud Instances Geolocation in Multicloud
**Authors:** Saïd Ider, Maryline Laurent (Samovar, Télécom SudParis, Institut Polytechnique de Paris)
**Published:** April 2025
**Funding:** France 2030 ANR Project ANR-23-PECL-0009 TRUSTINCloudS
**Code:** https://github.com/diasredi/GeoFINDR
**arXiv:** https://arxiv.org/abs/2504.18685

---

## Problem & Motivation

Cloud customers specify in SLAs where their data should be located (GDPR compliance, CLOUD Act avoidance, availability), but cannot verify that CSPs are honoring those declarations. The paper focuses on **verifying the physical location of a VM** (not data or cloud servers), because the VM is where encrypted data is decrypted and is thus the highest-risk asset. This is a **compliance verification** problem, not a general geolocation problem.

---

## Threat Model

Prior work (CBDG, IGOD, SPLITTER) assumes an *economically rational CSP* that moves data to cheaper regions. GeoFINDR uses a more realistic **dishonest CSP** model: the CSP actively lies about VM location and has:
1. Total control and unlimited access to data and hardware
2. High-performance architecture including **dark fibers** (private low-latency inter-DC links)
3. Multicloud architecture (large-scale cloud, subcontractors, etc.)

The distinguishing feature vs. VLOC (which also assumes dishonest CSP) is that GeoFINDR accounts for dark fibers, making the threat model more realistic.

**Key assumptions:**
1. There exists another machine with similar performance (DDR) to the target VM, connected to the Internet
2. Delay between Internet machines has low variance in time
3. Two machines on the same network with similar config get the same RTT to a third machine
4. The VM is an IaaS/PaaS connected to the Internet and able to run GeoFINDR
5. The VM can send/receive ICMP pings to/from RIPE Atlas landmarks

The audit is **internal** (run from inside the VM), enabling measurement of **in-cloud delays** (loopback RTT, proxy RTT) that external methods cannot observe.

---

## Technical Approach

GeoFINDR improves three key stages of delay-based geolocation. All steps are wrapped in a convergence loop.

### Step 1 — Greedy Audit Landmark Selection (→ LM_A)

Given the CSP-declared position and a `zone_size` radius, select `NB_LM` dispersed RIPE Atlas anchors using a greedy algorithm called **Dispoints** (O(n×m) complexity). Dispoints maximizes the sum of pairwise distances among the selected subset, ensuring geographic spread. Dispersed and nearby landmarks yield better accuracy than random or distant ones.

### Step 2 — DDR-based Sectorization (→ LM_S)

Rather than fitting a global Distance-Delay Relation (DDR) regression, GeoFINDR finds landmarks with **similar DDR behavior** to the VM. For a given RTT of ~20ms, distances to landmarks can range from 250 to 1100 km, making regression unreliable. The sectorization approach:
- Pings each LM_A landmark from the VM, recording measured RTT
- Collects all landmarks whose RTT to LM_A falls within `[measured_delay ± interval_percent]`
- LM_S = landmarks with the maximum occurrence count across all LM_A measurements

This avoids fitting a DDR model entirely — it identifies landmarks co-located or similarly-positioned relative to the VM.

### Step 3 — Position Estimation via Barycenter

Rather than finding the intersection of circles (which often yields an empty set for noisy measurements), multilateration is framed as an **optimization problem**: find the position minimizing squared distances to the estimated-distance circles around each LM_S landmark. Quality metric: SMRE (squared mean root error). The estimated position is the **barycenter of LM_S landmarks weighted by normalized measured delays**. This always returns a result, solving the zero-intersection problem.

### Step 4 — Convergence Loop

Steps 1–3 are in a do-while loop that re-centers the search area on the current estimate and repeats until `distance(prev_estimate, curr_estimate) < tolerance`. This corrects cases where the CSP declares a false location far from the truth.

**Full pseudocode (Algorithm 2):**
```
sector_coordinates ← declared_position
do:
  near_landmarks ← landmarks where distance < zone_size
  LM_A ← Dispoints(near_landmarks, NB_LM)
  similars ← {}
  for lm in LM_A:
    measured_delay ← Ping(lm.ip)
    add to similars: landmarks with RTT to lm in [measured_delay ± interval_percent]
  LM_S ← landmarks with max occurrences in similars
  measured_delays ← [Ping(lm.ip) for lm in LM_S]
  distance_scale ← Normalize(measured_delays)
  sector_coordinates ← Barycenter(LM_S, distance_scale)
while distance(prev, sector_coordinates) >= tolerance
estimated_position ← sector_coordinates
```

---

## Implementation

- **Language:** Python 3, using RIPE Atlas Cousteau library
- **Probing:** ICMP ping, 3 pings, 64-byte payload (same as RIPE Atlas default)
- **RTT** = Internet delay + in-cloud delay (measured from inside the VM)
- **Landmarks:** RIPE Atlas anchors (~850 worldwide, verified coordinates, homogeneous hardware, public IPs, permanent inter-anchor measurements)
- **4 tunable parameters:** `tolerance`, `zone_size`, `NB_LM`, `interval_percent`

**Additional in-cloud metrics:**
- `loopbackRTT`: ping to 127.0.0.1 — proxy for VM's internal processing speed
- `proxyRTT`: ping to cloud's public IP — estimates in-cloud routing delay; high value suggests dark fiber or complex flow routing

---

## Experimental Setup

- **Controlled environment:** Fixed VM at Telecom SudParis, Evry (25km south of Paris); 64-bit 8-core 1.60GHz, 32GB RAM, Linux 6.12.11, 1Gb/s link
- **Ground truth:** VM location is known; declared positions are synthetically varied
- **24 declared positions:** 1 true (Evry) + 23 false (world capitals: Amsterdam, Tokyo, Buenos Aires, Los Angeles, Moscow, etc.)
- **Success criterion:** estimated position within 50km of true position
- Two experiment types: parameter sensitivity (Section 5.2) and landmark meshing (Section 5.3)

---

## Key Results

### Parameter Sensitivity (optimal values)

| Parameter | Optimal Value | Accuracy | Audit Time |
|-----------|:-------------:|:--------:|:----------:|
| `tolerance` | 100 km | 22.6 km avg | 49.8s |
| `NB_LM` | 16 | **22.1 km best** | 67.7s |
| `interval_percent` | 35% | 25.8 km avg | 56s |
| `zone_size` | matched to area | small accuracy effect | up to 78.2s for large values |

Best result: **22.1 km** accuracy with `tolerance=100, zone_size=1000, NB_LM=16, interval_percent=35`

### Landmark Meshing Experiment

| Landmark Condition | True position accuracy | False position lie detection |
|-------------------|:---------------------:|:---------------------------:|
| Dense (Paris area landmarks present) | 22.1 km, 34.3s | avg 22.4 km, 87.4s |
| Dead zone (Paris landmarks removed) | 303.8 km, 396.9s | avg 164.3 km, 244.1s |

Even without nearby landmarks, the lie detection remains effective: declared lie of 3406.9 km, estimated lie of 3445.3 km (only 1.1% error). Accuracy degrades to continental scale but the fraud is still detected.

---

## Comparison with Prior Work

| Method | Audit type | RTT type | Landmarks | Threat model | Best accuracy |
|--------|-----------|----------|-----------|-------------|:-------------:|
| CBDG | external | Internet | PlanetLab | Economically rational | median 166km |
| IGOD | external | total | PlanetLab | Economically rational | avg 88.5km |
| SPLITTER | external | total | PlanetLab | Economically rational | best 30km, avg 139km |
| VLOC | internal | total | Web sites | Dishonest (no dark fiber) | best 150km |
| **GeoFINDR** | **internal** | **total** | **RIPE Atlas** | **Realistic dishonest** | **best 22.1km, avg 22.6km** |

---

## Limitations & Caveats

1. **Single controlled environment:** all experiments from one VM in Evry, France — no experiments from different VM locations or cloud providers
2. **RIPE Atlas density is uneven:** accuracy degrades to ~300km in landmark-sparse regions; requires correct `tolerance` tuning per scale
3. **Dark fiber and VPN not tested:** identified as future work; proxyRTT provides a signal but adversarial manipulation is not evaluated
4. **Parameter sensitivity:** `tolerance` must match the scale of nearby landmarks; wrong values cause premature termination or long audit times
5. **Not evaluated against RTT manipulation** by a sophisticated dishonest CSP (e.g. artificially inflating delays)
6. **No comparison of lie detection rate** against prior methods (Table 1 only compares accuracy, not F1/precision/recall for lie detection)

---

## Artifacts

- GeoFINDR code: https://github.com/diasredi/GeoFINDR
- Dispoints algorithm: https://github.com/diasredi/dispoints

---

## Relevance to Our CBG Benchmark Paper

**Similarities:**
- Both use RIPE Atlas anchors as landmarks, validating their suitability for city-scale geolocation
- Both involve delay-based multilateration from external VPs to a target host
- GeoFINDR's 22.6km average accuracy demonstrates RIPE Atlas landmark quality is sufficient for our use case

**Key differences — cite and differentiate:**

| Dimension | GeoFINDR | Our Paper |
|-----------|----------|-----------|
| Goal | CSP compliance verification (detect lying CSP) | Geolocation of IPs for mobile operator traffic analysis |
| Algorithm | DDR sectorization (find similar landmarks) — not a CBG variant | CBG multilateration (intersect distance-constrained regions) |
| VP model | Internal audit from within the target VM | External RTT from mobile operator VPs to target |
| Target IP types | Unicast VMs in cloud | Unicast and anycast cloud IPs |
| Benchmark scope | Single method, parameter sensitivity | 18 CBG pipeline combinations |
| Scale | One VM, 24 positions | Millions of IPs |

GeoFINDR is **not a CBG variant** and does not address anycast IPs. It should be cited as recent adjacent work demonstrating RIPE Atlas viability for fine-grained geolocation.
