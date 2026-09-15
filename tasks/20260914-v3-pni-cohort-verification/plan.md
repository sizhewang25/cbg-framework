# §8.1 PNI Cohort Verification (v3) — Plan

## Background

`papers/cbg-benchmark-as-network-operator/paper-flow-draft-v2.md` §8.1
(lines 404-467) argues a three-link chain for why Shortest-Ping reaches
near-perfect classification on traffic-weighted data:

1. Traffic weighting keeps targets **at PNI locations** with well-routed paths.
2. Min-RTT selects the VP **nearest the target's serving PNI**.
3. Nearest-seed snapping turns that VP's coordinate into the **right class**.

Two problems block it.

**No traffic-weighted data exists.** The `.tbweight.csv` files the three
`configs/*-weighted.yaml` arms point at are absent; the source CSVs carry no
`weight` column; only `.randweight` (synthetic uniform, a fixture) exists. Every
weighted number in the draft — **99.3% included** — is `PROVISIONAL_WEIGHTED`, a
hard-coded dict at `scripts/analysis/v3/modules/headline_table.py:191` that the
table's own footnote flags as *"not a result of this pipeline."* Real weights are
enterprise-sensitive and unobtainable.

**The PNI evidence was synthetic.** All PNI work to date ran against
`datasets/pni/SYNTHETIC-us-carrier-hotels.csv` — 9 fabricated sites, peer ASN
15169 — and on **as02 only**.

Two things changed. We now have an approximate **real** PNI list for as01,
`datasets/pni/as01-us-pni.approx.csv` (14 US metros, AS20940, read off the
operator dashboard). And we have a surrogate for the weighted arm: **filter
targets to those sitting at PNI locations, keep all VPs.**

The surrogate is a *fake* cohort and is treated as one. Its value is that it is
the **best case** of traffic-weighted filtering — the optimistic bound on what
weighting could deliver — and it lets the entire script stack be built and
validated now, so that if real weights ever land the pipeline is ready.

Related: [20260523-leakage-free-cbg-eval-protocol](../20260523-leakage-free-cbg-eval-protocol/)
(finding F2 below reopens it), [20260906-v3-section-8-1-evaluation](../20260906-v3-section-8-1-evaluation/),
[20260907-v3-section-8-1-figures](../20260907-v3-section-8-1-figures/).

## Context

### What the surrogate already shows

as01's existing classification labels crossed with the new PNI list, before any
new code:

| cohort | n targets | n regions | sping acc | closest VP→seed p50 | min_inflation p50 |
| --- | --: | --: | --: | --: | --: |
| ≤10 km from PNI | 160 | 8 | **1.000** | 24.7 km | 1.384 |
| ≤20 km | 240 | 12 | 0.812 | 24.7 km | 1.419 |
| ≤40 km | 299 | 15 | 0.849 | 23.7 km | 1.384 |
| all | 399 | 20 | 0.637 | 23.6 km | 1.444 |
| >40 km | 100 | 5 | **0.000** | 18.4 km | 1.591 |

Links 2-3 reproduce strongly with no traffic data. Two cohort correlations,
reported as **characterization of this cohort** and nothing more:
ρ(d_pni, `tg_seed_nearest_vp_km`) = **−0.304**;
ρ(d_pni, `min_inflation`) = **+0.581**.

The cliff between ≤10 km (1.000) and ≤20 km (0.812) sits at the scale where a
PNI and its target stop sharing a cell — which is what Phase 2 decomposes.

### Three structural facts that constrain every number

**F1 — the independent unit is the site, not the IP.**

| dataset | target_ids | distinct coordinates | ids per region |
| --- | --: | --: | --: |
| as01 | 399 | **20** | 19.9 |
| as02 | 412 | **22** | 18.7 |
| as03 | 458 | **23** | 19.9 |

All replicas of a coordinate share a seed, share VP distances, and get an
identical Shortest-Ping verdict. Accuracy is quantized in 5% steps. The headline
`n = 1,269` is ~65 independent regions. Plausibly real — a CDN has many server
IPs per facility — but the statistics must cluster.

**F2 — every region spans all five folds.** `fold_assignments` splits
`target_id`, not coordinate: all 20 as01 regions appear in all 5 folds. Each test
fold's coordinates sit in all five training folds, so location-sensitive LTD
calibration is fit on the geometry it is scored on.

**F3 — the answer space is continental, not metro-granular.**

| dataset | occupied cells | margin_km p50 | nearest_seed_km p50 |
| --- | --: | --: | --: |
| as01 | **18** | 150.5 | 301.0 |
| as02 | 22 | 172.5 | 345.0 |
| as03 | 22 | 159.7 | 319.4 |

H3-4 *cells* are ~45 km, but only 18-22 are occupied, so decision boundaries sit
~300 km apart. "Near-perfect classification" over 18 classes 300 km apart is a
substantially weaker claim than §1's "metro-granular" framing.

### Decisions already taken

- Cluster by coordinate, keep all IPs. Never quote a target-level `n` as independent.
- Cohort defined by the threshold-free **same-cell rule**, not a km cutoff.
- Slice existing outputs; **no benchmark re-run**.
- Terminology: **region** = a distinct target coordinate and its IP replicas.

## Goals

1. Run the full PNI stack on as01 against the real 14-site list.
2. Decompose link 3 into a testable two-condition mediation and explain the
   10→20 km cliff with it.
3. Build every cohort script against the best-case surrogate so the stack is
   ready if real weights land.
4. Report F1/F2/F3 as first-class caveats with region-level statistics beside
   the target-level ones.
5. Leave explicit what the surrogate cannot test.

## Approach

Six phases. Phases 0-3 and 5 need no benchmark run. Phase 4 is parked pending
discussion. Phase 6 needs a run and is gated.

- **Phase 0** — region ids, region-level accuracy with cluster-robust CIs, the
  fold-overlap statistic. New `modules/places.py`; `place_id` → `region_id`
  emitted from `build-proximity`.
- **Phase 1** — the five PNI commands on as01 with the real list. No new code.
- **Phase 2** — new `modules/pni_mediation.py`, command
  `breakdown-pni-mediation`. One boolean per target:
  **A** `pni_in_tg_cell` = `nearest_seed(tg_nearest_pni) == tg_seed`.
  **Argmin rule only — no half-gap variant, no distance threshold anywhere.**
  Report **accuracy split by A**, at target and region level.

  Given A, `correct ⟺ B` exactly (where B = `nearest_seed(sping_vp) ==
  nearest_seed(PNI)`), so the A=true row's accuracy *is* B's success rate and B
  needs no column of its own. The A=false row's accuracy is two errors
  cancelling. The draft's identity "% of PNI in TG cell == sping accuracy" holds
  iff the A=true row reads 1.000 — which the split measures rather than assumes.
- **Phase 3** — the cohort *is* condition A. Score all six variants split by A
  over the same strata. Best-case framing throughout.
- **Phase 4** — build the draft's **1a** (d(VP,TG) vs RTT scatter, weighted
  stacked on mesh) and **1b** (Pearson r of minRTT with d(VP,TG), by dataset) as
  real commands. They **cannot** be interpreted on the surrogate — real weighting
  prunes *flows*, the cohort prunes *targets* and keeps all 134 VPs, so the
  (distance, RTT) cloud cannot concentrate. The cohort is a **dev fixture** to
  exercise the code path; the comparison waits for real weights. **No metric
  substitution** — `min_inflation` is not a stand-in for 1b at this stage.
- **Phase 5** — a short subsection stating what remains unverifiable.
- **Phase 6** — *deferred, needs a run.* Re-cut folds by region (F2).

## Caveats

- **The cohort is not traffic weighting and must never be labelled so.** It
  selects on target position relative to PNIs, not on traffic. Name it the
  **PNI-co-located cohort**. It is the optimistic bound, not a measurement.
- **Do not over-read the cohort.** It is fake and it is 20 regions. Treat every
  cohort number as pipeline validation, not as a finding. Only threshold-free
  statistics measured on the full as01 mesh may be quoted as results.
- **No distance thresholds in reporting.** Splits are by the A binary. Distance
  bands were rejected: they invite readings the data cannot support.
- **Range restriction will confound 1b.** A subset compresses the x-range, which
  attenuates Pearson r regardless of routing quality. Emit the **x-range** and
  **residual RMSE** beside r so a lower TW r can be diagnosed as attenuation
  rather than misread as worse routing.
- **Mark dev-fixture outputs in metadata.** Record `flow_filtered: false` in the
  cohort's `meta.json`, mirroring the `pni_site_list_is_synthetic: true`
  convention, so 1a/1b artifacts built from it are self-labelling.
- **Reporting accuracy on a condition-A cohort is conditional.** `P(correct | A)`
  is close to `P(B | A)`. That is a decomposition, not a headline.
- **The surrogate is biased one way, which is the safe way.** Real weighting
  prunes low-traffic *flows* (`filter_weighted_flows.py` is flow-level), thinning
  each target's VP set. The surrogate filters targets only and keeps all flows,
  so CBG retains every constraint and is at its *best*. If CBG still trails
  Shortest-Ping here, it would trail further under real weighting.
- **Effective n is tiny.** 20 regions for as01, 8 in the ≤10 km cohort. Any
  as01-only PNI claim rests on ~20 independent observations.
- **The PNI list is approximate.** Metro identity read from a screenshot;
  coordinates substituted from known carrier hotels. CONUS-cropped (dashboard
  shows 94.3% of traffic), so an "off-list" verdict may mean off-screen. Detroit
  has no target within 50 km and may be a misread.
- **Expect feasibility selectivity to fall.** 14 real sites admit larger feasible
  sets than 9 synthetic ones, so `singleton_share` should drop from as02's 63.3%.
  Stating this before running it is the honest order.
- **Do not carry as02's PNI numbers as priors** — different peer, synthetic list.
- **Naming collision to watch:** "region" is also used for answer-space clusters
  in prior work and for `pni_region` (US state) in the PNI CSV. Here it means a
  distinct target coordinate. Keep `region_id` unambiguous in schema docs.
- `PROVISIONAL_WEIGHTED` stays flagged or is deleted. The surrogate must not be
  used to launder 99.3%.
