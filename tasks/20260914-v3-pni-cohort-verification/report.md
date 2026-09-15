# §8.1 PNI Cohort Verification (v3) — Report

**Status**: In Progress — Phase 0 complete on all three runs; Phase 1 as01 only
**Created**: 2026-09-14
**Last Updated**: 2026-09-14

## Summary

Phase 1 is run. Every failure recorded against the synthetic as02 fixture
reverses on as01's real 14-site list, and the RTT-side feasibility exclusion
(draft figure 2a) now has a measured, null-calibrated answer. The work was scoped
after establishing that §8.1's weighted numbers are a placeholder, that the
prior PNI evidence was synthetic, and that three structural properties of the
operator datasets constrain every number the section reports.

## Findings

### Pre-existing state of §8.1's evidence

- **No traffic-weighted data exists.** `.tbweight.csv` absent; source CSVs carry
  no `weight` column; only `.randweight` (synthetic uniform fixture). No
  `*-weighted` run under `outputs/benchmark/v2/` or `outputs/analysis/v3/`.
- **99.3% is hard-coded.** `PROVISIONAL_WEIGHTED` at
  `scripts/analysis/v3/modules/headline_table.py:191`, flagged by the table's own
  footnote as not a pipeline result.
- **Prior PNI work was synthetic and as02-only.** 9 fabricated sites, peer ASN
  15169, `pni_site_list_is_synthetic: true`.
- On that fixture the routing story did **not** hold: strategy verdict `mixed`
  (argmin 38.8%, `direct_no_pni` 26.6%, `tg_nearest` only 15.9%);
  `sel_pni_is_tg_nearest_share` 45.2%; `routing_tg_nearest` r² 0.795 vs air 0.835
  (24.7% residual variance *added*). Only the self-fulfilling `routing_selected`
  axis beat air (+3.4%). Interaction ρ = +0.337 for `routing_selected` but
  **−0.060** for the `tg_nearest` guard.

### Phase 0 results — the structural facts, now measured by the pipeline

`modules/places.py` is in, `build-proximity` emits `region_id`, and the three
reporting tables carry region-level numbers. Full suite 897 passed.

**F1 reproduced exactly** — 20 / 22 / 23 regions over 399 / 412 / 458 targets.
One addition: the distinct-coordinate count is **identical at 2 through 6
decimal places** on all three runs. The replicas are not merely close, they are
byte-identical, so region identity is exact equality and the rounding parameter
carries no assumption. `region_count_is_rounding_stable` records this per run.

**F2 is worse than recorded.** The plan noted complete fold leakage for as01.
It holds for **all three runs**: `share_regions_in_all_folds` = 1.0 with
`folds_per_region` min = p50 = max = 5 on as01, as02 and as03. Not one region
anywhere is confined to a single fold.

**F3 reproduced exactly** — 18 / 22 / 22 occupied cells; `margin_km` p50
150.5 / 172.5 / 159.7; `nearest_seed_km` p50 301.0 / 345.0 / 319.4.

**New, and it matters for Phase 2:** as01 has **20 regions but 18 seeds**. Two
pairs of regions share a cell, so those four sites are unresolvable by
construction — no method can separate them at h3-4, and condition A cannot be
diagnostic there.

#### What the region columns change in the published tables

The headline row now reads **n = 1,269 targets / 65 regions**. The taxonomy
table shows the strata are far smaller than they look:

| term | n_targets | n_regions | sping | Octant-Hull (target / region) |
| --- | --: | --: | --: | --: |
| `selection_miss` | 145 | **8** | 0.000 | 0.614 / 0.556 |
| `selection_hit` | 254 | **13** | 1.000 | 0.795 / 0.790 |
| `geometry_only` | 0 | 0 | — | — |

Octant-Hull's headline 0.614 on `selection_miss` — the stratum §8.2 calls the
CBG opportunity — is four or five sites out of eight.

#### The consequence: every published interval is 3-5x too narrow

The point estimates survive — replica counts are near-equal, so
`region_accuracy` matches `target_accuracy` to a thousandth. **Precision does
not.** Clustered-bootstrap intervals over regions against naive binomial
intervals over targets, top-1:

| run | method | accuracy | clustered 95% CI | naive binomial CI | width ratio |
| --- | --- | --: | --- | --- | --: |
| as01 | Shortest-Ping | 0.637 | [0.425, 0.838] | [0.587, 0.684] | **4.3x** |
| as01 | Octant-Hull | 0.729 | [0.570, 0.880] | [0.683, 0.772] | 3.5x |
| as02 | Shortest-Ping | 0.369 | [0.184, 0.550] | [0.322, 0.418] | 3.8x |
| as02 | Octant-Hull | 0.650 | [0.508, 0.824] | [0.602, 0.697] | 3.4x |
| as03 | Shortest-Ping | 0.432 | [0.217, 0.652] | [0.386, 0.479] | **4.7x** |
| as03 | Octant-Hull | 0.502 | [0.330, 0.683] | [0.455, 0.549] | 3.8x |

#### Only as02 supports "Octant-Hull beats Shortest-Ping"

Comparing two independent intervals is the wrong test for methods scored on the
same targets, so this is the **paired** clustered bootstrap: per-region
difference in correctness, resampled over regions, 20,000 draws.

| run | Octant-Hull − Shortest-Ping | clustered 95% CI | excludes 0 |
| --- | --: | --- | --- |
| as01 | +0.092 | [−0.135, +0.328] | **no** |
| as02 | +0.299 | [+0.055, +0.532] | yes |
| as03 | +0.070 | [−0.126, +0.272] | **no** |

This is the powerful version of the test — pairing removes between-target
difficulty — and two of three datasets still fail to separate. as01's headline
72.9% vs 63.7% is nine points over **twenty sites** and is consistent with no
difference. Only as02's 29.9-point gap survives.

The paper's per-dataset ordering claims need this attached, or they need to
rest on the pooled row, where 65 regions buy back some power. It does not touch
§8.1's shortest-ping mechanism argument, which is about *why* a rate is what it
is rather than about separating two rates.

**An anomaly worth chasing:** 8 + 13 = 21 > 20. One region appears in both
terms, so `has_proximate_sping_vp` is **not** a pure function of the
coordinate — the shortest-ping VP differs between IP replicas at one site,
which only RTT can cause. Verified: exactly **one** region of 20 splits, 15
`selection_hit` to 5 `selection_miss`, giving `region_homogeneity` = 0.95.
Small, but it means region-level verdicts are not always unanimous, and it is
the column that says so.

### Phase 0 extended to as02 and as03 (2026-09-15)

Run by subagent, key numbers independently re-derived. F1/F2/F3 confirmed with
no disagreement: 22 / 23 regions, complete fold leakage on **all five combos**
per run (min = p50 = max = 5 folds per region), 22 / 22 occupied cells,
`margin_km` p50 172.487 / 159.688, and `margin_km == nearest_seed_km / 2` on
every row. as03's 23 regions over 22 cells means two coordinates share a cell
(as01 is worse at 20 over 18).

#### Every interval is 2.5-4.7x too narrow, in all 36 cells

`ci_is_underpowered` is True in every (run, method, top-N) cell — no run
reaches 30 regions. No clustered interval was narrower than its naive
counterpart anywhere.

One ratio exceeds its own design-effect heuristic and is worth keeping rather
than smoothing: as03 Shortest-Ping top-1 is **4.692** against
`sqrt(458/23) = 4.462`. That cell has homogeneity 1.000, so the clustered
problem is exactly a binomial on 23 items, and a bootstrap percentile interval
on 23 Bernoulli draws is wider than sqrt-scaling a Clopper-Pearson interval on
458 predicts. `sqrt(n/R)` is an approximation, not a bound.

#### as03 supports no pairwise method claim at all

All 15 pairwise paired comparisons per run, top-1, one shared region-resample
matrix per run, 20,000 draws, **unadjusted marginal 95% intervals**:

| run | pairs excluding zero | of | expected under the global null |
| --- | --: | --: | --: |
| as01 | 7 | 15 | 0.75 |
| as02 | 5 | 15 | 0.75 |
| as03 | **0** | 15 | 0.75 |

as01 and as02 sit far enough above 0.75 that those finding *sets* survive any
reasonable multiplicity correction. **as03 separates on nothing.** Octant-Hull
over Vanilla — the largest gap on the other two runs — is 19.8 points at target
level (0.502 vs 0.308) with clustered CI **[−0.434, +0.048]**. Independently
reproduced.

as02's five are both Octant arms over Vanilla, and Octant-Hull over
Shortest-Ping, SoI and Spotter (Shortest-Ping − Octant-Hull = −0.299,
CI [−0.533, −0.048]).

**A tie convention that needs stating in the paper.** Shortest-Ping − SoI has
an interval touching exactly 0.000 on all three runs. One method weakly
dominates the other target-for-target (as01: 0 targets where SP wins and SoI
loses, 10 the other way; as02 1/0; as03 0/3), so the bootstrap difference
distribution is one-signed and its far percentile lands exactly on the
boundary. Counted here as **not** excluding zero (strict `lo > 0 or hi < 0`),
which is the conservative reading. The permissive convention gives 8 / 6 / 1
instead of 7 / 5 / 0. The differences are 0.025 / 0.002 / 0.007, so nothing
substantive turns on it, but the convention should be stated rather than left
to the reader.

#### New finding: the fitted variants are not functions of the coordinate

Not anticipated by the plan. `region_homogeneity` — the share of regions whose
~20 IP replicas all receive the same verdict — separates the methods sharply:

| method | as01 | as02 | as03 | regions split (as03) |
| --- | --: | --: | --: | --: |
| Shortest-Ping | 0.950 | 0.955 | **1.000** | 0 of 23 |
| SoI CBG | — | 1.000 | 0.957 | 1 of 23 |
| Vanilla | — | 0.864 | 0.783 | 5 of 23 |
| Spotter | — | 0.545 | 0.696 | 7 of 23 |
| Octant-Hull | — | 0.545 | 0.522 | 11 of 23 |
| Octant-Spline | — | 0.455 | **0.348** | **15 of 23** |

Shortest-Ping and SoI are near-pure functions of the coordinate, as the design
predicts. The fitted variants are not: probe a site through twenty of its own
addresses and Octant-Spline returns a different class for **15 of 23 as03
sites**. The ground truth is identical across those addresses by construction —
same coordinate, same seed, same VP distances — so the only input that varies
is the per-address RTT vector.

Two consequences, and they point in opposite directions:

* **A reliability defect the aggregate hides.** An operator asking "which metro
  is this prefix in?" gets an answer that depends on which address in the
  prefix was probed. No accuracy number in the paper exposes this; it is a
  variance property, not a bias property, and it is invisible at target level.
* **It does not rescue the effective-n argument.** Replicas give partially
  independent *measurements*, but the site remains the unit of generalization,
  so the clustered interval stays the right one for "accuracy at a new site".

It also changes what `region_accuracy` *means* per method. Where homogeneity is
1.000 it is a mean of verdicts; at 0.348 it is a mean of per-site success
rates. Both are defensible, they are not the same quantity, and the column
should not be read across methods without homogeneity beside it.

#### Caveat found in `region_rate`, now documented at the source

Strata subdivide regions, so inside a stratum a region can contribute one
address while its other nineteen sit elsewhere — and the cluster mean weights
that sliver like a whole site. as02 top-1 `selection_hit` with SoI CBG:
`accuracy` 0.993 (151/152) against `region_accuracy` 0.889, because eight
regions sit at 1.000 and one region contributes a single wrong address at
weight 1/9. Verified directly. Both figures are correct for their own
estimator; what is wrong is reading the region figure as "the target rate with
replication removed". Over a whole run the two agree closely — max divergence
0.001 / 0.028 / 0.003 — because there every region contributes all its
addresses. Now stated in `region_rate`'s docstring.

#### Smaller items

* as02's 40 `geometry_only` targets are **exactly 2 regions**, so
  "Octant-Hull 0.975 on geometry_only" is two sites (20/20 and 19/20).
* as02's taxonomy split is the mirror of as01's: region 20 at 19 `selection_miss`
  to 1 `selection_hit`, and that single address is what drags the SoI stratum
  figure above. as03 is a clean partition — 10 all-hit regions, 13 all-miss,
  homogeneity 1.000.
* `taxonomy_of` reads only `has_proximate_vp` / `has_proximate_sping_vp`, so the
  taxonomy split count is one number per run, independent of method and top-N.
* as03 `region_accuracy_when_false` is NaN for the two left-column flags: no
  False cases exist, already flagged as `zero_variance`. Not imputed.

### Phase 1 is as01-only, and cannot be extended (2026-09-15)

Only two PNI site lists exist: `datasets/pni/as01-us-pni.approx.csv` (14 sites,
peer 20940, real) and `datasets/pni/SYNTHETIC-us-carrier-hotels.csv` (9 sites,
peer 15169, fabricated). as02's `pni-graph/` still carries the synthetic list,
so anything read from that directory today describes the fixture. as03 has no
PNI list at all.

The feasibility bars, the linearity reversal and the strategy verdicts
therefore cannot be replicated for as02 or as03 without a dashboard read per
peer. The stack would run unchanged if those lists land — the config blocks are
patterned and `pni.load_pni_sites` validates on load.

### Phase 1 results — as01 against the real 14-site AS20940 list

Run 2026-09-14, all five commands, reproducible from
`--config configs/as01-260728-260802.yaml` alone.

#### Figure 2a is answered: the RTT exclusion is near-deterministic

`build-pni-feasibility`, sping-VP pairs only (399 targets) — exactly the
`(sping VP, TG)` speed-of-Internet violation check the draft calls for:

| statistic | as01 (real, 14 sites) | as02 (synthetic, 9 sites) |
| --- | --: | --: |
| `singleton_share` | **0.837** | 0.633 |
| `singleton_equals_tg_nearest_share` | **0.837** | 0.585 |
| `contains_tg_nearest_share` | **1.000** | 0.816 |
| `empty_set_rate` unexplained by SOI | **0.000** | 0.136 |
| `best_rank_is_1_share` | **1.000** | — |

For 83.7% of targets the shortest-ping RTT is low enough that **only the
target's own nearest PNI lies inside the VP–TG ellipse** — every other site is
physically excluded at 2/3 c. The target's nearest PNI is feasible for *every*
target, and as02's 13.6% "served off-list" residue vanishes entirely, which is
evidence the residue was a fixture artifact rather than a property of operator
serving.

`n_feasible` for sping pairs is bimodal — p50 1, p75 1, p90 13 — mirroring the
bimodal `d_pni` distribution. The wide-feasible-set tail is the far targets.

**Permutation null (200 trials, same-size site lists from the real bounding
box).** The sping scope is outside the null on all four statistics, 0/200:

| statistic | observed | null mean | null p95 | P(null ≥ obs) |
| --- | --: | --: | --: | --: |
| `singleton_share` | 0.837 | 0.062 | 0.160 | **0.000** |
| `singleton_equals_tg_nearest_share` | 0.837 | 0.047 | 0.148 | **0.000** |
| `n_feasible` p50 | 1 | 0 | 0 | **0.000** |
| `best_rank_is_1_share` | 1.000 | 0.229 | 0.313 | **0.000** |

The null's median `n_feasible` is **0**: random sites in the same bounding box
usually cannot explain the sping RTTs at all. The real sites can.

**Honest limit — the discrimination is sping-only.** On all 53,262 pairs the
same statistics are indistinguishable from the null (`singleton_share` 0.072 vs
null mean 0.059, P = 0.23). A distant VP's ellipse is large enough to admit
almost anything. This is not a weakness of the claim: it is why the draft's
figure is specifically the *shortest-ping-VP* bar plot. Say so rather than
quoting the pooled number.

#### The as02 routing failure reverses

`compare-pni-linearity --holdout-only`, scored on the 26,605 held-out pairs:

| axis | r² | Δr² vs air | residual variance removed | implied km/ms |
| --- | --: | --: | --: | --: |
| air (control) | 0.655 | — | — | 134.4 |
| `routing_selected` (argmin, self-fulfilling) | 0.686 | +0.030 | 8.8% | 129.7 |
| **`routing_tg_nearest`** (honest guard) | **0.704** | **+0.049** | **14.2%** | 131.8 |

On the as02 fixture `routing_tg_nearest` r² was 0.795 **below** air's 0.835 —
it *added* 24.7% residual variance, and only the self-fulfilling axis beat the
control. Here the honest guard beats air by the larger margin **and beats the
self-fulfilling axis too** (+0.019, 5.9% residual removed). The rule that is
not fitted to the data explains more than the one that is. That is the
strongest available form of this result.

#### The PNI correction is between-target, not within-target

Three independent diagnostics show the same asymmetry, and it is not a
contradiction — it is structural:

| diagnostic | target-side / within-target | VP-side / between-target |
| --- | --: | --: |
| `detect-pni-strategy` verdict | `direct_no_pni` 32.6% (plurality, κ=0.47) | **`tg_nearest` 88.2%** |
| `compare-pni-linearity` | per-target routing wins 54.1%, median Δr ≈ 1e-05 | pooled r² 0.704 > 0.655 |
| interaction Spearman(Δr, `d_pni`) | `tg_nearest` **−0.027** | — |

The reason is arithmetic. For a fixed target, the `tg_nearest` predictor is
`d(VP, P_tg) + d(P_tg, TG)`, whose second term is a per-target **constant**.
Since `P_tg` sits p50 13.6 km from the target, the first term is nearly
`d(VP,TG)`, so within one target the two axes rank that target's 134 VPs almost
identically — the verdict is undetermined for 25.3% of targets *by
construction*, and near-tied for most of the rest. Between targets the constant
is exactly the extra propagation needed to reach the serving site, and it is
what the pooled fit and the VP-side vote both pick up.

**This changes how the result should be stated.** The PNI does not explain which
VP is closest to a given target; it explains why different targets sit at
different RTT offsets. Quote the VP-side 88.2% and the pooled r², not the
target-side plurality — and say why the target side cannot see it.

#### Accuracy by PNI distance (`breakdown-sping-pni`, top-1, h3-4)

Quartile bins are the command's own strata; reporting uses the A binary
(Phase 2) instead. Recorded here as raw artifacts:

| bin (km to nearest PNI) | n | sping | SoI CBG | vanilla |
| --- | --: | --: | --: | --: |
| 0.66–4.1 | 100 | 1.000 | 1.000 | **0.120** |
| 4.1–13.6 | 80 | 0.750 | 0.750 | 0.763 |
| 13.6–107.5 | 119 | 0.790 | 0.790 | 0.782 |
| 107.5–915.7 | 100 | **0.000** | 0.100 | 0.100 |
| pooled | 399 | 0.637 | 0.662 | 0.441 |

Shortest-ping and SoI CBG are identical in the first three bins and diverge only
in the fourth. Vanilla CBG's inversion in bin 0 — 12% on the targets sitting
*at* an interconnect, where shortest-ping is perfect — is unexplained and worth
a look; it is the opposite of the ordering §8.1 predicts.

### Quotable: threshold-free statistics on the full as01 mesh

These need no cohort and no distance cutoff. Computed from as01's existing
labels joined to the new 14-site PNI list.

- **`has_proximate_vp` = 1.000 for every target.** A class-resolving VP always
  exists, so VP–target geometric proximity is *saturated* and cannot explain any
  variation in accuracy. It is a control, not a claim.
- **ρ(d_pni, d(spingVP, TG's nearest PNI)) = −0.020.** The shortest-ping VP sits
  near the target's nearest interconnect regardless of how far the target is
  from it.
- **ρ(d_pni, d(spingVP, TG)) = +0.669.** The same VP's distance to the *target*
  tracks that separation directly.

Together these say the outcome is governed by **selection**, not availability:
a well-placed VP is always present and min-RTT declines to pick it whenever the
target is not at its serving site. This is the draft's bullet 2, and it is the
only one of the three bullets currently supported by real data.

**Caveat carried with all three:** as01 is 20 regions (F1), and the PNI list is
approximate. Co-location is shown geometrically; it is *not* yet shown that the
path crossed that site — that needs the RTT-side feasibility exclusion
(draft figure 2a, Phase 1), which has not been run.

### Dev-fixture only: the PNI-co-located cohort

Cohort numbers are **pipeline validation, not findings** — the cohort is fake
and rests on 20 regions. Earlier distance-binned readings from it (a "B ≤ 0.676"
bound, a "cliff between 10 and 20 km") are **withdrawn**: they used distance
thresholds that reporting has since rejected in favour of the A binary.

as01 target distances to nearest PNI: p50 13.6 km, p75 107.5 km, p90 668 km —
bimodal, with a hard gap between 40 km and 100 km. Useful for sizing the dev
fixture; not a result.

Draft figures **1a and 1b cannot be interpreted on this cohort** for a
structural reason: real weighting prunes *flows*, the cohort prunes *targets*
and keeps all 134 VPs, so the (distance, RTT) cloud cannot concentrate no matter
how true the mechanism is. They are built now and run against real weights later.

### F1 — the independent unit is the site, not the IP

| dataset | target_ids | distinct coordinates | ids per region |
| --- | --: | --: | --: |
| as01 | 399 | **20** | 19.9 |
| as02 | 412 | **22** | 18.7 |
| as03 | 458 | **23** | 19.9 |

Accuracy quantized in 5% steps; the 1.000 and 0.000 above are 8 and 5 regions.
Headline `n = 1,269` is ~65 independent regions. Metro counts in the ≤50 km
cohort are almost all exactly 20 (Atlanta 40, Chicago 39), consistent with ~20
IPs sampled per facility.

### F2 — every region spans all five folds

All 20 as01 regions appear in all 5 folds; `fold_assignments` splits `target_id`,
not coordinate. Location-sensitive LTD calibration is fit on the geometry it is
scored on. Reopens
[20260523-leakage-free-cbg-eval-protocol](../20260523-leakage-free-cbg-eval-protocol/).

### F3 — the answer space is continental, not metro-granular

| dataset | occupied cells | margin_km p50 | nearest_seed_km p50 |
| --- | --: | --: | --: |
| as01 | **18** | 150.5 | 301.0 |
| as02 | 22 | 172.5 | 345.0 |
| as03 | 22 | 159.7 | 319.4 |

Decision boundaries ~300 km apart despite ~45 km cells. This is the honest form
of the grid-resolution caveat and it weakens §1's "metro-granular" framing.

### Draft errors identified

- **Link 3's identity is conditional, and the condition is measurable.** "% of
  PNI in TG cell **should equal** sping accuracy" holds iff B (`nearest_seed(
  sping_vp) == nearest_seed(PNI)`) holds throughout. Given A, `correct ⟺ B`
  **exactly**, so splitting accuracy by the A binary makes the A=true row's
  accuracy *be* B's rate: the identity is then read off rather than assumed, and
  the A=false row's accuracy is two errors cancelling. B needs no column.
- **Draft line 142 misdescribes seed placement.** It says each class is "seeded
  at the centroid of the targets it merges"; `answer_space.py:278` places seeds at
  **cell centres**, and the loader rejects target-centroid seed files
  (`answer_space.py:429-435`). Fix the draft.
- **Bullet 2's scatter is degenerate** where the target sits at a site, since
  `d(spingVP,PNI) = d(spingVP,TG)` there. Replaced by the feasibility bar plot,
  which is an RTT-based exclusion and stays discriminating. Available today from
  as02: 63.3% singleton feasible, 58.5% singleton = TG's nearest, 13.6% empty
  (all unexplained by SOI violation, i.e. served off-list).
- **Answer-space confound for any future weighted arm.** Seed *positions* are
  grid-fixed, but the *occupied cell set* depends on the target set. A weighted
  subset yields a strict subset of mesh seeds, enlarging the surviving Voronoi
  cells — a monotone bias favouring every method. `cross.py` has no
  answer-space-sharing mechanism. Score weighted targets against the **mesh**
  seed set.

## Conclusions

_To be filled on completion._
