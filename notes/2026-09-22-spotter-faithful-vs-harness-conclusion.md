# Spotter, faithfully implemented: what changed, how it was measured, what it means

**Date:** 2026-09-22
**Source paper:** Laki et al., *Spotter: A Model Based Active Geolocation Service*,
IEEE 2011 — `papers/references/`. Sections cited: III-A, III-B, IV-B, IV-C, V-A-2.
**Status:** conclusion note. Consolidates four investigations; supersedes none of them.

## The Spotter document set

Read in this order:

| note | stage audited | question it answers |
|---|---|---|
| [2026-05-17-spotter-normality-check.md](2026-05-17-spotter-normality-check.md) | LTD | Does the pooled landmark-independent normal hold on RIPE Atlas? (No on consumer probes; approximately yes anchor-to-anchor.) |
| [2026-09-18-spotter-normality-operator-mesh.md](2026-09-18-spotter-normality-operator-mesh.md) | LTD | The Fig. 3 panels and landmark test on the operator↔hypergiant meshes. |
| [2026-09-19-normal-dist-wrong-for-operator-hypergiant.md](2026-09-19-normal-dist-wrong-for-operator-hypergiant.md) | LTD | *Why* the pooled normal is the wrong model here — four independent reasons R1–R4. |
| [2026-09-18-spotter-mtl-fidelity-gap.md](2026-09-18-spotter-mtl-fidelity-gap.md) | **MTL + CTR** | The full audit, defect-by-defect, plus every measurement this note summarises. |
| **this note** | — | What was built, how it was tested, and what to conclude. |

The division of labour matters: the first three are about **Spotter's model**; the
fourth is about **our implementation of Spotter**. Conflating them is the specific
error this work corrects.

---

# 1. What "faithful" changed

## 1.1 What the previous implementation was

`spotter_cbg` — the variant that produced every published Spotter number — is
**Spotter's LTD inside Octant's geometry**:

```
normal_dist  →  planar_annulus_weighted  →  monte_carlo_medoid
```

`normal_dist` is faithful. Everything after it is not. Four specific departures,
each traced in the [fidelity-gap note](2026-09-18-spotter-mtl-fidelity-gap.md):

| # | departure | where | why it is not Spotter |
|---|---|---|---|
| D1 | **inclusion pruning** — drop any disk engulfing another | [`filter_redundant_outer_disks`](../scripts/framework/geometry.py#L273-L310) | Under a product of densities no constraint is non-binding; every landmark contributes a factor everywhere. |
| D2 | **hard AND** — intersect outer disks, subtract inner disks | [`compute_feasible_region_unweighted`](../scripts/libs/octant/octant_geolocation.py#L278-L292) | §IV-C's selling point is the *opposite*: "less prone to measurement errors" than CBG/Octant's strict constraints. Near-landmark space is low-density, never excluded. |
| D3 | **density-blind CTR** — reduce a hard polygon | `monte_carlo_medoid` | §III-B derives the estimate *from the surface*: argmax, distribution mean, or region centre. |
| D4 | **RTT-decay weight** `exp(−rtt/τ)` | [`planar_annulus_weighted`](../scripts/framework/v2/mtl/planar_annulus_weighted.py) | An Octant reliability heuristic. §IV-B's argument for one pooled model is precisely that per-landmark treatment is infeasible. RTT is already inside µ(d), σ(d), so it is double-counted. |

D1 and D4 have no counterpart in the paper at all. D2 inverts the paper's central
claim. D3 moves the estimator into a stage that has no density to estimate from.

## 1.2 The reduction that made the fix cheap

Eq. (2) with Gaussian `f_d`, at candidate position `x`:

```
log P(x) = Σᵢ [ −log σᵢ − (sᵢ(x) − µᵢ)² / (2σᵢ²) ] + const
```

`µᵢ = µ(dᵢ)` and `σᵢ = σ(dᵢ)` depend only on the **measured** RTTs, which are fixed
once the target is chosen. So `Σ log σᵢ` is constant in `x` and drops out:

```
argmax log P(x)  =  argmin Σᵢ zᵢ² ,   zᵢ = (sᵢ(x) − µᵢ) / σᵢ
```

**Spotter's MAP estimate is σ-weighted nonlinear least squares over two
parameters.** The HTM mesh of §V-A-2 serves the confidence-region and
visualisation outputs, not the point estimate. This is why the faithful path ends
up *cheaper* than the approximation it replaces — see §3.2.

## 1.3 What was built

Five pieces. All in the framework, all additive — no existing combo changes
behaviour.

**(a) σ survives to the MTL stage.** `Distance` gained optional
[`mu_km` / `sigma_km`](../scripts/framework/v2/types.py). Carried rather than
re-derived downstream because **the band is not invertible**: `lower_km` clamps at
0, `upper_km` is clipped by the 2/3·c envelope, and the half-width is `k·σ` for an
LTD-internal `k`. A density MTL computing `µ = (lo+hi)/2` would be wrong exactly
where the clipping bit. `SpotterRTTModel.predict_mu_sigma` exposes (µ, σ) under the
same refuse-to-extrapolate clamp the bounds path uses.

**(b) A third MTL family.** `DensityMTLMethod` + `DensityField` in
[`mtl/base.py`](../scripts/framework/v2/mtl/base.py). The distinction is not
cosmetic:

- annulus families answer *"which points satisfy every constraint?"* — a **set**
  question; one violation empties the answer;
- the density family answers *"how plausible is each point?"* — a **ranking**; a
  bad fit lowers a score.

Those fail differently, so the type system should not let a caller assume one and
receive the other. `MTLResult.density` is `None` for every geometric method, which
is what a density-aware CTR checks before refusing to run.

**(c) [`GaussianDensityMTL`](../scripts/framework/v2/mtl/gaussian_density.py)**
(`gaussian_density`). Evaluates `−½ Σ zᵢ²` per H3 cell — H3 rather than HTM because
the v3 analysis tree is already keyed on it. Accumulates one constraint at a time;
a global H3-4 pass with 130 VPs would otherwise be a 300 MB intermediate.
Coarse-to-fine: global pass at `coarse_resolution=2` (5,882 cells), then descent
into the top-64 cells' children with a one-ring neighbour margin, to
`resolution=4`. The pruning is the only approximation and it is recorded as one;
`coarse_resolution >= resolution` disables it.

**(d) One CTR** in
[`ctr/density_point.py`](../scripts/framework/v2/ctr/density_point.py):
**`density_argmax`**, §III-B's "the maximum" — the MAP up to the grid's own
resolution.

Four were built and measured. `density_mean` (probability-weighted spherical
mean), `density_region_center` (unweighted centre of the credible region) and
`density_mle` (the exact continuous MAP, `least_squares` on `min Σzᵢ²` seeded at
the argmax) were **removed** once argmax was chosen as the shipped estimator.
Their numbers are in §2.3 and Addendum 3.

Argmax was the one kept because it is the only one of the four **invariant to
how far the MTL truncated the field** — adding or dropping low-probability cells
cannot move the maximum — which is exactly the right property under
coarse-to-fine pruning. The cost is the grid's quantisation: an H3-4 cell is
~20 km edge, and measured against `density_mle` on the same objective that was
~21 km at p5 and ~20 km at p50 on as01. **That comparison is no longer
reproducible in-tree**; re-running it means re-adding the solver *and*
`DensityField.constraints`, which went with it.

**(e) `per_vp_offset`** on `normal_dist` — one additive RTT offset per VP, used
to test R4 (§3.5). **Removed.** It answered its question (δᵢ is real and stable
but regresses accuracy under a symmetric density), the two δ arms were retired
with it, and leaving an unused option on the deployed LTD invites accidental
use. Re-implement it against a skew-aware loss when that experiment is run; the
estimator is described in §3.5 and in
[R4's](2026-09-19-normal-dist-wrong-for-operator-hypergiant.md)
`spotter_assumption_breakdown.py`.

**Tests:** 31 new for the density stack (17 MTL, 14 CTR).
The two load-bearing ones assert the properties that make this Spotter rather than
Octant-with-σ: *one hostile constraint cannot empty the field*, and *the inner hole
is low-density rather than excluded*. Full framework suite: 181 passed.

---

# 2. The experiment

## 2.1 Design

**Three maintained Spotter arms**, isolating the geometry with the LTD held
fixed at `normal_dist`:

| combo | MTL + CTR | what it is |
|---|---|---|
| `spotter_cbg` | `planar_annulus_weighted` + `monte_carlo_medoid` | the incumbent — Spotter's LTD in Octant's hard geometry |
| `spotter_true` | `gaussian_density` + `density_mle` | faithful; exact continuous MAP of the paper's own objective |
| `spotter_true_grid` | `gaussian_density` + `density_argmax` | faithful; §III-B's literal grid reading, with a ~20 km cell floor |

plus the incumbent non-Spotter variants (`vanilla_cbg`, `million_scale_cbg`,
`octant_cbg_hull/spl`) for context. None of those four use `normal_dist`, so
they are unaffected by any LTD change and their numbers carry across revisions.

**Four exploratory arms were run and then retired** once they had answered their
question: `spotter_appr` / `spotter_appr_fil` (the max-count approximation, §2.2
and §3.8) and `spotter_cbg_delta` / `spotter_true_delta` (the per-VP offset,
§3.5-3.6). Their measurements are kept below as recorded history. They are no
longer in the config and **cannot be reproduced without re-adding them** — the
numbers stand, the arms do not.

**Dataset:** `as01-260728-260802-mesh` — AT&T × Akamai, 399 targets over 5 folds,
134 VPs all in AS7018 mainland US,
[config](../configs/as01-260728-260802-mesh.yaml). Every arm answers the same 399
targets, so all comparisons are paired.

**Why as01, and its limits.** It was the only mesh tree already populated with a
`spotter_cbg` baseline on reproducible folds (as02-mesh is empty, as03-mesh has
`vanilla_cbg` only; the published `as01/02/03` trees have all five combos but their
fold pinning is not recoverable — `run.json` records `source_kwargs: null`). It is
a **weak instrument**: 399 targets span only **20 distinct coordinates**, every
region appears in all five folds, and the classification metric quantises in 5%
steps. Treat the ordering as solid and the magnitudes as coarse.

## 2.2 Abandoned arm, and why it is worth recording

A Phase-1 arm (`spotter_appr`) was to test "drop the AND, keep the annuli, pick the
face contained in the most annuli" — expressible with no code change as uniform
weights plus `highest_weight_only`. It was **abandoned after 2h23m of CPU per
fold without completing one**, at ~73 s/target.

The profile is the useful part. With the real LTD bands (~1,500 km-wide annuli at
~3,000 km radius, so every boundary crosses every other):

| path | annuli | faces | union | polygonize | contains | total |
|---|---|---|---|---|---|---|
| annular, filter ON | 16 | 329 | 0.01 s | 0.02 s | 0.09 s | 0.11 s |
| annular, filter OFF | 131 | 33,563 | 0.39 s | 1.57 s | **71.42 s** | 73.38 s |

**97% of the blow-up is the containment loop, not the geometry** —
`sum(c.weight for c, a in annuli if a.contains(rep))` is 4.4M one-at-a-time Shapely
calls with no spatial index. Measured replacements on the same data: vectorised
`shapely.contains` 3.25 s (22×), `STRtree.query(predicate="contains")` **0.52 s
(137×)**. That is a standalone optimisation worth landing — it speeds up every
Octant variant — and is tracked separately from the Spotter question.

## 2.3 Results

Error distance in km, 399 targets pooled over 5 folds.

| combo | LTD + MTL + CTR | p5 | p25 | p50 | p75 | p95 | VPs used | mtl_ms | ctr_ms | MTL ok |
|---|---|---|---|---|---|---|---|---|---|---|
| vanilla_cbg | low_envelope + planar_circle + geo_centroid | 2.2 | 28.1 | **49.2** | 281.9 | 2144.4 | 16.3 | 73 | 0.3 | 292/399 |
| million_scale_cbg | speed_of_internet + planar_circle + geo_centroid | 2.2 | 25.3 | 50.3 | 247.6 | 2175.5 | 4.5 | 8 | 0.2 | 399/399 |
| octant_cbg_hull | bounded_spline + annulus_weighted + mc_medoid | 0.8 | 6.2 | 71.5 | 313.3 | 559.3 | 27.8 | 1906 | 522 | 399/399 |
| octant_cbg_spl | bounded_spline + annulus_weighted + mc_medoid | 0.9 | 6.3 | 79.9 | 345.9 | 694.8 | 25.6 | 1687 | 502 | 399/399 |
| **spotter_cbg** | normal_dist + annulus_weighted + mc_medoid | 208.9 | 383.0 | **527.1** | 935.8 | 2713.8 | 23.8 | 680 | 373 | 399/399 |
| spotter_true_grid | normal_dist + gaussian_density + density_argmax | 39.4 | 136.2 | 239.0 | 617.4 | 2523.6 | 128.2 | 121 | 0.3 | 399/399 |
| **spotter_true** | normal_dist + gaussian_density + density_mle | **18.6** | 140.9 | **218.6** | 609.1 | 2511.9 | 128.2 | 118 | 11 | 399/399 |

**These are the ORIGINAL-fit numbers**, i.e. the binned `np.polyfit` estimator
that `fit_mu_sigma` used when this comparison was run. The addendum at the end
of this note re-runs the three maintained arms under the monotone / log-σ fit
and supersedes the three Spotter rows. The four non-Spotter rows are unaffected.

### Retired exploratory arms (recorded history, not reproducible from the config)

| combo | p5 | p25 | p50 | p75 | p95 | what it established |
|---|---|---|---|---|---|---|
| spotter_appr_fil | 210.9 | 452.9 | 546.1 | 975.9 | 2717.7 | uniform weights are nearly inert → D4 was not load-bearing (§3.8) |
| spotter_cbg_delta | 116.9 | 255.9 | 410.4 | 801.3 | 2762.6 | δ **helps** under the annulus (−117 km) |
| spotter_true_delta | 67.8 | 176.6 | 273.2 | 558.0 | 2514.5 | δ **hurts** under the density (+55 km) → the interaction in §3.6 |

---

# 3. Key observations

## 3.1 The harness was costing Spotter 2.4× at the median, 11× at p5

> **SUPERSEDED — see the addendum.** These numbers were measured under the
> original binned `fit_mu_sigma`. Once µ is monotone and σ is fitted in log
> space, `spotter_cbg` p50 drops 527.1 → 209.4 and the geometry gap collapses
> from 308.5 km to 4.0 km. Most of what is attributed to geometry below is in
> fact the broken fit interacting with the annulus. Do not cite this section
> on its own.

**527.1 → 218.6 km p50** and **208.9 → 18.6 km p5**, from changing nothing but MTL
and CTR. Paired, `spotter_true` wins on **329/399 targets (82.5%)**, median gain
217.5 km. Every `spotter_cbg` number published so far understates Spotter by
roughly this much.

## 3.2 The faithful method is also ~6× cheaper

MTL 680 → 118 ms, CTR 373 → 11 ms; five folds in **16 seconds**. The polygon
machinery was more expensive than solving the exact problem, which removes the last
reason to keep the annular stack for Spotter. The filter existed to make the AND
both tractable and survivable; remove the AND and neither is needed.

## 3.3 `EMPTY_REGION` and `EXCLUSIVE_REGION` were never Spotter's failure modes

`spotter_true` reports **399/399 MTL success with zero errors**. Under a density
surface, "the truth falls inside some annulus's hole" is not an outcome that
exists. The previously recorded ~89–99% `EXCLUSIVE_REGION` rate for Spotter was a
property of our harness; the caveat is now attached to that finding. Octant's ~50%
stands — Octant really does impose hard annuli.

## 3.4 The root cause is adverse selection, not the AND alone

This is the mechanism, and it was the least expected result.

Over all **51,161** (target, VP) constraints, scored against whether the annulus
actually contains the true location:

| | n | contains truth | median \|z\| |
|---|---|---|---|
| **kept** by the circle filter | 9,487 | **37.8%** | 1.16 |
| **dropped** by it | 41,674 | **72.3%** | 0.58 |
| all | 51,161 | 65.9% | — |

The filter keeps 18.4% of constraints and they are **twice as likely to be wrong as
the ones it discards.** Structural, not luck: the rule drops disk `i` when
`radius_i > dist(i,j) + radius_j`, i.e. it keeps the **smallest** outer disks. A
small outer disk is a small `µ + kσ`, i.e. a short predicted distance — and the
pooled model systematically **under-predicts** here (worked example: a 6.78 ms VP
genuinely 481.5 km away receives the band [243, 297] km). *Smallest disk* and *most
under-predicting* are the same set.

**The filter's selection criterion is correlated with the model's error direction.
It is a bias amplifier.** This is
[R2](2026-09-19-normal-dist-wrong-for-operator-hypergiant.md) (a model-family
property: `µ + kσ` is a percentile, not an envelope) multiplied by a harness
property — a composite neither note had alone.

The AND then converts selection into displacement: on the worked target, **6 of the
8 surviving annuli exclude the truth**, each a veto, so the region is forced
somewhere the truth is not and the CTR faithfully reports the middle of the wrong
place. Under the density product that same constraint contributes `z ≈ 7.8` — a
large penalty the other 125 constraints outvote.

Three compounding effects, all removed at once: adverse selection (128.2
participants vs 23.8, and the discarded 41,674 were the accurate ones), veto →
penalty, and σ restored as a continuous weight rather than a binary keep/drop proxy.

## 3.5 R4's per-VP offset is real, stable — and makes geolocation worse

*(from the retired `spotter_*_delta` arms)*

The δᵢ experiment **refuted** the hypothesis that landmark-independence was the
binding constraint.

Not in doubt: the estimator recovers known synthetic offsets at corr 0.9997 (after
centring — δ is identifiable only up to a global constant, the mean being absorbed
by µ(d)), and δᵢ is extremely stable across folds (pairwise corr **0.995–1.000**,
drift 0.08 ms against a between-VP spread of 3.34 ms). Exactly the reproducible
topological property R4 describes.

Yet under the faithful geometry it costs 55 km at p50. Per-constraint accuracy on
held-out folds explains why:

| | median bias (truth − µ) | median abs err | median \|z\| | within 1σ |
|---|---|---|---|---|
| pooled | **+39.1 km** | 289.6 km | **0.666** | 0.660 |
| + δᵢ | **+59.2 km** | **277.3 km** | 0.719 | 0.661 |

δᵢ does what it was asked — **reduces scatter** (−4%) — while **increasing
systematic under-prediction by 20 km**. The cause is
[R1](2026-09-19-normal-dist-wrong-for-operator-hypergiant.md): δᵢ is fitted by least
squares against a residual with skew **−0.75**, so squared loss chases the heavy
left tail and shifts µ down for the typical case. Multilateration keys on the
systematic component, not the scatter — a shared downward bias shrinks every ring
together and drags the density peak toward the VP cloud.

**Consequence for the R4 note's plan:** its "what would settle it" lists #2 (add
δᵢ) before #3 (a skewed or one-sided density). Measured, **#2 alone is
counterproductive because its estimator inherits the defect #3 names.** The
dependency runs the other way: *fix R1 first; δᵢ is only safe under a
correctly-shaped density.* Re-fitting δᵢ under a skew-aware or quantile/envelope
loss is the experiment that tests R4 properly. The present result is evidence about
the **estimator**, not about whether R4 binds.

## 3.6 Geometry and calibration interact — the annulus arm is not a valid instrument

*(from the retired `spotter_*_delta` arms)*

p50, km:

| | pooled | + δ | δ effect |
|---|---|---|---|
| **annulus** | 527.1 | 410.4 | **−116.7 (helps)** |
| **density** | 218.6 | 273.2 | **+54.6 (hurts)** |

**The sign of δ's effect flips with the geometry.** Run this experiment on the
annulus alone and it reports "δ helps by 117 km" — clean, and the opposite of the
truth. This is the strongest single argument for having done the fidelity work
first, and the general lesson: **a mis-specified geometry cannot be used to
evaluate an LTD change.**

The annulus arm's improvement has **no established mechanism.** Two candidates were
tested and both refuted:

- *radius homogenisation weakening the filter's adverse selection* — no: δ slightly
  **reduces** participants (23.8 → 23.5) and **increases** outer-radius spread
  (CV 0.427 → 0.472);
- *downward bias shrinking the spurious inner hole, cancelling R3's outward push* —
  no: median inner radius moves only 1347 → 1332 km and the inner-hole miss rate
  goes **up** (17.5% → 18.6%); containment is flat (65.9% → 66.0%).

Working interpretation, untested: perturbing a mis-specified hard-constraint system
can improve it for reasons unrelated to the perturbation's merit. Anyone wanting to
use that 117 km should establish the mechanism first.

## 3.7 The grid costs about 20 km

`spotter_true_grid` (argmax cell) vs `spotter_true` (continuous MAP): 39.4 → 18.6 at
p5, 239.0 → 218.6 at p50. An H3-4 cell is ~20 km edge, so the gap is the grid's own
quantisation — and it lands in the same range as the accuracy thresholds the paper
reports at. Read the paper's literal grid reading as having a ~20 km floor.

Reassuringly, coarse-to-fine costs nothing: it produces the **identical argmax** as
a single global 288,122-cell pass on a unimodal field.

## 3.8 Uniform weighting is nearly inert — D4 did not matter

*(from the retired `spotter_appr_fil` arm)*

`spotter_cbg` → `spotter_appr_fil` moves p50 by 19 km (527 → 546) with identical
participants and timing. With `highest_weight_only`, containment-count differences
(≥1) dominate the within-count weight spread. So of the four departures, **D4 was
the one that did not bite**; D1 and D2 were load-bearing.

## 3.9 Spotter still loses to vanilla — and that part was never an artifact

Faithful p50 **218.6 km** against vanilla's **49.2**. The fidelity fix changes the
*magnitude* of Spotter's disadvantage, not its direction.

Why, precisely: the improvement is concentrated where the geometry was already
good, and absent where it was not —

| closest-VP RTT quartile | median min-RTT | spotter_cbg p50 | spotter_true p50 | gain |
|---|---|---|---|---|
| closest | 0.9 ms | 524.4 | 158.4 | 232.3 |
| near | 1.5 ms | 492.9 | 178.9 | 341.7 |
| far | 2.6 ms | 463.5 | 159.4 | 309.4 |
| farthest | 59.1 ms | 2455.8 | 2262.2 | 189.5 |

The first three quartiles have an effectively co-located VP and the harness still
returned ~500 km. That was information being destroyed on *easy* targets. The
farthest quartile barely moves, and p95 barely moves overall (2713.8 → 2511.9):
where no VP is near, the binding error is the **LTD's**. A single pooled (µ, σ) is
simply the wrong distance model for those paths — R1 through R4. **Geometry cannot
rescue a wrong distance model.**

And the standing explanation for vanilla's win is R4's mirror image: `low_envelope`
fits its bestline **per VP**, absorbing δᵢ into that VP's own intercept, so it never
makes the exchangeability assumption at all.

*(Caveat: vanilla's percentiles cover only its 292/399 successful MTLs — a
survivorship effect flattering its tail — while every Spotter arm answers 399/399.)*

## 3.10 Three independent cross-validations

The LTD-side and MTL-side analyses were written separately and agree where they
overlap:

| quantity | [R-note](2026-09-19-normal-dist-wrong-for-operator-hypergiant.md) script | this work's audit |
|---|---|---|
| annulus contains truth | 67.5% | 65.9% |
| truth inside inner hole | 17.5% | 17.5% |
| per-VP bias magnitude | 139 km p50 | same order as the 169 km vanilla gap |

Different scripts, different row sets. Useful confidence in both.

---

# 4. What to do with this

**Reporting.** Stop citing Laki et al. for `spotter_cbg` results, and stop framing
the Octant head-to-head as isolating the LTD — it isolates LTD *within a shared
non-Spotter geometry*, which is a different and weaker claim. §8.1 inherits this.
`spotter_true` is the citable Spotter.

**Open, in priority order.**

1. **Repeat on eu-de.** as01's 20 distinct coordinates are too coarse to carry a
   verdict. eu-de has zero no-proximity failures, so geometry effects are not
   confounded by fleet coverage.
2. **Fix R1 before R4** — a skewed or one-sided density (or simply `log σ`, which
   also removes the `σ(d) ≤ 0` pathology), then re-fit δᵢ under a matching loss.
   Falsifiable prediction: the sign of δ's effect reverses.
3. **Land the `STRtree` fix** (137×) as its own change. Pure speedup to existing
   Octant variants, no semantics, and it makes the abandoned filter-off ablation
   affordable if it is ever wanted for the record.
4. **Establish the annulus/δ mechanism** (§3.6) or leave that 117 km uncited.
5. **A density MTL for a non-Gaussian family.**
   [`GaussianDensityMTL._log_density`](../scripts/framework/v2/mtl/gaussian_density.py)
   hardcodes `−½Σz²`, so as built it **cannot test R1**. Supporting a skewed
   likelihood means parameterising the MTL by distribution family, not just
   (µ, σ) — contained, now that the `Distance` plumbing exists.

**Code.** Branch `spotter-approx` (worktree `../cbg-spotter-approx`). Commit
`53ddb22` carries the density stack; the monotone / log-σ fit and the config
prune to three Spotter arms are later work on the same branch. The main
checkout's `configs/as01-260728-260802-mesh.yaml` still lists only the five
original combos — `spotter_true` / `spotter_true_grid` live on the branch.

## Reproduce

```bash
# density stack + the delta arms (add --configfile's combos first)
export PATH="$PWD/.venv/bin:$PATH"
R=outputs/benchmark/v2/as01-260728-260802-mesh/generic_csv/anchors_to_probes
python -m snakemake -s scripts/benchmark/v2/Snakefile \
  --configfile configs/as01-260728-260802-mesh.yaml -j 5 \
  $(for f in 0 1 2 3 4; do echo $R/fold_$f/spotter_true/targets.parquet; done)

# tests
python -m pytest scripts/framework/ -q          # 181 passed
```

Note that requesting the cell paths explicitly is required: if `summary.parquet`
already exists, `rule all` reports "Nothing to be done" and silently skips new
combos.

---

# Addendum (2026-09-22): monotone µ(d) and log-σ(d)

`fit_mu_sigma` now fits µ by **monotonicity-constrained least squares on the raw
pairs** (no binning) and σ **in log space** from that fit's residuals. Motivation
and design in `~/.claude/plans/zany-dazzling-reddy.md`; the three defects it
targets are no monotonicity guarantee, bin width as an unexamined bandwidth
knob, and σ(d) going non-positive.

**Implementation.** µ is a cubic whose derivative is written in the degree-2
Bernstein basis with non-negative weights, solved by plain `scipy.optimize.nnls`
— which yields monotone *and* non-negative in one shot, since `p(0) = c >= 0`
plus `p' >= 0` implies `p >= 0`. SLSQP with grid derivative constraints was
tried first and **failed to converge** on saturating data. Coefficients are
expanded analytically; recovering them by `np.polyfit`-ing the constructed curve
is lossy and silently broke monotonicity just outside the sampled range.
`p_sigma` was **renamed `p_log_sigma`** so the 16 hand-built fixtures and 10
evaluation sites failed loudly rather than computing `exp(50)`; a `sigma_at()`
helper is now the single place the log is undone.

## Verification

| check | result |
|---|---|
| Divergence (Comcast→G-Core US, n=401, rtt≤80 ms) | unconstrained µ spans **−557 → 157,007 km** over 0–300 ms; monotone spans **0 → 45,108**, and µ(0) = 0 |
| **Saturation cost, REAL data** (AS7922, n=147,699, dist→18,806 km, knee at ~100 ms) | rmse **1315.5 → 1376.6, +4.6%** — against the +24% synthetic worst case |
| as01 regression | rmse 638.2 → 645.5 (+1.1%); µ(0) −228.5 → **0.0**; σ > 0 over 0–300 ms |
| tests | **1443 passed** across framework / libs.spotter / analysis.v3 / benchmark.v2 |

The saturation penalty is **not uniform**: on the saturating corpus the
sub-100 ms region degrades 784.5 → 1089.8 rmse (**+39%**) while the plateau is
almost untouched (1823.2 → 1855.8, +1.8%). So the cubic's inability to sit flat
costs accuracy where geolocation cares most, on wide-RTT data. On the deployed
operator meshes (as01, no saturation) the penalty is +0.17% and the question
does not arise. **The spline follow-up should be scoped by RTT range, not
deferred indefinitely.**

## End-to-end, all three arms on the corrected fit

**This revises §3.1.** All seven combos re-run so nothing is stale; the four
non-Spotter arms are unaffected (they do not use `normal_dist`).

| combo | p5 | p25 | p50 | p75 | p95 | VPs used | mtl_ms | ctr_ms |
|---|---|---|---|---|---|---|---|---|
| vanilla_cbg | 2.2 | 28.1 | **49.2** | 281.9 | 2144.4 | 16.3 | 73 | 0.3 |
| million_scale_cbg | 2.2 | 25.3 | 50.3 | 247.6 | 2175.5 | 4.5 | 8 | 0.2 |
| octant_cbg_hull | 0.8 | 6.2 | 71.5 | 313.3 | 559.3 | 27.8 | 1906 | 522 |
| octant_cbg_spl | 0.9 | 6.3 | 79.9 | 345.9 | 694.8 | 25.6 | 1687 | 502 |
| **spotter_cbg** | **46.9** | **92.4** | 209.4 | 750.2 | 2703.3 | 20.8 | 456 | 209 |
| **spotter_true_grid** | 54.6 | 144.8 | **198.1** | 647.9 | **2619.7** | 133.5 | 114 | 0.3 |
| **spotter_true** | 51.3 | 144.3 | 205.4 | **636.2** | 2624.1 | 133.5 | 114 | 10 |

For reference, the same three arms under the **original binned fit**:
spotter_cbg 527.1, spotter_true_grid 239.0, spotter_true 218.6 (p50).

### The geometry gap largely dissolves once the fit is fixed

| | spotter_cbg p50 | spotter_true p50 | gap attributed to geometry |
|---|---|---|---|
| original binned fit | 527.1 | 218.6 | **308.5 km** |
| monotone + log-σ fit | 209.4 | 205.4 | **4.0 km** |

**§3.1's headline — "the harness was costing Spotter 2.4× at the median" — does
not survive.** Most of that 308 km was the *broken fit interacting with* the
annulus geometry, not the geometry alone. Fixing µ and σ improves the annulus
arm by 2.5× (527.1 → 209.4) and the density arms barely at all, because the
density path was already robust to the defect that hurt the annulus.

The mechanism is §3.4 read backwards. The circle filter keeps the **smallest**
outer disks, and the old µ was mis-shaped at the low-RTT end (negative,
clamped to 0) with σ negative below 5.51 ms. That made "smallest disk" strongly
correlated with "most wrong", which is the adverse selection §3.4 measured. A
monotone non-negative µ with positive σ orders the radii sensibly, so the same
filter now selects far less destructively — participants 23.8 → 20.8, yet p50
more than halves.

### What still holds, and what is now a wash

**Holds.**
* The density MTL is **~4× cheaper**: MTL 456 → 114 ms, CTR 209 → 10 ms.
* It uses **all** the VPs: 133.5 vs 20.8 participants.
* `EMPTY_REGION` / `EXCLUSIVE_REGION` remain impossible for it, and it reports
  399/399 MTL success.
* It is better in the **upper** tail: p75 636 vs 750, p95 2624 vs 2703.
* Everything in §1 about *fidelity* is unchanged — `spotter_cbg` is still not
  Spotter, whatever its error happens to be.

**Now a wash or reversed.**
* p50 is effectively tied (209.4 vs 205.4, ~2%).
* The annulus is **better in the lower tail**: p5 46.9 vs 51.3, p25 92.4 vs
  144.3. Hard constraints are sharp when they are right; a soft density pays a
  little precision for robustness. That is a sensible division and it only
  became visible once the LTD stopped dominating both.

**Reading for the paper:** argue the density stack on *fidelity, cost, and tail
behaviour*, not on median accuracy. The median claim was an artifact of a
defective fit, and it is the kind of number that would not have survived
review.

### Why the low tail moved: a compensating error was removed

Under the old fit σ(d) had a real root at 5.51 ms, so
`predict_mu_sigma` returned `None` for **2,088 of 53,262 rows (3.9%)** — and
those rows are exactly the **closest** VPs (rtt ≤ 5.51 ms, true distance median
69 km), the most geometrically informative ones. They were silently absent from
every density-MTL run. Meanwhile the old µ was *negative* below ~5 ms and got
clamped to 0, which happened to be a good prediction for a co-located target.

So the old fit was right at the low end for two wrong reasons: it dropped the
near VPs it could not model, and it clamped a negative µ to zero. The new fit
admits all of them and predicts honestly — µ(1 ms) = 63.6 km, µ(3 ms) = 189.0 km
against a true median of 48 km for rtt < 5 ms. It **over-predicts** there,
because a monotone cubic through the origin must start rising immediately, while
the data has a **last-mile floor**: RTT below a few ms is access latency, not
propagation.

That is the same phenomenon as R4's per-VP offset δᵢ, seen globally rather than
per-VP. The principled fix is an RTT intercept — µ(d) = f(d − δ) — not a shape
change to f. Which makes the ordering from
[2026-09-18](2026-09-18-spotter-mtl-fidelity-gap.md) sharper: **R1 (shape) is now
done for σ; the next binding constraint is the intercept/offset, and it should be
fitted against the corrected σ.**


---

# Addendum 2 (2026-09-22): the binning is gone from the API too

Follow-on cleanup, for new-version clarity. No behaviour change — `spotter_cbg`
209.4, `spotter_true` 205.4, `spotter_true_grid` 198.1 before and after.

**`fit_mu_sigma` now returns `(p_mu, p_log_sigma)`.** The trailing
`centers, mus, sigmas` bin summary is gone. It had already stopped being the
fit input; keeping it as a descriptive by-product only invited the reader to
think the curve came from the dots.

**`n_bins` and `min_per_bin` are deprecated and rejected.** They default to
`None` and raise a `ValueError` naming the offending key if anything passes a
value — in `fit_mu_sigma`, in `SpotterRTTModel.fit`, and in
`NormalDistLTD.__init__` so a stale `ltd_kwargs` fails at *composition* time
rather than mid-run. Silently ignoring them was the alternative, and it would
have left 61 config files asserting a bandwidth with no effect; deleting the
parameters outright would have surfaced as an opaque `TypeError`.

Removed from **60 YAML files (568 lines)**. The one surviving `n_bins` in the
tree is `configs/template.yaml`'s density-quantile knob, which is unrelated.
The sample-count gate that used to borrow `min_per_bin` is now
`MIN_FIT_SAMPLES = 4` (the cubic's requirement), and `bin_size_ms` /
`cutoff_min_points` stay — they configure the `cutoff_rtt` scan, not the fit.

**`deg_mu` stays** and is still validated `== 3`. It is arguably the next thing
to retire, since the Bernstein construction admits no other degree, but it does
document that mu is a cubic and removing it would touch the same 60 files
again.

**Figure consumers removed.** `figure_spotter_normality` loses the panel-(a)
bin dots, the `*_bin_fit.csv` artifact, the `n_bins_used` / `bin_counts`
manifest keys, the `--n-bins` / `--min-per-bin` CLI options, and the
`centers/mus/sigmas/counts` fields on `Fit`. Panel (a) keeps the scatter cloud,
the µ±σ curves and the 2/3·c bound — the cloud already shows the population the
curve is fitted to, which is what the dots were standing in for.

`test_the_bin_dots_belong_to_the_drawn_curve` became
`test_the_fit_carries_no_bin_summary`, which asserts the fields and the CSV
suffix are absent rather than consistent. 1443 tests pass.


---

# Addendum 3 (2026-09-22): one CTR, one density arm

`density_argmax` is now the only density-aware CTR. `density_mean`,
`density_region_center` and `density_mle` are deleted, and with them
`_spherical_mean`, the `scipy.optimize` dependency in the CTR package, and
`DensityField.constraints` — whose only reader was `density_mle`. A field
nothing reads is a field that goes stale.

`density_point.py` **218 → 66 lines**; its tests 185 → 88.

**The two density arms merge.** `spotter_true` was `gaussian_density +
density_mle` and `spotter_true_grid` was `gaussian_density + density_argmax`;
under one CTR they are the same pipeline, so `spotter_true_grid` is gone and
`spotter_true` now *is* the argmax arm. Verified identical to the retired grid
arm: p50 198.1 either way. Three Spotter combos become two.

| combo | p5 | p25 | p50 | p75 | p95 | mtl_ms | ctr_ms |
|---|---|---|---|---|---|---|---|
| vanilla_cbg | 2.2 | 28.1 | **49.2** | 281.9 | 2144.4 | 73 | 0.3 |
| octant_cbg_hull | 0.8 | 6.2 | 71.5 | 313.3 | 559.3 | 1906 | 522 |
| spotter_cbg | 46.9 | 92.4 | 209.4 | 750.2 | 2703.3 | 456 | 209 |
| **spotter_true** | 54.6 | 144.8 | **198.1** | 647.9 | 2619.7 | **74** | **0.3** |

`ctr_ms` drops 9.7 → 0.3 ms: argmax is an `max()` over a list where the MLE ran
a least-squares solve per target.

**What was kept despite looking dead.** `credible_mass`, `_credible_region`,
`DensityField.probabilities()` and `MTLResult.intersection` all survive even
though no density CTR reads the region any more:
`analysis/v3/modules/map_mtl.py` serialises `intersection` for the world-map
viewer, the benchmark writer records `mtl_intersection_kind` from it, and
`DensityMTLMethod`'s contract promises it so a density-blind CTR degrades
instead of crashing.

**What this costs.** The exact-MAP reference is gone, so the ~20 km
grid-quantisation figure above is now a recorded measurement rather than a
reproducible one. A test asserts the three retired names are absent from
`CTR_REGISTRY`, so a stale combo id fails at composition instead of silently
resolving.
