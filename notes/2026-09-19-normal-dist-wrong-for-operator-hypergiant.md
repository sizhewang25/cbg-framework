# Why a pooled normal is the wrong delay-distance model on operator↔hypergiant meshes

**Date:** 2026-09-19
**Source paper:** Laki et al., *Spotter: A Model Based Active Geolocation Service*, 2011.
**Script:** [scripts/libs/cbg_feasibility/spotter_assumption_breakdown.py](../scripts/libs/cbg_feasibility/spotter_assumption_breakdown.py)
**Measurements:** [2026-09-18-spotter-normality-operator-mesh.md](2026-09-18-spotter-normality-operator-mesh.md) (the Fig. 3 panels and the landmark test)
**Related:** [2026-09-18-spotter-mtl-fidelity-gap.md](2026-09-18-spotter-mtl-fidelity-gap.md), [2026-05-17-spotter-normality-check.md](2026-05-17-spotter-normality-check.md)

## Thesis

`f_d(s) = N(μ(d), σ(d)²)` — Spotter's delay-distance model — is a poor
approximator on operator↔hypergiant measurement meshes, and it fails for
**four independent reasons**. Three are properties of the model family and one
is our harness's. None of them is a fitting error: the fit is good by its own
criteria (`σ_z` 0.91–1.03 against the paper's 1.035) and still wrong for the
job.

The short version: geolocation needs a **one-sided bound** on distance, and a
Gaussian is a **two-sided description of a centre**. Those are different
objects, and on this data the difference costs ~15 percentage points of
containment even when the two happen to have the same radius.

## The setting

| | as01 | as02 | as03 |
|---|---|---|---|
| peer | AT&T × Akamai | AT&T × Google | AT&T × Netflix |
| rows (guarded) | 49,629 | 53,877 | 60,214 |
| VPs | 134 | 134 | 134 |
| target sites (distinct coordinates) | 20 | 22 | 23 |

All VPs are in AS7018, mainland US. The VP side is operator access/aggregation
infrastructure; the target side is hypergiant serving infrastructure. This is
the regime Spotter's assumptions are *least* likely to be safe in, and — because
it is a single ASN in a single country — also the regime where the naive
expectation is that they should be *most* safe. That expectation is what this
note is about.

---

## R1 — The noise is one-sided; the family is symmetric

**The assumption.** At a fixed RTT, distance is distributed symmetrically about
a mean.

**Why it cannot hold.** Every delay term is additive. Queueing, backhaul,
serialization, and routing detours only ever *increase* RTT relative to the
propagation minimum; nothing in a network makes a packet arrive before the
speed of light allows. Condition on an RTT and that asymmetry moves the
distance distribution one way only: the observed distance can be far *below*
what the RTT would permit, and cannot be above it. In `z` terms the left tail
is heavy and the right tail is bounded.

**Measured** (guarded pass):

| | as01 | as02 | as03 |
|---|---|---|---|
| skew of `z` | **−0.75** | **−0.57** | **−0.65** |
| excess kurtosis | +0.81 | −0.00 | +0.09 |

Negative skew on all three, in the exact direction the physics predicts, while
the kurtosis is unremarkable — so this is not "heavy tails", it is a
**one-sided** distribution being described by a symmetric family. A Gaussian has
one shape parameter too few to carry it, and the deficiency is structural: no
choice of `μ(d)`, `σ(d)` fixes a skew.

**Whose fault:** Spotter's model family.

## R2 — A conditional mean cannot stand in for an envelope

**The assumption.** `μ(d) + kσ(d)` is a usable outer constraint.

**Why it cannot hold.** CBG's bestline is fitted to lie *above* the
delay-distance points — it is a **bound**, and its radius grows wherever the
data at that RTT is worst. `μ + σ` is the ~84th percentile of the conditional
distribution *by construction*, whatever the tail is doing. The two can be the
same size on average and behave completely differently where it matters.

**Measured**, on identical rows, against `low_envelope`'s per-VP bestline:

| | as01 | as02 | as03 |
|---|---|---|---|
| containment, CBG disk | **99.8%** | 99.9% | 99.9% |
| containment, Spotter outer-only disk | 85.0% | 79.0% | 83.9% |
| containment, Spotter annulus (k=1) | 67.5% | 63.4% | 69.3% |
| Spotter outer radius ÷ CBG radius, p50 | **1.00** | 0.91 | 0.91 |

The last two rows together are the finding. **The Spotter outer radius is the
same size as CBG's on as01 and ~9% smaller on the others, yet it contains ~15
points fewer truths.** What is lost is not the envelope's height — it is its
*envelope-ness*, i.e. the property of adapting to the worst case at each RTT
rather than tracking the centre.

**Whose fault:** Spotter's model family — though it is only a defect because we
consume the model as a constraint. See "What is ours, not Spotter's".

## R3 — The inner edge is invented, and it costs more than the outer one

**The assumption.** A lower bound on distance given RTT is meaningful.

**Why it cannot hold.** There is none. `distance ≤ (RTT/2)·v` is a physical
constraint; there is no corresponding floor, because a target 1 km away can
show 50 ms on a bad route. A disk has no hole for exactly this reason. Turning a
density into an annulus manufactures a lower bound that the physics does not
license.

**Measured** at `k = 1.0`, which is what the mesh configs deploy
(`target_coverage` unset, so `k` stays 1):

| | as01 | as02 | as03 |
|---|---|---|---|
| truth inside the inner hole (`z < −1`) | **17.5%** | 15.6% | **14.6%** |
| truth beyond the outer edge (`z > +1`) | 15.0% | 21.0% | 16.1% |
| total miss | 32.5% | 36.6% | 30.7% |
| a *perfect* N(0,1) would miss | 31.7% | 31.7% | 31.7% |

Two things to read here. First, the invented inner edge loses as many truths as
the outer edge does, and more on two of three datasets. Second — and this is the
part that is easy to get wrong — **the aggregate miss rate is what a perfect
normal would give.** The pooled model is well calibrated on average. Its problem
is not the average.

**Whose fault:** ours for the annulus, Spotter's for offering a two-sided
description of a one-sided phenomenon.

## R4 — Landmark-independence fails, and the residual is not noise at all

**The assumption** (Sec. IV-B, the paper's central methodological move): one
pooled `(μ, σ)` describes every landmark, so no per-landmark calibration is
needed.

**Why it cannot hold here.** Landmark-independence is an **exchangeability**
claim, not a precision claim: it needs each landmark's deviation from the pooled
law to be a draw from one common distribution. A fixed single-ASN topology
guarantees the opposite. Each VP's route to each serving site is stable, so its
deviation is a *reproducible property of its position in the topology*, measured
~400 times, not noise that averages out. **Determinism does not make the pooled
model fit better; it makes the per-landmark component detectable.** A noisier
network would look more landmark-independent.

**Measured.** Two-way decomposition of `Var(z)` on (VP, target site):

| share of `Var(z)` | as01 | as02 | as03 |
|---|---|---|---|
| VP main effect | 10.8% | 30.7% | 14.9% |
| target-site main effect | 23.1% | 28.1% | 36.2% |
| interaction (per-path) | 62.5% | 41.4% | 44.4% |
| **within-cell — the only actual noise** | **4.3%** | **0.5%** | **5.1%** |

Knowing which VP pinged which site determines 95–99.5% of the standardized
residual. There is essentially no exchangeable component for the pooled law to
describe. Note also that the **target side carries as much or more structure
than the landmark side** — plausibly the hypergiant's serving topology, though
that is a hypothesis this data cannot confirm.

The per-VP part has a specific and simple shape: **a fixed additive RTT offset.**
Giving each VP one free scalar `δᵢ`:

| | as01 | as02 | as03 |
|---|---|---|---|
| `η²(VP)` before → after | 10.8% → **1.2%** | 30.7% → **1.4%** | 14.9% → **1.2%** |
| `corr(per-VP mean z, δᵢ)` | −0.93 | −0.99 | −0.97 |
| per-VP bias in km, p50 / max | 139 / 463 | 150 / 987 | 156 / 739 |

One scalar per VP removes ~90% of the landmark effect, and the correlation says
the per-VP mean residual essentially *is* that offset. The shape — a constant in
RTT, identical whichever target is pinged — fits either fixed access/backhaul
latency or a fixed detour to the VP's backbone ingress PoP; traceroutes would
separate them.

**This is also why the Q-Q slopes sit below 1.** `Var(pooled) = within +
between`, and the Q-Q slope is `σ_VP / σ_pooled`, so between-VP variance alone
forces most slopes down. Predicted `√(1 − η²)` = 0.944 / 0.832 / 0.923 against
observed median slopes of 0.897 / 0.812 / 0.901 — the slope deficit visible in
panel (c) is the landmark effect, arithmetically.

**Whose fault:** Spotter's claim. And note the irony: **CBG's `low_envelope`
fits its bestline per VP**, so it absorbs `δᵢ` into that VP's own intercept and
never makes this assumption. The paper argues against per-landmark calibration
on sample-size grounds ("only a few hundred points… technically infeasible");
with ~400 samples per VP and one scalar to estimate, that objection does not
apply.

---

## What is ours, not Spotter's

The paper never forms a hard region. Eq. (2) is a **product of densities** over
cells, `P(T ∈ H | L₁⋈d₁, …) = A_H ∏ᵢ ∫_H g_dᵢ^Lᵢ(τ)dτ`, and the estimate is the
maximum or mean of that surface, or the union of the most probable cells at a
confidence level. Nothing can be excluded; a badly-fitting landmark just
contributes a flatter factor.

So `μ ± kσ` — and therefore the empty-intersection and `EXCLUSIVE_REGION`
failures — are **our harness's translation**, not Spotter's method. `spotter_cbg`
runs Octant hard-annulus geometry (`planar_annulus_weighted` +
`monte_carlo_medoid`), as [2026-09-18-spotter-mtl-fidelity-gap.md](2026-09-18-spotter-mtl-fidelity-gap.md)
already records. Any argument of the form "k=1 is a 68% band, so intersecting N
of them fails" is a criticism of our adaptation and must not be attributed to
the paper.

What survives the correction: R1, R3 and R4 still bite in the true formulation,
but as **drift instead of exclusion**. Each VP's density is centred on the wrong
radius by `δᵢ`, so the product peaks in the wrong place; the skew means each
factor is the wrong shape. The estimate degrades smoothly rather than failing
— which is probably how the paper got away with all three.

## Why the four compound

They are not four independent nuisances; they push the same way.

1. R1 says the true conditional distribution has a long left tail — i.e. many
   pairs sit well below `μ(d)`.
2. R3 says the annulus removes exactly that region with its inner edge.
3. R2 says the outer edge is a percentile rather than a bound, so it cannot
   compensate by being generous.
4. R4 says the whole construction is displaced per VP by 139–156 km median,
   and that the displacement differs across VPs — so **no single global `k` can
   repair it**: a `k` wide enough for the worst VP (64% miss at k=1) is
   wastefully loose for the best (10%). At `k = 2` the per-VP miss still spans
   0–14%.

That last point is the practical crux. Heterogeneous landmarks cannot be fixed
by widening a shared band; they need a per-landmark parameter, which is exactly
what the pooled model exists to avoid.

## Scope, honestly

Supported: three peers, **one operator** (AS7018), one country, one week, ~54k
measurements each. The mechanism arguments (R1, R3) are general on physical
grounds. R2 is general to any comparison of a quantile against an envelope. R4
is measured here and is *expected* wherever topology is stable, but "operator↔
hypergiant networks" as a class is an extrapolation from three datasets sharing
an operator and a geography. A second operator would test it cheaply.

Also note the target side is ~20 distinct coordinates, so `f_d(s)` is a discrete
mixture here rather than a density — see the limits section of
[the measurement note](2026-09-18-spotter-normality-operator-mesh.md). That
bounds how strong any *normality* verdict can be; it does not affect R2, R3 or
R4, which are containment and between-group comparisons.

## What would settle it

1. **Implement the real Spotter MTL** — the MAP `argmin Σ((s − μᵢ)/σᵢ)²` over
   the density product, instead of the annulus. Until then every `spotter_cbg`
   number confounds "Spotter's model is wrong here" with "our geometry
   misrepresents Spotter", and R2/R3 cannot be cleanly attributed.
2. **Add `δᵢ` to `normal_dist`** — one scalar per VP, fit leakage-safe on the
   train folds. Cheap, and it tests R4 directly: if accuracy jumps, the pooled
   assumption was the binding constraint.
3. **A skewed or one-sided density** (or simply `log σ`, which also fixes the
   `σ(d) ≤ 0` pathology) — tests R1.
4. **A second operator** — tests the scope claim.

Run 1 and 2 together and the four reasons become separable, which is the only
way this note's attribution column stops being an argument and becomes a
measurement.

## Reproduce

```bash
cd /home/nuwinslab/workspace/atnt/cbg-framework

# All three meshes. One function per reason, printed as a table and written to
# JSON. Takes a few minutes per dataset: `envelope` fits a per-VP CBG bestline
# (134 LP solves) and `landmark` scans a 501-point offset grid per VP.
.venv/bin/python -m scripts.libs.cbg_feasibility.spotter_assumption_breakdown
# -> scripts/libs/cbg_feasibility/outputs/spotter_assumption_breakdown/breakdown.json

# One dataset, when iterating.
.venv/bin/python -m scripts.libs.cbg_feasibility.spotter_assumption_breakdown \
    --dataset as01 --out-dir /tmp/breakdown
```

Which function backs which reason:

| reason | function | key manifest keys |
|---|---|---|
| R1 one-sided noise | `shape()` | `skew`, `excess_kurtosis` |
| R2 bound vs quantile | `envelope()` | `containment_cbg_disk`, `containment_spotter_outer_only`, `outer_over_cbg_radius` |
| R3 the invented inner edge | `sides()` → `sides_k1`, `sides_k2` | `miss_inside_inner_hole`, `miss_beyond_outer_edge`, `per_vp_miss` |
| R4 landmark dependence | `landmark()` | `variance_decomposition`, `offset_model` |

The Fig. 3 panels, the Q-Q plot and R4's permutation test come from the
measurement note's command:

```bash
.venv/bin/python -m scripts.analysis.v3.cli plot-spotter-normality \
    --csv datasets/final/as01-20260728-20260802.mainland.sanitized.csv \
    --out-dir /tmp/spotter
```

Both read the dataset CSVs directly — no database, no prior benchmark run. The
breakdown script reuses `plot-spotter-normality`'s loader, fit and
`standardize`, so the two cannot report different numbers for the same input.
