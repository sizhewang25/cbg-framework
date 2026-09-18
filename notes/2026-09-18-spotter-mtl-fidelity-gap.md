# Spotter's MTL stage is not Spotter

**Date:** 2026-09-18
**Source paper:** Laki et al., *Spotter: A Model Based Active Geolocation Service*, 2011
(`papers/references/`), §III-B, §IV-B, §V-A-2.
**Companion note:** [2026-05-17-spotter-normality-check.md](2026-05-17-spotter-normality-check.md)
— that note audits Spotter's **LTD** claim (pooled landmark-independent normal) on our
data. This note audits the **MTL and CTR** stages, which that note did not touch.

## Summary

Our `normal_dist` LTD is a faithful implementation of Spotter's delay-distance model.
Everything downstream of it is Octant's geometry, and Octant's geometry contradicts the
property Spotter is *defined by*. The variants labelled `spotter_cbg*` are therefore
"Spotter's LTD dropped into a CBG hard-constraint stack", not Spotter, and their numbers
are not comparable to the paper.

Three distinct defects, in increasing order of severity.

## Defect 1 — inclusion pruning has no counterpart in Spotter

[`filter_redundant_outer_disks`](../scripts/framework/geometry.py#L273-L310) drops any
disk that fully engulfs another. Its own docstring is honest about the two roles: a
mathematical no-op for disk intersection (`A ⊇ B ⇒ A ∩ B = B`), but an explicit
**heuristic** for annular MTLs, where it additionally suppresses the engulfing
constraint's inner-disk veto. It was introduced to stop the Octant AS7018 NA collapse.

Under a product of densities there is no such thing as a non-binding constraint. Every
landmark contributes a factor at every point; a wide-RTT landmark with large `µ(d)` still
multiplies in a density that is *low* wherever the tight landmarks' rings are, and so
actively shapes the answer. Removing it changes the product.

It is on by default ([`planar_annulus.py`](../scripts/framework/v2/mtl/planar_annulus.py),
`enable_circle_filter: bool = True`) and set explicitly for the live Spotter variant
([north_america_as7018_final.yaml:204-219](../scripts/benchmark/v2/config/north_america_as7018_final.yaml#L204-L219)).
Note the `_nofil` in `spotter_cbg_nofil` refers to VP prefiltering, **not** to this
filter — a naming trap worth remembering.

## Defect 2 — the hard AND inverts the paper's central contrast

§IV-C is explicit about what distinguishes Spotter: *"the latter models produce strict
constraints around the landmarks, while Spotter's underlying probabilistic approach is
less prone to measurement errors."*

Our implementation gives Spotter strict constraints twice over, in
[`compute_feasible_region_unweighted`](../scripts/libs/octant/octant_geolocation.py#L278-L292):

```python
positive_region = reduce(lambda a, b: a.intersection(b), outer_disks)
...
result = positive_region.difference(unary_union(inner_disks))
```

The inner-disk subtraction is the worse half. In the paper, near-landmark space is *low
density*; it is never *excluded*. We turn `µ − kσ` into a hard veto.

**This invalidates a previously recorded finding.** Spotter's ~89–99% `EXCLUSIVE_REGION`
rate is not a property of Spotter — under a density surface, "truth falls inside some
annulus's hole" is not an outcome that exists. Likewise `EMPTY_REGION`: the AND makes one
under-predicting VP sufficient to empty the feasible region (the same brittleness already
recorded for `SphericalCircleMTL`). Both failure classes are harness artifacts that we
have been reporting as measurements of Spotter.

## Defect 3 — CTR is doing the density surface's job

§III-B derives the point estimate *from the density surface*: argmax, distribution mean,
or centre of the confidence region. We run `monte_carlo_medoid` / `geometric_centroid`
over a hard polygon carrying no density at all.

Combined with the earlier finding that Spotter's accuracy is essentially
"EXCLUSIVE-but-correct centroid snapping", the picture is that the CTR stage is currently
producing the behaviour the density surface is supposed to produce, and being credited for
it. Fixing MTL alone would leave the evaluation still not measuring Spotter.

A related interface fact: `σ` is already gone before MTL runs.
[`Distance`](../scripts/framework/v2/types.py#L24-L32) carries only `upper_km` /
`lower_km`, so `NormalDistLTD` collapses `N(µ, σ²)` to a band and keeps only σ's *width*,
discarding its *shaping* role.

## Also non-Spotter: the RTT-decay weight

[`planar_annulus_weighted`](../scripts/framework/v2/mtl/planar_annulus_weighted.py) sets
`weight = exp(-rtt_ms / weight_tau_ms)`. That is an Octant reliability heuristic. §IV-B's
entire argument is that per-landmark calibration is infeasible, *hence* one pooled model —
per-landmark RTT-decay weighting is precisely the thing Spotter is defined by not doing.
And RTT is already inside `µ(d)` and `σ(d)`, so it is double-counted.

## The exact objective is cheaper than the approximation

Worth writing down explicitly, because it reframes the whole implementation question.
Spotter's Eq. (2) with Gaussian `f_d` gives, for a candidate position `x`:

```
log P(x) = Σᵢ [ −log σᵢ − (sᵢ(x) − µᵢ)² / (2σᵢ²) ] + const
```

where `sᵢ(x)` is the great-circle distance from landmark `i` to `x`, and
`µᵢ = µ(dᵢ)`, `σᵢ = σ(dᵢ)`. The `Σ log σᵢ` term depends only on the measured RTTs, which
are fixed for a given target, so it is constant in `x`. Maximizing the joint density is
therefore **exactly**:

```
minimize  Σᵢ zᵢ²        where  zᵢ = (sᵢ(x) − µᵢ) / σᵢ
```

Spotter's MAP estimate is σ-weighted nonlinear least squares — a 2-parameter optimization,
`O(n)` per iteration, no polygons and no grid. The HTM mesh in §V-A-2 is an implementation
choice serving the *visualization* and *confidence-region* outputs, not a requirement for
the point estimate.

This matters for cost. The face-decomposition path builds the planar arrangement of `2n`
circle boundaries → `O(n²)` faces, each needing `n` Shapely containment tests, so
`O(n³)`-ish per target. The polygon machinery is *more* expensive than solving the exact
problem.

## Assessment of the max-count approximation

Proposal considered: keep annuli `[µᵢ − kσᵢ, µᵢ + kσᵢ]`, drop the AND, and select the
face of the arrangement contained in the **largest number** of annuli.

Against `min Σzᵢ²`, maximizing `Σᵢ 1{|zᵢ| ≤ k}` replaces a quadratic penalty with a
top-hat. In M-estimator terms that is the 0-1 / "skipped" loss — MAP under a
uniform-plus-outlier (RANSAC consensus) model, not a Gaussian one. Consequences:

1. **Strictly better than what we have.** It is monotone where the AND is brittle: never
   empty if any face is covered by ≥1 annulus. It eliminates `EMPTY_REGION` and
   `EXCLUSIVE_REGION` as failure classes by construction, and it removes the *reason*
   `enable_circle_filter` exists — engulfing constraints no longer veto, they only vote
   where they cover. Defects 1 and 2 both dissolve.
2. **Coarse by construction.** The objective takes at most `n+1` distinct values, so with
   10 VPs there are ≤11 distinguishable scores over the whole plane. Within the winning
   face the objective is flat, so CTR is again choosing the point — Defect 3 survives
   untouched.
3. **Discards σ heterogeneity.** Under `Σzᵢ²` a tight VP dominates via `1/σᵢ`. Under
   max-count every VP contributes exactly 1 whether `σᵢ` is 50 km or 5000 km — throwing
   away the entire reason to fit `σ(d)`. Partial repair: weight the count by `1/σᵢ²`, or
   by log-density at the face's representative point. The existing `weight` field in
   `compute_feasible_region_weighted` supports this structurally; only the
   `exp(-rtt/τ)` expression would change.
4. **Introduces a tuning knob the paper does not have.** `k` does not exist in
   `Σzᵢ²`. Count is monotone in `k` (k→∞ ⇒ every face scores `n`, region = everything;
   k→0 ⇒ region shatters), so there is an interior optimum that depends on fleet geometry
   and on inflation — i.e. it will not transfer between the US (~35 km median VP) and DE
   (~4.6 km) regimes. Reporting a per-setup-tuned `k` is reporting a tuned approximation,
   not Spotter.
5. **Approximation error is correlated with the phenomenon under study.** It is a good
   approximation when bands are narrow relative to inter-landmark geometry and inflation
   is low (the eu-de regime): near the `Σzᵢ²` minimizer all residuals are small, so all
   indicators are on and the top-count face is small and well-placed. It degrades when
   bands are wide and overlapping — country-scale, high inflation — which is exactly where
   CBG already degrades. So its error will confound failure attribution.

**It is free to measure.** Max-count is already expressible as a config: all-equal
`weight` plus `highest_weight_only=True` in
[`compute_feasible_region_weighted`](../scripts/libs/octant/octant_geolocation.py#L301)
(the face weight is `sum(c.weight for c, a in annuli if a.contains(rep))`, which with unit
weights *is* the containment count). No new code required for the diagnostic.

## Recommendations

1. **Build the exact reference first.** `min Σzᵢ²` via `scipy.optimize.least_squares` on a
   local tangent plane — small, no grid, and it makes "how approximate is max-count?" a
   measurement rather than an argument. It also needs `NormalDistLTD` to surface
   `(µᵢ, σᵢ)` rather than a band, which is the same interface change any faithful variant
   requires.
2. **Then measure max-count against it**, on eu-de first: zero no-proximity failures there
   means geometry effects are not confounded by fleet coverage.
3. **Relabel now, regardless of outcome.** Drop the Laki citation from the variant
   descriptions, stop framing the Octant head-to-head as isolating LTD (it isolates LTD
   *within a shared non-Spotter geometry*, which is a different and weaker claim), and
   treat the current `spotter_*` numbers as not comparable to the paper.
4. **§8.1 inherits this** and needs the same caveat.
5. A faithful *region* output (as opposed to point estimate) still requires a density MTL
   family returning a scalar field over cells — h3-4 substituting for the paper's HTM —
   plus a density-aware CTR. That is an architecture extension, and step 1 should tell us
   whether it is worth paying for.

---

# Measured: the fidelity gap is worth 2.4x at the median

Phase 1 (config-only max-count approximation) and Phase 2 (faithful density
MTL) both run on `as01-260728-260802-mesh`, 399 targets over 5 folds. Error
distance in km, pooled:

| combo | LTD + MTL + CTR | p5 | p25 | p50 | p75 | p95 | mean VPs | mtl_ms | ctr_ms |
|---|---|---|---|---|---|---|---|---|---|
| vanilla_cbg | low_envelope + planar_circle + geometric_centroid | 2.2 | 28.1 | 49.2 | 281.9 | 2144.4 | 16.3 | 73 | 0.3 |
| million_scale_cbg | speed_of_internet + planar_circle + geometric_centroid | 2.2 | 25.3 | 50.3 | 247.6 | 2175.5 | 4.5 | 8 | 0.2 |
| octant_cbg_hull | bounded_spline + planar_annulus_weighted + mc_medoid | 0.8 | 6.2 | 71.5 | 313.3 | 559.3 | 27.8 | 1906 | 522 |
| octant_cbg_spl | bounded_spline + planar_annulus_weighted + mc_medoid | 0.9 | 6.3 | 79.9 | 345.9 | 694.8 | 25.6 | 1687 | 502 |
| **spotter_cbg** | normal_dist + planar_annulus_weighted + mc_medoid | 208.9 | 383.0 | 527.1 | 935.8 | 2713.8 | 23.8 | 680 | 373 |
| spotter_appr_fil | ditto, uniform weights | 210.9 | 452.9 | 546.1 | 975.9 | 2717.7 | 23.8 | 681 | 362 |
| spotter_true_grid | normal_dist + gaussian_density + density_argmax | 39.4 | 136.2 | 239.0 | 617.4 | 2523.6 | 128.2 | 121 | 0.3 |
| **spotter_true** | normal_dist + gaussian_density + density_mle | **18.6** | 140.9 | **218.6** | 609.1 | 2511.9 | 128.2 | 118 | 11 |

## Findings

**1. The harness was costing Spotter a factor of 2.4 at p50 and 11x at p5.**
527.1 -> 218.6 km median, 208.9 -> 18.6 km at p5, from changing nothing but the
MTL and CTR. Every `spotter_cbg` number published so far understates Spotter by
roughly that much. This is the quantification of Defects 1-3.

**2. The faithful method is also ~6x CHEAPER.** MTL 680 -> 118 ms, CTR 373 ->
11 ms. The density surface costs less than the polygon intersection it replaces,
which kills the last argument for keeping the annular stack for Spotter. (The
filter-off annular arm, `spotter_appr`, was abandoned after 2h23m of CPU per
fold without completing a single one: 73 s/target, of which 71.4 s was the naive
`contains` loop -- see "Cost" below.)

**3. EMPTY_REGION and EXCLUSIVE_REGION are gone, as predicted.** `spotter_true`
reports `mtl_success` 399/399 with no `mtl_error` at all. They were never
Spotter's failure modes; they were the harness's. This confirms the caveat now
attached to `finding_exclusive_region_verified`.

**4. All VPs now participate: 23.8 -> 128.2.** The circle filter was discarding
~81% of constraints (131 -> 16 on a sampled target). Under a density product
there is no redundant constraint, so every VP contributes.

**5. The grid costs ~21 km at p5 and ~20 km at p50.** `spotter_true_grid`
(argmax cell) vs `spotter_true` (continuous MAP): 39.4 -> 18.6 at p5,
239.0 -> 218.6 at p50. An H3-4 cell is ~20 km edge, so the gap is the grid's own
quantisation and it lands in the same range as the accuracy thresholds being
reported. Read the paper's literal grid reading as having a ~20 km floor.

**6. Uniform weighting is nearly inert.** `spotter_cbg` -> `spotter_appr_fil`
moves p50 by 19 km (527 -> 546) with identical participants and timing. The
`exp(-rtt/tau)` weight -- one of the three non-Spotter imports flagged above --
turns out not to matter. The AND and the filter were the load-bearing defects.

**7. Spotter still loses to vanilla and Octant on this dataset, and that part
was NOT an artifact.** Faithful p50 218.6 km against vanilla's 49.2 and
Octant-Hull's 71.5. Consistent with
[2026-05-17-spotter-normality-check.md](2026-05-17-spotter-normality-check.md):
landmark-independence fails on heterogeneous access networks, so a single pooled
(mu, sigma) is the wrong model for this VP population regardless of how the
geometry is done. The fidelity fix changes the magnitude of Spotter's
disadvantage, not its direction. Note vanilla's percentiles are computed over
its 292/399 successful MTLs -- a survivorship effect that flatters its tail --
whereas every Spotter arm answers 399/399.

## Cost, measured

Per-target, one sampled as01 target with the real LTD bands (~1,500 km-wide
annuli at ~3,000 km radius, so every boundary crosses every other):

| path | annuli | faces | union | polygonize | contains | total |
|---|---|---|---|---|---|---|
| annular, filter ON | 16 | 329 | 0.01s | 0.02s | 0.09s | 0.11s |
| annular, filter OFF | 131 | 33,563 | 0.39s | 1.57s | 71.42s | 73.38s |
| density (h3-4, coarse 2) | 131 | n/a | -- | -- | -- | ~0.13s |

The annular blow-up is **97% the containment loop**, not the geometry:
`sum(c.weight for c, a in annuli if a.contains(rep))` is 4.4M one-at-a-time
Shapely calls with no spatial index. Measured replacements on the same data:
vectorized `shapely.contains` 3.25s (22x), `STRtree.query(predicate="contains")`
**0.52s (137x)**. Worth fixing on its own account -- it speeds up every Octant
variant too -- and tracked separately from the Spotter question.

## What was built

* `Distance.mu_km` / `.sigma_km` (`framework/v2/types.py`) -- the distribution
  carried alongside the band, because the band is not invertible (inner clamps
  at 0, outer is clipped by 2/3*c, half-width is `k*sigma` for an internal `k`).
* `SpotterRTTModel.predict_mu_sigma` -- (mu, sigma) under the same
  refuse-to-extrapolate clamp the bounds path uses.
* `DensityField` + `DensityMTLMethod` (`framework/v2/mtl/base.py`) -- a third
  MTL family. The annulus families answer "which points satisfy every
  constraint" (a set question, one violation empties it); this one answers "how
  plausible is each point" (a ranking, a bad fit lowers a score).
* `GaussianDensityMTL` (`gaussian_density`) -- Eq. (2) per H3 cell,
  coarse-to-fine. Verified to give the identical argmax as a single global
  288,122-cell pass on a unimodal field.
* Four CTRs (`ctr/density_point.py`): `density_argmax`, `density_mean`,
  `density_region_center` -- the three estimators SS III-B names -- plus
  `density_mle`, the exact continuous MAP by least squares, which is the
  reference approximations should be scored against.
* 31 tests. The load-bearing ones assert the two properties that make this
  Spotter rather than Octant-with-sigma: one hostile constraint cannot empty the
  field, and the inner hole is low-density rather than excluded.

## Still open

1. **as01 is a weak instrument**: 399 targets over only 20 distinct
   coordinates, every region spanning all 5 folds, accuracy quantised in 5%
   steps. Treat the ordering as real and the values as coarse. The verdict
   belongs on a RIPE setup -- eu-de first, where zero no-proximity failures mean
   geometry effects are not confounded by fleet coverage.
2. **as02/as03 mesh trees are sparse** (as02 empty, as03 `vanilla_cbg` only), so
   neither is an incremental add and neither has a `spotter_cbg` baseline.
3. Recommendations 3-4 from the section above stand: relabel the published
   `spotter_*` results, and carry the caveat into SS8.1.
