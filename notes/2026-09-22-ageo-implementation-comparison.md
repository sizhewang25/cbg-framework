# `ageo` (active-geolocator) vs our framework — CBG / Octant / Spotter

**Date:** 2026-09-22
**Source repo:** [zackw/active-geolocator](https://github.com/zackw/active-geolocator)
(`lib/ageo/`, 1,476 lines: `calibration.py`, `ranging.py`, `ageo.py`), backing
Weinberg et al., *How to Catch when Proxies Lie: Verifying the Physical Locations of
Network Proxies with Active Geolocation*, IMC 2018.
**Companion notes:**
[2026-09-18-spotter-mtl-fidelity-gap.md](2026-09-18-spotter-mtl-fidelity-gap.md),
[2026-05-17-spotter-normality-check.md](2026-05-17-spotter-normality-check.md),
[2026-05-15-cbg-related-work-datasets.md](2026-05-15-cbg-related-work-datasets.md).

## Summary

`ageo` is the only other open implementation of all three of CBG, Octant and Spotter in
one codebase, so it is the closest thing we have to an external reference. It is **not**
a drop-in reference: its three algorithms differ *only* in the latency-to-distance
curve, and every one of them is combined by a **soft product of fuzzy memberships on a
raster**, never by hard set intersection. That single design choice is a direct answer
to two of our recorded failure modes (`SphericalCircleMTL` brittleness, the Octant
AS7018 NA collapse), and it is the most transferable idea in the repo.

Two verified defects in their Spotter (§5.1) mean their Spotter numbers should not be
cited as an independent check on ours.

Three things they have that we do not: a geography prior, soft constraint shoulders, and
per-estimate region-quality confidence (§6). Several things we have that are
measurably better, chiefly the monotone-µ fit (§5).

---

## 1. Architecture: the axes do not line up

| `ageo` stage | our stage |
|---|---|
| `Calibration.distance_range(rtts) -> (min_d, max_d)` | LTD -> `Distance(lower_km, upper_km)` |
| `ranging.{MinMax,Gaussian}`: distance -> **soft** probability | *(no analogue)* |
| `Observation`: rasterise that onto a lat/lon grid | — |
| `Location.intersection`: elementwise **product** of sparse pmfs | MTL (exact vector geometry) |
| `Map`: land/coastline **Bayesian prior**, multiplied in | *(none)* |
| `Location.centroid` (mass-weighted) / `.rep_pt` (MAP w/ tie-break) | CTR |

The geometry stage in `ageo` is byte-identical across CBG, Octant and Spotter. In our
framework the algorithm identity is spread across LTD **and** MTL (`vanilla_cbg` =
`low_envelope` + `spherical_circle`; `octant_*` = `bounded_spline` + `planar_annulus`).
Their algorithm grid is Calibration x Ranging = 2x3 with 8 combos hand-wired in
`calibrate` and `locate-from-db`; ours is a full registry cross-product
([`registry.py`](../scripts/framework/v2/registry.py)).

### 1.1 Soft constraints — the load-bearing difference

`ranging.MinMax` turns a calibrated `[min_d, max_d]` into a trapezoid in distance:
value **1.0** inside the calibrated band, falling linearly to **0.75** at the *empirical*
bounds (110,000–153,000 km/s, plus 55 ms at distance 0), and to **0** only at the
*physical* bounds (0–200,000 km/s). Their own comment states the motivation:

> Because all of the empirical calibration algorithms are liable to spit out an
> observation from time to time that's inconsistent with the global truth, we do not
> drop the probability straight to zero immediately at the limits suggested by the
> calibration.

Consequences, stated against our findings:

* **A single under-predicting disk cannot empty the region.** `finding_spherical_circle_brittle`
  (the inside-all filter ANDs constraints; one bad disk empties the feasible set) has no
  counterpart here — a bad constraint depresses a score, it cannot veto.
* **The annulus hole is a trough, not an exclusion.** For Octant, sorted bounds put 0 at
  d=0 rising to 0.75 at the empirical minimum. So `EXCLUSIVE_REGION` — which we measured
  at ~50% for Octant and ~89–99% for Spotter (`finding_exclusive_region_verified`) — is
  structurally unreachable. Their design has no give-up path at all; the worst outcome is
  a `vacuous` Location, and there is no fallback-to-nearest-VP.
* The tolerance dividend we attribute to centroid snapping
  (`finding_tolerance_dividend_scales_with_vp_proximity`) is, in their design, built into
  the constraint shape rather than recovered downstream.

---

## 2. CBG

Same LP, genuinely: minimise total slack above the line subject to `m*d_j + b <= rtt_j`,
`m >= 2/3 c` slope, `b >= 0`, then invert. Their `bounds = [(1,1), (1/100000, None), (0,
np.amin(cy))]` is in metres/ms, i.e. exactly our `THEORETICAL_SLOPE = 0.01` ms/km.

Four substantive deltas against
[`scripts/libs/cbg/rtt_model.py`](../scripts/libs/cbg/rtt_model.py) (`fit_bestline_lp`,
L113-L201):

1. **They bin the distance axis into ~804 bins of ~25 km and feed the LP one min-RTT per
   bin.** We feed raw pairs. This is the same equal-weight-per-bin vs equal-weight-per-pair
   tradeoff that `fit_mu_sigma`'s docstring argues about for Spotter's µ — and we went the
   opposite way on each. With their binning, a dense RTT region cannot drag the bestline.
   Empty bins are back-filled with the next higher bin's min-RTT so `linprog` never sees NaN.
2. **An artificial constraint point at (20,037,508 m, 237.16 ms)** — "ensures that the fit
   will not select a data point from a satellite link as a defining point for the line."
   We have no equivalent, so a single satellite-fed anchor can define our bestline and
   inflate every radius from that VP.
3. **`b <= min(minrtt)`** as an explicit intercept ceiling. Ours is unbounded above.
4. **`discard_infeasible` is two-sided** — drops faster-than-200,000 km/s *and*
   slower-than-110,000 km/s-after-55 ms. Our `filter_baseline` only drops the fast side.
   Near-harmless for a lower-envelope LP (slow points never bind), but it is load-bearing
   for their Octant and Spotter, which we filter equally loosely.

Minor: they also drop `distance == 0` rows ("CBG can't make constructive use of them"),
and they use WGS84 geodesic distance (`pyproj.Geod.inv`) where we use haversine on
R=6371 — sub-0.5%, ignorable.

---

## 3. Octant

Theirs is explicitly `QuasiOctant` and drops both the traceroute stage and the last-hop
height correction. On the pieces they kept, **ours is closer to the paper**:

| | `ageo` QuasiOctant | ours (`bounded_spline` + `planar_annulus`) |
|---|---|---|
| band | convex hull upper/lower chains | same |
| hull algorithm | `scipy.spatial.ConvexHull(qhull_options="QbB")` | hand-rolled monotone chain |
| best-guess curve | none | LSQ spline inside the hull |
| widening | none | multiplicative delta per VP to `target_coverage=0.9` |
| tail handling | cut upper at 50th pct, lower at 75th pct of obs; extrapolate at fixed empirical slopes | density cutoff (last bin with >= 5 pts), then sentinel-at-(10000 ms, 2/3 c) extension |
| inner bound in tail | keeps growing at 100 km/ms | held **flat** |
| combining | soft trapezoid product | hard annular intersection + redundant-disk filter |

Two observations worth carrying forward:

* **Their tail rule is percentile-based** — always defined, never degenerate. Ours is
  density-based (paper-faithful) but can put `cutoff_rtt` at `min_rtt` when no bin clears
  `cutoff_min_points`.
* **Their inner bound keeps growing in the tail**, which is *more* aggressive than ours.
  Safe under soft ranging; under our hard subtraction it would be a disaster. This is a
  clean illustration that constraint tightness and constraint hardness have to be chosen
  together — relevant framing for §8 of the paper.
* Their hull code is fragile in a way ours is not: it raises `ValueError` on "all hull
  points have same x-coord" and "hull split inappropriately - i=... v=...", neither of
  which a monotone chain can hit.

---

## 4. Spotter — the same objective, arrived at independently

`ranging.Gaussian` evaluates `N(mu, sigma^2)` pdf per landmark per grid cell;
`Location.intersection` multiplies them. That product is `exp(-1/2 * sum z_i^2)` up to a
constant — **literally** [`gaussian_density.py::_log_density`](../scripts/framework/v2/mtl/gaussian_density.py#L150-L160).

This is independent corroboration of `2026-09-18-spotter-mtl-fidelity-gap.md`: the
faithful reading of Eq. (2) is a density product, not an annular intersection, and the
only other implementation of the paper agrees.

Where they differ, and where we are better:

* **They fit the monotone cubic with SLSQP** (`3a > 0` and `B^2 - 4AC < 0` as `ineq`
  constraints, unit-square scaled, `maxiter=10000`, wrapped in a
  `MinimizationFailedWarning`). That is exactly the method `_monotone_nonneg_cubic`'s
  docstring reports as failing to converge on saturating data. Our NNLS/Bernstein
  reformulation (`finding_monotone_mu_log_sigma_fit`) is strictly better and is a
  publishable delta.
* **They fit sigma directly to windowed standard deviations** — 800 knots, 4-wide
  overlapping windows. That carries the slope-leak inflation quantified at +18.3% in
  `fit_mu_sigma`'s docstring; narrow overlapping windows shrink it but do not remove it.
  Our log-space fit from mu's residuals, with the `(gamma + log 2)/2` bias correction, has
  no such term and cannot produce a negative sigma.
* **They test landmark-independence; we assert it.** `calibrate` builds both per-landmark
  (`spo-m-1`, `spo-g-1`) and pooled (`spo-m-a`, `spo-g-a`) calibrations, and
  `calibration-report` scores them side by side ("Separate" vs "Combined"). Our
  `NormalDistLTD` is pooled-only by design. Given that
  `2026-05-17-spotter-normality-check.md` records the claim **failing** on
  `ping_10k_to_anchors`, their ablation is an experiment we are missing.
* **They run a Spotter-calibration-with-MinMax-ranging mode** (`spo-m`, using +/- 5 sigma
  as the band). That is the ablation our `spotter_cbg` vs `spotter_hybrid_cbg` split
  covers, but cleaner: same geometry, only the ranging changes.
* **Discretisation:** one dense raster pass over a bounded region (sparse CSR, no
  pruning) vs our coarse-to-fine H3 with `top_k` + `neighbor_ring`. Theirs has no
  pruning-bias caveat; ours is far cheaper.

### 4.1 Two verified defects in their Spotter

Both matter if anyone treats `ageo` as a reference implementation. Verified numerically
on a 5,000-point synthetic set (script in §8).

**Defect A — `windowed_moments` receives an unsorted x-axis.**
`discard_infeasible` returns `fobs[np.lexsort((fobs[:,1], fobs[:,0])),:]`; `np.lexsort`
takes the *last* key as primary, so rows are sorted by **column 0 = distance**. CBG then
uses `ys = obs[:,0]` (sorted — correct), but `Spotter.__init__` calls
`windowed_moments(obs[:,1], obs[:,0])`, so `xs` is **RTT and unsorted** — yet the knot
grid is `np.linspace(xs[0], xs[-1], nknots + 4)`. Those are the RTTs of the nearest and
farthest observations, not min/max RTT.

```
xs[0], xs[-1]     = 7.99, 209.11
true rtt min,max  = 1.45, 281.11
frac of obs inside their knot span = 0.966
```

Mild on well-behaved data, unbounded in principle. If `xs[0] > xs[-1]` the edges run
backwards, every window is empty, and the fit dies in their `raise RuntimeError("wtf")`
branch.

**Defect B — `np.percentile(self.rtts, .25)` is the 0.25th percentile, not Q1.**
A 100x slip, present in both `ranging.Gaussian.__init__` and `Spotter.distance_range`,
against a docstring that says explicitly "we use the first quartile as the representative
delay; this is because the delay distribution is unbounded upward."

```
np.percentile(rtt, .25) = 5.913    <- what the code does
np.percentile(rtt, 25)  = 64.431   <- what the docstring says
min(rtt)                = 1.448
```

So their Spotter is effectively running on a near-minimum RTT, behaving much more like
CBG/Octant's `np.amin(rtts)` than like the paper's spec.

Note this aggregation step has no counterpart in our framework at all: their Calibration
receives the **whole RTT vector** to one landmark and reduces it internally; our LTD
receives a single already-reduced latency per VP.

---

## 5. What they have that we do not

1. **A geography prior.** `ageo.Map` loads an HDF5 baseline built by
   `maps/make_geog_baseline.py` from shapefiles: 1 inside land, 0 well outside, graded
   values within a `fuzz` band at coastlines. It is normalised and multiplied into the
   posterior, so ocean cells are zeroed. Free accuracy on coastal targets; we have nothing
   equivalent. (Live in `calibrate`; commented out in `locate-from-db`.)
2. **Soft constraint shoulders** (§1.1).
3. **Per-estimate confidence.** `Location.area` (probability-weighted, on an equal-area
   projection) and `Location.covariance`. We report point error only.
4. **`rep_pt`** — the highest-probability cell, tie-broken by proximity to the
   mass-weighted centroid. A principled hybrid of our `density_argmax` and the removed
   `density_mean` (see `2026-09-22-spotter-faithful-vs-harness-conclusion.md`).

## 6. What we have that they do not

Per-VP delta coverage search; `filter_redundant_outer_disks`; NNLS/Bernstein monotone
fit; log-space sigma; the `Error` taxonomy plus lowest-latency-VP fallback (they only
have a `vacuous` flag); coarse-to-fine H3 pruning; the LTD/MTL/CTR factorisation itself;
five CTR variants against their two; ~50 test modules against their zero.

Portability: `ageo` will not run on a modern stack without work — `pyproj.transform`
(removed in pyproj 2.2), `np.array(disk.boundary)` (removed in Shapely 2.0), SLSQP
pinned "for scipy 0.17.1 reproducibility", `lib/six.py` and `lib/selectors34.py` vendored
for Python 2.7/3.4.

## 7. Different evaluation objective

Their harness is **verification**, not geolocation: `confusion-report` computes what
fraction of the estimated region's probability mass falls in each candidate country, and
`claimed-country-accuracy` asks whether a proxy is where it claims to be. That is the
IMC 2018 proxy-lies paper. Ours is distance-error percentiles plus cluster/airport
classification (`project_airport_eval_metric`, `project_answer_space_clustering`).

So their region-quality machinery (`area`, `contains_point`, per-region mass) has no
counterpart in our harness — and it is arguably the *right* scoring for a soft-constraint
design, which never yields a crisp region to measure. Worth remembering if we adopt §1.1.

## 8. Reproducing the §4.1 checks

```python
import numpy as np
rng = np.random.default_rng(0)
n = 5000
dist = rng.uniform(1, 15_000_000, n)        # metres, as ageo uses
rtt  = dist / 76_500 + rng.gamma(2, 8, n)   # ms, with a queueing tail

obs = np.column_stack((dist, rtt))
feas = np.logical_and(obs[:,1] * (100*1000) >= obs[:,0],
                      (obs[:,1] - 55) * (55*1000) <= obs[:,0])
f = obs[feas,:]
f = f[np.lexsort((f[:,1], f[:,0])),:]       # ageo.calibration.discard_infeasible
xs = f[:,1]                                  # what windowed_moments receives as 'xs'

assert np.all(np.diff(f[:,0]) >= 0)          # sorted by distance
assert not np.all(np.diff(xs) >= 0)          # NOT sorted by rtt   <- Defect A
print(xs[0], xs[-1], xs.min(), xs.max())
print(np.percentile(xs, .25), np.percentile(xs, 25))  #             <- Defect B
```

## 9. Follow-ups worth doing

1. **Port `ranging.MinMax`'s soft-shoulder trapezoid as a new MTL family** and re-run the
   AS7018 NA collapse and the `eu-de` RTT-inflation isolation against it. This is the
   highest-value item: it turns "CBG is brittle to one bad disk" from a finding into a
   finding *with a fix we measured*.
2. **Add two-sided `discard_infeasible`** to our Octant and Spotter fit paths (the slow
   side, which `filter_baseline` currently lets through).
3. **Add a pooled-vs-per-landmark Spotter ablation** matching their `spo-*-1` / `spo-*-a`
   split, to turn the normality-check caveat into a number.
4. Consider the (20,037 km, 237.16 ms) satellite anchor point for our CBG LP.
5. Consider a land prior. Cheap, and every target in our sets is on land.
