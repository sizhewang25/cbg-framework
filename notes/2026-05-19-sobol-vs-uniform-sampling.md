# Sobol QMC vs. uniform random sampling

Why [scripts/framework/geometry.py:195-238](../scripts/framework/geometry.py#L195-L238)
samples with Sobol QMC + rejection instead of `np.random.uniform`, and what
that choice buys the Monte-Carlo medoid centroid in
[scripts/framework/v2/ctr/monte_carlo_median.py](../scripts/framework/v2/ctr/monte_carlo_median.py).

## Setup

Both samplers produce N points in `[0,1]²`, which we then rescale to the
region's bounding box and reject-test with `region.contains(Point)`. The only
thing that differs is **how those N points are placed relative to each other**.

## Uniform random (`np.random.uniform`)

Each point is drawn independently. Pairs have no awareness of each other:

- **Clumping is expected, not pathological.** With N=1000 i.i.d. points in
  `[0,1]²`, you routinely see small subregions with no points at all (~10% of
  the area, by the standard "Poisson voids" argument) and other small
  subregions with 4-5 points piled up. This is just the variance of
  independent draws.
- **Monte-Carlo error convergence is `O(1/√N)`.** Halving the error needs 4×
  the samples.

For a *medoid* estimate, clumping matters: a cluster of samples in one corner
of the region drags the sum-of-distances minimum toward that corner, even
when the corner has no special geometric significance. With 1000 i.i.d.
samples, that cluster bias is real.

## Sobol QMC

Sobol is a **low-discrepancy sequence** — deterministic (or
scrambled-deterministic) points constructed in base-2 to maximize uniformity.
Key properties:

- **Self-avoiding.** After N points, every dyadic box of area `1/N` contains
  close to 1 point. No big voids, no clumps. This is the *discrepancy* bound:
  Sobol's star-discrepancy is `O((log N)^d / N)` vs. uniform random's
  `O(1/√N)`. For 2D and N=5000, that's roughly **~10× better worst-case
  coverage**.
- **Scrambled.** `qmc.Sobol(scramble=True, ...)` (used here) applies Owen
  scrambling — randomizes the sequence in a way that preserves the
  low-discrepancy property but adds genuine variance for unbiased estimation.
  So you get *both* uniformity *and* an honest random sample (just with much
  smaller variance than i.i.d.).
- **Convergence is `O((log N)^d / N)`** — close to `O(1/N)` for low `d`. So
  ~1000 Sobol points ≈ ~10000 uniform-random points for coverage quality.

## Why this matters for the medoid specifically

The medoid is sensitive to *spatial distribution* of samples, not just their
count. Two failure modes uniform random exhibits:

1. **Bias in cluster regions.** Uniform random's accidental clusters skew the
   medoid toward dense areas. Sobol's even spacing means the medoid reflects
   the *region's geometry*, not the *sampler's variance*.
2. **Disconnected regions (`MultiPolygon`).** Rejection sampling fills
   components in proportion to their area. With small N + uniform random, you
   can get *unlucky* and miss a small component entirely. Sobol's coverage
   guarantee makes this much less likely — for the area-4 vs area-16
   MultiPolygon in
   [test_monte_carlo_median.py:49](../scripts/framework/v2/ctr/tests/test_monte_carlo_median.py#L49),
   Sobol reliably lands samples in both.

## One subtlety the code handles correctly

The rejection loop draws Sobol points in the **bounding box**, then keeps the
ones inside the polygon. Sobol's uniformity is over the bounding box, *not*
over the polygon. For a polygon that fills <50% of its bounding box (a tilted
diamond, say), Sobol-over-rectangle still translates into much better
uniformity-over-polygon than i.i.d. would, because the rejected points don't
cluster — they're filtered uniformly. The `max_attempts_factor=20` budget on
[geometry.py:219](../scripts/framework/geometry.py#L219) ensures we keep
drawing until we have `n_samples` accepted points or exhaust 20×.

## TL;DR

| | Uniform random | Sobol QMC (scrambled) |
|---|---|---|
| Star-discrepancy (2D) | `O(1/√N)` | `O((log N)² / N)` |
| Clumps/voids | yes, by design | actively avoided |
| Bias for medoid | yes (cluster pull) | no |
| Coverage of small components | luck-dependent | reliable |
| Unbiased? | yes | yes (scrambling preserves) |
| Effective "for-free" N multiplier | 1× | ~10× |

So `MonteCarloMedianCTR(n_samples=1000)` with Sobol gives coverage quality
roughly equivalent to ~10K i.i.d. uniform samples, at a fraction of the
runtime. That's why the default stays at 1000 — runtime priority, but the
sampler choice means we don't pay an accuracy price for the lower count.

## Cross-references

- [scripts/framework/geometry.py:195-238](../scripts/framework/geometry.py#L195-L238) — `sample_points_in_region`
- [scripts/framework/geometry.py:130-150](../scripts/framework/geometry.py#L130-L150) — `sampled_medoid`
- [scripts/framework/v2/ctr/monte_carlo_median.py](../scripts/framework/v2/ctr/monte_carlo_median.py) — caller
- SciPy docs: [`scipy.stats.qmc.Sobol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
