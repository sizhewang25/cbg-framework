# Randomness / determinism assessment — v2 benchmark pipeline

**Date:** 2026-06-02
**Branch:** `add-nofil-combos`
**Question:** Given the same input target list and seed, is the per-target output randomness reproducible?

**Answer: Yes**, under two normal conditions (stable target ordering, same library versions). Verified by code trace + reading the `seed` column out of existing `targets.parquet` outputs.

---

## Seed chain (entry → Sobol sampler)

1. Config sets `seed: 42` — every `*_final.yaml` (and all 31 configs) has it. [europe_as3209_final.yaml:50](../scripts/benchmark/v2/config/europe_as3209_final.yaml#L50)
2. Snakefile forwards it as `--seed 42` (per-combo seed wins over run-level; empty flag ⇒ CTR self-randomizes). [Snakefile:82-86](../scripts/benchmark/v2/Snakefile#L82-L86)
3. CLI captures it as `base_seed`. [cli.py:125](../scripts/benchmark/v2/cli.py#L125)
4. Runner derives a **per-target** seed and resets the CTR RNG **before each target**:
   ```python
   has_stochastic_ctr = hasattr(model.ctr, "rng")          # True for the 2 stochastic CTRs
   target_seed = _derive_target_seed(spec.base_seed, target_index)
   if has_stochastic_ctr and target_seed is not None:
       model.ctr.rng = np.random.default_rng(target_seed)
   ```
   [runner.py:105-110](../scripts/benchmark/v2/runner.py#L105-L110)
5. `_derive_target_seed` = `SeedSequence([base_seed, target_index]).generate_state(1, uint32)`. [runner.py:157-166](../scripts/benchmark/v2/runner.py#L157-L166)
6. CTR passes `rng=self.rng` into `sample_points_in_region`, which derives the Sobol seed from it. [monte_carlo_medoid.py:45](../scripts/framework/v2/ctr/monte_carlo_medoid.py#L45), [geometry.py:212](../scripts/framework/geometry.py#L212)

## Why it's airtight

The **only** RNG sources in the whole v2 geolocate path are the two stochastic CTRs — confirmed by grep over `scripts/framework/v2/` (no other `np.random` / `default_rng` / `sample` / `choice` / `shuffle` / `permutation`):

| Component | Randomness? | Notes |
|---|---|---|
| `MonteCarloMedoidCTR` | yes — `self.rng` | reset per target by runner |
| `GeometricMedianCTR` | yes — `self.rng` (default `seed=42`) | reset per target by runner |
| `GeometricCentroidCTR` | none | area-weighted centroid |
| `SphericalCircleMTL`, `PlanarCircleMTL`, `PlanarAnnulusMTL`, `PlanarAnnulusWeightedMTL` | none | `planar_*` use a fixed `n_pts` grid, not random sampling |

Both stochastic CTRs expose `.rng`, so `has_stochastic_ctr` is `True` and the runner reconstructs the generator **fresh per target** from `(seed, target_index)`. Consequences:

- No RNG state leaks between targets.
- Reproducible regardless of combo execution order or parallel fold/combo scheduling (`-j N`).
- Reproducible regardless of how many RNG draws happen inside one target's geolocate.

## Empirical confirmation

The `seed` column of `targets.parquet` is bit-for-bit `SeedSequence([42, idx])`:

| target_index | seed |
|---|---|
| 0 | 3444837047 |
| 1 | 3329053876 |
| 2 | 955475868 |
| 3 | 2541583436 |
| 4 | 964687612 |

Identical across folds, combos, and ASNs — because the per-target seed depends only on `(base_seed, positional index)`, **not** on fold/combo/ASN/target identity. Since every config uses `42`, every output shows the same sequence. (This uniformity is what first *looked* like the seed wasn't config-controlled — it is; it's just `f(42, idx)`.)

## Caveats / conditions

1. **Order matters.** Seed is keyed to the positional index in `eval_targets`, not the target's stable identity. Same list in the same order ⇒ reproducible. A reshuffled list (even same targets) reassigns seeds. To make it order-invariant, key `_derive_target_seed` off a stable target ID instead of `target_index`.
2. **Same library versions.** `numpy.SeedSequence` is platform-independent, but the actual consumer is `scipy.stats.qmc.Sobol` — cross-machine reproducibility assumes the same scipy version.
3. **Latent footgun:** if a future config drops the `seed` key (and no per-combo seed), `base_seed=None` ⇒ `_derive_target_seed` returns `None` ⇒ runner skips the reset ⇒ `MonteCarloMedoidCTR(seed=None)` builds an unseeded `default_rng()` ⇒ non-deterministic. `GeometricMedianCTR` stays deterministic (hardcoded `seed=42`). All current configs set `seed: 42`, so this is not active.

## Out of scope (not on the v2 pipeline)

Unseeded `random.sample` / `randint` / `Pool(24)` in [measurement_utils.py](../scripts/utils/measurement_utils.py), [atlas_api.py](../scripts/ripe_atlas/atlas_api.py), and legacy [analysis.py](../scripts/analysis/analysis.py) belong to the old million-scale measurement code, not the v2 benchmark/analysis pipeline.
