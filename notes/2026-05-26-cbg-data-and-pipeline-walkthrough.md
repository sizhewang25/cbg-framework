# CBG v2 data + pipeline walkthrough — `example.yaml` + vanilla CBG

**Date:** 2026-05-26
**Config of interest:** [scripts/benchmark/v2/config/example.yaml](../scripts/benchmark/v2/config/example.yaml) (AS16509 → 5 anchor folds, `vanilla_cbg` combo)
**Frame:** root-causing three CBG observations on the all-6-ASN error CDF —
1. None of the CBG variants beat the shortest-ping baseline.
2. Tighter constraints (Octant, Spotter) often end with an empty MTL intersection.
3. Octant / Spotter have low SUCCESS rates.

This note walks the full path from yaml → CBG model output for that combo, so the downstream LTD/MTL deep dives have a shared anchor.

---

## Part 1 — How the stratified data reaches the CBG model

`example.yaml` declares:

```yaml
source: ripe_atlas_asn_corpora
setup: probes_to_anchors
slices: [fold_0, fold_1, fold_2, fold_3, fold_4]
source_kwargs:
  probe_data_dir: datasets/ripe_atlas/asn_corpora/probes/global
  probe_asn: 16509
  anchor_data_dir: datasets/ripe_atlas/asn_corpora/anchors/kfolds
  max_rtt_ms: 10000
```

Snakemake fans out across `(slice × combo)` cells. Each cell runs in two phases.

### Phase 1 — `materialize-inputs` per slice

`cli.py materialize-inputs` builds

```python
RipeAtlasASNCorporaSource(slice="fold_N", setup="probes_to_anchors", **source_kwargs)
```

…and `inputs.materialize_inputs(source, run_id=…)` ([inputs.py:78](../scripts/benchmark/v2/inputs.py#L78)) writes four parquets under
`inputs/ripe_atlas_asn_corpora/<run_id>/probes_to_anchors/fold_N/`:

- `vp_configs.parquet`
- `tg_configs.parquet`
- `fit_samples.parquet`
- `eval_observations.parquet`

Stratification mechanics live inside the source ([ripe_atlas_asn_corpora.py:239-287](../scripts/benchmark/v2/sources/ripe_atlas_asn_corpora.py#L239-L287)):

1. **`_load_probes`** — reads `probes_of_as_<asn>.json` → one VP corpus (all probes share the same ASN, already continent-filtered + city-deduped upstream).
2. **`_load_anchors`** — globs `anchor_fold_*.json`. Anchors in `anchor_fold_<N>.json` go into `eval_anchors`; anchors in every other fold go into `fit_anchors`. K is auto-discovered from the file count, and `fold_N >= K` raises.
3. **`_load_rtts`** — pulls `ping_10k_to_anchors` via `compute_rtts_per_dst_src`, then collapses to one `min_rtt` per `(anchor, probe)` and filters to anchors ∈ `eval ∪ fit` and probes in the corpus. RTTs ≤ 0 are dropped.

Then the four `iter_*` methods emit:

- `iter_vp_configs` — one row per probe in the corpus that appears in at least one RTT row ([line 124](../scripts/benchmark/v2/sources/ripe_atlas_asn_corpora.py#L124)).
- `iter_tg_configs` — every anchor across both eval and fit (static catalog).
- `iter_fit_samples` — `FitSample(vp_id=probe_ip, vp_coord=probe_coord, probe_coord=anchor_coord, latency=rtt)` for **anchor ∈ fit_anchors only** ([line 160-185](../scripts/benchmark/v2/sources/ripe_atlas_asn_corpora.py#L160-L185)). The `probe_coord` field name is a v1 quirk — it really means "known target coord" (= anchor coord here).
- `iter_eval_targets` — one `EvalTarget(target_id=anchor_ip, true_coord=anchor_coord, obs=[(probe_ip, probe_coord, rtt), …])` for **anchor ∈ eval_anchors only** ([line 187-207](../scripts/benchmark/v2/sources/ripe_atlas_asn_corpora.py#L187-L207)).

So **fit_anchors ∩ eval_anchors = ∅** by construction — the LTD trains on (K−1)/K of the anchors, and the held-out 1/K fold becomes eval. The split is anchor-side; the probe corpus is shared across both views.

### Phase 2 — `run-combo` reads the parquets

`cli.py run-combo` → `run_one_combo` ([runner.py:67](../scripts/benchmark/v2/runner.py#L67)):

```python
fit_samples  = load_fit_samples_parquet(inputs_dir / "fit_samples.parquet")          # 1 row per (vp, fit_anchor)
eval_targets = load_eval_targets_parquet(inputs_dir / "eval_observations.parquet")   # 1 EvalTarget per eval_anchor

model = CBGModel.from_config(ltd="low_envelope", mtl="spherical_circle",
                             ctr="boundary_vertex_mean", …)
fit_result = model.fit(fit_samples)                  # LTD learns per-VP RTT→distance
for target in eval_targets:
    result = model.geolocate(target.obs, …)          # LTD → MTL → CTR pipeline
```

- `load_fit_samples_parquet` ([inputs.py:215](../scripts/benchmark/v2/inputs.py#L215)) re-hydrates each row as `FitSample(vp_id, vp_coord=probe lat/lon, probe_coord=anchor lat/lon, latency=min_rtt)`.
- `load_eval_targets_parquet` ([inputs.py:232](../scripts/benchmark/v2/inputs.py#L232)) groups the flat observation rows by `target_id` and yields one `EvalTarget` per held-out anchor with its full probe-observation list.

**Net effect for vanilla CBG on fold_N:** the LTD only ever sees (probe, fit-anchor) RTT pairs; at eval time it scores every anchor in fold N. Leakage-free.

---

## Part 2 — Vanilla-CBG pipeline (fit → predict → multilaterate → centroid)

`CBGModel` ([model.py:78](../scripts/framework/v2/model.py#L78)) is a thin orchestrator of three stages registered in `registry.py`:

| Stage | Vanilla choice | File |
|---|---|---|
| LTD | `low_envelope` | [ltd/low_envelope.py](../scripts/framework/v2/ltd/low_envelope.py) |
| MTL | `spherical_circle` | [mtl/spherical_circle.py](../scripts/framework/v2/mtl/spherical_circle.py) |
| CTR | `boundary_vertex_mean` | [ctr/boundary_vertex_mean.py](../scripts/framework/v2/ctr/boundary_vertex_mean.py) |

`_validate_family_pairing` ([model.py:93](../scripts/framework/v2/model.py#L93)) only rejects `AnnulusMTL ← CircleLTD`. `low_envelope` is a `CircleLTDModel`, `spherical_circle` is a `CircleMTLMethod` → the pairing passes.

### Stage 1 — Fit (`model.fit(fit_samples)`)

`CBGModel.fit` forwards to `self.ltd.fit(samples)` ([model.py:109](../scripts/framework/v2/model.py#L109)). `LowEnvelopeLTD._fit` ([ltd/low_envelope.py:37](../scripts/framework/v2/ltd/low_envelope.py#L37)):

1. Bucket samples by `vp_id`. Each bucket holds `(haversine(vp_coord, anchor_coord), rtt)` pairs over the 4 fit folds.
2. For each VP, construct a `RTTDistanceModel` and call `model.fit(distances, rtts)`.

`RTTDistanceModel.fit` ([libs/cbg/rtt_model.py:58](../scripts/libs/cbg/rtt_model.py#L58)) runs:

- **`filter_baseline`** ([line 99](../scripts/libs/cbg/rtt_model.py#L99)) — drops any `(d, rtt)` with `rtt < 0.01·d`, i.e. below the 2/3·c floor. Mislabeled coords or one-way-path artifacts get culled here.
- **`fit_bestline_lp`** ([line 116](../scripts/libs/cbg/rtt_model.py#L116)) — Gueye et al. (IMC 2004) LP:

  ```
  min  Σ [d_j − (m·g_j + b)]
  s.t. m·g_j + b ≤ d_j   ∀ j
       m ≥ baseline_slope (= 0.01 ms/km, the 2/3·c floor)
       b ≥ 0
  ```

  Tightest affine RTT→distance line that lies above every observation in the (g_rtt, d) plane. Solver: HiGHS via `scipy.optimize.linprog`.

- Needs **≥ 3 points** ([line 138](../scripts/libs/cbg/rtt_model.py#L138)). VPs with fewer surviving samples → `fitted=False`. Those VPs become permanent `Error.VP_NOT_FITTED` at predict time.

Persisted on each VP: `slope`, `intercept`, `fitted`. `FittingResult.args["vps_fitted"]` is the ID list that survived.

### Stage 2 — Predict (LTD phase of `geolocate`)

`geolocate` ([model.py:112](../scripts/framework/v2/model.py#L112)) calls `self.ltd.predict_all(obs)` which for vanilla CBG maps to `LowEnvelopeLTD._predict` per VP ([ltd/low_envelope.py:77](../scripts/framework/v2/ltd/low_envelope.py#L77)):

```python
radius_km = submodel.predict_distance(latency)   # max(0, (rtt - intercept) / slope)
LTDResult(success=True, tg_distance=Distance(upper_km=radius_km))    # lower_km defaults to 0
```

One `LTDResult` per observation, each describing a **disk** centered at the VP with radius = predicted upper bound. VPs without a fitted submodel return `success=False, error=VP_NOT_FITTED` and are excluded by `ok = [r for r in ltd_results if r.success]` in `model.py:131`.

### Stage 3 — Multilaterate (`spherical_circle`)

`SphericalCircleMTL._multilaterate` ([mtl/spherical_circle.py:36](../scripts/framework/v2/mtl/spherical_circle.py#L36)) repacks successful `LTDResult`s into legacy `(lat, lon, rtt, radius_km, radius_rad)` tuples and calls `circle_intersections` ([geometry.py:317](../scripts/framework/geometry.py#L317)) with `preprocess=enable_circle_filter` (vanilla combo sets it true).

Inside `circle_intersections`:

1. Optional **redundant-circle preprocessing** ([line 273](../scripts/framework/geometry.py#L273)): if disk A fully contains disk B, A is dropped (the smaller is the binding constraint). Determinism: input-order.
2. **Pairwise spherical-cap boundary intersection** for every pair of disks: solve for the two great-circle crossings of disk_i and disk_j on the unit sphere ([line 345-377](../scripts/framework/geometry.py#L345-L377)).
3. **Inside-all filter** ([line 379-387](../scripts/framework/geometry.py#L379-L387)): keep only crossings that satisfy `haversine(point, c_k) ≤ d_k` for every disk k. Survivors are the vertices of the feasible region's boundary.
4. Single-disk degeneracy: 4 evenly-spaced perimeter points ([line 340-342](../scripts/framework/geometry.py#L340-L342)).

Return: `list[Coord]` of feasible-region vertices, wrapped as `MTLResult(success=True, intersection=...)`. If the inside-all filter empties out, `MTLResult(success=False, error=NO_INTERSECTION)`.

### Stage 4 — Centroid (`boundary_vertex_mean`)

`BoundaryVertexMeanCTR._select_centroid` ([ctr/boundary_vertex_mean.py:29](../scripts/framework/v2/ctr/boundary_vertex_mean.py#L29)) takes `MTLResult.intersection`. For `SphericalCircleMTL` this is `list[Coord]`, so it hits the second branch:

```python
verts = [(c.lat, c.lon) for c in intersection]
lat, lon = arithmetic_mean_centroid(verts)
CTRResult(success=True, tg_coord=Coord(lat, lon))
```

`arithmetic_mean_centroid` ([geometry.py:63](../scripts/framework/geometry.py#L63)) is a plain coordinate average — **not** an area centroid, **not** a great-circle-aware mean. Asymmetric regions (long thin slivers from co-linear VPs) bias the prediction toward whichever vertex cluster is densest, often near the closest VP.

### Stage 5 — Fallback

`model.geolocate` ([model.py:153](../scripts/framework/v2/model.py#L153)): if MTL or CTR fails and `enable_fallback=True`:

```python
nearest = min(obs, key=lambda x: x[2])      # smallest rtt
return GeoResult(coord=nearest[1], status=FALLBACK, error=last_error, ...)
```

So `status=FALLBACK` rows in `targets.parquet` are literally shortest-ping predictions.

---

## Where each observed pathology enters

1. **CBG ≤ shortest-ping** — `low_envelope` produces an **upper bound only**; the feasible region usually contains the closest VP. Mean of boundary vertices leans toward that VP, ≈ fallback's pick. SUCCESS error CDF tracks FALLBACK CDF.
2. **Tighter constraints → empty intersection** — any single under-predicting constraint trims away every pairwise crossing in `inside_all`. Annulus methods have two opportunities per VP (inner + outer) to violate, multiplying the risk.
3. **Octant / Spotter low SUCCESS** — same mechanism, compounded. Once you intersect annuli (`planar_annulus`), one VP whose `[lower, upper]` doesn't bracket the truth shrinks or empties the region. `EMPTY_REGION` / `NO_INTERSECTION` dominate their `targets.parquet` error column.

Next deep-dives (queued):
- Octant LTD failure modes — `bounded_spline.py`, especially the δ-band search.
- Octant/Spotter MTL — `planar_annulus.py`, polygon arithmetic and its empty-result paths.
- CTR sensitivity — boundary-vertex mean vs geometric centroid on thin slivers.
