# Region-overlap confidence levels — design & usage

**Date:** 2026-06-22
**Branch:** `main`
**Status:** current. Code `scripts/analysis/partvp/region_confidence.py`; results
`scripts/analysis/partvp/analysis/region_confidence.csv`,
`scripts/analysis/partvp/data/region_confidence_all.parquet`.

An **inference-observable** confidence signal for a CBG prediction: one an operator can compute from
the RTT measurements and the (known) answer space *alone*, without ground truth. It answers "can I
trust this prediction?" by asking how the CBG **MTL feasible region** sits relative to the
**answer-space cells** (clusters). It is the deployable replacement for the truth-anchored
"answer-space isolation" lever in the participating-VP study
(`papers/cbg-benchmark-as-network-operator/participating_vp_findings.md` §5/§7).

Why this rather than a landmass/geographic-feasibility filter (an earlier idea, dropped): the answer
space already encodes the spatial constraint, so we reuse CBG's *own* geometry instead of bolting on
an external plausibility gate. See the same paper's `discussion.md` §6.3.

---

## 1. Concept

For each prediction compute two **observables** (no truth needed):

- **`n_hit`** — number of distinct answer-space cluster disks (uniform radius **R**, default 50 km)
  that the MTL feasible region overlaps.
- **`d_hub`** — great-circle distance from the point estimate to its nearest cluster centroid (the
  "hub" it would snap to).

Intuition: a feasible region landing inside *exactly one* answer cell is unambiguous — that cell is
the answer regardless of where the point estimate is. A region straddling *many* cells, or touching
*none*, is not. This turns the geometry CBG discards (it keeps only the centroid point estimate) into
a confidence ranking.

### Confidence levels (priority order — evaluate top-down)

| level | rule | meaning |
| ----- | ---- | ------- |
| **L1** highest | `n_hit == 1` | region in exactly one cell (regardless of `d_hub`) |
| **L2** high    | `n_hit > 1` and `d_hub < R` | ambiguous region, point inside a hub |
| **L3** mid     | `n_hit > 1` and `d_hub ≥ R` | region present, point far from any hub |
| **L0** low/fail| `n_hit == 0` (empty / no overlap) **or** status ∈ {FALLBACK, ERROR} | no usable region |

L1 wins on intersection-count: a single-cell region is treated as unambiguous even if the point
estimate is >R from that cell's centroid. L0 is observably *one* bucket; ground truth is used **only
in validation** to split it into "snaps right anyway" vs "fail".

---

## 2. Design

### 2.1 Reconstructing the MTL region (the key trick)
The benchmark persists only `mtl_intersection_kind` (a type string), not the region geometry. But the
participating-VP instrumentation persists `mtl_participants` — for each VP that survived the
redundant-disk filter: `vp_lat`, `vp_lon`, `echoed_lower_km`, `echoed_upper_km` (the predicted
distance band). Each participant therefore defines an **annulus**, and the region is their
intersection. We rebuild it offline with the octant unweighted region builder:

```
compute_feasible_region_unweighted([AnnularConstraint(
    landmark_lat=vp_lat, landmark_lon=vp_lon,
    inner_radius_km=echoed_lower_km, outer_radius_km=echoed_upper_km, weight=1.0, ...)])
```
(`scripts/libs/octant/octant_geolocation.py`). Circle methods (vanilla/spotter) are the `lower=0`
special case (a disk), so the same call generalizes across all CBG variants. The region is built in
octant's equirectangular (lon=x, lat=y) planar frame.

### 2.2 Counting overlaps (`n_hit`) without cross-frame polygon math
Globally-dispersed cluster disks cannot share one planar frame, so we **sample** points inside the
region (`sample_points_in_region`, returns `[lat, lon]`), then for each sampled point query a
haversine `BallTree` of centroids with `query_radius(r = R / EARTH_RADIUS_KM)`. `n_hit` = size of the
union of returned centroid indices. Only the region *shape* carries planar error; the cell test
itself is exact haversine. Empty/None region ⇒ `n_hit = 0`.

### 2.3 Validation against truth
The script also recomputes the §1.2 tier labels (match = pred's nearest centroid == truth's nearest
centroid; within_r = error-to-centroid ≤ R) so the observable levels can be **calibrated** against
true outcomes in the same run. This mirrors the labeling in `extract_features.py`.

---

## 3. Requirements (what a dataset must provide)

To apply this to another dataset / run you need:

1. **`targets.parquet` with `mtl_participants` populated** — i.e. the run was executed with the
   participating-VP instrumentation (the runner joins surviving VPs back to their LTD distance band).
   Stale runs without this column are skipped with a warning. Columns used: `target_id`,
   `target_lat/lon`, `pred_lat/lon`, `status`, `mtl_participants`.
2. **An answer space** — either a precomputed `clusters/clusters.csv` (`cluster-eval` output) under
   the run, or none (the script clusters the run's pooled ground truth in-process via
   `cluster_ground_truth`, capped at R). See `notes/2026-06-21-cluster-eval-design-and-usage.md`.
3. Standard repo deps (shapely, sklearn, pyarrow) — no new dependencies.

The answer space defines "cells"; choose R to match the metro/region granularity you care about
(50 km = same-metro proxy for RIPE anchors). The same R is used for cluster disks *and* the within-R
tier check, so keep them consistent.

---

## 4. Usage

Single run (writes the per-run calibration CSV + optional per-target parquet):
```
.venv/bin/python -m scripts.analysis.partvp.region_confidence \
  --run-dir scripts/benchmark/v2/outputs/global_as16509_final \
  --out-csv  scripts/analysis/partvp/analysis/region_confidence.csv \
  --out-parquet scripts/analysis/partvp/data/region_confidence_global_as16509_final.parquet
```

Multiple runs into one combined table (repeat `--run-dir`):
```
.venv/bin/python -m scripts.analysis.partvp.region_confidence \
  --run-dir scripts/benchmark/v2/outputs/global_as16509_final \
  --run-dir scripts/benchmark/v2/outputs/global_as31898_final \
  --run-dir scripts/benchmark/v2/outputs_partvp/north_america_as7018_final_us \
  --out-csv scripts/analysis/partvp/analysis/region_confidence.csv \
  --out-parquet scripts/analysis/partvp/data/region_confidence_all.parquet
```

Key options:
- `--combos` (default `vanilla_cbg million_scale_cbg octant_cbg spotter_cbg`) — which CBG variants.
- `--radius-km` (default 50) — cluster-disk radius **and** within-R threshold.
- `--n-samples` (default 600) — region-interior samples for the overlap count (see §6 stability).
- `--clusters-dir` — force a specific answer space; default auto-detects `<run>/…/clusters/clusters.csv`
  and falls back to in-process clustering when absent.
- `--seed` (default 42); `--max-targets` (cap per combo, for a quick smoke test).

**Gotcha:** the participating-VP instrumented runs may live in a *different* outputs root than the
original run (e.g. the regional textbook-4 side runs are under `outputs_partvp/`, not `outputs/`).
Point `--run-dir` at the dir whose `targets.parquet` actually has `mtl_participants`.

---

## 5. Outputs

**`--out-csv`** — one row per `run_id × combo_id × level`:

| column | meaning |
| ------ | ------- |
| `n`, `frac` | count and share of predictions at this level |
| `p_correct` | P(snaps to correct centroid \| level) — the level's **precision** |
| `p_tier1` | P(within-R / Tier-1 \| level) |
| `p_tier2`, `p_tier3` | tier mix within the level |
| `recall_of_correct` | share of *all* truly-correct predictions captured by this level (**coverage**) |

**`--out-parquet`** (optional) — one row per `run_id, combo_id, target_id` with the raw observables
and labels: `n_hit`, `d_hub_km`, `level`, `status`, `match`, `within_r`, `tier`,
`error_to_centroid_km`. This is the feature table a future confidence *model* would consume.

---

## 6. Interpretation guidance

- **Use L1 as the high-confidence gate.** Empirically (RIPE, 6 runs) L1 precision is **0.66–0.91** for
  the three real CBG variants and captures **~30–56%** of all correct answers. Operating rule for any
  dataset: *trust a prediction iff its feasible region falls inside a single answer cell.*
- **Do not treat L1∪L2 as "high".** A multi-cell region snapping to a nearby hub (L2) is **not**
  reliably correct (~0.20–0.28 globally on RIPE): region ambiguity dominates the near-hub signal. L2
  was moderate only for `million_scale` regionally (larger regions, rarely L1). Report L1 alone as the
  confident set.
- **L3 (far point) ≈ low**, mostly wrong-cell.
- **L0 is variant-dependent, not uniformly failure.** A pure-collapse variant (spotter) has ~all mass
  in L0 at ~0.1 precision; a variant with a sane centroid fallback (octant) can still be right
  ~0.3–0.5 of the time at L0. Read L0 precision *per variant*, not globally.
- **Calibration is the deliverable, not a classifier.** The CSV tells you, per variant, how trustworthy
  each observable level is on *your* data. Re-run it on a new dataset before quoting the RIPE numbers —
  precision depends on the variant, the VP fleet geometry, and the answer-space density.
- **A near-total-L0 profile is a red flag** that the variant's bands never isolate a cell on your data
  (the spotter signature; cross-ref `finding_spherical_circle_brittle`).

---

## 7. Caveats

- **Planar shape error.** The region is reconstructed in octant's equirectangular frame, so its shape
  is approximate (worst for spherical-circle methods over long spans); the cell test is haversine.
- **Sampling stability.** Level assignment is **94% stable** between `--n-samples` 300 and 1200 on the
  RIPE global run; disagreements are borderline L1↔L3 (thin regions). Because `n_hit` only *grows* with
  more samples, reported **L1 precision is a mild lower bound** (denser sampling demotes some L1→L3).
  Bump `--n-samples` if you need tighter borderline behavior; 600 is a good default.
- **Scoring rule.** "Correct" uses the point-estimate's nearest centroid for all levels (consistent
  with the existing tier metric). An L1 variant scored against the single *overlapped* cell would be
  marginally cleaner but was not needed (L1 precision already high).
- **Regional answer spaces are small** (tens of cells), so regional level *counts* are noisy — treat
  regional precision as directional.

---

## 8. Cross-refs

- Paper writeups: `papers/cbg-benchmark-as-network-operator/participating_vp_findings.md` §7/§8;
  `discussion.md` §6.3/§6.4.
- Answer space / clustering: `notes/2026-06-21-cluster-eval-design-and-usage.md`;
  `project_answer_space_clustering` memory.
- Region geometry primitives: `scripts/libs/octant/octant_geolocation.py`
  (`compute_feasible_region_unweighted`, `sample_points_in_region`); km→degree circle conversion
  `notes/2026-05-19-km-to-degree-circle-conversion.md`; Sobol sampling
  `notes/2026-05-19-sobol-vs-uniform-sampling.md`.
- Participating-VP instrumentation & features: `scripts/analysis/partvp/extract_features.py`.
