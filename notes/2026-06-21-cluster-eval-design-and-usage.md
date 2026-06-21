# Cluster-based CBG classification evaluation — design & usage

**Date:** 2026-06-21
**Branch:** `main`
**Status:** current design (supersedes the earlier "answer-space eval flow" note).

A second way to score CBG, alongside the closest-airport metric
(`notes/2026-06-18-closest-airport-eval-decisions.md`). Instead of demanding
city-level `error_km`, we define a **finite answer space from the ground truth
itself** and turn accuracy into a **classification** problem: every ground-truth
coordinate is grouped into a coherent ≤R-radius region, each region's centroid
is one answer-space point, and a CBG prediction is *correct* when it snaps to the
same centroid the truth snaps to.

Why clustering rather than airports: snapping to airport hubs loses place
identity (GeoNames cities1000 city-match survives the snap only ~10–13%; see
`notes/2026-06-19-reverse-geocoder-granularity.md` and the
`finding_geonames_city_too_granular_for_metro` memory). Clustering the truth makes
the answer space self-defined and tunable by a single radius knob.

---

## 1. Design

### 1.1 Answer-space construction
`scripts/benchmark/v2/sources/cluster_ground_truth.py`

- **Complete-linkage agglomerative** over a precomputed haversine distance
  matrix — deterministic, no `k`, isolated points fall out as **singletons**
  (radius 0 = the "no nearby ground truth" case).
- The scoring-relevant bound is the **centroid radius** (max member→centroid ≤
  R), which complete linkage does NOT bound directly (it bounds *diameter*), and
  the two diverge — a diameter-2R region can have a mean-centroid radius ~72 km
  at R=50. So two deterministic stages: (a) complete-linkage at
  `distance_threshold = 2R` builds the largest candidate regions; (b) any region
  whose centroid radius still exceeds R is recursively 2-way bisected until it
  complies. Centroids are spherical means (antimeridian-safe).
- **Why not k-means:** it optimizes variance for a fixed `k` with no radius
  concept; capping radius needs a non-monotone search over `k` that over-splits
  dense metros and forces singletons by brute force. See the
  `project_answer_space_clustering` memory.
- RIPE asn_corpora @R=50: **713 coords → 257 regions**, 143 singletons (55.6% of
  regions / 20.1% of coords), centroid radius capped at exactly 50.0 (p95 38.7).

### 1.2 Classification & metrics
`scripts/analysis/plot_cluster_cdf.py`, `plot_cluster_match_bars.py`

- **Nearest-centroid Voronoi**: prediction and truth each snap to their nearest
  centroid (haversine `BallTree`); **match = same centroid**. A perfect
  prediction (pred = truth) is therefore always correct — the same property the
  airport metric gives hubs.
- **error-to-centroid** = great-circle distance from the prediction to the
  *truth's* centroid (the correct answer point). For matched rows it equals the
  prediction's own snap distance; for mismatched rows it is strictly larger (the
  misclassification penalty).
- Headline numbers per combo: **classification accuracy** (matched share),
  **within-R** rate (error-to-centroid ≤ R — the point-estimate scoring rule),
  per-cohort percentiles, and the **answer-space floor** (truth→nearest centroid,
  the resolution the metric quantizes away). Pools SUCCESS+FALLBACK across folds.

### 1.3 Materialization & decoupling
The answer space is **materialized once per `(source, setup)`** and the plots
consume it, rather than re-clustering inside every plot.

- `cli cluster-eval` clusters the run's pooled truth and persists `clusters/`
  next to the folds; the plots read it via `--clusters-dir` (single source of
  truth). Without `--clusters-dir` they fall back to clustering in process.
- **Scope = `(run, source, setup)`** because the answer space must match the
  targets actually evaluated (a foreign catalog would leave some truths far from
  every centroid, breaking "perfect prediction is correct"). Different
  runs/datasets get their own answer space automatically.
- **Geo = per-geo answer space.** A `--geo-level/--geo-value` subset is clustered
  on its own (e.g. Europe → 415 targets → 120 centroids) and stored under
  `clusters/geo/<level>/<value>/`; the plots auto-resolve that subdir when the
  matching filter is active. (Chosen over a fixed global space so a geo slice is
  scored against its own regions.)

---

## 2. Components

| file | role |
|---|---|
| `scripts/benchmark/v2/sources/cluster_ground_truth.py` | clustering engine; standalone CLI (CSV catalog → `clusters/` + `meta.json`) |
| `scripts/benchmark/v2/cli.py` → `cluster-eval` | merge folds + cluster → persisted answer space per `(source, setup)` |
| `scripts/analysis/plot_cluster_cdf.py` | per-combo all/matched/mismatched error-to-centroid CDFs |
| `scripts/analysis/plot_cluster_match_bars.py` | cross-combo accuracy ranking bars (+ within-R markers) |
| `scripts/visualization/cluster/plot_ground_truth_clusters.py` | map + size/radius stats of an answer space (reads a results dir) |
| `scripts/analysis/cluster.smk` | orchestration: cluster-eval (global + per-geo) → bars + cdf |

---

## 3. Usage

### 3.1 One run, via Snakemake (recommended)
Existing per-run configs already work (`cluster.smk` reads `run_id/source/setup`;
`radius_km` defaults to 50, `cluster_geos` to none):

```bash
snakemake -s scripts/analysis/cluster.smk \
    --configfile scripts/analysis/config/global_as16509_final.yaml -j 2
```
Add per-geo subsets in the config:
```yaml
radius_km: 50
cluster_geos:
  - {level: continent, value: Europe}
  - {level: country,   value: US}
```

### 3.2 Manual steps
```bash
# materialize the answer space (global + optional geo subset)
python -m scripts.benchmark.v2.cli cluster-eval --run-id global_as16509_final --radius-km 50
python -m scripts.benchmark.v2.cli cluster-eval --run-id global_as16509_final --radius-km 50 \
    --geo-level continent --geo-value Europe

# plot against it
CDIR=scripts/benchmark/v2/outputs/global_as16509_final/ripe_atlas_asn_corpora/probes_to_anchors/clusters
python -m scripts.analysis.plot_cluster_match_bars --run-dir <run> --clusters-dir $CDIR --radius-km 50
python -m scripts.analysis.plot_cluster_cdf        --run-dir <run> --clusters-dir $CDIR --radius-km 50
# geo: add --geo-level continent --geo-value Europe (auto-resolves clusters/geo/...)
```

### 3.3 Standalone (any target catalog, no benchmark run)
```bash
python -m scripts.benchmark.v2.sources.cluster_ground_truth \
    --targets datasets/ripe_atlas/asn_corpora/targets.csv --radius-km 50
python -m scripts.visualization.cluster.plot_ground_truth_clusters \
    --clusters-dir datasets/ripe_atlas/asn_corpora/clusters
```

**Keep `--radius-km` identical** across cluster-eval and the plots — it is the
only knob that sets answer-space granularity.

---

## 4. Outputs (all gitignored)

```
scripts/benchmark/v2/outputs/<run_id>/<source>/<setup>/
    targets.csv  vps.csv                         (merged unique entities across folds)
    clusters/{clusters,assignments}.csv  meta.json   (global answer space)
    clusters/geo/<level>/<value>/...             (per-geo answer spaces)
scripts/analysis/outputs/<run_id>/[geo/<l>/<v>/]cluster/
    <run_id>_cluster_accuracy.png                (bars)
    cdf/<combo>_cluster_cdf.png                  (one per combo)
```

---

## 5. Results snapshot — 6 `_final` configs @R=50 (713 → 257 centroids)

Verified end-to-end through `cluster.smk` (4/4 rules each).

| config | best combo | acc | worst combo | acc |
|---|---|---|---|---|
| global_as16509 | octant_cbg_hull_geo | 30.0% | spotter_cbg_c100 | 2.8% |
| global_as31898 | octant_cbg_top | 29.3% | spotter_cbg_c100 | 2.4% |
| na_as7922 | vanilla_cbg | 10.4% | spotter_cbg_top_geo | 1.3% |
| na_as7018 | octant_cbg_hull | 7.4% | spotter_cbg_top_geo | 1.3% |
| eu_as3209 | vanilla_cbg | 8.3% | octant_cbg | 2.0% |
| eu_as3215 | vanilla_cbg | 6.5% | spotter_cbg_top_geo | 1.3% |

Patterns (consistent with the CBG-limitations framing,
`project_cbg_bench_framing`): global ≫ NA ≈ EU; `octant_*_hull/top` and plain
`vanilla` lead while calibrated `spotter_c80/c100` collapse near the random
baseline (1/257 ≈ 0.4%). `within_R` tracks accuracy a few points lower.

---

## 6. Gotchas / knobs

- **Singletons** = isolated truths (radius 0); scored by nearest-centroid with
  the error distance as confidence.
- **Granularity:** smaller `--radius-km` → more centroids → harder. The size is
  printed each run (`N targets → M centroids`).
- **Cost:** clustering builds an NxN haversine matrix — fine for ~1k targets,
  watch memory for many thousands.
- **Geo path-safety:** continent/country values with spaces are stored with `_`
  (e.g. `North_America`); `cluster.smk` round-trips this for the `--geo-value`.
- **Determinism:** complete-linkage + split-repair is deterministic; precomputed
  `--clusters-dir` and the in-process fallback produce identical centroids at the
  same R on the same targets.
