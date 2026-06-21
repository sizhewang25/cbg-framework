# Cluster answer-space — CBG classification eval flow

**Date:** 2026-06-21
**Branch:** `main`
**Context:** A second finite answer space for scoring CBG, alongside the
closest-airport metric (`notes/2026-06-18-closest-airport-eval-decisions.md`).
Instead of snapping to airport hubs, the answer space is built **from the ground
truth itself**: all candidate truth coordinates are clustered into coherent
≤R-radius regions, and each region's centroid is one answer-space point. CBG
accuracy is then a **classification** problem — a prediction snaps to its nearest
centroid and is correct when that is the same centroid the truth snaps to.

Why this over airports: airport snapping loses place identity (cities1000 city
match survives the snap only ~10–13%, see
`notes/2026-06-19-reverse-geocoder-granularity.md` and the
`finding_geonames_city_too_granular_for_metro` memory). Clustering the truth
makes the answer space self-defined and tunable by a single radius knob.

---

## Pipeline (three stages, decoupled)

```
targets.csv ──▶ [1] cluster_ground_truth ──▶ clusters.csv / assignments.csv / meta.json
                                                   │
                                                   └─▶ [2] plot_ground_truth_clusters  (map + stats of the answer space)

benchmark run ──▶ [1'] cli cluster-eval ──▶ <run>/<source>/<setup>/{targets,vps}.csv
                          (merge folds + cluster)   clusters/  (+ clusters/geo/<l>/<v>/)
                                                   │
                                                   └─▶ [3] plot_cluster_cdf / plot_cluster_match_bars  --clusters-dir → scores CBG
```

The canonical pipeline is **materialized**: `cluster-eval` (stage [1']) clusters
each benchmark run's pooled truth once and persists it next to the folds, and the
plots read it via `--clusters-dir` (stage [3]) — the single source of truth,
orchestrated by `scripts/analysis/cluster.smk`. `cluster_ground_truth` (stage [1])
is the same clustering engine `cluster-eval` calls, also usable standalone on any
target catalog; `plot_ground_truth_clusters` (stage [2]) visualizes its results.
Without `--clusters-dir` the plots fall back to clustering in process.
**Keep `--radius-km` identical across stages** — it is the only knob that sets
answer-space granularity.

---

## [1] Clustering — `scripts/benchmark/v2/sources/cluster_ground_truth.py`

```bash
python -m scripts.benchmark.v2.sources.cluster_ground_truth \
    --targets datasets/ripe_atlas/asn_corpora/targets.csv --radius-km 50
```

- **Input:** any csv/json with `target_id/target_lat/target_lon`.
- **Method:** complete-linkage agglomerative over a precomputed haversine
  distance matrix (deterministic, no `k`, singletons fall out naturally). The
  scoring-relevant bound is the **centroid radius** (max member→centroid ≤ R),
  which complete linkage does NOT bound directly (it bounds diameter), so two
  stages: (a) complete-linkage at `distance_threshold = 2R` builds the largest
  candidate regions; (b) any region whose centroid radius still exceeds R is
  recursively 2-way bisected until it complies. Centroids are spherical means.
- **Why not k-means:** k-means optimizes variance for a fixed `k` with no radius
  concept; a cap can only be bolted on by searching `k` (non-monotone,
  over-splits dense metros, forces singletons by brute force). See the
  `project_answer_space_clustering` memory.
- **Outputs** (default `datasets/<targets-stem>/clusters/`):
  - `clusters.csv` — answer space: `cluster_id, centroid_lat, centroid_lon,
    n_members, radius_km, diameter_km, is_singleton`
  - `assignments.csv` — `target_id, target_lat, target_lon, cluster_id,
    dist_to_centroid_km`
  - `meta.json` — `radius_km, n_targets, n_clusters, n_singletons` (carries the
    cap R, which is not recoverable from the CSVs alone)

RIPE asn_corpora @R=50: **713 coords → 257 regions**, 143 singletons (55.6% of
regions / 20.1% of coords), centroid radius capped at exactly 50.0 (p95 38.7).

## [2] Answer-space visualization — `scripts/visualization/cluster/plot_ground_truth_clusters.py`

```bash
python -m scripts.visualization.cluster.plot_ground_truth_clusters \
    --clusters-dir datasets/ripe_atlas/asn_corpora/clusters
# optional: --radius-km (override meta.json), --extent LON0 LON1 LAT0 LAT1, --out
```

Reads the stage-[1] results dir (clusters.csv + assignments.csv + meta.json) and
never recomputes. Map = members colored by region, grey singletons, R-radius
geodesic footprints; side panels = region-size histogram + centroid-radius CDF
(cap utilization). Output default `scripts/visualization/cluster/outputs/`.

## [1'] Materialize the answer space — `cli cluster-eval`

Decoupled postprocessing (like `airport-eval`/`geo-eval`): once per `(source,
setup)` of a run, merge the unique targets (from each fold's `targets.parquet`)
and VPs (from each fold's inputs `vp_configs.parquet`) → `targets.csv` + `vps.csv`
at the folds' parent dir, reverse-geocode the targets for geo labels, then
cluster via `cluster_ground_truth` → `clusters/` (global) or
`clusters/geo/<level>/<value>/` (a per-geo subset, the chosen behavior).

```bash
python -m scripts.benchmark.v2.cli cluster-eval --run-id global_as16509_final --radius-km 50
python -m scripts.benchmark.v2.cli cluster-eval --run-id global_as16509_final --radius-km 50 \
    --geo-level continent --geo-value Europe   # → clusters/geo/continent/Europe/
```

Writes to `<v2_outputs>/<run_id>/<source>/<setup>/`. Global @R=50 reproduces
713→257; the Europe subset gives 415→120.

## [3] CBG classification eval — `scripts/analysis/`

Classify by nearest-centroid Voronoi: prediction and truth each snap to their
nearest centroid, **match = same centroid** (so a perfect prediction is always
correct). Error quantity = great-circle distance from the prediction to the
*truth's* centroid. Pools SUCCESS+FALLBACK across folds. With `--clusters-dir`
the plots read the materialized answer space (auto-resolving the per-geo subset
when `--geo-level/--geo-value` is active); without it they cluster in process.

```bash
CDIR=scripts/benchmark/v2/outputs/global_as16509_final/ripe_atlas_asn_corpora/probes_to_anchors/clusters

# per-combo CDFs (all / matched / mismatched), accuracy + within-R in the box/legend
python -m scripts.analysis.plot_cluster_cdf \
    --run-dir scripts/benchmark/v2/outputs/global_as16509_final --clusters-dir $CDIR --radius-km 50

# cross-combo accuracy ranking bars (+ within-R markers)
python -m scripts.analysis.plot_cluster_match_bars \
    --run-dir scripts/benchmark/v2/outputs/global_as16509_final --clusters-dir $CDIR --radius-km 50
```

`--config <yaml>` also works (run_id resolves the dir). Outputs under
`scripts/analysis/outputs/<run_id>/[geo/<l>/<v>/]cluster/{cdf/,...}`.

## Orchestration — `scripts/analysis/cluster.smk`

Wires [1'] → [2]/[3] as a DAG: `cluster_eval` (global + one per `cluster_geos`
entry) → `plot_cluster_match_bars` + `plot_cluster_cdf` (consuming `--clusters-dir`).

```bash
snakemake -s scripts/analysis/cluster.smk --configfile <run>.yaml -j 2
# config: run_id, source, setup, radius_km, cluster_geos: [{level, value}, ...]
```

Run the 6 `_final` AS configs:
```bash
for run in europe_as3209_final europe_as3215_final global_as16509_final \
           global_as31898_final north_america_as7018_final north_america_as7922_final; do
  python -m scripts.analysis.plot_cluster_cdf        --run-dir scripts/benchmark/v2/outputs/$run --radius-km 50
  python -m scripts.analysis.plot_cluster_match_bars --run-dir scripts/benchmark/v2/outputs/$run --radius-km 50
done
```

---

## Results snapshot — 6 `_final` configs @R=50 (713 targets → 257 centroids)

Accuracy = prediction snaps to the truth's centroid.

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
baseline (1/257 ≈ 0.4%). `within_R` tracks accuracy a few points lower
(predictions can hit the right centroid yet land just beyond R of it).

---

## Design notes / gotchas

- **Singletons** are the "no nearby ground truth" case — radius 0, scored by the
  nearest-ground-truth fallback with the error distance as confidence.
- **Answer-space size scales granularity:** smaller `--radius-km` → more
  centroids → harder classification. The size is printed each run
  (`N targets → M centroids`).
- **Cost:** clustering builds an NxN haversine matrix; fine for ~1k targets,
  watch memory for many thousands.
- **Bug fixed during dev:** `groupby().indices` returns positions into the
  dropna'd frame — index lat/lon from that same reset frame, not the original.
