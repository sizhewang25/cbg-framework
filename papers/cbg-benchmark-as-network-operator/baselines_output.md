# Shortest-Ping Baseline Results

Computed with `python -m scripts.analysis.plot_cluster_match_bars --run-dir <run_dir> --radius-km 50`
from the project root. Answer space: R=50 km complete-linkage clusters built from each run's pooled ground truth.
Baseline = snap the min-RTT VP to the nearest centroid; accuracy over **all** targets (n = total target set).

| run_id | baseline_acc | baseline_within_r | n | n_centroids |
| --- | ---: | ---: | ---: | ---: |
| global_as16509_final | 23.4% | 21.7% | 713 | 257 |
| north_america_as7018_final_us | 39.6% | 31.2% | 96 | 32 |
| europe_as3209_final_de | 50.0% | 49.0% | 96 | 21 |
| north_america_as7018_final | 5.3% | 4.2% | 713 | 257 |
| europe_as3209_final | 6.3% | 6.2% | 713 | 257 |

## Notes

- All 5 runs resolved their inputs directory automatically via `scripts/benchmark/v2/inputs/ripe_atlas_asn_corpora/<run_id>/probes_to_anchors/` (fold-parent mode, one `eval_observations.parquet` per fold). No `--inputs-dir` override was needed.
- The two geo-filtered runs (`_us`, `_de`) use a subset of ~96 targets and correspondingly fewer centroids (32 and 21 respectively), so their baseline rates are not directly comparable to the full-target runs (n=713, 257 centroids).
- `north_america_as7018_final` and `europe_as3209_final` both use the full 713-target, 257-centroid answer space with out-of-distribution VPs; their baseline accuracy is very low (5–6%) because the AS7018/AS3209 VP locations are geographically concentrated and rarely co-locate with a target's centroid across the full global target set.
- Machine-readable CSVs (one row per combo + a `shortest_ping_baseline` row) were written alongside the PNG figures in `scripts/analysis/outputs/<run_id>/cluster/`.
