# Benchmark World-Map Visualizations

This directory contains interactive Plotly world-map renderers for v2 benchmark
outputs.

## Cluster Classification Map

`cluster_world_map.py` renders a centroid-aware view of benchmark failures. It
reuses the base MTL map geometry, then adds:

- all answer-space centroids;
- truth and prediction centroid cells;
- per-target outcome and failure mechanism from `per_target_failures.parquet`;
- optional nearest-hub Voronoi cells clipped to a named landmass.

Run from the repository root with the project virtualenv:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml \
  --combo vanilla_cbg \
  --landmass "North America"
```

The output is written under:

```text
scripts/visualization/benchmark/v2/outputs_cluster/<run_id>/<combo>_cluster_map.html
```

For a scratch verification run that does not touch repository outputs, set
`--out-dir` to `/tmp`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml \
  --combo vanilla_cbg \
  --landmass "North America" \
  --out-dir /tmp/cbg-cluster-map-test
```

## Inputs

The renderer expects:

- benchmark `targets.parquet` files under `scripts/benchmark/v2/outputs/` or
  `scripts/benchmark/v2/outputs_partvp/`;
- cluster-eval `clusters/clusters.csv` when available;
- failure attribution at
  `scripts/analysis/partvp/outputs/analysis_fail/per_target_failures.parquet`.

If the attribution table is missing, build it first:

```bash
.venv/bin/python -m scripts.analysis.partvp.characterize_failures
```

## Viewing

Open the generated HTML in a browser. If you want feasible-region polygons to
load reliably, serve the output tree through a local HTTP server because
`file://` blocks the page's `fetch()` calls:

```bash
cd scripts/visualization/benchmark/v2/outputs_cluster
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/<run_id>/<combo>_cluster_map.html
```

## Voronoi Overlay

Pass `--landmass` to include nearest-hub cells:

```bash
--landmass "North America"
--landmass US
--landmass France
--landmass Europe
```

The landmass name is resolved from Natural Earth admin-0 data through Cartopy.
The overlay is for visual diagnosis of nearest-centroid snapping boundaries; it
is a lon/lat visualization-grade partition, not an equal-area construction.

## Batch Rendering

Use the Snakemake workflow to render the textbook cluster maps:

```bash
snakemake -s scripts/visualization/benchmark/v2/cluster_world_map.smk -j 4
```

The workflow uses `.venv/bin/python`, builds the attribution table if needed,
and writes HTML plus static feasible-region JSON files under
`outputs_cluster/`.
