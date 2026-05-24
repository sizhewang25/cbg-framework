# VP Selection — Cho 2024 Agreement Replication

Snakemake-orchestrated pipeline that replicates Cho et al. 2024's landmark-selection-effectiveness measurement on our IMC 2023 RIPE Atlas data (probes-as-VPs, anchors-as-targets), adapted to hard-GT data via per-anchor fake-country injection.

See [../../tasks/20260523-vp-selection-agreement-replication/](../../tasks/20260523-vp-selection-agreement-replication/) for the task plan and decision log.

## What this measures

The **ICLab CBG-style accept/reject verifier** (Niaki 2020 §App B, reused by Cho 2024 §III):

```
for each landmark ℓ measuring target T:
    d   = great-circle distance from ℓ to the nearest point on T's claimed country's border
    owtt = rtt(ℓ, T) / 2                                # one-way time, ms
    if 2 * d / rtt > S_calibrated:                       # implied propagation speed exceeds limit
        return REJECT
return ACCEPT
```

Cho asks: given the full landmark pool's verdict per (target, claim), how small a subset can preserve those verdicts? The smaller the subset that achieves identical verdicts, the more efficient the deployment.

Her data has natural REJECT cases — commercial VPNs sometimes lie about their location. Our anchors don't lie, so we **inject false country claims** into 30% of anchors before the sweep (one claim per target; uniform-random wrong country). The full pool then organically rejects most fakes, and the question becomes: does the subset preserve those rejections?

## Pipeline (4 stages, snakemake-orchestrated)

```
(a) calibrate ──► speed_calibration.json
(b) claims    ──► claims.parquet (one (target, claim, is_real) per anchor)
(c) borders   ──► border_distances.parquet (probe × country → km, AEQD-projected)
(d) sweep     ──► agreement_rows.parquet  + agreement_summary.json + agreement_curve.png
                  ▲
                  ├── selections/{strategy}_{seed}.parquet  ← 5 strategies × N seeds
                  ├── claims.parquet
                  ├── border_distances.parquet
                  └── speed_calibration.json
```

Run end-to-end on 8 cores:

```bash
snakemake -s scripts/vp_selection/Snakefile --cores 8
```

Inspect the DAG without running:

```bash
snakemake -s scripts/vp_selection/Snakefile --dry-run
```

Re-run only the last stage (after changing a strategy or seed count):

```bash
snakemake -s scripts/vp_selection/Snakefile --cores 8 --forcerun sweep
```

## The 5 selection strategies

| Name | Cho's name | Output shape | Default seeds | Algorithm |
|---|---|---|---|---|
| `random` | Random | K-subset per K (independent) | 100 | Uniform `rng.sample(pool, K)` |
| `cluster_as` | Clustering (AS) | K-subset per K (independent) | 100 | Stratified random per ASN: even base + round-robin remainder |
| `cluster_city` | Clustering (city) | K-subset per K (independent) | 100 | Same on city |
| `h1_as` | Hybrid 1 (AS) | Full N-sequence | 1 | Greedy Prim maximizing Σ pair-distance with ASN-cluster preference until all clusters covered |
| `h2_as` | Hybrid 2 (AS) | Full N-sequence | 100 | Random-100 seed + h1_as continuation; one sequence per random-100 init |

Edit [configs/strategies.yaml](configs/strategies.yaml) to add/remove strategies or change seed counts.

## How the "closest border" check works (AEQD projection)

The verifier needs `nearest_border_distance_km(probe, claimed_country)` — the great-circle km from a probe to the nearest point on the claimed country's polygon boundary.

The trick lives in [country_borders.py::nearest_border_distance_km](country_borders.py):

1. Load country polygons from a Natural Earth shapefile via `geopandas.read_file(...)`. Each country is a `shapely.MultiPolygon` in WGS84 (lon, lat).
2. If `polygon.contains(Point(lon, lat))` → return 0.0 (probe sits inside the country).
3. Otherwise: project the polygon into an **azimuthal-equidistant CRS centered on the probe**:

   ```python
   aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=km")
   polygon_aeqd = shapely.ops.transform(transformer.transform, polygon)
   return polygon_aeqd.distance(Point(0, 0))
   ```

   AEQD's defining property: **distances from the projection center are preserved as true great-circle km**. After we center on the probe, the probe is at the origin (0, 0) and every point on the polygon's border is at coordinates `(x_km, y_km)` — its actual km-distance from the probe in some direction. `shapely.distance(polygon, origin)` then returns the minimum Euclidean km to the polygon, which **is** the great-circle distance.

This is more accurate than the alternatives:

| Alternative | Tradeoff |
|---|---|
| Use country centroid + haversine | Wrong direction; bias up to several thousand km for large countries |
| Discretize boundary into vertices, take min haversine | Loses precision on long edges; can miss mid-segment nearest points |
| Bounding box in lon/lat | Distorted at high latitudes |

**Precomputing the table**: for the smoke run with ~10K probes × ~95 countries appearing as claims, we materialize ~950K (probe, country) → km entries via [borders_precompute.py](borders_precompute.py). Snakemake's `borders_shard` rule splits the probe pool across N shards (default 8) and runs the projections in parallel; `borders_merge` concatenates the per-shard parquets into one lookup table.

**Limitations**:
- Natural Earth low-res (110m) polygons: borders are accurate to ~10-50 km. Fine for speed-of-light checks; not for sub-km adjudication.
- AEQD is per-call; ~30 ms per projection in Python. Parallel snakemake scatter compresses ~30 min serial → ~4 min on 8 cores.
- Antimeridian / poles: handled correctly by AEQD (the projection unwraps relative to the center).

## Where the Natural Earth shapefile comes from

The pipeline uses the `pyogrio` package's bundled Natural Earth test fixture at `.venv/.../pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp` (ISO_A3-keyed, 177 countries). The `country_borders.load_country_polygons` loader auto-detects ISO_A2 vs ISO_A3 columns; the agreement runner translates anchor country codes via the vendored [upstream_csv/iso3166.csv](upstream_csv/iso3166.csv) as needed.

**For production / long-term use**: download the proper Natural Earth shapefile from <https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip> and unpack into `datasets/static_datasets/naturalearth_lowres/`. The loader picks it up automatically when present.

## Two metrics reported

1. **Agreement vs full pool** — Cho's metric. Per (strategy, K): % of targets where `sub_verdict == full_pool_verdict`. Curve approaches 1.0 as K → N (full pool by definition).
2. **Detection rate vs ground truth** — extra metric we get because we control the fake injection. Per (strategy, K): TPR (% of fakes correctly rejected) and FPR (% of reals wrongly rejected). The full-pool baseline tells us how strong the verifier is at all; the per-K curve tells us how much detection power the subset preserves.

The `agreement_summary.json` reports both per (strategy, k); the PNG plots three panels: agreement-vs-K, TPR-vs-K, FPR-vs-K.

## Two output shapes inside the selection step

- **Sampling strategies** (`random`, `cluster_as`, `cluster_city`) — one selection rule produces a parquet with `(strategy, seed, k, vp_id)` rows. For each K, the K rows for that (strategy, seed, k) ARE the K-subset.
- **Sequence strategies** (`h1_as`, `h2_as`) — one selection rule produces `(strategy, seed, position, vp_id)`. K-subset = rows with `position ≤ K`.

The `sweep` stage dispatches: sampling strategies need one verifier call per (K, seed, target, claim); sequence strategies need one first-violator scan per (seed, target, claim), and `sub_verdict(K)` for every K is derived from `first_violator_k`.

## Files

```
scripts/vp_selection/
├── README.md                     ← you are here
├── Snakefile                     ← workflow definition
├── configs/
│   └── strategies.yaml           ← strategy lineup + seed counts + pipeline knobs
├── calibrate_speed.py            ← Stage a CLI
├── claims.py                     ← Stage b CLI
├── borders_precompute.py         ← Stage c CLI (shard + merge modes)
├── selection_runner.py           ← Stage d.1 CLI (one rule per strategy×seed)
├── agreement.py                  ← Stage d.2 CLI (sweep + plot)
├── strategies.py                 ← select_vps + sample_vps + cluster algorithms
├── pair_distances.py             ← geodesic pair-distance generator
├── country_borders.py            ← AEQD nearest-border + polygon loader
├── iclab_verifier.py             ← the verifier itself (15 lines)
├── tests/                        ← unit tests (TDD-driven)
├── upstream_py/                  ← pristine Cho 2024 reference code
├── upstream_csv/                 ← pristine Cho 2024 reference data
└── outputs/                      ← gitignored generated artifacts
```

## Reference

Cho, Weinberg, Bhattacharya, Dai, Rauf. *Selection of Landmarks for Efficient Active Geolocation.* TMA 2024 (IFIP). Original code: <https://github.com/grace71/tma24-vp-ls> (vendored under `upstream_py/`).

Niaki et al. 2020 §App B (ICLab) — origin of the accept/reject verifier and the calibrated speed limit. Cho 2024 lifts ICLab's algorithm wholesale.
