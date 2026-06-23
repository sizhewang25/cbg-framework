# Representative CBG Failure Targets for Map Verification

This report lists representative failed targets from `scripts/analysis/partvp/outputs/analysis_fail/per_target_failures.parquet`. It is intended as a companion to the centroid-aware maps produced by `scripts/visualization/benchmark/v2/cluster_world_map.py`.

## How to verify

1. Render all maps with:

```bash
snakemake -s scripts/visualization/benchmark/v2/cluster_world_map.smk -j 4
```

Or render one map with the per-section command shown below.

2. Serve the map output if you want feasible-region polygons to load:

```bash
cd scripts/visualization/benchmark/v2/outputs_cluster
python -m http.server 8000
```

3. Open `http://localhost:8000/<run_id>/<variant>_cluster_map.html`, set the outcome/mechanism filters, and select the row whose dropdown label contains the `target_id` and `fold` shown here. The map metadata line should show the same mechanism and diagnostic features.

## Selection rule

For each config x variant, mechanisms are ranked by failure count. The report keeps up to the three largest mechanisms, then selects one target closest to that mechanism's median diagnostic value: `avail_min_vp_km` for `NO_PROXIMITY`, `frac_blockers` for `ERRONEOUS_CONTAINMENT`, `part_min_infl` for `RTT_INFLATION`, and `error_to_centroid_km` for `OTHER`. These are representative checks, not worst-case examples.

## Quick Map Commands

### global-global (`global_as16509_final`)

`vanilla_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/benchmark/v2/config/global_as16509_final.yaml \
  --combo vanilla_cbg
```

`million_scale_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/benchmark/v2/config/global_as16509_final.yaml \
  --combo million_scale_cbg
```

`octant_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/benchmark/v2/config/global_as16509_final.yaml \
  --combo octant_cbg
```

`spotter_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/benchmark/v2/config/global_as16509_final.yaml \
  --combo spotter_cbg
```

### europe-europe (`europe_as3215_eu`)

`vanilla_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_eu.yaml \
  --combo vanilla_cbg \
  --landmass Europe
```

`million_scale_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_eu.yaml \
  --combo million_scale_cbg \
  --landmass Europe
```

`octant_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_eu.yaml \
  --combo octant_cbg \
  --landmass Europe
```

`spotter_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_eu.yaml \
  --combo spotter_cbg \
  --landmass Europe
```

### europe-country (`europe_as3215_final_fr`)

`vanilla_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_final_fr.yaml \
  --combo vanilla_cbg \
  --landmass France
```

`million_scale_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_final_fr.yaml \
  --combo million_scale_cbg \
  --landmass France
```

`octant_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_final_fr.yaml \
  --combo octant_cbg \
  --landmass France
```

`spotter_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/europe_as3215_final_fr.yaml \
  --combo spotter_cbg \
  --landmass France
```

### na-na (`north_america_as7018_final_na`)

`vanilla_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml \
  --combo vanilla_cbg \
  --landmass "North America"
```

`million_scale_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml \
  --combo million_scale_cbg \
  --landmass "North America"
```

`octant_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml \
  --combo octant_cbg \
  --landmass "North America"
```

`spotter_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml \
  --combo spotter_cbg \
  --landmass "North America"
```

### na-us (`north_america_as7018_final_us`)

`vanilla_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_us.yaml \
  --combo vanilla_cbg \
  --landmass US
```

`million_scale_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_us.yaml \
  --combo million_scale_cbg \
  --landmass US
```

`octant_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_us.yaml \
  --combo octant_cbg \
  --landmass US
```

`spotter_cbg`:

```bash
.venv/bin/python -m scripts.visualization.benchmark.v2.cluster_world_map \
  --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_us.yaml \
  --combo spotter_cbg \
  --landmass US
```

## Examples

### global-global

- Run ID: `global_as16509_final`
- Config for map renderer: `scripts/benchmark/v2/config/global_as16509_final.yaml`

| variant | mechanism | mech n/share | target_id | fold | outcome | status | err->cell km | nearest VP km | next cell km | min infl | blocker frac | n part | true lat,lon | pred lat,lon |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `million_scale_cbg` | `NO_PROXIMITY` | 498 / 91% | `192.136.136.221` | `fold_4` | `WRONG` | `SUCCESS` | 494.0 | 492.8 | 153.2 | 1.96 | 0.00 | 2 | 38.8875, -77.4305 | 40.6394, -82.7522 |
| `million_scale_cbg` | `OTHER` | 48 / 9% | `113.23.232.22` | `fold_1` | `WRONG` | `SUCCESS` | 304.6 | 296.4 | 304.2 | 2.70 | 0.00 | 1 | 2.8985, 101.6515 | 1.3485, 103.8215 |
| `million_scale_cbg` | `ERRONEOUS_CONTAINMENT` | 3 / 1% | `193.1.201.140` | `fold_2` | `GIVE_UP` | `FALLBACK` | - | 7.6 | 269.7 | 19.86 | 0.00 | 2 | 53.3295, -6.3695 | 53.3515, -6.2595 |
| `octant_cbg` | `NO_PROXIMITY` | 440 / 83% | `37.10.42.14` | `fold_1` | `WRONG` | `SUCCESS` | 193.4 | 486.8 | 153.2 | 1.25 | 0.25 | 8 | 39.0115, -77.4595 | 40.6397, -78.0423 |
| `octant_cbg` | `OTHER` | 93 / 17% | `94.203.76.219` | `fold_3` | `WRONG` | `SUCCESS` | 432.4 | 20.5 | 104.4 | 2.35 | 0.50 | 2 | 25.0285, 55.1905 | 26.0263, 51.0169 |
| `spotter_cbg` | `NO_PROXIMITY` | 473 / 71% | `193.201.40.210` | `fold_3` | `WRONG` | `SUCCESS` | 1789.3 | 477.5 | 155.6 | 1.51 | 0.50 | 4 | 41.8995, 12.5085 | 57.0356, 4.0287 |
| `spotter_cbg` | `OTHER` | 115 / 17% | `23.138.112.22` | `fold_2` | `WRONG` | `SUCCESS` | 795.3 | 279.7 | 343.2 | 1.35 | 0.38 | 13 | 42.4695, -83.2425 | 36.3549, -78.4224 |
| `spotter_cbg` | `ERRONEOUS_CONTAINMENT` | 75 / 11% | `130.59.80.2` | `fold_2` | `WRONG` | `SUCCESS` | 1530.6 | 0.5 | 64.1 | 1.48 | 0.80 | 5 | 47.3805, 8.5475 | 58.3459, -5.4797 |
| `vanilla_cbg` | `NO_PROXIMITY` | 473 / 88% | `62.80.227.146` | `fold_3` | `WRONG` | `SUCCESS` | 1167.2 | 490.0 | 127.0 | 2.26 | 0.00 | 1 | 55.9195, 23.2885 | 50.1195, 8.6805 |
| `vanilla_cbg` | `OTHER` | 56 / 10% | `171.67.70.16` | `fold_2` | `WRONG` | `SUCCESS` | 377.1 | 43.7 | 141.9 | 1.28 | 0.00 | 3 | 37.4295, -122.1715 | 36.8746, -117.7401 |
| `vanilla_cbg` | `ERRONEOUS_CONTAINMENT` | 11 / 2% | `103.116.125.2` | `fold_1` | `GIVE_UP` | `FALLBACK` | - | 6.5 | 304.2 | 1.17 | 0.50 | 2 | 1.2985, 103.7915 | 1.3485, 103.8215 |

### europe-europe

- Run ID: `europe_as3215_eu`
- Config for map renderer: `scripts/analysis/partvp/cfg_textbook/europe_as3215_eu.yaml`
- Useful Voronoi overlay: `--landmass Europe`

| variant | mechanism | mech n/share | target_id | fold | outcome | status | err->cell km | nearest VP km | next cell km | min infl | blocker frac | n part | true lat,lon | pred lat,lon |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `million_scale_cbg` | `NO_PROXIMITY` | 349 / 90% | `46.244.63.73` | `fold_4` | `WRONG` | `SUCCESS` | 888.1 | 369.0 | 110.1 | 2.54 | 0.00 | 2 | 53.1105, 6.8805 | 60.1685, -0.4144 |
| `million_scale_cbg` | `OTHER` | 35 / 9% | `80.67.190.218` | `fold_4` | `WRONG` | `SUCCESS` | 316.7 | 0.8 | 219.0 | 2.52 | 0.00 | 2 | 48.1095, -1.6385 | 48.7152, 2.5588 |
| `million_scale_cbg` | `ERRONEOUS_CONTAINMENT` | 4 / 1% | `107.162.220.5` | `fold_4` | `GIVE_UP` | `FALLBACK` | - | 191.3 | 48.2 | 2.06 | 0.00 | 2 | 51.5175, -0.6405 | 48.8075, 2.1885 |
| `octant_cbg` | `NO_PROXIMITY` | 331 / 91% | `193.46.104.18` | `fold_2` | `WRONG` | `SUCCESS` | 316.2 | 397.3 | 86.0 | 3.40 | 0.00 | 22 | 47.7885, 13.0075 | 50.2162, 10.7303 |
| `octant_cbg` | `OTHER` | 13 / 4% | `46.29.176.71` | `fold_2` | `WRONG` | `SUCCESS` | 210.5 | 18.1 | 73.8 | 2.80 | 0.16 | 38 | 49.5115, 6.1075 | 51.3645, 6.8012 |
| `octant_cbg` | `ERRONEOUS_CONTAINMENT` | 12 / 3% | `213.225.160.239` | `fold_0` | `WRONG` | `SUCCESS` | 89.1 | 1.3 | 88.2 | 1.73 | 0.64 | 14 | 48.5795, 7.7485 | 48.4370, 6.7882 |
| `spotter_cbg` | `NO_PROXIMITY` | 350 / 86% | `185.144.72.8` | `fold_0` | `WRONG` | `SUCCESS` | 989.2 | 352.0 | 127.2 | 5.46 | 1.00 | 1 | 46.4795, 11.3315 | 55.5273, 8.0376 |
| `spotter_cbg` | `ERRONEOUS_CONTAINMENT` | 59 / 14% | `132.227.123.3` | `fold_3` | `WRONG` | `SUCCESS` | 1311.2 | 1.2 | 127.6 | 1.72 | 1.00 | 16 | 48.8515, 2.3605 | 42.5407, -11.9309 |
| `vanilla_cbg` | `NO_PROXIMITY` | 351 / 89% | `37.46.78.66` | `fold_1` | `WRONG` | `SUCCESS` | 1344.9 | 370.0 | 348.1 | 2.89 | 0.00 | 12 | 40.5105, -3.3385 | 51.3222, 4.1998 |
| `vanilla_cbg` | `OTHER` | 20 / 5% | `192.65.184.54` | `fold_0` | `WRONG` | `SUCCESS` | 332.7 | 15.1 | 93.1 | 2.46 | 0.00 | 26 | 46.2285, 6.0495 | 48.5574, 3.3522 |
| `vanilla_cbg` | `ERRONEOUS_CONTAINMENT` | 19 / 5% | `80.67.163.251` | `fold_1` | `GIVE_UP` | `FALLBACK` | - | 2.1 | 127.6 | 1.69 | 0.20 | 25 | 48.8585, 2.3785 | 48.8075, 2.1885 |

### europe-country

- Run ID: `europe_as3215_final_fr`
- Config for map renderer: `scripts/analysis/partvp/cfg_textbook/europe_as3215_final_fr.yaml`
- Useful Voronoi overlay: `--landmass France`

| variant | mechanism | mech n/share | target_id | fold | outcome | status | err->cell km | nearest VP km | next cell km | min infl | blocker frac | n part | true lat,lon | pred lat,lon |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `million_scale_cbg` | `OTHER` | 16 / 89% | `45.148.157.1` | `fold_1` | `WRONG` | `SUCCESS` | 364.0 | 66.9 | 124.4 | 3.84 | 0.00 | 1 | 47.6395, 6.8395 | 48.9005, 2.3005 |
| `million_scale_cbg` | `ERRONEOUS_CONTAINMENT` | 2 / 11% | `185.230.79.16` | `fold_3` | `GIVE_UP` | `FALLBACK` | - | 3.4 | 127.6 | 13.85 | 0.00 | 2 | 48.7085, 2.1585 | 48.9005, 2.3005 |
| `octant_cbg` | `OTHER` | 10 / 59% | `185.119.168.5` | `fold_0` | `WRONG` | `SUCCESS` | 348.1 | 2.7 | 215.4 | 2.21 | 0.11 | 27 | 43.6175, 1.4175 | 45.6060, 4.8194 |
| `octant_cbg` | `ERRONEOUS_CONTAINMENT` | 4 / 24% | `213.225.160.239` | `fold_0` | `WRONG` | `SUCCESS` | 199.1 | 1.3 | 119.0 | 2.96 | 0.83 | 12 | 48.5795, 7.7485 | 48.3501, 5.0706 |
| `octant_cbg` | `RTT_INFLATION` | 3 / 18% | `185.151.70.11` | `fold_3` | `WRONG` | `SUCCESS` | 582.5 | 2.0 | 215.4 | 4.14 | 0.25 | 4 | 43.5395, 1.5215 | 48.7302, 2.8427 |
| `spotter_cbg` | `ERRONEOUS_CONTAINMENT` | 14 / 56% | `185.151.70.11` | `fold_3` | `WRONG` | `SUCCESS` | 273.1 | 2.0 | 215.4 | 3.09 | 0.46 | 127 | 43.5395, 1.5215 | 45.9482, 2.3812 |
| `spotter_cbg` | `RTT_INFLATION` | 6 / 24% | `80.67.190.218` | `fold_4` | `WRONG` | `SUCCESS` | 386.1 | 0.8 | 219.0 | 2.19 | 0.32 | 102 | 48.1095, -1.6385 | 47.5022, 3.4547 |
| `spotter_cbg` | `OTHER` | 5 / 20% | `193.49.43.151` | `fold_2` | `WRONG` | `SUCCESS` | 264.5 | 2.8 | 96.1 | 1.85 | 0.33 | 112 | 45.2105, 5.6885 | 45.6948, 2.3683 |
| `vanilla_cbg` | `ERRONEOUS_CONTAINMENT` | 23 / 74% | `132.227.123.3` | `fold_3` | `GIVE_UP` | `FALLBACK` | - | 1.2 | 127.6 | 1.89 | 0.33 | 12 | 48.8515, 2.3605 | 48.8075, 2.1885 |
| `vanilla_cbg` | `OTHER` | 8 / 26% | `185.119.168.5` | `fold_0` | `WRONG` | `SUCCESS` | 415.2 | 2.7 | 215.4 | 2.40 | 0.00 | 9 | 43.6175, 1.4175 | 47.0936, -0.3231 |

### na-na

- Run ID: `north_america_as7018_final_na`
- Config for map renderer: `scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml`
- Useful Voronoi overlay: `--landmass North America`

| variant | mechanism | mech n/share | target_id | fold | outcome | status | err->cell km | nearest VP km | next cell km | min infl | blocker frac | n part | true lat,lon | pred lat,lon |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `million_scale_cbg` | `NO_PROXIMITY` | 53 / 68% | `192.33.139.235` | `fold_2` | `WRONG` | `SUCCESS` | 2058.2 | 482.0 | 431.8 | 1.76 | 0.00 | 4 | 33.4215, -112.0105 | 39.3134, -90.1485 |
| `million_scale_cbg` | `OTHER` | 25 / 32% | `23.157.112.130` | `fold_2` | `WRONG` | `SUCCESS` | 552.8 | 10.4 | 160.7 | 2.53 | 0.00 | 1 | 34.1675, -118.0915 | 37.8395, -122.2325 |
| `octant_cbg` | `NO_PROXIMITY` | 40 / 62% | `96.126.70.90` | `fold_4` | `WRONG` | `SUCCESS` | 1030.6 | 983.8 | 649.4 | 1.92 | 0.06 | 18 | 46.0795, -64.8115 | 36.8742, -66.2594 |
| `octant_cbg` | `OTHER` | 14 / 22% | `208.40.192.202` | `fold_4` | `WRONG` | `SUCCESS` | 505.4 | 159.1 | 278.6 | 1.68 | 0.25 | 16 | 40.4315, -80.0385 | 36.1017, -81.7994 |
| `octant_cbg` | `ERRONEOUS_CONTAINMENT` | 8 / 12% | `149.248.18.65` | `fold_2` | `WRONG` | `SUCCESS` | 502.5 | 10.8 | 160.7 | 1.60 | 0.86 | 14 | 34.0615, -118.2385 | 36.1834, -123.1111 |
| `spotter_cbg` | `NO_PROXIMITY` | 58 / 50% | `50.28.98.185` | `fold_2` | `WRONG` | `SUCCESS` | 779.7 | 481.8 | 431.8 | 1.68 | 0.88 | 26 | 33.4175, -112.0125 | 40.3320, -113.4937 |
| `spotter_cbg` | `ERRONEOUS_CONTAINMENT` | 31 / 27% | `23.174.128.243` | `fold_3` | `WRONG` | `SUCCESS` | 1146.1 | 10.2 | 141.9 | 1.63 | 0.90 | 21 | 38.6505, -121.4905 | 35.5431, -109.1566 |
| `spotter_cbg` | `OTHER` | 15 / 13% | `23.148.232.6` | `fold_3` | `WRONG` | `SUCCESS` | 817.7 | 33.8 | 282.2 | 1.66 | 0.74 | 42 | 40.5585, -74.4785 | 38.1329, -83.0143 |
| `vanilla_cbg` | `NO_PROXIMITY` | 44 / 57% | `172.93.18.42` | `fold_0` | `WRONG` | `SUCCESS` | 402.1 | 571.6 | 177.7 | 1.80 | 0.00 | 9 | 45.5185, -73.5215 | 42.0024, -72.1611 |
| `vanilla_cbg` | `ERRONEOUS_CONTAINMENT` | 18 / 23% | `177.124.130.57` | `fold_4` | `GIVE_UP` | `FALLBACK` | - | 1.1 | 329.0 | 1.56 | 0.50 | 6 | 25.7805, -80.1925 | 25.7705, -80.1925 |
| `vanilla_cbg` | `OTHER` | 15 / 19% | `23.157.112.130` | `fold_2` | `WRONG` | `SUCCESS` | 428.4 | 10.4 | 160.7 | 2.53 | 0.00 | 5 | 34.1675, -118.0915 | 36.4099, -121.9746 |

### na-us

- Run ID: `north_america_as7018_final_us`
- Config for map renderer: `scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_us.yaml`
- Useful Voronoi overlay: `--landmass US`

| variant | mechanism | mech n/share | target_id | fold | outcome | status | err->cell km | nearest VP km | next cell km | min infl | blocker frac | n part | true lat,lon | pred lat,lon |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `million_scale_cbg` | `NO_PROXIMITY` | 30 / 56% | `192.1.110.32` | `fold_4` | `WRONG` | `SUCCESS` | 334.6 | 333.5 | 309.2 | 3.23 | 0.00 | 1 | 42.3895, -71.1495 | 40.3975, -74.1395 |
| `million_scale_cbg` | `OTHER` | 24 / 44% | `206.144.243.4` | `fold_2` | `WRONG` | `SUCCESS` | 398.5 | 12.5 | 431.4 | 2.63 | 0.00 | 7 | 32.7885, -96.7885 | 36.1545, -98.4951 |
| `octant_cbg` | `NO_PROXIMITY` | 21 / 45% | `104.225.102.122` | `fold_0` | `WRONG` | `SUCCESS` | 601.3 | 481.9 | 431.8 | 1.89 | 0.42 | 12 | 33.4215, -112.0125 | 38.2364, -115.0465 |
| `octant_cbg` | `OTHER` | 18 / 38% | `208.86.250.253` | `fold_0` | `WRONG` | `SUCCESS` | 546.3 | 21.9 | 350.1 | 2.35 | 0.22 | 9 | 42.4515, -83.2585 | 44.6939, -89.2934 |
| `octant_cbg` | `ERRONEOUS_CONTAINMENT` | 7 / 15% | `151.139.52.113` | `fold_0` | `WRONG` | `SUCCESS` | 663.3 | 13.8 | 431.4 | 1.47 | 0.89 | 27 | 32.7985, -96.8215 | 35.1055, -90.1759 |
| `spotter_cbg` | `NO_PROXIMITY` | 35 / 39% | `35.186.175.121` | `fold_3` | `WRONG` | `SUCCESS` | 1059.3 | 324.1 | 153.2 | 1.71 | 0.75 | 55 | 39.0395, -77.4925 | 36.0088, -88.8094 |
| `spotter_cbg` | `ERRONEOUS_CONTAINMENT` | 27 / 30% | `151.139.52.113` | `fold_0` | `WRONG` | `SUCCESS` | 1203.5 | 13.8 | 431.4 | 1.45 | 0.90 | 60 | 32.7985, -96.8215 | 34.4419, -83.9663 |
| `spotter_cbg` | `RTT_INFLATION` | 15 / 17% | `23.157.112.121` | `fold_4` | `WRONG` | `SUCCESS` | 489.5 | 14.0 | 141.9 | 1.87 | 0.30 | 27 | 37.4685, -121.9215 | 37.1915, -116.4122 |
| `vanilla_cbg` | `NO_PROXIMITY` | 19 / 37% | `74.118.183.197` | `fold_0` | `WRONG` | `SUCCESS` | 535.2 | 527.0 | 431.8 | 1.90 | 0.00 | 4 | 37.0775, -113.6095 | 32.3063, -112.8397 |
| `vanilla_cbg` | `ERRONEOUS_CONTAINMENT` | 17 / 33% | `209.195.0.34` | `fold_1` | `GIVE_UP` | `FALLBACK` | - | 2.2 | 414.9 | 2.08 | 0.50 | 4 | 33.8915, -84.4705 | 33.9985, -84.4725 |
| `vanilla_cbg` | `OTHER` | 16 / 31% | `208.90.108.84` | `fold_4` | `WRONG` | `SUCCESS` | 503.6 | 36.2 | 344.2 | 2.51 | 0.00 | 24 | 36.0405, -94.1685 | 34.3867, -89.0068 |

## Notes

- `err->cell km` is distance from the prediction to the truth centroid, matching the map's `d->cell` label.
- `nearest VP km` is the available-fleet proximity signal used by the failure attribution script.
- `next cell km` is the truth centroid distance to the nearest competing answer-space centroid.
- `min infl` is the minimum participating-VP RTT inflation relative to the 2/3c physical slope.
- `blocker frac` is the fraction of participating VPs whose emitted distance band excludes the true VP-target distance.
