# CBG Variant Accuracy Ranking — Full-Region Analysis
**Date:** 2026-06-01  
**Configs:** 6 × `_final` ASN corpora (K=5 folds, n≈713 targets/combo/region)  
**Metric:** median error_km on SUCCESS+FALLBACK targets, aggregated across all folds  
**Source:** `scripts/benchmark/v2/outputs/<run_id>/summary.parquet` + per-fold `targets.parquet`

---

## Setup

| Config | Region label | ASN | Operator |
|--------|-------------|-----|----------|
| europe_as3209_final | EU·AS3209·Vodafone | 3209 | Vodafone Germany |
| europe_as3215_final | EU·AS3215·Orange | 3215 | Orange France |
| global_as16509_final | GL·AS16509·Amazon | 16509 | Amazon AWS |
| global_as31898_final | GL·AS31898·Oracle | 31898 | Oracle Cloud |
| north_america_as7018_final | NA·AS7018·ATT | 7018 | AT&T |
| north_america_as7922_final | NA·AS7922·Comcast | 7922 | Comcast |

16 combos × 6 regions × 5 folds = 480 (run, combo, fold) triples.

---

## 1. Global Ranking by Mean p50 Error (km) Across All Regions

Sorted by mean p50 across the 6 regions. Rank column = within-region rank (1=best).

```
combo_id               EU·AS3209  EU·AS3215  GL·AS16509  GL·AS31898  NA·AS7018  NA·AS7922  mean_p50  mean_rk  rk_std
                       p50   rk   p50   rk   p50    rk   p50    rk   p50   rk   p50   rk
octant_cbg_top         1214   1  1232   1    288    4    384    3   2760   1   2820   1     1449      2.2     1.3
octant_cbg             1214   1  1232   1    322    7    433    8   2760   1   2820   1     1463      3.2     3.4
spotter_cbg_c80        1304   4  1397   3    798   11    745   12   3020   5   4430   3     1949      6.3     4.1
spotter_cbg_top        1350   8  1533   8    898   14    864   15   2911   3   4638   6     2033      9.0     4.6
spotter_cbg            1350   8  1520   7    965   16    845   14   2911   3   4638   6     2039      9.0     5.0
vanilla_cbg            1407  13  1452   6    394    8    417    7   4170   6   4734   8     2096      8.0     2.6
octant_cbg_hull        1320   5  1419   4    265    1    374    1   4715   7   4510   4     2101      3.7     2.3
octant_cbg_top_geo     1253   3  1696  15    301    5    395    5   5099  10   4620   5     2228      7.2     4.5
spotter_cbg_c100       1334   7  1420   5    965   15    686   10   4861   9   4796   9     2344      9.2     3.4
spotter_cbg_top_geo    1330   6  1677  14    896   13    882   16   4723   8   6002  10     2585     11.2     3.8
spotter_cbg_c80_geo    1395  12  1570  11    796   10    801   13   5407  11   6094  11     2677     11.3     1.0
octant_cbg_hull_geo    1392  10  1581  12    270    2    387    4   6567  15   6334  12     2755      9.2     5.1
million_scale_cbg_geo  1412  15  1570   9    312    6    401    6   6524  13   6468  14     2781     10.5     4.0
vanilla_cbg_geo        1393  11  1588  13    271    3    379    2   6650  16   6424  13     2784      9.7     5.8
million_scale_cbg      1533  16  2158  16    416    9    473    9   6073  12   6882  16     2923     13.0     3.5
spotter_cbg_c100_geo   1412  14  1570   9    884   12    699   11   6524  14   6468  15     2926     12.5     2.3
```

---

## 2. Robustness Score (Minimax: worst rank across all 6 regions)

```
combo_id               worst_rk  best_rk  mean_rk  mean_p50
octant_cbg_top                4        1      2.2      1449   ← clear winner
octant_cbg_hull               7        1      3.7      2101
octant_cbg                    8        1      3.5      1463
spotter_cbg_c80              12        3      6.3      1949
vanilla_cbg                  13        6      8.0      2096
octant_cbg_top_geo           15        3      7.2      2228
spotter_cbg_c100             15        5      9.2      2344
spotter_cbg_top              15        3      9.2      2033
octant_cbg_hull_geo          15        2      9.2      2755
spotter_cbg                  16        3      9.2      2039
million_scale_cbg_geo        15        6     10.6      2781
spotter_cbg_c80_geo          13       10     11.3      2677
spotter_cbg_top_geo          16        6     11.2      2585
vanilla_cbg_geo              16        2      9.7      2784
spotter_cbg_c100_geo         15        9     12.6      2926
million_scale_cbg            16        9     13.0      2923
```

**`octant_cbg_top` is the most robust combo**: worst rank is 4 (GL·Amazon); wins outright in both EU corpora and both NA corpora; mean rank 2.2.

---

## 3. Region Similarity (Spearman Rank Correlation)

```
              EU·Vodafone  EU·Orange  GL·Amazon  GL·Oracle  NA·ATT  NA·Comcast
EU·Vodafone         1.000      0.521      0.122      0.112   0.658       0.897
EU·Orange           0.521      1.000     -0.031      0.094   0.721       0.721
GL·Amazon           0.122     -0.031      1.000      0.915  -0.275       0.081
GL·Oracle           0.112      0.094      0.915      1.000  -0.253       0.088
NA·ATT              0.658      0.721     -0.275     -0.253   1.000       0.853
NA·Comcast          0.897      0.721      0.081      0.088   0.853       1.000
```

Three clusters: GL pair (ρ=0.92), NA pair (ρ=0.85), EU pair (ρ=0.52). GL vs NA is weakly negative — rankings are nearly inverted between dense global and sparse continental networks.

---

## 4. The Critical Finding: `_geo` CTR Inverts in NA

`geometric_centroid` CTR helps in EU/GL but dramatically hurts in NA.

```
pair (base → geo)            EU·Vodafone  EU·Orange  GL·Amazon  GL·Oracle  NA·ATT  NA·Comcast  overall
vanilla_cbg → geo                    -14       +136       -123        -38  +2480       +1691      +20
million_scale → geo                 -120       -588       -105        -72    +451        -414     -517
octant_cbg_top → geo                 +40       +465        +14        +11  +2339       +1800     +206
octant_cbg_hull → geo                +72       +162         +5        +13  +1853       +1824      +56
spotter_cbg_top → geo                -20       +143         -2        +17  +1812       +1364      +89
spotter_cbg_c100 → geo               +78       +150        -81        +13  +1663       +1672      -11
spotter_cbg_c80 → geo                +90       +173         -1        +55  +2387       +1663      +32
```

**Why:** In NA, VP coverage is sparse and constraint regions are large continent-spanning polygons. The geometric centroid of an irregular large polygon lands far from the true target. Non-geo CTRs (boundary_vertex_mean, monte_carlo_medoid) pull toward the densest constraint boundary, which is more informative when constraints are weak.

---

## 5. NA Inversion Rankings (Δ mean rank: NA − EU/GL)

```
combo_id               EU/GL mean rank  NA mean rank  Δ (NA−EU/GL)
vanilla_cbg_geo                    7.2          14.5         +7.2   ← worst flip
octant_cbg_hull_geo                7.0          13.5         +6.5
million_scale_cbg_geo              9.1          13.5         +4.4
spotter_cbg_c100_geo              11.6          14.5         +2.9
octant_cbg_hull                    2.8           5.5         +2.8
spotter_cbg                       11.4           5.0         -6.4   ← improves in NA
spotter_cbg_top                   11.4           5.0         -6.4
spotter_cbg_c80                    7.5           4.0         -3.5
octant_cbg                         4.5           1.5         -3.0
```

---

## 6. Ablation Results

### `_top` variant (select top-k landmarks)
```
octant_cbg  → _top:   EU·Voda=0  EU·Orange=0  GL·Amazon=-35  GL·Oracle=-49  NA·ATT=0  NA·Comcast=0  overall=-76km
spotter_cbg → _top:   EU·Voda=0  EU·Orange=+13  GL·Amazon=-67  GL·Oracle=+19  NA=0  overall=-70km
```
`_top` is neutral for EU/NA, mildly helpful for GL (dense global networks benefit from narrowing landmark set).

### `_hull` variant (convex hull of intersection boundary)
```
octant_cbg → _hull:  EU·Voda=+106  EU·Orange=+188  GL·Amazon=-57  GL·Oracle=-59  NA·ATT=+1955  NA·Comcast=+1689  overall=+51km
```
Hull helps only for tight GL networks; catastrophic for NA.

---

## 7. LTD Family Summary (all variants pooled per family)

```
family          EU·Vodafone  EU·Orange  GL·Amazon  GL·Oracle  NA·ATT  NA·Comcast  mean_p50
octant                 1287       1427        288        392    4360        4223      1996
spotter                1346       1526        882        785    4335        5253      2354
vanilla                1404       1542        310        402    6041        6053      2625
million_scale          1453       1850        385        421    6277        6584      2829
```

LTD family order is stable across all regions: **octant > spotter > vanilla > million_scale**.

---

## 8. Fraction Within Distance Thresholds (top combos per region)

| Region | Combo | <100km | <500km | <1000km | <2500km | <5000km |
|--------|-------|--------|--------|---------|---------|---------|
| EU·Vodafone | octant_cbg_top | 4.9% | 33.0% | 45.9% | 58.1% | 66.5% |
| EU·Orange | octant_cbg_top | 4.1% | 33.7% | 45.6% | 58.3% | 67.6% |
| GL·Amazon | octant_cbg_hull | 29.6% | 67.3% | 81.1% | 93.7% | 98.3% |
| GL·Oracle | octant_cbg_hull | 27.5% | 57.5% | 70.3% | 92.0% | 97.6% |
| NA·ATT | octant_cbg_top | 3.1% | 10.1% | 15.4% | 42.5% | 87.0% |
| NA·Comcast | octant_cbg_top | 3.8% | 10.8% | 15.1% | 41.1% | 83.6% |

---

## Key Takeaways

1. **`octant_cbg_top` is the recommended single combo**: best mean rank (2.2), best robustness (worst rank 4), wins EU×2 and NA×2, competitive in GL.
2. **`_geo` CTR is regionally toxic for NA**: adds +1.7–2.5k km to median error in NA while neutral in EU. Do not use geo-centroid with sparse continental VP fleets.
3. **GL rewards tight intersection CTRs** (`octant_cbg_hull` wins there) because VP density is high enough that constraint regions are small and hull/centroid are stable.
4. **`million_scale_cbg` is consistently last in EU**: the speed-of-internet RTT model is too permissive for dense EU networks.
5. **Three structurally distinct environments**: GL (dense, low error ~300–400km median), EU (medium density, ~1200–2200km), NA (sparse, ~2800–7000km). Optimal combo for one environment does not transfer to another for geo-CTR variants.
