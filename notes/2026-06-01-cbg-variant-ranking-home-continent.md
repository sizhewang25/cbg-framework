# CBG Variant Accuracy Ranking — Home-Continent Analysis
**Date:** 2026-06-01  
**Companion to:** `2026-06-01-cbg-variant-ranking-full-region.md`  
**Filter:** Each region's targets restricted to its VP fleet's home continent  
— EU corpora: European anchors only (83/143 targets per fold ≈ 58%)  
— NA corpora: North American anchors only (24/143 targets per fold ≈ 17%)  
— GL corpora excluded (no single home continent)

---

## Setup

| Config | Region label | Home continent | n_targets/fold |
|--------|-------------|----------------|---------------|
| europe_as3209_final | EU·AS3209·Vodafone | Europe | 83 |
| europe_as3215_final | EU·AS3215·Orange | Europe | 83 |
| north_america_as7018_final | NA·AS7018·ATT | North America | 24 |
| north_america_as7922_final | NA·AS7922·Comcast | North America | 24 |

---

## 1. Per-Region Ranking (home-continent targets only)

### EU·AS3209·Vodafone

```
Rank  combo_id               p25    p50    p75    p95  succ%
1     octant_cbg_top_geo     201    416    881   1648  100.0   ← geo now #1 (was #3 full)
2     vanilla_cbg            173    425    828   2326   92.5
3     octant_cbg_hull_geo    194    427    807   1838  100.0
4     octant_cbg_top         200    429    834   1876  100.0
5     octant_cbg             206    431    834   1876  100.0
6     vanilla_cbg_geo        195    434    806   1838  100.0
7     octant_cbg_hull        194    437    835   1837  100.0
8     spotter_cbg_c100_geo   196    460    891   1864  100.0
9     million_scale_cbg_geo  184    461    891   1864  100.0
10    spotter_cbg_c80_geo    249    466    863   1864  100.0
11    spotter_cbg_top_geo    230    474    868   1870  100.0
12    spotter_cbg_c80        259    475    875   1860  100.0
13    spotter_cbg_top        231    475    861   1976  100.0
14    spotter_cbg            243    476    861   1976  100.0
15    spotter_cbg_c100       207    482    881   1891  100.0
16    million_scale_cbg      193    482   1146   2497  100.0
```

### EU·AS3215·Orange

```
Rank  combo_id               p25    p50    p75    p95  succ%
1     octant_cbg_top         245    420    890   1906  100.0
2     octant_cbg             259    425    890   1906  100.0
3     octant_cbg_top_geo     265    435   1063   2260  100.0
4     octant_cbg_hull        246    439   1005   1986  100.0
5     octant_cbg_hull_geo    290    474    986   2213  100.0
6     vanilla_cbg            240    475   1019   2409   96.6
7     vanilla_cbg_geo        313    483    995   2224  100.0
8     million_scale_cbg_geo  304    487   1023   2123  100.0
9     spotter_cbg_c100       302    495   1034   1996  100.0
10    spotter_cbg_c100_geo   338    504   1023   2123  100.0
11    spotter_cbg_c80        398    586   1075   1982  100.0
12    spotter_cbg_c80_geo    405    588   1077   2114  100.0
13    spotter_cbg_top        399    591   1122   2259  100.0
14    spotter_cbg            392    592   1100   2259  100.0
15    spotter_cbg_top_geo    407    624   1251   2123  100.0
16    million_scale_cbg      360    810   1473   3331   99.0
```

### NA·AS7018·ATT  ← rankings INVERT vs full-region

```
Rank  combo_id               p25    p50    p75    p95  succ%
1     vanilla_cbg_geo         89    262    733   1829  100.0   ← was rank 16 full-region!
2     million_scale_cbg_geo   45    269    710   1821  100.0   ← was rank 13 full-region
3     octant_cbg_hull_geo     43    276    695   1822  100.0   ← was rank 15
4     octant_cbg_hull         45    288    608   1795  100.0   ← was rank 7
5     million_scale_cbg      104    335    999   2056  100.0   ← was rank 12
6     vanilla_cbg             99    349    807   1795   87.7
7     octant_cbg_top         149    451    703   2969  100.0   ← was rank 1 full-region!
8     octant_cbg_top_geo     149    453    703   2089  100.0
9     octant_cbg             222    504    793   2982  100.0
...
16    spotter_cbg            467    647    978   2422  100.0
```

### NA·AS7922·Comcast  ← similar inversion

```
Rank  combo_id               p25    p50    p75    p95  succ%
1     octant_cbg_hull_geo     32    133    441   1919  100.0   ← was rank 12 full-region
2     octant_cbg_hull         31    135    425   2096  100.0   ← was rank 4
3     vanilla_cbg_geo         35    158    416   1850  100.0   ← was rank 13
4     vanilla_cbg             37    166    523   1557   77.9
5     spotter_cbg_c100        88    207    521   1958  100.0
6     spotter_cbg_c100_geo    83    230    518   1829  100.0
7     million_scale_cbg_geo   87    237    518   1829  100.0
...
14    octant_cbg             197    395    800   2151  100.0   ← was rank 1 full-region
16    million_scale_cbg      190    477   1143   2597  100.0
```

---

## 2. Cross-Region Table (home-continent targets, sorted by mean p50)

```
combo_id               EU·AS3209  EU·AS3215  NA·AS7018  NA·AS7922  mean_p50  mean_rk  rk_std
                       p50  rk    p50  rk    p50  rk    p50  rk
octant_cbg_hull         437   7    439   4    288   4    135   2       324      4.2     2.1
octant_cbg_hull_geo     427   3    474   5    276   3    133   1       327      3.0     1.6  ← robustness winner
vanilla_cbg_geo         434   6    483   7    262   1    158   3       334      4.2     2.8
vanilla_cbg             425   2    475   6    349   6    166   4       354      4.5     1.9
million_scale_cbg_geo   461   9    487   8    269   2    237   7       364      6.5     3.1
octant_cbg_top          429   4    420   1    451   7    371  10       418      5.5     3.9
octant_cbg_top_geo      416   1    435   3    453   8    386  12       422      6.0     5.0
octant_cbg              431   5    425   2    504   9    395  14       439      7.5     5.2
spotter_cbg_c100        482  15    495   9    601  10    207   5       446      9.8     4.1
spotter_cbg_c100_geo    460   8    504  10    619  12    230   6       453      9.0     2.6
spotter_cbg_c80         475  12    586  11    632  14    353   8       511     11.2     2.5
spotter_cbg_top         475  13    591  13    621  13    366   9       513     12.0     2.0
spotter_cbg_c80_geo     466  10    588  12    638  15    376  11       517     12.0     2.2
spotter_cbg_top_geo     474  11    624  15    613  11    387  13       524     12.5     1.9
million_scale_cbg       482  16    810  16    335   5    477  16       526     13.2     5.5
spotter_cbg             476  14    592  14    647  16    418  15       533     14.8     1.0
```

---

## 3. Robustness (Minimax: worst rank across 4 regions)

```
combo_id               worst_rk  best_rk  mean_rk  mean_p50
octant_cbg_hull_geo           5        1      3.0       327  ← home-continent winner
vanilla_cbg                   6        2      4.5       354
vanilla_cbg_geo               7        1      4.2       334
octant_cbg_hull               7        2      4.2       324
million_scale_cbg_geo         9        2      6.5       364
octant_cbg_top               10        1      5.5       418
octant_cbg_top_geo           12        1      6.0       422
```

`octant_cbg_hull_geo` has the best minimax (5) and best mean rank (3.0). `octant_cbg_top` (the full-region winner) drops to minimax=10 due to its poor NA home-continent performance.

---

## 4. Spearman Rank Correlation (home-continent)

```
              EU·AS3209  EU·AS3215  NA·AS7018  NA·AS7922
EU·AS3209         1.000      0.832      0.459      0.374
EU·AS3215         0.832      1.000      0.485      0.397
NA·AS7018         0.459      0.485      1.000      0.491
NA·AS7022         0.374      0.397      0.491      1.000
```

EU·AS3209 ↔ EU·AS3215 correlation jumps from **ρ=0.52 → 0.83** when restricted to home-continent targets — the two EU corpora have very similar within-Europe ranking structure.  
NA·ATT ↔ NA·Comcast drops from **ρ=0.85 → 0.49** — the two NA corpora have meaningfully different within-NA rankings (ATT favors `_geo`, Comcast favors `_hull`).

---

## 5. Geo-Centroid Lift (home-continent targets)

```
pair (base → geo)            EU·AS3209  EU·AS3215  NA·AS7018  NA·AS7922  overall
vanilla_cbg → geo                   +9         +8        -87         -9      +22
million_scale → geo                -20        -322        -66       -240     -155  ← consistent gain
octant_cbg_top → geo               -13        +14         +1        +15       -1   ← near neutral
octant_cbg_hull → geo               -9        +35        -12         -2      +14
spotter_cbg_top → geo               -1        +33         -8        +21        0
spotter_cbg_c100 → geo             -21         +9        +18        +23       -8
spotter_cbg_c80 → geo               -9         +2         +5        +23       +8
```

**Critical finding: _geo is near-neutral for home-continent.** The catastrophic NA `_geo` penalty seen in the full-region analysis (+1.7–2.5k km) completely disappears when restricted to NA home-continent targets. Geo-centroid is not intrinsically bad for NA — it is bad for *out-of-continent targets evaluated with a regional VP fleet*.

Only `million_scale_cbg_geo` shows a consistent gain across all four regions (−155 km overall).

---

## 6. Fraction Within Distance Thresholds (home-continent, top-3)

| Region | Combo | <100km | <500km | <1000km | <2500km | <5000km |
|--------|-------|--------|--------|---------|---------|---------|
| EU·Vodafone | octant_cbg_top_geo | 8.0% | 58.6% | 79.0% | 99.0% | 99.8% |
| EU·Vodafone | octant_cbg_hull_geo | 14.0% | 59.5% | 79.3% | 99.0% | 99.8% |
| EU·Orange | octant_cbg_top | 7.0% | 57.8% | 78.1% | 98.8% | 100.0% |
| NA·ATT | vanilla_cbg_geo | 27.0% | 63.9% | 81.1% | 98.4% | 99.2% |
| NA·ATT | octant_cbg_hull_geo | 36.9% | 64.8% | 81.1% | 98.4% | 99.2% |
| NA·Comcast | octant_cbg_hull_geo | 45.9% | 77.0% | 87.7% | 96.7% | 99.2% |
| NA·Comcast | vanilla_cbg_geo | 45.9% | 77.9% | 88.5% | 98.4% | 99.2% |

Home-continent coverage at 500km is dramatically better: EU ~58% vs ~33% full-region, NA ~65–78% vs ~10–11% full-region.

---

## 7. Home vs Full-Region Delta (p50, home − full)

```
combo_id               EU·AS3209       EU·AS3215       NA·AS7018       NA·AS7922
octant_cbg_top     429→1214 (-785)  420→1232 (-811)  451→2760(-2308)  371→2820(-2449)
octant_cbg         431→1214 (-782)  425→1232 (-806)  504→2760(-2256)  395→2820(-2425)
spotter_cbg_c80    475→1304 (-830)  586→1397 (-811)  632→3020(-2387)  353→4430(-4078)
vanilla_cbg        425→1407 (-982)  475→1452 (-978)  349→4170(-3821)  166→4734(-4567)
million_scale_cbg  482→1533(-1051)  810→2158(-1348)  335→6073(-5738)  477→6882(-6406)
octant_cbg_top_geo 416→1253 (-837)  435→1696(-1262)  453→5099(-4647)  386→4620(-4234)
vanilla_cbg_geo    434→1393 (-959)  483→1588(-1106)  262→6650(-6388)  158→6424(-6267)
```

NA home-continent accuracy is 6–25x better than full-region for `_geo` variants — the out-of-continent degradation is extreme.

---

## Key Takeaways vs Full-Region Analysis

1. **The `_geo` NA collapse was an out-of-continent artifact.** Home-continent NA targets are accurately handled by `_geo` variants (vanilla_cbg_geo ranks #1 in NA·ATT at 262km median). The villain is regional VP fleets trying to geolocate far-away targets with an inappropriate CTR.

2. **Home-continent winner: `octant_cbg_hull_geo`** (worst rank=5, mean rank=3.0, mean p50=327km) — up from rank 9–15 in full-region analysis. The convex-hull CTR is ideal when the VP fleet's constraint circles are tight (home-continent targets are close to VPs).

3. **EU rankings are stable home↔full** (octant top-4, spotter bottom). The EU corpora have enough non-EU anchors to add noise but not enough to flip the ranking order. EU·AS3209 ↔ EU·AS3215 correlation rises sharply (ρ=0.52 → 0.83), confirming the two EU ASNs are measuring the same phenomenon at home.

4. **NA rankings invert completely home↔full.** `octant_cbg_top` (full-region #1 for NA) drops to rank 7–14 at home. `vanilla_cbg_geo` (full-region #16 for NA) rises to rank 1–3 at home. The practical implication: for a deployment that only cares about geolocating NA targets with a NA VP fleet, choose `octant_cbg_hull_geo` or `vanilla_cbg_geo`, not `octant_cbg_top`.

5. **`million_scale_cbg_geo` is unexpectedly strong for NA home-continent**: ranks #2 in NA·ATT (269km) and #7 in NA·Comcast (237km). The speed-of-internet LTD model, when paired with geometric centroid, works well for dense home-continent measurements.

6. **The two NA corpora disagree at home more than globally** (ρ: 0.85 → 0.49). ATT vs Comcast probes cover the US differently, and those coverage differences matter more for tight intra-NA geolocation than for the broad continental patterns dominating full-region results.
