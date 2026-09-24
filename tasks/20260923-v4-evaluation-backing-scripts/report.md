# v4 Evaluation Backing Scripts — Report

**Status**: In Progress
**Created**: 2026-09-23
**Last Updated**: 2026-09-23

## Summary

Turning three ad-hoc measurements into pipeline artifacts so
`cbg-benchmark-paper/sections/evaluation.md` rests on regenerable numbers.

## Findings

Session measurements to be reproduced by the new commands (pooled
as01+as02+as03, 1,269 targets / 65 sites):

- Ring bound, max `error_km`: nside-128 ring0 50 / ring1 142 / ring2 193 /
  beyond 3,946 km. Ring-1 ~2.3-2.9x the nominal cell, ring-2 ~3.5-5.3x.
- Ring-1 serving-region agreement by rung: 36.4% (ns16), 57.4% (32),
  69.2% (64), 89.9% (128).
- Spotter at nside-128: ring0 0.0%, ring-1 agreement 100%, ring-2 95%,
  bounded recovery 25.1% (retired unbounded rule said 62.2% and ranked it 1st).
- ICC 0.64-0.99, design effect 13-19x, effective n 66-99. Paired site-level
  tests: Octant-Hull vs Shortest-Ping n.s. at every rung (p=0.23/0.92/0.27);
  vs Vanilla and Spotter significant at p<0.01 throughout.
- Micro vs macro rate agrees within 0.3 pp (replica CV 0.088) — rates are
  unbiased; the defect is precision.
- Per-site profile at nside-128: Shortest-Ping 13 all / 1 some / 51 none;
  Octant-Hull 3 / 26 / 36. 83.6% of (method, site) pairs unanimous.
- Class collapse: 65 sites -> 52 classes at nside-16 (25 sites merge into 12
  cells), 60 at nside-32, 63 at nside-64/128.

## Verified during planning (2026-09-23)

Reproduced **exactly** from on-disk artifacts before any code was written, so
the methodology is sound and the job is packaging rather than rediscovery:
§3 class collapse (52/60/63/63), §4 ring bound (50/142/193/3946 …
509/1078/1427/3946), §4 per-method recovery (all six rows), §5 ICC and
effective n (0.68/94, 0.97/67, 0.99/66, 0.79/81).

## Finding A — the site key, and a wrong number in the companion

No region id exists anywhere: `datasets/final/*.mainland.sanitized.csv` carries
only `vp_id, vp_lat, vp_lon, vp_asn, vp_country, target_id, target_lat,
target_lon, rtt_ms`, and the reconstruction manifests list `target_city` and
`target_asn` as **unrecoverable** (`tg_configs.parquet` is gone). The coordinate
pair is the only stable identity, and it is rounding-insensitive (2–6 dp agree).

The three meshes **share facilities**: 20+22+23 = 65 per-run sites, but only
**43 distinct coordinates** pooled — 7 in all three runs, 8 in exactly two, 28
in one. The companion's "1,269 targets over 65 unique site coordinates" was
therefore false; 65 counts site-*appearances*. Corrected in `evaluation.md`.

**Key = `(run_id, target_lat, target_lon)`**, because operators geolocate ASN by
ASN: a facility serving two autonomous systems poses two geolocation problems.
This reproduces every published number. Under the coordinate-only key effective
n falls to 44–74 and design effects rise to 17–29×; the "top four are tied"
conclusion holds either way, so no paper claim rested on the choice.

## Finding B — the error-distance section is under-supported

The *data* is complete and regenerable (`error_cdf.pooled.csv`, no script
needed). The *prose* is not:

1. Six cells are printed as `—` though they exist: Vanilla p5 = 9.53,
   p25 = 58.19, p95 = 1156.65; Spotter p5 = 71.36, p25 = 150.83,
   p95 = **1894.58** — the worst tail in the table, suppressed.
2. The read-out "sub-2 km at p5 for **every** method" is **false**, falsified by
   exactly those suppressed cells. Only 4 of 6 methods are sub-2 km.
3. §1 ranks by p50 to 0.1 km and §2b builds its "opposite recommendation"
   argument on that ranking, while §5 forbids ranking rates for clustering.
   Paired site bootstrap (2,000 resamples over the 65 sites):

   | contrast | p50 | p90 |
   |---|---|---|
   | SoI − Shortest-Ping | −19.6 [−61.5, +9.2] n.s. | −27.6 n.s. |
   | SoI − Octant-Hull | −35.0 [−112.1, +105.9] n.s. | **+200.4 p=0.020** |
   | Octant-Hull − Shortest-Ping | +15.4 [−130.2, +112.3] n.s. | **−228.0 p<0.001** |

   **No median difference survives; the tail differences do.** "Octant-Hull owns
   the tail" holds; "SoI owns the median" does not; and the punchline has no
   significant median advantage to flip *to*. The tail argument alone still
   carries the section.

Consequence: `significance` must cover **error percentiles as well as ring-0
rates**, or §1/§2b keeps resting on the ranking §5 exists to forbid.

## Delivered

**T1 — site key + class collapse (2026-09-23).**
`modules/sites.py` (the one site key, `solved_mask` re-exported,
`site_diagnostics` publishing 65 *and* 43), `modules/class_collapse.py` +
`class-collapse` command, wired into `create_analysis_artifacts.sh`, README
section, 22 tests. Full v4 suite green (298 passed).
`map_answer_space.distinct_sites` now calls into `sites.py`, so the change
closes a duplicate spelling instead of adding a fourth.

Output reproduces §3 exactly:

| rung | sites | classes | merged | cells | max/cell |
|---|--:|--:|--:|--:|--:|
| 128 | 65 | 63 | 4 | 2 | 2 |
| 64 | 65 | 63 | 4 | 2 | 2 |
| 32 | 65 | 60 | 10 | 5 | 2 |
| 16 | 65 | **52** | 25 | 12 | **3** |

The manifest publishes as measurements what the first draft had asserted: the
unioned class count (40/38/33/26 — the number to *avoid*, published beside the
reason) and the per-run quantizer floor (as01 caps at 18 classes for 20 sites,
ceiling 0.90, at every rung).

## Conclusions

<To be filled when T2–T8 land.>

## 2026-09-23 (later) — two additions from drafting §5.1

**`significance` must cover error percentiles, not only ring-0 rates.**
`evaluation.tex` §5.1 now quotes paired site-bootstrap intervals on `error_km`
p50/p90 (OCT-H vs S-P at p90: −228.0 km [−589.9, −107.3], p<0.001; SoI vs OCT-H
at p90: +200.4 [+35.9, +564.3], p=0.020; all p50 contrasts n.s.). These were
measured ad hoc and are not reproducible from the repo yet.

**New command needed: VP-adjacency breakdown.** §5.1 quotes: 97/1,269 targets
(7.6%) within 1.55 km of a VP (read off the S-P error CDF, which upper-bounds
target-to-nearest-VP distance); S-P and SoI bottom-63 errors lie entirely in
that group while only 14.3% of OCT-H's do; on those 97 targets OCT-H p50=5.61 /
p90=250.84 km and VAN fails 60/97 (61.9%) vs 21.7% overall. Needs a command
emitting the per-method bottom-k overlap and the restricted percentiles.

**Threshold-sweep table needed (§5.1 argument 2).** Fraction of all 1,269
targets resolved within X km (failures counted in the denominator), measured
2026-09-23:

| method | 40 km | 100 km | 500 km | 1000 km |
|---|--:|--:|--:|--:|
| S-P | 30.7 | 49.6 | 70.8 | 93.2 |
| SOI | 32.5 | **51.5** | 77.9 | 93.2 |
| OCT-H | 34.6 | 43.7 | **87.6** | **98.3** |
| OCT-S | **34.8** | 42.1 | 81.8 | 96.7 |
| VAN | 13.2 | 27.7 | 65.2 | 74.0 |
| SPO | 1.6 | 10.2 | 72.5 | 90.2 |

The four thresholds standard in prior work select **three different winners**
(OCT-S / SOI / OCT-H) on identical predictions. This is the load-bearing
evidence for §5.1's threshold argument and must become a reproducible command.

## 2026-09-23 — `plot-vp-proximity` DONE

`scripts/analysis/v4/modules/vp_proximity.py` + `@app.command("plot-vp-proximity")`
+ `tests/test_vp_proximity.py` (15 tests). Full v4 suite: 400 passed.

Writes `_cross/<datasets>@<arm>/vp_proximity/vp_proximity.{cohort}.{png,csv,manifest.json}`
for `--cohort p5|p25|all`, `--geo/--no-geo`, `--sping/--no-sping`.

**Note the directory shape differs from the rest of the package.** Everything
else is `_cross/<kind>/<datasets>/` (`cross.cross_dir`); this is
`_cross/<datasets>/<kind>/`, as specified. Worth reconciling before more
commands land, or the `_cross` tree carries two conventions.

Results, pooled as01+as02+as03 mesh:

- population: closest VP by geography p50 **13.8** km / max 374.5; by RTT p50
  **115.7** / max 3946. The 8.4x gap is RTT inflation, measured.
- p5 bound = **53.4 km**, every method, no exception.
- p25 bound breaks only for the full-constraint variants: OCT-H 720.7,
  SPO 600.7, OCT-S 400.3 vs S-P 25.3, SOI 48.2, VAN 140.5.

**Corrects two ad-hoc numbers reported earlier in the session.** The hand-rolled
version ranked cohorts on `error_km` without `classify.solved_mask`, so
Vanilla's cohort filled with FALLBACK rows — whose prediction *is* the
shortest-ping VP coordinate. Vanilla p5 bound 33.1 -> **69.0** km, p25 69.0 ->
**140.5** km. The wrong numbers were the flattering ones. Pinned by
`TestCohortExcludesUnanswered`.

Design record for the figure form: `violin-vs-dotstrip{,-p25}.png` in this
directory. Violins are defensible at p25 (9-20 distinct values, max tie 15-19%)
and misleading at p5 (2-11 distinct values, max tie 31-60%); `max_tie_share`
and `distinct_values` ship in every stats CSV row so this is checkable.
