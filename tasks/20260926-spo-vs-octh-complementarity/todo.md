# SPO vs OCT-H Complementarity — Todo

## Phase 0: Shared substrate — done 2026-09-26, `modules/contest.py`
- [x] **Four atomic figure modules**, one per claim, sharing one helper — decided 2026-09-26. Not one module with a `--figure` selector.
- [x] Key every per-site frame on `(run_id, site_id)`. `site_id` is assigned per run and collides across meshes; keying on it alone reported 20 sites where there are 34.
- [x] `site_contest(run, method_a, method_b)`: per-site join of correct counts with `solved_mask` applied, returning `k_a, n_a, k_b, n_b, tg_seed_id, tg_lat, tg_lon`.
- [x] Four-way category by **direct count comparison** — `neither` when both zero, else `a`/`b`/`tied`. No threshold anywhere.
- [x] `centroid_km`: distance from each site to the run's seed-cloud centroid. **On the sphere**, via `geodesy.spherical_centroid` + `elementwise_km` — not EPSG:5070 as this line first said. The partition is great-circle nearest seed; the plane only draws it.
- [x] McNemar exact (`binomtest(b, b+c, 0.5)`) plus the attainable floor `2*0.5**(b+c)`, so an underpowered mesh says so.
- [x] Test: a 9/20 vs 20/20 site is `SPO wins`, never `tied` or `only SPO` — the Los Angeles regression.
- [x] Test: a FALLBACK row never reaches any count.
- [x] Guards beyond the plan: two methods on different rosters, a repeated `tg_id`, and two answer spaces (disagreeing `tg_seed_id`) are each refused rather than merged.

## Phase 1: C1 — the XOR map — done 2026-09-26, `modules/figure_contest_map.py`
- [x] Port the prototype: one panel per mesh, four marker styles, per-site label `SPO | OCT-H | total`.
- [x] Drop McNemar and the stats box from the figure; both go to manifest + CSV. Pinned by a test that fails on `"McNemar"`, `"p="`, `"sites="` or `"tied="` anywhere in the figure's text.
- [x] Title only — no footnote, no caption block. The legend stays: it is the encoding, not commentary.
- [x] Label de-collision: repel from markers as well as from other labels. `FOM.place_labels` grew `obstacles=`/`obstacle_sep=`; `figure_outcome_map` does **not** pass it, because turning it on would move counts on a figure the paper already cites.
- [x] Relaxation runs in units of a label box, not degrees. A site label is 4x wider than tall, so one isotropic separation in degrees is wrong on one axis whichever value it takes. All three panels now report **0 overlapping label pairs** in the manifest.
- [x] Colour comes from `methods.LABEL_HUES`: SPO's red, OCT-H's green, so C1–C4 paint the two methods identically. `tied` and `neither` are neutral — neither is a method.
- [x] Verified against `plan.md` with no adjustment: as01 8–1 p=0.039, as02 5–10 p=0.302, as03 8–9 p=1.000, pooled 21–20 over 65 sites / 1,269 targets.

### Open decision for the paper (Phase 5 will need an answer)
A site label is eight characters at a fixed point size, so the default 1x3 row
**cannot be printed with labels**: authored at 16.2 in it scales to under 3 pt
at a 7 in `\textwidth`, and authored at 7 in each panel is 2.3 in, where one
label spans 11 deg of a 65 deg frame and twenty cannot be separated at all.
Both ways out are one flag, and both are rendered and tested:
- `--no-labels` — keeps the 1x3 row, counts go to the CSV twin only. Reads
  cleanly at 7 in; the spatial pattern (SPO peripheral, OCT-H interior) is
  what survives, which is what C1 argues.
- `--ncols 1 --panel-width 7` — keeps the labels, gives a 7 x 14 in figure.
  Legible, but too tall for a page.

## Phase 2: C2 — peripherality boxplot — done 2026-09-26, `modules/figure_peripherality.py`
- [x] Two horizontal boxplots, `SPO wins` and `OCT-H wins`, coloured by each method's palette hue.
- [x] Distance is **seed → centroid of all seeds**, both ends in the answer space. Axis label: `normalized distance from the TG's seed to the centroid of all seeds`.
- [x] X = centroid distance, **min-max normalised across the pooled sites**, linear scale. Normalised over all 65, not the 41 drawn, so both endpoints are real sites and the axis does not move when a site changes hands.
- [x] Record the raw km behind the normalisation in the CSV — a normalised axis alone cannot be quoted. The twin carries all 65 sites with a `drawn` flag, because a twin holding only the drawn rows could not reproduce its own axis.
- [x] Reproduced from the CSV alone: SPO-win min 1,020 km, p25 1,634, median 1,928; OCT-H from 91 km, median 1,459. SPO's lower quartile sits above OCT-H's median.
- [x] Follows `dc29b36`'s convention, not C1's: **no figure title**, `subject` in the manifest, furniture sized for a paper column (4.6 x 1.75 in).

### Convention drift to settle
C1 carries a `suptitle` (it holds the `a | b | total` label key); C2 and the
outcome bars carry none. Either C1 loses its title and the key moves to the
paper's caption, or it keeps it as the one titled figure in the set.

## Phase 3: C3 — stability — done 2026-09-26, `modules/figure_stability.py`
- [x] **Two separate figures**, not two panels: `paired_ratio_of_success.spo_vs_octh` and `paired_std_grid_offset.spo_vs_octh`, each with its own CSV twin and manifest, so the paper can place them apart.
- [x] **Empirical CDF**, not a violin: x = per-site success ratio 0→1, y = share of sites, one step curve per method in its own hue. Exact — no bins, no bandwidth, no smoothing choices. The violin was wrong twice before this (Scott's bandwidth invented a waist; clipping the KDE to [0,1] left a shape that did not read as a violin).
- [x] The claim is readable off the geometry: the jump at 0 is the share of sites missed entirely, the jump at 1 the share swept, and the two sum to the unanimity rate — verified against the manifest, SPO 0.354 + 0.585 = 0.938, OCT-H 0.185 + 0.415 = 0.600. Between them SPO is flat where OCT-H climbs.
- [x] Boxplot of the per-site **standard deviation of `pred_dist_to_tg_grid`** across replicas, one box per method, in **grid steps** (`contest.offset_spread`, ddof=1). Whiskers at **p5/p95**, no outliers — so SPO's worst site (15.8 grids) is deliberately off the page and survives only as `spread_max`.
- [x] **Decided**, after measuring the alternative: this is the spread of the error *magnitude*, so two replicas five grids out in opposite directions read as perfect agreement. Against the spread of the prediction cloud itself (RMS grid distance to the site's prediction centroid) the two rank the 65 sites at Spearman 0.90, reach the same conclusion, and differ on 11 sites. The simpler statistic wins on those terms. `test_opposite_answers_read_as_agreement` pins the behaviour so it is not "fixed" by accident.
- [x] Grid steps, not kilometres — a spread in km invites comparison against an error distance, and this figure is not about accuracy. A site whose replicas all sit 38 grids out scores 0; as01 has such a site.
- [x] Both figures are **4 x 3 in**, with a test measuring each axis label against the canvas — the figures save at a fixed size, so an overrunning label is clipped silently (it happened once on `figure_peripherality`).
- [x] Axis labels in the paper's wording, title case: ratio `Fraction of Correct Predictions per Site` / `Fraction of Sites`, spread `Std. Dev. of Grid Error Distance`. Note `figure_peripherality` is still sentence case — **worth unifying**.
- [x] **65 points per box**, pooled: as01 20 + as02 22 + as03 23 sites, each over ~20 replicas, 1,269 targets behind each box. No site is dropped.
- [x] **Solved rows only**, both panels — the same rule as `correct_mask`. SPO and OCT-H never fall back on these meshes so the two readings coincide; the rule is fixed because it will not always, and VAN is the method it bites. A site with fewer than two solved rows has no spread and is absent from (b) rather than drawn at 0.
- [x] Reproduced from the CSV alone: SPO unanimous on 61/65 (93.8%), OCT-H on 39/65 (60.0%) with 26 splits; per-site spread p50 **0.31** grids vs **1.22**, zero spread on **21** sites vs **5**.
- [x] `QUANTILES` now carries p5 and p95, so the spread figure's **drawn whisker ends are readable as numbers** (SPO p95 2.29 grids, OCT-H 5.52) instead of being a bound only the ink states. `spread_min`/`spread_max` sit beside them.
- [x] Naming: these two use `spo_vs_octh`; C1 and C2 use `SPO-vs-OCT-H`. **Worth unifying** — not done here, because it renames committed artifacts.
- [x] `pred_dist_to_tg_grid` added to `contest.TG_COLUMNS`, and `site_contest` now returns `n_solved_*` and `offset_sd_*`. C4 needs the column too.

## Phase 4: C4 — error where both succeed — done 2026-09-27, `modules/figure_error_diff.py`
- [x] Cohort: the **565 targets both methods place in the correct cell**, across 34 `(run, site_id)` pairs. `contest.both_correct_targets`, solved rows only.
- [x] CDF of `diff = offset_SPO - offset_OCT-H`, zero at the x centre, symmetric log either side (`linthresh` 1 grid, the smallest difference there is). Ticks spelled, not left to the symlog locator, which labels decades and leaves the 1–3 band where the median sits unmarked.
- [x] Sign convention `spo - octh` fixed: the mass sits **right** of zero. The axis label carries the convention, a test asserts the minuend reads first, and the manifest says the expectation of a left lean did not move it.
- [x] The three shares are the **curve's own geometry** — height left of zero 4.2%, jump at zero 9.2%, remainder 86.5% — and a test pins that correspondence.
- [x] Direction is shown by two **arrows** above the curve (`SPO better ←`, `→ OCT-H better`), not by tinted half-planes: a direction is what the axis encodes, and arrows read in greyscale where two pale tints do not. They sit in a reserved band above y=1, which a CDF can never reach, so they cannot collide with the data.
- [x] Axis labels: x `Diff. of Grid Error Distances (SPO - OCT-H)`, y `Fraction of Correct Predictions`.
- [x] CSV twin is the cohort itself, 565 rows. Every claim re-derived from it alone: shares 86.5/4.2/9.2, median +2, p95 +8, max +19, per mesh 86.9/85.7/86.8, per site 29 OCT-H / 2 SPO / 3 level.
- [x] Companion in the manifest only: 21 SPO-win sites, offset p5/p50/p95 **1/3/35** vs **0/4/12**, OCT-H more accurate on **11 of 21** (SPO 8, level 2). Compared on an inner join of per-site medians, so a site one method never solved is not judged — `n_sites_compared` reports it.

## Phase 5: Verification
- [ ] Reproduce every C1–C4 number in `plan.md` from the committed CSVs, not from a notebook.
- [ ] Full v5 suite green. Baseline is **373**, not the 370 first recorded; C1 takes it to 415 (+20 `test_contest.py`, +22 `test_figure_contest_map.py`).
- [ ] Render at paper column width and check nothing is clipped at final size.
- [ ] Confirm each figure carries title + figure only.
