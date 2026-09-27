# SPO vs OCT-H Complementarity — Todo

## Phase 0: Shared substrate — done 2026-09-26, `modules/contest.py`
- [x] **Four atomic figure modules**, one per claim, sharing one helper — decided 2026-09-26. Not one module with a `--figure` selector.
- [x] Key every per-site frame on `(run_id, site_id)`. `site_id` is assigned per run and collides across meshes; keying on it alone reported 20 sites where there are 34.
- [x] `site_contest(run, method_a, method_b)`: per-site join of correct counts with `solved_mask` applied, returning `k_a, n_a, k_b, n_b, tg_seed_id, tg_lat, tg_lon`.
- [x] Four-way category by **direct count comparison** — `neither` when both zero, else `a`/`b`/`tied`. No threshold anywhere.
- [x] `centroid_km`: distance from each site to the run's seed-cloud centroid, in EPSG:5070 via `projection.project`.
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

## Phase 2: C2 — peripherality boxplot
- [ ] Two horizontal boxplots, `SPO wins` and `OCT-H wins`, coloured by each method's palette hue.
- [ ] X = centroid distance, **min-max normalised across the pooled sites**, linear scale.
- [ ] Record the raw km behind the normalisation in the CSV — a normalised axis alone cannot be quoted.

## Phase 3: C3 — stability
- [ ] (a) Violin of per-site success ratio (`k/n`) by category, x = category, y = fraction. Overlay the 93.8% / 60.0% unanimity rates in the CSV, not on the figure.
- [ ] (b) Boxplot of the per-site **standard deviation of `pred_dist_to_tg_grid`** across replicas, one box per method.
- [ ] Decide and document whether (b) uses all targets or solved rows only — they differ for VAN but not for SPO/OCT-H, so state the rule once.

## Phase 4: C4 — error where both succeed
- [ ] Cohort: the **565 targets both methods place in the correct cell**, across 34 `(run, site_id)` pairs.
- [ ] CDF of `diff = offset_SPO - offset_OCT-H`, zero at the x centre, symmetric log scale either side.
- [ ] Sign convention `spo - octh` is fixed: the mass sits **right** of zero (OCT-H closer on 86.5%). Do not flip it to make the curve lean left.
- [ ] Mark the three shares on the curve's own geometry (4.2% / 9.2% / 86.5%) via the CSV, not as figure annotation.
- [ ] Emit to CSV: per-mesh shares (86.9 / 85.7 / 86.8), diff percentiles, and the per-site head-to-head (2 SPO / 3 equal / 29 OCT-H).
- [ ] Companion stat in the manifest only: on the 21 SPO-win sites, offset p5/p50/p95 1/3/35 vs 0/4/12, OCT-H better on 11 of 21.

## Phase 5: Verification
- [ ] Reproduce every C1–C4 number in `plan.md` from the committed CSVs, not from a notebook.
- [ ] Full v5 suite green. Baseline is **373**, not the 370 first recorded; C1 takes it to 415 (+20 `test_contest.py`, +22 `test_figure_contest_map.py`).
- [ ] Render at paper column width and check nothing is clipped at final size.
- [ ] Confirm each figure carries title + figure only.
