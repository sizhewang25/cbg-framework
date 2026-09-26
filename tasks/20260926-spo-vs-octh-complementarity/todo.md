# SPO vs OCT-H Complementarity — Todo

## Phase 0: Shared substrate
- [ ] Decide one module with a `--figure` selector vs four modules; read how `figure_outcome_bars` carries `layout` × `mode` before choosing.
- [ ] `site_contest(run, method_a, method_b)`: per-site join of correct counts with `solved_mask` applied, returning `k_a, n_a, k_b, n_b, tg_seed_id, tg_lat, tg_lon`.
- [ ] Four-way category by **direct count comparison** — `neither` when both zero, else `a`/`b`/`tied`. No threshold anywhere.
- [ ] `centroid_km`: distance from each site to the run's seed-cloud centroid, in EPSG:5070 via `projection.project`.
- [ ] McNemar exact (`binomtest(b, b+c, 0.5)`) plus the attainable floor `2*0.5**(b+c)`, so an underpowered mesh says so.
- [ ] Test: a 9/20 vs 20/20 site is `SPO wins`, never `tied` or `only SPO` — the Los Angeles regression.
- [ ] Test: a FALLBACK row never reaches any count.

## Phase 1: C1 — the XOR map
- [ ] Port the prototype: one panel per mesh, four marker styles, per-site label `SPO | OCT-H | total`.
- [ ] Drop McNemar and the stats box from the figure; both go to manifest + CSV.
- [ ] Title only — no footnote, no caption block.
- [ ] Label de-collision: repel from markers as well as from other labels (the prototype does not, and the as03 northeast cluster overlaps).

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
- [ ] Full v5 suite green (370 passing before this task).
- [ ] Render at paper column width and check nothing is clipped at final size.
- [ ] Confirm each figure carries title + figure only.
