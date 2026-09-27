# SPO vs OCT-H Complementarity — Lessons

## 2026-09-26

- **A majority rule over replicas is a silent classifier.** `correct_frac >= 0.5`
  turned Octant-Hull's 9/20 at Los Angeles into a clean loss, and the site was
  painted an exclusive Spotter win — on a site where Octant-Hull was the more
  accurate method. Compare counts directly, or emit an explicit `split`
  category. Never let a threshold decide a category silently.
- **`solved_mask` has now been forgotten twice**, once in analysis and once in
  plotting code. A FALLBACK row carries the baseline's coordinate, so omitting
  the mask credits a method for the answers it declined to produce.
- **Three geometric mechanisms were proposed and falsified in one session.**
  Each was plausible and each took one cheap query to kill. Test the
  discriminator before planning a module around it.
- **A figure changed the question twice.** The per-method maps showed the
  replica structure (~20 stacked markers, counts of exactly 20) and the
  Seattle/Portland pair more clearly than any table had. Plot before theorising.
- **Relax labels in the metric the labels are shaped in.** A spring
  relaxation with one isotropic separation assumes round labels. These are 4:1,
  so any single value in degrees is wrong on one axis: large enough to clear
  two labels side by side flings them apart vertically, small enough to sit
  right vertically leaves them overlapping horizontally. Scale the space by the
  label box first, then one rule is correct on both axes. And measure the
  result into the manifest — the relaxation can be handed an unsatisfiable
  cluster, and when it is, it should say so rather than look tuned.
- **Font size is absolute; map scale is not.** Keep a drawn label's dimensions
  in inches and convert at draw time from the panel width actually in use.
  Pinning them in degrees bakes in one figure size, and the first thing a paper
  asks for is a different one.
- **A figure the paper cites is not a place to enable an improvement in
  passing.** `place_labels` grew the obstacle repulsion its own docstring had
  been asking for; `figure_outcome_map` still does not pass it, because turning
  it on would silently move every count on a committed figure. The capability
  and the decision to use it are separate changes.
- **Pick the metric the thing is defined in, not the one it is drawn in.** The
  cell partition is great-circle nearest seed; EPSG:5070 exists so `cells` can
  draw it. Measuring peripherality in the plane was defended in a docstring
  with a rationale invented after the fact ("a property of the plane the
  partition is drawn in") — which is the tell. A projection in a number that
  scoring never touches is a smell even when, as here, it changes nothing.
- **A fixed-canvas figure clips its axis label silently.** `tight_layout` fits
  the label's height and lets a long one overrun both sides; saved without
  `bbox_inches="tight"` the overhang is simply cut, and the first version lost
  the last character of the x-label. Measure the label against the canvas in a
  test rather than picking a figure width that happens to work.
- **A KDE is a claim, and the default bandwidth makes it for you.** A violin of
  65 shares piled on 0 and 1 drew Spotter's four split sites as a waist the
  width of Octant-Hull's twenty-six, and spilled density past 1 on a variable
  that cannot exceed it. Reflection fixed the arithmetic and the shape still
  read wrong — after two attempts the answer was that a bounded variable
  concentrated on its endpoints should not be smoothed at all. An empirical CDF
  has no bins, no bandwidth and no choices, and the claim sits in its geometry:
  the jump at each end is the share at that end.
- **Report the percentiles the figure draws.** Whiskers at p5/p95 state a bound
  that nothing else in the artifact does, so the quantile list has to include
  them or the ink is unquotable.
- **Whiskers at p5/p95 with no outliers hide exactly the observation that
  argues the other way.** Legitimate, and it has to be said in the prose:
  the figure then reads "strictly more stable" when the tail says otherwise.
- **Never name a test fixture's site by its position in the source list.**
  `sites.site_ids` numbers sites in sorted key order, so the mapping is a
  permutation. A test that conflated the two asserted against the wrong site
  and passed anyway, for months if a second test had not needed the same site.
  Go through an explicit index.
- **"Spread" of what, exactly.** The standard deviation of an *error magnitude*
  is not the spread of the *answers*: two predictions equally far from the
  target in opposite directions have identical magnitudes and score as perfect
  agreement. Worth measuring the difference before choosing — here it was
  Spearman 0.90, same conclusion, 11 sites apart, and the simpler statistic was
  kept on those terms. When a known limitation is accepted rather than fixed,
  pin it with a test that says so, or the next reader files it as a bug.
- **Keep a consistency metric in the units of the grid, not kilometres.** A
  spread in km sits next to an error distance in km and gets read as accuracy,
  which is exactly what it is not: a site whose replicas all agree on one grid
  scores zero whether that grid is right or thirty-eight grids away.
- **Put the reading in the curve's geometry, not in an annotation.** The three
  shares C4 quotes are the height left of zero, the jump at zero, and what is
  left — so the figure needs no numbers written on it, and a test can assert
  that the geometry and the manifest agree. Direction is carried by two arrows
  above the curve rather than by tinted half-planes -- it is what the axis
  encodes, and an arrow survives greyscale where a pale tint does not.
- **Reserve annotation space in data units where the data has a ceiling.** A
  CDF stops at 1, so a band above it is empty whatever the data does. Placing
  the arrows there beats drawing them outside the axes, which `tight_layout`
  does not reserve room for and a fixed-canvas save then clips.
- **Align two grouped Series before comparing them.** A per-site head-to-head
  built from two `groupby().median()` calls compares fine until one method has
  no rows at some site, and then pandas raises rather than silently
  misaligning. Join explicitly and report how many pairs were comparable.
- **"A small portion" needs its denominator stated, and often two.** The same
  tail is 13.6% of Spotter's exclusive cohort and 2.6% of all evaluated
  targets — a factor of five apart, and each is the honest answer to a
  different question. Report both or the sentence is unfalsifiable.
- **Reach for symlog whenever zero is a real value.** It is not only for
  signed data: a non-negative offset that is genuinely 0 on 13% of a cohort
  cannot go on a log axis, and dropping those rows to make the axis work is
  hiding data to suit the drawing.
- **Count in the unit the metric grades in.** `cell_label` is a verdict about a
  Voronoi cell, so a per-site success ratio quietly weights a facility that
  shares a seed with another twice and reports a denominator the metric never
  uses. Two figures in one module can legitimately count different things —
  the spread is about identical inputs, which is a site property — but then
  the unit has to be named in the manifest and carried in the twin's key, or
  61 and 65 sit side by side with nothing saying which is which.
