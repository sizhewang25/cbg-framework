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
  that cannot exceed it.
- **Bound a density by reflection, not by clipping the drawing.** Clipping the
  violin body after the fact leaves a flat-topped shape that stops reading as a
  violin, which is a reviewer's first comment. Mirroring the sample about each
  bound before the KDE is bounded by construction, preserves the mass, and
  looks like the plot it is.
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
