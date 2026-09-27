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
