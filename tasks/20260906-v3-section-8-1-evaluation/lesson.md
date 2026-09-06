# §8.1 Evaluation Scripts (v3) — Lessons

## 2026-09-06

- **Measure the guarantee against the thing being classified.** The half-gap
  rule inherited from `eval_source` reads naturally as "is a VP close to the
  target", but the guarantee it is supposed to provide is about the *class*.
  Only `d(VP, tg_seed) < margin` implies the VP's argmin seed is `tg_seed`;
  `d(VP, target) < margin` does not, and the gap is the target's own offset from
  its seed. Measuring the convenient distance instead of the meaningful one
  would have silently broken the nesting the whole ladder is built on.
- **A min below a threshold *is* an existence claim** — but only when the min is
  taken over the same metric being thresholded. `closest_vp_km` is a min over
  VP→target distance, so testing *it* against a seed-distance threshold picks
  the wrong VP. Renaming the quantity (`tg_seed_nearest_vp_km`) was the fix;
  reusing the old name would have imported the bug.
- **Ask where a fact already lives before adding a column for it.** The latent
  (whole-roster) proximity columns were a good idea with an existing owner —
  `bipartite.py`'s `measured_nearest_vp_ratio_per_target`. Two copies of one
  fact in two directories are free to disagree, which is the failure mode the
  v3 README already argues against for the hierarchy rungs.

- **A name that presupposes the verdict will hide the finding.** The stratum
  called `structural_failure` is where Octant-Hull scores 39/40 and the baseline
  scores 0/40 — the single strongest case for CBG in the dataset, filed under a
  heading saying it could not be won. Named for what the target *requires*
  (`geometry_only`) rather than for an outcome, the same number reads correctly.
  Inheriting v2's gloss ("MTL required") without measuring it was the actual
  mistake; one query settled it.
- **"Does this increase coupling?" was the wrong question, and worth asking
  anyway.** The audit showed the dependency was pre-existing and the hard
  surface unchanged. What it *did* surface, one level over, was two independent
  readers of the same file — which is not coupling but duplication, and the
  failure mode is drift rather than fragility. The question found a real problem
  by not finding the one it asked about.
- **A degenerate case that "cannot happen" happened on two of four runs.**
  `density_bins` indexed `edges[b+1]` assuming at least two quantile edges; on a
  run where every seed shares one `nearest_seed_km` there is one. h3 at a fixed
  resolution puts many seeds at exactly the same pitch, so the tie case is
  normal rather than exotic. Fixed by making the contract "always at least two
  edges" instead of making every caller special-case it.
