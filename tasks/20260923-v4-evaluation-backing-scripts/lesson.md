# v4 Evaluation Backing Scripts — Lessons

## 2026-09-23

- A column marked "kept for one release, do not publish" became load-bearing
  for a paper section within days. Deprecation notes need to name the thing
  being retired precisely: here the *unbounded nearest-seed rule* is retired,
  while the *nearest-seed column* is now required evidence.
- FALLBACK rows carry a valid `ring` in `*_cells.parquet` because the fallback
  point is a real coordinate. Any new consumer that groups by ring without
  `solved_mask` silently inflates Vanilla (16.9% vs 6.3% ring0 at nside-128).
- Rank flips across the resolution ladder looked like a headline finding until
  the site-clustered paired tests showed the top four are tied. Check
  separability before building an argument on an ordering.

## 2026-09-23 (T1)

- **A count that sums per-run units reads like a count of things.** "65 unique
  site coordinates" was wrong in two places in the companion and twice in this
  task's own plan: 65 is the sum of three runs' site counts, and only 43
  physical coordinates are involved. Whenever a total is a sum over groups,
  publish the union beside it — `site_diagnostics` now emits both, and
  `class_collapse`'s manifest emits `n_classes_if_unioned` beside the reason
  not to use it.
- **The FALLBACK trap catches people who already know about it.** Computing
  ICC during planning — with the caveat in front of me — I grouped on `ring`
  without `solved_mask` and got Vanilla at 16.9% instead of 6.3%, which is the
  exact number the caveat names. A documented trap is not a mitigated one;
  `sites.py` re-exports `solved_mask` so the right predicate is in hand
  wherever the site key is.
- **Don't pick a test fixture by distance on a grid.** Two points 78 km apart
  landed in *different* nside-16 cells (407 km) because they straddled a
  boundary; the pair that merges from nside-64 down is 0.3° of longitude
  apart. HEALPix boundaries decide cell identity, not separation — sweep for
  the fixture, do not reason to it.
- **A manifest that quotes a measurement re-introduces the problem the task
  exists to fix.** The first draft hard-coded "26/33/38/40" and "as01 holds 20
  sites but 18 classes" as prose. Both are now computed
  (`_unioned_classes`, `_quantizer_floor`) — otherwise the manifest is another
  `PROVISIONAL_WEIGHTED` waiting to go stale.
- **Check the numbers a section suppresses, not just the ones it prints.** §1's
  six em-dashes hid the cells that falsify its own read-out sentence. The
  suppressed cells were the discriminating ones.
