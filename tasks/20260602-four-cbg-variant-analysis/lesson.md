# Four-CBG-Variant Analysis Across VP Setups — Lessons

## 2026-06-02

- The four variants are two clean LTD pairs sharing a geometry stack
  (vanilla/SOI = spherical_circle + boundary_vertex_mean; octant/spotter =
  planar_annulus + monte_carlo_medoid). Only within-pair contrasts isolate the
  LTD; disk-vs-annulus is a 3-change stack effect.
- `_nofil` = unweighted `planar_annulus` (every constraint ANDed equally), which
  is the collapse-prone configuration — fallback% must be reported separately
  from error.
- Data lives in two trees keyed by run_id (= `*_final` stem): per-fold
  `targets.parquet` under `scripts/benchmark/v2/outputs/<stem>/...`, and
  `eval_observations.parquet` (VP coords for closest-VP distance) under
  `scripts/benchmark/v2/inputs/ripe_atlas_asn_corpora/<stem>/...`. The analysis
  `outputs/` tree holds merged plots, NOT per-target parquets.
- Git worktrees do **not** contain gitignored data dirs (`inputs/`, `outputs/`).
  Subagents running scripts from a worktree need symlinks back to the main repo's
  data dirs, or `_io.py`'s `REPO_ROOT`-relative paths resolve to empty worktree
  paths. The final consolidated run was done in the main worktree (real data) to
  sidestep this entirely.
- Parallelization pattern that worked: commit the shared foundation first, branch
  N worktrees off that commit, give each subagent the exact shared API + fixed
  output filenames + a "don't touch foundation files" rule. Disjoint new files
  ⇒ conflict-free merges.
