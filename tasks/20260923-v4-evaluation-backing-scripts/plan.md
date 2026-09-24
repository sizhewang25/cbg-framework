# v4 Evaluation Backing Scripts — Plan

## Background

`cbg-benchmark-paper/sections/evaluation.md` (the drafting companion written
2026-09-23) restructures §Evaluation around three findings that were measured
ad-hoc in a session and exist nowhere in the repo:

1. **Ring-bounded serving-region recovery** — the replacement for v3's retired
   nearest-seed rule. Restricting the nearest-seed question to `ring in {1,2}`
   bounds it by construction (ring-1 caps at ~2.3-2.9x the nominal cell,
   ring-2 at ~3.5-5.3x), which is what the unbounded Voronoi rule lacked.
2. **Per-site stability + clustered significance** — 1,269 targets are 65
   sites (one coordinate within one run; 43 distinct coordinates); ICC 0.64-0.99, effective n 66-99. Paired site-level tests show
   the top four methods are statistically tied at every rung, while the
   per-site profile separates them operationally (Shortest-Ping resolves 13
   sites completely, Octant-Hull 3).
3. **Answer-space class collapse across rungs** — 65 sites -> 52 distinct
   classes at nside-16; coarsening destroys class distinctions, not just
   tolerance. *(Delivered 2026-09-23 as `class-collapse`.)*

Three of the companion's tables are marked **[SCRIPT NEEDED]**. Until they are
produced by the pipeline, the section rests on numbers that cannot be
regenerated — which is exactly the failure mode that put `PROVISIONAL_WEIGHTED`
into §8.1 and the retired nearest-seed rule into the current `evaluation.tex`.

## Context

- Package: `scripts/analysis/v4/` (typer CLI, `@app.command`), modules under
  `scripts/analysis/v4/modules/`, tests under `scripts/analysis/v4/tests/`.
- Input: `outputs/analysis/v4/<run>/target-cls-accuracy/healpix-<n>/*_cells.parquet`
  (columns `target_id, target_lat, target_lon, pred_lat, pred_lon, status,
  fold, tg_cell, tg_seed_id, pred_cell, ring, error_km, nearest_seed_id_retired`).
- Runs: `as01/as02/as03-260728-260802-mesh`, rungs nside 16/32/64/128.
- Cross-run pooling lives in `modules/cross.py`; figure conventions and the
  validated palette in `modules/figure_outcome_bars.py`.
- Driver: `scripts/analysis/v4/create_analysis_artifacts.sh` (arms: mesh /
  weighted; needs `.venv` on PATH).

## Goals

- `recovery` command + `plot-recovery` figure: per-method ring0, ring0+ring1
  recovered, ring0+ring1+ring2 recovered, and the per-ring agreement rates,
  with the measured ring bound (max `error_km` per ring) emitted to the
  manifest so the bound is published rather than asserted.
- `site-stability` command + figure: per-site resolution profile
  (all / some / none), micro-vs-macro rate, within-site unanimity.
- `significance` command: ICC, design effect, effective n, naive vs clustered
  CI, and paired site-level tests for every method pair at every rung.
- Class-collapse table across rungs (sites, distinct classes, merged sites),
  emitted from the answer-space builder rather than a one-off.
- Every companion table marked **[SCRIPT NEEDED]** regenerable by one command,
  with a manifest recording the policy decisions.

## Approach

Follow the v4 house style: each command writes CSV + PNG + manifest.json, the
manifest carries the policy prose, and cross-run outputs land under
`_cross/<slug>/<dataset-set>@<arm>/`. Add a `modules/sites.py` holding the one
canonical site key so three commands cannot drift apart (the lesson from
`finding-cluster-eval-consistency-audit`).

Reuse `classify.solved_mask` everywhere. Do not re-derive ring or seed logic.

Sequence: sites module -> recovery -> significance -> site-stability ->
class-collapse -> figures -> wire into `create_analysis_artifacts.sh` -> feed
numbers back into `evaluation.md` and strike the [SCRIPT NEEDED] markers.

## Caveats

- **`nearest_seed_id_retired` is scheduled for deletion.** `classify.py` says
  it is "kept for one release ... not for use, the manifest says not to
  publish it." §4 of the paper now depends on it. It must be promoted to a
  first-class column under a name that says what it is (e.g.
  `nearest_seed_id`), with the retirement note rewritten to say the *unbounded*
  rule is retired, not the column. Otherwise the next cleanup deletes the
  paper's evidence.
- **Never emit the word "Voronoi"** in a column name, figure label, or
  manifest. The metric is *ring-bounded serving-region recovery*; the old name
  invites the objection the bound exists to defuse.
- **FALLBACK rows carry a ring.** The fallback point is the shortest-ping VP,
  so a naive groupby credits Vanilla 16.9% ring0 at nside-128 instead of 6.3%.
  Denominator = every evaluated target; numerator = solved rows only.
- **The seed set is rung-dependent**, so cross-rung recovery trends move two
  things at once. The manifest must say so; the figure must not imply
  otherwise. *(Corrected 2026-09-23: the "19/22/23/23" first written here is
  not any run's ladder. Measured per run at nside 16/32/64/128 — as01
  15/17/18/18, as02 19/22/22/22, as03 18/21/23/23. as01 never reaches its 20
  sites at any rung, so its separability is capped at 0.90 by the quantizer.)*
- **~65 sites is too few for a cluster-robust sandwich estimator**
  (`finding-operator-datasets-20-regions`). Use paired site-level tests and a
  clustered bootstrap, not HC/CR standard errors.
- Ring-2 agreement is high-variance and method-dependent (4%-95% at nside-128).
  Report ring-1 as headline, ring-2 as a secondary line; do not publish a
  single lumped "recovered" number.
- Site key must be canonical, not a coordinate rounding chosen per script.
  *(Settled 2026-09-23: there is **no** region id — `*.mainland.sanitized.csv`
  carries only vp/target coordinates, `vp_asn`, `vp_country`, `rtt_ms`, and the
  reconstruction manifests list `target_city`/`target_asn` as unrecoverable.
  The key is `(run_id, target_lat, target_lon)` at 6 dp, owned by
  `modules/sites.py`. Rounding is stable 2–6 dp. **65 is the sum of per-run
  site counts, not a count of places** — pooled there are 43 distinct
  coordinates, 7 shared by all three runs.)*
