# v5 Evaluation Backing Scripts — Plan

## Background

`§1b` of `cbg-benchmark-paper/sections/evaluation.md` ("Where the near-exact
predictions come from") now carries the strongest pro-CBG argument in the
paper: the six methods' best cases are **different targets**, only the two
baselines' near-exact predictions are explained by VP proximity, and the CBG
gain comes from keeping the constraint radius that Shortest-Ping discards.

Roughly half the numbers backing that argument have **no saved artifact**.
They were derived ad hoc over `vp_proximity.load()` + `cohort_frame()` during
drafting. The companion document's own opening rule is that every number is
traceable to a file in `outputs/analysis/v4/`; §1b currently violates it and
says so in a `[PENDING MODULE]` block. This task closes that gap.

The v4 task (`20260923-v4-evaluation-backing-scripts`) delivered
`modules/vp_proximity.py` and the `plot-vp-proximity` command, which cover the
two VP distances, the per-method bounds, the violin quantiles and the gap
ratios. Everything below is the layer above that.

## Context

- Substrate: `as01/as02/as03-260728-260802-mesh` pooled, **1,269 targets**,
  six methods, nside 128 for `solved_mask`.
- **The active line is `scripts/analysis/v5/`, not v4.** v5 already carries
  `classify`, `cross`, `paths`, `methods`, `grid`, `sites`, `figure_error_cdf`
  and `figure_outcome_bars`. It does **not** yet carry `vp_proximity`, so
  porting that is step one of this task, not an assumption.
  Naming convention observed in v5: figure-producing modules are prefixed
  `figure_`; analysis modules are plain. `cohort_overlap` and `vp_ranking`
  emit CSVs only, so they stay unprefixed; the port becomes
  `figure_vp_proximity.py`.
- Reusable from v4 until ported: `scripts/analysis/v4/modules/vp_proximity.py`
  (`load`, `cohort_frame`, `stats_table`, `vp_distances`, `COHORTS`) and
  `modules/edges.resolve_source_csv` (gives `sping_rtt_ms`, `geo_vp_rtt_ms`,
  `n_vp` per target). v4's `test_vp_proximity.py` has 15 tests that must
  survive the port, `TestCohortExcludesUnanswered` above all.
- Output convention: `_cross/<combo_run_id>/<kind>/`, matching how
  `vp_proximity` writes today.
- Numbers to reproduce exactly (these are what §1b prints):
  - p5 union **254** distinct targets, degree histogram `{1:154, 2:78, 3:20, 4:2}`,
    max degree **4**; p25 union **686**, `{1:147, 2:191, 3:157, 4:88, 5:68, 6:35}`.
  - Overlap with Shortest-Ping's cohort — p5: SoI 96.8, OCT-S 11.1, OCT-H 9.5,
    SPO 0.0, VAN 0.0; p25: 94.0, 47.0, 42.6, 43.8, 25.6.
  - Refusal on S-P's cohort — VAN 57.1% (p5) / 59.3% (p25), all others 0%.
  - Error p50 on S-P's cohort — **S-P reference 0.46 / 8.77**; p5: SoI 0.5,
    OCT-S 41.1, OCT-H 7.5, SPO 149.1, VAN 85.2; p25: 13.8, 20.8, 20.8, 152.9,
    59.0. The reference row must be emitted, not left implicit: without it the
    err column has no scale and SoI's 13.8 km reads as good.
  - Per-target share closer than S-P on S-P's own cohort (p5 / p25) — OCT-H
    7.9 / 26.2, OCT-S 9.5 / 26.8, SoI 0.0 / 16.7, VAN 3.9 (solved only) / 0.0,
    SPO 0.0 / 0.0.
  - Baseline-beats-method by >1 km on the method's own cohort — SPO 87.3% (p5)
    / 83.9% (p25), VAN 51.4% (p25), Octants 0% / 14.8%.
  - Far-tail counts at p25, `d_sping >` 100/200/400 km — OCT-H 8/1/1,
    OCT-S 13/6/6, SPO 56/23/2, VAN 27/0/0, SoI and S-P 0/0/0.
  - Mechanism table: 7 as01 targets at closest 21.8 km @ 3.1 ms vs smallest-RTT
    140.5 km @ 2.7 ms (SoI radius 265 km, OCT-H error 2.1–3.7 km); `tg-ac636bd`
    at 154.2 km @ 20.1 ms vs 720.7 km @ 13.5 ms (radius 1350 km, error 5.3 km).

## Goals

1. Every number in §1b traces to a committed CSV under
   `_cross/<combo>/<kind>/`, and the `[PENDING MODULE]` block is deleted.
2. A `cohort_overlap` module: per-cohort set structure (union, degree
   histogram, pairwise overlap matrix) plus the behaviour-on-a-reference-cohort
   decomposition (shared / answered-not-best / refused, with error percentiles).
   Reference method is a parameter, defaulting to `shortest_ping`.
3. A `vp_ranking` module (or an extension of `vp_proximity`): per-target
   smallest-RTT vs closest VP with RTTs, implied SoI constraint radius, and
   the far-tail counts at configurable thresholds.
4. Tests that pin the two traps this analysis already fell into, and the two
   new ones below.
5. `report.md` records which §1b claims are now file-backed and which were
   *rejected* during drafting, so they are not re-proposed.

## Approach

Three steps, all on the v5 line, each mirroring how `vp_proximity` is laid out
(module + typer command + test file + stats CSV written beside any figure).

**Step 0 — port `vp_proximity` to v5** as `figure_vp_proximity.py`, against
v5's `classify`/`cross`/`paths`/`methods`. Carry all 15 tests across. Confirm
the ported module reproduces the committed v4 CSVs cell for cell before
anything is built on top of it; the numbers in Context are the check.

**Step 1 — `v5/modules/cohort_overlap.py`**
- Reuse the ported `load` and `cohort_frame` verbatim; do not re-derive
  cohorts, or the `solved_mask` trap re-opens.
- `set_structure(rows)` → union size, degree histogram, pairwise overlap.
- `against_reference(long, rows, reference="shortest_ping")` → per method:
  shared / answered-not-best / refused shares over the *reference's* cohort,
  plus error percentiles on solved rows only.
- `baseline_margin(rows, margin_km=1.0)` → share where `d_sping < error_km -
  margin`. The margin is required, not cosmetic: at zero margin Shortest-Ping
  scores 100% against itself on floating-point noise, since `d_sping` and
  `error_km` agree to ~0.006 km.
- Emits one tidy CSV per cohort; no figure (the overlap bar chart was
  considered and rejected — see Caveats).

**Step 2 — `v5/modules/vp_ranking.py`**
- `radii(distances, soi_kms=200_000)` → constraint radius implied by each RTT.
- `far_tail(rows, thresholds=(100, 200, 400))` → counts and distinct-value
  counts per method, since replicas make the raw count overstate independence.
- `mechanism_rows(rows, min_sping_km=100)` → the per-target table with both
  VPs, both RTTs, the radius, `n_vp` and the method's error.
- Classify each far-tail row as `short-range-unrankable` vs `indirect-routing`
  by whether the *closest* VP's RTT has unexplained delay beyond a threshold,
  rather than asserting one mechanism for all of them.

CLI: `report-cohort-overlap` and `report-vp-ranking`, same option shape as
`plot-vp-proximity` (`--run-id` repeatable, `--cohort` repeatable, `--method`,
`--nside`, `--outputs-root`, `--analysis-root`).

## Caveats

- **The `solved_mask` / FALLBACK trap.** A FALLBACK row carries a real
  `error_km` because the fallback prediction is the shortest-ping VP's
  coordinate. Any cohort built by ranking `error_km` without the mask silently
  fills with give-up rows; it moved Vanilla's p5 bound 69.0 → 33.1 km and the
  wrong number was the flattering one. Never rebuild cohorts locally.
- **Replica quantization.** ~20 IP replicas per coordinate. Octant-Spline's six
  targets past 400 km sit at an identical distance, so a raw count of 6 is
  ~1 independent observation. Every count this task emits must ship a
  distinct-value count beside it.
- **Privacy.** Target-based analysis only. No site names, no coordinates, no
  location strings in any output, figure or commit message. Target IDs are
  fine.
- **The overlap bar chart is rejected, deliberately.** Two failure modes
  documented in §1b: at p5 Vanilla and Spotter both plot 0% meaning opposite
  things (57% refusal vs answered-and-149-km-off), and at p25 bar height does
  not track quality (Spotter 43.8% at 152.9 km draws level with Octant-Hull
  42.6% at 20.8 km). Ship the table. Do not let a later pass "improve" this
  into a figure.
- **Do not claim "multilateration is stable."** Spotter has more far-VP reach
  than either Octant and is the worst method. The conditional claim is the
  only one that survives.
- **Framework repo is shared.** `modules/edges.py` and parts of `cli.py` were
  uncommitted work from a parallel session at the time the v4 task closed;
  check `git status` before assuming an import is on disk for everyone.
