# §8.1 Evaluation Scripts (v3) — Report

**Status**: In Progress
**Created**: 2026-09-06
**Last Updated**: 2026-09-06

## Summary

**All three phases complete.** The `tg_seed` rename landed with all eight
`topn_accuracy.csv` byte-identical; `build-proximity` labels every target on all
four runs at both grids; and `breakdown-accuracy`, `confusion-density` and
`table-accuracy` consume those labels. 442 tests pass.

## Verification so far

| check | result |
| --- | --- |
| `topn_accuracy.csv` unchanged by the rename | **8/8 byte-identical** (`cmp`) |
| `has_proximate_sping_vp` == `shortest_ping` top-1 correct | **exact, 8/8 run×grid** (947 targets) |
| `sping_vp_to_tg_seed_km` vs `shortest_ping`'s `error_to_tg_seed_km` | **max abs diff 0.0000 km** |
| diamond implications (all four) | **zero violations, 8/8** |
| `closest_vp_to_tg_km` vs `bipartite-graph`'s `nearest_measured_vp_km` | **max abs diff 0.0000 km** |
| `n_measured_vps` vs `degree_to_vp` | **equal on every target** |
| `topn_accuracy.csv` unchanged by the `io.load_sping_vp` refactor | **8/8 byte-identical** |
| `accuracy_by_taxonomy.csv` sums to the target count per (method, N) | **holds on all 4 runs** |
| `pytest scripts/analysis/v3/tests/ -q` | **442 passed** |

## Base rates, `h3-4`

|                     |   n | prox_vp | disc_vp | prox_sping | disc_sping | structural | selection | baseline_ok | occupied cells |
| ------------------- | --: | ------: | ------: | ---------: | ---------: | ---------: | --------: | ----------: | -------------: |
| as01-260728-260802  | 399 |  1.0000 |  1.0000 |     0.6366 |     0.6366 |          0 |       145 |         254 |              2 |
| as02-260728-260802  | 412 |  0.9029 |  0.8544 |     0.3689 |     0.3592 |         40 |       220 |         152 |              6 |
| as03-260728-260802  | 458 |  1.0000 |  1.0000 |     0.4323 |     0.4323 |          0 |       260 |         198 |              2 |
| as7018_us_test01    |  78 |  0.6282 |  0.5769 |     0.4359 |     0.3205 |         29 |        15 |          34 |              5 |

Three things this table already settles:

- **The zero-variance risk was real, and partial.** as01 and as03 have both
  left-column flags constant at 1.0 — every target has a VP inside its seed's
  half-gap. as02 and as7018 vary. So the left column separates on exactly two of
  the four runs and must be reported with `zero_variance` naming the others.
- **as02 is the operator run worth leaning on.** Six of the sixteen possible
  cross-tab cells are occupied, including the incomparable pair; as01/as03
  occupy two. Any §8.1 table that needs *contrast* on the left axis has to come
  from as02, and the three operator runs staying unpooled (D2) is what preserves
  that.
- **`structural_failure` on as7018 is 29/78 = 0.372**, which is v2's recorded
  `no_proximity_share` for that run to four decimals — despite v2's rule being
  cluster-keyed and VP→target and v3's being grid-keyed and VP→seed. Treat as a
  coincidence of this dataset until checked; it is *not* an invariant, and as01
  disagrees (v2 0.0, v3 0.0 — agrees; as02 v3 0.097 has no v2 counterpart to hand).

## Findings

### Design decisions taken (2026-09-06)

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | Half-gap rule primary, nearest-seed argmin as cross-check | More conservative, so a stronger geography guarantee |
| D1b | All distances measured **VP → seed**, not VP → target | Only then does `d < margin ⟹ argmin seed is tg_seed`, which is what makes the diamond's implications hold. VP→target leaks by `cell_offset_km` (p50 16–20 km, max 26 km at h3-4) |
| D2 | as01/02/03 reported separately, never pooled | Their differing peering/routing regimes *are* the comparison §8.1's "clean network topology matters" subsection rests on |
| D3 | Operator first, public in its own subsection | §7.3 declines the head-to-head. `setup` is a meaningless placeholder on as01–03 and gates nothing |
| D4 | Topology-paired dataset deferred | Needs VP+TG co-curation and a benchmark re-run |
| D5 | Ladder is a **diamond**, not a chain | `4 ⟹ 2 ⟹ 1` and `4 ⟹ 3 ⟹ 1`; 2 and 3 incomparable |
| D6 | Booleans at top-1; no `--top-n` on `build-proximity` | Top-N tolerance is an analytical decision applied at consumption |
| D7 | Dense answer region = **seed** property | Uses the tg_seed's `nearest_seed_km` |
| D8 | No latent/whole-roster columns | The latent half already has an owner in `bipartite.py` |
| D9 | `sping_` throughout v3 | Brevity; `eval_source`'s `shortest_ping_*` is v2 legacy, not a constraint |

### Two subtleties surfaced during design

- **`has_proximate_sping_vp` ⟺ Shortest-Ping correct at top-1**, by
  construction. Its separation for `shortest_ping` is a pipeline self-check
  rather than a result; at top-3 it is informative.
- **The left chain selects two different VPs.** Minimizing `rank(tg_seed | VP)`
  and minimizing `d(VP, tg_seed)` are different objectives — a distant VP in a
  sparse region can rank `tg_seed` first while a nearer VP in a dense region
  does not. Hence separate `*_vp_id` columns.

### One design change made during implementation

**The shortest-ping VP is read from `eval_source`, not re-minimized over RTT.**
The plan said "identify the sping VP by min RTT". Doing that would have made the
tautology approximate: `classify` scores the baseline on `eval_source`'s
`shortest_ping_vp_lat/lon`, so a second RTT minimization here would break ties
differently and the flag could disagree with the score on a target where two VPs
share a min RTT. Reading the VP from the same place `classify` does makes the
self-check exact — which is the entire reason for keeping a tautological column.
`meta.json` records how many sping VPs fall outside the roster (0 on all four
runs) as the diagnostic that would have surfaced a mismatch.

### Two decisions taken in Phase 2

**`io.load_sping_vp`** — raised by the user as a coupling question. The audit
found the coupling was pre-existing (`classify` already required
`shortest_ping_vp_lat/lon` and `bipartite` already imported
`eval_source.load_canonical_csv`); the real smell was that `classify` and
`proximity` *independently* resolved the same VP from the same file. They agreed
today and were free to drift tomorrow, which would silently invalidate the
tautology. One reader now, pinned by a structural test as well as a numeric one.

**`structural_failure` → `geometry_only`** (with `selection_miss` /
`selection_hit`). The old name, inherited from v2's `NO_PROXIMITY` and glossed
there as "MTL required", is contradicted by the data: on as02 the stratum's 40
targets are exactly two seeds whose nearest measured VP is 92 km and 153 km out,
with `tg_seed_best_rank == 1` throughout — the true seed is always the runner-up.
Shortest-Ping and Vanilla answer 0 of 40; **Octant-Hull answers 39**. It is a
ceiling on Shortest-Ping, not on CBG, and the name now says so. v2's vocabulary
is deliberately not reused for the other two terms either, since the two layers
key their taxonomy to different answer spaces and different endpoints.

## Findings from the consumers

**SoI CBG is the Shortest-Ping baseline in disguise.** Crossed with
`has_proximate_sping_vp` at top-1, `million_scale_cbg` scores 1.000 / 0.993 /
1.000 where the flag holds and 0.069 / 0.000 / 0.012 where it does not, on
as01 / as02 / as03 — φ = 0.947, 0.995, 0.987 against a *baseline correctness
indicator*. Its aggregate accuracy sits within three points of the baseline's on
all three runs, and this is why. Not a tautology: the flagged cell one row above
it is, which is exactly what the `is_tautological` marker exists to distinguish.

**Vanilla misses narrowly; the baseline misses by a region.** Median
`boundary_margin_km` over wrong top-1 rows: Vanilla 31 / 70 / 87 km versus
Shortest-Ping 670 / 606 / 281 km (as01 / as02 / as03). Vanilla's high fallback
rate and low accuracy are not the same failure as the baseline's — its answers
land near the boundary of the right cell. 42% / 27% / 55% of all top-1 mistakes
land on the true seed's *nearest* neighbour.

**Answer-space density costs accuracy, but not universally.** Shortest-Ping
across density tertiles: as01 0.33 → 0.71 → 0.82, as02 0.16 → 0.42 → 0.51, but
as03 0.67 → 0.25 → 0.44. The non-monotone run is the one to explain, not the one
to drop.

## Conclusions

The §8.1 scripts are complete and produce paper-ready output
(`_cross/accuracy-table/*/accuracy_table.h3-4.md`). Three findings are already
sitting in the artifacts and want writing up: SoI-as-baseline, the
Vanilla-vs-baseline failure-mode split, and `geometry_only` as the stratum where
Octant-Hull earns its keep. §8.2 and §8.3 remain, and both read against
`target-proximity/`.
