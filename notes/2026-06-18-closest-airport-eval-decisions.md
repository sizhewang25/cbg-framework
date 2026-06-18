# Closest-airport eval metric — design decisions

**Date:** 2026-06-18
**Branch:** `paper/cbg-bench-four-variants`
**Context:** Add an operator-facing accuracy metric to the v2 benchmark. By
default CBG is scored by `error_km` = haversine(ground-truth, prediction),
which demands ~city-level accuracy. Operators often only care whether the
estimate resolves to the **right airport (IATA metro code)**. This loosens the
accuracy requirement and reframes the lat/lon as "which airport is nearest."

---

## Decision 1 — Airport reference source: **OurAirports**

- File: `airports.csv` from `https://davidmegginson.github.io/ourairports-data/airports.csv`
  (public domain, continuously updated, ~85.6k rows).
- Columns we use: `type`, `name`, `latitude_deg`, `longitude_deg`,
  `iso_country`, `municipality` (city), `iata_code`.

### Filter applied (the semantic crux of the metric)

Keep `type ∈ {large_airport, medium_airport}` **and** non-null `iata_code`
**and** non-null `municipality`.

| filter | count |
|---|---|
| all rows | 85,597 |
| with IATA | 9,056 |
| large+medium + IATA | 4,562 |
| large+medium + IATA + non-null municipality | **4,441** ← chosen |
| large+medium + IATA + scheduled_service=yes | 3,274 |

**Why this filter:** the unfiltered set is 50% heliports / small / closed
strips. An operator reasons in recognizable metro codes (JFK, LHR, FRA), so a
nearby grass airstrip "winning" the nearest-airport test would make the metric
meaningless. large+medium+IATA is the right granularity. We additionally
require a non-null `municipality` (drops 121 city-less entries) — the metric is
operator/city-facing, so an airport with no associated city is not a useful
match target. `scheduled_service=yes` (3,274) remains a documented tightening
option if reviewers want only active commercial fields.

### Slimmed artifact

Store a slim copy at `datasets/static_datasets/ourairports_iata.parquet`
(columns: `iata_code, name, municipality, iso_country, latitude_deg,
longitude_deg, type`). Keeps the repo self-contained and the load fast; the
raw 85k CSV is not committed.

---

## Decision 2 — Nearest-airport search: **sklearn `BallTree(metric='haversine')`**

- Added `scikit-learn ^1.9.0` to deps (sklearn was not previously installed;
  scipy is). Chosen per user direction over the scipy-cKDTree-on-xyz route.
- BallTree gives **exact great-circle** nearest neighbor with no chord→arc
  conversion. Coords go in as **radians** `(lat, lon)`; query distance is in
  radians → multiply by `EARTH_RADIUS_KM = 6371.0` (same constant as
  [rtt_model.py](../scripts/libs/cbg/rtt_model.py#L24)) for km.
- Build the tree **once per combo run** (4,562 points), query `k=1` per target
  for both the prediction point and the ground-truth point. The min distance
  falls out of `k=1`.

---

## Decision 3 — Airport population is a **postprocessing step, decoupled from the runner**

The runner ([runner.py](../scripts/benchmark/v2/runner.py)) is **not touched**.
Airport columns are added by a separate, idempotent postprocessing pass over
existing `targets.parquet` files. Rationale:

- **Re-runnable** — re-run any time after the benchmark; no need to redo CBG
  when the airport set, filter, or match definition changes.
- **Backfill for free** — works on all existing outputs, not just new runs.
- **Faster** — one vectorized BallTree query over every row of a parquet at
  once, instead of a per-target query inside the run loop.
- **No schema coupling** — these columns are *not* added to `TARGETS_SCHEMA`
  (the runner's `ParquetWriter` keeps writing its current schema). The
  postprocessor appends columns on top via an atomic temp-file rewrite.

### Columns appended to each `targets.parquet`

| column | type | meaning |
|---|---|---|
| `truth_airport_iata` | string | nearest airport to ground truth (always set) |
| `truth_airport_km`   | float64 | distance truth → its nearest airport |
| `pred_airport_iata`  | string, null | nearest airport to prediction (null if no pred) |
| `pred_airport_km`    | float64, null | distance pred → its nearest airport |
| `airport_match`      | bool, null | `pred_airport_iata == truth_airport_iata` |

`truth_airport_km` characterizes the target set itself (how airport-dense each
region is — an EU target sits closer to an airport than a rural one), useful
context for interpreting `airport_match`. Re-running recomputes/overwrites the
columns (idempotent), so changing the filter just means re-running the pass.

---

## Decision 4 — Delivery: standalone CLI command + module

- New module `scripts/benchmark/v2/airports.py`: `AirportIndex` (loads slim
  parquet, builds `BallTree`), `lru_cache`d `load_airport_index()`. Loud error
  if the slim parquet is missing.
- New postprocessing entrypoint (Typer subcommand `cmd_airport_eval` in
  [cli.py](../scripts/benchmark/v2/cli.py), additive — does not couple the
  runner): takes an outputs root / glob, loads the index once, and rewrites
  each matched `targets.parquet` in place (atomic).
- **Summary**: the postprocessor emits its own `airport_summary.parquet`
  (per-combo `pred_airport_km` p5..p95/mean/std + `airport_match_rate` over
  SUCCESS/FALLBACK rows), independent of `cmd_summarize`. Keeps the airport
  feature fully self-contained.

---

## Open / adjustable

- Airport filter (large+medium vs +scheduled_service) — chosen large+medium,
  easy to retighten, then just re-run the postprocessor.
- Whether to also record `pred_truth_airport_gap_km` (distance between the two
  nearest airports) — deferred unless needed for analysis.
- In-place append vs sidecar file — chose in-place (one file has everything for
  analysis); atomic temp-rename guards against corruption.
