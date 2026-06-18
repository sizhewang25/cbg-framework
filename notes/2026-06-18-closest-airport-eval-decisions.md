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

Keep `type == 'large_airport'` **and** non-null `iata_code` **and** non-null
`municipality` **and** `scheduled_service == 'yes'`.

| filter (with IATA + municipality) | count |
|---|---|
| all rows | 85,597 |
| large+medium | 4,441 |
| large+medium + scheduled_service | 3,224 |
| **large only + scheduled_service** | **1,158** ← chosen |

**Why this filter:** the unfiltered set is 50% heliports / small / closed
strips. An operator reasons in recognizable metro codes (JFK, LHR, FRA), so a
nearby grass airstrip "winning" the nearest-airport test would make the metric
meaningless.

Two gates do the work. (1) **`scheduled_service == 'yes'`** is the in-dataset
proxy for "codes operators actually reference" — network operators encode metro
IATA codes in PoP/router **rDNS hostnames** (`sjc`, `sfo`, `lhr`, …), and those
are commercial passenger fields, not GA/military strips. It drops artifacts like
**PAO** (Palo Alto) and **NUQ** (Moffett), Bay Area fields that out-compete real
hubs (SJC/SFO/OAK) on raw distance. (2) **`type == 'large_airport'`** (hub
level): data centers and major PoPs colocate at large hubs, and an empirical
check showed **~82% of our targets already snap to a large airport** (227 of the
285 distinct ground-truth airports under large+medium were large). We chose the
hub-level set over large+medium for alignment with the colocation/hostname
framing.

Tradeoff (recorded): ~18% of targets sit nearest a *medium* airport that is
genuinely closer (median 10.5 km) — those mid-size metros now reassign to a
farther large hub, coarsening `truth_airport_km` slightly. A *true* "appears in
hostnames" filter would use an external hostname-code dictionary (CAIDA
Hoiho/DRoP-style); `scheduled_service` + large-only is the practical proxy. The
`scheduled_service` column is carried into the slim parquet for provenance.

### Slimmed artifact

Slim set lives at `datasets/static_datasets/ourairports_iata.parquet` (columns:
`iata_code, name, municipality, iso_country, latitude_deg, longitude_deg,
type, scheduled_service`). Per repo convention **nothing under `datasets/` is committed** (the big
reference files are all on-disk-only, regenerated/downloaded) — so this artifact
is **not** committed either. Regenerate it with:

```
python -m scripts.benchmark.v2.cli build-airports          # downloads + builds
python -m scripts.benchmark.v2.cli build-airports --src-csv path/to/airports.csv
```

Neither the slim parquet nor the raw 85k CSV is tracked in git.

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
| `pred_truth_airport_km` | float64, null | great-circle gap between the pred & truth airports (null if no pred) |
| `airport_match`      | bool, null | exact: `pred_airport_iata == truth_airport_iata` |

`truth_airport_km` characterizes the target set itself (how airport-dense each
region is — an EU target sits closer to an airport than a rural one), useful
context for interpreting the match metrics. Re-running recomputes/overwrites the
columns (idempotent), so changing the filter just means re-running the pass.

### Decision 3b — threshold match, not just exact equality

Exact nearest-IATA equality is too brittle: multi-airport metros (JFK/LGA/EWR,
FTY/MGE in Atlanta) score as misses even when the prediction is in the right
place. So we store the **continuous** `pred_truth_airport_km` (airport-to-airport
gap) per target — *threshold-free* — and apply the cutoff only at summary time.
The summary headline is `airport_match_rate_within_40km` (40 km = this repo's
city-level `THRESHOLD_DISTANCES` bin and the literature's city-level accuracy);
the exact `airport_match_rate` is kept as the strict reference. The threshold is
configurable (`--threshold-km`); since it's applied at summary time, a different
threshold needs no per-target recompute. Caveat: snapping both endpoints to
nearby airports makes the gap ≈ `error_km` in dense-airport regions, so the
40 km airport match tracks a 40 km `error_km` threshold there — the value is the
operator framing (deliverable is an airport code) and that sparse regions
genuinely differ.

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

- Airport filter — chosen large-only + scheduled_service (1,158): hub-level,
  aligned with data-center/rDNS-hostname colocation (~82% of targets already
  snap to a large hub). Evolution: large+medium (4,441) → +scheduled_service
  (3,224, dropped PAO/NUQ) → large-only (1,158). A hostname-code dictionary
  (Hoiho/DRoP) is the stronger-but-heavier alternative.
- Whether to also record `pred_truth_airport_gap_km` (distance between the two
  nearest airports) — deferred unless needed for analysis.
- In-place append vs sidecar file — chose in-place (one file has everything for
  analysis); atomic temp-rename guards against corruption.
