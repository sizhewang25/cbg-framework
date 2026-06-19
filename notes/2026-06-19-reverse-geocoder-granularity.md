# reverse_geocoder granularity + how geo_eval uses it

**Date:** 2026-06-19
**Code of interest:** [scripts/benchmark/v2/geo_eval.py](../scripts/benchmark/v2/geo_eval.py) — `_reverse_geocode_cc` ([L45-60](../scripts/benchmark/v2/geo_eval.py#L45-L60)), `annotate_targets_geo` ([L63-74](../scripts/benchmark/v2/geo_eval.py#L63-L74))

---

## What the library is

`reverse_geocoder` (the `rg` import) is a fully **offline** `(lat, lon) → place`
lookup. It ships the **GeoNames cities1000** dataset (~150k places with
population ≥ 1000), builds a **k-d tree** over their coordinates, and answers
each query with the **single nearest city** by Euclidean distance in lat/lon
space. It is **nearest-neighbor, not point-in-polygon**.

- `rg.search(coords, mode=1)` — `mode=1` = single-process (no fork). We use it
  so the benchmark CLI stays fork-free + deterministic (no worker-count-dependent
  ordering; plays nice with Snakemake). `mode=2` (default) forks workers.
- The k-d tree loads once per process on the first call, then is reused.
- First call prints `Loading formatted geocoded file...` to stderr.

## Granularity ladder (every result dict has all six fields)

Verified live in `.venv` on 2026-06-19:

| field    | level                       | Boston           | Guadeloupe       |
| -------- | --------------------------- | ---------------- | ---------------- |
| `lat`/`lon` | coords of the matched **city** (not the query point) | 42.35843, -71.05977 | 16.26738, -61.58543 |
| `name`   | **city**                    | Boston           | Baie-Mahault     |
| `admin2` | county / district           | Suffolk County   | Guadeloupe       |
| `admin1` | state / province / region   | Massachusetts    | Guadeloupe       |
| `cc`     | **country**, ISO 3166-1 α-2 | US               | GP               |

So: **country (`cc`) → region (`admin1`) → county (`admin2`) → city (`name`)**.

### Caveats

- **Country (`cc`) is the robust level**; city/admin are best-effort. Accuracy
  is bounded by city coverage — dense regions resolve to a city a few km away,
  sparse/remote/offshore points can match a far-away city, degrading the finer
  levels.
- **`admin1`/`admin2` are free-text GeoNames names, not codes** (`"Massachusetts"`,
  `"Ile-de-France"`) — NOT standardized to ISO 3166-2.
- **No street/postcode/building level.** That needs a polygon/address dataset
  (Nominatim etc.). This library tops out at city.

## How geo_eval uses it

`_reverse_geocode_cc` pulls **only `cc`** and discards the rest — country is the
granularity that holds up under nearest-neighbor matching, and it's all
`continent_of` needs. `annotate_targets_geo` then appends two columns to an
already-written `targets.parquet` (decoupled from the runner, like
[airport_eval.py](../scripts/benchmark/v2/airport_eval.py)):

- `target_country`   = `cc`
- `target_continent` = `continent_of(cc)`

Both populated on **every** row regardless of prediction status (labels describe
the ground-truth target, not the prediction). Idempotent.

### Why coordinate-derived, not source-provided country

Deliberate ([geo_eval.py:9-16](../scripts/benchmark/v2/geo_eval.py#L9-L16)).
Coordinate-derived codes resolve **overseas territories to their physical
location** (Guadeloupe → `GP` → North America) rather than the administrative
parent (`FR` → Europe). The `country_code`-based continent split in
`plot_error_cdf.py` has to bbox-guard against exactly that mislabel
(`continents.continent_bbox_contains`); going straight from coords sidesteps it.

## Relation to GenericCSVSource optional geo columns

Distinct from the `vp_*/target_* continent/region/city` optional columns added
to `GenericCSVSource` (commit 6f487f5) — those are **source-provided metadata**
on `vp_configs`/`tg_configs.parquet`. geo_eval's `target_country/continent` are
**derived labels** on `targets.parquet`, intentionally so (the source-provided
country can carry the admin-parent mislabel this avoids).

If you ever want to *populate* those CSV `region`/`city` columns from coords,
`admin1` → region and `name` → city is the natural map — but they'd be
approximate (nearest-neighbor caveat), whereas `cc` → country is solid.
