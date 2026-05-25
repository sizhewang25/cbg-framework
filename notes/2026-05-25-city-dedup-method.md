# Probe / anchor city dedup: 0.1° grid, not geocoder names

**Decision (2026-05-25):** use a 0.1° lat/lon grid (~11 km at the equator) as the
canonical "city" primitive for VP-corpus design, dedup, and geographic-diversity
analysis. Do **not** dedup by geocoded city names from Nominatim or
`reverse_geocoder`. Keep the geocoders strictly for human-readable labels.

This was confirmed after running both backends against the 723-anchor corpus
and finding that they answer different questions, each correctly:

- **Nominatim** returns the *containing administrative city* via polygon
  containment. Anchor in Borgfelde resolves to Hamburg.
- **`reverse_geocoder`** returns the *nearest GeoNames city center* via kdtree.
  Same anchor resolves to Borgfelde itself (the nearest GeoNames cities1000
  point).

## Agreement numbers (723 anchors, both backends)

| Level | Agreement |
|---|---|
| Country code | **98.3%** (711/723) |
| Exact city string | 36.2% |
| ASCII-folded (Roma↔Rome) | 40.9% |
| Plus offline-name-in-Nominatim-address (Berlin↔Mitte) | 49.2% |
| Truly different city | 48.5% — almost all suburb-vs-metro |

The ~50% city disagreement is **structural** (different abstraction layers),
not noise. Using either as a dedup key would over-merge (Nominatim folds
Amstelveen into Amsterdam) or under-merge (`reverse_geocoder` keeps Mitte
separate from Berlin). Country code is reliable; city names are not.

## Why the 0.1° grid is the right primitive here

- **Naming-free.** No ambiguity from local vs anglicized names, language
  differences, suburb vs metro, or admin-polygon shape.
- **Tunable.** Grid size controls the granularity directly — 0.1° (~11 km) for
  metro-level dedup, 0.01° (~1 km) for building-level if we ever need it.
- **No external deps and instant.** No Nominatim rate limit, no GeoNames file,
  no kdtree load — just `(round(lat*10)/10, round(lon*10)/10)`.
- **Matches our actual use case.** For CBG VP-corpus design we want to count
  *independent measurement points*, and two probes within 11 km contribute
  essentially redundant RTT — regardless of whether they're in
  "Amstelveen" or "Amsterdam" administratively.

## Where the geocoders still earn their keep

Labels only. Concretely:

- `anchor_city.json` / `probe_city.json` (Nominatim) — display strings, full
  address hierarchy when we want to talk about a specific anchor in a paper or
  notebook.
- `anchor_city_offline.json` (`reverse_geocoder`) — same display purpose, sub-
  second to regenerate.
- The notebook's per-ASN city-diversity rerank in
  [scripts/processing/ripe_atlas/visualize_probes_anchors.ipynb](../scripts/processing/ripe_atlas/visualize_probes_anchors.ipynb)
  uses the grid for the unique-cities count and the geocoder outputs not at
  all — which is the right separation.

## Implications

- Any future analysis that wants "city-level" probe counts should use the
  0.1° grid, not the geocoder JSON.
- The two `*_city.json` files in `datasets/ripe_atlas/` are downstream-of-decision
  artifacts: nothing in the pipeline depends on them, they're for human reading.
- If we later need polygon-based metro grouping (e.g., to compare against
  Octant's per-metro VP selection), we should reach for Natural Earth admin1
  polygons via `shapely`, not for either of the city geocoders.
