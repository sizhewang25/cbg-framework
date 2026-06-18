"""Geographic (continent/country) postprocessing over benchmark `targets.parquet`.

Sibling of [airport_eval.py](airport_eval.py) and built on the same principle:
**decoupled from the runner**. It annotates already-written `targets.parquet`
files in place with the *target's* continent and country, so downstream analysis
can group eval metrics by geography (continent/country subsets) without an
external join — and it backfills existing outputs without re-running CBG.

The labels are derived from the **ground-truth coordinates** (`target_lat`,
`target_lon`) via an offline `reverse_geocoder` kdtree lookup, not from any
source-provided country field. Coordinate-derived country codes resolve overseas
territories to their physical location (Guadeloupe → ``GP`` → North America,
not its administrative parent ``FR`` → Europe), which is exactly the mislabel
the `country_code`-based continent split in `plot_error_cdf.py` has to
bbox-guard against (see `continents.continent_bbox_contains`). Going straight
from coordinates sidesteps it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from scripts.benchmark.v2.schema import SUMMARY_STATS
from scripts.processing.ripe_atlas.continents import continent_of

# Columns appended to each targets.parquet (not part of TARGETS_SCHEMA — the
# runner stays untouched; these live only in postprocessed files).
GEO_COLUMNS = (
    "target_continent",  # canonical continent name (continents.continent_of)
    "target_country",    # ISO 3166-1 alpha-2 code from reverse geocoding
)

# Rows that count as a real prediction for error_km stats.
_SCORED_STATUSES = ("SUCCESS", "FALLBACK")

_STAT_Q = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}


def _reverse_geocode_cc(lats, lons) -> list[str]:
    """Offline ``(lat, lon)`` → ISO alpha-2 country code via a kdtree.

    Uses `reverse_geocoder` (GeoNames cities1000, mode=1 single-process so the
    benchmark CLI stays fork-free and deterministic). The kdtree is loaded once
    per process on the first call. Coordinates are always present on target rows
    (`TARGETS_SCHEMA` marks `target_lat/lon` non-nullable), so no NaN handling
    is needed here.
    """
    import reverse_geocoder as rg

    coords = [(float(a), float(b)) for a, b in zip(lats, lons)]
    if not coords:
        return []
    results = rg.search(coords, mode=1)
    return [r["cc"] for r in results]


def annotate_targets_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Append `target_continent` + `target_country`, derived from target coords.

    Both columns are populated on every row regardless of prediction status
    (the labels describe the ground-truth target, not the prediction).
    Idempotent — re-running overwrites the columns in place.
    """
    out = df.copy()
    cc = _reverse_geocode_cc(out["target_lat"].to_numpy(), out["target_lon"].to_numpy())
    out["target_country"] = cc
    out["target_continent"] = [continent_of(c) for c in cc]
    return out


def _error_stat_block(series: pd.Series, prefix: str = "error_km") -> dict:
    """p5..p95/mean/std for an error series, NaN-filled when empty."""
    s = series.dropna()
    out: dict = {}
    for stat in SUMMARY_STATS:
        if len(s) == 0:
            out[f"{prefix}_{stat}"] = float("nan")
        elif stat == "mean":
            out[f"{prefix}_mean"] = float(s.mean())
        elif stat == "std":
            out[f"{prefix}_std"] = float(s.std())
        else:
            out[f"{prefix}_{stat}"] = float(s.quantile(_STAT_Q[stat]))
    return out


def summarize_geo(df: pd.DataFrame) -> list[dict]:
    """Per-geo-bucket eval summary — one row per ``(group_level, group_value)``.

    Emits an overall ``(all, all)`` row plus a `continent` breakdown and a
    `country` breakdown. `error_km` stats are over SUCCESS+FALLBACK rows only;
    the `n_*` counts cover every target in the bucket so the success rate per
    subset stays visible. Empty input is safe.
    """
    rows: list[dict] = []

    def _emit(level: str, value: str, sub: pd.DataFrame) -> None:
        scored = sub[sub["status"].isin(_SCORED_STATUSES)]
        row = {
            "group_level": level,
            "group_value": value,
            "n_targets": len(sub),
            "n_success": int((sub["status"] == "SUCCESS").sum()),
            "n_fallback": int((sub["status"] == "FALLBACK").sum()),
            "n_error": int((sub["status"] == "ERROR").sum()),
        }
        row.update(_error_stat_block(scored["error_km"]))
        rows.append(row)

    _emit("all", "all", df)
    if "target_continent" in df.columns:
        for cont, sub in df.groupby("target_continent", dropna=False):
            _emit("continent", str(cont), sub)
    if "target_country" in df.columns:
        for cc, sub in df.groupby("target_country", dropna=False):
            _emit("country", str(cc), sub)
    return rows


def process_parquet(path: Path) -> list[dict]:
    """Annotate a single targets.parquet in place (atomic) and return its
    per-bucket summary rows."""
    path = Path(path)
    df = pd.read_parquet(path)
    annotated = annotate_targets_geo(df)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet")
    os.close(fd)
    try:
        annotated.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    return summarize_geo(annotated)
