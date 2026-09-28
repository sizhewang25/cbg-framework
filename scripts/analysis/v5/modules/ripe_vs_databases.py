"""CBG methods against the commercial geolocation databases, on one CDF.

The IMC 2023 paper's Figure 7 asks the question this figure asks: a CBG curve
is only worth drawing if the reader can see what it is competing with. A
measurement-based method that lands inside 40 km half the time sounds good
until an IP-to-location lookup -- free, instant, no vantage points -- does
better on the same targets.

Two databases ride along, both shipped in `datasets/static_datasets/` and both
keyed by anchor IPv4, which is exactly this package's `tg_id`:

  * `ip_info_geo_anchors.json`    -- IPinfo, a `loc: "lat,lon"` string per IP
  * `maxmind_free_geo_anchors.json` -- MaxMind GeoLite2 free, a `[lat, lon]` pair

## They are series, not methods

A database has no combo id, no fold, and nothing was fitted: the same lookup
answers every target in every fold. So it gets no position in
`methods.TERM_ORDER` and no hue from the validated six-term palette -- widening
that palette would re-order the legend of every other figure in the package for
the sake of this one. The terms, hues and dashes live here instead, and
`figure_error_cdf.plot_cdf` takes them through its `order` / `style_fn` hooks.
They are drawn dashed and dash-dot for the same reason the S-P baseline is:
they are the reference, not the result.

## The same distance, the same population

The CBG curves are `figure_error_cdf.load_errors` unchanged -- the identical
arrays behind `error_cdf.png`, so a number here and a number there cannot
disagree. The database curves are computed the same way the benchmark computes
`pred_dist_to_tg_km`: great-circle from the looked-up coordinate to the TG's
own `tg_lat`/`tg_lon`, read out of the scored parquet rather than re-derived
from the corpora, so both sides of the comparison use one ground truth.

A TG the database has no record for is **unanswered**, not an error of zero or
infinity -- the same status a CBG method's FALLBACK gets. Under the default
`exclude` policy it drops out and the database curve rests on its own
population; under `sentinel` it re-enters at 10,000 km, and the height of the
curve at the sentinel is the database's coverage. `n_solved` in the CSV is the
covered count, so a curve drawn on partial coverage is never silently read as a
full one. On the RIPE anchor roster both databases cover every TG, so today the
two policies differ only in the CBG methods' refusals.

Command: `plot-ripe-vs-databases`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import figure_error_cdf as E
from scripts.analysis.v5.modules.geodesy import elementwise_km
from scripts.analysis.v5.modules.methods import method_colors, method_label, method_order
from scripts.analysis.v5.modules.paths import (
    MissingArtifactError,
    RIPE_VS_DATABASES_KIND,
    RunPaths,
    grid_slug,
)
from scripts.analysis.v5.modules.status import SHORTEST_PING

#: Where the shipped lookups live. Both are anchor-keyed snapshots taken with
#: the reproducibility dataset, not live API calls -- so the figure is
#: reproducible and costs nothing to redraw.
DEFAULT_DB_DIR = Path("datasets/static_datasets")

IPINFO = "ipinfo"
MAXMIND = "maxmind_free"

#: Series id -> (filename, term). Ordered as they are drawn.
DATABASES: dict[str, tuple[str, str]] = {
    MAXMIND: ("maxmind_free_geo_anchors.json", "MM"),
    IPINFO: ("ip_info_geo_anchors.json", "IPI"),
}

#: Term -> full name, the local twin of `methods.METHOD_TERMS`. Kept here
#: because `test_methods` pins that table to the hue table, and these two have
#: no hue in the validated palette by design (see the module docstring).
DB_TERMS: dict[str, str] = {
    "MM": "MaxMind GeoLite2 (free)",
    "IPI": "IPinfo",
}

#: Term -> hue. Chosen away from the six method hues: MM's burnt orange against
#: VAN's #eda100 yellow, IPI's orchid against SPO's #e34948 red. The dash
#: patterns below carry the distinction a second time, so the pair is
#: separable in greyscale and under colour-vision simulation.
DB_HUES: dict[str, str] = {
    "MM": "#b5651d",
    "IPI": "#b455a8",
}

#: Term -> linestyle. Mirrors the paper's figure: MaxMind dashed, IPinfo
#: dash-dot, and neither solid, because solid is a measured result here.
DB_DASHES: dict[str, tuple] = {
    "MM": (0, (5, 2)),
    "IPI": (0, (6, 2, 1, 2)),
}

STEM = "ripe_vs_databases"

#: Read from the scored parquet: the TG roster plus its ground truth. Any
#: method's file carries the same three columns for the same TGs.
ROSTER_COLUMNS: tuple[str, ...] = ("tg_id", "tg_lat", "tg_lon")


def artifact_names(policy: str = E.EXCLUDE) -> tuple[str, str, str]:
    """`(png, csv, manifest)`. `.sentinel.` infix under that policy, as in
    `figure_error_cdf` -- the two populations must not overwrite each other."""
    stem = f"{STEM}.{E.SENTINEL}" if policy == E.SENTINEL else STEM
    return (f"{stem}.png", f"{stem}.csv", f"{stem}.manifest.json")


# ---- the databases -----------------------------------------------------------


def _parse_ipinfo(payload: dict) -> dict[str, tuple[float, float]]:
    """`{ip: (lat, lon)}` from IPinfo's `loc: "lat,lon"` string.

    Records without a parseable `loc` are dropped rather than defaulted: a
    database that has no answer for an IP is uncovered, and inventing (0, 0)
    for it would put the Gulf of Guinea in the error distribution.
    """
    out: dict[str, tuple[float, float]] = {}
    for ip, rec in payload.items():
        loc = (rec or {}).get("loc")
        if not loc or "," not in loc:
            continue
        lat_s, _, lon_s = loc.partition(",")
        try:
            out[ip] = (float(lat_s), float(lon_s))
        except ValueError:
            continue
    return out


def _parse_maxmind(payload: dict) -> dict[str, tuple[float, float]]:
    """`{ip: (lat, lon)}` from MaxMind's `[lat, lon]` pair."""
    out: dict[str, tuple[float, float]] = {}
    for ip, rec in payload.items():
        if not rec or len(rec) < 2:
            continue
        try:
            out[ip] = (float(rec[0]), float(rec[1]))
        except (TypeError, ValueError):
            continue
    return out


_PARSERS = {IPINFO: _parse_ipinfo, MAXMIND: _parse_maxmind}


def load_database(series: str, db_dir: Path = DEFAULT_DB_DIR) -> dict[str, tuple[float, float]]:
    """One database as `{ip: (lat, lon)}`."""
    filename, _ = DATABASES[series]
    path = Path(db_dir) / filename
    if not path.exists():
        raise MissingArtifactError(
            f"{path} missing; {series} is shipped in datasets/static_datasets/"
        )
    with path.open() as fh:
        return _PARSERS[series](json.load(fh))


# ---- the roster --------------------------------------------------------------


def load_roster(
    run: RunPaths,
    nside: int = E.SOURCE_NSIDE,
    *,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """The run's TGs and their ground truth, one row each.

    Read from a scored parquet rather than from the source CSV so the
    coordinate a database is measured against is byte-identical to the one the
    CBG methods were measured against. Folds are disjoint, so a TG appears once
    per method; `drop_duplicates` guards the case where that stops being true.
    """
    cls_dir = run.classify_dir(nside, root=analysis_root)
    scored = E.scored_methods(cls_dir)
    if not scored:
        raise MissingArtifactError(
            f"{cls_dir} holds no *_tgs.parquet; run `classify --run-id {run.run_id}` first"
        )
    from scripts.analysis.v5.modules import classify as C

    path = cls_dir / C.TGS_PARQUET.format(method=scored[0])
    roster = pd.read_parquet(path, columns=list(ROSTER_COLUMNS))
    return roster.drop_duplicates("tg_id").reset_index(drop=True)


def database_errors(
    roster: pd.DataFrame,
    coords: dict[str, tuple[float, float]],
) -> dict:
    """One database's entry, in `figure_error_cdf.load_errors`' shape.

    Great-circle from the looked-up coordinate to the TG's own, over the TGs
    the database covers. Uncovered TGs count as `n_failed` -- unanswered, the
    same bucket a CBG refusal lands in -- so the sentinel policy draws the
    database's coverage and `exclude` rests it on its own population.
    """
    hit = roster[roster["tg_id"].isin(coords)]
    if len(hit):
        db_lat = np.array([coords[i][0] for i in hit["tg_id"]], dtype=float)
        db_lon = np.array([coords[i][1] for i in hit["tg_id"]], dtype=float)
        errors = np.asarray(
            elementwise_km(
                hit["tg_lat"].to_numpy(dtype=float),
                hit["tg_lon"].to_numpy(dtype=float),
                db_lat,
                db_lon,
            ),
            dtype=float,
        )
    else:
        errors = np.empty(0, dtype=float)
    finite = np.isfinite(errors)
    return {
        "errors": errors[finite],
        "tg_ids": set(roster["tg_id"]),
        "n_tgs": int(len(roster)),
        "n_solved": int(len(hit)),
        "n_failed": int(len(roster) - len(hit)),
        "n_no_distance": int((~finite).sum()),
    }


def load_databases(
    roster: pd.DataFrame,
    *,
    db_dir: Path = DEFAULT_DB_DIR,
    series: list[str] | None = None,
) -> dict[str, dict]:
    """`{series_id: entry}` for the databases, against one roster."""
    chosen = list(series) if series else list(DATABASES)
    return {s: database_errors(roster, load_database(s, db_dir)) for s in chosen}


# ---- labels, order, style ----------------------------------------------------


def series_label(series: str) -> str:
    """Term for a series id: the databases' own, else the package's."""
    if series in DATABASES:
        return DATABASES[series][1]
    return method_label(series)


def series_order(loaded: dict[str, dict]) -> list[str]:
    """Methods in `TERM_ORDER`, then the databases, in `DATABASES` order.

    Databases last because they are the reference the CBG curves are read
    against -- the same reason S-P is drawn last within the methods.
    """
    dbs = [s for s in DATABASES if s in loaded]
    return method_order([s for s in loaded if s not in DATABASES]) + dbs


def series_style(series: str, colors: dict[str, str]) -> dict:
    """Line kwargs. Databases dashed in their own hue; methods as usual."""
    if series in DATABASES:
        term = DATABASES[series][1]
        return {
            "color": DB_HUES[term],
            "linestyle": DB_DASHES[term],
            "linewidth": 1.3,
            "zorder": 3,
        }
    return E._curve_style(series, colors)


def term_table(order: list[str]) -> dict[str, str]:
    """`term -> full name` for everything on the panel, for the manifest."""
    from scripts.analysis.v5.modules.methods import METHOD_TERMS

    terms = {series_label(s) for s in order}
    merged = {**METHOD_TERMS, **DB_TERMS}
    return {t: merged[t] for t in merged if t in terms}


# ---- artifacts ---------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "unanswered_policy", "sentinel_km",
    "method", "method_label", "is_database", "is_baseline",
    *E.COUNT_KEYS, "n_plotted", "n_censored", *[E.pcol(p) for p in E.PERCENTILES],
)


def _manifest(
    table: pd.DataFrame,
    loaded: dict[str, dict],
    order: list[str],
    *,
    run_id: str,
    nside: int,
    png_name: str,
    csv_name: str,
    min_x_km: float,
    max_x_km: float,
    policy: str,
    sentinel_km: float,
    db_dir: Path,
) -> str:
    censored = policy == E.SENTINEL
    body: dict = {
        "figure": png_name,
        "csv": csv_name,
        "run": run_id,
        "dataset": cross.dataset_slug([run_id]),
        "series": order,
        "databases": {
            s: {
                "term": DATABASES[s][1],
                "name": DB_TERMS[DATABASES[s][1]],
                "source": str(Path(db_dir) / DATABASES[s][0]),
                "n_covered": int(loaded[s]["n_solved"]),
                "n_uncovered": int(loaded[s]["n_failed"]),
            }
            for s in order
            if s in DATABASES
        },
        "terms": term_table(order),
        "baseline": SHORTEST_PING,
        "dist_column": E.DIST_COLUMN,
        "percentiles": list(E.PERCENTILES),
        "source_rung": {
            "nside": int(nside),
            "slug": grid_slug(nside),
            "note": (
                "which rung's *_tgs.parquet supplied the CBG distances and the TG "
                "ground truth. It does not affect the output: pred_dist_to_tg_km is "
                "prediction-to-TG and identical at every rung."
            ),
        },
        "method_rows": (
            "figure_error_cdf.load_errors, unchanged -- the same arrays behind "
            "error_cdf.png, so the two figures cannot disagree."
        ),
        "database_rows": (
            "great-circle from the database's looked-up coordinate to the TG's own "
            "tg_lat/tg_lon, read from the same parquet as the CBG rows. A TG with no "
            "record is unanswered (n_failed), never an error of zero."
        ),
        "unanswered": {
            "policy": policy,
            "sentinel_km": float(sentinel_km) if censored else None,
            "n_censored": {s: int(e.get("n_censored", 0)) for s, e in loaded.items()},
            "note": (
                "unanswered rows -- a method's refusals and a database's uncovered "
                "TGs alike -- are drawn at sentinel_km, so each curve's height there "
                "is its answer rate."
                if censored
                else "unanswered rows are dropped; each curve rests on its own "
                "population. n_solved is a database's covered count."
            ),
        },
        "palette": {
            "methods": method_colors([s for s in order if s not in DATABASES]),
            "databases": {s: DB_HUES[DATABASES[s][1]] for s in order if s in DATABASES},
            "note": (
                "database hues and dashes are local to this figure: a database has no "
                "combo id and no position in methods.TERM_ORDER, and widening the "
                "validated six-term palette would re-order every other legend."
            ),
        },
        "x_axis": {"min_km": float(min_x_km), "max_km": float(max_x_km), "scale": "log"},
        "clamped_below_min": E._clamped(loaded, min_x_km),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_run(
    run: RunPaths,
    *,
    nside: int = E.SOURCE_NSIDE,
    methods: list[str] | None = None,
    databases: list[str] | None = None,
    db_dir: Path = DEFAULT_DB_DIR,
    analysis_root: Path | None = None,
    min_x_km: float = E.X_MIN_KM,
    max_x_km: float | None = None,
    unanswered: str = E.EXCLUDE,
    sentinel_km: float = E.SENTINEL_KM,
) -> Path:
    """One run's methods-vs-databases CDF, into `ripe-vs-databases/`.

    Its own kind directory rather than `classify/`: this figure reads the
    scored parquets but adds a source outside the benchmark entirely, and a
    reader picking through `classify/` should not find a curve that no run
    produced.
    """
    loaded = dict(E.load_errors(run, nside, methods=methods, analysis_root=analysis_root))
    roster = load_roster(run, nside, analysis_root=analysis_root)
    loaded.update(load_databases(roster, db_dir=db_dir, series=databases))

    policy = unanswered
    png_name, csv_name, manifest_name = artifact_names(policy)
    drawn = E.censor(loaded, policy=policy, sentinel_km=sentinel_km)
    order = series_order(drawn)

    table = E.percentile_table(drawn)
    # `percentile_table` labels through `methods.method_label`, which does not
    # know the databases. Re-label here rather than registering them globally.
    table["method_label"] = table["method"].map(series_label)
    table["is_database"] = table["method"].isin(DATABASES)
    table.insert(0, "run_id", run.run_id)
    table.insert(1, "dataset", cross.dataset_slug([run.run_id]))
    table["unanswered_policy"] = policy
    table["sentinel_km"] = float(sentinel_km) if policy == E.SENTINEL else np.nan

    out_dir = run.analysis_dir(RIPE_VS_DATABASES_KIND, root=analysis_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    table[[c for c in CSV_COLUMNS if c in table.columns]].to_csv(
        out_dir / csv_name, index=False
    )

    resolved_max = E.resolve_x_max(policy, max_x_km, sentinel_km)
    colors = method_colors([s for s in order if s not in DATABASES])
    png = E.plot_cdf(
        drawn,
        table,
        out_dir / png_name,
        title="CBG against the geolocation databases",
        subtitle=(
            f"{run.run_id} · n={E._n_tgs(drawn):,} TGs · "
            f"{E._subtitle_tail(policy, sentinel_km)}"
        ),
        min_x_km=min_x_km,
        max_x_km=resolved_max,
        sentinel_km=sentinel_km if policy == E.SENTINEL else None,
        order=order,
        style_fn=lambda s: series_style(s, colors),
        label_fn=series_label,
    )
    (out_dir / manifest_name).write_text(
        _manifest(
            table, drawn, order,
            run_id=run.run_id, nside=nside, png_name=png_name, csv_name=csv_name,
            min_x_km=min_x_km, max_x_km=resolved_max,
            policy=policy, sentinel_km=sentinel_km, db_dir=db_dir,
        )
    )
    return png
