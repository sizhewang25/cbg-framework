"""How close a vantage point was, for the TGs a method placed best. Ported from v4.

The error CDF says how far a prediction landed from the TG. It cannot say
*why* the easy ones were easy, and on these meshes the answer is VP proximity:
the TGs every method resolves to within a kilometre are the TGs a VP is
sitting on. This module measures that, per method, over a cohort the caller
selects, as two violins per method.

## Two distances, and the gap between them is the finding

Each TG is measured by many VPs. Two of them matter:

* the **geographically closest** VP (`geo_vp_dist_to_tg_km`) -- how much
  proximity the fleet actually offers for that TG;
* the **smallest-RTT** VP (`sping_vp_dist_to_tg_km`) -- the one latency
  nominates, and the coordinate S-P returns, so its distance *is* S-P's
  `pred_dist_to_tg_km` -- up to tie-breaking: where two VPs share the minimum
  RTT (0.1 ms resolution; 5 of 1,269 TGs on as01-03) the benchmark may have
  picked the other one. `n_sping_vp_ties` counts them per TG.

They are not the same VP. `geo <= sping` always, and on the pooled as01-03
meshes the median TG has a VP 13.8 km away while its smallest-RTT VP is
115.7 km away. The fleet is dense; latency routinely fails to say so. Drawing
one without the other invites reading a sparse-coverage story into what is an
RTT-inflation story, which is why `--geo/--sping` default to both on.

## The cohort is each method's own best TGs

`--cohort p5` takes the 5% of TGs *that method* placed most accurately, `p25`
the best 25%, `p95` the best 95%, `all` the whole population. The cohorts hold
different TGs for different methods, which is the point: the question is "what
did this method's easy cases have in common", not "how did every method do on
one fixed subset".

`p95` and `all` are not the same cohort. `all` is the evaluated population,
unanswered rows included; `p95` ranks on `pred_dist_to_tg_km` and so drops both
the unanswered rows and each method's own worst 5%. Read it against `all` to
see what that tail was carrying.

Selection runs over `status.solved_mask` rows only -- an unanswered TG has no
distance to rank on. A FALLBACK row still carries a coordinate (the S-P VP's),
so ranking without the mask fills a method's "most accurate" cohort with the
rows where it gave up. That is a different denominator from the outcome bars,
where a refusal counts as wrong, and the manifest records both counts.

## S-P's own row is circular

S-P predicts the smallest-RTT VP's coordinate, so ranking its TGs by
`pred_dist_to_tg_km` ranks them by the very quantity the orange violin draws.
Its row is still emitted -- it is the reference the others are read against --
and `circular` in the stats CSV marks it.

## A note on shape

At `p5` the distributions collapse to 2-11 distinct values with a single tie
holding 31-60% of the cohort, because ~20 IP replicas share a site and so share
its VP geometry exactly. A violin there reports density between values that
hold nothing. The extrema bars and the stats CSV carry the bound regardless,
and `max_tie_share` is in the CSV for every row so the smoothing can be checked.

Command: `plot-vp-proximity`. Writes `_cross/vp-proximity/<datasets>[@<arm>]/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from scripts.analysis.v5.modules import classify as C  # noqa: E402
from scripts.analysis.v5.modules import cross, edges
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.answer_space import BENCHMARK_TG_COLUMNS
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.modules.mapping import INK, INK_2, MUTED
from scripts.analysis.v5.modules.methods import method_label, method_term_table
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask

#: Which rung's `*_tgs.parquet` supplies `pred_dist_to_tg_km` and `status`.
#: Nothing here varies with it -- the distance is to the raw TG -- so this
#: selects a file, not an answer.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: `_cross/<KIND>/<datasets>[@<arm>]/`.
KIND = "vp-proximity"

#: The column cohorts are ranked on.
RANK_COLUMN = "pred_dist_to_tg_km"

#: The two measures, and the column each lives in.
GEO = "geo"
SPING = "sping"
MEASURE_COLUMNS = {GEO: "geo_vp_dist_to_tg_km", SPING: "sping_vp_dist_to_tg_km"}
MEASURE_LABELS = {
    GEO: "geographically closest VP",
    SPING: "smallest-RTT VP (what S-P returns)",
}

#: VP blue and TG orange (`mapping.LANDMASS_EDGE`, `mapping.TARGET_FILL`),
#: validated together in v4 (`validate_palette.js --mode light`: 5/5 pass,
#: worst CVD dE 24.7 protan). Method identity is carried by **row position**,
#: not hue, so these do not collide with `methods.LABEL_HUES`.
MEASURE_HUES = {GEO: "#2a78d6", SPING: "#eb6834"}

GRID = "#e6e4dd"

#: Cohort -> fraction of the population kept. `all` is the reference.
COHORTS: dict[str, float | None] = {
    "p5": 0.05,
    "p25": 0.25,
    "p95": 0.95,
    "all": None,
}

#: `max` is the *bound* the prose quotes, so it is reported beside these.
STAT_QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95)

CSV_NAME = "vp_proximity.{cohort}.csv"
PNG_NAME = "vp_proximity.{cohort}.png"
MANIFEST_NAME = "vp_proximity.{cohort}.manifest.json"

#: Canonical-CSV columns this module reads, in benchmark names.
_EDGE_COLUMNS = ("target_id", "vp_id", "rtt_ms", "vp_lat", "vp_lon", "target_lat", "target_lon")


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/vp-proximity/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def vp_distances(run: RunPaths, *, source_csv: Path | None = None) -> pd.DataFrame:
    """One row per TG: the two VP distances, and what backs them.

    Columns: `tg_id`, `geo_vp_dist_to_tg_km`, `geo_vp_rtt_ms`,
    `sping_vp_dist_to_tg_km`, `sping_vp_rtt_ms`, `n_sping_vp_ties` (VPs at the
    minimum RTT; > 1 means the smallest-RTT VP is a tie-break), `n_vp`.

    The RTT minimum per `(TG, VP)` pair is the one every LTD in the framework
    built its constraint from.
    """
    from scripts.libs.canonical.schema import load_canonical_csv

    path = edges.resolve_source_csv(run, source_csv)
    df = load_canonical_csv(path)[list(_EDGE_COLUMNS)].rename(columns=BENCHMARK_TG_COLUMNS)
    df = df.groupby(["tg_id", "vp_id"], as_index=False).agg(
        rtt_ms=("rtt_ms", "min"),
        vp_lat=("vp_lat", "first"),
        vp_lon=("vp_lon", "first"),
        tg_lat=("tg_lat", "first"),
        tg_lon=("tg_lon", "first"),
    )
    df["d_km"] = haversine_km(df.vp_lat.values, df.vp_lon.values, df.tg_lat.values, df.tg_lon.values)
    by_tg = df.groupby("tg_id", sort=False)
    geo = df.loc[by_tg["d_km"].idxmin()].set_index("tg_id")
    rtt = df.loc[by_tg["rtt_ms"].idxmin()].set_index("tg_id")
    out = pd.DataFrame(
        {
            MEASURE_COLUMNS[GEO]: geo["d_km"],
            "geo_vp_rtt_ms": geo["rtt_ms"],
            MEASURE_COLUMNS[SPING]: rtt["d_km"],
            "sping_vp_rtt_ms": rtt["rtt_ms"],
            "n_sping_vp_ties": (df["rtt_ms"] == by_tg["rtt_ms"].transform("min"))
            .groupby(df["tg_id"]).sum(),
            "n_vp": by_tg.size(),
        }
    ).rename_axis("tg_id").reset_index()
    # The definition guarantees it; a violation means the two VPs were picked
    # off different frames and every number downstream is untrustworthy.
    bad = out[out[MEASURE_COLUMNS[GEO]] > out[MEASURE_COLUMNS[SPING]] + 1e-6]
    if len(bad):
        raise ValueError(
            f"{run.run_id}: {len(bad)} TGs have geo > sping VP distance, which the "
            f"definition forbids (the closest VP cannot be further than any "
            f"particular VP). First: {bad.tg_id.iloc[0]!r}."
        )
    return out


def _tgs(run: RunPaths, method: str, nside: int, analysis_root: Path | None) -> pd.DataFrame:
    path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=method)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(path, columns=["tg_id", RANK_COLUMN, "status"])


def scored_methods(
    run: RunPaths, nside: int = SOURCE_NSIDE, *, analysis_root: Path | None = None
) -> list[str]:
    suffix = C.TGS_PARQUET.format(method="")
    d = run.classify_dir(nside, root=analysis_root)
    return sorted(p.name[: -len(suffix)] for p in d.glob("*" + suffix))


def load(
    runs: list[RunPaths],
    *,
    methods: list[str] | None = None,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
    source_csv: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """One row per `(run, TG, method)` with both VP distances repeated onto it,
    the shape both the cohort filter and the stats table consume, plus the
    provenance the manifest needs."""
    dist_frames, err_frames = [], []
    run_tgs: dict[str, set[str]] = {}
    available: dict[str, set[str]] = {}
    for run in runs:
        d = vp_distances(run, source_csv=(source_csv or {}).get(run.run_id))
        d["run_id"] = run.run_id
        dist_frames.append(d)
        run_tgs[run.run_id] = set(d.tg_id)
        available[run.run_id] = set(scored_methods(run, nside, analysis_root=analysis_root))
    cross.guard_disjoint_tgs(run_tgs, remedy="Pass one --run-id at a time.")

    chosen = sorted(set.intersection(*available.values())) if available else []
    if methods:
        missing = sorted(set(methods) - set(chosen))
        if missing:
            raise ValueError(
                f"{missing} are not scored in every run "
                f"({ {r: sorted(m) for r, m in available.items()} }). "
                f"Score them, or drop them from --method."
            )
        chosen = [m for m in chosen if m in methods]
    if not chosen:
        raise MissingArtifactError(
            f"no method is scored at {nside} in every run; run `classify` first."
        )

    for run in runs:
        for m in chosen:
            t = _tgs(run, m, nside, analysis_root)
            t["solved"] = solved_mask(t)
            t["method"] = m
            t["run_id"] = run.run_id
            err_frames.append(t)

    dist = pd.concat(dist_frames, ignore_index=True)
    err = pd.concat(err_frames, ignore_index=True)
    long = err.merge(dist, on=["run_id", "tg_id"], how="left", validate="m:1")
    if long[MEASURE_COLUMNS[GEO]].isna().any():
        n = int(long[MEASURE_COLUMNS[GEO]].isna().sum())
        raise ValueError(
            f"{n} scored TGs have no edge in the canonical CSV, so no VP "
            f"distance could be computed for them. The scored population and "
            f"the measurement table disagree; do not pool them."
        )
    meta = {
        "run_ids": [r.run_id for r in runs],
        "nside": int(nside),
        "methods": chosen,
        "n_tgs": int(len(dist)),
        "n_vp_per_tg_median": float(dist.n_vp.median()),
    }
    return long, meta


def cohort_k(long: pd.DataFrame, cohort: str) -> int | None:
    """How many rows per method `cohort` asks for. `None` for `all`.

    A fraction of the POOLED TG count, not of what a method answered, so every
    method is asked for the same number and the bounds are comparable.
    """
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; known: {sorted(COHORTS)}")
    frac = COHORTS[cohort]
    if frac is None:
        return None
    return int(round(frac * long.groupby("method").size().max()))


def cohort_frame(long: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """The rows `cohort` keeps, per method.

    `all` keeps every evaluated row, answered or not. A percentile cohort ranks
    on `pred_dist_to_tg_km` over solved rows only.

    A method that answered fewer TGs than k supplies everything it has and no
    more. That **short** cohort is not padded -- the rows it lacks are its
    worst, so borrowing any would flatter its bound. The manifest carries
    `cohort_k_requested` beside `n_cohort_per_method`, and the title gives the
    range rather than one number.
    """
    k = cohort_k(long, cohort)
    if k is None:
        return long.copy()
    out = (
        long[long.solved]
        .sort_values(RANK_COLUMN, kind="mergesort")
        .groupby("method", sort=False)
        .head(k)
    )
    return out.copy()


def _describe(values: pd.Series) -> dict:
    v = values.dropna()
    rounded = v.round(2)
    counts = rounded.value_counts()
    out = {
        "n": int(len(v)),
        "min_km": float(v.min()),
        "max_km": float(v.max()),
        "mean_km": float(v.mean()),
        "distinct_values": int(rounded.nunique()),
        "max_tie_share": float(counts.iloc[0] / len(v)) if len(v) else float("nan"),
    }
    for q in STAT_QUANTILES:
        out[f"p{int(q * 100)}_km"] = float(v.quantile(q))
    return out


def stats_table(
    long: pd.DataFrame, cohort_rows: pd.DataFrame, *, measures: list[str]
) -> pd.DataFrame:
    """One row per `(scope, method, measure)`. `scope="population"` carries the
    all-TG reference the cohort rows are read against."""
    rows = []
    dedup = long.drop_duplicates(subset=["run_id", "tg_id"])
    for measure in measures:
        col = MEASURE_COLUMNS[measure]
        pop = _describe(dedup[col])
        rows.append({"scope": "population", "method": "", "method_label": "all TGs",
                     "measure": measure, "circular": False, **pop})
        for method, g in cohort_rows.groupby("method", sort=False):
            d = _describe(g[col])
            rows.append({
                "scope": "cohort",
                "method": method,
                "method_label": method_label(method),
                "measure": measure,
                # S-P's pred_dist_to_tg_km IS the sping distance, so ranking
                # its cohort ranks it by the orange violin itself.
                "circular": method == SHORTEST_PING,
                **d,
                "lift_p50": (pop["p50_km"] / d["p50_km"]) if d["p50_km"] else float("nan"),
            })
    return pd.DataFrame(rows)


def order_methods(
    cohort_rows: pd.DataFrame, methods: list[str], measures: list[str]
) -> list[str]:
    """Rows sorted by their bound on the primary measure, tightest first.

    The bound is the claim, so the figure reads as a ladder: methods whose
    accurate TGs always had a near VP at the top, methods that stay accurate
    when latency points far away at the bottom. A method does not keep one row
    across cohorts; the orders genuinely differ.
    """
    primary = MEASURE_COLUMNS[SPING if SPING in measures else measures[0]]
    bound = cohort_rows.groupby("method")[primary].max().reindex(methods)
    return list(bound.sort_values(kind="mergesort").index)


def _violin(ax, values: np.ndarray, centre: float, color: str, half: float) -> float:
    """One violin, drawn in km onto a log axis. Returns the max.

    The density is estimated in **log space** and drawn at `10 ** xs`: a KDE
    on raw km would put one bandwidth across both the 0.5 km and the 700 km
    ends. It is cut at the data range, because the largest observation is the
    bound this figure reports.
    """
    lx = np.log10(np.asarray(values, dtype=float))
    lo, hi = float(lx.min()), float(lx.max())
    if np.ptp(lx) > 0 and len(lx) > 2:
        kde = gaussian_kde(lx)
        xs = np.linspace(lo, hi, 400)
        dens = kde(xs)
        dens = dens / dens.max() * half
        ax.fill_between(10 ** xs, centre - dens, centre + dens, facecolor=color,
                        alpha=0.55, edgecolor=color, lw=1.1, zorder=2)
    ax.plot([10 ** lo, 10 ** hi], [centre, centre], color=color, lw=0.9,
            alpha=0.75, zorder=1)
    for v in (lo, hi):
        ax.plot([10 ** v] * 2, [centre - 0.07, centre + 0.07], color=color,
                lw=1.6, zorder=3)
    med = 10 ** float(np.median(lx))
    ax.plot([med] * 2, [centre - half * 0.75, centre + half * 0.75],
            color="white", lw=2.6, zorder=4)
    ax.plot([med] * 2, [centre - half * 0.75, centre + half * 0.75],
            color=INK, lw=1.3, zorder=5)
    return 10 ** hi


def title_for(cohort: str, cohort_rows: pd.DataFrame, n_tgs: int) -> str:
    sizes = cohort_rows.groupby("method").size()
    lo = int(sizes.min()) if len(sizes) else 0
    hi = int(sizes.max()) if len(sizes) else 0
    if COHORTS[cohort] is None:
        return f"all {n_tgs:,} evaluated TGs"
    if lo == hi:
        return f"{cohort} cohort: the {hi:,} TGs each method placed most accurately"
    # A short cohort: one number here would be true of some rows and not others.
    return (f"{cohort} cohort: each method's {lo:,}-{hi:,} most accurately "
            f"placed TGs (fewer where it answered fewer)")


def plot(
    long: pd.DataFrame,
    cohort_rows: pd.DataFrame,
    *,
    cohort: str,
    measures: list[str],
    meta: dict,
    out_png: Path,
) -> Path:
    order = order_methods(cohort_rows, meta["methods"], measures)
    fig, ax = plt.subplots(figsize=(11.5, 1.0 + 0.82 * len(order)))
    paired = len(measures) == 2
    half = 0.185 if paired else 0.30
    offsets = {measures[0]: -0.205, measures[1]: 0.205} if paired else {measures[0]: 0.0}

    for i, method in enumerate(order):
        g = cohort_rows[cohort_rows.method == method]
        for measure in measures:
            vals = g[MEASURE_COLUMNS[measure]].dropna().values
            if not len(vals):
                continue
            hi = _violin(ax, vals, i + offsets[measure], MEASURE_HUES[measure], half)
            ax.annotate(
                f"{hi:,.0f}" if hi >= 10 else f"{hi:,.1f}",
                (hi, i + offsets[measure]), xytext=(7, 0), textcoords="offset points",
                va="center", fontsize=8.2, color=INK_2, zorder=6,
            )

    # Limits hug the data rather than snapping to whole decades; the right
    # margin is larger because the bound labels sit outside the violin.
    cols = [MEASURE_COLUMNS[m] for m in measures]
    data_lo = float(cohort_rows[cols].min().min())
    data_hi = float(cohort_rows[cols].max().max())
    span = np.log10(data_hi) - np.log10(data_lo)
    ax.set_xscale("log")
    ax.set_xlim(10 ** (np.log10(data_lo) - 0.03 * span), data_hi * 2.2)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=99))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([method_label(m) for m in order], fontsize=10, color=INK)
    ax.set_ylim(len(order) - 0.42, -0.62)
    ax.set_xlabel("VP distance to the TG (km, log scale)", fontsize=10, color=INK_2)
    ax.grid(axis="x", which="major", color=GRID, lw=0.9)
    ax.grid(axis="x", which="minor", color=GRID, lw=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="minor", length=2, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    legend = ax.legend(
        handles=[Patch(facecolor=MEASURE_HUES[m], alpha=0.55, edgecolor=MEASURE_HUES[m],
                       label=MEASURE_LABELS[m]) for m in measures],
        loc="upper center", bbox_to_anchor=(0.5, -0.155 - 0.02 * (6 - len(order))),
        ncol=len(measures), frameon=False, fontsize=9.5,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)
    # The term lookup, so the figure reads without the paper beside it.
    terms = "   ·   ".join(f"{t}: {name}" for t, name in method_term_table(order).items())
    ax.annotate(terms, xy=(0.5, -0.30 - 0.03 * (6 - len(order))), xycoords="axes fraction",
                ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.set_title(title_for(cohort, cohort_rows, meta["n_tgs"]), loc="left",
                 fontsize=11.5, color=INK, pad=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def _manifest(meta: dict, cohort: str, measures: list[str],
              cohort_rows: pd.DataFrame, stats: pd.DataFrame) -> str:
    n_per = cohort_rows.groupby("method").size().to_dict() if len(cohort_rows) else {}
    # From the pooled TG count, not from `cohort_rows` -- those are the rows k
    # already selected, so asking them what k was is circular.
    frac = COHORTS[cohort]
    k = None if frac is None else int(round(frac * meta["n_tgs"]))
    order = order_methods(cohort_rows, meta["methods"], measures)
    body = {
        "figure": "vp_proximity",
        "cohort": cohort,
        "cohort_fraction": frac,
        "cohort_k_requested": k,
        "n_cohort_short": {
            method_label(m): int(v) for m, v in n_per.items() if k is not None and int(v) < k
        },
        "measures": measures,
        "measure_columns": {m: MEASURE_COLUMNS[m] for m in measures},
        "rank_column": RANK_COLUMN,
        "datasets": cross.dataset_slug(meta["run_ids"]),
        "run_ids": meta["run_ids"],
        "source_nside": meta["nside"],
        "n_tgs_pooled": meta["n_tgs"],
        "n_vp_per_tg_median": meta["n_vp_per_tg_median"],
        "n_cohort_per_method": {method_label(m): int(v) for m, v in n_per.items()},
        "row_order": [method_label(m) for m in order],
        "method_terms": method_term_table(meta["methods"]),
        "policy": {
            "cohort_selection": (
                "Each method's own smallest pred_dist_to_tg_km, over solved rows "
                "only (status.solved_mask). An unanswered TG has no distance to "
                "rank on. This is NOT the outcome-bars denominator, where a "
                "refusal counts as wrong. The cohort size k (cohort_k_requested) "
                "is a fraction of the POOLED TG count, so every method is ASKED "
                "for the same number. A method that cannot supply k is listed "
                "in n_cohort_short and its bound rests on fewer rows."
            ),
            MEASURE_COLUMNS[GEO]: (
                "Great-circle distance to the geographically closest VP that "
                "measured this TG."
            ),
            MEASURE_COLUMNS[SPING]: (
                "Great-circle distance to the smallest-RTT VP, which is the "
                "coordinate S-P returns; geo <= sping always."
            ),
            "shortest_ping_is_circular": (
                "S-P's pred_dist_to_tg_km equals its sping VP distance, so its "
                "cohort is ranked by the quantity its own orange violin draws. "
                "Kept as the reference row, marked `circular` in the CSV."
            ),
            "kde": (
                "Fitted in log10 space and cut at the data range so the drawn "
                "right edge is the true maximum. Interior density is smoothed; "
                "see distinct_values and max_tie_share per row."
            ),
            "max_is_the_claim": (
                "max_km is the bound: no TG in this cohort lay further than this "
                "from that VP."
            ),
        },
        "bounds": {
            f"{row.measure}:{row.method_label}": row.max_km
            for row in stats[stats.scope == "cohort"].itertuples()
        },
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    cohorts: list[str] | None = None,
    methods: list[str] | None = None,
    geo: bool = True,
    sping: bool = True,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
    source_csv: dict[str, Path] | None = None,
) -> list[Path]:
    """PNG, stats CSV and manifest per cohort. Returns the PNGs."""
    measures = [m for m, on in ((GEO, geo), (SPING, sping)) if on]
    if not measures:
        raise ValueError("nothing to draw: --no-geo and --no-sping cannot both be set.")
    cohorts = list(dict.fromkeys(cohorts or ["p25"]))
    unknown = [c for c in cohorts if c not in COHORTS]
    if unknown:
        # Before the CSVs are read: the distances take seconds per mesh.
        raise ValueError(f"unknown cohort {unknown}; known: {sorted(COHORTS)}")
    nside = G.validate_nside(nside)
    long, meta = load(runs, methods=methods, nside=nside, analysis_root=analysis_root,
                      source_csv=source_csv)
    out_dir = output_dir(meta["run_ids"], analysis_root=analysis_root)
    written = []
    for cohort in cohorts:
        rows = cohort_frame(long, cohort)
        stats = stats_table(long, rows, measures=measures)
        stats.to_csv(out_dir / CSV_NAME.format(cohort=cohort), index=False)
        (out_dir / MANIFEST_NAME.format(cohort=cohort)).write_text(
            _manifest(meta, cohort, measures, rows, stats)
        )
        written.append(
            plot(long, rows, cohort=cohort, measures=measures, meta=meta,
                 out_png=out_dir / PNG_NAME.format(cohort=cohort))
        )
    return written
