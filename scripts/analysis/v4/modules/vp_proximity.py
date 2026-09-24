"""How close a vantage point was, for the targets a method placed best.

The error CDF says how far a prediction landed from the truth. It cannot say
*why* the easy ones were easy, and on these meshes the answer is vantage point
proximity: the targets every method resolves to within a kilometre are the
targets a VP is sitting on. This module measures that, per method, over a
cohort the caller selects.

## Two distances, and the gap between them is the finding

Each target is measured by many VPs. Two of them matter:

* the **geographically closest** VP -- how much proximity the fleet actually
  offers for that target;
* the **smallest-RTT** VP -- the one latency nominates, and the coordinate
  Shortest-Ping returns, so its distance *is* Shortest-Ping's error.

They are not the same VP. `d_geo <= d_sping` always, and on the pooled operator
meshes the median target has a VP 13.8 km away while its smallest-RTT VP is
115.7 km away. The fleet is dense; latency routinely fails to say so. Drawing
one without the other invites reading a sparse-coverage story into what is an
RTT-inflation story, which is why `--geo/--sping` default to both on.

## The cohort is each method's own best targets

`--cohort p5` takes the 5% of targets *that method* placed most accurately,
`p25` the best 25%, `p95` the best 95%, `all` the whole population. The cohorts
therefore hold different targets for different methods, which is the point: the
question is "what did this method's easy cases have in common", not "how did
every method do on one fixed subset".

`p95` and `all` are not the same cohort, and the difference is the reason both
exist. `all` is the evaluated population, unanswered rows included; `p95` ranks
on `error_km` and so drops both the unanswered rows and each method's own worst
5%. It is the near-whole population with the tail that would set the bound
trimmed off -- read it against `all` to see what that tail was carrying.

Selection runs over `classify.solved_mask` rows only -- an unanswered target
has no error to rank on. That is a different denominator from the outcome
bars, where a refusal counts as wrong, and the manifest records both counts so
the difference is visible rather than inferred.

## Shortest-Ping's own row is circular

Shortest-Ping predicts the smallest-RTT VP's coordinate, so its error *is*
`d_sping`, and ranking its targets by error is ranking them by the very
quantity the orange violin draws. Its row is still emitted -- it is the
reference the others are read against -- and `circular` in the stats CSV marks
it so no reader mistakes it for a result.

## A note on shape

At `p25` these distributions carry 9-20 distinct values per method and a KDE
summarises them fairly. At `p5` they collapse to 2-11 distinct values with a
single tie holding 31-60% of the cohort, because ~20 IP replicas share a site
and therefore share its VP geometry exactly. A violin drawn there reports
density between values that hold nothing. The figure still draws it, because
the extrema bars and the stats CSV carry the bound regardless, but
`max_tie_share` is in the CSV for every row so the smoothing can be checked
rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from scripts.analysis.v4.modules import classify, edges
from scripts.analysis.v4.modules.cross import (
    arm,
    dataset_slug,
    guard_disjoint_targets,
)
from scripts.analysis.v4.modules.mapping import INK, INK_2, MUTED
from scripts.analysis.v4.modules.methods import method_label
from scripts.analysis.v4.modules.paths import (
    CLS_ACCURACY_KIND,
    DEFAULT_ANALYSIS_ROOT,
    MissingArtifactError,
    RunPaths,
    grid_slug,
    resolve_run,
)
from scripts.framework.geometry import EARTH_RADIUS_KM

#: The rung whose `*_cells.parquet` supplies `error_km` and `status`. Like the
#: error CDF, nothing here varies with it -- `error_km` is
#: prediction-to-truth -- so this selects which file to open, not which answer
#: to get.
SOURCE_NSIDE = 128

#: Where this module writes, under `_cross/<datasets>[@<arm>]/`.
KIND = "vp_proximity"

#: The two measures, and the column each lives in.
GEO = "geo"
SPING = "sping"
MEASURE_COLUMNS = {GEO: "d_geo_km", SPING: "d_sping_km"}
MEASURE_LABELS = {
    GEO: "geographically closest VP",
    SPING: "smallest-RTT VP (what Shortest-Ping returns)",
}

#: Hues. `#2a78d6` is the package's VP blue (`mapping.VP_EDGE`, itself v3's
#: `map_bipartite._VP_COLOR`) and `#eb6834` its target orange; the pair is
#: already validated together for the answer-space map and re-checked for this
#: figure (`validate_palette.js --mode light`: 5/5 pass, worst CVD dE 24.7
#: protan, normal-vision floor 33.6).
#:
#: Method identity is carried by **row position** here, not by hue, so these
#: two do not collide with `methods.LABEL_HUES` inside this figure. That is
#: also why the figure cannot simply adopt those hues: it draws two series per
#: method, and a method's own colour has nothing to say about which of its two
#: violins is which.
MEASURE_HUES = {GEO: "#2a78d6", SPING: "#eb6834"}

GRID = "#e6e4dd"

#: Cohort names the CLI accepts, mapped to the fraction of the population they
#: keep. `all` is the whole evaluated set and is the reference every cohort is
#: read against.
COHORTS: dict[str, float | None] = {
    "p5": 0.05,
    "p25": 0.25,
    "p95": 0.95,
    "all": None,
}

#: Percentiles the stats CSV reports. `max` is the one the prose quotes -- it
#: is the *bound*, the claim that no target in this cohort was further than
#: this from a VP -- so it is reported beside the quantiles rather than left to
#: be read off a whisker.
STAT_QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95)

CSV_NAME = "vp_proximity.{cohort}.csv"
PNG_NAME = "vp_proximity.{cohort}.png"
MANIFEST_NAME = "vp_proximity.{cohort}.manifest.json"


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/<datasets>[@<arm>]/vp_proximity/`, created.

    Note the ordering differs from `cross.cross_dir`, which is
    `_cross/<kind>/<datasets>/`. This one groups by dataset set first so that
    everything derived from one combination of runs sits together.
    """
    name = dataset_slug(run_ids)
    shared = arm(run_ids)
    if shared is not None:
        name = f"{name}@{shared}"
    out = (analysis_root or DEFAULT_ANALYSIS_ROOT) / "_cross" / name / KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance, vectorised over numpy arrays.

    Written here rather than imported: `framework.geometry.haversine` takes two
    coordinate tuples and this runs over ~170k edge rows per mesh. The radius
    constant is shared, so the two cannot drift on the only value that would
    change an answer.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def vp_distances(run: RunPaths, *, source_csv: Path | None = None) -> pd.DataFrame:
    """One row per target: the two VP distances, and what backs them.

    Columns: `target_id`, `d_geo_km`, `d_sping_km`, `sping_rtt_ms`,
    `geo_vp_rtt_ms`, `n_vp`.

    Read through `edges.resolve_source_csv`, which owns the "which CSV is this
    arm's" question and refuses a mesh superset standing in for a weighted arm.
    The RTT minimum per `(target, vp)` pair matches `edges.load_min_rtt`, so the
    VP nominated here is the VP every LTD in the framework built its constraint
    from.
    """
    from scripts.libs.canonical.schema import load_canonical_csv

    path = edges.resolve_source_csv(run, source_csv)
    df = load_canonical_csv(path)
    df = (
        df.groupby(["target_id", "vp_id"], as_index=False)
        .agg(
            rtt_ms=("rtt_ms", "min"),
            vp_lat=("vp_lat", "first"),
            vp_lon=("vp_lon", "first"),
            target_lat=("target_lat", "first"),
            target_lon=("target_lon", "first"),
        )
    )
    df["d_km"] = _haversine_km(
        df.vp_lat.values, df.vp_lon.values, df.target_lat.values, df.target_lon.values
    )
    by_target = df.groupby("target_id", sort=False)
    geo_idx = by_target["d_km"].idxmin()
    rtt_idx = by_target["rtt_ms"].idxmin()
    out = pd.DataFrame(
        {
            "d_geo_km": df.loc[geo_idx].set_index("target_id")["d_km"],
            "geo_vp_rtt_ms": df.loc[geo_idx].set_index("target_id")["rtt_ms"],
            "d_sping_km": df.loc[rtt_idx].set_index("target_id")["d_km"],
            "sping_rtt_ms": df.loc[rtt_idx].set_index("target_id")["rtt_ms"],
            "n_vp": by_target.size(),
        }
    ).reset_index()
    # The definition guarantees it; a violation means the two VPs were picked
    # off different frames and every number downstream is untrustworthy.
    bad = out[out.d_geo_km > out.d_sping_km + 1e-6]
    if len(bad):
        raise ValueError(
            f"{run.run_id}: {len(bad)} targets have d_geo > d_sping, which the "
            f"definition forbids (the closest VP cannot be further than any "
            f"particular VP). First: {bad.target_id.iloc[0]!r}."
        )
    return out


def _cells(run: RunPaths, method: str, nside: int) -> pd.DataFrame:
    path = (
        run.analysis_dir(CLS_ACCURACY_KIND)
        / grid_slug(nside)
        / classify.CELLS_PARQUET.format(method=method)
    )
    if not path.exists():
        raise MissingArtifactError(
            f"{path} is absent; run `classify --run-id {run.run_id}` first."
        )
    return pd.read_parquet(path, columns=["target_id", "error_km", "status"])


def scored_methods(run: RunPaths, nside: int = SOURCE_NSIDE) -> list[str]:
    d = run.analysis_dir(CLS_ACCURACY_KIND) / grid_slug(nside)
    suffix = classify.CELLS_PARQUET.format(method="")
    return sorted(p.name[: -len(suffix)] for p in d.glob("*" + suffix))


def load(
    run_ids: list[str],
    *,
    methods: list[str] | None = None,
    nside: int = SOURCE_NSIDE,
    outputs_root: Path | None = None,
    source_csv: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Pooled `(run_id, target_id)` rows carrying both distances and per-method
    error, plus the provenance the manifest needs.

    Returns a long frame with one row per `(run, target, method)` and the two
    distance columns repeated onto it, which is the shape both the cohort
    filter and the stats table consume.
    """
    runs = [resolve_run(r, outputs_root) if outputs_root else resolve_run(r)
            for r in run_ids]
    per_run_targets: dict[str, set[str]] = {}
    dist_frames, err_frames = [], []
    available: dict[str, set[str]] = {}
    for run in runs:
        d = vp_distances(run, source_csv=(source_csv or {}).get(run.run_id))
        d["run_id"] = run.run_id
        dist_frames.append(d)
        per_run_targets[run.run_id] = set(d.target_id)
        available[run.run_id] = set(scored_methods(run, nside))
    guard_disjoint_targets(per_run_targets)

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

    for run in runs:
        for m in chosen:
            c = _cells(run, m, nside)
            c["solved"] = classify.solved_mask(c)
            c["method"] = m
            c["run_id"] = run.run_id
            err_frames.append(c)

    dist = pd.concat(dist_frames, ignore_index=True)
    err = pd.concat(err_frames, ignore_index=True)
    long = err.merge(dist, on=["run_id", "target_id"], how="left", validate="m:1")
    if long[MEASURE_COLUMNS[GEO]].isna().any():
        n = int(long[MEASURE_COLUMNS[GEO]].isna().sum())
        raise ValueError(
            f"{n} scored targets have no edge in the canonical CSV, so no VP "
            f"distance could be computed for them. The scored population and "
            f"the measurement table disagree; do not pool them."
        )
    meta = {
        "run_ids": [r.run_id for r in runs],
        "nside": int(nside),
        "methods": chosen,
        "n_targets": int(len(dist)),
        "n_vp_per_target_median": float(dist.n_vp.median()),
    }
    return long, meta


def cohort_k(long: pd.DataFrame, cohort: str) -> int | None:
    """How many rows per method `cohort` asks for. `None` for `all`.

    A fraction of the POOLED target count, not of what a method answered, so
    every method is asked for the same number and the bounds are comparable.
    Whether a method can *supply* it is a separate question -- see
    `cohort_frame`.
    """
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; known: {sorted(COHORTS)}")
    frac = COHORTS[cohort]
    if frac is None:
        return None
    return int(round(frac * long.groupby("method").size().max()))


def cohort_frame(long: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """The rows `cohort` keeps, per method.

    `all` keeps every evaluated row, answered or not, because the population
    reference has to be the population. A percentile cohort ranks on
    `error_km`, so it can only see answered rows.

    At small fractions every method fills `cohort_k` and the cohorts are
    equal-sized. At large ones they need not be: a method that answered fewer
    targets than k supplies everything it has and no more. That is a **short**
    cohort, and it is not padded -- there is nothing to pad it with, and
    ranking is by error so the rows it lacks are the worst ones, which would
    flatter its bound if borrowed from anywhere. It is reported instead: the
    manifest carries `cohort_k_requested` beside `n_cohort_per_method`, the
    figure's title gives the range rather than one number, and a reader
    comparing bounds across a short cohort is comparing different n.
    """
    k = cohort_k(long, cohort)
    if k is None:
        return long.copy()
    out = (
        long[long.solved]
        .sort_values("error_km", kind="mergesort")
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
    all-target reference the cohort rows are read against."""
    rows = []
    dedup = long.drop_duplicates(subset=["run_id", "target_id"])
    for measure in measures:
        col = MEASURE_COLUMNS[measure]
        pop = _describe(dedup[col])
        rows.append({"scope": "population", "method": "", "method_label": "all targets",
                     "measure": measure, "circular": False, **pop})
        for method, g in cohort_rows.groupby("method", sort=False):
            d = _describe(g[col])
            rows.append({
                "scope": "cohort",
                "method": method,
                "method_label": method_label(method),
                "measure": measure,
                # Shortest-Ping's error IS d_sping, so ranking its cohort by
                # error ranks it by the orange violin itself.
                "circular": method == classify.SHORTEST_PING,
                **d,
                "lift_p50": (pop["p50_km"] / d["p50_km"]) if d["p50_km"] else float("nan"),
            })
    return pd.DataFrame(rows)


def order_methods(
    cohort_rows: pd.DataFrame, methods: list[str], measures: list[str]
) -> list[str]:
    """Rows sorted by their bound on the primary measure, tightest first.

    The bound is the claim, so ordering by it makes the figure a ladder: the
    methods whose accurate targets always had a near VP sit at the top, and the
    ones that stay accurate when latency points far away fall to the bottom.
    Combo-id order buries exactly that.

    `figure_outcome_bars` ranks each panel by its own in-cell share for the same
    reason, and with the same consequence -- a method does not keep one row
    across cohorts, which is deliberate: the orders genuinely differ.
    """
    primary = MEASURE_COLUMNS[SPING if SPING in measures else measures[0]]
    bound = (
        cohort_rows.groupby("method")[primary].max().reindex(methods)
    )
    return list(bound.sort_values(kind="mergesort").index)


def _violin(ax, values: np.ndarray, centre: float, color: str, half: float) -> float:
    """One violin, drawn in kilometres onto a real log axis. Returns the max.

    The density is estimated in **log space** and then drawn at `10 ** xs`,
    because the axis is logarithmic: a KDE fitted on raw kilometres would put
    one bandwidth across both the 0.5 km end and the 700 km end, and the shape
    near zero would be an artifact of the far tail.

    Drawing in kilometres rather than in log10 units is what lets matplotlib
    own the ticks. The earlier version plotted `log10(x)` on a linear axis and
    hand-placed ticks at 0.5/1/5/10/50, which reads as an irregular ladder
    because those are not the decade marks a log axis is expected to carry.

    The KDE is cut at the data range. An uncut one puts its right edge past the
    largest observation, and the largest observation is the bound this figure
    exists to report.
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


def plot(
    long: pd.DataFrame,
    cohort_rows: pd.DataFrame,
    *,
    cohort: str,
    measures: list[str],
    meta: dict,
    out_png: Path,
) -> Path:
    labels = order_methods(cohort_rows, meta["methods"], measures)
    fig, ax = plt.subplots(figsize=(11.5, 1.0 + 0.82 * len(labels)))
    paired = len(measures) == 2
    half = 0.185 if paired else 0.30
    offsets = {measures[0]: -0.205, measures[1]: 0.205} if paired else {measures[0]: 0.0}

    for i, method in enumerate(labels):
        g = cohort_rows[cohort_rows.method == method]
        for measure in measures:
            col = MEASURE_COLUMNS[measure]
            vals = g[col].dropna().values
            if not len(vals):
                continue
            hi = _violin(ax, vals, i + offsets[measure], MEASURE_HUES[measure], half)
            ax.annotate(
                f"{hi:,.0f}" if hi >= 10 else f"{hi:,.1f}",
                (hi, i + offsets[measure]), xytext=(7, 0), textcoords="offset points",
                va="center", fontsize=8.2, color=MEASURE_HUES[measure], zorder=6,
            )

    # No population reference rules. The all-target medians are one row of the
    # stats CSV (`scope="population"`), and drawing them put two dashed lines
    # through every violin for a comparison the cohort rows do not need.

    cols = [MEASURE_COLUMNS[m] for m in measures]
    # Limits hug the data; neither end is snapped out to a whole decade.
    # That is the convention for a log axis -- matplotlib's own autoscale does
    # not snap either -- and a partial leading decade is normal rather than a
    # defect: the unlabelled 2..9 minors are the texture a reader locates a
    # value from, which is why the extreme edge does not need a number on it.
    # Snapping the low end out to 0.1 km here cost a third of the width to
    # whitespace, for one label under a region holding no data.
    #
    # The left margin is a fixed fraction of the drawn log span. The right is
    # larger because the bound labels sit outside the violin.
    #
    # If a future cohort ever spans under ~2 decades, decade labels alone go
    # too sparse and the convention flips to labelling the 2x and 5x minors.
    # Nothing here does: the narrowest is p5 at ~0.46-69 km.
    data_lo = float(cohort_rows[cols].min().min())
    data_hi = float(cohort_rows[cols].max().max())
    span = np.log10(data_hi) - np.log10(data_lo)
    lo = 10 ** (np.log10(data_lo) - 0.03 * span)
    hi = data_hi * 2.2
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=99)
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([method_label(m) for m in labels], fontsize=10)
    ax.set_ylim(len(labels) - 0.42, -0.62)
    ax.set_xlabel("distance from target to vantage point (km, log scale)",
                  fontsize=10, color=INK_2)
    # The decade lines carry the reading; the 2..9 minors are the log texture
    # that makes a position between decades legible, so they are drawn lighter
    # rather than at the same weight.
    ax.grid(axis="x", which="major", color=GRID, lw=0.9)
    ax.grid(axis="x", which="minor", color=GRID, lw=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="minor", length=2, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    ax.legend(
        handles=[Patch(facecolor=MEASURE_HUES[m], alpha=0.55,
                       edgecolor=MEASURE_HUES[m], label=MEASURE_LABELS[m])
                 for m in measures],
        loc="upper center", bbox_to_anchor=(0.5, -0.155 - 0.02 * (6 - len(labels))),
        ncol=len(measures), frameon=False, fontsize=9.5,
    )
    sizes = cohort_rows.groupby("method").size()
    lo = int(sizes.min()) if len(sizes) else 0
    hi = int(sizes.max()) if len(sizes) else 0
    if COHORTS[cohort] is None:
        title = f"all {meta['n_targets']:,} evaluated targets"
    elif lo == hi:
        title = (f"{cohort} cohort: the {hi:,} targets each method "
                 f"placed most accurately")
    else:
        # A short cohort: someone answered fewer targets than the cohort asks
        # for. One number here would be true of some rows and not others.
        title = (f"{cohort} cohort: each method's {lo:,}-{hi:,} most accurately "
                 f"placed targets (fewer where it answered fewer)")
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    return out_png


def _manifest(meta: dict, cohort: str, measures: list[str],
              cohort_rows: pd.DataFrame, stats: pd.DataFrame) -> str:
    n_per = (cohort_rows.groupby("method").size().to_dict() if len(cohort_rows) else {})
    # From the pooled target count, not from `cohort_rows` -- those are the
    # rows k already selected, so asking them what k was is circular.
    frac = COHORTS[cohort]
    k = None if frac is None else int(round(frac * meta["n_targets"]))
    body = {
        "figure": "vp_proximity",
        "cohort": cohort,
        "cohort_fraction": COHORTS[cohort],
        "cohort_k_requested": k,
        "n_cohort_short": {
            method_label(m): int(v) for m, v in n_per.items()
            if k is not None and int(v) < k
        },
        "measures": measures,
        "datasets": dataset_slug(meta["run_ids"]),
        "run_ids": meta["run_ids"],
        "source_nside": meta["nside"],
        "n_targets_pooled": meta["n_targets"],
        "n_vp_per_target_median": meta["n_vp_per_target_median"],
        "n_cohort_per_method": {method_label(k): int(v) for k, v in n_per.items()},
        "row_order": [
            method_label(m)
            for m in order_methods(cohort_rows, meta["methods"], measures)
        ],
        "policy": {
            "cohort_selection": (
                "Each method's own smallest error_km, over solved rows only "
                "(classify.solved_mask). An unanswered target has no error to "
                "rank on. This is NOT the outcome-bars denominator, where a "
                "refusal counts as wrong. The cohort size k "
                "(cohort_k_requested) is a fraction of the POOLED target "
                "count, so every method is ASKED for the same number and a "
                "method that answered fewer draws from a smaller pool. At "
                "large fractions it may not be able to supply k at all; those "
                "methods are listed in n_cohort_short and their bound rests "
                "on fewer rows than the rest."
            ),
            "d_geo_km": (
                "Great-circle distance to the geographically closest VP that "
                "measured this target."
            ),
            "d_sping_km": (
                "Great-circle distance to the smallest-RTT VP, which is the "
                "coordinate Shortest-Ping returns; d_geo <= d_sping always."
            ),
            "shortest_ping_is_circular": (
                "Shortest-Ping's error equals d_sping_km, so its cohort is "
                "ranked by the quantity its own orange violin draws. Kept as "
                "the reference row, marked `circular` in the CSV."
            ),
            "kde": (
                "Cut at the data range so the drawn right edge is the true "
                "maximum, which is the bound this figure reports. Interior "
                "density is still smoothed; see distinct_values and "
                "max_tie_share per row."
            ),
            "max_is_the_claim": (
                "max_km is the bound: no target in this cohort lay further "
                "than this from that VP."
            ),
        },
        "bounds": {
            f"{row.measure}:{row.method_label}": row.max_km
            for row in stats[stats.scope == "cohort"].itertuples()
        },
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    run_ids: list[str],
    *,
    cohorts: list[str] | None = None,
    methods: list[str] | None = None,
    geo: bool = True,
    sping: bool = True,
    nside: int = SOURCE_NSIDE,
    outputs_root: Path | None = None,
    analysis_root: Path | None = None,
) -> list[Path]:
    """PNG, stats CSV and manifest per cohort. Returns the PNGs."""
    measures = [m for m, on in ((GEO, geo), (SPING, sping)) if on]
    if not measures:
        raise ValueError("nothing to draw: --no-geo and --no-sping cannot both be set.")
    cohorts = cohorts or ["p25"]
    long, meta = load(run_ids, methods=methods, nside=nside, outputs_root=outputs_root)
    out_dir = output_dir(run_ids, analysis_root=analysis_root)
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
