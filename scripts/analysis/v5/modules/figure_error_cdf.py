"""Error-distance CDF, one curve per method, log x. Ported from v4.

The companion to the outcome bars. Those say *where* a prediction landed --
which serving region, which ring, or no answer. They cannot say *how far off*
it was, and the two can disagree: a method rarely in the right cell may still
be consistently close, and one often in the right cell may have a catastrophic
tail. The disagreement is the finding, so the package needs both halves.

## Distance to the TG, not to the seed

The column is `pred_dist_to_tg_km`: prediction to the raw TG coordinate. v5
also writes `pred_dist_to_seed_km`, but the seeds are regrouped at every rung,
so that distance moves with the grid. The TG distance does not -- the answer
space defines the labels and stays out of a distance that has its own ground
truth. So the figure is identical at every rung, and the artifacts carry **no
`healpix-<n>` in their names** and land in the rung-free parent of the rung
directories (`classify/`). `test_figure_error_cdf` re-checks the premise
against real runs.

## Unanswered rows are excluded

A `FALLBACK` row still carries a coordinate -- the shortest-ping VP's -- so
filtering on NaN will not drop it, and pooling it would pull a method's curve
toward the baseline exactly where the method failed. `status.solved_mask` is
the shared predicate, the one `classify.summarize` applies before taking
`pred_dist_to_tg_km_p50`/`p90`. Those rows are the outcome bars' dark-grey "no
answer" segment, so the population behind a curve here is that figure's
non-grey stack, method for method. The `plotted/total` column in the
percentile box shows where a curve rests on a subset (VAN, with its
fallbacks).

It also carries the case a hand-written filter gets wrong: S-P's rows are all
`BASELINE`, never `SUCCESS`, so `status == "SUCCESS"` would empty the baseline.

## ...unless they are parked at the 10,000 km sentinel

`--unanswered sentinel` puts every unanswered row back in at `SENTINEL_KM`
instead of dropping it, so each curve is drawn over the same denominator --
the whole TG roster -- and reaches y=1 only at the sentinel. A method's refusal
rate then reads straight off the figure: the height of its curve just left of
the sentinel line is the fraction it answered. `exclude` cannot show that,
because there every curve reaches 1 on its own population and VAN's refusals
are invisible in the shape.

The cost is that the percentiles move with the policy, and a censored one is
not a distance anyone measured -- it means "this method had not answered that
share of its TGs yet". A censored cell prints the sentinel plainly, so the
footnote and the labelled sentinel line carry that reading; the CSV declares
`unanswered_policy`/`sentinel_km`, and the artifacts take a `.sentinel.` infix
so the two populations cannot overwrite each other. Only the `exclude` files
join to `accuracy.csv`.

## Pooling concatenates rows; it cannot average percentiles

`--layout pooled` draws one curve per method over every selected run's solved
rows at once -- a micro-pool, so a dataset weighs by its TG count. Averaging
the runs' published p50s is a different quantity, and on as01-03 it reverses
the leader (SOI first at 96.1 km pooled; OCT-H first on the mean of p50s). A
method enters only if **every** run scored it, and TG ids are checked
disjoint.

## The baseline is dark grey and dashed

S-P is drawn recessive, because the CDF shows the CBG methods *against* a
reference. The grey is `_INK_2`, **not** `methods.OTHER_HUE` (== `_MUTED`):
any unpublished method falls into that bucket, and a baseline sharing its hex
would be told apart by the dash alone.

Command: `plot-error-cdf`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v5.modules import classify as C  # noqa: E402
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.methods import (
    method_colors,
    method_label,
    method_order,
    method_term_table,
)
from scripts.analysis.v5.modules.paths import (
    CLASSIFY_KIND,
    MissingArtifactError,
    RunPaths,
    grid_slug,
)
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask

#: The column this figure is about. A constant so the "distance to the raw TG,
#: never through the seed" decision is greppable.
DIST_COLUMN = "pred_dist_to_tg_km"

#: Only what the curve and the percentile box need.
READ_COLUMNS: tuple[str, ...] = ("tg_id", "status", DIST_COLUMN)

#: Computed per method and written to the CSV. 50 and 90 are the two
#: `accuracy.csv` publishes, so this CSV joins to it on its headline numbers,
#: and 90 is the rank cascade's tail term.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 90, 95)

#: Axis text sizes (pt) for `PAPER_FIGSIZE`. Set for the printed page rather
#: than scaled from the canvas: at 4 inches wide a proportional scale would put
#: the tick labels under 5pt.
_TITLE_PT = 9.0
_SUBTITLE_PT = 6.0
_LABEL_PT = 8.0
_TICK_PT = 7.0
_LEGEND_PT = 6.5
_GUIDE_PT = 5.5

#: Panel size (inches): a single paper column. Everything else on the figure is
#: sized for it.
PAPER_FIGSIZE: tuple[float, float] = (4.0, 3.0)

#: y at a quarter each, so the gridlines are the quartiles and the reader takes
#: p25/p50/p75 off the panel without a table.
Y_TICKS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Axis names. Short: the column name, the row policy and the units convention
#: live in the manifest, which is where the numbers are read from anyway.
X_LABEL = "Distance (km)"
Y_LABEL = "CDF"

#: Reference verticals (km). Neutral ink, **not** green/red: green is the
#: Octant family and red is SPO, so coloured guides would read as series.
THRESHOLDS_KM: tuple[int, ...] = (100, 500, 1000)

#: Fixed x range, both bounds, so per-run and pooled figures share one axis.
#: The floor is 0.1 km because sub-kilometre errors are not rare (observed
#: minimum on as01-03: 0.106 km); the manifest records any clamped count.
X_MIN_KM = 0.1
DEFAULT_X_MAX_KM = 10_000.0

#: What to do with a TG the method did not answer.
EXCLUDE = "exclude"
SENTINEL = "sentinel"
UNANSWERED_POLICIES: tuple[str, ...] = (EXCLUDE, SENTINEL)

#: Where `sentinel` parks an unanswered row. Beyond any error a method could
#: plausibly *make* on this roster, and short of the 20,015 km antipodal
#: maximum, so a real distance can never be mistaken for a censored one.
SENTINEL_KM = 10_000.0

#: The sentinel layout's default right edge: half the Earth's circumference,
#: the largest surface error that exists. It has to clear `SENTINEL_KM` or the
#: riser to y=1 would be drawn on the spine, and it clears every real value
#: too, so nothing falls off the edge unnoticed.
SENTINEL_X_MAX_KM = 20_015.1

PER_RUN = "per-run"
POOLED = "pooled"
LAYOUTS: tuple[str, ...] = (PER_RUN, POOLED)

STEM = "error_cdf"


def artifact_names(layout: str, policy: str = EXCLUDE) -> tuple[str, str, str]:
    """`(png, csv, manifest)` for a layout and unanswered policy.

    No `{slug}`: the figure does not vary with the rung, so the name does not
    pretend it might. It *does* vary with the policy -- same axes, different
    population and different percentiles -- so `sentinel` takes its own infix
    rather than overwriting the file `accuracy.csv` joins to.
    """
    parts = [STEM]
    if layout == POOLED:
        parts.append("pooled")
    if policy == SENTINEL:
        parts.append(SENTINEL)
    stem = ".".join(parts)
    return (f"{stem}.png", f"{stem}.csv", f"{stem}.manifest.json")


#: `layout -> (png, csv, manifest)` under the default policy.
NAMES: dict[str, tuple[str, str, str]] = {layout: artifact_names(layout) for layout in LAYOUTS}

#: Which rung's parquets to read. Any would do -- that is the point -- so the
#: finest is the default and `--nside` exists to demonstrate nothing changes.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: Summed when runs are pooled.
COUNT_KEYS: tuple[str, ...] = ("n_tgs", "n_solved", "n_failed", "n_no_distance")

#: The CDF's way out of a pooling refusal (`--layout compare` is the outcome
#: bars' and does not exist here).
REMEDY_COMMON = "--layout per-run to keep each dataset on its own figure."
REMEDY_DISJOINT = "Use --layout per-run, which keeps each dataset on its own figure."

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

# ---- loading ----------------------------------------------------------------


def scored_methods(cls_dir: Path) -> list[str]:
    """Method ids with a `*_tgs.parquet` in `cls_dir`. Ranked later, by p50."""
    suffix = C.TGS_PARQUET.format(method="")
    return sorted(p.name[: -len(suffix)] for p in Path(cls_dir).glob(f"*{suffix}"))


def _load_tgs(
    run: RunPaths,
    method: str,
    nside: int,
    *,
    analysis_root: Path | None = None,
    columns: tuple[str, ...] = READ_COLUMNS,
) -> pd.DataFrame:
    """One run's per-TG scored rows for one method, `columns` wide."""
    path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=method)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(path, columns=list(columns))


def load_errors(
    run: RunPaths,
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> dict[str, dict]:
    """Per method: the solved rows' distances, the row counts, and the TG ids.

    `n_solved - n_no_distance` equals `accuracy.csv`'s `n_solved` by
    construction: both come from `solved_mask` on the same frame, and
    `summarize` counts an answered row without a coordinate as failed.

    The ids are the **whole** scored population, not the solved subset, because
    the pooled guard protects the denominator, and `n_tgs` is a denominator.
    """
    cls_dir = run.classify_dir(nside, root=analysis_root)
    chosen = list(methods) if methods else scored_methods(cls_dir)
    if not chosen:
        raise MissingArtifactError(
            f"{cls_dir} holds no *_tgs.parquet; run `classify --run-id {run.run_id}` first"
        )
    out: dict[str, dict] = {}
    for method in chosen:
        df = _load_tgs(run, method, nside, analysis_root=analysis_root)
        solved = solved_mask(df)
        values = df.loc[solved, DIST_COLUMN].to_numpy(dtype=float)
        finite = np.isfinite(values)
        out[method] = {
            "errors": values[finite],
            "tg_ids": set(df["tg_id"]),
            "n_tgs": int(len(df)),
            "n_solved": int(solved.sum()),
            "n_failed": int((~solved).sum()),
            # Answered, yet no distance. Zero on every run today; carried so it
            # cannot start happening quietly.
            "n_no_distance": int((~finite).sum()),
        }
    return out


def error_matrix(
    run: RunPaths,
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """`load_errors` keyed by TG: a `tg_id x method` frame, plus each TG's site.

    The paired twin. `load_errors` hands back each method's distances as a bare
    array, which is all a CDF needs and exactly what a per-TG comparison cannot
    use. Same row policy: a cell is the method's distance where `solved_mask`
    accepts the row and the distance is finite, NaN otherwise -- so a FALLBACK
    row, which carries the shortest-ping VP's coordinate, is NaN here and not
    S-P's number under another method's name.

    Every method must cover the same TGs. `classify` scores them all over the
    run's evaluated roster, so a difference means a half-rescored run, and
    intersecting would pair the methods over a population nobody asked for.

    The site key (`sites.site_key`, `(run_id, tg_lat, tg_lon)`) rides along
    because ~20 IP replicas share a site: counts of TGs overstate the number
    of independent observations, and a caller that reports one should be able
    to report the other.
    """
    from scripts.analysis.v5.modules.sites import site_key

    cls_dir = run.classify_dir(nside, root=analysis_root)
    chosen = list(methods) if methods else scored_methods(cls_dir)
    if not chosen:
        raise MissingArtifactError(
            f"{cls_dir} holds no *_tgs.parquet; run `classify --run-id {run.run_id}` first"
        )
    cols: dict[str, pd.Series] = {}
    sites: pd.Series | None = None
    for method in chosen:
        df = _load_tgs(
            run, method, nside, analysis_root=analysis_root,
            columns=(*READ_COLUMNS, "tg_lat", "tg_lon"),
        )
        dist = df[DIST_COLUMN].astype(float).where(solved_mask(df))
        dist = dist.where(np.isfinite(dist))
        cols[method] = dist.set_axis(df["tg_id"])
        if sites is None:
            sites = site_key(df, run_id=run.run_id).set_axis(df["tg_id"])
    first = cols[chosen[0]].index
    odd = [m for m in chosen[1:] if not cols[m].index.sort_values().equals(first.sort_values())]
    if odd:
        sizes = {m: int(len(cols[m])) for m in chosen}
        raise ValueError(
            f"{run.run_id}: methods cover different TG sets ({sizes}); a per-TG "
            f"comparison needs one roster. Re-run `classify --run-id {run.run_id}`, "
            f"or pin a consistent set with --method."
        )
    frame = pd.DataFrame(cols).reindex(first)
    frame.index.name = "tg_id"
    return frame, sites.reindex(first)


def load_per_run(
    runs: list[RunPaths],
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> dict[str, dict[str, dict]]:
    """`{run_id: load_errors(run)}`, read once."""
    return {
        run.run_id: load_errors(run, nside, methods=methods, analysis_root=analysis_root)
        for run in runs
    }


def pool(loaded: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Every run's rows for a method concatenated into one population."""
    common = cross.guard_common_methods(
        {rid: set(entries) for rid, entries in loaded.items()}, remedy=REMEDY_COMMON
    )
    cross.guard_disjoint_tgs(
        {
            rid: set().union(*(entries[m]["tg_ids"] for m in common))
            for rid, entries in loaded.items()
        },
        remedy=REMEDY_DISJOINT,
    )
    out: dict[str, dict] = {}
    for method in common:
        parts = [entries[method] for entries in loaded.values()]
        out[method] = {
            "errors": np.concatenate([p["errors"] for p in parts]),
            "tg_ids": set().union(*(p["tg_ids"] for p in parts)),
            **{k: int(sum(p[k] for p in parts)) for k in COUNT_KEYS},
        }
    return out


def pooled_errors(
    runs: list[RunPaths],
    nside: int = SOURCE_NSIDE,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> dict[str, dict]:
    """`pool` over `load_per_run` -- the whole pooled population in one call."""
    return pool(load_per_run(runs, nside, methods=methods, analysis_root=analysis_root))


def censor(
    loaded: dict[str, dict],
    *,
    policy: str = EXCLUDE,
    sentinel_km: float = SENTINEL_KM,
) -> dict[str, dict]:
    """Apply the unanswered-row policy. `exclude` is a pass-through.

    `sentinel` appends one `sentinel_km` value per unanswered row, so
    `len(errors) == n_tgs` for every method and the curves share a denominator.
    Unanswered means whatever `solved_mask` rejected **plus** the answered rows
    with no coordinate: both are TGs this figure has no distance for, and both
    would otherwise shrink one method's denominator and not another's.

    Pool first, censor second -- the counts this reads are the ones `pool` sums.
    """
    if policy not in UNANSWERED_POLICIES:
        raise ValueError(
            f"unknown unanswered policy {policy!r}; pick from {list(UNANSWERED_POLICIES)}"
        )
    out: dict[str, dict] = {}
    for method, entry in loaded.items():
        values = np.asarray(entry["errors"], dtype=float)
        # From n_tgs, not n_failed + n_no_distance: the identity the figure
        # needs is len(errors) == n_tgs, so derive it from the denominator.
        n_censored = int(entry["n_tgs"] - len(values)) if policy == SENTINEL else 0
        out[method] = {
            **entry,
            "errors": (
                np.concatenate([values, np.full(n_censored, float(sentinel_km))])
                if n_censored
                else values
            ),
            "n_censored": n_censored,
            # A measured distance out here would be indistinguishable from a
            # censored one. Zero on every run today; the manifest carries it so
            # it cannot start happening quietly.
            "n_real_at_or_above_sentinel": (
                int((values >= sentinel_km).sum()) if policy == SENTINEL else 0
            ),
        }
    return out


def resolve_x_max(
    policy: str, max_x_km: float | None = None, sentinel_km: float = SENTINEL_KM
) -> float:
    """The right edge: the caller's, else the policy's default.

    `sentinel` needs a wider axis than `exclude` -- its rightmost mass sits
    *at* `SENTINEL_KM`, which is `exclude`'s edge -- so the two layouts cannot
    share one default. Within a layout the bound is still fixed, which is what
    per-run and pooled comparability actually rests on.
    """
    if max_x_km is not None:
        return float(max_x_km)
    return SENTINEL_X_MAX_KM if policy == SENTINEL else DEFAULT_X_MAX_KM


# ---- the table --------------------------------------------------------------

#: The sort cascade, best first: p50 (drawn as the dashed horizontal), then the
#: p90 tail, then the larger population, then the id -- total and stable.
_RANK_KEYS: tuple[str, ...] = (
    f"{DIST_COLUMN}_p50", f"{DIST_COLUMN}_p90", "n_solved", "method",
)
_RANK_ASC: tuple[bool, ...] = (True, True, False, True)


def pcol(p: int) -> str:
    """`pred_dist_to_tg_km_p<p>` -- the name `accuracy.csv` uses for 50 and 90."""
    return f"{DIST_COLUMN}_p{p}"


def percentile_table(loaded: dict[str, dict]) -> pd.DataFrame:
    """One row per method: the counts plus the reported percentiles, ranked.

    `Series.quantile` (linear) rounded to 3dp -- the same call
    `classify.summarize` makes, so p50/p90 agree with `accuracy.csv` digit for
    digit. That holds under `exclude` only: `censor` changes the population,
    and a quantile of a censored sample is a different quantity. The `sentinel`
    artifacts are named apart for exactly that reason.
    """
    rows: list[dict] = []
    for method, entry in loaded.items():
        values = pd.Series(entry["errors"], dtype=float)
        row = {
            "method": method,
            "method_label": method_label(method),
            "is_baseline": method == SHORTEST_PING,
            **{k: int(entry[k]) for k in COUNT_KEYS},
            "n_plotted": int(len(values)),
            # 0 under `exclude`; under `sentinel`, n_plotted - n_censored is
            # the number of rows that are a measurement.
            "n_censored": int(entry.get("n_censored", 0)),
        }
        for p in PERCENTILES:
            row[pcol(p)] = round(float(values.quantile(p / 100)), 3) if len(values) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(list(_RANK_KEYS), ascending=list(_RANK_ASC)).reset_index(drop=True)


def curve_order(table: pd.DataFrame) -> list[str]:
    """Legend and box order: `methods.TERM_ORDER`, not the table's ranking.

    Fixed, so a method holds the same row in every figure in the package and
    the legend does not reshuffle when the data moves. Rank is still on the
    panel -- the box's p50 column is sorted-looking or not, and that is itself
    readable -- and the CSV is still written best-first.
    """
    return method_order(table["method"])


# ---- drawing ----------------------------------------------------------------


def _cdf(values: np.ndarray, min_x_km: float = X_MIN_KM):
    """`(x, y)` for an empirical CDF, x clamped up to the log floor.

    The clamp is a rendering concession applied here only; `percentile_table`
    reads the unclamped values.
    """
    xs = np.sort(np.maximum(values, min_x_km))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def _draw_guides(ax, min_x_km: float, max_x_km: float) -> None:
    """The 100/500/1,000 km verticals, under everything.

    No median rule any more: `Y_TICKS` puts a gridline on 0.5, which is the
    same line drawn twice.
    """
    for km in THRESHOLDS_KM:
        if min_x_km < km < max_x_km:
            ax.axvline(km, color=_GRID, linestyle=":", linewidth=1.2, zorder=1)
            # Rotated under the top spine: on a log axis 500 and 1,000 km are a
            # third of a decade apart, so horizontal labels collide.
            ax.annotate(
                f"{km}", xy=(km, 0.995), xytext=(-2, 0), textcoords="offset points",
                fontsize=_GUIDE_PT, color=_MUTED, ha="right", va="top", rotation=90, zorder=1,
            )


def _draw_sentinel(ax, sentinel_km: float, max_x_km: float) -> None:
    """The censoring line. Darker than the threshold guides, and labelled.

    Those are reference marks on a measured axis; this one is the boundary
    between measurement and "no answer", so it has to read as a different kind
    of thing. Still ink, never a series colour.
    """
    if not sentinel_km < max_x_km:
        raise ValueError(
            f"sentinel {sentinel_km:g} km must sit inside the axis (max {max_x_km:g} km), "
            "or the rise to y=1 is drawn on the spine; raise --max-x-km"
        )
    ax.axvline(sentinel_km, color=_INK_2, linestyle="-.", linewidth=1.2, alpha=0.8, zorder=1)
    ax.annotate(
        f"unanswered → {sentinel_km:,.0f} km", xy=(sentinel_km, 0.995),
        xytext=(-3, 0), textcoords="offset points", fontsize=_GUIDE_PT, color=_INK_2,
        ha="right", va="top", rotation=90, zorder=1,
    )


def _style_axes(
    ax, min_x_km: float, max_x_km: float, *, sentinel_km: float | None = None
) -> None:
    """Log x on the fixed range, `Y_TICKS` on y, and the quiet spines."""
    from matplotlib.ticker import FuncFormatter

    ax.set_xscale("log")
    # `%g`: a ScalarFormatter renders the 0.1 km decade as a bare "0".
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(min_x_km, max_x_km)
    ax.set_ylim(0, 1)
    ax.set_yticks(list(Y_TICKS))
    ax.set_yticklabels([f"{y:g}" for y in Y_TICKS])
    # Plain axis names. What the distance *is*, and which rows are behind the
    # curve, are the manifest's job now that the figure carries no footnote.
    ax.set_xlabel(X_LABEL, fontsize=_LABEL_PT, color=_INK_2)
    ax.set_ylabel(Y_LABEL, fontsize=_LABEL_PT, color=_INK_2)
    ax.grid(True, which="both", color=_GRID, linewidth=0.5, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelsize=_TICK_PT)


def _curve_style(method: str, colors: dict[str, str]) -> dict:
    """Colour is the method; the baseline is recessive grey and dashed."""
    if method == SHORTEST_PING:
        # _INK_2, not _MUTED: _MUTED is byte-for-byte `methods.OTHER_HUE`.
        return {"color": _INK_2, "linestyle": "--", "linewidth": 1.3, "zorder": 3}
    return {"color": colors[method], "linestyle": "-", "linewidth": 1.2, "zorder": 2}


def plot_cdf(
    loaded: dict[str, dict],
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    min_x_km: float = X_MIN_KM,
    max_x_km: float = DEFAULT_X_MAX_KM,
    sentinel_km: float | None = None,
    figsize: tuple[float, float] = PAPER_FIGSIZE,
    dpi: int = 300,
    order: list[str] | None = None,
    style_fn=None,
    label_fn=None,
) -> Path:
    """One panel, one curve per method, log x. Both layouts draw through here.

    Honest limit: CDF curves **cross**, so no single ranking holds across the
    axis. The legend does not claim one -- it is `methods.TERM_ORDER`, fixed.

    Sized for a paper column, so the panel carries curves, a key and two axis
    names and nothing else. Every number that used to sit under it -- the
    percentiles, the row policy, the term glossary -- is in the CSV and the
    manifest written beside it.

    `sentinel_km` is the censoring mark, set when `loaded` came through
    `censor(policy=SENTINEL)`. It only draws and labels -- the sentinel values
    are already in `loaded`, and this function never invents rows.

    `order`, `style_fn` and `label_fn` exist for the one figure that draws series this
    package does not own: `ripe_vs_databases` puts two geolocation databases on
    these axes, and they are not methods -- they have no combo id, no
    `TERM_ORDER` position and no hue in the validated six-term palette. Rather
    than widen that palette (whose order every other figure's legend inherits),
    that caller passes its own order, its own `method -> line kwargs` and its
    own `method -> legend term`. Left unset, all three fall back to the
    package's: `curve_order`, `_curve_style` over `method_colors`, and
    `method_label`.
    """
    from matplotlib.lines import Line2D

    order = list(order) if order is not None else curve_order(table)
    colors = method_colors(order)
    style = style_fn if style_fn is not None else (lambda m: _curve_style(m, colors))
    label = label_fn if label_fn is not None else method_label

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    _draw_guides(ax, min_x_km, max_x_km)
    if sentinel_km is not None:
        _draw_sentinel(ax, sentinel_km, max_x_km)

    # Baseline last, so the dashed reference reads on top. Draw order is not
    # legend order, hence the explicit handles below.
    variants = [m for m in order if m != SHORTEST_PING]
    for method in variants + [m for m in order if m == SHORTEST_PING]:
        values = loaded[method]["errors"]
        if len(values) == 0:
            continue
        xs, ys = _cdf(values, min_x_km)
        ax.plot(xs, ys, alpha=0.95, gid=method, **style(method))

    _style_axes(ax, min_x_km, max_x_km, sentinel_km=sentinel_km)

    handles = [
        Line2D(
            [], [],
            label=f"{label(m)} (baseline)" if m == SHORTEST_PING else label(m),
            **{k: v for k, v in style(m).items() if k != "zorder"},
        )
        for m in order
    ]
    legend = ax.legend(
        handles=handles, loc="upper left", fontsize=_LEGEND_PT, frameon=False,
        handlelength=1.6, handletextpad=0.5, labelspacing=0.3, borderpad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(_INK_2)

    ax.set_title(title, fontsize=_TITLE_PT, fontweight="bold", color=_INK, pad=9)
    ax.annotate(
        subtitle, xy=(0.5, 1.005), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=_SUBTITLE_PT, color=_INK_2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


# ---- artifacts --------------------------------------------------------------

#: The CSV twin's columns. `run_id` and `dataset` lead so per-run and pooled
#: files concatenate into one frame.
#: `unanswered_policy`/`sentinel_km` ride along so a concatenation of the two
#: policies' files cannot silently average a censored percentile with a real one.
CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "unanswered_policy", "sentinel_km",
    "method", "method_label", "is_baseline",
    *COUNT_KEYS, "n_plotted", "n_censored", *[pcol(p) for p in PERCENTILES],
)


def _clamped(loaded: dict[str, dict], min_x_km: float) -> dict[str, int]:
    """Per method, how many drawn points the log floor moved. Empty is good."""
    return {
        m: int((e["errors"] < min_x_km).sum())
        for m, e in loaded.items()
        if (e["errors"] < min_x_km).any()
    }


def _manifest(
    layout: str,
    table: pd.DataFrame,
    loaded: dict[str, dict],
    *,
    run_ids: list[str],
    nside: int,
    png_name: str,
    csv_name: str,
    min_x_km: float,
    max_x_km: float,
    policy: str = EXCLUDE,
    sentinel_km: float = SENTINEL_KM,
    per_run: dict[str, int] | None = None,
) -> str:
    order = curve_order(table)
    censored = policy == SENTINEL
    body: dict = {
        "figure": png_name,
        "csv": csv_name,
        "layout": layout,
        "runs": list(run_ids),
        "dataset": cross.dataset_slug(run_ids),
        "methods": order,
        "method_terms": method_term_table(order),
        "baseline": SHORTEST_PING,
        "dist_column": DIST_COLUMN,
        "percentiles": list(PERCENTILES),
        "source_rung": {
            "nside": int(nside),
            "slug": grid_slug(nside),
            "note": (
                "which rung's *_tgs.parquet was read. It does not affect the output: "
                f"{DIST_COLUMN} is prediction-to-TG and is identical at every rung, "
                "which is why neither the figure nor its name carries a grid."
            ),
        },
        "row_policy": (
            "solved rows only, via status.solved_mask -- FALLBACK and ERROR "
            "excluded, S-P's all-BASELINE rows counted as solved. n_plotted equals "
            "accuracy.csv's n_solved; n_failed is the outcome bars' 'no answer'."
        )
        + (
            " Then every unanswered row re-enters at the sentinel, so n_plotted "
            "is the whole roster and n_plotted - n_censored is accuracy.csv's "
            "n_solved."
            if censored
            else ""
        ),
        "distance_policy": (
            f"{DIST_COLUMN}: prediction to the raw TG coordinate. Not "
            "pred_dist_to_seed_km, which moves with the rung because seeds are "
            "regrouped at every grid_km."
        ),
        "percentile_policy": (
            "pandas Series.quantile (linear) rounded to 3dp -- the call "
            "classify.summarize makes, so p50/p90 agree with accuracy.csv."
        )
        + (
            " Taken over the censored sample here, so they do NOT agree with "
            "accuracy.csv: a value at the sentinel means the method had not "
            "answered that share of its TGs, not that it missed by that far."
            if censored
            else ""
        ),
        "unanswered": {
            "policy": policy,
            "sentinel_km": float(sentinel_km) if censored else None,
            "n_censored": {m: int(e.get("n_censored", 0)) for m, e in loaded.items()},
            "n_real_at_or_above_sentinel": {
                m: int(e.get("n_real_at_or_above_sentinel", 0))
                for m, e in loaded.items()
                if e.get("n_real_at_or_above_sentinel", 0)
            },
            "note": (
                "each unanswered TG -- solved_mask rejects, plus answered rows "
                "with no coordinate -- is drawn at sentinel_km, so every curve "
                "shares the roster as its denominator and the height at the "
                "sentinel is the method's answer rate. Real distances at or "
                "past the sentinel would be indistinguishable from censored "
                "ones; the count above is how many there were."
                if censored
                else "unanswered rows are dropped; each curve rests on its own "
                "solved population. This is the file accuracy.csv joins to."
            ),
        },
        "curve_order": (
            "methods.TERM_ORDER -- fixed, so a method holds the same legend row "
            "in every figure. CDF curves cross, so no ranking holds across the "
            "axis and the legend does not assert one; the CSV rows are still "
            "written best-first (p50, then p90, then n_solved, then method id)."
        ),
        "panel": {
            "figsize_in": list(PAPER_FIGSIZE),
            "x_label": X_LABEL,
            "y_label": Y_LABEL,
            "y_ticks": list(Y_TICKS),
            "note": (
                "sized for a paper column: curves, key and axis names only. The "
                "percentiles, the row policy and the method glossary are in this "
                "file and the CSV rather than printed under the axes."
            ),
        },
        "baseline_encoding": (
            "dark grey (_INK_2) and dashed; deliberately not methods.OTHER_HUE, "
            "which any unpublished method takes."
        ),
        "x_axis": {
            "scale": "log",
            "min_km": min_x_km,
            "max_km": max_x_km,
            "sentinel_km": float(sentinel_km) if censored else None,
            "n_clamped_to_floor": _clamped(loaded, min_x_km),
            "clamp_note": (
                f"a log axis cannot render 0, so the drawn curve clamps distances "
                f"below {min_x_km} km up to the floor. The CSV is unclamped."
            ),
        },
        "counts": {m: {k: int(e[k]) for k in COUNT_KEYS} for m, e in loaded.items()},
    }
    if layout == POOLED:
        total = int(table["n_tgs"].max()) if len(table) else 0
        body["pooling"] = {
            "rule": (
                "micro-pool: each run's solved rows concatenated into one "
                "population, so a dataset weighs by its TG count."
            ),
            "runs": per_run or {},
            "n_tgs": total,
            "largest_share": (
                round(max(per_run.values()) / total, 4) if per_run and total else None
            ),
            "percentiles": (
                "true pooled quantiles over the concatenated per-TG distances, "
                "never a mean of the runs' published percentiles."
            ),
            "coverage": (
                "strict -- a method absent from any run is refused, and "
                "overlapping TG ids are refused."
            ),
        }
    return json.dumps(body, indent=2) + "\n"


def _write(
    loaded: dict[str, dict],
    out_dir: Path,
    layout: str,
    *,
    run_ids: list[str],
    nside: int,
    subtitle: str,
    min_x_km: float,
    max_x_km: float,
    policy: str = EXCLUDE,
    sentinel_km: float = SENTINEL_KM,
    per_run: dict[str, int] | None = None,
) -> Path:
    """Table, CSV twin, PNG and manifest for one layout. Returns the PNG.

    `loaded` arrives uncensored; the policy is applied here, once, so the
    table and the curve can never disagree about which rows are in.
    """
    png_name, csv_name, manifest_name = artifact_names(layout, policy)
    drawn = censor(loaded, policy=policy, sentinel_km=sentinel_km)
    mark = sentinel_km if policy == SENTINEL else None
    table = percentile_table(drawn)
    table.insert(0, "run_id", "+".join(sorted(run_ids)))
    table.insert(1, "dataset", cross.dataset_slug(run_ids))
    table["unanswered_policy"] = policy
    table["sentinel_km"] = float(sentinel_km) if policy == SENTINEL else np.nan
    table[[c for c in CSV_COLUMNS if c in table.columns]].to_csv(out_dir / csv_name, index=False)
    png = plot_cdf(
        drawn, table, out_dir / png_name,
        title="Error distance to the TG", subtitle=subtitle,
        min_x_km=min_x_km, max_x_km=max_x_km, sentinel_km=mark,
    )
    (out_dir / manifest_name).write_text(
        _manifest(
            layout, table, drawn, run_ids=run_ids, nside=nside, png_name=png_name,
            csv_name=csv_name, min_x_km=min_x_km, max_x_km=max_x_km,
            policy=policy, sentinel_km=sentinel_km, per_run=per_run,
        )
    )
    return png


def _subtitle_tail(policy: str, sentinel_km: float) -> str:
    """How the panel names its own population, in the subtitle."""
    return (
        f"unanswered at {sentinel_km:,.0f} km"
        if policy == SENTINEL
        else "answered rows only"
    )


def _n_tgs(loaded: dict[str, dict]) -> int:
    """The population behind the panel; every method was scored on one roster."""
    return max((e["n_tgs"] for e in loaded.values()), default=0)


def build_for_run(
    run: RunPaths,
    *,
    nside: int = SOURCE_NSIDE,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    min_x_km: float = X_MIN_KM,
    max_x_km: float | None = None,
    unanswered: str = EXCLUDE,
    sentinel_km: float = SENTINEL_KM,
) -> Path:
    """One run's CDF, written into `classify/`, beside its `healpix-<n>/` rungs."""
    loaded = load_errors(run, nside, methods=methods, analysis_root=analysis_root)
    return _write(
        loaded,
        run.analysis_dir(CLASSIFY_KIND, root=analysis_root),
        PER_RUN,
        run_ids=[run.run_id],
        nside=nside,
        subtitle=(
            f"{run.run_id} · n={_n_tgs(loaded):,} TGs · "
            f"{_subtitle_tail(unanswered, sentinel_km)}"
        ),
        min_x_km=min_x_km,
        max_x_km=resolve_x_max(unanswered, max_x_km, sentinel_km),
        policy=unanswered,
        sentinel_km=sentinel_km,
    )


def build_for_runs(
    runs: list[RunPaths],
    *,
    layouts: tuple[str, ...] = (PER_RUN,),
    nside: int = SOURCE_NSIDE,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    min_x_km: float = X_MIN_KM,
    max_x_km: float | None = None,
    unanswered: str = EXCLUDE,
    sentinel_km: float = SENTINEL_KM,
) -> list[Path]:
    """Render the requested layouts; returns the PNG paths, layout-major.

    `per-run` writes one figure per run into that run's own tree; `pooled`
    writes one into the cross-dataset directory, beside the outcome bars.

    `unanswered` picks the population: `exclude` (the default, and the only one
    that joins to `accuracy.csv`) or `sentinel`, which parks unanswered TGs at
    `sentinel_km` so every curve shares a denominator. The two write different
    filenames, so building both leaves both on disk.
    """
    ordered = tuple(dict.fromkeys(layouts)) or (PER_RUN,)
    unknown = [x for x in ordered if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layout {unknown}; pick from {list(LAYOUTS)}")
    if unanswered not in UNANSWERED_POLICIES:
        raise ValueError(
            f"unknown unanswered policy {unanswered!r}; pick from {list(UNANSWERED_POLICIES)}"
        )
    nside = G.validate_nside(nside)
    max_x_km = resolve_x_max(unanswered, max_x_km, sentinel_km)
    run_ids = [r.run_id for r in runs]

    out: list[Path] = []
    for layout in ordered:
        if layout == PER_RUN:
            out.extend(
                build_for_run(
                    run, nside=nside, methods=methods, analysis_root=analysis_root,
                    min_x_km=min_x_km, max_x_km=max_x_km,
                    unanswered=unanswered, sentinel_km=sentinel_km,
                )
                for run in runs
            )
            continue
        by_run = load_per_run(runs, nside, methods=methods, analysis_root=analysis_root)
        loaded = pool(by_run)
        per_run = {rid: _n_tgs(e) for rid, e in by_run.items()}
        label = cross.dataset_slug(run_ids).upper()
        out.append(
            _write(
                loaded,
                cross.cross_dir(run_ids, analysis_root=analysis_root),
                POOLED,
                run_ids=run_ids,
                nside=nside,
                subtitle=(
                    f"{label} pooled · n={_n_tgs(loaded):,} TGs · "
                    f"{_subtitle_tail(unanswered, sentinel_km)}"
                ),
                min_x_km=min_x_km,
                max_x_km=max_x_km,
                policy=unanswered,
                sentinel_km=sentinel_km,
                per_run=per_run,
            )
        )
    return out
