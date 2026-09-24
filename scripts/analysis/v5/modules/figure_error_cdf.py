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

#: Reported per method in the box under the legend. 50 and 90 are the two
#: `accuracy.csv` publishes, so this CSV joins to it on its headline numbers.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 90, 95)

#: Reference verticals (km). Neutral ink, **not** green/red: green is the
#: Octant family and red is SPO, so coloured guides would read as series.
THRESHOLDS_KM: tuple[int, ...] = (100, 500, 1000)

#: Fixed x range, both bounds, so per-run and pooled figures share one axis.
#: The floor is 0.1 km because sub-kilometre errors are not rare (observed
#: minimum on as01-03: 0.106 km); the manifest records any clamped count.
X_MIN_KM = 0.1
DEFAULT_X_MAX_KM = 10_000.0

PER_RUN = "per-run"
POOLED = "pooled"
LAYOUTS: tuple[str, ...] = (PER_RUN, POOLED)

#: `layout -> (png, csv, manifest)`. No `{slug}`: the figure does not vary with
#: the rung, so the name does not pretend it might.
NAMES: dict[str, tuple[str, str, str]] = {
    PER_RUN: ("error_cdf.png", "error_cdf.csv", "error_cdf.manifest.json"),
    POOLED: (
        "error_cdf.pooled.png",
        "error_cdf.pooled.csv",
        "error_cdf.pooled.manifest.json",
    ),
}

#: Which rung's parquets to read. Any would do -- that is the point -- so the
#: finest is the default and `--nside` exists to demonstrate nothing changes.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: Summed when runs are pooled.
COUNT_KEYS: tuple[str, ...] = ("n_tgs", "n_solved", "n_failed", "n_no_distance")

#: The CDF's way out of a pooling refusal (`--layout compare` is the outcome
#: bars' and does not exist here).
REMEDY_COMMON = "--layout per-run to keep each dataset on its own figure."
REMEDY_DISJOINT = "Use --layout per-run, which keeps each dataset on its own figure."

_NO_VALUE = "—"

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

#: The row and distance policies, printed under every layout's axes.
FOOTNOTE = (
    "Unanswered rows excluded: a FALLBACK coordinate is the shortest-ping VP's, "
    "so its error is the baseline's wearing a method's name.\nThose are the "
    "outcome bars' “no answer” segment — the n above is that figure's "
    "answered stack. Distance is prediction to the raw\nTG; neither grid nor "
    "seed enters it, which is why this figure is the same at every rung."
)


# ---- loading ----------------------------------------------------------------


def scored_methods(cls_dir: Path) -> list[str]:
    """Method ids with a `*_tgs.parquet` in `cls_dir`. Ranked later, by p50."""
    suffix = C.TGS_PARQUET.format(method="")
    return sorted(p.name[: -len(suffix)] for p in Path(cls_dir).glob(f"*{suffix}"))


def _load_tgs(
    run: RunPaths, method: str, nside: int, *, analysis_root: Path | None = None
) -> pd.DataFrame:
    """One run's per-TG scored rows for one method, three columns wide."""
    path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=method)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(path, columns=list(READ_COLUMNS))


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
    digit.
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
        }
        for p in PERCENTILES:
            row[pcol(p)] = round(float(values.quantile(p / 100)), 3) if len(values) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(list(_RANK_KEYS), ascending=list(_RANK_ASC)).reset_index(drop=True)


def curve_order(table: pd.DataFrame) -> list[str]:
    """The ranked method ids -- legend order, and the percentile box's."""
    return list(table["method"])


# ---- drawing ----------------------------------------------------------------


def _cdf(values: np.ndarray, min_x_km: float = X_MIN_KM):
    """`(x, y)` for an empirical CDF, x clamped up to the log floor.

    The clamp is a rendering concession applied here only; `percentile_table`
    reads the unclamped values.
    """
    xs = np.sort(np.maximum(values, min_x_km))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def _draw_guides(ax, min_x_km: float, max_x_km: float) -> None:
    """The 100/500/1,000 km verticals and the median line, under everything."""
    for km in THRESHOLDS_KM:
        if min_x_km < km < max_x_km:
            ax.axvline(km, color=_GRID, linestyle=":", linewidth=1.2, zorder=1)
            # Rotated under the top spine: on a log axis 500 and 1,000 km are a
            # third of a decade apart, so horizontal labels collide.
            ax.annotate(
                f"{km}", xy=(km, 0.995), xytext=(-3, 0), textcoords="offset points",
                fontsize=7.5, color=_MUTED, ha="right", va="top", rotation=90, zorder=1,
            )
    ax.axhline(0.5, color=_GRID, linestyle="--", linewidth=1.2, zorder=1)


def _style_axes(ax, min_x_km: float, max_x_km: float) -> None:
    """Log x on the fixed range, 0-1 y, and the package's quiet spines."""
    from matplotlib.ticker import FuncFormatter

    ax.set_xscale("log")
    # `%g`: a ScalarFormatter renders the 0.1 km decade as a bare "0".
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(min_x_km, max_x_km)
    ax.set_ylim(0, 1)
    ax.set_xlabel("pred_dist_to_tg (km)", fontsize=10.5, color=_INK_2)
    ax.set_ylabel("Fraction of TGs answered", fontsize=10.5, color=_INK_2)
    ax.grid(True, which="both", color=_GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelsize=9)


def _curve_style(method: str, colors: dict[str, str]) -> dict:
    """Colour is the method; the baseline is recessive grey and dashed."""
    if method == SHORTEST_PING:
        # _INK_2, not _MUTED: _MUTED is byte-for-byte `methods.OTHER_HUE`.
        return {"color": _INK_2, "linestyle": "--", "linewidth": 2.2, "zorder": 3}
    return {"color": colors[method], "linestyle": "-", "linewidth": 2.0, "zorder": 2}


def _percentile_box(ax, table: pd.DataFrame, legend) -> None:
    """The monospace percentile table, hung under `legend`, in legend order.

    Needs a drawn canvas, so the legend's extent is real.
    """
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    header = f"{'':<8}{'plotted':>11}" + "".join(f"{'p' + str(p):>7}" for p in PERCENTILES)
    lines = [header]
    for _, row in table.iterrows():
        counts = f"{int(row['n_plotted'])}/{int(row['n_tgs'])}"
        cells = "".join(
            f"{row[pcol(p)]:>7.0f}" if np.isfinite(row[pcol(p)]) else f"{_NO_VALUE:>7}"
            for p in PERCENTILES
        )
        lines.append(f"{row['method_label'][:8]:<8}{counts:>11}{cells}")
    ax.text(
        0.02, box.ymin - 0.03, "\n".join(lines), transform=ax.transAxes,
        fontsize=7, va="top", ha="left", color=_INK_2, family="monospace",
        bbox=dict(boxstyle="round", facecolor=_SURFACE, edgecolor=_GRID, alpha=0.95),
    )


def plot_cdf(
    loaded: dict[str, dict],
    table: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    min_x_km: float = X_MIN_KM,
    max_x_km: float = DEFAULT_X_MAX_KM,
    figsize: tuple[float, float] = (8.6, 6.6),
    dpi: int = 200,
) -> Path:
    """One panel, one curve per method, log x. Both layouts draw through here.

    Honest limit: CDF curves **cross**, so no single ranking holds across the
    axis. The legend is ordered by p50 and the percentile box beneath it shows
    p5..p95 in the same order, so the reader sees where the order stops holding.
    """
    from matplotlib.lines import Line2D

    order = curve_order(table)
    colors = method_colors(order)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    _draw_guides(ax, min_x_km, max_x_km)

    # Baseline last, so the dashed reference reads on top. Draw order is not
    # legend order, hence the explicit handles below.
    variants = [m for m in order if m != SHORTEST_PING]
    for method in variants + [m for m in order if m == SHORTEST_PING]:
        values = loaded[method]["errors"]
        if len(values) == 0:
            continue
        xs, ys = _cdf(values, min_x_km)
        ax.plot(xs, ys, alpha=0.95, gid=method, **_curve_style(method, colors))

    _style_axes(ax, min_x_km, max_x_km)

    handles = [
        Line2D(
            [], [],
            label=f"{method_label(m)} (baseline)" if m == SHORTEST_PING else method_label(m),
            **{k: v for k, v in _curve_style(m, colors).items() if k != "zorder"},
        )
        for m in order
    ]
    legend = ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(_INK_2)

    ax.set_title(title, fontsize=13, fontweight="bold", color=_INK, pad=16)
    ax.annotate(
        subtitle, xy=(0.5, 1.005), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=9.5, color=_INK_2,
    )
    # The term lookup, so the figure reads without the paper beside it.
    entries = [f"{t}: {name}" for t, name in method_term_table(order).items()]
    half = (len(entries) + 1) // 2
    terms = "\n".join("   ·   ".join(r) for r in (entries[:half], entries[half:]) if r)
    ax.annotate(
        f"{terms}\n\n{FOOTNOTE}", xy=(0.5, -0.135), xycoords="axes fraction",
        ha="center", va="top", fontsize=7.5, color=_MUTED,
    )

    fig.canvas.draw()
    _percentile_box(ax, table, legend)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


# ---- artifacts --------------------------------------------------------------

#: The CSV twin's columns. `run_id` and `dataset` lead so per-run and pooled
#: files concatenate into one frame.
CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "method", "method_label", "is_baseline",
    *COUNT_KEYS, "n_plotted", *[pcol(p) for p in PERCENTILES],
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
    per_run: dict[str, int] | None = None,
) -> str:
    order = curve_order(table)
    body: dict = {
        "figure": png_name,
        "csv": csv_name,
        "layout": layout,
        "runs": list(run_ids),
        "dataset": cross.dataset_slug(run_ids),
        "arm": cross.arm(run_ids),
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
        ),
        "distance_policy": (
            f"{DIST_COLUMN}: prediction to the raw TG coordinate. Not "
            "pred_dist_to_seed_km, which moves with the rung because seeds are "
            "regrouped at every grid_km."
        ),
        "percentile_policy": (
            "pandas Series.quantile (linear) rounded to 3dp -- the call "
            "classify.summarize makes, so p50/p90 agree with accuracy.csv."
        ),
        "curve_order": (
            "p50 ascending, then p90, then n_solved descending, then method id. "
            "CDF curves cross, so this is the order they reach the median line; "
            "the percentile box shows p5..p95 in the same order."
        ),
        "baseline_encoding": (
            "dark grey (_INK_2) and dashed; deliberately not methods.OTHER_HUE, "
            "which any unpublished method takes."
        ),
        "x_axis": {
            "scale": "log",
            "min_km": min_x_km,
            "max_km": max_x_km,
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
    per_run: dict[str, int] | None = None,
) -> Path:
    """Table, CSV twin, PNG and manifest for one layout. Returns the PNG."""
    png_name, csv_name, manifest_name = NAMES[layout]
    table = percentile_table(loaded)
    table.insert(0, "run_id", "+".join(sorted(run_ids)))
    table.insert(1, "dataset", cross.dataset_slug(run_ids))
    table[[c for c in CSV_COLUMNS if c in table.columns]].to_csv(out_dir / csv_name, index=False)
    png = plot_cdf(
        loaded, table, out_dir / png_name,
        title="Error distance to the TG", subtitle=subtitle,
        min_x_km=min_x_km, max_x_km=max_x_km,
    )
    (out_dir / manifest_name).write_text(
        _manifest(
            layout, table, loaded, run_ids=run_ids, nside=nside, png_name=png_name,
            csv_name=csv_name, min_x_km=min_x_km, max_x_km=max_x_km, per_run=per_run,
        )
    )
    return png


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
    max_x_km: float = DEFAULT_X_MAX_KM,
) -> Path:
    """One run's CDF, written into `classify/`, beside its `healpix-<n>/` rungs."""
    loaded = load_errors(run, nside, methods=methods, analysis_root=analysis_root)
    return _write(
        loaded,
        run.analysis_dir(CLASSIFY_KIND, root=analysis_root),
        PER_RUN,
        run_ids=[run.run_id],
        nside=nside,
        subtitle=f"{run.run_id} · n={_n_tgs(loaded):,} TGs · answered rows only",
        min_x_km=min_x_km,
        max_x_km=max_x_km,
    )


def build_for_runs(
    runs: list[RunPaths],
    *,
    layouts: tuple[str, ...] = (PER_RUN,),
    nside: int = SOURCE_NSIDE,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    min_x_km: float = X_MIN_KM,
    max_x_km: float = DEFAULT_X_MAX_KM,
) -> list[Path]:
    """Render the requested layouts; returns the PNG paths, layout-major.

    `per-run` writes one figure per run into that run's own tree; `pooled`
    writes one into the cross-dataset directory, beside the outcome bars.
    """
    ordered = tuple(dict.fromkeys(layouts)) or (PER_RUN,)
    unknown = [x for x in ordered if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layout {unknown}; pick from {list(LAYOUTS)}")
    nside = G.validate_nside(nside)
    run_ids = [r.run_id for r in runs]

    out: list[Path] = []
    for layout in ordered:
        if layout == PER_RUN:
            out.extend(
                build_for_run(
                    run, nside=nside, methods=methods, analysis_root=analysis_root,
                    min_x_km=min_x_km, max_x_km=max_x_km,
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
                subtitle=f"{label} pooled · n={_n_tgs(loaded):,} TGs · answered rows only",
                min_x_km=min_x_km,
                max_x_km=max_x_km,
                per_run=per_run,
            )
        )
    return out
