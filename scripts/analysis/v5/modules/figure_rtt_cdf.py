"""The RTT distribution of each dataset, as one CDF per dataset on one axis.

Every other v5 figure describes what a *method* did. This one describes the
**input**: for an edge in this mesh, what RTT did it carry? No method, no
cohort, no answer space -- one curve per dataset, drawn from the canonical
`(vp_id, target_id, rtt_ms)` CSV through `edges.load_min_rtt`.

It exists because the LTD is the only part of a CBG stack that touches RTT
directly, and every finding about LTD behaviour -- inflation, the bounded
spline's bin occupancy, where `speed_of_internet` starts refusing -- is a
statement about where these curves sit. A reader comparing as01, as02 and as03
results needs to know first whether the three meshes even present the same
latency, and on these three they very nearly do.

## Why one curve per dataset and not one pooled curve

The sibling cross figures pool, and guard the pooling with
`cross.guard_disjoint_tgs`, because they report a single micro-averaged
number whose denominator must be one population. Nothing here is pooled: each
dataset keeps its own curve and its own denominator, so a shared TG id would
not corrupt anything -- it would appear on two curves, which is what a reader
comparing two meshes that share a target would want to see. The guard is
deliberately **not** called; see `TestNoPoolingGuard`.

## The x cut is a real cut, and the manifest says how much it hides

A linear axis truncated at `x_max` looks identical whether the tail beyond it
holds nothing or holds a third of the population -- the curve simply runs off
the right edge below 100%. That is the failure mode this module's manifest
exists to close: `share_within_xmax_pct` and `max_ms` are recorded per dataset,
and when any dataset has mass beyond the cut the figure marks it in-panel
rather than leaving the reader to assume the curve completed.

On as01-03 at the default 200 ms cut this never fires: the largest RTT in any
of the three meshes is 92.4 ms, so 200 ms wastes more than half the axis.
The default is kept anyway because it is the cut the RTT literature reports
at, and a figure whose bounds move with its data cannot be compared against
the next dataset. Pass `--x-max` to zoom.

## Edges, not targets -- and the replica inflation that follows

The unit here is an **edge**, not a TG: a TG measured by 134 VPs contributes
134 observations. That is the right unit for an RTT distribution, but it
carries the replica structure with it. ~20 IP replicas share a site and so
share its VP geometry, so the curve is roughly a 20x oversample of each
site's latency profile, and a site with more replicas pulls the curve toward
its own RTTs. `n_tgs`, `n_vps` and `n_distinct` are in the CSV so the
oversampling is visible; do not read a 2 ms shift between datasets as a
routing difference without checking that their site counts match.

Command: `plot-rtt-cdf`. Writes `_cross/rtt-cdf/<datasets>[@<arm>]/`.
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

from scripts.analysis.v5.modules import cross, edges  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

KIND = "rtt-cdf"

PNG_NAME = "rtt_cdf.png"
CSV_NAME = "rtt_cdf.csv"
MANIFEST_NAME = "rtt_cdf.manifest.json"

#: Default right edge, in ms. See the module docstring: 200 ms is the cut the
#: RTT literature reports at, and a fixed bound is what makes two runs of this
#: figure comparable. It is generous for the as0X meshes on purpose.
DEFAULT_X_MAX_MS = 200.0

#: X scales this figure will draw. Plain `log` is absent deliberately: it is a
#: legitimate choice for RTT and would work on these meshes, but it is one
#: dataset away from silently dropping rows. `load_canonical_csv` already
#: refuses `rtt_ms <= 0`, so nothing here is zero *today*; a future source that
#: rounds sub-millisecond RTTs to 0 would be filtered upstream rather than
#: visibly missing from the curve. `symlog` keeps a linear window over the
#: small values and stays defined at 0, so it degrades honestly.
X_SCALES = ("linear", "symlog")
DEFAULT_X_SCALE = "linear"

#: Tick spacing on a linear x axis, in ms. None leaves matplotlib's automatic
#: locator, which picks a round step from the data range. Ignored on `symlog`,
#: where a constant spacing is meaningless -- the decades are the ticks.
DEFAULT_X_STEP_MS: float | None = None

#: Where `symlog` hands over from linear to logarithmic, in ms. 1 ms because
#: that is the floor of a plausible RTT -- the smallest observation on the
#: as0X meshes is 0.58 ms -- so the linear window holds the handful of
#: same-facility edges and the log region holds everything a reader compares.
#: Set it above the bulk of the data and the "log" axis is linear in disguise;
#: `share_below_linthresh_pct` in the manifest is the check.
DEFAULT_LINTHRESH_MS = 1.0

#: Reported in the CSV and in the in-panel legend. `min` and `max` are emitted
#: alongside rather than as p0/p100, since a reader asking for the extremes is
#: not asking for an interpolated quantile.
PERCENTILES = (5, 10, 25, 50, 75, 90, 95, 99)

#: `dataset` head -> `(legend label, colour, linestyle)`.
#:
#: The label cannot be derived: `as7018` is "RIPE MIX-ASN" because its VPs are
#: RIPE probes spread across many ASNs, while `as01-03` are one operator's own
#: single-AS meshes, and no part of a run id says so. Hence a table, and hence
#: `_style`'s fallback for a dataset that is not in it.
#:
#: The operator meshes are **solid**, so hue alone separates them -- which
#: rules out the blue/orange/green trio, since orange against green is
#: precisely the pair deuteranopes lose. Blue/orange/purple stays separable
#: under deuteranopia and in greyscale, because the three also differ in
#: lightness. The RIPE mesh is grey and dashed on both counts at once: it is a
#: different kind of dataset, not a fourth operator mesh, and the reference
#: curve should read as a reference.
DISPLAY = {
    "as01": ("PRO AS01", "#1f6fd0", "-"),
    "as02": ("PRO AS02", "#e8762c", "-"),
    "as03": ("PRO AS03", "#7b52ab", "-"),
    "as7018": ("RIPE MIX-ASN", "#8a8a8a", "--"),
}

#: Colours for a dataset absent from `DISPLAY`, cycled in first-appearance
#: order. Dotted, so an unlabelled series is visibly not one of the named ones
#: rather than silently borrowing a named one's look.
_FALLBACK_HUES = ("#2f8f4e", "#b3123b", "#0f7d7d", "#8a5a00")


def _style(dataset: str, index: int) -> tuple[str, str, str]:
    """`(label, colour, linestyle)` for one dataset.

    Falls back to the uppercased head rather than raising: a new mesh should
    draw and be obviously unstyled, not stop the figure.
    """
    if dataset in DISPLAY:
        return DISPLAY[dataset]
    return (dataset.upper(), _FALLBACK_HUES[index % len(_FALLBACK_HUES)], ":")


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/rtt-cdf/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def load(
    runs: list[RunPaths], *, source_csv: dict[str, Path] | None = None
) -> tuple[pd.DataFrame, dict]:
    """`(run_id, dataset, tg_id, vp_id, rtt_ms)` for every edge in every run.

    Reads each run's canonical CSV through `edges.load_min_rtt`, so the rows
    are the observations the benchmark itself ran on -- same NaN and
    `rtt_ms <= 0` drops, same minimum-per-pair deduplication -- and a
    traffic-weighted arm still raises `MeshSupersetError` rather than
    quietly describing the mesh it was pruned from.

    `dataset` is carried alongside `run_id` because it is what the legend and
    the CSV key on, and deriving it twice is how the two drift apart.
    """
    if not runs:
        raise ValueError("pass at least one run")
    overrides = source_csv or {}
    frames = []
    for run in runs:
        df = edges.load_min_rtt(run, source_csv=overrides.get(run.run_id))
        df.insert(0, "dataset", cross.short_dataset(run.run_id))
        df.insert(0, "run_id", run.run_id)
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    meta = {
        "run_ids": [r.run_id for r in runs],
        "datasets": [cross.short_dataset(r.run_id) for r in runs],
    }
    return long, meta


def stats_table(
    long: pd.DataFrame,
    *,
    x_max_ms: float = DEFAULT_X_MAX_MS,
    linthresh_ms: float = DEFAULT_LINTHRESH_MS,
) -> pd.DataFrame:
    """One row per dataset: extrema, percentiles, and what the x cut hides.

    Row order follows first appearance in `long`, not sort order, so the CSV,
    the legend and the caller's `--run-id` order are the same sequence.
    """
    rows = []
    for run_id in long["run_id"].drop_duplicates():
        sub = long[long["run_id"] == run_id]
        v = sub["rtt_ms"].to_numpy()
        q = np.percentile(v, PERCENTILES)
        rows.append(
            {
                "run_id": run_id,
                "dataset": sub["dataset"].iloc[0],
                "n_edges": int(len(v)),
                "n_tgs": int(sub["tg_id"].nunique()),
                "n_vps": int(sub["vp_id"].nunique()),
                "n_distinct": int(len(np.unique(v))),
                "min_ms": float(v.min()),
                **{f"p{p}_ms": float(x) for p, x in zip(PERCENTILES, q)},
                "max_ms": float(v.max()),
                "mean_ms": float(v.mean()),
                "share_within_xmax_pct": float((v <= x_max_ms).mean() * 100),
                "share_below_linthresh_pct": float((v < linthresh_ms).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def validate_axis(
    x_scale: str,
    *,
    x_max_ms: float,
    linthresh_ms: float,
    x_step_ms: float | None = None,
) -> str:
    """Reject an x axis before anything is read off disk. Returns the scale.

    The linthresh check is the substantive one: a `symlog` axis whose linear
    window swallows the data is a log axis in name only, and reads as one, so
    a threshold at or above `x_max` is refused rather than drawn.
    """
    if x_scale not in X_SCALES:
        raise ValueError(f"--x-scale must be one of {list(X_SCALES)}, got {x_scale!r}")
    if x_max_ms <= 0:
        raise ValueError(f"--x-max must be positive, got {x_max_ms}")
    if x_step_ms is not None:
        if x_step_ms <= 0:
            raise ValueError(f"--x-step must be positive, got {x_step_ms}")
        if x_step_ms > x_max_ms:
            raise ValueError(
                f"--x-step {x_step_ms} exceeds --x-max {x_max_ms}, so the axis "
                f"would carry a single tick at 0"
            )
    if x_scale == "symlog":
        if linthresh_ms <= 0:
            raise ValueError(f"--linthresh must be positive, got {linthresh_ms}")
        if linthresh_ms >= x_max_ms:
            raise ValueError(
                f"--linthresh {linthresh_ms} is at or above --x-max {x_max_ms}, so "
                f"the whole axis would be the linear window and 'symlog' would "
                f"draw a linear axis under a log label"
            )
    return x_scale


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values against the **fraction** of observations at or below each.

    A fraction in [0, 1], not a percentage. The drawn axis is the CDF itself,
    which is what the y label says, so the curve carries the same units as the
    label. The stats CSV and the manifest still speak in percent -- those are
    prose numbers a reader quotes ("99.8% of edges"), not axis coordinates.
    """
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def clipped(stats: pd.DataFrame) -> pd.DataFrame:
    """The datasets with mass beyond the x cut, if any.

    Separated from `plot` so the same verdict reaches the figure, the manifest
    and a test without three roundings of the same comparison.
    """
    return stats[stats["share_within_xmax_pct"] < 100.0 - 1e-9]


def clip_notes(stats: pd.DataFrame) -> list[tuple[str, str]]:
    """`(text, colour)` per clipped dataset, in drawn order. Empty when none.

    One note per dataset rather than one worst-case sentence for the panel.
    The earlier wording ("up to 1.5% of a dataset's edges lie beyond it") named
    no dataset, so on a four-curve figure a reader could not tell *which* curve
    stops short -- and that is the only thing they need in order to read the
    panel correctly.

    Each note carries the dataset's **p99** rather than its max. The max is one
    observation and says nothing about the shape of what was cut; a p99 above
    the axis is the statement that the truncation is structural rather than a
    single outlier. `observed_max_ms` stays in the manifest for whoever wants
    the extreme.

    Coloured to match the curve, so the note attaches to a line without a
    second legend. Pure, and separate from `plot`, so the wording is testable
    without reading artists back off a closed figure.
    """
    notes = []
    for i, row in enumerate(stats.itertuples()):
        if row.share_within_xmax_pct >= 100.0 - 1e-9:
            continue
        label, colour, _ = _style(row.dataset, i)
        notes.append((f"{label}  (p99: {row.p99_ms:.1f} ms)", colour))
    return notes


#: Where each dataset's median label sits, as a CDF value. Stepped down by
#: index rather than placed at 0.5 for all four: the medians on the as0X+RIPE
#: set span 34.7-42.0 ms, which on a symlog axis is a few millimetres, so four
#: labels at one height would overprint. The ladder is what keeps them legible
#: and also makes the reading order match the legend.
_MEDIAN_LABEL_TOP = 0.46
_MEDIAN_LABEL_STEP = 0.075


def label_anchor(stats: pd.DataFrame, *, x_max_ms: float) -> float | None:
    """x for the median labels: just right of the **rightmost** drawn median.

    One anchor for all of them, not `p50 * 1.05` each. Per-label anchoring
    reads fine with one dataset and fails with four: the medians on the
    as0X+RIPE set span 34.7-42.0 ms, so every label but the last one lands on
    top of a later dataset's dropline. A shared anchor puts them in a column
    clear of all four lines, which also lets them be read as a small table.

    None when no median is drawable.
    """
    drawable = [p for p in stats["p50_ms"] if 0 < p <= x_max_ms]
    return max(drawable) * 1.08 if drawable else None


def _median_dropline(ax, row, colour: str, *, index: int, x_max_ms: float,
                     anchor: float | None) -> None:
    """A dashed vertical at this dataset's p50, with the value in the column.

    Drawn from the axis floor up to 0.5, where the median meets its own curve
    by definition, so the line terminates on the thing it describes instead of
    crossing the whole panel.

    A median beyond the x cut is skipped rather than clamped to the edge: a
    label pinned at `x_max` reads as "the median is x_max", which is a
    stronger and wronger claim than showing nothing.
    """
    p50 = row.p50_ms
    if not (0 < p50 <= x_max_ms):
        return
    ax.vlines(p50, 0, 0.5, color=colour, ls=(0, (4, 2)), lw=0.9, zorder=1)
    if anchor is not None:
        ax.text(
            anchor,
            _MEDIAN_LABEL_TOP - _MEDIAN_LABEL_STEP * index,
            f"{p50:.1f} ms",
            color=colour, fontsize=6.5, ha="left", va="center", zorder=4,
        )


def plot(
    long: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    meta: dict,
    out_png: Path,
    x_max_ms: float = DEFAULT_X_MAX_MS,
    x_scale: str = DEFAULT_X_SCALE,
    linthresh_ms: float = DEFAULT_LINTHRESH_MS,
    x_step_ms: float | None = DEFAULT_X_STEP_MS,
    show_medians: bool = True,
    annotate_clipping: bool = True,
) -> Path:
    """One step curve per dataset, sized as a small descriptive panel.

    The legend carries dataset names only. `n` is in the stats CSV -- on a
    figure this size the parenthetical ran wider than the panel, and a legend
    is for identifying curves rather than for tabulating them. The median
    comes back onto the panel as a dropline instead, where it sits at its own
    x position rather than in a list.

    A dataset whose distribution runs past `x_max` gets a note in the top
    right, in its own colour, carrying its p99. That corner is free under both
    legend placements. `--no-annotate-clipping` suppresses them, which leaves
    a curve able to stop short of 1.0 with nothing on the panel saying why --
    the manifest still records it, but the caption then has to.
    """
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    anchor = label_anchor(stats, x_max_ms=x_max_ms) if show_medians else None

    for i, row in enumerate(stats.itertuples()):
        v = long.loc[long["run_id"] == row.run_id, "rtt_ms"].to_numpy()
        x, y = _ecdf(v)
        label, colour, linestyle = _style(row.dataset, i)
        ax.step(x, y, where="post", color=colour, ls=linestyle, lw=1.4, label=label)
        if show_medians:
            _median_dropline(ax, row, colour, index=i, x_max_ms=x_max_ms,
                             anchor=anchor)

    # A linear axis truncated mid-distribution looks exactly like one that
    # captured everything, so say so in the panel when it did not. The
    # manifest carries the per-dataset shares; this is only the flag.
    if show_medians:
        # The droplines stop at 0.5, so say what 0.5 is. Without it the four
        # verticals read as arbitrary marks rather than as medians.
        ax.axhline(0.5, color="#b9b9b9", lw=0.6, ls=(0, (1, 2)), zorder=0)

    if annotate_clipping:
        # **Above** the axes, right-aligned, not inside the top right corner.
        # That corner looks free and is not: every curve that does reach 1.0
        # runs flat along the top edge from its maximum to x_max, so a note
        # placed there lands on the very lines it is describing. Outside the
        # frame there is nothing to collide with and the notes read as a
        # caption fragment, which is what they are.
        notes = clip_notes(stats)
        for k, (text, colour) in enumerate(notes):
            ax.text(1.0, 1.03 + 0.085 * (len(notes) - 1 - k), text,
                    transform=ax.transAxes, fontsize=6.5, ha="right",
                    va="bottom", color=colour)

    if x_scale == "symlog":
        ax.set_xscale("symlog", linthresh=linthresh_ms, linscale=0.35)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(
            ticker.SymmetricalLogLocator(base=10, linthresh=linthresh_ms,
                                         subs=list(range(2, 10)))
        )
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif x_step_ms:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step_ms))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, x_max_ms)
    ax.set_ylim(0, 1)
    ax.set_xlabel("RTT (ms)", fontsize=8)
    ax.set_ylabel("CDF", fontsize=8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    # `:g` would print "0.25" and "0.5"; a CDF axis reads as a ladder, so the
    # ticks are held to a common two-decimal width.
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(labelsize=7.5, length=3)
    ax.grid(alpha=0.25, lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Placement follows the scale, not taste. On a linear axis the CDF leaves
    # the lower right empty; symlog stretches the sub-10 ms region across the
    # left half and drags the curves' rise into exactly that corner, so a fixed
    # "lower right" puts the legend on top of the data.
    ax.legend(loc="upper left" if x_scale == "symlog" else "lower right",
              fontsize=7, frameon=False,
              handlelength=2.2, borderaxespad=0.3, labelspacing=0.35)

    fig.tight_layout(pad=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _manifest(
    meta: dict,
    stats: pd.DataFrame,
    *,
    x_max_ms: float,
    x_scale: str = DEFAULT_X_SCALE,
    linthresh_ms: float = DEFAULT_LINTHRESH_MS,
    x_step_ms: float | None = DEFAULT_X_STEP_MS,
    annotate_clipping: bool = True,
) -> str:
    cut = clipped(stats)
    return json.dumps(
        {
            "figure": PNG_NAME,
            "csv": CSV_NAME,
            "kind": KIND,
            "run_ids": meta["run_ids"],
            "datasets": meta["datasets"],
            "n_edges": {r.dataset: int(r.n_edges) for r in stats.itertuples()},
            "scope": (
                "population: one row per (vp, tg) edge at its minimum RTT, read "
                "from each run's canonical CSV. No method, no cohort, no "
                "solved_mask -- this describes the input, not an output. Each "
                "dataset keeps its own denominator; nothing is pooled."
            ),
            "x_axis": {
                "scale": x_scale,
                "min_ms": 0.0,
                "max_ms": float(x_max_ms),
                "tick_step_ms": float(x_step_ms) if x_step_ms else None,
                "share_within_xmax_pct": {
                    r.dataset: float(r.share_within_xmax_pct) for r in stats.itertuples()
                },
                "observed_max_ms": {
                    r.dataset: float(r.max_ms) for r in stats.itertuples()
                },
                "clipped_datasets": sorted(cut["dataset"]),
                "annotated_on_figure": bool(annotate_clipping),
                **(
                    {
                        "symlog": {
                            "linthresh_ms": float(linthresh_ms),
                            "share_below_linthresh_pct": {
                                r.dataset: float(r.share_below_linthresh_pct)
                                for r in stats.itertuples()
                            },
                            "note": (
                                "below linthresh the axis is linear, above it "
                                "logarithmic. Check share_below_linthresh_pct: if "
                                "a large share of a dataset sits in the linear "
                                "window, that dataset's curve is not being read on "
                                "a log axis however the axis is labelled."
                            ),
                        }
                    }
                    if x_scale == "symlog"
                    else {}
                ),
                "note": (
                    "a linear axis truncated mid-distribution is visually "
                    "identical to one that captured everything -- the curve "
                    "just runs off the right edge below 100%. Cite "
                    "share_within_xmax_pct, not the picture. annotated_on_figure "
                    "says whether the panel itself warns; when it is false and "
                    "clipped_datasets is non-empty, the caption is the only "
                    "place a reader can learn that the axis truncated."
                ),
            },
            "unit": {
                "n_tgs": {r.dataset: int(r.n_tgs) for r in stats.itertuples()},
                "n_vps": {r.dataset: int(r.n_vps) for r in stats.itertuples()},
                "note": (
                    "the unit is an edge, so a TG measured by 134 VPs "
                    "contributes 134 observations and ~20 IP replicas per site "
                    "oversample that site's latency profile ~20x. A shift of a "
                    "few ms between datasets can be a difference in site mix "
                    "rather than in routing; check n_tgs before reading it as "
                    "the latter."
                ),
            },
            "pooling": (
                "cross.guard_disjoint_tgs is deliberately NOT called. Nothing "
                "here shares a denominator across runs, so a TG present in two "
                "datasets belongs on both curves rather than being an error."
            ),
        },
        indent=2,
    )


def build_for_runs(
    runs: list[RunPaths],
    *,
    x_max_ms: float = DEFAULT_X_MAX_MS,
    x_scale: str = DEFAULT_X_SCALE,
    linthresh_ms: float = DEFAULT_LINTHRESH_MS,
    x_step_ms: float | None = DEFAULT_X_STEP_MS,
    show_medians: bool = True,
    annotate_clipping: bool = True,
    analysis_root: Path | None = None,
    source_csv: dict[str, Path] | None = None,
) -> list[Path]:
    """PNG, stats CSV and manifest. Returns the PNG in a list, as siblings do."""
    x_scale = validate_axis(x_scale, x_max_ms=x_max_ms, linthresh_ms=linthresh_ms,
                            x_step_ms=x_step_ms)
    long, meta = load(runs, source_csv=source_csv)
    stats = stats_table(long, x_max_ms=x_max_ms, linthresh_ms=linthresh_ms)

    out_dir = output_dir(meta["run_ids"], analysis_root=analysis_root)
    stats.to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(
        _manifest(meta, stats, x_max_ms=x_max_ms, x_scale=x_scale,
                  linthresh_ms=linthresh_ms, x_step_ms=x_step_ms,
                  annotate_clipping=annotate_clipping)
    )
    return [
        plot(long, stats, meta=meta, out_png=out_dir / PNG_NAME, x_max_ms=x_max_ms,
             x_scale=x_scale, linthresh_ms=linthresh_ms, x_step_ms=x_step_ms,
             show_medians=show_medians, annotate_clipping=annotate_clipping)
    ]
