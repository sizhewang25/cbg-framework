"""How far out each method is on the targets only *it* got right.

The companion to `figure_error_diff`, and its complement: that one compares
the two where both placed the target in the correct cell, this one looks at
the two cohorts where exactly one did. Together with the targets neither
reached they partition every evaluated target -- 565 both, 243 Spotter only,
236 Octant-Hull only, 225 neither, of 1,269.

## The claim

Spotter's exclusive wins are bought at distances Octant-Hull never reaches.
Its median offset on them is 4 grids against Octant-Hull's 1.5, its 90th
percentile 11 against 6.5, and its worst 35 grids -- about 1,780 km -- where
Octant-Hull's exclusive cohort stops at 8. Those are the peripheral sites
whose cells are large enough that a nearest-seed verdict credits a prediction
most of a continent away.

And the far tail is a small part of it. 13.6% of Spotter's exclusive cohort
lies beyond 10 grids, which is 2.6% of all evaluated targets. The curve says
so: it is already at 0.86 where Octant-Hull's has ended.

## Each curve against its own cohort

The two are normalised separately, so the y axis reads "of the predictions
this method got right and the other did not, what share is this close". That
is the comparison the shapes are for. It is deliberately *not* a share of all
targets -- the cohorts are nearly the same size (19.1% and 18.6%) and putting
both on a common denominator would compress the tail this figure exists to
show. The sizes are in the manifest.

## Symmetric log, for the zeros

Octant-Hull lands 31 of its exclusive targets in the target's own grid, an
offset of exactly 0, which a log axis cannot draw. `symlog` with a one-grid
linear window shows them and still separates a tail that runs to 35. Spotter
has no such target: it never answers in the right grid on a target
Octant-Hull missed.

## What it does not say

Nothing about whether either method is *right* to be credited at those
distances -- that is the evaluation section's argument, and this figure is
the measurement under it.

Command: `plot-exclusive-error`. Needs `build-answer-space` and `classify` on
every run. Writes `_cross/exclusive-error/<datasets>[@<arm>]/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import contest as CT  # noqa: E402
from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules import methods  # noqa: E402
from scripts.analysis.v5.modules.methods import method_colors, method_label  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

#: `_cross/<KIND>/<datasets>[@<arm>]/`.
KIND = "exclusive-error"

SOURCE_NSIDE = CT.SOURCE_NSIDE
DEFAULT_METHOD_A = CT.DEFAULT_METHOD_A
DEFAULT_METHOD_B = CT.DEFAULT_METHOD_B

#: The two cohorts drawn, and which method's offset each reads. `both` and
#: `neither` are in the twin and off the figure: on `both` the two offsets are
#: comparable and `figure_error_diff` compares them, and on `neither` there is
#: no correct prediction to measure.
DRAWN: tuple[str, str] = (CT.COHORT_ONLY_A, CT.COHORT_ONLY_B)
OFFSET_OF = {CT.COHORT_ONLY_A: "offset_a", CT.COHORT_ONLY_B: "offset_b"}

PNG_NAME = "exclusive_error_cdf.{pair}.png"
CSV_NAME = "exclusive_error_cdf.{pair}.csv"
MANIFEST_NAME = "exclusive_error_cdf.{pair}.manifest.json"

_FIG_W = 4.0
_FIG_H = 3.0

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

_CDF_PT = 1.8
_LABEL_PT = 8.5

#: `symlog`'s linear window. One grid, so the zeros are drawn at zero rather
#: than pushed off a log axis or quietly dropped.
_LINTHRESH = 1.0

#: Candidate ticks, filtered to the data. Spelled rather than left to
#: matplotlib, which labels a symlog axis at decades and would leave the
#: 1-to-5 band -- where both medians sit -- unmarked.
_TICKS = (0, 1, 2, 5, 10, 20, 50, 100)

#: Where the tail is counted, in grids. Reported as a share of the cohort and
#: as a share of every evaluated target, because "a small portion" is only
#: meaningful against a stated denominator.
TAIL_GRIDS: tuple[int, ...] = (2, 5, 10, 20)

QUANTILES = (0.5, 0.9, 0.99)

#: Each curve's longest offset, marked where it ends. The two ceilings are
#: the claim -- 35 grids against 8 -- and a step curve's last riser is easy
#: to miss, especially the one that is a single target.
#:
#: `_HEADROOM` is how far the y axis runs past 1.0 to hold the numbers. In
#: data units, which is safe because a CDF cannot exceed 1.
_END_DASH = (0, (3, 2))
_END_PT = 1.0
_HEADROOM = 0.12
_END_LABEL_Y = 1.02

_X_LABEL = "Grid Error Distance (symlog)"
_Y_LABEL = "Fraction of Correct Predictions"


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/exclusive-error/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def pair_slug(method_a: str, method_b: str) -> str:
    terms = [
        "".join(ch for ch in method_label(m) if ch.isalnum()).lower()
        for m in (method_a, method_b)
    ]
    return "_vs_".join(terms)


def cohort_labels(method_a: str, method_b: str) -> dict[str, str]:
    """`cohort -> legend label`."""
    return {
        CT.COHORT_ONLY_A: f"{method_label(method_a)} Correct Only",
        CT.COHORT_ONLY_B: f"{method_label(method_b)} Correct Only",
    }


def cohort_ink(method_a: str, method_b: str) -> dict[str, str]:
    hues = method_colors([method_a, method_b])
    return {CT.COHORT_ONLY_A: hues[method_a], CT.COHORT_ONLY_B: hues[method_b]}


# -- the numbers ----------------------------------------------------------


def cohort_offsets(targets: pd.DataFrame, cohort: str) -> np.ndarray:
    """The offsets of the method that was right, for one exclusive cohort.

    Reading the *other* method's offset here would be meaningless: it did not
    place the target in the correct cell, so how far out it landed is not a
    measure of anything this figure is about.
    """
    if cohort not in OFFSET_OF:
        raise ValueError(f"{cohort!r} is not an exclusive cohort; pick from {list(DRAWN)}")
    rows = targets[targets["cohort"] == cohort]
    return rows[OFFSET_OF[cohort]].to_numpy()


def csv_columns() -> list[str]:
    """The twin: every evaluated target, so the partition is checkable."""
    return list(CT.TARGET_COLUMNS)


def cohort_stats(targets: pd.DataFrame, cohort: str) -> dict:
    """One cohort's size, shape and tail.

    The tail is reported twice on purpose. As a share of the cohort it says
    how much of this method's exclusive success is far out; as a share of
    every evaluated target it says how much of the whole picture that is. The
    claim needs both, and quoting either alone overstates or buries it.
    """
    n_all = len(targets)
    v = cohort_offsets(targets, cohort)
    out = {
        "cohort": cohort,
        "n_targets": int(v.size),
        "share_of_all": round(float(v.size) / n_all, 4) if n_all else None,
        "n_at_zero": int((v == 0).sum()),
    }
    if not v.size:
        return out
    out["offset_min"] = int(v.min())
    for q in QUANTILES:
        out[f"offset_p{int(q * 100)}"] = float(np.quantile(v, q))
    out["offset_max"] = int(v.max())
    for g in TAIL_GRIDS:
        beyond = int((v > g).sum())
        out[f"n_beyond_{g}"] = beyond
        out[f"share_of_cohort_beyond_{g}"] = round(beyond / v.size, 4)
        out[f"share_of_all_beyond_{g}"] = round(beyond / n_all, 4) if n_all else None
    return out


def partition(targets: pd.DataFrame) -> dict:
    """The four cohort sizes, which must account for every target."""
    counts = targets["cohort"].value_counts()
    out = {"n_targets": int(len(targets))}
    for c in CT.COHORTS:
        out[f"n_{c}"] = int(counts.get(c, 0))
        out[f"share_{c}"] = (
            round(float(counts.get(c, 0)) / len(targets), 4) if len(targets) else None
        )
    return out


# -- drawing --------------------------------------------------------------


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The empirical CDF as a staircase. Exact -- no bins, no bandwidth."""
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def visible_ticks(values: np.ndarray) -> list[int]:
    hi = float(np.max(values)) if len(values) else 1.0
    return [t for t in _TICKS if t <= hi * 1.2]


def render(
    targets: pd.DataFrame,
    method_a: str,
    method_b: str,
    out_png: Path,
    *,
    fig_size: tuple[float, float] = (_FIG_W, _FIG_H),
    dpi: int = 150,
) -> Path:
    """One step curve per exclusive cohort, each against its own denominator."""
    ink = cohort_ink(method_a, method_b)
    names = cohort_labels(method_a, method_b)
    series = {c: cohort_offsets(targets, c) for c in DRAWN}
    drawn = [v for v in series.values() if v.size]
    if not drawn:
        raise ValueError(
            "neither method has a target the other missed; there is no "
            "exclusive cohort to draw"
        )

    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.set_xscale("symlog", linthresh=_LINTHRESH)

    for cohort in DRAWN:
        v = series[cohort]
        if not v.size:
            continue
        x, y = ecdf(v)
        ax.step(
            x, y, where="post", color=ink[cohort], linewidth=_CDF_PT,
            label=names[cohort],
        )
        # Where this cohort stops, and how far out that is. The ceilings are
        # the claim and a last riser of one target is easy to miss.
        end = float(v.max())
        ax.vlines(
            end, 0.0, 1.0, color=ink[cohort], linewidth=_END_PT,
            linestyle=_END_DASH, zorder=2,
        )
        ax.annotate(
            f"{int(end)}", xy=(end, _END_LABEL_Y), ha="center", va="bottom",
            fontsize=8.0, color=ink[cohort],
        )

    everything = np.concatenate(drawn)
    ticks = visible_ticks(everything)
    # A little past the longest offset, not a lot: the far tail is the
    # subject, and dead space to the right of it reads as though the data
    # stopped earlier than it did.
    ax.set_xlim(-_LINTHRESH * 0.35, float(everything.max()) * 1.35)
    ax.set_ylim(-0.02, 1.0 + _HEADROOM)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel(_X_LABEL, fontsize=_LABEL_PT, color=_INK)
    ax.set_ylabel(_Y_LABEL, fontsize=_LABEL_PT, color=_INK)
    ax.tick_params(labelsize=_LABEL_PT, colors=_INK, length=3, width=0.7)
    ax.grid(True, color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.0, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=_SURFACE)
    plt.close(fig)
    return out_png


def _manifest(
    data: CT.ContestData, targets: pd.DataFrame, *, png_name: str, csv_name: str
) -> str:
    a, b = method_label(data.method_a), method_label(data.method_b)
    body = {
        "figure": png_name,
        "csv": csv_name,
        "subject": (
            f"How far out {a} and {b} land on the targets only one of them "
            f"places in the correct cell"
        ),
        "subject_note": (
            "the figure draws no title of its own -- the paper's caption names it"
        ),
        "method_a": data.method_a,
        "method_b": data.method_b,
        "datasets": cross.dataset_slug(data.run_ids),
        "run_ids": data.run_ids,
        "grid": G.describe(data.nside),
        "source_nside": data.nside,
        "grid_km": round(G.grid_km(data.nside), 1),
        "method_terms": {
            m: {"term": method_label(m), "name": methods.METHOD_TERMS.get(method_label(m))}
            for m in (data.method_a, data.method_b)
        },
        "partition": partition(targets),
        "drawn": list(DRAWN),
        "stats": [cohort_stats(targets, c) for c in DRAWN],
        "encoding": {
            "curve": "step, the empirical CDF of the offset; exact, no bins",
            "axes": (
                f"x: grid error distance of the method that was right, symlog "
                f"with a linear window of {_LINTHRESH}. y: the share of that "
                f"method's exclusive cohort at or below it."
            ),
            "cohorts": cohort_labels(data.method_a, data.method_b),
            "end_marker": (
                "a dashed rule in the cohort's hue where its curve ends, "
                "labelled with that longest offset in grids -- the two "
                "ceilings are the claim, and a last riser carrying one target "
                "is easy to miss"
            ),
        },
        "policy": {
            "offset": (
                "Each cohort reads the offset of the method that placed the "
                "target correctly. The other method's offset there measures "
                "nothing this figure is about -- it did not reach the cell."
            ),
            "denominator": (
                "Each curve is normalised to its own cohort, so the y axis "
                "compares shapes. A common denominator would compress the tail "
                "the figure exists to show, and the two cohorts are nearly the "
                "same size anyway. Both sizes are in `partition`."
            ),
            "scale": (
                "symlog about zero: an offset of 0 is a real and common value "
                f"for {b} here and a log axis cannot draw it."
            ),
            "tail": (
                "Reported as a share of the cohort and as a share of every "
                "evaluated target. 'A small portion' means nothing without a "
                "stated denominator, and the two differ by a factor of five."
            ),
        },
        "caveat": (
            "This measures how far out each method is when it alone is "
            "credited. It says nothing about whether a nearest-seed verdict "
            "should credit a prediction at that distance -- that is the "
            "section's argument, and this is the measurement under it."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    method_a: str = DEFAULT_METHOD_A,
    method_b: str = DEFAULT_METHOD_B,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> Path:
    """PNG, CSV twin and manifest for one method pair. Returns the PNG."""
    nside = G.validate_nside(nside)
    data = CT.load(
        runs, method_a=method_a, method_b=method_b, nside=nside,
        analysis_root=analysis_root,
    )
    targets = CT.paired_targets(data)
    pair = pair_slug(method_a, method_b)
    names = {k: v.format(pair=pair) for k, v in
             zip(("png", "csv", "man"), (PNG_NAME, CSV_NAME, MANIFEST_NAME))}
    out_dir = output_dir(data.run_ids, analysis_root=analysis_root)
    png = render(targets, method_a, method_b, out_dir / names["png"])
    targets.reindex(columns=csv_columns()).to_csv(out_dir / names["csv"], index=False)
    (out_dir / names["man"]).write_text(
        _manifest(data, targets, png_name=names["png"], csv_name=names["csv"])
    )
    return png
