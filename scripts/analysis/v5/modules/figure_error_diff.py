"""Where both methods are right, which of them is nearer -- and by how much.

The cell metric ranks Spotter first. The error distance ranks it last. This is
the figure that reconciles the two, on the only cohort where the question is
fair: the **565 targets both methods place in the correct cell**, across 34
`(run, site_id)` pairs.

On those, Octant-Hull is nearer on 86.5% and Spotter on 4.2%, with the two
level on the remaining 9.2%. The median difference is 2 grids, the 95th
percentile 8, and the worst 19. It is not a mesh effect: the share runs 86.9%,
85.7% and 86.8% on as01, as02 and as03. Per site, Octant-Hull has the nearer
median on 29 of the 34 pairs and Spotter on 2.

## Why this cohort, and not each method's own correct set

A method that answers fewer targets is not thereby more accurate on the ones
it does answer. Comparing each method over its own correct predictions scores
them on different populations, and the populations differ in exactly the way
that matters -- Spotter's correct set is the peripheral sites where the cell
is large. Restricting to targets both got right is what makes the difference a
paired quantity at all.

## The sign is fixed

`diff = offset_SPO - offset_OCT-H`, so positive means Spotter is further out
and the mass sits to the **right** of zero. An earlier expectation had the
curve leaning left; the data says the opposite and the convention does not
move to suit it. `contest.both_correct_targets` owns the definition.

## Symmetric log, because zero is the subject

The differences run from -6 to +19 grids and pile up between 0 and 3. A linear
axis spends most of its width on a tail carrying a few per cent of the
targets, and a plain log axis cannot show zero at all -- which is the value
9.2% of the cohort takes. `symlog` about zero keeps the crossing point at the
centre where the reading is, and still separates the tail.

## What this is not

It is not the 21-site statistic from the peripherality figure, which asks a
different question over a different population and is reported in the
manifest as a companion so the two are not conflated. And it says nothing
about the targets only one method reached -- that is the contest map's half of
the argument, and it is the half that favours Spotter.

Command: `plot-error-diff`. Needs `build-answer-space` and `classify` on every
run. Writes `_cross/error-diff/<datasets>[@<arm>]/`.
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
from scripts.analysis.v5.modules.methods import method_label  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import solved_mask  # noqa: E402

#: `_cross/<KIND>/<datasets>[@<arm>]/`.
KIND = "error-diff"

SOURCE_NSIDE = CT.SOURCE_NSIDE
DEFAULT_METHOD_A = CT.DEFAULT_METHOD_A
DEFAULT_METHOD_B = CT.DEFAULT_METHOD_B

DIFF_COL = "diff"

PNG_NAME = "paired_error_diff.{pair}.png"
CSV_NAME = "paired_error_diff.{pair}.csv"
MANIFEST_NAME = "paired_error_diff.{pair}.manifest.json"

#: Matching `figure_stability`.
_FIG_W = 4.0
_FIG_H = 3.0

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

_CDF_PT = 1.8
_ZERO_PT = 1.0

#: The direction arrows, in a reserved band above the curve. `_HEADROOM` is
#: how far the y-axis runs past 1.0 to make that band, and `_ARROW_Y` where
#: in it the arrows sit -- both in data units, so the band is empty whatever
#: the curve does: a CDF cannot exceed 1.
#:
#: Placed here rather than outside the axes because `tight_layout` does not
#: reserve space for an annotation drawn past the axes edge, and a figure
#: saved at a fixed canvas would clip it.
_HEADROOM = 0.18
_ARROW_Y = 1.09
_ARROW_PT = 0.9

#: `symlog`'s linear window about zero. One grid: the smallest difference
#: there is, so nothing inside the window is ever compressed.
_LINTHRESH = 1.0

#: Candidate ticks, filtered to the data. Spelled rather than left to
#: matplotlib, which labels a symlog axis at decades and leaves the 1-to-3
#: band -- where the median sits -- unmarked.
_TICKS = (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50)

_Y_LABEL = "Fraction of Correct Predictions"

#: Reported in the manifest, on the difference and on each method's offsets.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/error-diff/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def pair_slug(method_a: str, method_b: str) -> str:
    """`spo_vs_octh`, as `figure_stability` names its pair."""
    terms = [
        "".join(ch for ch in method_label(m) if ch.isalnum()).lower()
        for m in (method_a, method_b)
    ]
    return "_vs_".join(terms)


def x_label(method_a: str, method_b: str) -> str:
    """The axis carries the sign convention, because the figure turns on it."""
    return (
        f"Diff. of Grid Error Distances "
        f"({method_label(method_a)} - {method_label(method_b)})"
    )


def direction_labels(method_a: str, method_b: str) -> dict[str, str]:
    """`side -> what that direction means`, for the arrows above the curve.

    Arrows rather than tinted half-planes: a direction is what the axis
    actually encodes, and two filled bands behind a step curve spend a lot of
    ink saying it. They also read in greyscale, which two pale tints of the
    methods' hues do not.
    """
    return {
        CT.CLOSER_A: f"{method_label(method_a)} better",
        CT.CLOSER_B: f"{method_label(method_b)} better",
    }


# -- the numbers ----------------------------------------------------------


def csv_columns() -> list[str]:
    """The twin is `contest.both_correct_targets`, one row per target."""
    return list(CT.PAIRED_COLUMNS)


def shares(paired: pd.DataFrame) -> dict:
    """The three shares the curve's geometry carries, as numbers."""
    n = len(paired)
    counts = paired["closer"].value_counts()
    return {
        "n_targets": int(n),
        **{
            f"share_{k}": (round(float(counts.get(k, 0)) / n, 4) if n else None)
            for k in (CT.CLOSER_A, CT.CLOSER_B, CT.CLOSER_EQUAL)
        },
    }


def diff_stats(paired: pd.DataFrame) -> dict:
    """The difference's shape. Percentiles, not a mean: it is integral,
    skewed, and the median is the number the prose quotes."""
    d = paired[DIFF_COL]
    out = {"diff_min": int(d.min()), "diff_max": int(d.max())}
    for q in QUANTILES:
        out[f"diff_p{int(q * 100)}"] = float(d.quantile(q))
    return out


def per_mesh(paired: pd.DataFrame) -> list[dict]:
    """The same shares per dataset, so "not a mesh effect" is checkable."""
    return [
        {"dataset": ds, **shares(rows), **diff_stats(rows)}
        for ds, rows in paired.groupby("dataset", sort=True)
    ]


def per_site(paired: pd.DataFrame) -> dict:
    """Head-to-head over `(run_id, site_id)` pairs, by median difference.

    A second denominator on the same cohort: 565 targets are ~20 replicas of
    34 sites, so the target-level shares are not 565 independent observations
    and this is the count that respects the site key.
    """
    med = paired.groupby(list(CT.SITE_KEY))[DIFF_COL].median()
    return {
        "n_site_pairs": int(len(med)),
        f"n_sites_{CT.CLOSER_A}": int((med < 0).sum()),
        f"n_sites_{CT.CLOSER_B}": int((med > 0).sum()),
        f"n_sites_{CT.CLOSER_EQUAL}": int((med == 0).sum()),
    }


def win_site_companion(data: CT.ContestData, table: pd.DataFrame) -> dict:
    """The 21-site statistic, for the manifest only.

    A different question on a different population: over *every* solved target
    at the sites method `a` wins outright, not over the both-correct cohort.
    Reported so the two are not conflated -- and it points the same way.
    """
    wins = table.loc[table["category"] == CT.A_WINS, list(CT.SITE_KEY)]
    keys = set(map(tuple, wins.to_numpy()))
    out: dict = {"n_win_sites": len(keys), "cohort": "every solved target at those sites"}
    if not keys:
        return out
    medians = {}
    for key, method in (("a", data.method_a), ("b", data.method_b)):
        rows = []
        for run_id in data.run_ids:
            f = data.frames[(run_id, method)]
            f = f[solved_mask(f).to_numpy()]
            f = f[[(run_id, s) in keys for s in f["site_id"]]]
            rows.append(f.assign(run_id=run_id))
        got = pd.concat(rows, ignore_index=True)
        offsets = got[CT.C.GRID_OFFSET]
        out[f"offset_p5_{key}"] = float(offsets.quantile(0.05))
        out[f"offset_p50_{key}"] = float(offsets.quantile(0.5))
        out[f"offset_p95_{key}"] = float(offsets.quantile(0.95))
        medians[key] = got.groupby(["run_id", "site_id"])[CT.C.GRID_OFFSET].median()
    # Inner join: a site where one of them solved nothing cannot be judged
    # more accurate either way, and comparing the two Series unaligned is a
    # pandas error rather than a silent wrong answer only by luck.
    a_med, b_med = medians["a"].align(medians["b"], join="inner")
    out["n_sites_compared"] = int(len(a_med))
    out["n_sites_b_more_accurate"] = int((b_med < a_med).sum())
    out["n_sites_a_more_accurate"] = int((a_med < b_med).sum())
    return out


# -- drawing --------------------------------------------------------------


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The empirical CDF as a staircase. Exact -- no bins, no bandwidth."""
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def visible_ticks(values: np.ndarray) -> list[int]:
    lo, hi = float(np.min(values)), float(np.max(values))
    return [t for t in _TICKS if lo - 1 <= t <= hi + 1]


def render(
    paired: pd.DataFrame,
    method_a: str,
    method_b: str,
    out_png: Path,
    *,
    fig_size: tuple[float, float] = (_FIG_W, _FIG_H),
    dpi: int = 150,
) -> Path:
    """One step curve, zero at the centre, symmetric log either side."""
    names = direction_labels(method_a, method_b)
    d = paired[DIFF_COL].to_numpy()
    x, y = ecdf(d)

    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    ax.set_xscale("symlog", linthresh=_LINTHRESH)
    # Symmetric about zero, which is the whole point of the axis. Floored at
    # the linear window: a cohort on which the two methods never differ has a
    # range of zero and would otherwise collapse the limits onto a point.
    span = max(abs(float(d.min())), abs(float(d.max())), _LINTHRESH) * 1.35
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.02, 1.0 + _HEADROOM)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax.axvline(0.0, color=_INK_2, linewidth=_ZERO_PT, zorder=2)
    ax.step(x, y, where="post", color=_INK, linewidth=_CDF_PT, zorder=3)

    # Which side means whom -- the one thing a reader must not have to work
    # out from the sign. Mixed coordinates: the band's height is in data units
    # (a CDF stops at 1, so it is always clear) and the reach is a fraction of
    # the axes, so it does not depend on the range the data happens to span.
    mixed = ("axes fraction", "data")
    for side, tip, anchor, align in (
        (CT.CLOSER_A, 0.02, 0.46, "right"),
        (CT.CLOSER_B, 0.98, 0.54, "left"),
    ):
        ax.annotate(
            names[side], xy=(tip, _ARROW_Y), xytext=(anchor, _ARROW_Y),
            xycoords=mixed, textcoords=mixed, ha=align, va="center",
            fontsize=8.0, color=_INK_2,
            arrowprops={"arrowstyle": "->", "color": _INK_2, "linewidth": _ARROW_PT},
        )

    ticks = visible_ticks(d)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel(x_label(method_a, method_b), fontsize=8.5, color=_INK)
    ax.set_ylabel(_Y_LABEL, fontsize=8.5, color=_INK)
    ax.tick_params(labelsize=8.5, colors=_INK, length=3, width=0.7)
    ax.grid(True, color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
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
    data: CT.ContestData,
    paired: pd.DataFrame,
    companion: dict,
    *,
    png_name: str,
    csv_name: str,
) -> str:
    a, b = method_label(data.method_a), method_label(data.method_b)
    body = {
        "figure": png_name,
        "csv": csv_name,
        "subject": (
            f"On the targets both {a} and {b} place in the correct cell, how "
            f"much further out {a} lands"
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
        "method_terms": {
            m: {"term": method_label(m), "name": methods.METHOD_TERMS.get(method_label(m))}
            for m in (data.method_a, data.method_b)
        },
        "pooled": {**shares(paired), **diff_stats(paired), **per_site(paired)},
        "per_mesh": per_mesh(paired),
        "companion_win_sites": companion,
        "encoding": {
            "curve": "step, the empirical CDF of the difference; exact, no bins",
            "axes": (
                f"x: {DIFF_COL}, in grid steps, symlog about zero with a linear "
                f"window of {_LINTHRESH}. y: the share of the cohort at or "
                f"below it."
            ),
            "directions": {
                s: n for s, n in direction_labels(data.method_a, data.method_b).items()
            },
            "reading": (
                "The curve's height where it crosses zero is the share on "
                "which a is at least as near; the jump *at* zero is the share "
                "on which they are level. Everything to the right of the jump "
                "is a target b placed nearer."
            ),
        },
        "policy": {
            "cohort": (
                "Targets both methods place in the correct cell, solved rows "
                "only. Comparing each method over its own correct set would "
                "score them on different populations, and the populations "
                "differ in the way that matters: a's correct set is the "
                "peripheral sites where cells are large."
            ),
            "sign": (
                f"diff = offset_{a} - offset_{b}. Positive means {a} is "
                f"further out, so the mass sits right of zero. Fixed: an "
                f"earlier expectation had it leaning left, and the convention "
                f"does not move to suit that."
            ),
            "scale": (
                "symlog about zero. A linear axis spends its width on a tail "
                "carrying a few per cent; a log axis cannot show zero, which "
                "is the value 9.2% of the cohort takes."
            ),
            "denominators": (
                "The target-level shares rest on ~20 replicas of 34 sites, so "
                "they are not that many independent observations. per_site "
                "gives the head-to-head on the site key beside them."
            ),
        },
        "caveat": (
            f"This is the half of the comparison that favours {b}, and it is "
            f"silent on the targets only one method reached -- the contest map "
            f"carries that half, and it favours {a}. companion_win_sites is a "
            f"different question on a different population and is reported so "
            f"the two are not conflated."
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
    paired = CT.both_correct_targets(data)
    if not len(paired):
        raise ValueError(
            f"{method_a} and {method_b} place no target in the correct cell "
            f"together; there is no paired cohort to compare"
        )
    companion = win_site_companion(data, CT.contest_table(data))
    pair = pair_slug(method_a, method_b)
    names = {k: v.format(pair=pair) for k, v in
             zip(("png", "csv", "man"), (PNG_NAME, CSV_NAME, MANIFEST_NAME))}
    out_dir = output_dir(data.run_ids, analysis_root=analysis_root)
    paired.reindex(columns=csv_columns()).to_csv(out_dir / names["csv"], index=False)
    png = render(paired, method_a, method_b, out_dir / names["png"])
    (out_dir / names["man"]).write_text(
        _manifest(data, paired, companion, png_name=names["png"], csv_name=names["csv"])
    )
    return png
