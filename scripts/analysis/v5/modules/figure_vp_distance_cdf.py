"""The two VP distances per TG, and the gap between them, as one CDF.

`figure_vp_proximity` answers "how close was a VP, for the TGs a method placed
best" -- per method, per cohort. This module answers the question underneath
it, which is not about methods at all: **for a TG in these meshes, how far is
the nearest VP, how far is the one latency nominates, and how far apart are
those two?** One panel, the whole population, no method on the axis.

## Why the third curve is the one that carries the claim

Drawing `d_geo` and `d_sp` alone shows one distribution sitting left of the
other. That is *stochastic dominance*, and it is weaker than the claim the
paper makes. `d_geo <= d_sp` is a per-TG inequality: it says that for every
single TG, the smallest-RTT VP is no closer than the closest VP. Two marginal
curves are consistent with that and also consistent with its failing on
individual TGs, because a CDF discards the pairing.

The gap curve restores it. `d_sp - d_geo` is computed per TG, so its support
lying at or above zero *is* the pointwise inequality. `min_gap_km` is written
to the manifest for exactly this reason: a log axis cannot render a negative
value, so a violation would be invisible on the figure rather than obvious.
Never let the picture carry that claim on its own -- cite the number.

## The zero mass is the finding, and a log axis cannot draw it

For a share of TGs the two VPs coincide and the gap is exactly 0. On as01-03
that is 243 of 1,269 (19.1%), and `log(0)` does not exist, so those TGs cannot
be plotted where they belong. Dropping them would silently delete a fifth of
the population and, worse, delete precisely the TGs the surrounding argument
is about.

Instead the gap curve *starts* at `x_min` already carrying that share on the
y-axis, and the figure marks and annotates the intercept. Read the left
intercept as "the share of TGs whose gap is zero", not as a value at
`x_min`. `zero_gap_share_pct` is in the manifest and the CSV.

The smallest **positive** gap on these meshes is 1.71 km, so the flat run from
the axis floor to 1.71 km is a real void in the distribution and not a
rendering choice. `min_positive_gap_km` records it.

## Quantization, and why a CDF rather than a violin or a box

~20 IP replicas share a site and so share its VP geometry exactly, so 1,269
TGs carry only 43 distinct `d_geo` values, 63 distinct `d_sp` and 54 distinct
gaps. A KDE would report density between values that hold nothing. A CDF is a
step function over the observed values and invents nothing between them, which
is why the steps in this figure are the data rather than an artifact of it.
`n_distinct` per series is in the CSV.

Command: `plot-vp-distance-cdf`. Writes `_cross/vp-distance-cdf/<datasets>[@<arm>]/`.
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

from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules.figure_vp_proximity import (  # noqa: E402
    MEASURE_COLUMNS,
    MEASURE_HUES,
    SOURCE_NSIDE,
    load,
)
from scripts.analysis.v5.modules.figure_vp_proximity import GEO as _GEO_KEY  # noqa: E402
from scripts.analysis.v5.modules.figure_vp_proximity import SPING as _SPING_KEY  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

KIND = "vp-distance-cdf"

PNG_NAME = "vp_distance_cdf.png"
CSV_NAME = "vp_distance_cdf.csv"
MANIFEST_NAME = "vp_distance_cdf.manifest.json"

#: Series drawn, in legend order: (key, label, colour, linestyle).
GEO = "d_geo"
SPING = "d_sp"
GAP = "gap"

#: `d_geo` reuses `figure_vp_proximity.MEASURE_HUES[GEO]` exactly, so the
#: nearest-VP series is the same blue in the violins and here. `d_sp` does
#: NOT: the violins draw it in that module's orange, which this figure spends
#: on the gap instead. Orange therefore means `d_sp` in one figure and
#: `d_sp - d_geo` in the other -- deliberate, but state it in both captions.
#: The gap is dashed so the green/orange pair, the one combination deuteranopes
#: lose, is never carried by hue alone.
_GEO_HUE = MEASURE_HUES[_GEO_KEY]       # "#2a78d6"
_SPING_HUE = "#2f8f4e"
_GAP_HUE = MEASURE_HUES[_SPING_KEY]     # "#eb6834"
_SERIES = (
    (GEO, r"$d_\mathrm{geo}$  (nearest VP)", _GEO_HUE, "-"),
    (SPING, r"$d_\mathrm{sp}$  (smallest-RTT VP)", _SPING_HUE, "-"),
    (GAP, r"$d_\mathrm{sp}-d_\mathrm{geo}$  (gap)", _GAP_HUE, "--"),
)

#: Fixed so this panel and the error CDF are read on one axis.
X_MIN_KM = 0.1
X_MAX_KM = 10_000.0

PERCENTILES = (5, 25, 50, 75, 90, 95)


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/vp-distance-cdf/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def population(long: pd.DataFrame) -> pd.DataFrame:
    """One row per TG carrying both distances and their gap.

    `long` is per `(run, TG, method)`; the two distances are TG properties
    repeated onto every method's row, so this deduplicates. Keying on
    `(run_id, tg_id)` rather than `tg_id` alone is deliberate: TGs are
    guaranteed disjoint across runs by `cross.guard_disjoint_tgs`, but relying
    on that here would make this function wrong the moment the guard is
    relaxed.
    """
    out = (
        long.drop_duplicates(subset=["run_id", "tg_id"])
        .loc[:, ["run_id", "tg_id", MEASURE_COLUMNS[_GEO_KEY], MEASURE_COLUMNS[_SPING_KEY]]]
        .reset_index(drop=True)
    )
    out[GAP] = out[MEASURE_COLUMNS[_SPING_KEY]] - out[MEASURE_COLUMNS[_GEO_KEY]]
    violations = int((out[GAP] < -1e-9).sum())
    if violations:
        raise ValueError(
            f"{violations} TGs have d_sp < d_geo, which is arithmetically "
            f"impossible: the smallest-RTT VP cannot be closer than the "
            f"closest VP. The two VPs were selected off different frames and "
            f"every distance here is garbage. Do not plot this."
        )
    return out


def series(pop: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three drawn series, by key."""
    return {
        GEO: pop[MEASURE_COLUMNS[_GEO_KEY]].to_numpy(),
        SPING: pop[MEASURE_COLUMNS[_SPING_KEY]].to_numpy(),
        GAP: pop[GAP].to_numpy(),
    }


def stats_table(pop: pd.DataFrame) -> pd.DataFrame:
    """One row per series: percentiles, extrema, and the quantization counts.

    `zero_share_pct` is meaningful only for the gap, where it is the share of
    TGs whose two VPs coincide. It is emitted for all three so the column is
    total, and is 0 for the distances unless a VP sits exactly on a TG.
    """
    rows = []
    for key, values in series(pop).items():
        q = np.percentile(values, PERCENTILES)
        positive = values[values > 0]
        rows.append(
            {
                "series": key,
                "n": int(len(values)),
                "n_distinct": int(len(np.unique(values))),
                "min_km": float(values.min()),
                "max_km": float(values.max()),
                "mean_km": float(values.mean()),
                **{f"p{p}_km": float(v) for p, v in zip(PERCENTILES, q)},
                "n_zero": int((values == 0).sum()),
                "zero_share_pct": float((values == 0).mean() * 100),
                "min_positive_km": float(positive.min()) if len(positive) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values against the share of the population at or below each."""
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x) * 100


def _gap_curve(gap: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The gap ECDF with its zero mass folded onto the left edge.

    Returns `(x, y, zero_share_pct)`. The first point sits at `X_MIN_KM`
    already carrying the zero share, so the curve's left intercept reads as
    that share rather than as a value at the axis floor.
    """
    n = len(gap)
    zero_share = float((gap == 0).mean() * 100)
    positive = np.sort(gap[gap > 0])
    x = np.concatenate([[X_MIN_KM], positive])
    y = np.concatenate([[zero_share], zero_share + np.arange(1, len(positive) + 1) / n * 100])
    return x, y, zero_share


def plot(pop: pd.DataFrame, *, meta: dict, out_png: Path) -> Path:
    """The three-curve panel, sized as a small descriptive figure.

    No in-figure annotation and no title: at this size both crowd the axes,
    and the caption is the right place for them. That moves one burden onto
    the caption, which **must** explain the gap curve's left intercept --
    without it a reader sees the dashed curve begin at 19% for no visible
    reason and reads a rendering fault. `zero_gap.share_pct` in the manifest
    is the number the caption needs.
    """
    values = series(pop)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))

    for key, label, colour, linestyle in _SERIES:
        if key == GAP:
            x, y, zero_share = _gap_curve(values[GAP])
        else:
            x, y = _ecdf(values[key])
        ax.step(x, y, where="post", color=colour, ls=linestyle, lw=1.4, label=label)

    # The intercept marker stays: it is the only thing distinguishing "the
    # curve starts here carrying 19.1%" from "the curve was clipped". The
    # share is printed beside it rather than left to the caption alone, since
    # a reader who misses the caption misreads the whole curve.
    ax.plot([X_MIN_KM], [zero_share], marker="o", ms=3.5, color=_GAP_HUE,
            clip_on=False, zorder=5)
    ax.text(X_MIN_KM * 1.18, zero_share + 3.0, f"{zero_share:.1f}%",
            fontsize=7, color=_GAP_HUE, ha="left", va="bottom", zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(X_MIN_KM, X_MAX_KM)
    ax.set_ylim(0, 100)
    ax.set_xlabel("distance (km)", fontsize=8)
    ax.set_ylabel("share of TGs (%)", fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(labelsize=7.5, length=3)
    ax.grid(alpha=0.25, lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=False,
              handlelength=1.8, borderaxespad=0.3, labelspacing=0.35)

    fig.tight_layout(pad=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _manifest(meta: dict, pop: pd.DataFrame, stats: pd.DataFrame) -> str:
    gap = pop[GAP].to_numpy()
    positive = gap[gap > 0]
    return json.dumps(
        {
            "figure": PNG_NAME,
            "csv": CSV_NAME,
            "kind": KIND,
            "run_ids": meta["run_ids"],
            "nside": meta["nside"],
            "n_tgs": int(len(pop)),
            "scope": (
                "population: one row per TG, methods deduplicated away. The two "
                "distances are TG properties, not method outputs, so no cohort "
                "and no solved_mask applies -- every evaluated TG is here, "
                "including those every method refused."
            ),
            "pointwise_inequality": {
                "min_gap_km": float(gap.min()),
                "holds": bool((gap >= -1e-9).all()),
                "note": (
                    "d_geo <= d_sp is a per-TG claim. The two marginal curves "
                    "show only stochastic dominance, which is weaker; this "
                    "number is what establishes the pointwise inequality. A log "
                    "x-axis cannot render a negative gap, so a violation would "
                    "be invisible on the figure -- cite this, not the picture."
                ),
            },
            "zero_gap": {
                "n": int((gap == 0).sum()),
                "share_pct": float((gap == 0).mean() * 100),
                "min_positive_gap_km": float(positive.min()) if len(positive) else None,
                "note": (
                    "log(0) does not exist, so these TGs cannot be drawn at "
                    "their value. The gap curve begins at x_min already "
                    "carrying this share; read the left intercept as the share, "
                    "not as a value at x_min. The flat run from x_min to "
                    "min_positive_gap_km is a real void, not a rendering choice."
                ),
            },
            "quantization": {
                "n_distinct": {
                    r.series: int(r.n_distinct) for r in stats.itertuples()
                },
                "note": (
                    "~20 IP replicas share a site and so share its VP geometry "
                    "exactly. A CDF is a step function over observed values and "
                    "invents nothing between them, which is why a CDF is drawn "
                    "here where figure_vp_proximity draws violins."
                ),
            },
            "x_axis": {
                "scale": "log",
                "min_km": X_MIN_KM,
                "max_km": X_MAX_KM,
                "fixed_note": (
                    "both bounds fixed to match figure_error_cdf, so the two "
                    "panels are read on one axis"
                ),
            },
        },
        indent=2,
    )


def build_for_runs(
    runs: list[RunPaths],
    *,
    methods: list[str] | None = None,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
    source_csv: dict[str, Path] | None = None,
) -> list[Path]:
    """PNG, stats CSV and manifest. Returns the PNG in a list, as siblings do."""
    nside = G.validate_nside(nside)
    long, meta = load(runs, methods=methods, nside=nside, analysis_root=analysis_root,
                      source_csv=source_csv)
    pop = population(long)
    stats = stats_table(pop)

    out_dir = output_dir(meta["run_ids"], analysis_root=analysis_root)
    stats.to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(_manifest(meta, pop, stats))
    return [plot(pop, meta=meta, out_png=out_dir / PNG_NAME)]
