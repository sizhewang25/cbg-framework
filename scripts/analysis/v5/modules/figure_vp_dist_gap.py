"""The gap `d_sp - d_geo` at three scopes: the population, and a reference
method's two best-case cohorts.

`figure_vp_distance_cdf` draws the gap once, over every TG. This module draws
only the gap, three times, restricted to nested populations: all TGs, then the
reference method's most accurately placed 25%, then its most accurate 5%. One
quantity, one axis, three scopes, so the curves are directly comparable and
the nesting (63 subset of 317 subset of 1,269) is what the figure is about.

## What it is for

S-P returns the smallest-RTT VP's location, so its error *is* `d_sp`, and
`d_sp = d_geo + gap` splits that error into a geographic term and a term that
measures how faithfully latency orders VPs by distance. This figure asks
whether S-P's near-exact predictions are the TGs where the second term
vanishes. On as01-03 they are.

## Read the maximum, not the zero share

The obvious summary is the share of each scope whose gap is exactly zero
(19.1% / 67.5% / 100% on as01-03). Two problems with quoting that alone.

At p5 a zero gap is close to **forced**, so the 100% is not a finding. S-P's
p5 bound is 1.55 km and S-P's error is `d_sp`, so every TG in that cohort has
`d_sp <= 1.55` km; `gap <= d_sp` bounds the gap by 1.55 km; and no TG in the
pooled population has a gap anywhere in (0, 1.71) km. A nonzero gap therefore
cannot fit inside the cohort. `forced_zero_gap` in the manifest carries the
three numbers that make that argument, per cohort, so the claim can be checked
rather than assumed -- on a mesh where the cohort bound clears the void, the
same 100% *would* be a finding and the flag says so.

At p25 the zero share understates the result badly. The 32.5% of that cohort
with a nonzero gap miss by at most 5.62 km, which is not meaningfully
different from zero at the scale of a 113.8 km population median. The claim
that survives both objections is about the whole cohort rather than a share:
**the gap never exceeds `max_km` anywhere in the cohort**, against a bound of
25.3 km that would have admitted four times that. `max_km` leads the CSV.

## Quantization

The p25 cohort's 317 TGs carry three distinct gap values (0, 5.39, 5.62 km),
because ~20 IP replicas share a site and so share its VP geometry exactly.
That is roughly three independent observations, not 317, and `n_distinct` is
emitted per scope so no count is read without it.

## Drawing choices

All three curves are solid, and hue separates the scopes: grey for the
population, blue for p25, orange for p5. Every curve is the same quantity, so
the key to this figure is not the key to `figure_vp_distance_cdf`, where blue
is `d_geo` and orange is the gap. Say so in both captions. The gap is dashed
in that figure *because* it sits beside two other measures; here nothing is
dashed, since there is only one measure to distinguish.

Zero mass folds onto the left edge exactly as in `figure_vp_distance_cdf`
(`_gap_curve` is imported, not reimplemented). Each curve is then carried to
the right edge at its final height, so an all-zero cohort draws as a flat line
at 100% rather than as a lone marker that reads like a plotting fault. That
extension is not invention: a CDF that has reached 100% is flat thereafter.

Command: `plot-vp-dist-gap`. Writes `_cross/vp-dist-gap/<datasets>[@<arm>]/`.
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
from scripts.analysis.v5.modules.figure_vp_distance_cdf import (  # noqa: E402
    GAP,
    X_MAX_KM,
    X_MIN_KM,
    _gap_curve,
    population,
)
from scripts.analysis.v5.modules.figure_vp_proximity import (  # noqa: E402
    COHORTS,
    RANK_COLUMN,
    SOURCE_NSIDE,
    cohort_frame,
    load,
)
from scripts.analysis.v5.modules.methods import method_label  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import SHORTEST_PING  # noqa: E402

KIND = "vp-dist-gap"

PNG_NAME = "vp_dist_gap.png"
CSV_NAME = "vp_dist_gap.csv"
MANIFEST_NAME = "vp_dist_gap.manifest.json"

#: The population scope, drawn alongside the cohorts. Not a cohort name.
ALL = "all"

#: Cohorts drawn beside the population, tightest last so it is on top.
DEFAULT_COHORTS = ("p25", "p5")

#: Scope -> hue. Grey for the population, so the reference curve recedes and
#: the two cohorts read against it. **Hue means scope here, not measure**:
#: `figure_vp_distance_cdf` spends this blue on `d_geo` and this orange on the
#: gap, whereas every curve in this figure *is* the gap. Both captions have to
#: say so, or a reader carrying one figure's key into the other misreads both.
SCOPE_HUES = {ALL: "#898781", "p25": "#2a78d6", "p5": "#eb6834"}

PERCENTILES = (50, 75, 90, 95)


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/vp-dist-gap/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


def scopes(
    long: pd.DataFrame,
    *,
    reference: str = SHORTEST_PING,
    cohorts: tuple[str, ...] = DEFAULT_COHORTS,
) -> dict[str, pd.DataFrame]:
    """`scope -> one row per TG with its gap`, population first.

    Cohort membership comes from `cohort_frame`, never from a local ranking:
    a FALLBACK row carries a real `pred_dist_to_tg_km` (the fallback
    prediction is the shortest-ping VP's own coordinate), so ranking without
    `solved_mask` fills a "most accurate" cohort with give-up rows. The gap
    itself is a TG property and needs no mask -- the mask decides *which TGs*
    are in the cohort, not what their gap is.
    """
    if reference not in set(long.method):
        raise ValueError(
            f"reference method {reference!r} is not scored here; "
            f"available: {sorted(set(long.method))}"
        )
    pop = population(long).set_index(["run_id", "tg_id"])
    out = {ALL: pop.reset_index()}
    for cohort in cohorts:
        if cohort not in COHORTS or COHORTS[cohort] is None:
            raise ValueError(
                f"{cohort!r} is not a percentile cohort; "
                f"choose from {sorted(c for c, f in COHORTS.items() if f is not None)}"
            )
        rows = cohort_frame(long, cohort)
        keys = rows[rows.method == reference].set_index(["run_id", "tg_id"]).index
        if not len(keys):
            raise ValueError(
                f"cohort {cohort!r} is empty for {reference!r}: k rounds to 0 "
                f"on {len(pop)} TGs, or the reference solved nothing. An empty "
                f"scope draws as no curve at all, which reads as a missing "
                f"method rather than as an absent cohort."
            )
        out[cohort] = pop.loc[keys].reset_index()
    return out


def cohort_bounds(
    long: pd.DataFrame,
    *,
    reference: str = SHORTEST_PING,
    cohorts: tuple[str, ...] = DEFAULT_COHORTS,
) -> dict[str, float]:
    """`cohort -> the reference's worst error in it`, the bound the gap lives under.

    This is the number the "is a zero gap forced?" argument turns on: the gap
    cannot exceed it, because `gap <= d_sp` and `d_sp` *is* the reference's
    error when the reference is S-P.
    """
    out = {}
    for cohort in cohorts:
        rows = cohort_frame(long, cohort)
        out[cohort] = float(rows[rows.method == reference][RANK_COLUMN].max())
    return out


def stats_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per scope. `max_km` leads because it is the claim.

    `zero_share_pct` is kept, but it is the weaker summary: see the module
    docstring on why the p5 share is close to forced and the p25 share
    understates its cohort.
    """
    rows = []
    for scope, frame in frames.items():
        values = frame[GAP].to_numpy()
        positive = values[values > 0]
        rows.append(
            {
                "scope": scope,
                "n": int(len(values)),
                "max_km": float(values.max()),
                "n_distinct": int(len(np.unique(values))),
                "n_zero": int((values == 0).sum()),
                "zero_share_pct": float((values == 0).mean() * 100),
                "min_positive_km": float(positive.min()) if len(positive) else float("nan"),
                "n_positive": int(len(positive)),
                **{
                    f"p{p}_km": float(v)
                    for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))
                },
                "mean_km": float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def forced_zero_gap(
    frames: dict[str, pd.DataFrame], bounds: dict[str, float]
) -> dict[str, dict]:
    """Per cohort, whether an all-zero gap follows from the cohort bound alone.

    A nonzero gap in a cohort must be at least the smallest positive gap in
    the *population*, and at most the cohort's bound on the reference's error.
    When the bound falls below that floor, no nonzero gap can fit and a 100%
    zero share carries no information about the data. Reporting the verdict
    rather than the share keeps the paper from resting on an arithmetic
    identity, and keeps it able to claim the result on a mesh where the bound
    clears the floor.
    """
    pop = frames[ALL][GAP].to_numpy()
    positive = pop[pop > 0]
    floor = float(positive.min()) if len(positive) else float("inf")
    out = {}
    for cohort, bound in bounds.items():
        out[cohort] = {
            "cohort_bound_km": bound,
            "population_min_positive_gap_km": floor,
            "forced": bool(bound < floor),
            "note": (
                "gap <= d_sp <= cohort_bound_km, and no TG in the population "
                "has a gap in (0, population_min_positive_gap_km). When "
                "cohort_bound_km is below that floor, every gap in the cohort "
                "is zero by arithmetic and the zero share is not a finding."
            ),
        }
    return out


def _curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """`_gap_curve` carried to the right edge at its final height.

    A CDF that has reached 100% is flat thereafter, so the extension adds no
    claim. Without it an all-zero cohort is a single point at `X_MIN_KM` and
    draws as a lone marker, which reads as a plotting fault rather than as the
    result that every TG in the cohort has a zero gap.
    """
    x, y, zero_share = _gap_curve(values)
    return (
        np.concatenate([x, [X_MAX_KM]]),
        np.concatenate([y, [y[-1]]]),
        zero_share,
    )


def plot(frames: dict[str, pd.DataFrame], *, meta: dict, out_png: Path) -> Path:
    """The three-scope gap panel, sized to match its sibling figure.

    The left-edge marker and its share are drawn per curve: without them a
    reader sees three curves begin at three unexplained heights and reads a
    clipped axis. The caption must still say that the intercept *is* the zero
    share, since the marker alone does not say what it means.
    """
    fig, ax = plt.subplots(figsize=(4.6, 2.5))

    reference_term = method_label(meta["reference"])
    for scope, frame in frames.items():
        values = frame[GAP].to_numpy()
        x, y, zero_share = _curve(values)
        label = "all TGs" if scope == ALL else f"{reference_term} {scope}"
        colour = SCOPE_HUES.get(scope, SCOPE_HUES[ALL])
        ax.step(x, y, where="post", color=colour, ls="-", lw=1.4,
                label=f"{label}  ($n$={len(values):,})")
        ax.plot([X_MIN_KM], [zero_share], marker="o", ms=3.5, color=colour,
                clip_on=False, zorder=5)
        ax.text(X_MIN_KM * 1.18, zero_share - 2.0, f"{zero_share:.1f}%",
                fontsize=7, color=colour, ha="left", va="top", zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(X_MIN_KM, X_MAX_KM)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"$d_\mathrm{sp}-d_\mathrm{geo}$ (km)", fontsize=8)
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


def _manifest(meta: dict, stats: pd.DataFrame, forced: dict[str, dict]) -> str:
    by_scope = stats.set_index("scope")
    return json.dumps(
        {
            "figure": PNG_NAME,
            "csv": CSV_NAME,
            "kind": KIND,
            "run_ids": meta["run_ids"],
            "nside": meta["nside"],
            "reference": meta["reference"],
            "reference_term": method_label(meta["reference"]),
            "n_tgs": meta["n_tgs"],
            "scope": (
                "one curve per scope, nested: all TGs, then the reference's "
                "most accurate 25% and 5%. Cohorts come from "
                "figure_vp_proximity.cohort_frame, so solved rows only and k "
                "from the pooled TG count; the gap itself is a TG property "
                "and no mask applies to it."
            ),
            "max_gap_km": {
                scope: float(by_scope.loc[scope, "max_km"]) for scope in by_scope.index
            },
            "zero_gap_share_pct": {
                scope: float(by_scope.loc[scope, "zero_share_pct"])
                for scope in by_scope.index
            },
            "forced_zero_gap": forced,
            "quote_this": (
                "max_gap_km per cohort, against max_gap_km['all'] and the "
                "population median. The zero share is the weaker summary: at "
                "p5 it is forced by the cohort bound (see forced_zero_gap), "
                "and at p25 it understates a cohort whose nonzero gaps are "
                "all within a few km of zero."
            ),
            "quantization": {
                "n_distinct": {
                    scope: int(by_scope.loc[scope, "n_distinct"]) for scope in by_scope.index
                },
                "note": (
                    "~20 IP replicas share a site and so share its VP geometry "
                    "exactly. Read every n beside its n_distinct: a cohort of "
                    "317 TGs carrying 3 gap values is ~3 independent "
                    "observations."
                ),
            },
            "x_axis": {
                "scale": "log",
                "min_km": X_MIN_KM,
                "max_km": X_MAX_KM,
                "zero_mass_note": (
                    "log(0) does not exist. Each curve begins at min_km "
                    "already carrying its zero share; read the left intercept "
                    "as that share, not as a value at min_km."
                ),
            },
        },
        indent=2,
    )


def build_for_runs(
    runs: list[RunPaths],
    *,
    methods: list[str] | None = None,
    reference: str = SHORTEST_PING,
    cohorts: tuple[str, ...] = DEFAULT_COHORTS,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
    source_csv: dict[str, Path] | None = None,
) -> list[Path]:
    """PNG, stats CSV and manifest. Returns the PNG in a list, as siblings do."""
    nside = G.validate_nside(nside)
    long, meta = load(runs, methods=methods, nside=nside, analysis_root=analysis_root,
                      source_csv=source_csv)
    meta = {**meta, "reference": reference}
    frames = scopes(long, reference=reference, cohorts=cohorts)
    stats = stats_table(frames)
    forced = forced_zero_gap(frames, cohort_bounds(long, reference=reference, cohorts=cohorts))

    out_dir = output_dir(meta["run_ids"], analysis_root=analysis_root)
    stats.to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(_manifest(meta, stats, forced))
    return [plot(frames, meta=meta, out_png=out_dir / PNG_NAME)]
