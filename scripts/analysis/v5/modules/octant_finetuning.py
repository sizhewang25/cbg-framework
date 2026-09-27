"""Octant weight-scorer sweep: the error CDF and the paired per-target table.

Two artifact sets, deliberately separate, because they answer different
questions and only one of them is paired.

## What the sweep varies

`planar_annulus_weighted` scores each face of the annulus arrangement by the
SUM of the weights of the annuli covering it and, under
`highest_weight_only`, returns the heaviest face. The weight was fixed at
`exp(-rtt/50ms)`, never calibrated. `configs/as0*-260728-260802-wsweep.yaml`
sweeps it over three families -- uniform, inverse-power `rtt^-k` and
negative-exponential `exp(-rtt/tau)` -- on both Octant variants.

Arms are ordered and reported by an **example ratio**,
`w(REF_FAST_MS)/w(REF_SLOW_MS)`, not by parameter value. That ratio is the
quantity the count-imbalance argument is about -- a face needs its few narrow
annuli to outweigh the surplus of wide ones -- and it is the only axis on
which the two families are commensurable. It is an *example* because it
depends on the interval it is read over as well as on the scorer; see
`REF_FAST_MS`.

## The CDF says which GROUP an arm is in; it cannot say what it costs

`plot_cdf` draws every arm's error distribution. That is the right view for
"does the weight function matter at all": the arms fall into three visually
distinct groups and the gap between groups is the headline. It is the wrong
view for choosing among the steep arms, for two reasons.

They overlap -- on as01 OCT-H the six steep arms sit within 0.5 km of each
other at p25 and 4.8 km at p50, so they render as one thick line. That is the
honest message, since a site-clustered bootstrap cannot separate them either,
but a reader takes it for a rendering fault unless the caption says so.

Worse, a CDF is UNPAIRED. It compares marginal distributions, so it cannot see
that `1/rtt` beats tau=50 on 107 targets and loses on 4 while `exp(-rtt/1)`
beats it on 160 and loses on 34, once by 5,354 km. Those two arms have
indistinguishable CDFs and very different risk.

## The table is the paired half, and it is where the decision lives

`build_table` reports, per arm against a baseline, the share of targets that
improve / regress / are unchanged, and the magnitude of each conditional on
moving that way. On as01 the regression column orders the steep arms
monotonically by steepness -- `1/rtt` breaks 4 targets by at most 74 km,
`exp(-rtt/1)` breaks 34 by up to 5,354 -- the only column in either artifact
that separates arms the CDF and the bootstrap both call tied.

### Ties are a first-class outcome, not a rounding artifact

50-72% of targets get a bit-identical answer under every weight function: the
arrangement's top face does not move. Ties are their own column rather than
folded into "not better", because the useful sentence is the win:loss ratio
among targets that DID move (`1/rtt`, 107:4), not the raw improvement share.

### Gains and regressions are reported symmetrically, with `n` beside them

Both sides carry p50 and p95, so an arm's upside and downside are read off the
same statistics. The cohorts are very unequal, though -- on as01 gains run
107-198 targets and regressions 4-34 -- so `n` sits next to every percentile.
At n=4 a p95 interpolates between a handful of points and is the maximum under
another name; the CSV carries `gain_max`/`reg_max` so that reading is checkable
rather than implied.

## Why this reads the benchmark tree directly

`error_km` comes from each combo's `targets.parquet`, not from `classify`'s
`*_tgs.parquet`. The sweep compares weight functions on distance to the raw TG
coordinate; no answer space enters, so requiring `classify` first would add a
dependency the question does not have. Same precedent as `figure_ltd_model`,
which reads `fit_checkpoint.pkl` from the benchmark tree.

## Scope: the arms the sweep config declares, and nothing else

Every artifact here is drawn from one `-wsweep` run. The shipped
`exp(-rtt/50)` lives in a different run id (the parent mesh run) and is
deliberately absent: mixing a combo from another run into a figure titled
after this one invites the reader to assume both came from the same config.

The consequence is worth stating plainly, because it bounds what these
artifacts can claim: they show what the weight FUNCTION buys over not
weighting at all, not that the shipped default is mis-set. That second claim
needs the tau=50 arm and has to be made from the parent run's own numbers.

`unweighted` is therefore the baseline for every paired comparison -- it is
the sweep's own control, scored under the same config, same folds, same
code.

Commands: `plot-octant-cdf`, `report-octant-finetuning`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.analysis.v5.modules.paths import (
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
)

#: Reference RTTs for the EXAMPLE RATIO, `w(REF_FAST)/w(REF_SLOW)`: the factor
#: by which a scorer prefers a 0.5 ms vantage point over a 5 ms one.
#:
#: The ratio is a property of the scorer AND the interval it is read over, not
#: of the scorer alone, because the two families key off different things --
#: `exp(-rtt/tau)` off the DIFFERENCE `REF_SLOW - REF_FAST`, `rtt^-k` off the
#: RATIO `REF_SLOW / REF_FAST`. Moving the window therefore reorders the arms:
#: over (3, 55) ms, which is this dataset's own near/far contrast, `exp(-rtt/1)`
#: is the extreme at 3.8e22; over (0.5, 5) it is 90, between `1/rtt` and
#: `1/rtt^2`. Both orderings are correct for their window, so the window is
#: named in the column header, the manifest and the caption -- an "example"
#: ratio, not a constant of the method.
REF_FAST_MS = 0.5
REF_SLOW_MS = 5.0


def example_ratio(tag: str) -> float:
    """`w(REF_FAST_MS) / w(REF_SLOW_MS)` for one arm."""
    if tag == "unw":
        return 1.0
    if tag.startswith("ip"):
        return (REF_SLOW_MS / REF_FAST_MS) ** float(tag[2:])
    return math.exp((REF_SLOW_MS - REF_FAST_MS) / float(tag[3:]))


#: Sweep suffix -> (label, family). The example ratio is derived, not stored,
#: so changing the reference window cannot leave a stale literal behind.
_ARM_SPEC: tuple[tuple[str, str, str], ...] = (
    ("unw",   "unweighted",   "control"),
    ("ip1",   "1/rtt",        "inverse-power"),
    ("ip2",   "1/rtt^2",      "inverse-power"),
    ("ip3",   "1/rtt^3",      "inverse-power"),
    ("tau10", "exp(-rtt/10)", "neg-exponential"),
    ("tau5",  "exp(-rtt/5)",  "neg-exponential"),
    ("tau1",  "exp(-rtt/1)",  "neg-exponential"),
)
ARMS: tuple[tuple[str, str, float, str], ...] = tuple(
    (t, l, example_ratio(t), f) for t, l, f in _ARM_SPEC
)
ARM_TAGS: tuple[str, ...] = tuple(t for t, _, _, _ in ARMS)
ARM_LABEL: dict[str, str] = {t: l for t, l, _, _ in ARMS}
ARM_RATIO: dict[str, float] = {t: d for t, _, d, _ in ARMS}
ARM_FAMILY: dict[str, str] = {t: f for t, _, _, f in ARMS}

#: Default baseline for every paired comparison: the sweep's own control,
#: scored under the same config, folds and code as the arms it anchors.
BASELINE_TAG = "unw"
BASELINES: tuple[str, ...] = ARM_TAGS

#: The two Octant combos the sweep forks, and their published terms.
VARIANTS: tuple[tuple[str, str], ...] = (
    ("octant_cbg_hull", "OCT-H"),
    ("octant_cbg_spl", "OCT-S"),
)

#: The arm the paired table supports adopting: tied with the rest on p50 and
#: p90, with by far the mildest regression profile, and the only steep family
#: member scale-free in RTT (`rtt -> c*rtt` scales every weight by one
#: constant, which the face argmax discards).
#:
#: Deliberately NOT marked on the figure. A CDF cannot show what an arm costs
#: on the targets it makes worse, which is the whole basis for preferring this
#: one, so flagging it there would assert a conclusion the panel does not
#: carry. Readers of the module and the manifest still get the pointer.
PICK = "ip1"

# ---- house style -------------------------------------------------------------
# Matches figure_error_cdf.py: same surface, inks, grid and panel size, so this
# figure sits beside the package's others without a second visual language.

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

#: COMPOSITE ENCODING: hue carries the decay family, dash pattern carries the
#: steepness rank within it. Six distinct hues was tried first and does not
#: validate -- three lightness steps per hue cannot clear the normal-vision
#: floor (dE 15) while staying inside the L 0.43-0.77 band; the best attempt
#: reached dE 10.5 between adjacent blues. The dataviz remedy for that is to
#: cut series, facet, or add a secondary encoding, and a secondary encoding is
#: the one that keeps all six arms on one panel.
#:
#: It also earns its place: the paper's claim is that the two families land on
#: one curve when ordered by the example ratio, so hue=family and dash=rank lets a
#: reader see the interleaving directly.
#:
#: The two hues are v5's validated palette entries, re-checked with the dataviz
#: validator at `--pairs all` on a #ffffff surface: CVD separation dE 21.6
#: (protan), normal-vision dE 32.3, contrast >= 3:1 -- all PASS.
FAMILY_HUE: dict[str, str] = {
    "inverse-power": "#2a78d6",
    "neg-exponential": "#e34948",
    "control": _INK_2,
}

#: Steepness rank within a family -> dash pattern, shallowest first. Shared
#: across families on purpose, so the pattern reads as "how steep" and the hue
#: as "which family" rather than the two being entangled.
RANK_DASH: tuple[tuple[int, ...] | None, ...] = (
    None,          # solid
    (5, 1.6),      # dashed
    (1.4, 1.4),    # dotted
)

_TITLE_PT, _SUBTITLE_PT, _LABEL_PT = 9.0, 6.0, 8.0
_TICK_PT, _LEGEND_PT = 7.0, 6.5
#: Wider than figure_error_cdf.py's 4x3: seven arms need seven legend rows, and
#: at the shared _LEGEND_PT a two-column key does not fit a single column's
#: width without overlapping the curves.
PAPER_FIGSIZE: tuple[float, float] = (5.6, 3.4)
Y_TICKS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
X_MIN_KM = 1.0
X_LABEL = "error distance (km)"
Y_LABEL = "fraction of targets"

#: Percentiles the CSV publishes per arm.
PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 90, 95)


def artifact_names(variant: str, baseline: str, kind: str) -> tuple[str, ...]:
    """`(png, csv, manifest)` for the CDF, `(csv, tex, manifest)` for the table.

    The baseline is in the table's stem because it changes the claim, not the
    view: two baselines are two different results and must not overwrite each
    other. The CDF takes no baseline, so its stem carries none.
    """
    if kind == "cdf":
        stem = f"error_cdf.{variant}"
        return (f"{stem}.png", f"{stem}.csv", f"{stem}.manifest.json")
    stem = f"paired.{variant}.vs-{baseline}"
    return (f"{stem}.csv", f"{stem}.tex", f"{stem}.manifest.json")


# ---- loading -----------------------------------------------------------------


def load_combo(run: RunPaths, combo: str) -> pd.DataFrame | None:
    """One combo's per-target rows, pooled over every fold that has FINISHED.

    A fold counts only when its `run.json` exists: `targets.parquet` is written
    a row group per target as the runner streams, so a mid-flight fold is on
    disk and short, and reading it would silently shrink the population.
    """
    frames = []
    for fold_id in run.fold_ids:
        cdir = run.combo_dir(combo, fold_id)
        if not (cdir / "run.json").exists() or not (cdir / "targets.parquet").exists():
            continue
        d = pd.read_parquet(cdir / "targets.parquet",
                            columns=["target_id", "target_lat", "target_lon", "error_km"])
        d["fold"] = fold_id
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def assemble(run: RunPaths, variant: str) -> tuple[pd.DataFrame, list[str]]:
    """Wide frame: one row per target, one column per scored arm.

    Every column comes from this one `-wsweep` run -- no combo is pulled in
    from another run id. Inner-joined across arms, so a half-finished sweep
    narrows every comparison to the slowest arm's targets rather than
    comparing different populations. The caller gets the arm list so it can
    say which arms, and how complete.
    """
    wide: pd.DataFrame | None = None
    present: list[str] = []
    for tag in ARM_TAGS:
        arm = load_combo(run, f"{variant}_{tag}")
        if arm is None:
            continue
        if wide is None:
            arm = arm.copy()
            arm["site"] = (arm.target_lat.round(4).astype(str) + ","
                           + arm.target_lon.round(4).astype(str))
            wide = arm[["target_id", "site", "fold", "error_km"]].rename(
                columns={"error_km": tag})
        else:
            wide = wide.merge(
                arm[["target_id", "error_km"]].rename(columns={"error_km": tag}),
                on="target_id", how="inner")
        present.append(tag)
    if wide is None:
        raise MissingArtifactError(
            f"no sweep arms scored for {variant} under {run.run_id}; run "
            f"./cli.sh --configfile configs/{run.run_id}.yaml first")
    return wide, present


def fold_coverage(run: RunPaths, variant: str) -> dict[str, int]:
    """Finished folds per arm, so a partial run can be declared rather than
    quietly narrowing the inner join in `assemble`."""
    out = {}
    for tag in ARM_TAGS:
        d = load_combo(run, f"{variant}_{tag}")
        out[tag] = 0 if d is None else d.fold.nunique()
    return out


# ---- artifact 1: the CDF -----------------------------------------------------


def rank_within_family(tag: str) -> int:
    """Steepness rank of an arm among its own family, shallowest = 0."""
    fam = ARM_FAMILY[tag]
    sibs = sorted((t for t in ARM_TAGS if ARM_FAMILY[t] == fam), key=ARM_RATIO.get)
    return sibs.index(tag)


def _style_for(tag: str) -> dict:
    """Hue by family, dash by steepness rank -- never by position in a list,
    so dropping an arm cannot repaint the survivors."""
    if ARM_FAMILY[tag] == "control":
        # The control is the baseline every table pairs against, so it is drawn
        # recessive and heavier: a reference, not a competitor.
        return dict(color=FAMILY_HUE["control"], linewidth=1.7,
                    dashes=(3, 1.4, 1, 1.4), zorder=5)
    # Every scorer arm is drawn identically apart from hue and dash: the
    # figure reports the sweep, it does not argue for a member of it. Which
    # arm to adopt is a call the paired table supports and this panel cannot,
    # since a CDF cannot show what an arm costs where it loses.
    style = dict(color=FAMILY_HUE[ARM_FAMILY[tag]], linewidth=1.5, zorder=4)
    dash = RANK_DASH[rank_within_family(tag)]
    if dash is not None:
        style["dashes"] = dash
    return style


def cdf_table(wide: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """The CSV twin: percentiles per arm, so the figure carries no footnote."""
    rows = []
    for tag in present:
        v = wide[tag]
        row = {
            "arm": ARM_LABEL[tag],
            "arm_tag": tag,
            "family": ARM_FAMILY[tag],
            "example_ratio": ARM_RATIO[tag],
            "n": len(v),
        }
        row.update({f"p{p}_km": v.quantile(p / 100) for p in PERCENTILES})
        row["lt40km_pct"] = 100 * (v < 40).mean()
        row["lt100km_pct"] = 100 * (v < 100).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("example_ratio").reset_index(drop=True)


def plot_cdf(wide: pd.DataFrame, present: list[str], term: str, run_id: str,
             out_png: Path, *, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    def curve(v):
        s = np.sort(v)
        return s, np.arange(1, len(s) + 1) / len(s)

    # Drawn shallowest-first so the steep arms, which sit left, land on top.
    for tag in sorted(present, key=ARM_RATIO.get):
        ax.plot(*curve(wide[tag].values), **_style_for(tag))

    ax.set_xscale("log")
    # Upper bound from the data: every curve reaches 1.0 by ~2,000 km on as0*,
    # and a round 20,000 leaves most of a decade of empty panel squeezing the
    # part a reader compares.
    hi = max(wide[t].max() for t in present)
    ax.set_xlim(X_MIN_KM, 10 ** math.ceil(math.log10(hi)))
    ax.set_ylim(0, 1)
    ax.set_yticks(Y_TICKS)
    ax.set_xlabel(X_LABEL, fontsize=_LABEL_PT, color=_INK_2)
    ax.set_ylabel(Y_LABEL, fontsize=_LABEL_PT, color=_INK_2)
    ax.grid(True, which="major", color=_GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=_MUTED, labelsize=_TICK_PT)
    for sp in ax.spines.values():
        sp.set_color(_AXIS)
        sp.set_linewidth(0.8)

    # Legend grouped by family, each family shallowest-first, so the dash
    # pattern reads as a steepness ladder down each column.
    order = [t for t in present if ARM_FAMILY[t] == "control"]
    for fam in ("inverse-power", "neg-exponential"):
        order += sorted((t for t in present if ARM_FAMILY[t] == fam),
                        key=ARM_RATIO.get)
    # TEX_LABEL's `$...$` forms render natively as matplotlib mathtext, so the
    # legend and the LaTeX table typeset the same expression -- `1/rtt^2` shows
    # a real superscript rather than a literal caret.
    handles = [
        Line2D([], [], label=TEX_LABEL.get(ARM_LABEL[t], ARM_LABEL[t]),
               **_style_for(t))
        for t in order
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=_LEGEND_PT,
                    frameon=False, handlelength=2.6, ncol=2,
                    columnspacing=1.1, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(_INK_2)

    ax.set_title(f"{term}: error CDF across weight scorers",
                 fontsize=_TITLE_PT, fontweight="bold", color=_INK, pad=9)
    # Kept short enough for the panel: at _SUBTITLE_PT a longer string overruns
    # the axes and bbox_inches="tight" clips it. Hue=family and dash=steepness
    # belong in the caption and the manifest, which have room.
    ax.text(0.5, 1.005, f"{run_id} · n={len(wide)} targets",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=_SUBTITLE_PT, color=_INK_2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)


# ---- artifact 2: the paired table --------------------------------------------


def build_table(wide: pd.DataFrame, present: list[str], *,
                baseline: str = BASELINE_TAG) -> pd.DataFrame:
    if baseline not in present:
        raise ValueError(f"baseline {baseline!r} not among scored arms {present}")
    rows = []
    for tag in [t for t in present if t != baseline]:
        d = wide[tag] - wide[baseline]          # negative = improvement
        gain, reg = -d[d < 0], d[d > 0]
        rows.append({
            "arm": ARM_LABEL[tag],
            "arm_tag": tag,
            "example_ratio": ARM_RATIO[tag],
            "p50_km": wide[tag].median(),
            "better_pct": 100 * (d < 0).mean(),
            "worse_pct": 100 * (d > 0).mean(),
            "tie_pct": 100 * (d == 0).mean(),
            "n_gain": len(gain),
            "gain_p50": gain.quantile(.50) if len(gain) else np.nan,
            "gain_p95": gain.quantile(.95) if len(gain) else np.nan,
            "gain_max": gain.max() if len(gain) else np.nan,
            "n_reg": len(reg),
            "reg_p50": reg.quantile(.50) if len(reg) else np.nan,
            "reg_p95": reg.quantile(.95) if len(reg) else np.nan,
            "reg_max": reg.max() if len(reg) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("example_ratio").reset_index(drop=True)


#: Plain label -> LaTeX. The plain forms carry a bare `^`, which is math-mode
#: only: emitted as-is the table does not compile ("Missing $ inserted").
#: Written out rather than escaped programmatically so the typeset form is
#: legible here and `rtt` stays upright inside math mode.
TEX_LABEL: dict[str, str] = {
    "unweighted":   r"unweighted",
    "1/rtt":        r"$1/\mathrm{rtt}$",
    "1/rtt^2":      r"$1/\mathrm{rtt}^{2}$",
    "1/rtt^3":      r"$1/\mathrm{rtt}^{3}$",
    "exp(-rtt/1)":  r"$e^{-\mathrm{rtt}/1}$",
    "exp(-rtt/5)":  r"$e^{-\mathrm{rtt}/5}$",
    "exp(-rtt/10)": r"$e^{-\mathrm{rtt}/10}$",
}


def tex_range(v: float) -> str:
    """Dynamic range in LaTeX. Scientific notation past 4 digits, where the
    plain form (`3.83e+22`) is neither readable nor typeset maths."""
    if v < 1e4:
        return f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    exp = int(math.floor(math.log10(v)))
    return rf"${v / 10 ** exp:.1f}\times10^{{{exp}}}$"


def to_latex(t: pd.DataFrame, term: str, run_id: str, baseline_label: str,
             n_targets: int) -> str:
    L = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{term}: per-target effect of the face-weight scorer on "
        rf"\texttt{{{run_id}}}, paired against {TEX_LABEL.get(baseline_label, baseline_label)} "
        rf"over $n={n_targets}$ targets. Gains and regressions are conditional "
        r"on a target moving in that direction, and carry the same statistics "
        r"so the two sides read off one scale; note their cohorts differ "
        r"greatly in size, so $n$ is given beside each. Ties are targets whose "
        r"prediction is bit-identical under both scorers. Arms are ordered by "
        rf"the example ratio $w({REF_FAST_MS:g}\,\mathrm{{ms}})/"
        rf"w({REF_SLOW_MS:g}\,\mathrm{{ms}})$ -- how strongly each prefers a "
        r"near vantage point over a moderate one. That ratio depends on the "
        r"interval it is read over, not on the scorer alone.}",
        rf"\label{{tab:octant-weight-sweep-{term.lower()}-{baseline_label}}}".replace(
            "(", "").replace(")", "").replace("/", "").replace("-rtt", "rtt"),
        r"\begin{tabular}{lrrrrr rrr rrr}", r"\toprule",
        r"& & & \multicolumn{3}{c}{share of targets} "
        r"& \multicolumn{3}{c}{gain (km)} & \multicolumn{3}{c}{regression (km)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r"scorer & ex. ratio & p50 & better & worse & tie & $n$ & p50 & p95 "
        r"& $n$ & p50 & p95 \\",
        r"\midrule",
    ]
    for _, r in t.iterrows():
        def f(v):
            return "--" if pd.isna(v) else f"{v:.0f}"
        L.append(
            f"{TEX_LABEL.get(r['arm'], r['arm'])} & {tex_range(r.example_ratio)} & "
            f"{r.p50_km:.1f} & {r.better_pct:.1f}\\% & {r.worse_pct:.1f}\\% & "
            f"{r.tie_pct:.1f}\\% & {int(r.n_gain)} & {f(r.gain_p50)} & "
            f"{f(r.gain_p95)} & {int(r.n_reg)} & {f(r.reg_p50)} & "
            f"{f(r.reg_p95)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ---- manifests ---------------------------------------------------------------


def _common_manifest(run: RunPaths, variant: str, term: str,
                     wide: pd.DataFrame, present: list[str]) -> dict:
    coverage = fold_coverage(run, variant)
    complete = len(set(coverage[t] for t in present)) == 1
    return {
        "run_id": run.run_id,
        "variant": variant,
        "term": term,
        "n_targets": int(len(wide)),
        "n_sites": int(wide.site.nunique()),
        "arms_scored": present,
        "fold_coverage": {t: coverage[t] for t in present},
        "complete": complete,
        "partial_note": (
            None if complete else
            "arms have unequal fold coverage; every figure and table here is "
            "the INNER JOIN across arms, so it reports whatever the "
            "least-complete arm has reached. Directional only."
        ),
        "arm_ordering": (
            f"by the example ratio w({REF_FAST_MS:g}ms)/w({REF_SLOW_MS:g}ms), "
            f"not parameter value: that ratio is what the count-imbalance "
            f"argument is about, and it is the only axis on which the "
            f"inverse-power and negative-exponential families are "
            f"commensurable. It depends on the interval it is read over as "
            f"well as on the scorer -- over (3, 55) ms exp(-rtt/1) is the "
            f"extreme at 3.8e22, over ({REF_FAST_MS:g}, {REF_SLOW_MS:g}) it is "
            f"90, between 1/rtt and 1/rtt^2. Hence 'example'."
        ),
        "scope": (
            "every column comes from this one -wsweep run; no combo is pulled "
            "in from another run id. The shipped exp(-rtt/50) lives in the "
            "parent mesh run and is deliberately absent, so these artifacts "
            "show what the weight FUNCTION buys over not weighting at all -- "
            "not that the shipped default is mis-set, which needs that arm."
        ),
        "source": (
            "error_km from each combo's targets.parquet, not classify's "
            "*_tgs.parquet: the sweep compares weight functions on distance to "
            "the raw TG coordinate, so no answer space enters."
        ),
        "site_note": (
            f"{int(wide.site.nunique())} distinct coordinates behind "
            f"{len(wide)} targets -- the dataset holds ~20 IP replicas per "
            f"site, so any interval computed by resampling TARGETS rather than "
            f"sites is overstated by roughly sqrt(20)."
        ),
    }


def cdf_manifest(run, variant, term, wide, present, table) -> dict:
    body = _common_manifest(run, variant, term, wide, present)
    steep = [t for t in present if t != "unw"]
    spread = {
        f"p{p}": float(max(wide[t].quantile(p / 100) for t in steep)
                       - min(wide[t].quantile(p / 100) for t in steep))
        for p in (25, 50, 75, 90)
    } if steep else {}
    body.update({
        "artifact": "error CDF",
        "encoding": {
            "rule": "hue = decay family; dash pattern = steepness rank within it",
            "family_hue": dict(FAMILY_HUE),
            "why_composite": (
                "six distinct hues was tried and does not validate: three "
                "lightness steps per hue cannot clear the normal-vision floor "
                "(dE 15) inside the L 0.43-0.77 band, best attempt dE 10.5 "
                "between adjacent blues. A secondary encoding is the remedy "
                "that keeps all arms on one panel, and it also shows the "
                "families interleaving on the example-ratio axis."
            ),
            "validated": (
                "dataviz validate_palette.js --pairs all, light, surface "
                "#ffffff, on the two family hues: CVD dE 21.6 (protan), "
                "normal-vision dE 32.3, contrast >= 3:1 -- all PASS."
            ),
        },
        "steep_arm_spread_km": spread,
        "overlap_note": (
            "the steep arms overlap because they are tied, not because the "
            "figure is broken; a site-clustered bootstrap cannot separate them "
            "either. Say so in the caption."
        ),
        "unpaired_warning": (
            "a CDF compares marginal distributions, so it cannot show what an "
            "arm costs on the targets it makes worse -- the steep arms differ "
            "sharply there while their curves nearly coincide. See the paired "
            "table."
        ),
        "percentiles": table.to_dict(orient="records"),
        "panel": {"figsize_in": list(PAPER_FIGSIZE), "x_label": X_LABEL,
                  "y_label": Y_LABEL, "y_ticks": list(Y_TICKS), "x_scale": "log"},
    })
    return body


def table_manifest(run, variant, term, wide, present, baseline, table) -> dict:
    body = _common_manifest(run, variant, term, wide, present)
    body.update({
        "artifact": "paired per-target table",
        "baseline": ARM_LABEL[baseline],
        "baseline_tag": baseline,
        "baseline_choice": (
            "the sweep's own control, scored under the same config, folds and "
            "code as the arms it anchors. It answers 'what does the weight "
            "function buy over not weighting at all'."
        ),
        "tie_note": (
            "ties are targets whose prediction is bit-identical under both "
            "scorers -- the arrangement's top face does not move. They run "
            "50-72% here, so the useful statistic is the win:loss ratio among "
            "targets that DID move, not the raw improvement share."
        ),
        "symmetry_note": (
            "gains and regressions carry the same statistics (n, p50, p95, "
            "max) so an arm's upside and downside are read off one scale. The "
            "cohorts are not comparable in size -- on as01 gains run 107-198 "
            "targets and regressions 4-34 -- so n is reported beside every "
            "percentile: at n=4 a p95 is the maximum under another name, and "
            "the max column makes that checkable."
        ),
        "rows": table.to_dict(orient="records"),
    })
    return body


# ---- orchestration -----------------------------------------------------------


def write_cdf(run: RunPaths, variant: str, *, analysis_root: Path | None = None,
              dpi: int = 300) -> Path:
    term = dict(VARIANTS)[variant]
    wide, present = assemble(run, variant)
    table = cdf_table(wide, present)

    png_name, csv_name, man_name = artifact_names(variant, BASELINE_TAG, "cdf")
    out = run.octant_finetuning_dir(root=analysis_root)
    plot_cdf(wide, present, term, run.run_id, out / png_name, dpi=dpi)
    table.to_csv(out / csv_name, index=False)
    (out / man_name).write_text(json.dumps(
        cdf_manifest(run, variant, term, wide, present, table), indent=2))
    return out / png_name


def write_table(run: RunPaths, variant: str, *, baseline: str = BASELINE_TAG,
                analysis_root: Path | None = None) -> Path:
    term = dict(VARIANTS)[variant]
    wide, present = assemble(run, variant)
    t = build_table(wide, present, baseline=baseline)

    csv_name, tex_name, man_name = artifact_names(variant, baseline, "table")
    out = run.octant_finetuning_dir(root=analysis_root)
    t.to_csv(out / csv_name, index=False)
    (out / tex_name).write_text(
        to_latex(t, term, run.run_id, ARM_LABEL[baseline], len(wide)))
    (out / man_name).write_text(json.dumps(
        table_manifest(run, variant, term, wide, present, baseline, t), indent=2))
    return out / csv_name
