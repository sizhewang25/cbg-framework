"""Octant parameter sweeps: the error CDF and the paired per-target table.

Two artifact sets, deliberately separate, because they answer different
questions and only one of them is paired.

## The two sweeps this module serves

Both vary one knob of the Octant stack against an in-tree control, and both
report it the same two ways, so they share every line below the `Sweep`
registry. What differs between them is data, not code -- see `Sweep`.

**The weight-scorer sweep** (`octant_cbg_hull` / `octant_cbg_spl`,
`configs/as0*-260728-260802-wsweep.yaml`). `planar_annulus_weighted` scores
each face of the annulus arrangement by the SUM of the weights of the annuli
covering it and, under `highest_weight_only`, returns the heaviest face. The
weight was fixed at `exp(-rtt/50ms)`, never calibrated. The sweep runs it over
three families -- uniform, inverse-power `rtt^-k` and negative-exponential
`exp(-rtt/tau)` -- against an unweighted control, on both Octant variants.

Its arms are ordered and reported by an **example ratio**,
`w(REF_FAST_MS)/w(REF_SLOW_MS)`, not by parameter value. That ratio is the
quantity the count-imbalance argument is about -- a face needs its few narrow
annuli to outweigh the surplus of wide ones -- and it is the only axis on
which the two families are commensurable. It is an *example* because it
depends on the interval it is read over as well as on the scorer; see
`REF_FAST_MS`.

**The spline-coverage sweep** (`octant_ssw`,
`configs/as0*-260728-260802-ssweep.yaml`). `bounded_spline` widens a per-VP
LSQ spline to `(spline/delta, spline*delta)` and clips it to the convex hull,
picking `delta` so the band contains `target_coverage` of that VP's own
training samples. That target was likewise fixed, at 0.9, and never
calibrated. The sweep runs 0.95 / 0.90 / 0.75 / 0.50 against a control that
fits no spline at all -- Octant's bare hull, which is the live comparison
because at the shipped coverage the spline variant is WORSE than the hull on
all three datasets, at p50 and p90 both.

Its arms are ordered by the coverage itself, DESCENDING: coverage rises with
looseness, so that runs the arms from the loosest bound to the tightest, with
the hull baseline as the limit of the axis rather than a point on it.

One caveat belongs with every number this sweep publishes. Each arm is a
BLEND of "spline at this coverage" and "hull": `find_delta_for_coverage`
demands `|coverage(delta) - target| <= 0.01`, `BoundedSplineLTD._fit` swallows
the failure at debug level, and that VP silently reverts to bare hull bounds.
Since the baseline IS the hull, this pulls every arm toward the baseline, so
the measured spline effect is a LOWER BOUND. The manifest says so under
`confound_note`; the per-arm fallback count has to be published beside the
error stats.

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

Every artifact here is drawn from ONE run. `assemble` globs `f"{variant}_{tag}"`
under a single `RunPaths` and inner-joins on `target_id`; no combo is ever
pulled in from another run id. Mixing a combo from another run into a figure
titled after this one invites the reader to assume both came from the same
config.

Each sweep therefore carries its control IN-TREE, and that control is the
baseline for every paired comparison -- scored under the same config, the same
folds and the same code as the arms it anchors.

For the weight sweep the control is `unweighted`, and the scope bounds what
its artifacts can claim: they show what the weight FUNCTION buys over not
weighting at all, not that the shipped default is mis-set. The shipped
`exp(-rtt/50)` lives in a different run id (the parent mesh run) and is
deliberately absent, so that second claim needs the tau=50 arm and has to be
made from the parent run's own numbers.

For the spline sweep the control is `nospl`, Octant's bare hull. It is
recomputed in-tree rather than joined from the pro mesh for exactly this
reason -- which also buys a free correctness gate, since it must come out
bit-identical to the wsweep's `octant_cbg_hull_ip2` arm, whose kwargs and
folds it shares.

Commands: `plot-octant-cdf`, `report-octant-finetuning`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules.paths import (
    DEFAULT_OUTPUTS_ROOT,
    OCTANT_FINETUNING_KIND,
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
    """`w(REF_FAST_MS) / w(REF_SLOW_MS)` for one WEIGHT-SWEEP arm.

    The weight sweep's ordering scalar, and only that one's. It stays derived
    rather than stored so that moving the reference window cannot leave a
    stale literal behind (see REF_FAST_MS) -- which is exactly why it has to
    REFUSE another sweep's tag rather than parse one. The final branch used to
    be an unguarded `float(tag[3:])`, so a spline-coverage tag fell through it:
    `example_ratio("cov95")` returned `exp(4.5/95) = 1.0485`, a
    plausible-looking number that ordered the coverage arms backwards and
    collapsed all four into one indistinguishable cluster. Sweeps whose arms
    are not weight functions carry an explicit `Arm.order` instead.
    """
    if tag == "unw":
        return 1.0
    if tag.startswith("ip"):
        return (REF_SLOW_MS / REF_FAST_MS) ** float(tag[2:])
    if tag.startswith("tau"):
        return math.exp((REF_SLOW_MS - REF_FAST_MS) / float(tag[3:]))
    raise ValueError(
        f"example_ratio is the weight sweep's ordering scalar and knows only "
        f"unw / ipK / tauN; got {tag!r}. Another sweep's arms carry an "
        f"explicit Arm.order.")


# ---- house style -------------------------------------------------------------
# Matches figure_error_cdf.py: same surface, inks, grid and panel size, so this
# figure sits beside the package's others without a second visual language.
#
# Defined above the sweep specs because `Sweep.family_hue` names `_INK_2`.

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"


# ---- the sweeps --------------------------------------------------------------
# This module serves TWO sweeps over the same two artifacts, and everything
# that differs between them lives in one `Sweep` value rather than in
# module-level dicts. That is not tidiness. `ARM_TAGS`/`ARM_FAMILY`/
# `FAMILY_HUE` and friends used to be flat, and a flat namespace fails three
# ways at once here: tags from one sweep get rendered with the other's
# semantics (`example_ratio("cov95")` returning 1.0485); `cdf_manifest`
# SERIALIZES `family_hue`, so merely adding an entry for a second sweep
# rewrites the FIRST sweep's published manifest bytes; and the CLI's combo
# discovery matched on `split("_")[-1] in ARM_TAGS`, which a union of tags
# turns into a false positive on every `octant_cbg_hull` on disk.


@dataclass(frozen=True)
class Arm:
    """One arm of a sweep: a combo-id suffix plus how to render and order it."""

    tag: str      # combo id is f"{variant}_{tag}"
    label: str    # the CSV `arm` column, the legend, the caption
    order: float  # the ordering scalar; see Sweep.order_ascending
    family: str   # hue bucket; exactly one family per sweep is "control"


@dataclass(frozen=True)
class Sweep:
    """Everything that differs between the sweeps this module serves.

    Byte-identity note for anyone editing a string here: the published
    weight-sweep artifacts are a regression gate, so every `SWEEP_WEIGHT`
    field below is the pre-registry literal COPIED, not retyped. The prose
    blocks in particular are compared verbatim.
    """

    key: str
    variants: tuple[tuple[str, str], ...]   # (variant prefix, published term)
    arms: tuple[Arm, ...]
    baseline: str
    pick: str | None
    families: tuple[str, ...]               # LEGEND order; "control" first
    #: SERIALIZED into the CDF manifest, so its insertion order is
    #: load-bearing, and it is deliberately NOT `families`: the legend puts
    #: control first, the manifest kept it last.
    family_hue: dict[str, str]
    rank_dash: tuple[tuple[float, ...] | None, ...]
    order_column: str                       # CSV header for `Arm.order`
    order_ascending: bool
    order_tex_header: str
    order_tex_fmt: Callable[[float], str]
    row_tex_header: str                     # the table's first column head
    title: str                              # figure title, after "{term}: "
    tex_label_prefix: str
    tex_labels: dict[str, str]
    caption_effect: str                     # what the sweep varies
    caption_tie: str                        # what a tie means here
    caption_ordering: str
    prose: dict[str, str]

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(a.tag for a in self.arms)

    def arm(self, tag: str) -> Arm:
        for a in self.arms:
            if a.tag == tag:
                return a
        raise ValueError(f"{tag!r} is not an arm of the {self.key} sweep")

    def label(self, tag: str) -> str:
        return self.arm(tag).label

    def order(self, tag: str) -> float:
        return self.arm(tag).order

    def family(self, tag: str) -> str:
        return self.arm(tag).family

    def term(self, variant: str) -> str:
        return dict(self.variants)[variant]

    def ordered(self, tags) -> list[str]:
        """`tags` sorted along this sweep's ordering axis, gentlest first."""
        return sorted(tags, key=self.order, reverse=not self.order_ascending)

    def tex(self, label: str) -> str:
        """LaTeX form of an arm label, degrading to the plain label."""
        return self.tex_labels.get(label, label)

    def rank_within_family(self, tag: str) -> int:
        """Rank of an arm among its own family, gentlest = 0."""
        fam = self.family(tag)
        sibs = self.ordered(t for t in self.tags if self.family(t) == fam)
        return sibs.index(tag)

    def style_for(self, tag: str) -> dict:
        """Hue by family, dash by within-family rank -- never by position in a
        list, so dropping an arm cannot repaint the survivors."""
        if self.family(tag) == "control":
            # The control is the baseline every table pairs against, so it is
            # drawn recessive and heavier: a reference, not a competitor.
            return dict(color=self.family_hue["control"], linewidth=1.7,
                        dashes=(3, 1.4, 1, 1.4), zorder=5)
        # Every non-control arm is drawn identically apart from hue and dash:
        # the figure reports the sweep, it does not argue for a member of it.
        # Which arm to adopt is a call the paired table supports and this panel
        # cannot, since a CDF cannot show what an arm costs where it loses.
        style = dict(color=self.family_hue[self.family(tag)], linewidth=1.5,
                     zorder=4)
        dash = self.rank_dash[self.rank_within_family(tag)]
        if dash is not None:
            style["dashes"] = dash
        return style


def _validate(s: Sweep) -> None:
    """Refuse a malformed spec at IMPORT, not mid-write.

    Every check here stands for a failure that used to surface after the CSV
    and the manifest were already computed -- `RANK_DASH[3]` raising
    `IndexError` inside `plot_cdf` being the worst, since the artifact triple
    is half-written by then.
    """
    where = f"sweep {s.key!r}"
    if not s.families or s.families[0] != "control":
        raise ValueError(f"{where}: families must start with 'control'")
    if unknown := {a.family for a in s.arms} - set(s.families):
        raise ValueError(
            f"{where}: arms use undeclared families {sorted(unknown)}")
    if set(s.family_hue) != set(s.families):
        raise ValueError(
            f"{where}: family_hue keys {sorted(s.family_hue)} != families "
            f"{sorted(s.families)}; the dict is serialized, so a stray key "
            f"gets published")
    if s.baseline not in s.tags:
        raise ValueError(f"{where}: baseline {s.baseline!r} is not an arm")
    if s.family(s.baseline) != "control":
        raise ValueError(f"{where}: baseline {s.baseline!r} is not the control")
    if s.pick is not None and s.pick not in s.tags:
        raise ValueError(f"{where}: pick {s.pick!r} is not an arm")
    orders = [a.order for a in s.arms]
    if len(set(orders)) != len(orders):
        raise ValueError(
            f"{where}: duplicate Arm.order values make row order depend on "
            f"insertion order")
    # `rank_dash` is indexed only for non-control families; the control's dash
    # is hardcoded in `style_for`.
    widest = max((sum(a.family == f for a in s.arms)
                  for f in s.families if f != "control"), default=0)
    if len(s.rank_dash) < widest:
        raise ValueError(
            f"{where}: rank_dash has {len(s.rank_dash)} patterns but the "
            f"largest non-control family has {widest} arms")
    for a in s.arms:
        if a.label not in s.tex_labels:
            raise ValueError(
                f"{where}: arm {a.tag!r} label {a.label!r} has no tex_labels "
                f"entry, so the LaTeX table would emit a bare `^`")


_WEIGHT_ARM_SPEC: tuple[tuple[str, str, str], ...] = (
    ("unw",   "unweighted",   "control"),
    ("ip1",   "1/rtt",        "inverse-power"),
    ("ip2",   "1/rtt^2",      "inverse-power"),
    ("ip3",   "1/rtt^3",      "inverse-power"),
    ("tau10", "exp(-rtt/10)", "neg-exponential"),
    ("tau5",  "exp(-rtt/5)",  "neg-exponential"),
    ("tau1",  "exp(-rtt/1)",  "neg-exponential"),
)

#: Plain label -> LaTeX, for the weight arms. The plain forms carry a bare `^`,
#: which is math-mode only: emitted as-is the table does not compile ("Missing
#: $ inserted"). Written out rather than escaped programmatically so the
#: typeset form is legible here and `rtt` stays upright inside math mode.
_WEIGHT_TEX: dict[str, str] = {
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
    plain form (`3.83e+22`) is neither readable nor typeset maths.

    The WEIGHT sweep's formatter, hence a `Sweep.order_tex_fmt` entry rather
    than a module-wide helper: it floors at one decimal, so it renders a
    coverage of 0.95 as "1.0" and 0.75 as "0.8" -- silently collapsing a
    coverage arm onto its baseline.
    """
    if v < 1e4:
        return f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    exp = int(math.floor(math.log10(v)))
    return rf"${v / 10 ** exp:.1f}\times10^{{{exp}}}$"


SWEEP_WEIGHT = Sweep(
    key="weight-scorer",
    variants=(("octant_cbg_hull", "OCT-H"), ("octant_cbg_spl", "OCT-S")),
    # The example ratio is derived, not stored, so changing the reference
    # window cannot leave a stale literal behind.
    arms=tuple(Arm(t, l, example_ratio(t), f) for t, l, f in _WEIGHT_ARM_SPEC),
    # The sweep's own control, scored under the same config, folds and code as
    # the arms it anchors.
    baseline="unw",
    # The arm the paired table supports adopting: tied with the rest on p50 and
    # p90, with by far the mildest regression profile, and the only steep
    # family member scale-free in RTT (`rtt -> c*rtt` scales every weight by
    # one constant, which the face argmax discards).
    #
    # Deliberately NOT marked on the figure. A CDF cannot show what an arm
    # costs on the targets it makes worse, which is the whole basis for
    # preferring this one, so flagging it there would assert a conclusion the
    # panel does not carry. Readers of the module and the manifest still get
    # the pointer.
    pick="ip1",
    families=("control", "inverse-power", "neg-exponential"),
    # COMPOSITE ENCODING: hue carries the decay family, dash pattern carries
    # the steepness rank within it. Six distinct hues was tried first and does
    # not validate -- three lightness steps per hue cannot clear the
    # normal-vision floor (dE 15) while staying inside the L 0.43-0.77 band;
    # the best attempt reached dE 10.5 between adjacent blues.
    #
    # The two hues are v5's validated palette entries, re-checked with the
    # dataviz validator at `--pairs all` on a #ffffff surface: CVD separation
    # dE 21.6 (protan), normal-vision dE 32.3, contrast >= 3:1 -- all PASS.
    family_hue={
        "inverse-power": "#2a78d6",
        "neg-exponential": "#e34948",
        "control": _INK_2,
    },
    # Steepness rank within a family -> dash pattern, shallowest first. Shared
    # across families on purpose, so the pattern reads as "how steep" and the
    # hue as "which family" rather than the two being entangled.
    rank_dash=(
        None,          # solid
        (5, 1.6),      # dashed
        (1.4, 1.4),    # dotted
    ),
    order_column="example_ratio",
    order_ascending=True,
    order_tex_header="ex. ratio",
    order_tex_fmt=tex_range,
    row_tex_header="scorer",
    title="error CDF across weight scorers",
    tex_label_prefix="octant-weight-sweep",
    tex_labels=_WEIGHT_TEX,
    caption_effect="the face-weight scorer",
    caption_tie="under both scorers",
    caption_ordering=(
        rf"Arms are ordered by the example ratio $w({REF_FAST_MS:g}\,"
        rf"\mathrm{{ms}})/w({REF_SLOW_MS:g}\,\mathrm{{ms}})$ -- how strongly "
        r"each prefers a near vantage point over a moderate one. That ratio "
        r"depends on the interval it is read over, not on the scorer alone."
    ),
    prose={
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
        "spread_key": "steep_arm_spread_km",
        "encoding_rule":
            "hue = decay family; dash pattern = steepness rank within it",
        "encoding_why": (
            "six distinct hues was tried and does not validate: three "
            "lightness steps per hue cannot clear the normal-vision floor "
            "(dE 15) inside the L 0.43-0.77 band, best attempt dE 10.5 "
            "between adjacent blues. A secondary encoding is the remedy "
            "that keeps all arms on one panel, and it also shows the "
            "families interleaving on the example-ratio axis."
        ),
        "encoding_validated": (
            "dataviz validate_palette.js --pairs all, light, surface "
            "#ffffff, on the two family hues: CVD dE 21.6 (protan), "
            "normal-vision dE 32.3, contrast >= 3:1 -- all PASS."
        ),
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
        "lossy_note": (
            "one pooled number hides a real spread: the "
            "weighted-over-unweighted p50 gain runs 14.6x on as01 and 1.8x "
            "on as03. Read p50_km_per_dataset, or use --layout per-run, "
            "before quoting the pooled figure as the result."
        ),
    },
)


SWEEP_SPLINE = Sweep(
    key="spline-coverage",
    variants=(("octant_ssw", "OCT-S"),),
    # `order` is a literal here where SWEEP_WEIGHT derives it, and the
    # asymmetry is deliberate: the target coverage IS the config key the arm
    # sets, so there is nothing upstream for it to go stale against. The
    # example ratio is derived because it is read over a window that can move.
    arms=(
        Arm("nospl", "hull, no spline", 1.00, "control"),
        Arm("cov95", "coverage 0.95",   0.95, "spline-coverage"),
        Arm("cov90", "coverage 0.90",   0.90, "spline-coverage"),
        Arm("cov75", "coverage 0.75",   0.75, "spline-coverage"),
        Arm("cov50", "coverage 0.50",   0.50, "spline-coverage"),
    ),
    baseline="nospl",
    # No pick. The offline refit says no coverage setting beats the hull on
    # p90, so there is nothing to adopt, and None records that rather than
    # leaving a stale weight-sweep pointer here.
    pick=None,
    families=("control", "spline-coverage"),
    # One series family, so no within-palette pair is at stake -- only contrast
    # against the #ffffff surface and separation from the control grey. Reuses
    # SWEEP_WEIGHT's validated blue; `encoding_validated` below is this sweep's
    # OWN statement, not a copy, because the pair validated there
    # (#2a78d6 vs #e34948) is not a pair this sweep ever draws.
    family_hue={"spline-coverage": "#2a78d6", "control": _INK_2},
    # Four arms in one family, so four patterns -- a monotone DENSITY ladder
    # (solid, long dash, medium dash, dot) so the pattern reads as a coverage
    # ladder. Deliberately no dash-dot: `style_for`'s control branch is a
    # dash-dot and the control shares this panel, so a dash-dot arm would read
    # as a second reference line.
    rank_dash=(
        None,          # solid
        (6, 1.8),      # long dash
        (2.8, 1.6),    # medium dash
        (1.2, 1.2),    # dot
    ),
    order_column="target_coverage",
    # DESCENDING: coverage rises with looseness, so gentlest-first runs from
    # the baseline down to 0.50.
    order_ascending=False,
    order_tex_header="coverage",
    order_tex_fmt=lambda v: rf"${v:.2f}$",
    row_tex_header="bound",
    title="error CDF across spline coverage targets",
    tex_label_prefix="octant-spline-coverage",
    #: `$\delta_{c}$` rather than a bare number, so the first column names the
    #: PARAMETER and the coverage column carries its value -- mirroring the
    #: weight sweep, where the first column is the scorer and the second its
    #: derived ratio. A bare number in both would print the same figure twice.
    tex_labels={
        "hull, no spline": r"hull (no spline)",
        "coverage 0.95": r"$\delta_{0.95}$",
        "coverage 0.90": r"$\delta_{0.90}$",
        "coverage 0.75": r"$\delta_{0.75}$",
        "coverage 0.50": r"$\delta_{0.50}$",
    },
    caption_effect="the spline band's target coverage",
    caption_tie="under both bounds",
    caption_ordering=(
        r"Arms are ordered by the spline's target coverage, the fraction of "
        r"the vantage point's own training samples the band is required to "
        r"contain. A looser band admits more annulus area, so the ordering "
        r"runs from the hull baseline -- no spline at all -- down to $0.50$."
    ),
    prose={
        "arm_ordering": (
            "by the spline band's target coverage, descending: coverage is "
            "monotone in the band's multiplicative half-width delta, so this "
            "runs the arms from the loosest bound to the tightest. The hull "
            "baseline is the LIMIT of that axis, not a point on it -- it fits "
            "no spline, so it has no coverage, and its 1.0 is a sentinel."
        ),
        "scope": (
            "every column comes from this one -ssweep run; no combo is pulled "
            "in from another run id. The baseline is scored IN-TREE rather "
            "than joined from the pro mesh, so the comparison shares one "
            "config, one fold assignment and one code path."
        ),
        "spread_key": "coverage_arm_spread_km",
        "encoding_rule":
            "hue = spline arm vs hull control; dash pattern = coverage rank",
        "encoding_why": (
            "one series family, so hue alone cannot separate four coverage "
            "arms: four lightness steps in one hue is worse than the three "
            "that already fail the normal-vision floor (dE 15) inside the "
            "L 0.43-0.77 band. Dash density carries the coverage ladder "
            "instead, and the control keeps its dash-dot so it still reads as "
            "a reference rather than a fifth arm."
        ),
        "encoding_validated": (
            "one series hue only, so no within-palette pair is drawn. What "
            "matters is #2a78d6 against the #ffffff surface and against the "
            "#52514e control -- re-check with dataviz validate_palette.js "
            "--pairs all --mode light --surface #ffffff before publication."
        ),
        "overlap_note": (
            "the coverage arms are expected to separate on the TAIL, not the "
            "median: the offline refit moves p90 481 -> 500 -> 683 -> 749 km "
            "from coverage 1.00 to 0.75 while p50 stays inside the noise at "
            "~20 effective sites. Read p90, and say so in the caption."
        ),
        "unpaired_warning": (
            "a CDF compares marginal distributions, so it cannot show what an "
            "arm costs on the targets it makes worse. See the paired table."
        ),
        "baseline_choice": (
            "the hull bound -- Octant with no spline at all, scored under the "
            "same config, folds and code as the arms it anchors. It answers "
            "'what does the spline band buy over the convex hull', which is "
            "the live question: at the shipped coverage the spline variant is "
            "WORSE than the hull on all three datasets, at p50 and p90 both."
        ),
        "tie_note": (
            "ties are targets whose prediction is bit-identical under both "
            "bounds -- the arrangement's top face does not move. Expect many: "
            "at the shipped coverage 41% of per-VP constraints are already "
            "bit-identical to the hull's, 21% because the RTT is above "
            "cutoff_rtt and 20% because spline*delta clips to the hull on "
            "both sides."
        ),
        "symmetry_note": (
            "gains and regressions carry the same statistics (n, p50, p95, "
            "max) so an arm's upside and downside are read off one scale, with "
            "n beside every percentile because the two cohorts are not "
            "comparable in size."
        ),
        "confound_note": (
            "every arm is a BLEND of 'spline at this coverage' and 'hull'. "
            "find_delta_for_coverage requires |coverage(delta) - target| <= "
            "0.01 and raises otherwise; BoundedSplineLTD._fit swallows that at "
            "debug level and the VP falls back to bare hull bounds. On as01 "
            "fold_0 that is 17/28/25/39 of 134 VPs at coverage "
            ".95/.90/.75/.50 -- a proportion that varies for reasons unrelated "
            "to coverage, because the datasets hold only ~20 distinct probe "
            "coordinates, so coverage(delta) is a staircase with ~5% risers, "
            "coarser than the 1% tolerance it is tested against. Since the "
            "baseline IS the hull, this pulls every arm TOWARD the baseline: "
            "read the measured spline effect as a lower bound, and publish the "
            "per-arm fallback count beside it."
        ),
        "lossy_note": (
            "one pooled number hides a real spread; the three datasets differ "
            "sharply in how much the hull already wins by. Read "
            "p50_km_per_dataset, or use --layout per-run, before quoting the "
            "pooled figure as the result."
        ),
    },
)


SWEEPS: tuple[Sweep, ...] = (SWEEP_WEIGHT, SWEEP_SPLINE)
for _s in SWEEPS:
    _validate(_s)

_BY_VARIANT: dict[str, Sweep] = {v: s for s in SWEEPS for v, _ in s.variants}

#: Every variant prefix any registered sweep forks, and its published term.
VARIANTS: tuple[tuple[str, str], ...] = tuple(
    (v, t) for s in SWEEPS for v, t in s.variants
)

#: Kept at module level for the weight sweep's callers. Per-sweep defaults come
#: from `Sweep.baseline`, which is what every writer actually reads.
BASELINE_TAG = SWEEP_WEIGHT.baseline


def sweep_for(variant: str) -> Sweep:
    """The sweep a variant prefix belongs to.

    Raises `ValueError`, not `KeyError`: the CLI wraps each artifact write in
    `except (MissingArtifactError, ValueError)` and turns it into a skip line,
    so a `KeyError` here tracebacks out of a loop that meant to continue.
    """
    try:
        return _BY_VARIANT[variant]
    except KeyError:
        raise ValueError(
            f"unknown variant {variant!r}; known: {sorted(_BY_VARIANT)}"
        ) from None


def sweep_combos() -> frozenset[str]:
    """Every combo id any registered sweep scores.

    Exact ids, for the CLI's run discovery. The predicate this replaced was
    `c.startswith(f"{v}_") and c.split("_")[-1] in ARM_TAGS`, which matches
    `octant_cbg_hull` itself as soon as "hull" is any sweep's tag, and cannot
    tell `octant_cbg_hull_geo` from an arm.
    """
    return frozenset(f"{v}_{a.tag}"
                     for s in SWEEPS for v, _ in s.variants for a in s.arms)


def variants_present(run: RunPaths) -> list[str]:
    """Variant prefixes this run holds at least one scored arm of."""
    combos = set(run.combo_ids)
    return [v for s in SWEEPS for v, _ in s.variants
            if any(f"{v}_{a.tag}" in combos for a in s.arms)]

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


#: Layouts. `per-run` writes into each run's own `octant-finetuning/`;
#: `pooled` micro-pools every run's targets into one population and writes to
#: `_cross/octant-finetuning/<datasets>@<arm>/`.
PER_RUN = "per-run"
POOLED = "pooled"
LAYOUTS: tuple[str, ...] = (PER_RUN, POOLED)


def assemble_pooled(runs: list[RunPaths], variant: str
                    ) -> tuple[pd.DataFrame, list[str]]:
    """Micro-pool: every run's per-target rows concatenated into one frame.

    Micro, not macro -- rows are concatenated, so a dataset weighs by its
    target count (as01 399, as02 412, as03 458). Averaging the runs' published
    percentiles is a different quantity and would give a 458-target dataset the
    same say as a 399-target one.

    Two guards, both strict, both borrowed from `cross`. Every run must score
    the same arms, or a pooled column would rest on a different denominator
    from its neighbours. And no target id may appear in two runs, or it lands
    in the denominator twice.

    Pooling is lossy in a way worth stating: the weighted-over-unweighted gain
    runs 14.6x on as01 and 1.8x on as03, and one pooled number reports neither.
    `--layout per-run` is what shows that spread; the manifest records the
    per-run p50s so a reader of the pooled artifact can still see it.
    """
    frames, scored, tg_ids = {}, {}, {}
    for run in runs:
        wide, present = assemble(run, variant)
        wide = wide.assign(run_id=run.run_id, dataset=cross.short_dataset(run.run_id))
        # No namespacing pass here: `assemble` keys sites on the run id via
        # `sites.site_key`, so two runs holding the same coordinate are already
        # two sites. `dataset` is a display name and must never be the key --
        # it is declared in a config, so two runs can legitimately share one.
        frames[run.run_id], scored[run.run_id] = wide, set(present)
        tg_ids[run.run_id] = set(wide.target_id)

    common = cross.guard_common_methods(
        scored, remedy="--layout per-run to keep each dataset on its own figure.")
    cross.guard_disjoint_tgs(
        tg_ids,
        remedy="Use --layout per-run, which keeps each dataset on its own figure.")
    cross.guard_distinct_labels(
        {r.run_id: cross.short_dataset(r.run_id) for r in runs},
        remedy="Use --layout per-run, which keeps each dataset on its own figure.")

    keep = [t for t in sweep_for(variant).tags if t in common]
    cols = ["target_id", "site", "fold", "run_id", "dataset"] + keep
    pooled = pd.concat([f[cols] for f in frames.values()], ignore_index=True)
    return pooled, keep


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
    sweep = sweep_for(variant)
    wide: pd.DataFrame | None = None
    present: list[str] = []
    for tag in sweep.tags:
        arm = load_combo(run, f"{variant}_{tag}")
        if arm is None:
            continue
        if wide is None:
            arm = arm.copy()
            # `sites.site_key`, not a local "lat,lon": it keys on the run id
            # too, so the pooled frame needs no second namespacing pass. The
            # hand-rolled key this replaced was unique only within a run, and
            # the prefix `assemble_pooled` bolted on to fix that was itself
            # derived from a parsed run id -- the collapse this module hit.
            arm["site"] = S.site_key(
                arm.rename(columns={"target_lat": S.SITE_COLUMNS[0],
                                    "target_lon": S.SITE_COLUMNS[1]}),
                run_id=run.run_id,
            )
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
            f"./cli.sh --configfile <that run's config> first. The config is "
            f"named by {run.target_space_json}, and is not always "
            f"configs/<run_id>.yaml -- the grafted runs record the config they "
            f"were produced against.")
    return wide, present


def fold_coverage(run: RunPaths, variant: str) -> dict[str, int]:
    """Finished folds per arm, so a partial run can be declared rather than
    quietly narrowing the inner join in `assemble`."""
    out = {}
    for tag in sweep_for(variant).tags:
        d = load_combo(run, f"{variant}_{tag}")
        out[tag] = 0 if d is None else d.fold.nunique()
    return out


# ---- artifact 1: the CDF -----------------------------------------------------


def rank_within_family(tag: str, *, sweep: Sweep = SWEEP_WEIGHT) -> int:
    """Rank of an arm among its own family, gentlest = 0."""
    return sweep.rank_within_family(tag)


def _style_for(tag: str, *, sweep: Sweep = SWEEP_WEIGHT) -> dict:
    """Hue by family, dash by within-family rank. See `Sweep.style_for`."""
    return sweep.style_for(tag)


def cdf_table(wide: pd.DataFrame, present: list[str], *,
              sweep: Sweep = SWEEP_WEIGHT) -> pd.DataFrame:
    """The CSV twin: percentiles per arm, so the figure carries no footnote."""
    rows = []
    for tag in present:
        v = wide[tag]
        row = {
            "arm": sweep.label(tag),
            "arm_tag": tag,
            "family": sweep.family(tag),
            sweep.order_column: sweep.order(tag),
            "n": len(v),
        }
        row.update({f"p{p}_km": v.quantile(p / 100) for p in PERCENTILES})
        row["lt40km_pct"] = 100 * (v < 40).mean()
        row["lt100km_pct"] = 100 * (v < 100).mean()
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values(sweep.order_column, ascending=sweep.order_ascending)
            .reset_index(drop=True))


def plot_cdf(wide: pd.DataFrame, present: list[str], term: str, run_id: str,
             out_png: Path, *, sweep: Sweep = SWEEP_WEIGHT,
             dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    def curve(v):
        s = np.sort(v)
        return s, np.arange(1, len(s) + 1) / len(s)

    # Drawn gentlest-first so the aggressive arms, which sit left, land on top.
    for tag in sweep.ordered(present):
        ax.plot(*curve(wide[tag].values), **sweep.style_for(tag))

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

    # Legend grouped by family in the sweep's declared order (control first),
    # each family gentlest-first, so the dash pattern reads as a ladder down
    # each column. Iterating `sweep.families` rather than a hardcoded tuple is
    # what stops this emitting an EMPTY legend for a sweep whose families are
    # named anything else -- matplotlib draws an empty frame rather than
    # raising, so that failure was invisible.
    order: list[str] = []
    for fam in sweep.families:
        order += sweep.ordered(t for t in present if sweep.family(t) == fam)
    # The `$...$` tex_labels forms render natively as matplotlib mathtext, so
    # the legend and the LaTeX table typeset the same expression -- `1/rtt^2`
    # shows a real superscript rather than a literal caret.
    handles = [
        Line2D([], [], label=sweep.tex(sweep.label(t)), **sweep.style_for(t))
        for t in order
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=_LEGEND_PT,
                    frameon=False, handlelength=2.6, ncol=2,
                    columnspacing=1.1, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(_INK_2)

    ax.set_title(f"{term}: {sweep.title}",
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
                sweep: Sweep = SWEEP_WEIGHT,
                baseline: str | None = None) -> pd.DataFrame:
    baseline = baseline or sweep.baseline
    if baseline not in present:
        raise ValueError(f"baseline {baseline!r} not among scored arms {present}")
    rows = []
    for tag in [t for t in present if t != baseline]:
        d = wide[tag] - wide[baseline]          # negative = improvement
        gain, reg = -d[d < 0], d[d > 0]
        rows.append({
            "arm": sweep.label(tag),
            "arm_tag": tag,
            sweep.order_column: sweep.order(tag),
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
    return (pd.DataFrame(rows)
            .sort_values(sweep.order_column, ascending=sweep.order_ascending)
            .reset_index(drop=True))


def to_latex(t: pd.DataFrame, term: str, run_id: str, baseline_label: str,
             n_targets: int, *, sweep: Sweep = SWEEP_WEIGHT) -> str:
    L = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{term}: per-target effect of {sweep.caption_effect} on "
        rf"\texttt{{{run_id}}}, paired against {sweep.tex(baseline_label)} "
        rf"over $n={n_targets}$ targets. Gains and regressions are conditional "
        r"on a target moving in that direction, and carry the same statistics "
        r"so the two sides read off one scale; note their cohorts differ "
        r"greatly in size, so $n$ is given beside each. Ties are targets whose "
        rf"prediction is bit-identical {sweep.caption_tie}. "
        rf"{sweep.caption_ordering}}}",
        rf"\label{{tab:{sweep.tex_label_prefix}-{term.lower()}-{baseline_label}}}"
        .replace("(", "").replace(")", "").replace("/", "")
        .replace("-rtt", "rtt").replace(",", "").replace(" ", "-")
        .replace(".", ""),
        r"\begin{tabular}{lrrrrr rrr rrr}", r"\toprule",
        r"& & & \multicolumn{3}{c}{share of targets} "
        r"& \multicolumn{3}{c}{gain (km)} & \multicolumn{3}{c}{regression (km)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        rf"{sweep.row_tex_header} & {sweep.order_tex_header} & p50 & better "
        r"& worse & tie & $n$ & p50 & p95 & $n$ & p50 & p95 \\",
        r"\midrule",
    ]
    for _, r in t.iterrows():
        def f(v):
            return "--" if pd.isna(v) else f"{v:.0f}"
        L.append(
            f"{sweep.tex(r['arm'])} & "
            f"{sweep.order_tex_fmt(r[sweep.order_column])} & "
            f"{r.p50_km:.1f} & {r.better_pct:.1f}\\% & {r.worse_pct:.1f}\\% & "
            f"{r.tie_pct:.1f}\\% & {int(r.n_gain)} & {f(r.gain_p50)} & "
            f"{f(r.gain_p95)} & {int(r.n_reg)} & {f(r.reg_p50)} & "
            f"{f(r.reg_p95)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ---- manifests ---------------------------------------------------------------


def _common_manifest(runs: list[RunPaths], variant: str, term: str,
                     wide: pd.DataFrame, present: list[str]) -> dict:
    sweep = sweep_for(variant)
    pooled = len(runs) > 1
    coverage = fold_coverage(runs[0], variant) if not pooled else {
        t: min(fold_coverage(r, variant)[t] for r in runs) for t in present}
    complete = len(set(coverage[t] for t in present)) == 1
    body = {
        "layout": POOLED if pooled else PER_RUN,
        "run_id": "+".join(r.run_id for r in runs) if pooled else runs[0].run_id,
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
        "arm_ordering": sweep.prose["arm_ordering"],
        "scope": sweep.prose["scope"],
        "source": (
            "error_km from each combo's targets.parquet, not classify's "
            "*_tgs.parquet: the sweep compares weight functions on distance to "
            "the raw TG coordinate, so no answer space enters."
        ),
        "site_note": (
            f"{int(wide.site.nunique())} distinct coordinates behind "
            f"{len(wide)} targets -- the datasets hold ~20 IP replicas per "
            f"site, so any interval computed by resampling TARGETS rather than "
            f"sites is overstated by roughly sqrt(20)."
        ),
    }
    if pooled:
        per_run = {
            d: {t: float(g[t].median()) for t in present}
            for d, g in wide.groupby("dataset")
        }
        body["pooling"] = {
            "rule": (
                "micro-pool: each run's rows concatenated into one population, "
                "so a dataset weighs by its target count. Percentiles are "
                "recomputed over the concatenation, never averaged across runs."
            ),
            "n_per_dataset": {d: int(len(g)) for d, g in wide.groupby("dataset")},
            "p50_km_per_dataset": per_run,
            "lossy_note": sweep.prose["lossy_note"],
        }
    return body


def cdf_manifest(runs, variant, term, wide, present, table) -> dict:
    sweep = sweep_for(variant)
    body = _common_manifest(runs, variant, term, wide, present)
    steep = [t for t in present if t != sweep.baseline]
    spread = {
        f"p{p}": float(max(wide[t].quantile(p / 100) for t in steep)
                       - min(wide[t].quantile(p / 100) for t in steep))
        for p in (25, 50, 75, 90)
    } if steep else {}
    body.update({
        "artifact": "error CDF",
        "encoding": {
            "rule": sweep.prose["encoding_rule"],
            "family_hue": dict(sweep.family_hue),
            "why_composite": sweep.prose["encoding_why"],
            "validated": sweep.prose["encoding_validated"],
        },
        sweep.prose["spread_key"]: spread,
        "overlap_note": sweep.prose["overlap_note"],
        "unpaired_warning": sweep.prose["unpaired_warning"],
        "percentiles": table.to_dict(orient="records"),
        "panel": {"figsize_in": list(PAPER_FIGSIZE), "x_label": X_LABEL,
                  "y_label": Y_LABEL, "y_ticks": list(Y_TICKS), "x_scale": "log"},
    })
    return body


def table_manifest(runs, variant, term, wide, present, baseline, table) -> dict:
    sweep = sweep_for(variant)
    body = _common_manifest(runs, variant, term, wide, present)
    body.update({
        "artifact": "paired per-target table",
        "baseline": sweep.label(baseline),
        "baseline_tag": baseline,
        "baseline_choice": sweep.prose["baseline_choice"],
        "tie_note": sweep.prose["tie_note"],
        "symmetry_note": sweep.prose["symmetry_note"],
        "rows": table.to_dict(orient="records"),
    })
    # Emitted only by sweeps that declare one. Adding a key unconditionally
    # would rewrite the weight sweep's already-published manifest bytes, which
    # are a regression gate -- so `Sweep.pick` stays deliberately UNSERIALIZED
    # for now, exactly as `PICK` was. Publishing it is a separate, visible
    # change with its own re-baseline.
    if "confound_note" in sweep.prose:
        body["confound_note"] = sweep.prose["confound_note"]
    return body


# ---- orchestration -----------------------------------------------------------


def _out_dir(runs: list[RunPaths], analysis_root: Path | None) -> Path:
    """Per-run artifacts land in the run's own kind directory; pooled ones in
    `_cross/octant-finetuning/<datasets>@<arm>/`, keyed by the dataset set and
    the shared arm so a two-run pool cannot overwrite a three-run one."""
    if len(runs) == 1:
        return runs[0].octant_finetuning_dir(root=analysis_root)
    return cross.cross_dir([r.run_id for r in runs], analysis_root=analysis_root,
                           kind=OCTANT_FINETUNING_KIND)


def _label(runs: list[RunPaths]) -> str:
    return (runs[0].run_id if len(runs) == 1
            else cross.dataset_slug([r.run_id for r in runs]))


def write_cdf(runs: RunPaths | list[RunPaths], variant: str, *,
              analysis_root: Path | None = None, dpi: int = 300) -> Path:
    runs = [runs] if isinstance(runs, RunPaths) else list(runs)
    sweep = sweep_for(variant)
    term = sweep.term(variant)
    wide, present = (assemble(runs[0], variant) if len(runs) == 1
                     else assemble_pooled(runs, variant))
    table = cdf_table(wide, present, sweep=sweep)

    # The cdf branch ignores the baseline -- see `artifact_names` -- but it is
    # passed this sweep's own rather than the weight sweep's, so no foreign tag
    # is threaded through a call that could later start using it.
    png_name, csv_name, man_name = artifact_names(variant, sweep.baseline, "cdf")
    out = _out_dir(runs, analysis_root)
    plot_cdf(wide, present, term, _label(runs), out / png_name,
             sweep=sweep, dpi=dpi)
    table.to_csv(out / csv_name, index=False)
    (out / man_name).write_text(json.dumps(
        cdf_manifest(runs, variant, term, wide, present, table), indent=2))
    return out / png_name


def write_table(runs: RunPaths | list[RunPaths], variant: str, *,
                baseline: str | None = None,
                analysis_root: Path | None = None) -> Path:
    runs = [runs] if isinstance(runs, RunPaths) else list(runs)
    sweep = sweep_for(variant)
    baseline = baseline or sweep.baseline
    term = sweep.term(variant)
    wide, present = (assemble(runs[0], variant) if len(runs) == 1
                     else assemble_pooled(runs, variant))
    t = build_table(wide, present, sweep=sweep, baseline=baseline)

    csv_name, tex_name, man_name = artifact_names(variant, baseline, "table")
    out = _out_dir(runs, analysis_root)
    t.to_csv(out / csv_name, index=False)
    (out / tex_name).write_text(
        to_latex(t, term, _label(runs), sweep.label(baseline), len(wide),
                 sweep=sweep))
    (out / man_name).write_text(json.dumps(
        table_manifest(runs, variant, term, wide, present, baseline, t), indent=2))
    return out / csv_name
