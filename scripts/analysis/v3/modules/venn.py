"""Set-overlap of correct classifications across methods (§8.2's regression view).

Consumes `classify`'s per-target output and turns each method into a set: the
targets it classified into the *right* seed. Overlaps then answer the question an
aggregate accuracy number hides — which targets does CBG win that Shortest-Ping
loses, and which does it lose that Shortest-Ping already had. That trade is the
point: "CBG beats the baseline" can be true in aggregate while a variant
regresses on targets the baseline solved.

Correctness uses the same rule as the accuracy table: `truth_seed_rank < N`
(N=1 by default), with non-SUCCESS rows counted as failures per §7.2 — so a
fallback never enters a set even though it carries a coordinate.

The headline figure is a 2-set Venn of Shortest-Ping against "at least one CBG
variant works", since that is the trade RQ2 rests on: the CBG-only region counts
rescues, the Shortest-Ping-only region counts regressions. An UpSet plot carries
the per-method detail, because at six methods a Venn would need 63 regions.

Venns are drawn **unweighted** — fixed-size, evenly positioned circles with the
counts and percentages written into the regions. The *collapsed* sets nest
(`all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` on every run we have), and no
area-proportional layout can render nesting; area-proportional drawing warned on
every run for exactly that reason.

**Cross-run mode.** Two or more `--run-id`s are pooled into one population under
`_cross/venn-diagram/<datasets>/`. as01/02/03 have disjoint target sets and the
same six methods, so pooling is a concat rather than a join — see
`pooled_membership`. The pooled directory adds `plot_ring_venn`: the
conventional presentation "6-way Venn" of six equal circles on a ring, with the
share of the pooled target population written into every region.

That figure's circles are a **fixed template** — identical centres and radii in
every run — so nothing about a region's size or position carries data and every
quantity on it is a printed label. Six circles realize 31 of the 63 possible
combinations, so on real data some observed intersections have no region; the
footnote and `ring_coverage_table` name them rather than letting a reader take
the labels for the whole population.

The pooled directory carries a **second, opposite figure**: `plot_euler`. Where
the ring fixes the geometry and prints every number, the Euler layout fits the
geometry *to* the numbers and prints none — circle area is the set size, the area
two circles share is the size of their intersection, methods that never agree are
drawn apart and a method contained in another is drawn inside it. Neither figure
subsumes the other. The ring can state that the six-way region holds 12.7% and
cannot show that Shortest-Ping's correct set sits inside SoI CBG's; the Euler
layout shows the containment at a glance and can quantify nothing. Circles are
overdetermined past two sets, so `EulerLayout.misplaced` measures the share of
targets the drawing puts in the wrong region and the figure prints it.

Every figure is backed by the same membership matrix, written alongside so the
counts are checkable without reading a figure. Every artifact is suffixed with
its top-N, so top-1 and top-3 coexist rather than overwriting; cross-run
artifacts carry the grid slug too, since that directory is not grid-scoped.

Command: `plot-venn`.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from scripts.analysis.v3.modules.grid import (  # noqa: E402
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    get_grid,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules import cross, io  # noqa: E402
from scripts.analysis.v3.modules.classify import SHORTEST_PING  # noqa: E402
from scripts.analysis.v3.modules.paths import (  # noqa: E402
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    grid_slug,
    resolve_run,
)

#: This module's artifact kind under `_cross/`, the sibling of `pareto`'s
#: `"cost-accuracy"`.
CROSS_KIND = "venn-diagram"

#: Display labels. Keyed by method id; the Octant-Spline combo id differs by run
#: (`octant_cbg_spl` on the operator runs, `octant_cbg` on the RIPE run).
LABELS: dict[str, str] = {
    SHORTEST_PING: "Shortest-Ping",
    "million_scale_cbg": "SoI CBG",
    "vanilla_cbg": "Vanilla CBG",
    "octant_cbg_hull": "Octant-Hull CBG",
    "octant_cbg_spl": "Octant-Spline CBG",
    "octant_cbg": "Octant-Spline CBG",
    "spotter_cbg": "Spotter CBG",
}

#: Display label for the collapsed "at least one CBG variant is correct" set.
CBG_ANY_LABEL = "≥1 CBG"

#: Display label for the Shortest-Ping side of that same collapse. Deliberately
#: not `label_for(SHORTEST_PING)` ("Shortest-Ping"): this pairing is a figure of
#: its own, and reads better without the hyphen alongside the terse "≥1 CBG".
SP_VS_CBG_LABEL = "Shortest Ping"

#: Baseline first, then calibration-free, then increasingly fitted.
PREFERRED_ORDER: tuple[str, ...] = (
    SHORTEST_PING,
    "million_scale_cbg",
    "vanilla_cbg",
    "octant_cbg_hull",
    "octant_cbg_spl",
    "octant_cbg",
    "spotter_cbg",
)

def artifact_name(
    stem: str, ext: str, top_n: int, grid: str | None = None
) -> str:
    """`("overlap_upset", "png", 3)` -> `"overlap_upset.top3.png"`.

    Every overlap artifact is a function of `top_n`, so the suffix is
    mandatory: without it a top-3 run silently overwrites the top-1 files.

    `grid` is the `paths.grid_slug` (`"h3-4"`), and is **optional because the
    two output trees are quantized differently**. Per-run artifacts already live
    under `target-cls-accuracy/<grid>-<res>/`, so repeating it in the filename
    would be noise — and would rename files that already exist on disk. The
    cross-run directory is keyed by dataset set only, so there the slug is the
    sole thing keeping an h3-4 run from overwriting a healpix-128 one.
    """
    grid_part = f"{grid}." if grid else ""
    return f"{stem}.{grid_part}top{top_n}.{ext}"


def label_for(method: str) -> str:
    """Display name, with every CBG variant marked as one.

    Shortest-Ping is the only non-CBG method here, so anything else gets a
    `CBG` suffix — including the ablation arms, which have no entry in `LABELS`
    and would otherwise appear as bare combo ids indistinguishable from the
    baseline at a glance.
    """
    label = LABELS.get(method)
    if label is not None:
        return label
    return method if method == SHORTEST_PING else f"{method} CBG"


# ---------------------------------------------------------------------------
# § palette
# ---------------------------------------------------------------------------
#
# Lives here rather than in `pareto.py` (where it was written) because it is
# keyed on `label_for` and `PREFERRED_ORDER` above, and because both modules now
# draw variants. `pareto` imports it back; the dependency only runs one way.

#: Fixed variant -> hue, assigned by **identity** (`PREFERRED_ORDER`) and never
#: by rank in the current selection, so `--method` cannot repaint the survivors.
#:
#: Validated with the dataviz skill's `validate_palette.js` against the
#: reference 8-hue categorical theme, `--pairs all` on white — the right check
#: here, since every variant is visible at once. Every 6/7/8-slot prefix of that
#: theme FAILS (green vs orange is dE 3.2 under protanopia), and an exhaustive
#: search over its hues found exactly two passing 6-subsets; this is the better
#: one, worst dE 6.9 (deutan) / 7.6 (tritan). Orange is the hue that had to go.
#:
#: dE 6.9 sits in the 6-8 band that is legal *only* with secondary encoding.
#: That is satisfied three times over: each variant owns its own x column (they
#: never interleave spatially), the legend names every one, and the CSV is the
#: table view. Aqua/yellow/magenta are also below 3:1 on white (2.82/2.17/2.69),
#: a contrast WARN that obliges visible labels or a table view — the legend and
#: CSV again. No 6-subset of this theme clears 3:1 for all six (only five hues
#: do), so at six variants that is unavoidable rather than a shortcut; the 2 px
#: cost line and ringed >=8 px markers give each variant more ink than a dot.
_VARIANT_HUES: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Past the palette's capacity a 9th series is never a generated hue — it folds
#: into one "other" bucket. as7018 carries 11 ablation arms on top of the five
#: published variants, and they belong in that bucket.
_C_OTHER = "#898781"

_C_GRID = "#e1e0d9"
_C_INK = "#0b0b0b"
_C_INK_2 = "#52514e"
_C_MUTED = "#898781"
_SURFACE = "#ffffff"


def _build_label_hues() -> dict[str, str]:
    """Display label -> hue, fixed once from `PREFERRED_ORDER`.

    Keyed on the *label* rather than the combo id so `octant_cbg_spl` and
    `octant_cbg` land on one hue: they are one paper variant whose id differs
    per run, which is why `LABELS` already maps both onto "Octant-Spline CBG".
    Two hues would invent a distinction the runs do not contain. That aliasing
    is also what makes the six published variants fit the six validated hues
    exactly.
    """
    hues: dict[str, str] = {}
    for method in PREFERRED_ORDER:
        label = label_for(method)
        if label in hues:
            continue
        if len(hues) < len(_VARIANT_HUES):
            hues[label] = _VARIANT_HUES[len(hues)]
    return hues


#: Computed once, at import, from a constant order — never from the methods
#: present in a given call. This is what makes colour stable under `--method`.
_LABEL_HUES: dict[str, str] = _build_label_hues()


def method_colors(methods) -> dict[str, str]:
    """Variant -> hue, stable under filtering.

    Each hue is pinned to a variant's *identity* via `_LABEL_HUES`, which is
    built from a fixed order at import time. Filtering the method pool with
    `--method` therefore cannot repaint the survivors — colour follows the
    entity, never its rank in the current selection, and the same variant is
    the same colour in every figure of a sweep.

    Anything `PREFERRED_ORDER` does not name — as7018's 11 ablation arms —
    folds into the single `_C_OTHER` bucket rather than being handed a generated
    hue, because no palette distinguishes 17 series.
    """
    return {m: _LABEL_HUES.get(label_for(m), _C_OTHER) for m in methods}


def available_methods(cls_dir: Path) -> list[str]:
    """Method ids with a `*_seed_distances.parquet`, in display order."""
    found = {
        p.name[: -len("_seed_distances.parquet")]
        for p in Path(cls_dir).glob("*_seed_distances.parquet")
    }
    ordered = [m for m in PREFERRED_ORDER if m in found]
    return ordered + sorted(found - set(ordered))


def build_membership(
    cls_dir: Path, methods: list[str], *, top_n: int = 1
) -> pd.DataFrame:
    """Boolean matrix: one row per target, one column per method.

    True means the method placed that target in a seed ranked better than
    `top_n`, and that the pipeline actually solved it (fallbacks are failures).
    """
    cls_dir = Path(cls_dir)
    cols: dict[str, pd.Series] = {}
    for method in methods:
        path = cls_dir / f"{method}_seed_distances.parquet"
        if not path.exists():
            raise MissingArtifactError(f"{path} missing; run `classify` first")
        df = pd.read_parquet(
            path, columns=["target_id", "status", "truth_seed_rank"]
        ).set_index("target_id")
        solved = (df["status"] == "BASELINE") | df["status"].isin(
            io.CBG_SUCCESS_STATUSES
        )
        rank = df["truth_seed_rank"]
        cols[method] = (rank >= 0) & (rank < top_n) & solved

    membership = pd.DataFrame(cols)
    if membership.isna().any().any():
        # Methods disagreeing on the target set would make every intersection
        # count ambiguous.
        counts = {m: int(membership[m].notna().sum()) for m in methods}
        raise ValueError(
            f"methods cover different target sets ({counts}); cannot form set "
            f"overlaps over a common denominator"
        )
    return membership.astype(bool)


#: Separator between the run id and the target id in a pooled membership index.
#: `::` cannot occur in either half — operator ids are `tg-<hex>` and as7018's
#: are raw IPs — so the key stays reversible by `str.split(RUN_KEY_SEP, 1)`.
RUN_KEY_SEP = "::"


def pooled_membership(
    runs: dict[str, RunPaths],
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    top_n: int = 1,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """One membership matrix over the targets of several runs.

    The runs are stacked, not joined. SCHEMA.md §1 forbids parsing `target_id`,
    and the three operator runs have **disjoint** target sets (verified: all
    three pairwise intersections are empty), so a target belongs to exactly one
    run and pooling is a concat. Rows are re-keyed `<run_id>::<target_id>`
    anyway: the disjointness is a property of these datasets rather than of the
    schema, and without the prefix a future run that reused an id would silently
    collide two different targets into one row.

    When the method set is discovered rather than given, every run must have
    scored **exactly** the same methods. Equality rather than a subset test: a
    method one run lacks would be all-False across that run's targets,
    understating it and inflating the "none correct" region, while a method only
    one run has would be dropped without a word. Testing only one direction also
    makes the answer depend on argument order — as01 + as7018 would error or
    silently drop as7018's ablation arms depending on which was named first.
    Passing `methods` opts out: the caller has stated the set, so extras are
    theirs to exclude and only presence is checked.

    Returns the pooled matrix and the per-row `run_id`, which the caller needs
    for the membership CSV and the manifest.
    """
    if len(runs) < 2:
        raise ValueError(f"pooling needs >= 2 runs, got {sorted(runs)}")

    frames: list[pd.DataFrame] = []
    origins: list[pd.Series] = []
    chosen: list[str] | None = list(methods) if methods else None
    first_run = ""

    for run_id, run in runs.items():
        cls_dir = run.cls_accuracy_dir(
            root=analysis_root, grid=grid, resolution=resolution
        )
        if not cls_dir.is_dir():
            raise MissingArtifactError(
                f"{cls_dir} missing; run `classify` for {run_id} at {grid}-{resolution}"
            )
        found = available_methods(cls_dir)
        if chosen is None and methods is None:
            chosen = found
            first_run = run_id
            if len(chosen) < 2:
                raise ValueError(
                    f"need >= 2 methods to show overlap, got {chosen} in {cls_dir}"
                )

        if methods is not None:
            # An explicit set only has to be *present*; the caller has already
            # said which methods they mean, so extras are theirs to exclude.
            absent = [m for m in chosen if m not in found]
            if absent:
                raise ValueError(
                    f"{run_id} has not scored {absent} (has {found}); a method "
                    f"missing from one run would be counted wrong on every one "
                    f"of its targets."
                )
        elif set(found) != set(chosen):
            # Discovered sets must match **exactly**, not merely overlap. A
            # subset test would make the result depend on which run was named
            # first: pooling as01 with as7018 would either error (as7018 first,
            # as01 missing its 10 ablation arms) or silently drop those arms
            # (as01 first) — same two runs, two different figures.
            only_here = sorted(set(found) - set(chosen))
            only_there = sorted(set(chosen) - set(found))
            raise ValueError(
                f"{run_id} and {first_run} scored different methods: "
                f"{only_here} only in {run_id}, {only_there} only in {first_run}. "
                f"Pooling them would count an unscored method as wrong on every "
                f"target of the run that lacks it. Pin the shared set with "
                f"--method, or score the missing ones."
            )

        one = build_membership(cls_dir, chosen, top_n=top_n)
        one.index = [f"{run_id}{RUN_KEY_SEP}{t}" for t in one.index]
        frames.append(one)
        origins.append(pd.Series(run_id, index=one.index, name="run_id"))

    pooled = pd.concat(frames)
    if not pooled.index.is_unique:
        dupes = pooled.index[pooled.index.duplicated()].unique().tolist()
        raise ValueError(
            f"pooled target keys are not unique ({dupes[:5]}); one run_id appears "
            f"twice in the selection"
        )
    return pooled, pd.concat(origins)


def intersection_table(membership: pd.DataFrame) -> pd.DataFrame:
    """Every non-empty exact intersection, largest first.

    "Exact" means the set of methods that got the target right is precisely this
    combination — so rows partition the target population and sum to n_targets.
    """
    methods = list(membership.columns)
    keys = membership.apply(
        lambda r: tuple(m for m in methods if r[m]), axis=1
    )
    grouped = keys.value_counts()
    rows = [
        {
            "methods": "|".join(label_for(m) for m in combo) if combo else "(none)",
            "n_methods": len(combo),
            "n_targets": int(n),
            "share": round(float(n) / len(membership), 4),
        }
        for combo, n in grouped.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["n_targets", "n_methods"], ascending=[False, True]
    ).reset_index(drop=True)


#: Letters the generic Venn tools key their inputs on — `A ^ B`, `A ^ B ^ C`.
#: Method labels are far too long for that role ("Octant-Spline CBG"), so the
#: spec carries both: a letter as the key and the real label alongside it.
SET_IDS = "ABCDEFGH"


def venn_spec(membership: pd.DataFrame, *, exclusive: bool = False) -> dict:
    """The membership matrix as a generic Venn-tool input document.

    Mirrors the shape those generators ask for: a set count, one entry per set
    with a name and a size, and one entry per *combination* of two or more sets.
    Written as JSON so the numbers can be pasted into a drawing tool without
    re-deriving them from `overlap_membership.csv`.

    `relations` are **cumulative** by default — `A^B` is the full `|A ∩ B|`,
    which counts targets that are also in `C`. That is the set-theoretic
    reading, and the one those forms assume: it is what makes `size` and the
    relations satisfy inclusion-exclusion, so a tool can solve for the regions
    itself. `exclusive=True` switches to the disjoint reading instead — `A^B`
    becomes targets in `A` and `B` and **nothing else** — which is what
    `intersection_table` and the ring figure report. The two differ wherever a
    higher-order region is non-empty, so the convention is recorded in the
    document rather than left for a reader to infer.

    Combinations with no targets are included with a zero. A Venn tool needs a
    value for every relation it will draw, and omitting the empties would make
    a reader guess whether the region is empty or the number is missing.
    """
    methods = list(membership.columns)
    if len(methods) > len(SET_IDS):
        raise ValueError(
            f"venn spec covers at most {len(SET_IDS)} sets, got {len(methods)}"
        )
    ids = {m: SET_IDS[i] for i, m in enumerate(methods)}
    columns = {m: membership[m].to_numpy() for m in methods}

    relations = {}
    for k in range(2, len(methods) + 1):
        for combo in combinations(methods, k):
            mask = columns[combo[0]].copy()
            for m in combo[1:]:
                mask = mask & columns[m]
            if exclusive:
                for m in methods:
                    if m not in combo:
                        mask = mask & ~columns[m]
            relations[" ^ ".join(ids[m] for m in combo)] = int(mask.sum())

    return {
        "number_of_sets": len(methods),
        "n_targets": int(len(membership)),
        # Targets no method got right sit outside every circle, so no Venn
        # region holds them and the count would otherwise vanish.
        "n_none": int((~membership.any(axis=1)).sum()),
        "relation_convention": "exclusive" if exclusive else "cumulative",
        "sets": [
            {
                "id": ids[m],
                "name": label_for(m),
                "method": m,
                "size": int(membership[m].sum()),
            }
            for m in methods
        ],
        "relations": relations,
    }


def pairwise_table(membership: pd.DataFrame) -> pd.DataFrame:
    """Per method pair: won-only, lost-only, both, neither.

    `a_only` is the regression column when `a` is the baseline — targets
    Shortest-Ping solves and the variant does not.
    """
    rows = []
    for a, b in combinations(membership.columns, 2):
        sa, sb = membership[a], membership[b]
        rows.append(
            {
                "method_a": label_for(a),
                "method_b": label_for(b),
                "both": int((sa & sb).sum()),
                "a_only": int((sa & ~sb).sum()),
                "b_only": int((~sa & sb).sum()),
                "neither": int((~sa & ~sb).sum()),
                "jaccard": round(
                    float((sa & sb).sum() / max((sa | sb).sum(), 1)), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def _region_labeller(total: int):
    """Format each Venn region as `n` over `(pct%)` of the target population."""

    def fmt(value) -> str:
        n = int(value if not hasattr(value, "__len__") else len(value))
        pct = 100.0 * n / total if total else 0.0
        return f"{n}\n({pct:.1f}%)"

    return fmt


def plot_venn(
    membership: pd.DataFrame, out_path: Path, *, title: str, weighted: bool = False
) -> Path:
    """Classic 2- or 3-set Venn. Requires 2 or 3 methods.

    Column names are taken as the display labels verbatim — the caller must
    already have named them (`label_for`, or a hand-picked string like the
    SP-vs-CBG collapse's "Shortest Ping"). Applying `label_for` in here too
    would double up on any column that is already a display name rather than
    a raw method id, which is exactly how this used to render "≥1 CBG CBG".

    Unweighted (fixed-size, evenly positioned circles) by default rather than
    area-proportional, because the correctness sets are routinely nested —
    `all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` holds on every run we have — and a
    3-circle area-proportional layout cannot render a strict 3-way chain: at
    least one region's exact-zero area is unsatisfiable by any real circle
    configuration, so the solver distorts the whole figure to approximate it.
    The counts carry the information instead of the areas.

    `weighted=True` switches to the real area-proportional layout — circle and
    overlap size driven by the actual subset sizes — so a near-zero region
    reads as near-total inclusion instead of an evenly-sized sliver. Only
    supported at 2 sets: the chain-nesting failure above is specifically a
    3-circle problem, and every current caller of `weighted=True` is the 2-set
    Shortest-Ping-vs-CBG collapse.

    Set labels are drawn as a legend rather than matplotlib-venn's default
    text beside each circle: at near-total overlap (the `weighted=True` case
    this exists for) the smaller circle's own label sits inside the bigger
    circle's territory, which reads as if it belongs to the wrong set.
    """
    # `venn2_unweighted` / `venn3_unweighted` are deprecated in matplotlib-venn
    # 1.1.2 *and* broken — they forward `normalize_to` into a custom layout that
    # rejects it. Drive the layout algorithm directly instead.
    from matplotlib.patches import Patch
    from matplotlib_venn import venn2, venn3
    from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm as Venn2Layout
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm as Venn3Layout

    methods = list(membership.columns)
    if len(methods) not in (2, 3):
        raise ValueError(f"Venn needs 2 or 3 methods, got {len(methods)}")
    if weighted and len(methods) != 2:
        raise ValueError(
            f"weighted=True is only supported at 2 sets (got {len(methods)}); "
            f"a 3-circle area-proportional layout cannot render the chain "
            f"nesting these sets routinely have"
        )
    sets = [set(membership.index[membership[m]]) for m in methods]
    labels = tuple(str(m) for m in methods)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    if len(methods) == 2:
        draw = venn2
        layout = Venn2Layout() if weighted else Venn2Layout(fixed_subset_sizes=(1, 1, 1))
        region_ids = ("10", "01")
    else:
        draw, layout = venn3, Venn3Layout(fixed_subset_sizes=(1,) * 7)
        region_ids = ("100", "010", "001")
    v = draw(
        sets,
        # No text beside the circles — the legend below carries the names.
        set_labels=None,
        ax=ax,
        layout_algorithm=layout,
        subset_label_formatter=_region_labeller(len(membership)),
    )
    # Swatch colour comes from the rendered "this set only" patch rather than
    # a second, independent colour choice, so the legend can never disagree
    # with what is actually on the canvas. That region is occasionally empty
    # (no patch drawn), hence the grey fallback.
    handles = [
        Patch(
            facecolor=(
                v.get_patch_by_id(rid).get_facecolor()
                if v.get_patch_by_id(rid) is not None
                else "#999999"
            ),
            edgecolor="none",
            label=lab,
        )
        for rid, lab in zip(region_ids, labels)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
    )

    # The "no method correct" region falls outside every circle, so a Venn
    # cannot show it. It is part of the denominator and often large, so state
    # it rather than leaving the reader to subtract.
    n_total = len(membership)
    n_none = int((~membership.any(axis=1)).sum())
    pct_none = 100.0 * n_none / n_total if n_total else 0.0
    ax.annotate(
        f"All Failed: {n_none} ({pct_none:.1f}%)",
        xy=(0.5, -0.14),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def collapse_to_sp_vs_cbg(membership: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse per-method membership to `Shortest-Ping` vs `≥1 CBG`.

    Returns the 2-column frame and the size of the CBG pool that was collapsed.
    The pool size matters for interpretation and must reach the figure: on
    `as7018_us_test01` the default method set is 16 combos including 11 ablation
    arms, and the CBG-only region reads 40 over all of them versus 31 over the
    five published variants.
    """
    if SHORTEST_PING not in membership.columns:
        raise ValueError(
            f"membership has no {SHORTEST_PING!r} column; the Shortest-Ping "
            f"baseline is required for the SP-vs-CBG view. Got: "
            f"{list(membership.columns)}"
        )
    cbg = membership.drop(columns=[SHORTEST_PING])
    if cbg.shape[1] == 0:
        raise ValueError("no CBG methods to collapse; need at least one besides the baseline")
    collapsed = pd.DataFrame(
        {
            SP_VS_CBG_LABEL: membership[SHORTEST_PING],
            CBG_ANY_LABEL: cbg.any(axis=1),
        },
        index=membership.index,
    )
    return collapsed, cbg.shape[1]


def plot_sp_vs_cbg_venn(
    membership: pd.DataFrame, out_path: Path, *, title: str
) -> Path:
    """Area-proportional 2-set Venn: Shortest-Ping vs "at least one CBG works".

    This is the rescue-vs-regression trade an aggregate accuracy number hides.
    The CBG-only region counts targets CBG rescues; the Shortest-Ping-only
    region counts the ones it regresses on. Weighted rather than fixed-size —
    unlike the 3-way chain `plot_venn`'s docstring warns about, this is exactly
    2 sets, so a near-zero regression region can render as near-total
    inclusion instead of an evenly-sized sliver that overstates it.
    """
    collapsed, n_cbg = collapse_to_sp_vs_cbg(membership)
    subtitle = f"≥1 of {n_cbg} CBG variant{'s' if n_cbg != 1 else ''}"
    return plot_venn(collapsed, out_path, title=f"{title}\n{subtitle}", weighted=True)


#: Header for the per-column numbers (exact disjoint intersection shares).
COLUMN_METRIC_LABEL = "Intersections (%)"

#: Header for the per-row numbers (per-method success rate). Deliberately a
#: different word from the column header: these do not sum to 100%.
ROW_METRIC_LABEL = "True (%)"


def plot_upset(
    membership: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    min_subset_size: int = 1,
) -> Path:
    """UpSet plot — the readable form once arity exceeds three.

    Both bar charts are suppressed and their magnitudes written as text
    instead: shares above each column, per-method success rate to the right of
    each row. The two number sets are *not* the same kind of quantity, which is
    why each carries its own header —

    * columns are **exact, disjoint** intersections ("these methods correct,
      all others wrong"), so they partition the targets and sum to 100%;
    * rows are set totals, which overlap and therefore do not.

    Category order is pinned to the caller's column order
    (`sort_categories_by=None`) so the dot pattern means the same thing in
    every run's figure. Columns are ranked by intersection size, largest first,
    so the numbers read monotonically left to right. That ordering is
    data-dependent — the same dot combination sits at a different x in another
    run — so compare columns by their dots, never by position.
    """
    from upsetplot import UpSet, from_indicators

    renamed = membership.rename(columns={m: label_for(m) for m in membership.columns})
    total = len(renamed)
    # upsetplot draws the first category at the *bottom*, so hand it the
    # reversed order to get the caller's order reading top-to-bottom.
    data = from_indicators(list(renamed.columns)[::-1], renamed)

    upset = UpSet(
        data,
        subset_size="count",
        show_counts=False,
        sort_by="cardinality",
        sort_categories_by=None,
        min_subset_size=min_subset_size,
        # Both bars off; the numbers are annotated onto the matrix below.
        intersection_plot_elements=0,
        totals_plot_elements=0,
    )

    n_cols = len(upset.intersections)
    fig = plt.figure(figsize=(max(7.0, 0.62 * n_cols + 4.2), 0.46 * len(renamed.columns) + 2.6))
    axes = upset.plot(fig=fig)
    ax = axes["matrix"]

    # `intersections` is in plotted order, so position i sits at x == i.
    shares = 100.0 * upset.intersections.to_numpy() / total if total else []
    y_top = len(renamed.columns) - 0.5
    for i, pct in enumerate(shares):
        ax.text(
            i, y_top + 0.30, f"{pct:.1f}", ha="center", va="bottom",
            fontsize=7.5, rotation=90, clip_on=False,
        )
    ax.text(
        -0.9, y_top + 0.30, COLUMN_METRIC_LABEL, ha="right", va="bottom",
        fontsize=8, fontweight="bold", clip_on=False,
    )

    # `totals` is indexed by category in the same order as the y ticks.
    x_right = n_cols - 0.4
    ticks = ax.get_yticks()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    for y, lab in zip(ticks, labels):
        # A method with zero correct targets is dropped from `totals`, so read
        # it defensively rather than KeyError-ing on the worst-performing arm.
        pct = 100.0 * float(upset.totals.get(lab, 0)) / total if total else 0.0
        ax.text(
            x_right, y, f"{pct:.1f}", ha="left", va="center",
            fontsize=8, clip_on=False,
        )
    ax.text(
        x_right, y_top + 0.30, ROW_METRIC_LABEL, ha="left", va="bottom",
        fontsize=8, fontweight="bold", clip_on=False,
    )

    ax.set_ylim(-0.6, y_top + 0.2)
    # Title on the matrix axis, not the figure: the rotated column numbers sit
    # above the axis, so a figure-level suptitle lands on top of them. `pad` is
    # in points and must clear the tallest number plus its header.
    ax.set_title(title, fontsize=11, pad=44)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


#: Circle radius divided by the radius of the ring the centres sit on. Fixed at
#: the value that reproduces the reference figure: every circle passes close to
#: the ring's centre, so all `n` overlap in the middle and the region count is
#: maximal. The layout is stable for anything in 1.05–1.55 (the region set is
#: identical throughout); 1.2 is what the reference looks like.
RING_RATIO = 1.2

#: Circle radius in figure units. Arbitrary — the axes are scaled to the union —
#: but fixing it keeps `ring_centres` returning literal coordinates a test can
#: compare.
RING_CIRCLE_RADIUS = 1.0

#: Ring layouts are only legible in this range. Below 3 there is no ring (use
#: `plot_venn`); above 8 the smallest region falls under ~30 px at 200 dpi and
#: its label no longer fits inside it.
RING_MIN_SETS, RING_MAX_SETS = 3, 8


def ring_centres(n: int) -> list[tuple[float, float]]:
    """Circle centres for an `n`-set ring, first at 12 o'clock then clockwise.

    Pure function of `n` — **never of the data**. That is the whole premise of
    this figure: it is a template, so a region's position and size say nothing
    about its count and only the printed label does. Anything that made the
    geometry depend on the membership matrix would reintroduce the
    area-encoding this layout was chosen to avoid.
    """
    d = RING_CIRCLE_RADIUS / RING_RATIO
    out = []
    for i in range(n):
        angle = math.pi / 2 - 2 * math.pi * i / n
        out.append((d * math.cos(angle), d * math.sin(angle)))
    return out


@lru_cache(maxsize=8)
def ring_regions(n: int) -> dict[frozenset[int], object]:
    """Every non-empty region of the `n`-circle ring, keyed by its member set.

    A region is the intersection of its members minus the union of everything
    else, so the regions partition the union and each one means "exactly these
    methods, no others" — the same reading as `intersection_table`.

    There are exactly `n*(n-1)+1` of them, and they are precisely the subsets
    that are **contiguous around the ring** (plus the all-`n` centre): six
    circles yield 31, not the 63 a mathematical 6-set Venn needs. That gap is
    not a defect of this implementation — no six circles can realize 63 regions
    — and it is why `ring_coverage_table` exists.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    disks = [
        Point(c).buffer(RING_CIRCLE_RADIUS, quad_segs=256) for c in ring_centres(n)
    ]
    regions: dict[frozenset[int], object] = {}
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            patch = disks[combo[0]]
            for i in combo[1:]:
                patch = patch.intersection(disks[i])
                if patch.is_empty:
                    break
            if patch.is_empty:
                continue
            others = [disks[i] for i in range(n) if i not in combo]
            if others:
                patch = patch.difference(unary_union(others))
            # Boolean ops on 256-segment circles leave slivers of ~1e-12 where
            # two arcs graze; the true regions are all >1e-3 of the union, so
            # this threshold separates them by nine orders of magnitude.
            if patch.is_empty or patch.area < 1e-9:
                continue
            regions[frozenset(combo)] = patch
    return regions


#: Single-letter aliases for the ring's circles, in position order from 12
#: o'clock. Eight letters is exactly `RING_MAX_SETS`, and the alphabet is what
#: needed because a set's identity has to fit where it is drawn, and
#: "Octant-Spline CBG" only just does on a circle of its own.
RING_LETTERS = "ABCDEFGH"


def ring_letter_map(order: list[str]) -> dict[str, str]:
    """Method id -> its single letter, assigned by **ring position**.

    Position rather than identity, unlike `method_colors`: the letter is a
    coordinate on this figure ("the circle at 12 o'clock is A"), so it has to
    follow `--ring-order`. Colour is the thing that stays pinned to the variant
    across figures, and the two together let a reader match a key region to its
    circle either way.
    """
    if len(order) > len(RING_LETTERS):
        raise ValueError(
            f"the ring takes at most {len(RING_LETTERS)} methods, got {len(order)}"
        )
    return {method: RING_LETTERS[i] for i, method in enumerate(order)}


def region_key(order: list[str], members) -> str:
    """`{vanilla_cbg, shortest_ping}` -> `"AC"`: the members' letters, in ring order.

    Concatenated rather than joined with `∧`. The five-member regions of a
    six-ring are ~12 pt across, so every separator character is one the label
    cannot afford; the caption states the convention instead.
    """
    letters = ring_letter_map(order)
    return "".join(letters[m] for m in order if m in members)


def ring_order_for(membership: pd.DataFrame, ring_order=None) -> list[str]:
    """Validate an explicit ring order, or take the frame's column order.

    An explicit order must be a **permutation** of the columns, not a subset: a
    method left off the ring would still be in the membership matrix, so its
    targets would fall into intersections the ring has no region for and vanish
    from the figure without appearing in the undrawn count either.
    """
    columns = list(membership.columns)
    if ring_order is None:
        order = columns
    else:
        order = list(ring_order)
        if sorted(order) != sorted(columns):
            raise ValueError(
                f"ring order must be a permutation of the scored methods; got "
                f"{order} against {columns}. Drop methods with --method, not by "
                f"omitting them from --ring-order."
            )
    if not RING_MIN_SETS <= len(order) <= RING_MAX_SETS:
        raise ValueError(
            f"the ring layout takes {RING_MIN_SETS}-{RING_MAX_SETS} methods, got "
            f"{len(order)}"
        )
    return order


def exact_combination_counts(
    membership: pd.DataFrame, order: list[str]
) -> dict[frozenset[str], int]:
    """Method-id keyed twin of `intersection_table`, for indexing regions by.

    `intersection_table` keys on display *labels* and is the human-readable
    artifact; the figure needs to look regions up by method id, and the two
    must agree exactly — a test asserts every drawn label equals that table's
    count for the same combination.
    """
    keys = membership.apply(lambda r: frozenset(m for m in order if r[m]), axis=1)
    if keys.empty:
        return {}
    return {k: int(v) for k, v in keys.value_counts().items()}


def ring_coverage_table(
    membership: pd.DataFrame, ring_order=None
) -> pd.DataFrame:
    """Every observed intersection, with whether the ring can draw it.

    This is the audit trail for the figure's central omission. The ring realizes
    `n*(n-1)+1` of the `2**n - 1` possible combinations, so on real data some
    non-empty intersections have nowhere to go — around 11-18% of solved targets
    on as01+as02+as03, depending on the ring order. Without this table a reader
    would take the figure's labels for the whole population and be wrong.

    Targets no method got right are **not** listed: they sit outside the union
    rather than in an undrawable region, so counting them as undrawn would
    conflate "the layout cannot show this" with "there is nothing to show".
    """
    order = ring_order_for(membership, ring_order)
    drawable = {frozenset(order[i] for i in combo) for combo in ring_regions(len(order))}
    rows = []
    for combo, n in exact_combination_counts(membership, order).items():
        if not combo:
            continue
        members = [m for m in order if m in combo]
        rows.append(
            {
                "methods": "|".join(label_for(m) for m in members),
                # The same letters the Euler figure puts on its circles, so a
                # row can be traced to the sets it is an intersection of.
                "region": region_key(order, combo),
                "n_methods": len(combo),
                "n_targets": n,
                "share": round(float(n) / len(membership), 4),
                "drawn": combo in drawable,
            }
        )
    return (
        pd.DataFrame(
            rows,
            columns=["methods", "region", "n_methods", "n_targets", "share", "drawn"],
        )
        .sort_values(["drawn", "n_targets"], ascending=[False, False])
        .reset_index(drop=True)
    )


#: Printed on every ring figure. The reference layout looks like an
#: area-proportional diagram and is not one: the circles are identical in every
#: run, so a region being large means the template put it there, not that many
#: targets landed in it. Every quantity on this figure is a printed label.
RING_CAVEAT = (
    "Circle positions and sizes are a fixed template and carry no data — "
    "only the printed percentages do.\n"
    "Each number is the share of the target population solved by exactly that "
    "combination of methods."
)


def _label_anchor(patch):
    """A point comfortably inside `patch`, even when it is a curved sliver.

    `representative_point` only promises to be inside, which for a crescent puts
    the label hard against an arc. `polylabel` returns the pole of
    inaccessibility — the point furthest from any edge — which is what keeps a
    5-set sliver's label off its own boundary.
    """
    from shapely.ops import polylabel

    geom = patch
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    try:
        return polylabel(geom, tolerance=1e-3)
    except Exception:
        return geom.representative_point()


def _region_font_size(area_share: float) -> float:
    """Font tier from the region's share of the union's *area*, not its count.

    Tiered rather than continuous so equally-sized regions get equal type: a
    smooth scale would make the six singles differ by a fraction of a point for
    no reason. The floor is 6.5 pt because below that the digits stop resolving
    at 200 dpi.
    """
    for threshold, size in ((0.03, 12.0), (0.01, 10.0), (0.003, 8.5)):
        if area_share >= threshold:
            return size
    return 6.5


def _draw_circles(ax, order: list[str], colors: dict[str, str], centres, radii) -> None:
    """Fill every circle, then outline every circle. Shared by both layouts.

    Two passes rather than one, so no circle's fill lands on top of another's
    edge. Translucent fills so overlaps blend: no point is covered by more than
    one fill per circle, so the stack tops out at `n` rather than compounding an
    arbitrary number of times over the same pixel.

    **Largest first**, so the biggest set sits at the bottom of the stack and the
    smallest on top. Same-`zorder` artists draw in insertion order, so this is
    the whole mechanism. It matters on the Euler layout, where the radii differ:
    drawing in method order lets a 60%-of-the-population circle be laid over a
    34% one, burying the smaller set's outline in the larger's fill. On the ring
    every radius is equal, so the sort is stable and the ring order survives.
    """
    from matplotlib.patches import Circle

    stacked = sorted(range(len(order)), key=lambda i: -float(radii[i]))
    for i in stacked:
        ax.add_patch(
            Circle(centres[i], radii[i], facecolor=colors[order[i]],
                   edgecolor="none", alpha=0.22, zorder=1)
        )
    for i in stacked:
        ax.add_patch(
            Circle(centres[i], radii[i], facecolor="none",
                   edgecolor=colors[order[i]], linewidth=1.8, zorder=2)
        )


def _draw_ring_outer_labels(
    ax, order: list[str], colors: dict[str, str], texts: dict[str, str]
) -> None:
    """Name each circle just outside its own arc, radially aligned.

    Both ring figures label the same circles in the same places; only the text
    differs (a success rate on the data figure, a letter on the key), so the
    caller supplies it per method.
    """
    n = len(order)
    for i, method in enumerate(order):
        angle = math.pi / 2 - 2 * math.pi * i / n
        reach = RING_CIRCLE_RADIUS / RING_RATIO + RING_CIRCLE_RADIUS + 0.06
        x, y = reach * math.cos(angle), reach * math.sin(angle)
        ax.annotate(
            texts[method],
            xy=(x, y),
            ha="center" if abs(x) < 0.2 else ("left" if x > 0 else "right"),
            va="center" if abs(y) < 0.2 else ("bottom" if y > 0 else "top"),
            fontsize=10, fontweight="bold", color=colors[method], zorder=5,
            linespacing=1.35,
        )


def _annotate_figure(ax, *, title: str, subtitle: str, note: str) -> None:
    """Title above, subtitle under it, footnote below the axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", color=_C_INK, pad=16)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0.5, 1.005), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9.5, color=_C_INK_2,
        )
    ax.annotate(
        note, xy=(0.5, -0.015), xycoords="axes fraction",
        ha="center", va="top", fontsize=8.5, color=_C_MUTED,
    )


def _finish_ring_axes(ax, *, title: str, subtitle: str, note: str) -> None:
    """Square, unframed axes scaled to the ring template."""
    span = RING_CIRCLE_RADIUS / RING_RATIO + RING_CIRCLE_RADIUS
    ax.set_xlim(-span - 0.55, span + 0.55)
    ax.set_ylim(-span - 0.5, span + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    _annotate_figure(ax, title=title, subtitle=subtitle, note=note)


def plot_ring_venn(
    membership: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    subtitle: str = "",
    ring_order=None,
    coverage_ref: str = "overlap_ring_coverage.csv",
) -> Path:
    """The `n`-way ring Venn: fixed circles, percentages written into the regions.

    Six equal circles on a ring is the conventional presentation "6-way Venn".
    It is not a mathematical 6-set Venn — that needs 63 regions and cannot be
    drawn with circles at all — so a fraction of the observed intersections have
    no region here. Both the footnote and `ring_coverage_table` say which, by
    name and by count; that disclosure is what makes the figure honest rather
    than optional polish.

    The geometry never changes: same centres, same radii, every run. Only the
    labels move. That is deliberate — three earlier forms of this figure tried
    to encode counts in area and each one either lost the higher-order regions
    or implied a containment the sets do not have.

    At six methods the middle of the ring is thirteen regions inside a circle's
    radius, so a percentage there cannot also carry the names of the methods it
    belongs to. `plot_euler` is the other half: the same six sets with the
    geometry fitted to the intersections and no number printed anywhere.
    """
    from matplotlib.patheffects import withStroke
    from shapely.ops import unary_union

    order = ring_order_for(membership, ring_order)
    n = len(order)
    total = len(membership)
    if total == 0:
        raise ValueError("no targets to draw")

    regions = ring_regions(n)
    union_area = unary_union(list(regions.values())).area
    counts = exact_combination_counts(membership, order)
    colors = method_colors(order)

    drawable = {frozenset(order[i] for i in combo) for combo in regions}
    n_solved = total - counts.get(frozenset(), 0)
    undrawn = {
        combo: c for combo, c in counts.items() if combo and combo not in drawable
    }
    n_undrawn = sum(undrawn.values())

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    _draw_circles(
        ax, order, colors, ring_centres(n), [RING_CIRCLE_RADIUS] * n
    )

    halo = [withStroke(linewidth=2.4, foreground=_SURFACE)]
    for combo, patch in regions.items():
        key = frozenset(order[i] for i in combo)
        n_here = counts.get(key, 0)
        pct_here = 100.0 * n_here / total if total else 0.0
        point = _label_anchor(patch)
        ax.annotate(
            f"{pct_here:.1f}%",
            xy=(point.x, point.y),
            ha="center", va="center", zorder=4,
            fontsize=_region_font_size(patch.area / union_area),
            # An empty region is drawn, not omitted: "0.0%" is a finding, and a
            # blank space would read as a region the figure forgot.
            color=_C_INK if n_here else _C_MUTED,
            fontweight="bold" if n_here else "normal",
            path_effects=halo,
        )

    _draw_ring_outer_labels(
        ax,
        order,
        colors,
        {
            m: f"{label_for(m)}\n"
               f"{100.0 * int(membership[m].sum()) / total if total else 0.0:.1f}%"
            for m in order
        },
    )

    note = RING_CAVEAT
    if undrawn:
        pct = 100.0 * n_undrawn / n_solved if n_solved else 0.0
        note += (
            f"\n{len(undrawn)} observed combination(s) — {n_undrawn} targets, "
            f"{pct:.1f}% of the {n_solved} solved — have no region in this "
            f"layout; see {coverage_ref}."
        )
    _finish_ring_axes(ax, title=title, subtitle=subtitle, note=note)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# § Euler layout — the geometry carries the numbers
# ---------------------------------------------------------------------------
#
# The ring above is a template: every quantity on it is a printed label. This
# section is the opposite bargain. Circle areas are the set sizes, positions are
# fitted to the observed intersections, and nothing is printed inside a region —
# a pair that never agrees is drawn apart, a set contained in another is drawn
# inside it, and the reader gets the relationships by looking rather than by
# reading 31 numbers.
#
# What no fit can buy is exactness. `n` circles have 2n degrees of freedom
# (3n before the sizes are pinned) against `2**n - 1` regions to satisfy, so at
# six sets the system is overdetermined by an order of magnitude and *some*
# combination is always misdrawn. `EulerLayout.misplaced` measures exactly how
# much, in the only unit that matters here — the share of targets sitting in a
# region the picture assigns to a different combination — and it is printed on
# the figure rather than kept in a log.

#: Circle centres are searched inside this half-width, in the same units as the
#: radii (where the whole target population has area 1). Wide enough that six
#: disjoint circles fit, tight enough to keep the sampling grid dense.
EULER_BOX = 1.6

#: Region areas are measured by sampling this many points per side over
#: `EULER_BOX`. 320 puts ~10 samples in a region holding 0.1% of the population,
#: which is the resolution the objective needs; the reported fit is re-measured
#: at `EULER_FIT_GRID`.
EULER_GRID = 320

#: Denser grid for the numbers that reach the figure and the CSV, so the quoted
#: misplacement is not an artifact of the optimizer's own sampling.
EULER_FIT_GRID = 600

#: Squared-error weight on combinations with **no** targets. Above 1 because
#: "these four methods are never all right together" is a fact about the data,
#: and a layout that draws that region anyway is making one up; the asymmetry
#: buys the topology at a small cost in area accuracy.
EULER_EMPTY_WEIGHT = 8.0

#: Weight on a term pulling the centres toward the origin. Small enough to be
#: swamped by any real area error (the pooled operator runs sit at a loss of
#: ~4.5e-3, this contributes ~6e-6) and large enough to pick the compact layout
#: when two are equally faithful — which is every layout, once all the circles
#: that must be disjoint are.
COMPACTNESS = 1e-6

#: Restarts of the local search, the first from the MDS seed and the rest jittered
#: around it. Three because on the pooled operator runs restarts 4-6 never
#: improved the fit by more than 0.3pp of misplacement and each one costs ~8s.
EULER_RESTARTS = 3


def set_shares(membership: pd.DataFrame, order: list[str]) -> "np.ndarray":
    """Share of the target population each method gets right, in ring order."""
    return np.array([float(membership[m].mean()) for m in order])


def combination_shares(membership: pd.DataFrame, order: list[str]) -> "np.ndarray":
    """Exact-combination shares indexed by bitmask: `1 << i` is `order[i]`.

    The array twin of `intersection_table` — same disjoint reading, same
    denominator, addressable by the bitmask arithmetic the layout needs. Index 0
    is the "no method correct" share, which lives outside every circle and is
    therefore excluded from every fit term below.
    """
    n = len(order)
    cols = np.column_stack([membership[m].to_numpy(dtype=bool) for m in order])
    keys = (cols * (1 << np.arange(n))).sum(axis=1)
    return np.bincount(keys, minlength=1 << n) / len(membership)


def circle_radii(shares: "np.ndarray") -> "np.ndarray":
    """Radii whose **areas** are the set shares — never the radii themselves.

    Encoding a quantity as a radius exaggerates it by squaring; area is what a
    reader integrates when comparing two circles, so area is what carries the
    number.
    """
    return np.sqrt(np.asarray(shares, dtype=float) / math.pi)


def lens_area(r1: float, r2: float, d: float) -> float:
    """Area shared by two circles of radii `r1`, `r2` whose centres are `d` apart."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = max(-1.0, min(1.0, (d * d + r1 * r1 - r2 * r2) / (2 * d * r1)))
    b = max(-1.0, min(1.0, (d * d + r2 * r2 - r1 * r1) / (2 * d * r2)))
    return (
        r1 * r1 * math.acos(a)
        + r2 * r2 * math.acos(b)
        - 0.5 * math.sqrt(
            max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        )
    )


#: Gap left between two circles whose sets never co-occur, as a fraction of the
#: smaller radius. Zero would put them tangent, which reads as "they just barely
#: touch" — the one relationship the data is certain they do not have.
DISJOINT_MARGIN = 0.06


def separation_for_overlap(r1: float, r2: float, target: float) -> float:
    """Centre distance at which two circles share exactly `target` area.

    The three cases are the three Euler relationships, and the first two are
    where this function earns its place — they are the ones the reader is
    entitled to read straight off the picture:

    * `target == 0` -> **disjoint**, pushed past tangency by `DISJOINT_MARGIN`;
    * `target >= min(area)` -> **contained**, one circle fully inside the other;
    * otherwise a bisection on `lens_area`, which is monotone in `d`.
    """
    from scipy.optimize import brentq

    lo, hi = abs(r1 - r2), r1 + r2
    if target <= 0:
        return hi + DISJOINT_MARGIN * min(r1, r2)
    if target >= math.pi * min(r1, r2) ** 2:
        return lo
    return float(brentq(lambda d: lens_area(r1, r2, d) - target, lo + 1e-12, hi - 1e-12))


def _mds_centres(radii: "np.ndarray", pair_target: "np.ndarray") -> "np.ndarray":
    """Starting positions: classical MDS on the ideal pairwise distances.

    Every pair's distance is individually solvable (`separation_for_overlap`);
    what is not simultaneously satisfiable is all of them at once in the plane.
    MDS gives the least-squares compromise, which is a far better start than any
    ring or random scatter — the local search that follows only has to fix the
    higher-order regions.
    """
    n = len(radii)
    d = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d[i, j] = d[j, i] = separation_for_overlap(
            radii[i], radii[j], float(pair_target[i, j])
        )
    j_mat = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * j_mat @ (d ** 2) @ j_mat
    vals, vecs = np.linalg.eigh(gram)
    take = np.argsort(vals)[::-1][:2]
    return vecs[:, take] * np.sqrt(np.maximum(vals[take], 0.0))


def _area_sampler(n: int, radii: "np.ndarray", grid: int):
    """Region areas by point sampling, as a function of the centres.

    Shapely booleans over `2**n - 1` combinations are exact but far too slow to
    sit inside an optimizer loop. A fixed grid turns the whole partition into one
    `bincount` over per-point bitmasks: every point is charged to exactly the
    combination that contains it, so the areas are disjoint and sum to the union
    by construction — the same invariant `intersection_table` has.
    """
    lin = (np.arange(grid) + 0.5) / grid * 2 * EULER_BOX - EULER_BOX
    gx, gy = (a.ravel() for a in np.meshgrid(lin, lin))
    cell = (2 * EULER_BOX / grid) ** 2
    bits = (1 << np.arange(n)).astype(np.int64)
    r2 = (radii ** 2)[:, None]

    def areas(flat_centres) -> "np.ndarray":
        c = np.asarray(flat_centres, dtype=float).reshape(n, 2)
        inside = (gx[None, :] - c[:, 0:1]) ** 2 + (gy[None, :] - c[:, 1:2]) ** 2 <= r2
        return np.bincount((inside * bits[:, None]).sum(axis=0), minlength=1 << n) * cell

    return areas


class EulerLayout:
    """A fitted layout plus everything needed to judge it.

    `observed` and `drawn` are both bitmask-indexed exact-combination shares, so
    they are directly comparable term by term and `misplaced` is a distance
    between two distributions over the same regions.
    """

    def __init__(self, order, centres, radii, observed, drawn, loss):
        self.order = list(order)
        self.centres = centres
        self.radii = radii
        self.observed = observed
        self.drawn = drawn
        self.loss = float(loss)

    @property
    def misplaced(self) -> float:
        """Share of targets the picture puts in the wrong region.

        Total variation between the observed and drawn region distributions:
        move that much probability mass and the two agree. Index 0 (no method
        correct) is excluded — it is outside every circle in both, so it is not
        a region the layout can get wrong.
        """
        return 0.5 * float(np.abs(self.drawn[1:] - self.observed[1:]).sum())

    @property
    def placed(self) -> float:
        """The complement: share of targets whose region the picture gets right."""
        return 1.0 - self.misplaced

    def pair_error(self) -> float:
        """Largest absolute error over the 2-set intersections, in share units.

        Reported separately from `misplaced` because pairs are the relationship
        a reader actually reads off a Euler diagram; the higher-order regions
        are the ones circles cannot control.
        """
        worst = 0.0
        n = len(self.order)
        for i, j in combinations(range(n), 2):
            both = (1 << i) | (1 << j)
            keys = [k for k in range(1, 1 << n) if k & both == both]
            worst = max(
                worst, abs(float(self.drawn[keys].sum() - self.observed[keys].sum()))
            )
        return worst


def fit_euler_layout(
    membership: pd.DataFrame,
    order: list[str],
    *,
    restarts: int = EULER_RESTARTS,
    grid: int = EULER_GRID,
    fit_grid: int = EULER_FIT_GRID,
) -> EulerLayout:
    """Place `n` circles so their overlaps match the observed intersections.

    Radii are fixed by the set sizes, so only the centres are free: `2n`
    parameters against `2**n - 1` regions. The objective is the weighted squared
    error over **every** combination, not just the pairs — fitting pairs alone is
    analytically tidy and produces a picture whose three- and four-way regions
    are whatever happens to fall out, which is precisely the failure the printed
    ring template was chosen to avoid. Combinations observed empty are weighted
    up (`EULER_EMPTY_WEIGHT`) so the layout separates circles rather than
    inventing a region.

    Deterministic: the MDS start is a function of the data and the jitters are
    seeded by restart index, so the same membership matrix always yields the same
    figure.
    """
    from scipy.optimize import minimize

    n = len(order)
    if not 2 <= n <= len(RING_LETTERS):
        raise ValueError(f"an Euler layout takes 2-{len(RING_LETTERS)} sets, got {n}")
    if len(membership) == 0:
        raise ValueError("no targets to lay out")

    observed = combination_shares(membership, order)
    radii = circle_radii(set_shares(membership, order))
    if not (radii > 0).all():
        dead = [m for m, r in zip(order, radii) if r <= 0]
        raise ValueError(
            f"{dead} classified no target correctly, so there is no circle to "
            f"draw for them; drop them with --method"
        )

    cols = np.column_stack([membership[m].to_numpy(dtype=bool) for m in order])
    pair_target = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        pair_target[i, j] = pair_target[j, i] = float((cols[:, i] & cols[:, j]).mean())

    areas = _area_sampler(n, radii, grid)
    weights = np.where(observed > 0, 1.0, EULER_EMPTY_WEIGHT)[1:]

    def loss(flat) -> float:
        diff = areas(flat)[1:] - observed[1:]
        # Areas are only measured inside the sampling box, so a circle that
        # wandered out would be scored on the part that stayed in. The bounds
        # below make that unreachable; the compactness term breaks the ties that
        # sent it there — with every pair already disjoint the objective is
        # exactly flat, and a flat plateau is where a direct search wanders.
        pull = COMPACTNESS * float((np.asarray(flat) ** 2).sum())
        return float((weights * diff * diff).sum()) + pull

    limit = EULER_BOX - float(radii.max())
    bounds = [(-limit, limit)] * (2 * n)
    start = np.clip(_mds_centres(radii, pair_target).ravel(), -limit, limit)
    best = None
    for attempt in range(max(1, restarts)):
        x0 = start if attempt == 0 else np.clip(
            start + np.random.default_rng(attempt).normal(0.0, 0.1, start.size),
            -limit, limit,
        )
        res = minimize(
            loss, x0, method="Powell", bounds=bounds,
            options={"maxiter": 20000, "xtol": 1e-4, "ftol": 1e-7},
        )
        if best is None or res.fun < best.fun:
            best = res

    centres = np.asarray(best.x, dtype=float).reshape(n, 2)
    centres -= centres.mean(axis=0)  # centre the figure, which changes nothing
    drawn = _area_sampler(n, radii, fit_grid)(centres.ravel())
    return EulerLayout(order, centres, radii, observed, drawn, best.fun)


def euler_fit_table(layout: EulerLayout, n_targets: int) -> pd.DataFrame:
    """Observed against drawn, one row per combination — the figure's audit trail.

    Rows for combinations that are empty in the data but drawn anyway are the
    ones to read first: they are the regions the picture asserts and the data
    denies. `delta` is drawn minus observed, so those are exactly the positive
    rows with `n_targets == 0`.
    """
    order = layout.order
    n = len(order)
    rows = []
    for key in range(1, 1 << n):
        members = [order[i] for i in range(n) if key >> i & 1]
        obs, drawn = float(layout.observed[key]), float(layout.drawn[key])
        if obs == 0 and drawn < 1e-4:
            continue
        rows.append(
            {
                "methods": "|".join(label_for(m) for m in members),
                "region": region_key(order, set(members)),
                "n_methods": len(members),
                "n_targets": int(round(obs * n_targets)),
                "observed_share": round(obs, 4),
                "drawn_share": round(drawn, 4),
                "delta": round(drawn - obs, 4),
            }
        )
    return (
        pd.DataFrame(
            rows,
            columns=["methods", "region", "n_methods", "n_targets",
                     "observed_share", "drawn_share", "delta"],
        )
        .sort_values(["observed_share", "drawn_share"], ascending=False)
        .reset_index(drop=True)
    )


#: Printed on every Euler figure. The counterpart to `RING_CAVEAT`, and its
#: mirror image: there, nothing but the labels carried data; here, nothing *but*
#: the geometry does.
EULER_CAPTION = (
    "Circle area = share of targets correct; shared area = share both get right."
)


#: Blank layout units left around the circles. Labels sit on their own circles
#: now, so this only has to clear the stroke and the type's own height.
EULER_MARGIN = 0.22

#: How far in from its own boundary a boundary-anchored label sits, as a share
#: of that circle's radius. Far enough that the type clears the stroke, near
#: enough that it still reads as belonging to the arc rather than the interior.
BOUNDARY_LABEL_INSET = 0.16

#: Minimum distance between two labels, in layout units — roughly the height of
#: a name at the size they are drawn.
LABEL_MIN_GAP = 0.26

#: A set is named inside its own exclusive lobe when that lobe is at least this
#: share of its circle. Below it the lobe is a crescent, and a name centred in a
#: crescent reads as belonging to whatever fills the rest of the circle.
LOBE_LABEL_SHARE = 0.12


def _spread_angles(angles: list[float], min_gap: float) -> list[float]:
    """Nudge angles apart until adjacent ones clear `min_gap`, keeping their order.

    Order is preserved so the labels stay in the same rotational sequence as the
    circles they name, which is what keeps the leader lines from crossing. A few
    relaxation passes are enough for the handful of labels a legible Euler
    diagram can carry; if the ring is too crowded to satisfy the gap, the passes
    simply end with the best spacing they reached.
    """
    if len(angles) < 2:
        return list(angles)
    out = list(angles)
    for _ in range(64):
        moved = False
        for i in range(len(out)):
            j = (i + 1) % len(out)
            gap = (out[j] - out[i]) % (2 * math.pi)
            if gap < min_gap:
                push = (min_gap - gap) / 2
                out[i] -= push
                out[j] += push
                moved = True
        if not moved:
            break
    return out


def _euler_label_points(layout: EulerLayout) -> list[tuple[float, float]]:
    """Where to write each circle's name. Always a point **on that circle**.

    Two placements, in order of preference:

    *Inside its own exclusive lobe*, when the lobe is at least
    `LOBE_LABEL_SHARE` of the circle. That is the placement that reads as "this
    circle is Vanilla CBG", and it is what a well-separated set gets.

    *Just inside its own boundary, on the bearing away from the layout's
    centroid*, otherwise. When a set is nearly contained in another
    (Shortest-Ping sits inside SoI CBG on the operator runs, one target short of
    total) it has no lobe to speak of, and a name centred in a crescent reads as
    belonging to whatever fills the rest of the circle. The outward arc is the
    part of that circle least covered by the others, so a name there is
    attributable to the arc it sits on.

    Every point is within its own circle, which is what lets the figure drop
    leader lines: a label never has to be connected to the thing it names,
    because it is already on it.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    disks = [
        Point(c).buffer(r, quad_segs=128)
        for c, r in zip(layout.centres, layout.radii)
    ]
    hub = layout.centres.mean(axis=0)
    points: dict[int, tuple[float, float]] = {}
    on_boundary: list[int] = []
    for i, disk in enumerate(disks):
        others = [d for k, d in enumerate(disks) if k != i]
        lobe = disk.difference(unary_union(others)) if others else disk
        if not lobe.is_empty and lobe.area >= LOBE_LABEL_SHARE * disk.area:
            point = _label_anchor(lobe)
            points[i] = (point.x, point.y)
        else:
            on_boundary.append(i)

    # Near-coincident circles (Shortest-Ping and SoI CBG differ by one target,
    # so the fit puts them almost on top of each other) leave the hub on almost
    # the same bearing and would stack their names. Spreading the *angles*
    # rather than walking outwards keeps each label on its own circle, which is
    # the property the whole placement rests on.
    bearings = {i: math.atan2(*(layout.centres[i] - hub)[::-1]) for i in on_boundary}
    # Ties would leave the spreading with no order to preserve; index breaks them.
    ranked = sorted(on_boundary, key=lambda i: (bearings[i], i))
    smallest = min((layout.radii[i] for i in on_boundary), default=1.0)
    spread = _spread_angles(
        [bearings[i] for i in ranked], LABEL_MIN_GAP / max(float(smallest), 1e-6)
    )
    for i, angle in zip(ranked, spread):
        reach = layout.radii[i] * (1.0 - BOUNDARY_LABEL_INSET)
        points[i] = tuple(
            layout.centres[i] + reach * np.array([math.cos(angle), math.sin(angle)])
        )

    return [points[i] for i in range(len(disks))]


#: What the space outside every circle is called. It is a real region of the
#: population — the targets no method placed correctly — and leaving it blank
#: invites reading the union of the circles as the whole population when on the
#: pooled operator runs it is 85% of it.
#:
#: Its percentage is `observed[0]` and is exact, but — unlike every circle — its
#: **area is not to scale**: the space outside the union is whatever frame is
#: left over after the layout, not a fitted share. It is the one label here whose
#: number and ink are unrelated, which is why it is drawn muted and outside.
OUTSIDE_LABEL = "None"


def plot_euler(
    layout: EulerLayout,
    out_path: Path,
    *,
    title: str,
    subtitle: str = "",
    fit_ref: str = "overlap_euler_fit.csv",
    caption: bool = True,
) -> Path:
    """Draw the fitted layout: names on the circles, numbers nowhere.

    No region carries a label. At six sets there are thirty-one of them and the
    combination a region stands for is legible from the arcs bounding it, so
    printing every one buries the relationship the layout exists to show.

    Names only — no letter ids. The letters still key `euler_fit_table`'s
    `region` column, but that table also names the methods in full, and the
    ring's own region key is where a letter legend belongs. On this figure they
    were a second line of type per circle buying nothing the colour did not
    already say.

    `caption=False` drops the note under the figure for slide use. It defaults
    on because the layout misplaces a real fraction of the targets — 16% at six
    sets — and a figure that says nothing about that asserts an exactness it
    does not have.
    """
    from matplotlib.patheffects import withStroke

    order = layout.order
    colors = method_colors(order)

    # Window the union, not the origin: the fit centres the circles on their own
    # mean, which is not the middle of what they cover. The canvas then takes the
    # window's aspect, because a fitted layout is rarely square and a square
    # figure would pad the short axis with blank canvas.
    lo = (layout.centres - layout.radii[:, None]).min(axis=0)
    hi = (layout.centres + layout.radii[:, None]).max(axis=0)
    mid = (lo + hi) / 2
    half = (hi - lo) / 2 + EULER_MARGIN
    fig, ax = plt.subplots(
        figsize=(9.0, float(np.clip(9.0 * half[1] / half[0], 5.0, 11.0)))
    )
    ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
    ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_circles(ax, order, colors, layout.centres, layout.radii)

    halo = [withStroke(linewidth=2.6, foreground=_SURFACE)]
    # Centred just under the lowest circle rather than at the foot of the frame:
    # in data units, so it tracks the layout instead of drifting with the margin.
    # `EULER_MARGIN` guarantees the band below `lo[1]` is clear of every circle
    # whatever the fit produced.
    ax.annotate(
        f"{OUTSIDE_LABEL}\n{100 * float(layout.observed[0]):.1f}%",
        xy=(mid[0], lo[1] - 0.03),
        ha="center", va="top", zorder=6,
        fontsize=10.5, fontweight="bold", color=_C_MUTED,
        linespacing=1.35, path_effects=halo,
    )

    for i, (lx, ly) in enumerate(_euler_label_points(layout)):
        method = order[i]
        # `pi * r**2` is not a re-derivation of the share — by `circle_radii`
        # it *is* the circle's drawn area, so the printed number and the ink
        # cannot drift apart. Sets only: an intersection's drawn area is fitted
        # and would disagree with its observed share by up to the figure's
        # error, which is what the caption reports instead.
        ax.annotate(
            f"{label_for(method)}\n{100 * math.pi * layout.radii[i] ** 2:.1f}%",
            xy=(lx, ly),
            ha="center", va="center", zorder=6,
            fontsize=10.5, fontweight="bold", color=colors[method],
            linespacing=1.35, path_effects=halo,
        )

    # One line, not five. The figure's own accuracy is the only thing a reader
    # cannot get from the geometry, so it stays; the rest of what the old
    # caption spelled out is what the picture is already showing them.
    # Two lines, not one: `bbox_inches="tight"` grows the canvas to fit the
    # widest artist, so a single ~200-character note pads the figure out
    # sideways and shrinks the circles it is describing.
    note = (
        f"{EULER_CAPTION}\n"
        f"Fitted layout: {100 * layout.placed:.1f}% of targets land in the "
        f"region drawn (largest pair error {100 * layout.pair_error():.1f}%) "
        f"— see {fit_ref}."
    ) if caption else ""
    _annotate_figure(ax, title=title, subtitle=subtitle, note=note)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def render_overlap(
    run: RunPaths,
    cls_dir: Path,
    *,
    methods: list[str] | None = None,
    venn_methods: list[str] | None = None,
    top_n: int = 1,
) -> dict[str, Path]:
    """Write the membership matrix, both count tables, and the figure(s)."""
    cls_dir = Path(cls_dir)
    chosen = methods or available_methods(cls_dir)
    if len(chosen) < 2:
        raise ValueError(f"need >= 2 methods to show overlap, got {chosen}")

    membership = build_membership(cls_dir, chosen, top_n=top_n)
    written: dict[str, Path] = {}

    def _out(stem: str, ext: str) -> Path:
        return cls_dir / artifact_name(stem, ext, top_n)

    out = membership.rename(columns={m: label_for(m) for m in membership.columns})
    out.to_csv(_out("overlap_membership", "csv"))
    written["membership"] = _out("overlap_membership", "csv")

    intersection_table(membership).to_csv(
        _out("overlap_intersections", "csv"), index=False
    )
    written["intersections"] = _out("overlap_intersections", "csv")
    pairwise_table(membership).to_csv(_out("overlap_pairwise", "csv"), index=False)
    written["pairwise"] = _out("overlap_pairwise", "csv")

    # Skipped rather than raised past the letter cap, the way the ring is: the
    # spec is an input document for a drawing tool, and at as7018's 17 methods
    # it would carry 131,054 relations for a figure no tool will draw.
    if len(chosen) <= len(SET_IDS):
        _out("overlap_venn_spec", "json").write_text(
            json.dumps(venn_spec(membership), indent=2)
        )
        written["venn_spec"] = _out("overlap_venn_spec", "json")

    n = len(membership)
    suffix = "" if top_n == 1 else f", top-{top_n}"
    base_title = f"{run.run_id} — correct classifications ({n} targets{suffix})"

    # The headline is always Shortest-Ping vs "≥1 CBG works" — the
    # rescue-vs-regression trade RQ2 rests on.
    if SHORTEST_PING in chosen and len(chosen) >= 2:
        written["venn"] = plot_sp_vs_cbg_venn(
            membership, _out("overlap_venn", "png"), title=base_title
        )

    if len(chosen) >= 3:
        written["upset"] = plot_upset(
            membership, _out("overlap_upset", "png"), title=base_title
        )

    # Per-method Venn only on explicit request. There is no useful automatic
    # triple: the sets nest, so a "baseline plus the two most accurate" pick
    # just redraws the nesting.
    if venn_methods:
        missing = [m for m in venn_methods if m not in membership.columns]
        if missing:
            raise ValueError(f"--venn-method names unscored methods: {missing}")
        written["venn_methods"] = plot_venn(
            membership[list(venn_methods)].rename(columns=label_for),
            _out("overlap_venn_methods", "png"),
            title=base_title,
        )
    return written


def render_cross_overlap(
    runs: dict[str, RunPaths],
    out_dir: Path,
    *,
    analysis_root: Path | None,
    grid: str,
    resolution: int,
    methods: list[str] | None = None,
    top_n: int = 1,
    ring_order: list[str] | None = None,
) -> dict[str, Path]:
    """Pool several runs into one artifact set under `_cross/venn-diagram/`.

    The same tables and figures `render_overlap` writes per run, plus the ring
    Venn and its backing `overlap_ring_coverage.csv`, computed over the pooled
    target population. Filenames carry the grid slug because — unlike the
    per-run directory — this one is keyed by dataset set alone.
    """
    # `cross_dir` creates its own directory, but `--out-dir` names an arbitrary
    # path the caller has not necessarily made yet.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = grid_slug(grid, resolution)
    membership, origin = pooled_membership(
        runs,
        analysis_root=analysis_root,
        grid=grid,
        resolution=resolution,
        top_n=top_n,
        methods=methods,
    )
    chosen = list(membership.columns)
    written: dict[str, Path] = {}

    def _out(stem: str, ext: str) -> Path:
        return out_dir / artifact_name(stem, ext, top_n, slug)

    labelled = membership.rename(columns={m: label_for(m) for m in chosen})
    labelled.insert(0, "run_id", origin)
    labelled.to_csv(_out("overlap_membership", "csv"))
    written["membership"] = _out("overlap_membership", "csv")

    intersection_table(membership).to_csv(
        _out("overlap_intersections", "csv"), index=False
    )
    written["intersections"] = _out("overlap_intersections", "csv")
    pairwise_table(membership).to_csv(_out("overlap_pairwise", "csv"), index=False)
    written["pairwise"] = _out("overlap_pairwise", "csv")

    if len(chosen) <= len(SET_IDS):
        spec = venn_spec(membership)
        spec["run_ids"] = sorted(runs)
        _out("overlap_venn_spec", "json").write_text(json.dumps(spec, indent=2))
        written["venn_spec"] = _out("overlap_venn_spec", "json")

    n = len(membership)
    per_run = origin.value_counts().sort_index()
    suffix = "" if top_n == 1 else f", top-{top_n}"
    names = " + ".join(cross.short_dataset(r) for r in sorted(runs))
    base_title = f"{names} — correct classifications ({n} targets{suffix})"
    subtitle = " · ".join(
        f"{cross.short_dataset(r)} {int(c)}" for r, c in per_run.items()
    )

    # Same arity guard as the UpSet below: under three methods there is no ring
    # to lay out, and `plot_venn` already draws two or three sets exactly.
    order: list[str] | None = None
    coverage: pd.DataFrame | None = None
    layout: EulerLayout | None = None
    if RING_MIN_SETS <= len(chosen) <= RING_MAX_SETS:
        order = ring_order_for(membership, ring_order)
        coverage = ring_coverage_table(membership, order)
        coverage.to_csv(_out("overlap_ring_coverage", "csv"), index=False)
        written["ring_coverage"] = _out("overlap_ring_coverage", "csv")

        written["ring_venn"] = plot_ring_venn(
            membership,
            _out("overlap_ring_venn", "png"),
            title=base_title,
            subtitle=subtitle,
            ring_order=order,
            coverage_ref=artifact_name("overlap_ring_coverage", "csv", top_n, slug),
        )
        # The other half of the pair: the ring prints every number on a
        # template that means nothing, this fits the geometry to the numbers and
        # prints none of them. Neither is sufficient alone — the ring cannot show
        # that Shortest-Ping is contained in SoI CBG, and the Euler layout cannot
        # tell you the six-way region is 12.7%.
        layout = fit_euler_layout(membership, order)
        euler_fit_table(layout, n).to_csv(_out("overlap_euler_fit", "csv"), index=False)
        written["euler_fit"] = _out("overlap_euler_fit", "csv")
        written["euler"] = plot_euler(
            layout,
            _out("overlap_euler", "png"),
            title=f"{names} — where the methods agree ({n} targets{suffix})",
            subtitle=subtitle,
            fit_ref=artifact_name("overlap_euler_fit", "csv", top_n, slug),
        )
    elif ring_order:
        raise ValueError(
            f"--ring-order was given but the ring takes {RING_MIN_SETS}-"
            f"{RING_MAX_SETS} methods and {len(chosen)} were scored"
        )

    if SHORTEST_PING in chosen:
        written["venn"] = plot_sp_vs_cbg_venn(
            membership, _out("overlap_venn", "png"), title=base_title
        )
    if len(chosen) >= 3:
        written["upset"] = plot_upset(
            membership, _out("overlap_upset", "png"), title=base_title
        )

    manifest = out_dir / artifact_name("manifest", "json", top_n, slug)
    manifest.write_text(
        json.dumps(
            {
                "run_ids": sorted(runs),
                "setups": {rid: r.setup for rid, r in sorted(runs.items())},
                "n_targets_total": n,
                "n_targets_per_run": {r: int(c) for r, c in per_run.items()},
                "methods": chosen,
                "labels": {m: label_for(m) for m in chosen},
                "grid": grid,
                "resolution": int(resolution),
                "top_n": int(top_n),
                "ring_order": order,
                "ring_drawn": order is not None,
                # Position-keyed, so it follows --ring-order rather than the
                # method identity; this is what names the Euler figure's circles.
                "ring_letters": (
                    None if order is None
                    else {v: label_for(k) for k, v in ring_letter_map(order).items()}
                ),
                # The one thing a reader of the figure must not assume: the
                # circles are a template, so size and position carry nothing.
                "ring_venn_encoding": "fixed-template circles; every quantity is "
                                      "a printed region count, never an area",
                "n_regions_drawn": (
                    None if coverage is None else int(coverage["drawn"].sum())
                ),
                "n_regions_undrawn": (
                    None if coverage is None else int((~coverage["drawn"]).sum())
                ),
                "n_targets_undrawn": (
                    None if coverage is None
                    else int(coverage.loc[~coverage["drawn"], "n_targets"].sum())
                ),
                # The Euler figure's honesty number: the rest of the targets
                # are in a region its circles could not realize.
                "euler_placed_share": (
                    None if order is None else round(layout.placed, 4)
                ),
                "euler_max_pair_error": (
                    None if order is None else round(layout.pair_error(), 4)
                ),
                "n_targets_none_correct": int(
                    len(membership) - membership.any(axis=1).sum()
                ),
            },
            indent=2,
        )
    )
    written["manifest"] = manifest
    return written



# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-venn")
    def plot_venn_cmd(
        run_id: list[str] = typer.Option(
            None, "--run-id",
            help="Run to plot (repeatable). One run writes back into that run's "
                 "target-cls-accuracy/; two or more pool them into "
                 "_cross/venn-diagram/. Omit with --all-runs.",
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Plot every run under --outputs-root, "
                                      "each on its own (never pooled)."
        ),
        method: list[str] = typer.Option(
            None,
            "--method",
            help="Methods to include (repeatable). Default: baseline + every "
                 "scored combo. <=3 renders a Venn, >3 renders an UpSet.",
        ),
        venn_method: list[str] = typer.Option(
            None,
            "--venn-method",
            help="2 or 3 methods for an extra per-method Venn "
                 "(overlap_venn_methods.top<N>.png). Off by default — the "
                 "headline Venn is always Shortest-Ping vs >=1 CBG. "
                 "Single-run mode only.",
        ),
        allow_mixed_setups: bool = typer.Option(
            False, "--allow-mixed-setups",
            help="Permit pooling anchors_to_probes with probes_to_anchors runs.",
        ),
        top_n: int = typer.Option(
            1, help="Correct means the true seed ranks below this N."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
        ring_order: list[str] = typer.Option(
            None, "--ring-order",
            help="Clockwise order of the ring Venn's circles, from 12 o'clock "
                 "(repeatable). Must be a permutation of the scored methods. "
                 "Default: display order, so the figure means the same thing in "
                 "every run. Cross-run mode only.",
        ),
        out_dir: Path = typer.Option(
            None, help="Override the pooled output directory (cross-run mode only)."
        ),
    ) -> None:
        """Venn / UpSet / ring Venn of which targets each method gets right.

        Reads target-cls-accuracy/ (from `classify`). One run writes the
        membership matrix, intersection tables and figures back into it. Two or
        more runs are **pooled** — their targets are disjoint — into
        `_cross/venn-diagram/<datasets>/`, which additionally carries the ring
        Venn — fixed-template circles with the exact count in every region —
        and the Euler diagram, whose circles are instead sized and positioned to
        match those counts, with nothing printed inside them.
        """
        ids = list(run_id or [])
        if all_runs == bool(ids):
            raise typer.BadParameter(
                "pass exactly one of --run-id (repeatable) or --all-runs"
            )
        if venn_method and len(venn_method) not in (2, 3):
            raise typer.BadParameter("--venn-method takes 2 or 3 methods")
        if ring_order and len(ids) <= 1:
            raise typer.BadParameter(
                "--ring-order applies to the pooled ring Venn, which only "
                "cross-run mode draws"
            )
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
        methods = list(method) if method else None

        # Unlike `classify`, this command has no answer space to read the grid
        # from — it consumes target-cls-accuracy/, so it has to reconstruct the
        # directory name from the CLI. Passing a --grid/--resolution that was
        # never scored is therefore a missing-directory error, not silent.
        if len(ids) > 1:
            if venn_method:
                raise typer.BadParameter(
                    "--venn-method is single-run only; the pooled figures are "
                    "the ring Venn, the SP-vs-CBG Venn and the UpSet"
                )
            runs = {rid: resolve_run(rid, outputs_root) for rid in ids}
            cross.guard_one_setup(runs, allow_mixed=allow_mixed_setups)
            target_dir = out_dir or cross.cross_dir(
                analysis_root, runs.keys(), kind=CROSS_KIND
            )
            for res in resolutions:
                written = render_cross_overlap(
                    runs,
                    target_dir,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=res,
                    methods=methods,
                    top_n=top_n,
                    ring_order=list(ring_order) if ring_order else None,
                )
                kinds = ", ".join(f"{k}={v.name}" for k, v in written.items())
                typer.echo(
                    f"{'+'.join(sorted(runs))}: {g.name} {g.resolution_arg}={res} · "
                    f"{kinds} -> {target_dir}"
                )
            return

        if out_dir is not None:
            raise typer.BadParameter(
                "--out-dir applies to cross-run mode; a single run always writes "
                "back into its own target-cls-accuracy/"
            )
        runs_list = (
            discover_runs(outputs_root) if all_runs
            else [resolve_run(ids[0], outputs_root)]
        )
        for run, res in [(r, x) for r in runs_list for x in resolutions]:
            cls_dir = run.cls_accuracy_dir(
                root=analysis_root, grid=g.name, resolution=res
            )
            written = render_overlap(
                run,
                cls_dir,
                methods=methods,
                venn_methods=list(venn_method) if venn_method else None,
                top_n=top_n,
            )
            kinds = ", ".join(f"{k}={v.name}" for k, v in written.items())
            typer.echo(
                f"{run.run_id}: {g.name} {g.resolution_arg}={res} · "
                f"{kinds} -> {cls_dir}"
            )
