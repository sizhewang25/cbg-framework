"""Set-overlap of correct classifications across methods (§8.2's regression view).

Consumes `classify`'s per-target output and turns each method into a set: the
targets it classified into the *right* seed. Overlaps then answer the question an
aggregate accuracy number hides — which targets does CBG win that Shortest-Ping
loses, and which does it lose that Shortest-Ping already had. That trade is the
point: "CBG beats the baseline" can be true in aggregate while a variant
regresses on targets the baseline solved.

Correctness uses the same rule as the accuracy table: `tg_seed_rank < N`
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

**The rescue view.** `--rescue-view` writes a second artifact set,
`rescue_overlap_*`, over a *restricted* population: only the targets
Shortest-Ping got wrong, and only the CBG variants as sets. The headline
collapse can say that CBG rescues what the baseline loses; it cannot say which
variants do it, because the baseline is still one of the sets and the
denominator is still every target. Restricting the rows fixes both at once — the
baseline column is dropped because over those rows it is all-False, so as a set
it is empty, which is also why this pass omits the Shortest-Ping collapse and
nothing else.

Every percentage in that set is a share of **the baseline's failures**, not of
the population. On the pooled operator runs at top-1 that denominator is 665 of
1,269 targets: CBG rescues 495 of them (74.4%), 170 are missed by every variant,
and **no target is rescued by all five** — Octant-Hull 335, Octant-Spline 253,
Spotter 224, Vanilla 131, SoI 13. Five sets is inside the ring's arity range, so
the pooled directory draws the ring and the Euler layout here too; a variant
that rescues nothing has a zero-radius circle, which the ring prints as a
deliberate "0.0%" and the Euler fit refuses, so that figure is skipped with the
manifest naming why.

**Where the code lives.** This module is the `plot-venn` command and the two
`render_*` functions that assemble an artifact set. Both delegate the writing to
one `_write_artifact_set`, so the four passes — two modes against two views —
cannot drift into disagreeing about what an artifact set contains. The figures
themselves are in `diagram/`, split by the bargain each makes: `diagram/venn/` fixes the
geometry and prints the numbers (classic Venn, ring template, UpSet),
`diagram/euler/` fits the geometry to the numbers, and `diagram/common/` holds
what both need — labels, palette, the membership matrix and the count tables.
The names below are re-exported so this module stays the single import surface
for the whole toolkit; `pareto.py` and the tests both rely on that.

Command: `plot-venn`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import typer

from scripts.analysis.v3.modules import cross
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    RunPaths,
    discover_runs,
    grid_slug,
    resolve_run,
)

# Re-exported, not merely used: this module is the import surface for the whole
# overlap toolkit. `pareto.py` reads `label_for`/`method_colors`/`PREFERRED_ORDER`
# off it, `test_io_and_venn.py` reaches for most of the rest, and the two
# `render_*` functions below use nearly all of it anyway. The definitions live
# under `diagram/`.
from scripts.analysis.v3.modules.diagram.common import (
    LABELS,
    PREFERRED_ORDER,
    RING_LETTERS,
    RUN_KEY_SEP,
    SET_IDS,
    artifact_name,
    available_methods,
    build_membership,
    exact_combination_counts,
    intersection_table,
    label_for,
    method_colors,
    pairwise_table,
    pooled_membership,
    region_key,
    restrict_to_baseline_failures,
    ring_letter_map,
    venn_spec,
)
from scripts.analysis.v3.modules.diagram.euler import (
    EULER_CAPTION,
    LABEL_MIN_GAP,
    OUTSIDE_LABEL,
    EulerLayout,
    circle_radii,
    combination_shares,
    euler_fit_table,
    fit_euler_layout,
    lens_area,
    plot_euler,
    separation_for_overlap,
    set_shares,
)
from scripts.analysis.v3.modules.diagram.venn import (
    CBG_ANY_LABEL,
    COLUMN_METRIC_LABEL,
    RING_CAVEAT,
    RING_CIRCLE_RADIUS,
    RING_MAX_SETS,
    RING_MIN_SETS,
    RING_RATIO,
    ROW_METRIC_LABEL,
    SP_VS_CBG_LABEL,
    collapse_to_sp_vs_cbg,
    plot_ring_venn,
    plot_sp_vs_cbg_venn,
    plot_upset,
    plot_venn,
    ring_centres,
    ring_coverage_table,
    ring_order_for,
    ring_regions,
)

#: This module's artifact kind under `_cross/`, the sibling of `pareto`'s
#: `"cost-accuracy"`.
CROSS_KIND = "venn-diagram"

#: The re-export surface, declared rather than left implicit: most of the
#: names above are imported for callers rather than for the two `render_*`
#: functions below, and without this they read as dead imports.
__all__ = [
    "CBG_ANY_LABEL",
    "COLUMN_METRIC_LABEL",
    "CROSS_KIND",
    "EULER_CAPTION",
    "EulerLayout",
    "LABELS",
    "LABEL_MIN_GAP",
    "OUTSIDE_LABEL",
    "PREFERRED_ORDER",
    "RING_CAVEAT",
    "RING_CIRCLE_RADIUS",
    "RING_LETTERS",
    "RING_MAX_SETS",
    "RING_MIN_SETS",
    "RING_RATIO",
    "ROW_METRIC_LABEL",
    "RUN_KEY_SEP",
    "SET_IDS",
    "SHORTEST_PING",
    "SP_VS_CBG_LABEL",
    "artifact_name",
    "available_methods",
    "build_membership",
    "circle_radii",
    "collapse_to_sp_vs_cbg",
    "combination_shares",
    "euler_fit_table",
    "exact_combination_counts",
    "fit_euler_layout",
    "intersection_table",
    "label_for",
    "lens_area",
    "method_colors",
    "pairwise_table",
    "plot_euler",
    "plot_ring_venn",
    "plot_sp_vs_cbg_venn",
    "plot_upset",
    "plot_venn",
    "pooled_membership",
    "region_key",
    "register",
    "render_cross_overlap",
    "render_overlap",
    "restrict_to_baseline_failures",
    "ring_centres",
    "ring_coverage_table",
    "ring_letter_map",
    "ring_order_for",
    "ring_regions",
    "separation_for_overlap",
    "set_shares",
    "venn_spec",
]


#: Filename stem for the full-population artifact set, and for the rescue
#: view's. Paired here because the `written` dict's `rescue_` key prefix and the
#: files' `rescue_overlap` stem have to move together — a reader matching a
#: logged key to a file on disk is relying on exactly that.
_STEM = "overlap"
_RESCUE_STEM = "rescue_overlap"


class _RingFit(NamedTuple):
    """What the ring/Euler pass produced, for the manifest to report.

    `layout` is None when the Euler fit was skipped, with `euler_skipped` naming
    the methods that caused it. The ring still draws in that case, so the two
    figures cannot be reported by one flag.
    """

    order: list[str]
    coverage: pd.DataFrame
    layout: EulerLayout | None
    euler_skipped: list[str]


def _write_artifact_set(
    membership: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    top_n: int,
    slug: str | None = None,
    title: str,
    subtitle: str = "",
    euler_title: str | None = None,
    origin: pd.Series | None = None,
    run_ids: list[str] | None = None,
    ring: bool = False,
    ring_order: list[str] | None = None,
) -> tuple[dict[str, Path], _RingFit | None]:
    """Membership matrix, both count tables, the spec, and the figures the arity allows.

    The one place an overlap artifact is written, so the passes — two modes
    (per-run, pooled) against two views (full population, rescue) — cannot drift
    into disagreeing about what an artifact set contains. *Which* figures appear
    is a function of the frame rather than of a flag: the Shortest-Ping collapse
    needs the baseline column, the UpSet needs three methods.

    `ring` is the exception, and a real mode difference rather than a data one:
    the per-run directory deliberately carries no ring or Euler figure even at
    six methods.

    Titles and the filename `stem` are passed in rather than built here, so that
    extracting this function moved no format string and the artifacts stay
    byte-identical to the two functions it was lifted out of.
    """
    def _out(name: str, ext: str) -> Path:
        return out_dir / artifact_name(name, ext, top_n, slug)

    written: dict[str, Path] = {}
    chosen = list(membership.columns)
    n = len(membership)

    labelled = membership.rename(columns={m: label_for(m) for m in chosen})
    if origin is not None:
        labelled.insert(0, "run_id", origin)
    labelled.to_csv(_out(f"{stem}_membership", "csv"))
    written["membership"] = _out(f"{stem}_membership", "csv")

    intersection_table(membership).to_csv(
        _out(f"{stem}_intersections", "csv"), index=False
    )
    written["intersections"] = _out(f"{stem}_intersections", "csv")
    pairwise_table(membership).to_csv(_out(f"{stem}_pairwise", "csv"), index=False)
    written["pairwise"] = _out(f"{stem}_pairwise", "csv")

    # Skipped rather than raised past the letter cap, the way the ring is: the
    # spec is an input document for a drawing tool, and at as7018's 17 methods
    # it would carry 131,054 relations for a figure no tool will draw.
    if len(chosen) <= len(SET_IDS):
        spec = venn_spec(membership)
        if run_ids is not None:
            spec["run_ids"] = run_ids
        _out(f"{stem}_venn_spec", "json").write_text(json.dumps(spec, indent=2))
        written["venn_spec"] = _out(f"{stem}_venn_spec", "json")

    # Same arity guard as the UpSet below: under three methods there is no ring
    # to lay out, and `plot_venn` already draws two or three sets exactly.
    fit: _RingFit | None = None
    if ring and RING_MIN_SETS <= len(chosen) <= RING_MAX_SETS:
        order = ring_order_for(membership, ring_order)
        coverage = ring_coverage_table(membership, order)
        coverage.to_csv(_out(f"{stem}_ring_coverage", "csv"), index=False)
        written["ring_coverage"] = _out(f"{stem}_ring_coverage", "csv")

        written["ring_venn"] = plot_ring_venn(
            membership,
            _out(f"{stem}_ring_venn", "png"),
            title=title,
            subtitle=subtitle,
            ring_order=order,
            coverage_ref=artifact_name(f"{stem}_ring_coverage", "csv", top_n, slug),
        )
        # The other half of the pair: the ring prints every number on a
        # template that means nothing, this fits the geometry to the numbers and
        # prints none of them. Neither is sufficient alone — the ring cannot show
        # that Shortest-Ping is contained in SoI CBG, and the Euler layout cannot
        # tell you the six-way region is 12.7%.
        #
        # A method correct on no target in this population has a zero-radius
        # circle, and `fit_euler_layout` refuses to draw one. Skip the figure
        # rather than let that abort the command: the ring renders such a set
        # correctly — it prints a deliberate "0.0%" — and the fit's own remedy
        # ("drop them with --method") would change the question being asked
        # wherever the empty set is one of the variants under comparison. The
        # manifest names what was skipped so the gap is never silent.
        dead = [m for m in order if not membership[m].any()]
        layout = None if dead else fit_euler_layout(membership, order)
        if layout is not None:
            euler_fit_table(layout, n).to_csv(
                _out(f"{stem}_euler_fit", "csv"), index=False
            )
            written["euler_fit"] = _out(f"{stem}_euler_fit", "csv")
            written["euler"] = plot_euler(
                layout,
                _out(f"{stem}_euler", "png"),
                title=euler_title or title,
                subtitle=subtitle,
                fit_ref=artifact_name(f"{stem}_euler_fit", "csv", top_n, slug),
            )
        fit = _RingFit(
            order=order, coverage=coverage, layout=layout, euler_skipped=dead
        )
    elif ring and ring_order:
        raise ValueError(
            f"--ring-order was given but the ring takes {RING_MIN_SETS}-"
            f"{RING_MAX_SETS} methods and this view has {len(chosen)}"
        )

    # The headline is always Shortest-Ping vs "≥1 CBG works" — the
    # rescue-vs-regression trade RQ2 rests on.
    if SHORTEST_PING in chosen and len(chosen) >= 2:
        written["venn"] = plot_sp_vs_cbg_venn(
            membership, _out(f"{stem}_venn", "png"), title=title
        )

    if len(chosen) >= 3:
        written["upset"] = plot_upset(
            membership, _out(f"{stem}_upset", "png"), title=title
        )
    return written, fit


def _ring_manifest_fields(fit: _RingFit | None) -> dict[str, object]:
    """The ring/Euler half of a manifest, or all-None when neither was drawn."""
    coverage = None if fit is None else fit.coverage
    layout = None if fit is None else fit.layout
    return {
        "ring_order": None if fit is None else fit.order,
        "ring_drawn": fit is not None,
        # Position-keyed, so it follows --ring-order rather than the method
        # identity; this is what names the Euler figure's circles.
        "ring_letters": (
            None if fit is None
            else {v: label_for(k) for k, v in ring_letter_map(fit.order).items()}
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
        "euler_drawn": fit is not None and layout is not None,
        # Names the skip above rather than leaving a missing figure unexplained.
        "euler_skipped_methods": (
            None if fit is None else (fit.euler_skipped or None)
        ),
        # The Euler figure's honesty number: the rest of the targets
        # are in a region its circles could not realize.
        "euler_placed_share": (
            None if layout is None else round(layout.placed, 4)
        ),
        "euler_max_pair_error": (
            None if layout is None else round(layout.pair_error(), 4)
        ),
    }


def _guard_rescue(chosen: list[str], *, baseline: str = SHORTEST_PING) -> None:
    """Refuse a rescue view that cannot be drawn — before anything is written.

    Checked up front rather than at the filter, so a `--method`/`--rescue-view`
    combination that cannot work fails before the full pass has left half an
    artifact set on disk.
    """
    if baseline not in chosen:
        raise ValueError(
            f"--rescue-view needs the {baseline!r} baseline among the scored "
            f"methods, since the rescue view is defined relative to the "
            f"baseline's failures. Got: {chosen}"
        )
    cbg = [m for m in chosen if m != baseline]
    if len(cbg) < 2:
        raise ValueError(
            f"--rescue-view needs >= 2 CBG variants besides {baseline}, got "
            f"{cbg}; with one there is no overlap left to show"
        )


def _write_rescue_view(
    membership: pd.DataFrame,
    out_dir: Path,
    *,
    top_n: int,
    slug: str | None,
    name: str,
    suffix: str,
    origin: pd.Series | None = None,
    run_ids: list[str] | None = None,
    ring: bool = False,
    ring_order: list[str] | None = None,
) -> tuple[dict[str, Path], _RingFit | None, pd.DataFrame]:
    """The second artifact pass: CBG variants over the baseline's failures alone.

    The full-population figures can say that CBG rescues what Shortest-Ping
    loses; they cannot say *which* variants do it, because the baseline is still
    one of the sets and the denominator is still every target. Restricting the
    population answers that, and costs no new figure code — every table and
    layout in `diagram/` is a pure function of the membership frame.

    Returns the `rescue_`-prefixed paths, the ring fit, and the restricted frame
    itself: the caller needs the frame for the manifest, since every count in
    that block is denominated in it rather than in the full population.
    """
    rescue = restrict_to_baseline_failures(membership)
    # `intersection_table` sorts an empty frame by a column it never built, so
    # an unrestricted-away population surfaces as a bare KeyError otherwise.
    if rescue.empty:
        raise ValueError(
            f"--rescue-view: {SHORTEST_PING} was correct on all "
            f"{len(membership)} targets, so there is nothing left to rescue"
        )
    origin = None if origin is None else origin.loc[rescue.index]
    per_run = None if origin is None else origin.value_counts().sort_index()
    n = len(rescue)
    baseline = LABELS[SHORTEST_PING]
    written, fit = _write_artifact_set(
        rescue,
        out_dir,
        stem=_RESCUE_STEM,
        top_n=top_n,
        slug=slug,
        title=(
            f"{name} — CBG rescues of {baseline} failures "
            f"({n} of {len(membership)} targets{suffix})"
        ),
        subtitle=(
            "" if per_run is None
            else " · ".join(
                f"{cross.short_dataset(r)} {int(c)}" for r, c in per_run.items()
            )
        ),
        euler_title=(
            f"{name} — where the CBG variants agree on {baseline}'s failures "
            f"({n} targets{suffix})"
        ),
        origin=origin,
        run_ids=run_ids,
        ring=ring,
        # The clockwise sequence the caller asked for, minus the baseline.
        # `ring_order_for` demands a permutation of the frame's columns, so the
        # full order cannot be reused verbatim; filtering keeps their intent and
        # satisfies that requirement by construction.
        ring_order=(
            None if ring_order is None
            else [m for m in ring_order if m != SHORTEST_PING]
        ),
    )
    return {f"rescue_{k}": v for k, v in written.items()}, fit, rescue


def _rescue_manifest_fields(
    rescue: pd.DataFrame, fit: _RingFit | None, *, origin: pd.Series | None = None
) -> dict[str, object]:
    """The manifest's `rescue_view` block.

    `population` and `denominator` are the load-bearing entries: they are the
    only thing stopping a reader from comparing a share in here against a
    full-population share elsewhere in the same file.
    """
    per_run = None if origin is None else origin.value_counts().sort_index()
    correct = rescue.any(axis=1)
    return {
        "baseline": SHORTEST_PING,
        "population": "targets the baseline classified incorrectly",
        "denominator": "restricted",
        "n_targets": int(len(rescue)),
        "n_targets_per_run": (
            None if per_run is None else {r: int(c) for r, c in per_run.items()}
        ),
        "methods": list(rescue.columns),
        "labels": {m: label_for(m) for m in rescue.columns},
        "n_rescued_by_any": int(correct.sum()),
        "share_rescued_by_any": round(float(correct.mean()), 4),
        "n_targets_none_correct": int((~correct).sum()),
        "n_correct_per_method": {m: int(rescue[m].sum()) for m in rescue.columns},
        **_ring_manifest_fields(fit),
    }


def render_overlap(
    run: RunPaths,
    cls_dir: Path,
    *,
    methods: list[str] | None = None,
    venn_methods: list[str] | None = None,
    top_n: int = 1,
    rescue_view: bool = False,
) -> dict[str, Path]:
    """Write the membership matrix, both count tables, and the figure(s)."""
    cls_dir = Path(cls_dir)
    chosen = methods or available_methods(cls_dir)
    if len(chosen) < 2:
        raise ValueError(f"need >= 2 methods to show overlap, got {chosen}")
    if rescue_view:
        _guard_rescue(chosen)

    membership = build_membership(cls_dir, chosen, top_n=top_n)

    n = len(membership)
    suffix = "" if top_n == 1 else f", top-{top_n}"
    base_title = f"{run.run_id} — correct classifications ({n} targets{suffix})"

    written, _ = _write_artifact_set(
        membership, cls_dir, stem=_STEM, top_n=top_n, title=base_title
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
            cls_dir / artifact_name("overlap_venn_methods", "png", top_n),
            title=base_title,
        )

    if rescue_view:
        rescue_written, _, _ = _write_rescue_view(
            membership,
            cls_dir,
            top_n=top_n,
            slug=None,
            name=run.run_id,
            suffix=suffix,
        )
        written.update(rescue_written)
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
    rescue_view: bool = False,
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
    if rescue_view:
        _guard_rescue(chosen)

    n = len(membership)
    per_run = origin.value_counts().sort_index()
    suffix = "" if top_n == 1 else f", top-{top_n}"
    names = " + ".join(cross.short_dataset(r) for r in sorted(runs))
    base_title = f"{names} — correct classifications ({n} targets{suffix})"
    subtitle = " · ".join(
        f"{cross.short_dataset(r)} {int(c)}" for r, c in per_run.items()
    )

    written, fit = _write_artifact_set(
        membership,
        out_dir,
        stem=_STEM,
        top_n=top_n,
        slug=slug,
        title=base_title,
        subtitle=subtitle,
        euler_title=f"{names} — where the methods agree ({n} targets{suffix})",
        origin=origin,
        run_ids=sorted(runs),
        ring=True,
        ring_order=ring_order,
    )

    rescue: pd.DataFrame | None = None
    rescue_fit: _RingFit | None = None
    if rescue_view:
        rescue_written, rescue_fit, rescue = _write_rescue_view(
            membership,
            out_dir,
            top_n=top_n,
            slug=slug,
            name=names,
            suffix=suffix,
            origin=origin,
            run_ids=sorted(runs),
            ring=True,
            ring_order=ring_order,
        )
        written.update(rescue_written)

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
                **_ring_manifest_fields(fit),
                "n_targets_none_correct": int(
                    len(membership) - membership.any(axis=1).sum()
                ),
                # Nested rather than flattened: every count in here is
                # denominated in the restricted population, so mixing it into a
                # top level that means "all targets" would invite exactly the
                # comparison the block's own `denominator` warns against.
                "rescue_view": (
                    None if rescue is None
                    else _rescue_manifest_fields(
                        rescue, rescue_fit, origin=origin.loc[rescue.index]
                    )
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
        rescue_view: bool = typer.Option(
            False, "--rescue-view",
            help="Also write a second artifact set (rescue_overlap_*) over the "
                 "CBG variants alone, restricted to the targets Shortest-Ping "
                 "got wrong — which variants rescue what the baseline loses. "
                 "Percentages there are shares of the baseline's failures, not "
                 "of the whole population. Both modes.",
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
                    rescue_view=rescue_view,
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
                rescue_view=rescue_view,
            )
            kinds = ", ".join(f"{k}={v.name}" for k, v in written.items())
            typer.echo(
                f"{run.run_id}: {g.name} {g.resolution_arg}={res} · "
                f"{kinds} -> {cls_dir}"
            )
