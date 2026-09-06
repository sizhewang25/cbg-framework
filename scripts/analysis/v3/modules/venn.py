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

**Where the code lives.** This module is the `plot-venn` command and the two
`render_*` functions that assemble an artifact set. The figures themselves are
in `diagram/`, split by the bargain each makes: `diagram/venn/` fixes the
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
    "ring_centres",
    "ring_coverage_table",
    "ring_letter_map",
    "ring_order_for",
    "ring_regions",
    "separation_for_overlap",
    "set_shares",
    "venn_spec",
]


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
