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
counts and percentages written into the regions. The sets are routinely nested
(`all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` on every run we have), and no
area-proportional layout can render nesting; area-proportional drawing warned on
every run for exactly that reason.

Both figures are backed by the same membership matrix, written alongside so the
counts are checkable without reading a figure. Every artifact is suffixed with
its top-N, so top-1 and top-3 coexist rather than overwriting.

Command: `plot-venn`.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from scripts.analysis.v3.modules import healpix as hx  # noqa: E402
from scripts.analysis.v3.modules import io  # noqa: E402
from scripts.analysis.v3.modules.classify import SHORTEST_PING  # noqa: E402
from scripts.analysis.v3.modules.paths import (  # noqa: E402
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)

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

def artifact_name(stem: str, ext: str, top_n: int) -> str:
    """`("overlap_upset", "png", 3)` -> `"overlap_upset.top3.png"`.

    Every overlap artifact is a function of `top_n`, so the suffix is
    mandatory: without it a top-3 run silently overwrites the top-1 files.
    """
    return f"{stem}.top{top_n}.{ext}"


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


def plot_venn(membership: pd.DataFrame, out_path: Path, *, title: str) -> Path:
    """Classic 2- or 3-set Venn, drawn unweighted. Requires 2 or 3 methods.

    Unweighted (fixed-size, evenly positioned circles) rather than
    area-proportional, because the correctness sets are routinely nested —
    `all-CBG ⊆ Shortest-Ping ⊆ ≥1-CBG` holds on every run we have — and no
    area-proportional layout can render nesting. The counts carry the
    information instead of the areas.
    """
    # `venn2_unweighted` / `venn3_unweighted` are deprecated in matplotlib-venn
    # 1.1.2 *and* broken — they forward `normalize_to` into a custom layout that
    # rejects it. Drive the layout algorithm directly instead.
    from matplotlib_venn import venn2, venn3
    from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm as Venn2Layout
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm as Venn3Layout

    methods = list(membership.columns)
    if len(methods) not in (2, 3):
        raise ValueError(f"Venn needs 2 or 3 methods, got {len(methods)}")
    sets = [set(membership.index[membership[m]]) for m in methods]
    labels = tuple(label_for(m) for m in methods)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    if len(methods) == 2:
        draw, layout = venn2, Venn2Layout(fixed_subset_sizes=(1, 1, 1))
    else:
        draw, layout = venn3, Venn3Layout(fixed_subset_sizes=(1,) * 7)
    draw(
        sets,
        set_labels=labels,
        ax=ax,
        layout_algorithm=layout,
        subset_label_formatter=_region_labeller(len(membership)),
    )
    # The "no method correct" region falls outside every circle, so a Venn
    # cannot show it. It is part of the denominator and often large, so state
    # it rather than leaving the reader to subtract.
    n_total = len(membership)
    n_none = int((~membership.any(axis=1)).sum())
    pct_none = 100.0 * n_none / n_total if n_total else 0.0
    ax.annotate(
        f"none correct: {n_none} ({pct_none:.1f}%)",
        xy=(0.5, -0.02),
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
            label_for(SHORTEST_PING): membership[SHORTEST_PING],
            CBG_ANY_LABEL: cbg.any(axis=1),
        },
        index=membership.index,
    )
    return collapsed, cbg.shape[1]


def plot_sp_vs_cbg_venn(
    membership: pd.DataFrame, out_path: Path, *, title: str
) -> Path:
    """Unweighted 2-set Venn: Shortest-Ping vs "at least one CBG works".

    This is the rescue-vs-regression trade an aggregate accuracy number hides.
    The CBG-only region counts targets CBG rescues; the Shortest-Ping-only
    region counts the ones it regresses on.
    """
    collapsed, n_cbg = collapse_to_sp_vs_cbg(membership)
    subtitle = f"≥1 of {n_cbg} CBG variant{'s' if n_cbg != 1 else ''}"
    return plot_venn(collapsed, out_path, title=f"{title}\n{subtitle}")


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
            membership[list(venn_methods)],
            _out("overlap_venn_methods", "png"),
            title=base_title,
        )
    return written


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("plot-venn")
    def plot_venn_cmd(
        run_id: str = typer.Option(
            None, help="Run to plot. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Plot every run under --outputs-root."
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
                 "headline Venn is always Shortest-Ping vs >=1 CBG.",
        ),
        top_n: int = typer.Option(
            1, help="Correct means the true seed ranks below this N."
        ),
        nside: list[int] = typer.Option(
            [hx.DEFAULT_NSIDE],
            "--nside",
            help="Which scored answer space(s) to read (repeatable), matching "
                 "target-cls-accuracy/nside-<x>/ written by `classify`.",
        ),
        sweep: bool = typer.Option(
            False,
            "--sweep",
            help=f"Shorthand for the full hierarchy {list(hx.NSIDE_HIERARCHY)}.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Venn / UpSet of which targets each method classifies correctly.

        Reads target-cls-accuracy/ (from `classify`) and writes the membership
        matrix, intersection tables and figure(s) back into it.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if venn_method and len(venn_method) not in (2, 3):
            raise typer.BadParameter("--venn-method takes 2 or 3 methods")

        nsides = list(hx.NSIDE_HIERARCHY) if sweep else list(dict.fromkeys(nside))

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run, ns in [(r, n) for r in runs for n in nsides]:
            cls_dir = run.cls_accuracy_dir(root=analysis_root, nside=ns)
            written = render_overlap(
                run,
                cls_dir,
                methods=list(method) if method else None,
                venn_methods=list(venn_method) if venn_method else None,
                top_n=top_n,
            )
            kinds = ", ".join(f"{k}={v.name}" for k, v in written.items())
            typer.echo(f"{run.run_id}: nside={ns} · {kinds} -> {cls_dir}")
