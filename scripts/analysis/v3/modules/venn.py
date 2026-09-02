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

Two renderings, because arity decides readability. Up to three methods draw as a
classic Venn. Beyond that the region count explodes (6 methods = 63 regions), so
an UpSet plot carries all of them with sorted intersection bars. Both are backed
by the same membership matrix, which is written alongside so the counts are
checkable without reading a figure.

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
    "octant_cbg_hull": "Octant-Hull",
    "octant_cbg_spl": "Octant-Spline",
    "octant_cbg": "Octant-Spline",
    "spotter_cbg": "Spotter",
}

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

MEMBERSHIP_CSV = "overlap_membership.csv"
INTERSECTIONS_CSV = "overlap_intersections.csv"
VENN_PNG = "overlap_venn.png"
UPSET_PNG = "overlap_upset.png"


def label_for(method: str) -> str:
    return LABELS.get(method, method)


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


def plot_venn(membership: pd.DataFrame, out_path: Path, *, title: str) -> Path:
    """Classic 2- or 3-set Venn. Requires 2 or 3 methods."""
    from matplotlib_venn import venn2, venn3

    methods = list(membership.columns)
    if len(methods) not in (2, 3):
        raise ValueError(f"Venn needs 2 or 3 methods, got {len(methods)}")
    sets = [set(membership.index[membership[m]]) for m in methods]
    labels = tuple(label_for(m) for m in methods)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    draw = venn2 if len(methods) == 2 else venn3
    draw(sets, set_labels=labels, ax=ax)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_upset(membership: pd.DataFrame, out_path: Path, *, title: str) -> Path:
    """UpSet plot — the readable form once arity exceeds three."""
    from upsetplot import UpSet, from_indicators

    renamed = membership.rename(columns={m: label_for(m) for m in membership.columns})
    data = from_indicators(list(renamed.columns), renamed)

    fig = plt.figure(figsize=(11.0, 6.4))
    UpSet(
        data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
        min_subset_size=1,
    ).plot(fig=fig)
    fig.suptitle(title, fontsize=11)
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

    out = membership.rename(columns={m: label_for(m) for m in membership.columns})
    out.to_csv(cls_dir / MEMBERSHIP_CSV)
    written["membership"] = cls_dir / MEMBERSHIP_CSV

    intersection_table(membership).to_csv(cls_dir / INTERSECTIONS_CSV, index=False)
    written["intersections"] = cls_dir / INTERSECTIONS_CSV
    pairwise_table(membership).to_csv(cls_dir / "overlap_pairwise.csv", index=False)
    written["pairwise"] = cls_dir / "overlap_pairwise.csv"

    n = len(membership)
    suffix = "" if top_n == 1 else f", top-{top_n}"
    base_title = f"{run.run_id} — correct classifications ({n} targets{suffix})"

    if len(chosen) <= 3:
        written["venn"] = plot_venn(membership, cls_dir / VENN_PNG, title=base_title)
    else:
        written["upset"] = plot_upset(membership, cls_dir / UPSET_PNG, title=base_title)
        # A 3-set Venn stays useful as the headline: baseline against two
        # variants. Default to the baseline plus the two most accurate.
        triple = venn_methods
        if triple is None:
            ranked = sorted(chosen, key=lambda m: -int(membership[m].sum()))
            head = [SHORTEST_PING] if SHORTEST_PING in chosen else []
            rest = [m for m in ranked if m not in head][: 3 - len(head)]
            triple = head + rest
        if len(triple) == 3:
            written["venn"] = plot_venn(
                membership[triple], cls_dir / VENN_PNG, title=base_title
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
            help="Exactly 3 methods for the companion Venn when >3 are plotted. "
                 "Default: baseline plus the two most accurate.",
        ),
        top_n: int = typer.Option(
            1, help="Correct means the true seed ranks below this N."
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
        if venn_method and len(venn_method) != 3:
            raise typer.BadParameter("--venn-method takes exactly 3 methods")

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run in runs:
            cls_dir = run.cls_accuracy_dir(root=analysis_root)
            written = render_overlap(
                run,
                cls_dir,
                methods=list(method) if method else None,
                venn_methods=list(venn_method) if venn_method else None,
                top_n=top_n,
            )
            kinds = ", ".join(f"{k}={v.name}" for k, v in written.items())
            typer.echo(f"{run.run_id}: {kinds} -> {cls_dir}")
