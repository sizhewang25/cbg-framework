"""The boolean matrix every figure in this package is computed from.

One row per target, one column per method, True where that method put the target
in the right seed. Everything downstream — the tables, the ring, the Euler fit —
is a function of this frame, so the rules about what counts as correct and what
may be pooled with what are enforced once, here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.classify import SHORTEST_PING
from scripts.analysis.v3.modules.diagram.common.labels import PREFERRED_ORDER
from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths


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
            path, columns=["target_id", "status", "tg_seed_rank"]
        ).set_index("target_id")
        solved = (df["status"] == "BASELINE") | df["status"].isin(
            io.CBG_SUCCESS_STATUSES
        )
        rank = df["tg_seed_rank"]
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


def restrict_to_baseline_failures(
    membership: pd.DataFrame, *, baseline: str = SHORTEST_PING
) -> pd.DataFrame:
    """The rows `baseline` got **wrong**, with the baseline column dropped.

    The rescue view's population, and the answer to a question the full matrix
    cannot put: not "does CBG beat the baseline" but "of the targets the baseline
    loses, which variants get them back". On the pooled operator runs at top-1
    that is 665 of 1,269 targets.

    Restricting the rows is what **re-denominates every percentage downstream**.
    A share is now a share of what there was to rescue rather than of the whole
    population, and no caller has to divide by a total other than the
    `len(membership)` it already reports — which is the only reason the ring, the
    Euler fit and both count tables need no changes to read correctly here.

    Dropping the column *follows from* the same filter rather than being a second
    decision: over these rows the baseline is all-False by construction, so as a
    set it is empty, as a Venn circle it is a circle with nothing in it, and as
    an UpSet row it is a row of blanks. Keeping it would also make the
    Shortest-Ping-vs-CBG collapse draw a figure whose left set is empty by
    definition, which is why the rescue pass omits that one artifact.

    Raises if `baseline` is absent: without it there is no definition of the
    targets to restrict to. Callers holding a parallel per-row series (the pooled
    `run_id`) restrict it themselves with `.loc[result.index]`; keeping that out
    of here is what leaves this a frame-in/frame-out function.
    """
    if baseline not in membership.columns:
        raise ValueError(
            f"membership has no {baseline!r} column; the rescue view is defined "
            f"relative to the baseline's failures. Got: {list(membership.columns)}"
        )
    return membership.loc[~membership[baseline]].drop(columns=[baseline])


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
