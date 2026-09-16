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


#: Set on the returned frame: how many baseline-only targets `build_membership`
#: dropped to reach a common denominator. Non-zero only on an arm whose
#: benchmark ran on a subset of the eval source — see `_align_to_scored_targets`.
#: An attribute rather than a return value because three commands call this and
#: only the ones that report a denominator need to look.
DROPPED_ATTR = "n_baseline_only_targets"


def _align_to_scored_targets(
    cols: dict[str, pd.Series], methods: list[str], *, baseline: str = SHORTEST_PING
) -> tuple[pd.Index | None, int]:
    """The population every column is read over: the targets the CBG arms scored.

    **The CBG arms must agree with each other; the baseline is aligned to them.**
    The asymmetry is not a convenience, it is where the two populations come
    from. A CBG column is the run's fold parquets — the targets the benchmark
    actually evaluated. The baseline column is `eval_source/*_eval_per_target.csv`,
    read straight off the eval source and never passed through the benchmark, so
    it covers every target the eval CSV describes.

    On a mesh run those coincide and this function changes nothing. On a
    **traffic-weighted** arm they cannot: the filter prunes flows, a target that
    loses every flow vanishes from the weighted CSV, and the eval source is the
    pre-filter mesh — so the baseline carries targets no CBG variant has an
    answer for. Those targets cannot enter a set-overlap figure (there is no
    membership to record for five of the six columns) and they cannot stay in
    the denominator either, so they are dropped and counted.

    Two disagreeing *CBG* arms are still an error. They read the same fold
    parquets of the same run, so a difference there is a broken run rather than
    a filtered one, and silently intersecting it would hide that.

    Returns the population index and how many baseline-only targets it excludes;
    `(None, 0)` when there is nothing to align — no CBG columns, or no baseline.
    """
    cbg = [m for m in methods if m != baseline]
    if not cbg:
        return None, 0

    first = set(cols[cbg[0]].index)
    disagree = [m for m in cbg[1:] if set(cols[m].index) != first]
    if disagree:
        counts = {m: int(len(cols[m])) for m in cbg}
        raise ValueError(
            f"CBG methods cover different target sets ({counts}); cannot form "
            f"set overlaps over a common denominator. These read the same run's "
            f"fold parquets, so they should not differ — re-run `classify`, or "
            f"pin a consistent set with --method"
        )
    if baseline not in cols:
        return None, 0

    scored = cols[cbg[0]].index
    missing = scored.difference(cols[baseline].index)
    if len(missing):
        # The other direction, and not alignable: a target the benchmark scored
        # but the eval source does not describe has no baseline answer to
        # compare against, and dropping it would shrink the CBG arms' own
        # denominator to hide an inconsistent run.
        raise ValueError(
            f"{len(missing)} target(s) scored by the CBG arms are absent from "
            f"the {baseline!r} baseline (e.g. {missing[:5].tolist()}); the "
            f"baseline is read from eval_source, so it should be a superset"
        )
    return scored, int(len(cols[baseline].index.difference(scored)))


def build_membership(
    cls_dir: Path, methods: list[str], *, top_n: int = 1
) -> pd.DataFrame:
    """Boolean matrix: one row per target, one column per method.

    True means the method placed that target in a seed ranked better than
    `top_n`, and that the pipeline actually solved it (fallbacks are failures).

    Rows are the targets the CBG arms scored. Where the Shortest-Ping baseline
    covers more than that — a traffic-weighted arm, whose benchmark ran on the
    traffic-carrying subset while its eval source spans the pre-filter mesh —
    the extra rows are dropped and counted in `frame.attrs[DROPPED_ATTR]`, so
    every column is read over one denominator. `_align_to_scored_targets` is
    where that rule and its limits live.
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

    scored, dropped = _align_to_scored_targets(cols, methods)
    membership = pd.DataFrame(cols)
    if scored is not None and dropped:
        # Masked rather than reindexed: `.loc[scored]` would reorder the rows to
        # the CBG arms' order, rewriting every existing membership CSV for a
        # change that drops nothing on a mesh run.
        membership = membership[membership.index.isin(set(scored))]
    if membership.isna().any().any():
        # Unreachable once aligned, kept as the backstop for the paths
        # `_align_to_scored_targets` returns `(None, 0)` on.
        counts = {m: int(membership[m].notna().sum()) for m in methods}
        raise ValueError(
            f"methods cover different target sets ({counts}); cannot form set "
            f"overlaps over a common denominator"
        )
    membership = membership.astype(bool)
    membership.attrs[DROPPED_ATTR] = dropped
    return membership


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
    # Summed explicitly: `concat` only carries `attrs` through when every input
    # agrees, and a pool of a mesh run with a weighted one is exactly the case
    # where they do not.
    pooled.attrs[DROPPED_ATTR] = sum(int(f.attrs.get(DROPPED_ATTR, 0)) for f in frames)
    if not pooled.index.is_unique:
        dupes = pooled.index[pooled.index.duplicated()].unique().tolist()
        raise ValueError(
            f"pooled target keys are not unique ({dupes[:5]}); one run_id appears "
            f"twice in the selection"
        )
    return pooled, pd.concat(origins)
