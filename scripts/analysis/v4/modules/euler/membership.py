"""The boolean matrix every Euler artifact is computed from.

One row per target, one column per method, True where that method's prediction
landed **inside the tolerance the figure is drawn at**. Everything downstream —
the intersection tables, the circle radii, the fitted overlaps — is a function
of this frame, so the rule about what counts as correct is enforced once, here.

## top-N *is* the ring

v3's `top_n` ranked class seeds by distance: "the truth's seed is among the N
nearest to the prediction". That is the rule v4 retired — a Voronoi partition
labels every point on Earth, so rank 1 can be 2,360 km away.

Here `top_n` selects a **rung of the containment ladder** instead, and the
mapping is fixed:

| `top_n` | rule | what it asks |
|---|---|---|
| 1 | `ring == 0` | in the truth's own cell |
| 2 | `ring <= 1` | in that cell or one of its 8 neighbours |
| 3 | `ring <= 2` | out to the second ring |

Cumulative, so a set at `top_n=2` contains the same targets it held at
`top_n=1` plus the ones one ring out. That nesting is what makes a sweep over
`top_n` readable as a tolerance dial: every circle can only grow.

A FALLBACK or ERROR row is **False, not absent** — the denominator is every
target the run evaluated, matching `classify`. A method that declines to answer
has not earned a smaller denominator than one that answers badly.

## Why no baseline alignment step

v3's equivalent carried `_align_to_scored_targets`, which cut the Shortest-Ping
baseline down to the CBG arms' targets because v3 built the baseline straight
from `eval_source/*_eval_per_target.csv` — every target the eval CSV describes,
which on a traffic-weighted arm is the pre-filter mesh and therefore *more*
targets than any CBG arm has.

v4 has no such step because it has no such gap: `classify.load_shortest_ping_frame`
builds the baseline over the run's **evaluated roster**, dropping eval-source
targets the run never evaluated and keeping a prediction-less row for evaluated
targets the eval source lacks. One denominator, upstream, for every method.
`guard_one_population` below asserts that rather than assuming it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths

#: `top_n` -> the largest ring that still counts as correct. `top_n=1` is
#: `ring == 0`, i.e. same cell.
MAX_TOP_N = H.MAX_RING + 1

#: How each rung reads in a title or a legend.
TOLERANCE_LABELS: dict[int, str] = {
    1: "in the cell",
    2: "within 1 ring",
    3: "within 2 rings",
}

#: Separator between the run id and the target id in a pooled index. `::`
#: cannot occur in either half — the operator runs' ids are `tg-<hex>` — so the
#: key stays reversible by `str.split(RUN_KEY_SEP, 1)`.
RUN_KEY_SEP = "::"

#: Baseline first, then calibration-free, then increasingly fitted. A display
#: order, so a figure's circles carry the same letters run to run.
PREFERRED_ORDER: tuple[str, ...] = (
    C.SHORTEST_PING,
    "million_scale_cbg",
    "vanilla_cbg",
    "octant_cbg_hull",
    "octant_cbg_spl",
    "octant_cbg",
    "spotter_cbg",
    "spotter_hybrid_cbg",
)


def validate_top_n(top_n: int) -> int:
    """`top_n` must name a rung of the ladder `classify` actually scored."""
    n = int(top_n)
    if not 1 <= n <= MAX_TOP_N:
        raise ValueError(
            f"top_n must be 1..{MAX_TOP_N} (ring 0..{H.MAX_RING}); got {top_n}. "
            f"Past the last ring the metric says 'unplaced' rather than a "
            f"distance, so there is no wider set to draw."
        )
    return n


def available_methods(cls_dir: Path) -> list[str]:
    """Method ids with a `*_cells.parquet` in `cls_dir`, in display order."""
    suffix = C.CELLS_PARQUET.format(method="")
    found = {
        p.name[: -len(suffix)] for p in Path(cls_dir).glob(f"*{suffix}")
    }
    ordered = [m for m in PREFERRED_ORDER if m in found]
    return ordered + sorted(found - set(ordered))


def correct_at(cells: pd.DataFrame, top_n: int) -> pd.Series:
    """Per-target: did this method land within `top_n - 1` rings, having answered?

    `ring == -1` covers both "answered, but further out than the metric grades"
    and "never answered", so the `>= 0` test is not redundant with the upper
    bound — without it an unplaced row would pass `ring <= 2` on the sign.
    """
    ring = cells["ring"]
    return (ring >= 0) & (ring <= validate_top_n(top_n) - 1) & C.solved_mask(cells)


def load_cells(
    run: RunPaths, method: str, nside: int, *, analysis_root: Path | None = None
) -> pd.DataFrame:
    """One run's per-target scored rows for one method at one rung."""
    path = run.cls_accuracy_dir(nside, root=analysis_root) / C.CELLS_PARQUET.format(
        method=method
    )
    if not path.exists():
        raise MissingArtifactError(
            f"{path} missing; run `classify --run-id {run.run_id} "
            f"--nside {nside}` first"
        )
    return pd.read_parquet(path, columns=["target_id", "status", "ring"])


def guard_one_population(cols: dict[str, pd.Series], where: str) -> None:
    """Every method must cover exactly the same targets.

    Asserted rather than intersected. `classify` builds all of them — baseline
    included — over the run's evaluated roster, so a difference here is a
    broken or half-rescored run, and silently scoring the intersection would
    report every share against a denominator nobody asked for. That is the
    failure v3 papered over with a drop-and-count, which then left its accuracy
    table comparing arms over two different populations.
    """
    names = list(cols)
    first = cols[names[0]].index
    odd = [m for m in names[1:] if not cols[m].index.equals(first)]
    if odd:
        sizes = {m: int(len(cols[m])) for m in names}
        raise ValueError(
            f"{where}: methods cover different target sets ({sizes}); cannot "
            f"form set overlaps over a common denominator. `classify` scores "
            f"every method over the run's evaluated roster, so this means the "
            f"rung was scored in two passes with different --method sets — "
            f"re-run `classify` for this run, or pin a consistent set with "
            f"--method."
        )


def build_membership(
    run: RunPaths,
    methods: list[str],
    nside: int,
    *,
    top_n: int = 1,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Boolean matrix for one run: one row per target, one column per method."""
    cols: dict[str, pd.Series] = {}
    for method in methods:
        cells = load_cells(run, method, nside, analysis_root=analysis_root)
        cols[method] = correct_at(cells, top_n).set_axis(cells["target_id"])
    guard_one_population(cols, run.run_id)
    return pd.DataFrame(cols).astype(bool)


def pooled_membership(
    runs: list[RunPaths],
    nside: int,
    *,
    top_n: int = 1,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """One membership matrix over several runs' targets, and the per-row run id.

    The runs are **stacked, not joined**: the three mesh runs have disjoint
    target sets, so a target belongs to exactly one run and pooling is a concat.
    Rows are re-keyed `<run_id>::<target_id>` anyway — disjointness is a
    property of these datasets rather than of the schema, and without the
    prefix a future run reusing an id would silently collide two targets into
    one row. `guard_disjoint_targets` still refuses the collision outright,
    because a re-keyed duplicate would sit in the denominator twice.

    Coverage is **strict**: a method absent from any input run is refused
    rather than pooled over the runs that carry it. Pooled over a subset it
    would be all-False across the missing run's targets, understating that
    method and inflating the "no method correct" region outside every circle —
    a silent wrong answer where a refusal is a fixable one.
    """
    if not runs:
        raise ValueError("pooling needs at least one run")

    scored = {
        r.run_id: set(
            available_methods(r.cls_accuracy_dir(nside, root=analysis_root))
        )
        for r in runs
    }
    if methods:
        wanted = list(dict.fromkeys(methods))
        absent = {
            rid: sorted(set(wanted) - found)
            for rid, found in scored.items()
            if set(wanted) - found
        }
        if absent:
            raise ValueError(
                f"--method names methods that are not scored at nside={nside}: "
                f"{absent}. A method missing from one run would count as wrong "
                f"on every one of its targets."
            )
        chosen = wanted
    else:
        common = set.intersection(*scored.values())
        partial = sorted(set.union(*scored.values()) - common)
        if partial:
            where = {
                m: sorted(r for r, ms in scored.items() if m in ms) for m in partial
            }
            raise ValueError(
                f"cannot pool: {partial} are not scored in every run ({where}). "
                f"Pooling them would count an unscored method as wrong on every "
                f"target of the run that lacks it. Pass --method to pick a "
                f"common subset."
            )
        chosen = [m for m in PREFERRED_ORDER if m in common] + sorted(
            common - set(PREFERRED_ORDER)
        )

    if len(chosen) < 2:
        raise ValueError(
            f"need >= 2 methods to show an overlap, got {chosen} at nside={nside}"
        )

    frames: list[pd.DataFrame] = []
    origins: list[pd.Series] = []
    seen: dict[str, set[str]] = {}
    for run in runs:
        one = build_membership(
            run, chosen, nside, top_n=top_n, analysis_root=analysis_root
        )
        seen[run.run_id] = set(one.index)
        one.index = [f"{run.run_id}{RUN_KEY_SEP}{t}" for t in one.index]
        frames.append(one)
        origins.append(pd.Series(run.run_id, index=one.index, name="run_id"))
    guard_disjoint_targets(seen)

    pooled = pd.concat(frames)
    if not pooled.index.is_unique:
        dupes = pooled.index[pooled.index.duplicated()].unique().tolist()
        raise ValueError(
            f"pooled target keys are not unique ({dupes[:5]}); one run id "
            f"appears twice in the selection"
        )
    return pooled, pd.concat(origins)


def guard_disjoint_targets(targets: dict[str, set[str]]) -> None:
    """No target id may appear in two runs.

    One shared id lands in the pooled denominator twice, which reweights that
    target and breaks the "every target counts once" claim the pooled shares
    rest on. The run-id prefix would hide it — two rows, two keys, one target —
    so the check is on the bare ids.
    """
    runs = sorted(targets)
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            shared = targets[a] & targets[b]
            if shared:
                raise ValueError(
                    f"{a} and {b} share {len(shared)} target ids (e.g. "
                    f"{sorted(shared)[:5]}); each would sit in the pooled "
                    f"denominator twice."
                )
