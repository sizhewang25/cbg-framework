"""Shared plumbing for the commands that pool several runs into one artifact set.

Most v3 commands are run-scoped: they read one run and write back into that
run's own `outputs/analysis/v3/<run_id>/` subtree, so `RunPaths.analysis_dir`
gives them a home for free. `plot-pareto` and the cross-run half of `plot-venn`
are not — their numbers are a function of a *set* of runs and belong to none of
them. This module holds the four things they need: the directory those artifacts
live in, the name of the dataset set, and the two guards on which sets may
legally be pooled at all — one on the run *family*, one on the target sets.

Extracted from `pareto.py`, which owned all of it while it was the only
cross-run command. The one signature change is `cross_dir`, which took the
artifact kind as a hard-coded literal and now takes it as a required keyword —
see that function for why it cannot be defaulted.

`guard_disjoint_targets` arrived later, from `figure_error_scatter.py`, when
`table-headline` grew a pooled row and became its second caller. A table command
must not import matplotlib, and that module pulls it in at import time, so the
guard moved here rather than being imported across — the same reason
`short_label` lives in `labels.py` and `_C_AXIS` in `palette.py`. It took a
plotting frame and a plotting remedy on the way in and takes neither now; see
its docstring.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from collections.abc import Iterable, Mapping

from scripts.analysis.v3.modules.paths import DEFAULT_ANALYSIS_ROOT, RunPaths

#: Cross-dataset artifacts have no per-run home (`RunPaths.analysis_dir` is
#: run-scoped), so they get a sibling directory. The leading underscore means it
#: can never collide with a `run_id`.
CROSS_DIRNAME = "_cross"


def short_dataset(run_id: str) -> str:
    """`as01-260728-260802` -> `as01`; anything else unchanged.

    The date range is identical across the runs being compared (it is what
    makes them comparable), so printing it three times in a legend costs width
    and carries no information. Only a trailing all-numeric tail is stripped,
    so `as7018_us_test01` survives intact.
    """
    parts = run_id.split("-")
    if len(parts) > 1 and all(p.isdigit() for p in parts[1:]):
        return parts[0]
    return run_id


def dataset_set_slug(run_ids) -> str:
    """A stable directory name for one *set* of datasets.

    The dataset set is a parameter of every number in these artifacts, so it
    has to appear in the path — the cost channel, grid, top-N and fit policy are
    all in the filename, but without this a run over `as7018_us_test01` writes
    the same `pareto_runtime.healpix-128.top1.csv` as a run over as01+as02+as03
    and silently replaces it. Long sets are truncated and hashed so the name
    stays a usable directory while still being unique.
    """
    short = sorted(short_dataset(r) for r in run_ids)
    slug = "+".join(short)
    if len(slug) <= 60:
        return slug
    digest = hashlib.sha1("+".join(sorted(run_ids)).encode()).hexdigest()[:8]
    return f"{len(short)}sets-{digest}"


def cross_dir(analysis_root: Path | None = None, run_ids=None, *, kind: str) -> Path:
    """`<analysis_root>/_cross/<kind>/<dataset-set-slug>/`, created on demand.

    `kind` names the artifact family — `"cost-accuracy"` for `plot-pareto`,
    `"venn-diagram"` for `plot-venn`. It is a **required keyword with no
    default**: the two families write different files over the same dataset
    set, and a default would let a new caller inherit whichever kind happened
    to be written first and quietly deposit its output in the other command's
    directory. The slug is the leaf rather than the parent so that one kind's
    dataset sets sit together, matching the `cost-accuracy/` tree already on
    disk.
    """
    base = (analysis_root or DEFAULT_ANALYSIS_ROOT) / CROSS_DIRNAME / kind
    if run_ids is not None:
        base = base / dataset_set_slug(run_ids)
    base.mkdir(parents=True, exist_ok=True)
    return base


def guard_one_setup(runs: dict[str, RunPaths], *, allow_mixed: bool) -> None:
    """Refuse to pool `anchors_to_probes` with `probes_to_anchors`.

    SCHEMA.md §7: the two run families swap the VP and target roles, so their
    costs and accuracies describe different experiments. Pooling them onto one
    frontier would compare a 134-VP fleet against a 53-VP one as though the
    difference were the variant's.
    """
    setups = sorted({r.setup for r in runs.values()})
    if len(setups) > 1 and not allow_mixed:
        import typer

        by_setup = {
            s: sorted(rid for rid, r in runs.items() if r.setup == s) for s in setups
        }
        raise typer.BadParameter(
            f"selected runs span {len(setups)} setups: "
            + "; ".join(f"{s} = {v}" for s, v in by_setup.items())
            + ". These swap the VP/target roles (SCHEMA.md §7), so one artifact over "
            "both would not be a like-for-like comparison. Pass --allow-mixed-setups "
            "to override, or select runs from one setup."
        )


def guard_disjoint_targets(
    ids_by_run: Mapping[str, Iterable], *, remedy: str
) -> dict[tuple[str, str], int]:
    """Refuse to pool runs that share a `target_id`.

    A shared target would be counted once per run in the pooled denominator, so
    the pooled number would stop being a rate over a population — it would be a
    rate over a multiset. The three operator runs are disjoint by construction
    (each draws its targets from its own peer ASN, and 399 + 412 + 458 = 1,269
    on both the union and the sum), but that is a property of the data rather
    than of the pipeline, and a re-run with an overlapping target list would
    otherwise pool silently.

    Keyed by run rather than taking a frame, because the two callers hold their
    target ids in different shapes — one has a long points frame with a
    `dataset` column, the other has one parquet per run — and the only thing the
    guard needs from either is a set per run.

    `remedy` is the caller's sentence about what to do instead, appended to the
    refusal. It is required rather than defaulted: the useful advice is
    caller-specific (`--layout compare` for the band figures, "run the command
    once per dataset" for a table) and a generic default would be advice nobody
    can act on. Returns the pairwise overlap sizes, empty when disjoint.
    """
    sets = {run_id: set(ids) for run_id, ids in ids_by_run.items()}
    order = list(sets)
    overlaps: dict[tuple[str, str], int] = {}
    for i, left in enumerate(order):
        for right in order[i + 1 :]:
            shared = sets[left] & sets[right]
            if shared:
                overlaps[(left, right)] = len(shared)
    if overlaps:
        import typer

        worst = max(overlaps.items(), key=lambda kv: kv[1])
        raise typer.BadParameter(
            f"the selected runs share targets ({worst[1]} ids in common between "
            f"{worst[0][0]} and {worst[0][1]}, {len(overlaps)} overlapping pair(s)), "
            "so pooling them would count those targets once per run in the "
            f"denominator. {remedy}"
        )
    return overlaps
