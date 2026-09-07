"""Shared plumbing for the commands that pool several runs into one artifact set.

Most v3 commands are run-scoped: they read one run and write back into that
run's own `outputs/analysis/v3/<run_id>/` subtree, so `RunPaths.analysis_dir`
gives them a home for free. `plot-pareto` and the cross-run half of `plot-venn`
are not — their numbers are a function of a *set* of runs and belong to none of
them. This module holds the three things both need: the directory those
artifacts live in, the name of the dataset set, and the guard on which sets may
legally be pooled at all.

Extracted from `pareto.py`, which owned all of it while it was the only
cross-run command. The one signature change is `cross_dir`, which took the
artifact kind as a hard-coded literal and now takes it as a required keyword —
see that function for why it cannot be defaulted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
