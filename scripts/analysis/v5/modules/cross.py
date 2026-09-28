"""Where cross-dataset figures land, and what they refuse to pool.

Ported from v4, then split along a seam v4 does not have. Naming a pooled
artifact is two jobs with opposite requirements, and the ported code conflated
them:

* **Identity** -- the directory a pool writes to, and the namespace its sites
  live in. Needs uniqueness and nothing else.
* **Display** -- a panel title, a CSV `dataset` column, a LaTeX caption. Wants
  brevity.

Brevity was the only reason to parse, and parsing is what broke. v4's
`short_dataset` read the dataset off the run id by splitting on the first
hyphen, which assumes the shape `<dataset>-<arm>`; `pro-as01-mesh` and
`pro-as02-mesh` both read `pro`. So the two jobs now have separate answers:

* Display comes from `labels.dataset_label` -- declared in the config, under
  `analysis.common.dataset_label`, falling back to the run id.
* Identity comes from a hash of the run ids (`cross_dir`) and from the run id
  itself (`sites.site_key`). Neither can collapse, whatever a run is called.

The directory is `_cross/<kind>/<n>-runs-<hash>/`, with a `runs.json` beside
the artifacts because a hash does not read as anything. It replaced
`<datasets>@<arm>`, where `arm` was the run-id remainder every run shared:
that existed to stop the mesh and traffic-weighted arms of one dataset set
overwriting each other under a heads-only name, and the hash does it without
parsing.

Three guards, all strict: a method must be scored in every run, no TG id may
appear in two runs, and no two runs may display the same label. The first two
keep the pooled denominator one population; the third keeps a `groupby` on the
display name honest, since a declared label is free text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.analysis.v5.modules.labels import dataset_label
from scripts.analysis.v5.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
)

#: Where cross-dataset figures land -- keyed by the run set, so a two-run
#: comparison cannot overwrite a three-run one.
CROSS_KIND = "classify"

#: Hex digits of the run-set digest kept in a directory name. Six is ~17M
#: values against the tens of run sets this tree will ever hold, and short
#: enough that the name still scans.
SLUG_HASH_CHARS = 6

#: Written into every cross directory: a hashed name does not say what it
#: holds, and a reader should not have to open a manifest to find out.
RUNS_JSON = "runs.json"

#: Each guard's default closing clause: the outcome bars' own wording, kept
#: verbatim so that module's messages are byte-identical before and after the
#: guards moved here. The two read differently because they attach to different
#: sentences -- `guard_common_methods` ends "...or <clause>", the other stands
#: alone -- so they are two constants rather than one shared string.
COMPARE_REMEDY_COMMON = "--layout compare to keep each dataset on its own panel."
COMPARE_REMEDY_DISJOINT = (
    "Use --layout compare, which keeps each dataset on its own panel."
)


def labels_for(run_ids: list[str], *, outputs_root: Path | str | None = None
               ) -> dict[str, str]:
    """`{run_id: display label}`, in the order given.

    One place resolves a label, so a figure's panel title, its CSV column and
    its manifest cannot disagree about what a run is called.
    """
    root = outputs_root or DEFAULT_OUTPUTS_ROOT
    return {r: dataset_label(r, root) for r in run_ids}


def short_dataset(run_id: str, *, outputs_root: Path | str | None = None) -> str:
    """The run's declared dataset label; the run id if it declares none.

    Was `run_id.split("-")[0]`. See `labels` for why that had to go and what
    the fallback costs: a run with no declared label gets its full run id here,
    which is correct but long on a panel title.
    """
    return dataset_label(run_id, outputs_root or DEFAULT_OUTPUTS_ROOT)


def dataset_slug(run_ids: list[str], *, outputs_root: Path | str | None = None
                 ) -> str:
    """`as01+as02+as03` -- the pooled **display** name, not a directory name.

    Deliberately still readable, and deliberately no longer the directory:
    this string is printed on pooled figures and written into the LaTeX the
    paper includes, where a hash would be worse than useless. `cross_dir` is
    what needs to be collision-proof, and it hashes instead.

    Sorted and de-duplicated, so the name does not depend on `--run-id` order.
    Two runs sharing a label collapse to one entry here -- `guard_distinct_labels`
    is what refuses to pool them in the first place.
    """
    return "+".join(sorted(set(labels_for(run_ids, outputs_root=outputs_root).values())))


def runs_hash(run_ids: list[str], *, chars: int = SLUG_HASH_CHARS) -> str:
    """A short digest of the run set, order-independent and stable.

    Over sorted run ids rather than labels: the digest is identity, and a
    declared label can repeat or be edited after the fact.
    """
    joined = "\n".join(sorted(set(run_ids)))
    return hashlib.sha256(joined.encode()).hexdigest()[:chars]


def cross_name(run_ids: list[str]) -> str:
    """`3-runs-8f2a1c`. The count is there so the name says something."""
    return f"{len(set(run_ids))}-runs-{runs_hash(run_ids)}"


def cross_dir(
    run_ids: list[str], *, analysis_root: Path | None = None, kind: str = CROSS_KIND
) -> Path:
    """`_cross/<kind>/<n>-runs-<hash>/`, created, with a `runs.json` in it.

    `kind` defaults to `classify`. The name is content-addressed, so a two-run
    pool cannot overwrite a three-run one and two arms of the same dataset set
    cannot overwrite each other -- neither of which the old
    `<datasets>[@<arm>]` name could guarantee without parsing run ids.

    `runs.json` is rewritten on every call. It is derived entirely from
    `run_ids`, so rewriting it repairs a directory whose file was lost rather
    than churning content.
    """
    out = (analysis_root or DEFAULT_ANALYSIS_ROOT) / "_cross" / kind / cross_name(run_ids)
    out.mkdir(parents=True, exist_ok=True)
    (out / RUNS_JSON).write_text(
        json.dumps({"run_ids": sorted(set(run_ids)), "kind": kind}, indent=2)
    )
    return out


def guard_common_methods(
    scored: dict[str, set[str]], *, remedy: str = COMPARE_REMEDY_COMMON
) -> list[str]:
    """Every run must score the same methods. Returns them, sorted.

    Strict on purpose. The alternative -- v3's `pool_method_counts`, which sums
    over the runs that *carry* a method -- leaves bars in one panel resting on
    different denominators, so the panel's `n=` is true of some bars and not
    others and a reader has no way to tell which. Here a method absent from any
    input run is refused, and the caller narrows the set with `--method`.
    """
    common = set.intersection(*scored.values()) if scored else set()
    partial = sorted(set.union(*scored.values()) - common) if scored else []
    if partial:
        where = {
            m: sorted(r for r, ms in scored.items() if m in ms) for m in partial
        }
        raise ValueError(
            f"cannot pool: {partial} are not scored in every run ({where}). "
            f"Pooling them would put their bars on a different denominator "
            f"from the rest. Pass --method to pick a common subset, or "
            f"{remedy}"
        )
    return sorted(common)


def guard_disjoint_tgs(
    tgs: dict[str, set[str]], *, remedy: str = COMPARE_REMEDY_DISJOINT
) -> None:
    """No TG id may appear in two runs.

    One shared id lands in the pooled denominator twice, which silently
    reweights that TG and breaks the "every TG counts once" claim the
    micro-average rests on.
    """
    runs = sorted(tgs)
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            shared = tgs[a] & tgs[b]
            if shared:
                sample = sorted(shared)[:5]
                raise ValueError(
                    f"{a} and {b} share {len(shared)} TG ids (e.g. "
                    f"{sample}); each would sit in the pooled denominator "
                    f"twice. {remedy}"
                )


def guard_distinct_labels(
    labels: dict[str, str], *, remedy: str = COMPARE_REMEDY_DISJOINT
) -> dict[str, str]:
    """No two runs may display the same label. Returns them unchanged.

    The other two guards protect the pooled denominator. This one protects
    every `groupby("dataset")` downstream -- the per-dataset medians, the panel
    split, the outcome-bar selection. A label is free text declared in a
    config, so two runs really can claim `as01`, and nothing about the pooled
    frame would look wrong afterwards: the groups would simply merge and report
    one dataset where there were two.

    It is also the guard that makes the run-id fallback safe. Two runs with no
    declared label fall back to their run ids, which are distinct by
    construction, so they pass here rather than silently pooling.
    """
    seen: dict[str, list[str]] = {}
    for run_id, label in labels.items():
        seen.setdefault(label, []).append(run_id)
    clashes = {label: runs for label, runs in seen.items() if len(runs) > 1}
    if clashes:
        detail = "; ".join(
            f"{label!r} is declared by {sorted(runs)}" for label, runs in sorted(clashes.items())
        )
        raise ValueError(
            f"cannot pool: {detail}. Every per-dataset number is grouped on "
            f"this label, so the runs would merge into one row and report a "
            f"dataset that does not exist. Give each run its own "
            f"`analysis.common.dataset_label`, or {remedy}"
        )
    return labels
