"""Where cross-dataset figures land, and what they refuse to pool.

Ported from v4. The directory is `_cross/classify/<datasets>[@<arm>]/`, keyed by
the dataset set and by the arm (the run-id remainder every run shares), so a
two-run comparison cannot overwrite a three-run one and the mesh and
traffic-weighted arms of one dataset set sit side by side.

Both guards are strict: a method must be scored in every run, and no TG id may
appear in two runs, or the pooled denominator stops being one population.
"""

from __future__ import annotations

from pathlib import Path

from scripts.analysis.v5.modules.paths import DEFAULT_ANALYSIS_ROOT

#: Where cross-dataset figures land -- keyed by the dataset set, so a two-run
#: comparison cannot overwrite a three-run one.
CROSS_KIND = "classify"

#: Each guard's default closing clause: the outcome bars' own wording, kept
#: verbatim so that module's messages are byte-identical before and after the
#: guards moved here. The two read differently because they attach to different
#: sentences -- `guard_common_methods` ends "...or <clause>", the other stands
#: alone -- so they are two constants rather than one shared string.
COMPARE_REMEDY_COMMON = "--layout compare to keep each dataset on its own panel."
COMPARE_REMEDY_DISJOINT = (
    "Use --layout compare, which keeps each dataset on its own panel."
)


def dataset_slug(run_ids: list[str]) -> str:
    """`as01-...-mesh, as02-...` -> `as01+as02+as03`.

    Keyed on the datasets rather than on a count, so the directory names the
    comparison it holds.
    """
    heads = sorted({r.split("-")[0] for r in run_ids})
    return "+".join(heads)


def short_dataset(run_id: str) -> str:
    """`as01-260728-260802-mesh` -> `as01`."""
    return run_id.split("-")[0]


def arm(run_ids: list[str]) -> str | None:
    """The run-id remainder every run shares, or None if they differ.

    `as01-260728-260802-mesh` + `as02-260728-260802-mesh` -> `260728-260802-mesh`.

    `dataset_slug` keeps only the head of each run id, so the mesh arm and the
    traffic-weighted arm of the same three datasets collapse to one name --
    `as01+as02+as03` either way. Pooling both would then write the weighted
    figures over the mesh ones, and the comparison between the arms is the whole
    reason both are run.

    Derived rather than declared, and with no vocabulary of arm names: anything
    the run ids share is the arm, whether that is `-mesh`, `-weighted`,
    `-mesh-reciprocal` or a date range alone. Mixed remainders yield None, which
    is the heads-only name -- a set spanning two arms is not an arm.
    """
    tails = {r.split("-", 1)[1] if "-" in r else "" for r in run_ids}
    if len(tails) != 1:
        return None
    return tails.pop() or None


def cross_dir(
    run_ids: list[str], *, analysis_root: Path | None = None, kind: str = CROSS_KIND
) -> Path:
    """`_cross/<kind>/<datasets>[@<arm>]/`, created. `kind` defaults to `classify`.

    The arm is a directory-name concern only. `dataset_slug` also supplies the
    `dataset` column of every CSV twin and the label in the pooled figures'
    subtitles, where a date range would be noise.
    """
    name = dataset_slug(run_ids)
    shared = arm(run_ids)
    if shared is not None:
        name = f"{name}@{shared}"
    out = (analysis_root or DEFAULT_ANALYSIS_ROOT) / "_cross" / kind / name
    out.mkdir(parents=True, exist_ok=True)
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
