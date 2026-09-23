"""Where cross-dataset figures land, and what they refuse to pool.

Three v4 figures now write into one directory keyed by the dataset set --
`plot-outcome-bars`, `plot-euler` and `plot-error-cdf` -- and two of them merge
several runs into one population. The directory rule and the two refusals that
make merging honest live here rather than in whichever figure needed them
first. That is the threshold `methods.py` was split out at, and by the time
`figure_error_cdf` arrived `dataset_slug`/`cross_dir` already had two verbatim
copies and `guard_disjoint_targets` two near-copies.

## Why the remedy is a parameter

Both guards end by naming a way out, and the way out is the *caller's*
vocabulary rather than the guard's: the outcome bars offer `--layout compare`,
the error CDF `--layout per-run`. A shared guard that hardcoded one figure's
flag would print a remedy the reader cannot take. So the closing sentence is
passed in, and the default is the outcome bars' original wording, unchanged to
the character -- the messages that module raised before this split are the
messages it raises after it.

## The directory is keyed by dataset AND arm

`dataset_slug` keeps only each run id's head, which is what makes
`as01+as02+as03` readable -- and what made the mesh and traffic-weighted arms of
those same three datasets share one directory. `cross_dir` therefore appends the
remainder the runs share (`arm`), so the two arms sit side by side instead of
one overwriting the other. Only the path carries it; the label the figures print
stays `dataset_slug`.

## One copy still outstanding

`euler.membership.guard_disjoint_targets` is **not** folded in here. It says
the same thing in its own words, carries no remedy clause, and sits directly
under a `pytest.raises(match=...)` in `test_figure_euler.py`. Unifying it is a
separate change with its own test edit rather than a rider on this one; it is
named here so the remaining duplicate is recorded instead of forgotten.
"""

from __future__ import annotations

from pathlib import Path

from scripts.analysis.v4.modules.paths import DEFAULT_ANALYSIS_ROOT

#: Where cross-dataset figures land -- keyed by the dataset set, so a two-run
#: comparison cannot overwrite a three-run one.
CROSS_KIND = "cls-accuracy"

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


def cross_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/cls-accuracy/<datasets>[@<arm>]/`, created.

    The arm is a directory-name concern only. `dataset_slug` also supplies the
    `dataset` column of every CSV twin and the label in the pooled figures'
    subtitles, where a date range would be noise.
    """
    name = dataset_slug(run_ids)
    shared = arm(run_ids)
    if shared is not None:
        name = f"{name}@{shared}"
    out = (analysis_root or DEFAULT_ANALYSIS_ROOT) / "_cross" / CROSS_KIND / name
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


def guard_disjoint_targets(
    targets: dict[str, set[str]], *, remedy: str = COMPARE_REMEDY_DISJOINT
) -> None:
    """No target id may appear in two runs.

    One shared id lands in the pooled denominator twice, which silently
    reweights that target and breaks the "every target counts once" claim the
    micro-average rests on. v3 guards the same thing in
    `cross.guard_disjoint_targets` for the same reason.
    """
    runs = sorted(targets)
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            shared = targets[a] & targets[b]
            if shared:
                sample = sorted(shared)[:5]
                raise ValueError(
                    f"{a} and {b} share {len(shared)} target ids (e.g. "
                    f"{sample}); each would sit in the pooled denominator "
                    f"twice. {remedy}"
                )
