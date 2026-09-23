"""Rename a combo's output directories, and the artifacts derived from them.

A combo id is the **whole cache key**: `<run>/<source>/<setup>/<fold>/<combo>/`.
Nothing in the path or the parquet records which code version produced it, so
re-running a combo whose implementation changed overwrites the old numbers in
place. Preserving them means renaming the directory first, which is what this
does.

## Why it is a command and not a shell loop

`outputs/` is gitignored, so the rename is an unversioned filesystem mutation
that no review and no CI can see. A test that reads a specific combo id then
passes or fails depending on whether a given machine happened to run the loop.
Making it a tested, idempotent command is the most that can be done about that:
re-running it is a no-op, a half-finished run can be finished, and the
`--dry-run` output is reviewable.

## The three things that have to move together

1. The **combo directories** themselves.
2. `combo_id` **inside each `run.json`**. `_summarize_combo` reads
   `meta["combo_id"]`, not the directory name, and does not check uniqueness --
   so a directory renamed without its metadata yields two rows per slice both
   claiming the old id, and any downstream `groupby("combo_id")` silently
   averages two different implementations together.
3. The **derived analysis artifacts**, which are keyed on the combo id string.
   Left behind, `<old>_cells.parquet` keeps holding the old implementation's
   predictions under a name that now means the new one, and the analysis layer
   globs by filename -- so a figure drawn before the artifacts are rebuilt
   reports old data under the new label with no signal at all. They are
   *renamed* rather than deleted: they are a valid scoring of the arm being
   preserved, and moving them keeps that evidence attributable.

## Matching is by exact suffix, never by prefix

`spotter_cbg_c80`, `spotter_cbg_c100` and `spotter_cbg_top` are **different
combos** from a coverage sweep, and all three are prefixed by `spotter_cbg`. So
a derived artifact is matched only when its name is exactly the old id or the
old id followed by one of `derived_suffixes`. The suffixes are passed in as
data rather than imported from the analysis layer, which the benchmark layer
may not depend on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

#: What the analysis layers append to a combo id. Passed as data, not imported.
DEFAULT_DERIVED_SUFFIXES: tuple[str, ...] = (
    "_cells.parquet",
    "_seed_distances.parquet",
)


class RenameRefused(Exception):
    """The rename would lose or mix data, so nothing was done."""


@dataclass
class RenamePlan:
    """What would move. Build with `plan_rename`, apply with `apply`."""

    combo_dirs: list[Path] = field(default_factory=list)
    derived: list[Path] = field(default_factory=list)
    already_done: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not self.combo_dirs and not self.derived


def _load_meta(combo_dir: Path) -> Optional[dict]:
    path = combo_dir / "run.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def plan_rename(
    outputs_root: Path,
    old: str,
    new: str,
    *,
    run_ids: Optional[Iterable[str]] = None,
    require_mtl: Optional[str] = None,
    derived_roots: Iterable[Path] = (),
    derived_suffixes: Iterable[str] = DEFAULT_DERIVED_SUFFIXES,
) -> RenamePlan:
    """Decide what moves, without moving anything.

    `require_mtl` is the safety gate, and it reads `run.json` rather than the
    path: several runs can share a combo id while having run *different*
    compositions under it, and only the ones matching should be renamed.
    Non-matching directories are recorded in `skipped` with the reason, so the
    caller can see they were considered and passed over.

    Derived artifacts are only considered under the runs whose combo directory
    moved -- see the comment at `renamed_runs`.

    Raises `RenameRefused` when a directory would be clobbered, rather than
    merging two combos' folds.
    """
    if old == new:
        raise RenameRefused(f"old and new combo ids are both {old!r}")
    outputs_root = Path(outputs_root)
    wanted = set(run_ids) if run_ids else None
    plan = RenamePlan()

    for src in sorted(outputs_root.glob(f"*/*/*/fold_*/{old}")):
        if not src.is_dir():
            continue
        run_id = src.relative_to(outputs_root).parts[0]
        if wanted is not None and run_id not in wanted:
            continue
        meta = _load_meta(src)
        if meta is None:
            plan.skipped.append((src, "no readable run.json"))
            continue
        if require_mtl is not None and meta.get("mtl") != require_mtl:
            plan.skipped.append(
                (src, f"mtl is {meta.get('mtl')!r}, not {require_mtl!r}")
            )
            continue
        dst = src.with_name(new)
        if dst.exists():
            raise RenameRefused(
                f"{dst} already exists; renaming {src} onto it would merge two "
                f"combos' folds. Remove or rename the destination first."
            )
        plan.combo_dirs.append(src)

    # Idempotency: a destination already in place with no source left is a
    # completed rename, not a problem.
    for done in sorted(outputs_root.glob(f"*/*/*/fold_*/{new}")):
        if not done.with_name(old).exists():
            plan.already_done.append(done)

    # Derived artifacts are renamed **only for the runs whose combo directory
    # actually moved**. Scoping this to the renamed runs is not tidiness: the
    # same combo id can name different compositions in different runs, and
    # `require_mtl` filters on exactly that. On this repo, `spotter_cbg` is the
    # density arm in the three `-mesh` runs and the Octant-geometry hybrid in
    # six others -- so an unscoped sweep would relabel the hybrid's scorings as
    # the density arm's, which is worse than leaving them alone.
    renamed_runs = {
        src.relative_to(outputs_root).parts[0] for src in plan.combo_dirs
    }

    suffixes = tuple(derived_suffixes)
    for root in derived_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            try:
                run_of = path.relative_to(root).parts[0]
            except (ValueError, IndexError):
                continue
            if run_of not in renamed_runs:
                continue
            name = path.name
            if name == old:
                target = new
            else:
                hit = next((s for s in suffixes if name == old + s), None)
                if hit is None:
                    continue
                target = new + hit
            dst = path.with_name(target)
            if dst.exists():
                raise RenameRefused(
                    f"{dst} already exists; refusing to overwrite a derived "
                    f"artifact. Delete it and re-run the analysis stage instead."
                )
            plan.derived.append(path)

    return plan


def apply(plan: RenamePlan, old: str, new: str) -> RenamePlan:
    """Move everything in `plan` and patch each `run.json`'s `combo_id`."""
    for src in plan.combo_dirs:
        dst = src.with_name(new)
        src.rename(dst)
        meta_path = dst / "run.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("combo_id") == old:
            meta["combo_id"] = new
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for path in plan.derived:
        name = path.name
        target = new if name == old else new + name[len(old):]
        path.rename(path.with_name(target))
    return plan
