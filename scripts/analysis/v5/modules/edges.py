"""The run's canonical `(vp_id, target_id, rtt_ms)` CSV, located. Ported from v4.

Only the VP-proximity figure opens the measurement table; every other v5 module
reads the benchmark's scored outputs. Locating the CSV is four steps and one
refusal. The refusal is the point: on a traffic-weighted arm the recorded CSV
is the **mesh** superset, and silently accepting it would measure VP distances
over edges the weighted run never had.

The CSV speaks benchmark names (`target_id`, `target_lat`, ...). Callers rename
to `tg_*` at load time, the one place those names appear in v5.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.v5.modules.paths import (
    REPO_ROOT,
    MissingArtifactError,
    RunPaths,
)


class MeshSupersetError(MissingArtifactError):
    """The only CSV on record describes a strictly larger edge set than this arm.

    A subclass, not a flag: absence is degradable, this is not. The file
    parses and every RTT in it is real, but they are measurements the weighted
    run never made.
    """


def eval_basename(run: RunPaths) -> str:
    """The eval sidecars' shared stem, e.g. `as01-...mainland.sanitized`.

    Taken off a real sidecar rather than built from `run_id`, which does not
    track the CSV stem.
    """
    return run.eval_file("eval_stats.json").name[: -len("_eval_stats.json")]


def _under_repo(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_source_csv(run: RunPaths, override: Path | None = None) -> Path:
    """Path to the run's canonical edge CSV.

    Resolution order:

    1. `eval_source/<basename>_eval_stats.json`'s `csv` key, recorded by the
       benchmark relative to the repo root.
    2. `target_space.json`'s `csv`, for a target space no combo has run on.
    3. A glob of `datasets/**/<basename>.csv`, for a run whose `eval_*`
       predates the `csv` key.

    Refuses a `csv_is_mesh_superset` target space, and checks that **first**:
    `eval_stats.json` names the mesh on exactly that arm, so resolving in
    recorded-path order would hand the mesh back and never reach the check.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise MissingArtifactError(f"--source-csv {p} does not exist")
        return p

    space = {}
    if run.target_space_json.exists():
        try:
            space = json.loads(run.target_space_json.read_text())
        except json.JSONDecodeError:
            space = {}
    if space.get("csv_is_mesh_superset"):
        raise MeshSupersetError(
            f"{run.run_id}: target_space.json records {space.get('csv')} as a "
            f"MESH SUPERSET of this arm's edge set (the traffic filter runs on "
            f"the fly, so no file holds the pruned flows). Derive a weighted CSV "
            f"via scripts/processing/source/derive_traffic_weighted_cbg_data.smk "
            f"and point the config's `weighted_csv_path` at it, or pass "
            f"--source-csv explicitly to accept the mesh graph."
        )

    recorded = None
    try:
        recorded = json.loads(run.eval_file("eval_stats.json").read_text()).get("csv")
    except MissingArtifactError:
        pass
    if recorded:
        p = _under_repo(recorded)
        if p.exists():
            return p

    if space.get("csv"):
        p = _under_repo(space["csv"])
        if p.exists():
            return p

    try:
        stem = eval_basename(run)
    except MissingArtifactError:
        stem = None
    if stem:
        hits = sorted((REPO_ROOT / "datasets").rglob(f"{stem}.csv"))
        if hits:
            return hits[0]

    raise MissingArtifactError(
        f"cannot locate the canonical edge CSV for {run.run_id}: "
        f"{stem or '<unknown basename>'}.csv is not under datasets/ and "
        f"eval_stats.json records {recorded!r}. Pass --source-csv <path>."
    )
