"""The run's canonical `(vp_id, target_id, rtt_ms)` CSV, located and reduced.

v4's `bipartite.py` is RTT-free end to end: it co-quantizes the target and VP
rosters onto the grid and never opens the measurement table. The case viewer
does need it -- it draws every VP's observed RTT against the target and the
inflation that implies -- so the resolution lives here rather than being bolted
onto a module none of whose own paths would call it.

Locating the CSV is four steps and one refusal, which is why it is a named
function with its own tests rather than a few lines inside the viewer. The
refusal is the point: on a traffic-weighted arm the recorded CSV is the **mesh**
superset, and silently accepting it would show the viewer edges the weighted run
never measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.v4.modules.paths import (
    MissingArtifactError,
    REPO_ROOT,
    RunPaths,
)
from scripts.libs.canonical.schema import load_canonical_csv

#: The columns the viewer reads off the reduced table.
MIN_RTT_COLUMNS = ("target_id", "vp_id", "rtt_ms")


class MeshSupersetError(MissingArtifactError):
    """The only CSV on record describes a strictly larger edge set than this arm.

    A subclass, not a flag, because callers need to treat it differently from
    an absent CSV. Absence is degradable -- a map can drop its observation
    layer and still be a map. This is not: the file parses, the paths resolve,
    and every RTT in it is real, but they are measurements the weighted run
    never made. Degrading here would draw a graph the run does not have.
    """


def eval_basename(run: RunPaths) -> str:
    """The eval sidecars' shared stem, e.g. `as01-...mainland.sanitized`.

    Taken off a real sidecar rather than built from `run_id`, which does not
    track the CSV stem -- `as01-260728-260802-mesh` scores
    `as01-20260728-20260802.mainland.sanitized`.
    """
    return run.eval_file("eval_stats.json").name[: -len("_eval_stats.json")]


def _under_repo(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_source_csv(run: RunPaths, override: Path | None = None) -> Path:
    """Path to the run's canonical edge CSV.

    Resolution order, and why each step exists:

    1. `eval_source/<basename>_eval_stats.json`'s `csv` key, recorded by the
       benchmark relative to the repo root. Read rather than reconstructed,
       because no config carries the path -- the operator runs' canonical CSVs
       were reconstructed from run outputs and their configs say `benchmark: {}`.
    2. `target_space.json`'s `csv`, written by `materialize-target-space`. The
       pre-benchmark case: a target space exists but no combo has run, so there
       is no `eval_source/` yet.
    3. A glob of `datasets/**/<basename>.csv`, for a run whose `eval_*` predates
       the `csv` key.

    Refuses a `csv_is_mesh_superset` target space: see the module docstring.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise MissingArtifactError(f"--source-csv {p} does not exist")
        return p

    # The refusal is checked FIRST, before any path is resolved, because it is
    # a fact about the ARM and not about which file happened to be recorded
    # where. `materialize-target-space --with-eval-source` scores the very CSV
    # it just flagged a superset (benchmark/v2/cli.py), so `eval_stats.json`
    # names the mesh too -- and resolving in recorded-path order would hand
    # back the mesh and never reach the check. That is the whole failure this
    # function exists to prevent, so it cannot sit behind a step that shadows
    # it on the one arm where it matters.
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


def load_min_rtt(run: RunPaths, *, source_csv: Path | None = None) -> pd.DataFrame:
    """`(target_id, vp_id, rtt_ms)`, one row per pair at its minimum RTT.

    Deduplicated to the minimum because a dataset can measure a pair repeatedly
    and the viewer shows one number per VP. Minimum rather than mean: it is what
    every LTD in the framework consumes, so the RTT drawn beside a VP is the RTT
    the constraint was built from.

    Read through `load_canonical_csv`, which owns the schema, lowercases the
    header and drops NaN rows and `rtt_ms <= 0` exactly as the benchmark's own
    source does -- so these are the observations the benchmark ran on.
    """
    path = Path(source_csv) if source_csv is not None else resolve_source_csv(run)
    df = load_canonical_csv(path)
    return (
        df.groupby(["target_id", "vp_id"], as_index=False)["rtt_ms"]
        .min()
        .reset_index(drop=True)
    )
