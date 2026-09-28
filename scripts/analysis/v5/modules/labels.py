"""The dataset a run belongs to, declared rather than parsed.

`cross.short_dataset` used to read the dataset off the run id by splitting on
the first hyphen -- `as01-260728-260802-mesh` -> `as01`. That assumes the shape
`<dataset>-<arm>`, which any prefix breaks: `pro-as01-mesh`, `pro-as02-mesh`
and `pro-as03-mesh` all read `pro`. The collapse was not only cosmetic. The
pooled octant-finetuning frame namespaced its sites on that name, so three
datasets' sites merged into one and the site-clustered bootstrap resampled
units that do not exist.

So the name is **declared**, in the config, under `analysis.common`:

    analysis:
      common:
        dataset_label: as01

## Why this module reads a config at all

`paths.combo_ids` states v5's rule: read the output tree, not a config, so
there is one source of truth. This is the documented exception, kept in one
module so it stays one exception. The label is a *display* name -- nothing in
the tree records it, and nothing can derive it, which is the whole point.

The run reaches its config through `target_space.json`'s `config` key, written
by the benchmark alongside the `csv` key that `edges.resolve_source_csv`
already reads the same way.

## Never raises

Every break in the chain falls back to the run id, which is unique by
construction and so is always a correct-if-verbose label. The breaks are real,
not hypothetical: five runs have no `target_space.json` at all
(`as0{1,2,3}-260728-260802`, `as01-260728-260802-mesh-heapfix`,
`as7018_us_test01`), and `as01-materialization-test`'s manifest names a config
that has since been deleted. A missing label is not an error -- only the
`pro-*` configs declare one.

Identity is a separate concern and never comes from here: sites key on
`run_id` (`sites.site_key`), and the pooled cross directory keys on a hash of
the run ids. A declared label is free text, so two runs may well declare the
same one -- `cross.guard_distinct_labels` is what refuses to pool those.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from scripts.analysis.v5.modules.paths import (
    DEFAULT_OUTPUTS_ROOT,
    REPO_ROOT,
    MissingArtifactError,
    resolve_run,
)

#: Where the label is declared, as a path through the config mapping.
LABEL_PATH = ("analysis", "common", "dataset_label")


def _under_repo(value: str) -> Path:
    """Manifest paths are repo-root-relative (the benchmark's `_repo_relative`)."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def config_path(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> Path | None:
    """The config that produced `run_id`, or None if it cannot be reached.

    Note the run id is **not** the config stem in general: the `-mesh-grafted`
    runs record the non-grafted config they were produced against, which is why
    this reads the manifest rather than guessing `configs/<run_id>.yaml`.
    """
    try:
        run = resolve_run(run_id, root)
    except (MissingArtifactError, OSError):
        return None
    if not run.target_space_json.exists():
        return None
    try:
        recorded = json.loads(run.target_space_json.read_text()).get("config")
    except (json.JSONDecodeError, OSError):
        return None
    if not recorded:
        return None
    p = _under_repo(recorded)
    return p if p.exists() else None


@lru_cache(maxsize=None)
def dataset_label(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> str:
    """The run's declared `analysis.common.dataset_label`, else the run id.

    Memoized: a pooled figure asks for the same handful of run ids once per
    panel, per CSV row group and again for the manifest, and each miss is a
    directory scan plus a YAML parse.
    """
    path = config_path(run_id, root)
    if path is None:
        return run_id
    try:
        cfg = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return run_id

    node = cfg
    for key in LABEL_PATH:
        if not isinstance(node, dict):
            return run_id
        node = node.get(key)
    # A non-scalar label would be carried into a filename and a CSV column, so
    # it is refused the same way an absent one is.
    if node is None or isinstance(node, (dict, list)):
        return run_id
    return str(node)
