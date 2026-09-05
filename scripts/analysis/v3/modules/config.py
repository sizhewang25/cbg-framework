"""Unified per-run config: one YAML supplying every v3 command's parameters.

    python -m scripts.analysis.v3.cli --config configs/as7018_us_test01.yaml classify

The file has a reserved `benchmark:` section and an `analysis:` section, and
`analysis:` holds one **sub-block per CLI command** plus a `common:` block:

    run_id: as7018_us_test01
    benchmark: {}
    analysis:
      common:            {grid: h3, resolution: [4]}
      build-answer-space: {}
      classify:          {topn: "1,3"}
      plot-venn:         {top_n: 1}
      plot-answer-space: {us_only: true}
      plot-pareto:       {cost: runtime}

**This is a defaults mechanism, not a second code path.** `default_map` returns
click's own `ctx.default_map`, so an explicit flag always beats the config and a
config value always beats the typer default — the ordering is click's, not
hand-rolled here, and no command signature changes to support it.

Everything is validated against the *live* CLI: `_command_params` introspects the
registered click commands, so block names, key names, `run_id` arity and which
params are paths are all read off the real signatures and cannot drift from them.
That check is the point of this module — click silently ignores a `default_map`
key it does not recognize, so a typo'd `topN:` would otherwise be a no-op rather
than an error.

Relative paths resolve against the **repo root**, matching this layer's own
`DEFAULT_OUTPUTS_ROOT` / `DEFAULT_ANALYSIS_ROOT` (see `paths.py`). The wider repo
resolves config paths against three different bases — cwd, the config file's own
directory, and repo root — so v3 picks one and states it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import typer
import yaml

from scripts.analysis.v3.modules.grid import DEFAULT_GRID
from scripts.analysis.v3.modules.paths import REPO_ROOT

#: The sub-block whose keys apply to every command that declares them.
COMMON_BLOCK = "common"

#: Sections + identity keys allowed at the top level.
#:
#: `run_id` is the only identity key here. `source` and `setup` are deliberately
#: absent: `paths.discover_runs` reads them off the output tree, so declaring
#: them would create a second source of truth that can disagree with the tree.
#: `slices` / `combos` likewise have no v3 equivalent — `fold_ids` and
#: `combo_ids` are discovered too.
TOP_LEVEL_KEYS = frozenset({"run_id", "benchmark", "analysis"})


def load(path: str | Path) -> dict[str, Any]:
    """Parse a unified config and check its top level.

    Per-command validation needs the live CLI, so it happens in `default_map`.
    """
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise typer.BadParameter(f"{p} is not valid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{p} must be a YAML mapping, got {type(raw).__name__}")

    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        raise typer.BadParameter(
            f"{p}: unknown top-level key(s) {unknown}. "
            f"Allowed: {sorted(TOP_LEVEL_KEYS)}. Per-command settings belong under "
            f"`analysis:`, in a sub-block named after the command."
        )
    return raw


def _command_params(app: typer.Typer) -> dict[str, dict[str, click.Parameter]]:
    """`{command_name: {param_name: click_param}}` for the registered commands."""
    group = typer.main.get_command(app)
    return {
        name: {p.name: p for p in cmd.params}
        for name, cmd in group.commands.items()  # type: ignore[attr-defined]
    }


def _resolve(value: Any, *, multiple: bool) -> Any:
    """Make path values absolute against `REPO_ROOT`, leaving absolutes alone."""
    if multiple:
        return [_resolve(v, multiple=False) for v in value]
    p = Path(str(value))
    return str(p if p.is_absolute() else REPO_ROOT / p)


#: Sentinel for "this command cannot take the config's run_id".
_UNFIT = object()


def grid_on_argv(argv: list[str]) -> str | None:
    """The `--grid` value on the command line, if any.

    Needed for the same reason as `all_runs_on_cli`: the group callback runs
    before the subcommand's arguments are parsed, so argv is the only way to
    see them. Used to keep a config-pinned `resolution` from being paired with
    a grid it was not written for -- see `default_map`.
    """
    for i, tok in enumerate(argv):
        if tok == "--grid" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--grid="):
            return tok.split("=", 1)[1]
    return None


def _run_id_for(value: Any, param: click.Parameter) -> Any:
    """Fit a top-level `run_id` to one command's arity, or return `_UNFIT`.

    `plot-pareto` and `plot-venn` take `--run-id` repeatably (both pool
    datasets); the rest score one run at a time. So a cross-run config fits those
    two and does not fit the others — reported as a named error only for the
    command actually invoked, since a config that lists three runs is perfectly
    valid for the command it was written for.
    """
    ids = list(value) if isinstance(value, (list, tuple)) else [value]
    if param.multiple:
        return [str(i) for i in ids]
    return str(ids[0]) if len(ids) == 1 else _UNFIT


def default_map(
    cfg: dict[str, Any],
    app: typer.Typer,
    *,
    all_runs_on_cli: bool = False,
    grid_on_cli: str | None = None,
    command: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build `ctx.default_map` from a loaded config, keyed by command name.

    Every block is key-checked regardless of `command`, so a typo anywhere in
    the file fails fast. `command` only scopes the one error that is
    command-specific: whether the config's `run_id` fits that command's arity.

    `all_runs_on_cli` drops the config's `run_id`. Every command enforces
    `--run-id` XOR `--all-runs`, and the group callback runs before the
    subcommand's arguments are parsed (`ctx.args` is still empty there), so
    without this an explicit `--all-runs` would collide with a config-supplied
    `run_id` and be rejected — and the documented pipeline uses `--all-runs`
    throughout.

    `grid_on_cli` drops the config's `resolution` when it names a *different*
    grid than the config does. A resolution is meaningful only against its own
    grid — h3 res 4 is 45 km, HEALPix nside 4 is 1630 km — which is why the CLI
    defaults the resolution per grid rather than globally. Overriding just
    `--grid` would otherwise silently inherit the other grid's number and build
    an answer space nobody asked for.
    """
    params = _command_params(app)
    analysis = cfg.get("analysis") or {}
    if not isinstance(analysis, dict):
        raise typer.BadParameter("`analysis:` must be a mapping of command blocks")

    common = analysis.get(COMMON_BLOCK) or {}
    if not isinstance(common, dict):
        raise typer.BadParameter(f"`analysis.{COMMON_BLOCK}:` must be a mapping")

    blocks = {k: v for k, v in analysis.items() if k != COMMON_BLOCK}
    for name in sorted(blocks):
        if name not in params:
            raise typer.BadParameter(
                f"`analysis.{name}` is not a command. Known commands: "
                f"{sorted(params)}"
            )

    # A `common:` key applies only where the param exists (`top_n` is on two of
    # five commands), so one matching nothing at all is a typo, not a no-op.
    for key in sorted(common):
        if not any(key in p for p in params.values()):
            raise typer.BadParameter(
                f"`analysis.{COMMON_BLOCK}.{key}` matches no command's parameters"
            )

    out: dict[str, dict[str, Any]] = {}
    for name, cmd_params in params.items():
        merged = {k: v for k, v in common.items() if k in cmd_params}

        block = blocks.get(name) or {}
        if not isinstance(block, dict):
            raise typer.BadParameter(f"`analysis.{name}:` must be a mapping")
        for key, value in block.items():
            if key not in cmd_params:
                raise typer.BadParameter(
                    f"unknown key `{key}` in `analysis.{name}`. Valid keys: "
                    f"{sorted(cmd_params)}"
                )
            merged[key] = value

        if "run_id" not in merged and cfg.get("run_id") is not None:
            merged["run_id"] = cfg["run_id"]
        # `--run-id` XOR `--all-runs`, so a config asking for every run cannot
        # also carry a run_id -- from either the CLI flag or its own block.
        if all_runs_on_cli or merged.get("all_runs"):
            merged.pop("run_id", None)
        elif "run_id" in merged:
            fitted = _run_id_for(merged["run_id"], cmd_params["run_id"])
            if fitted is _UNFIT:
                if name == command:
                    n = len(merged["run_id"])
                    raise typer.BadParameter(
                        f"`run_id` lists {n} runs, but `{name}` scores one run at "
                        f"a time. Pass `--run-id <one>` or `--all-runs`, or use a "
                        f"single-run config."
                    )
                merged.pop("run_id")
            else:
                merged["run_id"] = fitted

        # A config that omits `grid` was still written against the CLI's default
        # one, so that is what an override is compared against.
        if grid_on_cli is not None and grid_on_cli != merged.get("grid", DEFAULT_GRID):
            merged.pop("resolution", None)

        for key, value in merged.items():
            param = cmd_params[key]
            if isinstance(param.type, click.types.Path):
                merged[key] = _resolve(value, multiple=param.multiple)

        if merged:
            out[name] = merged
    return out
