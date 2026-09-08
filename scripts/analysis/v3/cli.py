"""Typer CLI for the v3 analysis layer — wiring only.

Every command body lives in `modules/`; this file imports those modules and
registers them, so adding a command means adding a module and one name to
`_COMMAND_MODULES`.

    python -m scripts.analysis.v3.cli build-answer-space --run-id as01-260728-260802
    python -m scripts.analysis.v3.cli build-bipartite-graph --run-id as01-260728-260802
    python -m scripts.analysis.v3.cli classify --run-id as01-260728-260802
    python -m scripts.analysis.v3.cli build-proximity --run-id as01-260728-260802
    python -m scripts.analysis.v3.cli plot-venn --run-id as01-260728-260802

A unified config can supply the parameters instead, one sub-block per command
(see `modules/config.py`). `--config` belongs to the *group*, so it goes before
the command name:

    python -m scripts.analysis.v3.cli --config configs/as7018_us_test01.yaml classify
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from scripts.analysis.v3.modules import (
    accuracy_table,
    answer_space,
    bipartite,
    breakdown,
    classify,
    config as config_mod,
    confusion,
    figure_error_cdf,
    figure_error_scatter,
    figure_outcome_bars,
    figure_proximity_inflation,
    headline_table,
    map_answer_space,
    map_bipartite,
    map_mtl,
    pareto,
    proximity,
    venn,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="v3 CBG analysis CLI — answer space, classification scoring, set overlap.",
)

#: Modules exposing `register(app)`. Order fixes `--help` listing order.
_COMMAND_MODULES = (
    answer_space,
    bipartite,
    classify,
    proximity,
    breakdown,
    confusion,
    accuracy_table,
    headline_table,
    figure_outcome_bars,
    figure_proximity_inflation,
    figure_error_cdf,
    figure_error_scatter,
    venn,
    map_answer_space,
    map_bipartite,
    map_mtl,
    pareto,
)

for _module in _COMMAND_MODULES:
    _module.register(app)


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(
        None,
        "--config",
        help="Unified run config (configs/<run_id>.yaml). Supplies each command's "
             "parameters from its `analysis.<command>` sub-block. Explicit flags "
             "still win. Must come before the command name.",
    ),
) -> None:
    """Load a unified config, if given, as click's per-command defaults.

    Setting `ctx.default_map` is the whole mechanism: click applies it beneath
    anything passed on the command line, so no command signature has to know a
    config exists. `--all-runs` is read from `sys.argv` because the group
    callback runs before the subcommand's own arguments are parsed — see
    `config.default_map`.
    """
    if config is not None:
        ctx.default_map = config_mod.default_map(
            config_mod.load(config),
            app,
            all_runs_on_cli="--all-runs" in sys.argv,
            grid_on_cli=config_mod.grid_on_argv(sys.argv),
            command=ctx.invoked_subcommand,
        )


if __name__ == "__main__":
    app()
