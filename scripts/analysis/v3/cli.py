"""Typer CLI for the v3 analysis layer — wiring only.

Every command body lives in `modules/`; this file imports those modules and
registers them, so adding a command means adding a module and one name to
`_COMMAND_MODULES`.

    python -m scripts.analysis.v3.cli build-answer-space --run-id as01-260728-260802
    python -m scripts.analysis.v3.cli classify --run-id as01-260728-260802
    python -m scripts.analysis.v3.cli plot-venn --run-id as01-260728-260802
"""

from __future__ import annotations

import typer

from scripts.analysis.v3.modules import answer_space, classify, map_answer_space, venn

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="v3 CBG analysis CLI — answer space, classification scoring, set overlap.",
)

#: Modules exposing `register(app)`. Order fixes `--help` listing order.
_COMMAND_MODULES = (answer_space, classify, venn, map_answer_space)

for _module in _COMMAND_MODULES:
    _module.register(app)


if __name__ == "__main__":
    app()
