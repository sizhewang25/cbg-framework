"""The unified config is a defaults mechanism, and a validated one.

Two properties are load-bearing and easy to lose. First, precedence: an explicit
flag must beat the config, and the config must beat the CLI default, with no
third code path. Second, validation: click *silently ignores* a `default_map`
key it does not recognize, so without an explicit check a typo'd key would be a
no-op rather than an error — the classic YAML-config failure mode.

These are also the first tests to exercise v3's typer layer at all.
"""

from __future__ import annotations

import glob
from pathlib import Path

import click
import pytest
import typer
import yaml
from typer.testing import CliRunner

from scripts.analysis.v3.cli import app
from scripts.analysis.v3.modules import config as cfg_mod

runner = CliRunner()


def _write(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "unified.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def _dm(tmp_path: Path, cfg: dict, **kw) -> dict:
    return cfg_mod.default_map(cfg_mod.load(_write(tmp_path, cfg)), app, **kw)


# ---- the callback is inert without --config ---------------------------------

def test_no_config_leaves_every_default_untouched():
    """`--config` is opt-in: without it nothing consults a config at all."""
    res = runner.invoke(app, ["classify", "--help"])
    assert res.exit_code == 0
    # The group option exists, but the command's own defaults are unchanged.
    assert "--config" in runner.invoke(app, ["--help"]).output


# ---- validation: the unknown-key trap ---------------------------------------

def test_unknown_param_key_is_rejected(tmp_path):
    """A typo'd key would otherwise be silently dropped by click."""
    with pytest.raises(typer.BadParameter) as exc:
        _dm(tmp_path, {"analysis": {"classify": {"topN": "1,3"}}})
    msg = str(exc.value)
    assert "topN" in msg and "analysis.classify" in msg
    assert "topn" in msg  # the valid keys are listed, so the fix is obvious


def test_unknown_command_block_is_rejected(tmp_path):
    with pytest.raises(typer.BadParameter) as exc:
        _dm(tmp_path, {"analysis": {"plot_venn": {"top_n": 2}}})
    # snake_case is the likely mistake; the message names the real commands.
    assert "plot_venn" in str(exc.value) and "plot-venn" in str(exc.value)


def test_unknown_top_level_key_is_rejected(tmp_path):
    """`combos:` at the top level is a v2 habit, and would do nothing here."""
    with pytest.raises(typer.BadParameter) as exc:
        cfg_mod.load(_write(tmp_path, {"run_id": "r", "combos": ["vanilla_cbg"]}))
    assert "combos" in str(exc.value)


def test_a_config_may_be_empty(tmp_path):
    assert _dm(tmp_path, {}) == {}
    (tmp_path / "empty.yaml").write_text("")
    assert cfg_mod.load(tmp_path / "empty.yaml") == {}


# ---- precedence -------------------------------------------------------------

def test_config_supplies_defaults_and_the_cli_overrides_them(tmp_path):
    """End-to-end precedence through click, not through our own merge."""
    path = _write(
        tmp_path,
        {"run_id": "cfg-run", "analysis": {"common": {"grid": "healpix"}}},
    )
    seen: dict = {}

    probe = typer.Typer()

    @probe.callback()
    def _main(ctx: typer.Context) -> None:
        ctx.default_map = cfg_mod.default_map(cfg_mod.load(path), app)

    # Mirror `classify`'s signature closely enough to observe the merge.
    @probe.command("classify")
    def _classify(
        run_id: str = typer.Option(None),
        grid: str = typer.Option("h3", "--grid"),
    ) -> None:
        seen.update(run_id=run_id, grid=grid)

    @probe.command("other")
    def _other() -> None:  # keeps `probe` a group
        pass

    assert runner.invoke(probe, ["classify"]).exit_code == 0
    assert seen == {"run_id": "cfg-run", "grid": "healpix"}

    assert runner.invoke(probe, ["classify", "--grid", "h3"]).exit_code == 0
    assert seen == {"run_id": "cfg-run", "grid": "h3"}  # flag wins, rest survives


# ---- the common block -------------------------------------------------------

def test_common_applies_only_where_the_param_exists(tmp_path):
    """`top_n` is a param of some commands, `grid` of all of them."""
    dm = _dm(tmp_path, {"analysis": {"common": {"grid": "healpix", "top_n": 3}}})
    assert all(b["grid"] == "healpix" for b in dm.values())
    assert {name for name, b in dm.items() if "top_n" in b} == {
        "plot-venn",
        "plot-pareto",
        "plot-outcome-bars",
    }
    assert "top_n" not in dm["classify"]


def test_a_common_key_matching_no_command_is_rejected(tmp_path):
    with pytest.raises(typer.BadParameter) as exc:
        _dm(tmp_path, {"analysis": {"common": {"radius_km": 50}}})
    assert "radius_km" in str(exc.value)


def test_a_command_block_overrides_common(tmp_path):
    dm = _dm(
        tmp_path,
        {"analysis": {"common": {"grid": "h3"}, "classify": {"grid": "healpix"}}},
    )
    assert dm["classify"]["grid"] == "healpix"
    assert dm["plot-venn"]["grid"] == "h3"


# ---- run_id arity -----------------------------------------------------------

def test_run_id_arity_follows_each_command(tmp_path):
    """`plot-pareto` pools datasets, so its `--run-id` is repeatable."""
    dm = _dm(tmp_path, {"run_id": "solo"})
    assert dm["classify"]["run_id"] == "solo"
    assert dm["plot-pareto"]["run_id"] == ["solo"]


def test_a_multi_run_config_fits_pareto_and_is_named_for_the_others(tmp_path):
    cfg = {"run_id": ["a", "b", "c"]}
    dm = _dm(tmp_path, cfg, command="plot-pareto")
    assert dm["plot-pareto"]["run_id"] == ["a", "b", "c"]
    # Dropped, not injected, for the commands that were not invoked -- and with
    # nothing else to set, they get no block at all rather than an empty one.
    assert "classify" not in dm
    # ... but named when one of them IS the command being run.
    with pytest.raises(typer.BadParameter) as exc:
        _dm(tmp_path, cfg, command="classify")
    assert "3 runs" in str(exc.value) and "classify" in str(exc.value)


def test_all_runs_suppresses_the_configs_run_id(tmp_path):
    """Every command enforces `--run-id` XOR `--all-runs`.

    The group callback cannot see the subcommand's parsed args, so `--all-runs`
    is detected from argv; without this the two would collide and be rejected.
    """
    cfg = {"run_id": "solo"}
    assert "classify" not in _dm(tmp_path, cfg, all_runs_on_cli=True)
    # Same when the config itself asks for every run.
    dm = _dm(tmp_path, {**cfg, "analysis": {"common": {"all_runs": True}}})
    assert "run_id" not in dm["classify"] and dm["classify"]["all_runs"] is True


# ---- paths ------------------------------------------------------------------

def test_relative_paths_resolve_against_the_repo_root(tmp_path, monkeypatch):
    """Not the cwd — run from elsewhere so the distinction actually bites."""
    monkeypatch.chdir(tmp_path)
    dm = _dm(tmp_path, {"analysis": {"common": {"analysis_root": "outputs/x"}}})
    assert dm["classify"]["analysis_root"] == str(cfg_mod.REPO_ROOT / "outputs" / "x")
    assert str(tmp_path) not in dm["classify"]["analysis_root"]


def test_absolute_paths_are_left_alone(tmp_path):
    dm = _dm(tmp_path, {"analysis": {"common": {"analysis_root": str(tmp_path)}}})
    assert dm["classify"]["analysis_root"] == str(tmp_path)


def test_repeatable_path_params_resolve_elementwise(tmp_path):
    dm = _dm(tmp_path, {"analysis": {"plot-pareto": {"accuracy_csv": ["a.csv", "b.csv"]}}})
    assert dm["plot-pareto"]["accuracy_csv"] == [
        str(cfg_mod.REPO_ROOT / "a.csv"),
        str(cfg_mod.REPO_ROOT / "b.csv"),
    ]


def test_path_params_are_detected_from_the_live_signatures():
    """Which params are paths is read off click, so it cannot drift.

    Pinned here because the resolution above is silent when it misses one: a
    relative root would then be taken against the cwd by click instead.
    """
    group = typer.main.get_command(app)
    found = {
        name: sorted(
            p.name for p in cmd.params if isinstance(p.type, click.types.Path)
        )
        for name, cmd in group.commands.items()
    }
    assert found["classify"] == ["analysis_root", "answer_space", "outputs_root"]
    assert found["plot-pareto"] == [
        "accuracy_csv",
        "analysis_root",
        "out_dir",
        "outputs_root",
    ]
    assert found["build-answer-space"] == ["analysis_root", "outputs_root"]
    assert found["build-bipartite-graph"] == [
        "analysis_root",
        "answer_space",
        "outputs_root",
        "source_csv",
    ]
    assert found["plot-bipartite-graph"] == [
        "analysis_root",
        "bipartite_dir",
        "outputs_root",
    ]
    assert found["plot-mtl-map"] == ["analysis_root", "outputs_root"]


# ---- the shipped configs ----------------------------------------------------

@pytest.mark.parametrize(
    "path", sorted(glob.glob(str(cfg_mod.REPO_ROOT / "configs" / "*.yaml")))
)
def test_every_shipped_config_validates(path):
    """Guards the checked-in files themselves against key drift in the CLI."""
    dm = cfg_mod.default_map(cfg_mod.load(path), app)
    assert dm, f"{path} set nothing"


# ---- grid / resolution stay paired ------------------------------------------

def test_grid_on_argv_reads_both_spellings():
    assert cfg_mod.grid_on_argv(["--config", "c", "classify", "--grid", "healpix"]) == "healpix"
    assert cfg_mod.grid_on_argv(["classify", "--grid=healpix"]) == "healpix"
    assert cfg_mod.grid_on_argv(["classify", "--grid"]) is None  # dangling
    assert cfg_mod.grid_on_argv(["classify"]) is None


def test_a_cli_grid_override_drops_the_configs_resolution(tmp_path):
    """h3 res 4 is 45 km; HEALPix nside 4 is 1630 km.

    A resolution only means something against its own grid, so overriding just
    `--grid` must fall back to the new grid's default rather than inherit the
    old grid's number. Without this the config silently builds a 1630 km answer
    space.
    """
    cfg = {"analysis": {"common": {"grid": "h3", "resolution": [4]}}}
    dm = _dm(tmp_path, cfg, grid_on_cli="healpix")
    assert "resolution" not in dm["build-answer-space"]
    assert dm["build-answer-space"]["grid"] == "h3"  # click's own override wins


def test_the_same_grid_on_the_cli_keeps_the_configs_resolution(tmp_path):
    """`--grid h3` on an h3 config is a no-op, so a pinned res 3 must survive."""
    cfg = {"analysis": {"common": {"grid": "h3", "resolution": [3]}}}
    dm = _dm(tmp_path, cfg, grid_on_cli="h3")
    assert dm["build-answer-space"]["resolution"] == [3]


def test_a_per_command_grid_is_respected_when_dropping_resolution(tmp_path):
    cfg = {
        "analysis": {
            "common": {"resolution": [4]},
            "classify": {"grid": "healpix"},
            "plot-venn": {"grid": "h3"},
        }
    }
    dm = _dm(tmp_path, cfg, grid_on_cli="healpix")
    assert dm["classify"]["resolution"] == [4]      # config already said healpix
    assert "resolution" not in dm["plot-venn"]      # config said h3


def test_a_config_omitting_grid_is_compared_against_the_cli_default(tmp_path):
    """`--grid h3` on a grid-less config is a no-op, so res 3 must survive."""
    cfg = {"analysis": {"common": {"resolution": [3]}}}
    assert _dm(tmp_path, cfg, grid_on_cli="h3")["build-answer-space"]["resolution"] == [3]
    # Dropped, and with nothing left the block is omitted rather than emptied.
    assert "build-answer-space" not in _dm(tmp_path, cfg, grid_on_cli="healpix")
