"""Locating the canonical edge CSV, and reducing it to one RTT per pair.

The load-bearing test is `TestMeshSupersetIsRefused`. Every other step of the
resolution order fails loudly when it is wrong -- a missing file raises. That
one fails *quietly*: the path exists and parses, it is simply the wrong
population, and a viewer drawn from it would show VP observations the weighted
run never made.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.analysis.v4.modules import edges as E
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths


BASENAME = "ds-a"
CANONICAL = pd.DataFrame(
    {
        "vp_id": ["vp-1", "vp-1", "vp-2", "vp-2"],
        "vp_lat": [40.0, 40.0, 41.0, 41.0],
        "vp_lon": [-74.0, -74.0, -75.0, -75.0],
        "target_id": ["tg-1", "tg-1", "tg-1", "tg-2"],
        "target_lat": [40.5, 40.5, 40.5, 42.0],
        "target_lon": [-74.5, -74.5, -74.5, -76.0],
        # tg-1/vp-1 measured twice: 9.0 is the minimum, 21.0 the decoy.
        "rtt_ms": [21.0, 9.0, 13.0, 4.0],
    }
)


def _run(tmp_path, *, stats: dict | None = None, space: dict | None = None) -> RunPaths:
    """A benchmark tree with just the two provenance files resolution reads."""
    run = RunPaths(run_id="r", root=tmp_path, source="generic_csv", setup="s")
    run.setup_dir.mkdir(parents=True, exist_ok=True)
    run.eval_source_dir.mkdir(parents=True, exist_ok=True)
    if stats is not None:
        (run.eval_source_dir / f"{BASENAME}_eval_stats.json").write_text(
            json.dumps(stats)
        )
    if space is not None:
        run.target_space_json.write_text(json.dumps(space))
    return run


def _csv(tmp_path, name: str = "edges.csv"):
    path = tmp_path / name
    CANONICAL.to_csv(path, index=False)
    return path


class TestResolutionOrder:
    def test_eval_stats_wins(self, tmp_path):
        csv = _csv(tmp_path)
        other = _csv(tmp_path, "other.csv")
        run = _run(
            tmp_path,
            stats={"csv": str(csv)},
            space={"csv": str(other)},
        )
        assert E.resolve_source_csv(run) == csv

    def test_target_space_is_the_pre_benchmark_fallback(self, tmp_path):
        """A materialized target space with no combo run yet has no eval_source."""
        csv = _csv(tmp_path)
        run = _run(tmp_path, space={"csv": str(csv)})
        assert E.resolve_source_csv(run) == csv

    def test_a_recorded_path_that_does_not_exist_falls_through(self, tmp_path):
        """Recorded, not guaranteed: a moved dataset must not dead-end here."""
        csv = _csv(tmp_path)
        run = _run(
            tmp_path,
            stats={"csv": "datasets/gone.csv"},
            space={"csv": str(csv)},
        )
        assert E.resolve_source_csv(run) == csv

    def test_override_wins_over_everything(self, tmp_path):
        csv = _csv(tmp_path)
        override = _csv(tmp_path, "override.csv")
        run = _run(tmp_path, stats={"csv": str(csv)})
        assert E.resolve_source_csv(run, override=override) == override

    def test_override_must_exist(self, tmp_path):
        run = _run(tmp_path, stats={"csv": str(_csv(tmp_path))})
        with pytest.raises(MissingArtifactError, match="does not exist"):
            E.resolve_source_csv(run, override=tmp_path / "nope.csv")

    def test_nothing_to_resolve_names_the_remedy(self, tmp_path):
        run = _run(tmp_path, stats={})
        with pytest.raises(MissingArtifactError, match="--source-csv"):
            E.resolve_source_csv(run)

    def test_no_eval_source_at_all_is_still_a_clean_refusal(self, tmp_path):
        """`eval_basename` raises when the sidecar is absent; it must not escape
        as a bare glob failure with no remedy in it."""
        run = _run(tmp_path)
        with pytest.raises(MissingArtifactError, match="--source-csv"):
            E.resolve_source_csv(run)


class TestMeshSupersetIsRefused:
    """`MeshSupersetError` is a subclass of `MissingArtifactError` so a caller
    that degrades on absence can still re-raise this one; `map_mtl._load_edges`
    is that caller."""

    def test_it_is_a_missing_artifact_error(self):
        from scripts.analysis.v4.modules.paths import MissingArtifactError

        assert issubclass(E.MeshSupersetError, MissingArtifactError)

    def test_it_raises_rather_than_returning_the_mesh(self, tmp_path):
        csv = _csv(tmp_path)
        run = _run(tmp_path, space={"csv": str(csv), "csv_is_mesh_superset": True})
        with pytest.raises(MissingArtifactError, match="MESH SUPERSET"):
            E.resolve_source_csv(run)

    def test_the_message_names_both_ways_out(self, tmp_path):
        csv = _csv(tmp_path)
        run = _run(tmp_path, space={"csv": str(csv), "csv_is_mesh_superset": True})
        with pytest.raises(MissingArtifactError) as exc:
            E.resolve_source_csv(run)
        assert "derive_traffic_weighted_cbg_data.smk" in str(exc.value)
        assert "--source-csv" in str(exc.value)

    def test_an_override_still_accepts_the_mesh_deliberately(self, tmp_path):
        """The refusal is about silence, not about the mesh being unusable."""
        csv = _csv(tmp_path)
        run = _run(tmp_path, space={"csv": str(csv), "csv_is_mesh_superset": True})
        assert E.resolve_source_csv(run, override=csv) == csv

    def test_it_outranks_a_recorded_eval_stats_csv(self, tmp_path):
        """The case that makes the refusal worth having, and the one an earlier
        draft got backwards.

        `materialize-target-space --with-eval-source` scores the very CSV it
        just flagged a superset, so on a real on-the-fly weighted arm
        `eval_stats.json` names the mesh too. Checking the flag after the
        recorded path would therefore return the mesh and never fire -- on
        exactly the arm the check exists for. The flag is a fact about the arm,
        so it is checked before any path is resolved."""
        csv = _csv(tmp_path)
        run = _run(
            tmp_path,
            stats={"csv": str(csv)},
            space={"csv": str(csv), "csv_is_mesh_superset": True},
        )
        with pytest.raises(E.MeshSupersetError):
            E.resolve_source_csv(run)

    def test_it_fires_even_when_the_recorded_csv_is_gone(self, tmp_path):
        """No path needs to resolve for the arm to be the wrong population."""
        run = _run(
            tmp_path,
            stats={"csv": "datasets/gone.csv"},
            space={"csv": "datasets/also-gone.csv", "csv_is_mesh_superset": True},
        )
        with pytest.raises(E.MeshSupersetError):
            E.resolve_source_csv(run)

    def test_a_corrupt_target_space_does_not_suppress_resolution(self, tmp_path):
        """Unreadable is not the same as flagged: fall through rather than
        refuse, or a malformed sidecar would block every run."""
        csv = _csv(tmp_path)
        run = _run(tmp_path, stats={"csv": str(csv)})
        run.target_space_json.write_text("{not json")
        assert E.resolve_source_csv(run) == csv


class TestLoadMinRtt:
    def test_one_row_per_pair_at_the_minimum(self, tmp_path):
        run = _run(tmp_path, stats={"csv": str(_csv(tmp_path))})
        out = E.load_min_rtt(run)
        assert list(out.columns) == list(E.MIN_RTT_COLUMNS)
        assert len(out) == 3
        got = out.set_index(["target_id", "vp_id"])["rtt_ms"]
        assert got[("tg-1", "vp-1")] == 9.0   # not the 21.0 decoy
        assert got[("tg-1", "vp-2")] == 13.0
        assert got[("tg-2", "vp-2")] == 4.0

    def test_non_positive_rtts_are_dropped_upstream(self, tmp_path):
        """Inherited from `load_canonical_csv`, which mirrors what the
        benchmark's own source did at materialize time -- so these are the
        observations the run was actually built on."""
        frame = CANONICAL.copy()
        frame.loc[frame.index[1], "rtt_ms"] = 0.0
        path = tmp_path / "zeroed.csv"
        frame.to_csv(path, index=False)
        run = _run(tmp_path, stats={"csv": str(path)})
        out = E.load_min_rtt(run)
        # The 9.0 row is gone, so the pair falls back to its other observation.
        assert out.set_index(["target_id", "vp_id"])["rtt_ms"][("tg-1", "vp-1")] == 21.0

    def test_source_csv_bypasses_resolution(self, tmp_path):
        run = _run(tmp_path)          # no provenance files at all
        out = E.load_min_rtt(run, source_csv=_csv(tmp_path))
        assert len(out) == 3


class TestOnRealRuns:
    def test_as01_resolves_and_reduces(self):
        from scripts.analysis.v4.modules.paths import resolve_run

        try:
            run = resolve_run("as01-260728-260802-mesh")
            path = E.resolve_source_csv(run)
        except MissingArtifactError as exc:
            pytest.skip(f"benchmark output not present: {exc}")
        assert path.exists()
        out = E.load_min_rtt(run, source_csv=path)
        assert not out.duplicated(["target_id", "vp_id"]).any()
        assert (out["rtt_ms"] > 0).all()
