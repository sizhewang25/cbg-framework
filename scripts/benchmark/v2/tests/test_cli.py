"""CLI tests — use Typer's CliRunner to invoke commands in-process."""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from typer.testing import CliRunner

from scripts.benchmark.v2.cli import app


# Canonical-schema synth CSV: vp_* = anchor side (acting as VP),
# target_* = probe side (the entity being geolocated).
_SYNTH_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,vp_asn,vp_country,target_id,target_lat,target_lon,target_asn,target_country,rtt_ms
    1.1.1.1,33.0,-84.0,20473,US,1001,33.5,-84.5,7922,US,5.0
    1.1.1.1,33.0,-84.0,20473,US,1002,32.5,-83.5,7922,US,6.0
    1.1.1.1,33.0,-84.0,20473,US,1003,33.5,-83.5,7922,US,7.0
    1.1.1.1,33.0,-84.0,20473,US,1004,32.5,-84.5,7922,US,5.5
    2.2.2.2,47.0,-122.0,40,US,1005,46.5,-122.5,7922,US,8.0
    2.2.2.2,47.0,-122.0,40,US,1006,47.5,-122.5,7922,US,9.0
    2.2.2.2,47.0,-122.0,40,US,1007,47.5,-121.5,7922,US,7.5
""").strip() + "\n"


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.csv_path = root / "canonical.csv"
        self.csv_path.write_text(_SYNTH_CSV)
        self.inputs_root = root / "inputs"
        self.outputs_root = root / "outputs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _materialize(
        self, source: str = "generic_csv", slice: str = "fold_0", run_id: str = "cli-test",
    ) -> Path:
        # Call the source directly with our synth path via inputs.materialize_inputs.
        from scripts.benchmark.v2.inputs import materialize_inputs
        from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource

        src = GenericCSVSource(
            slice=slice, setup="anchors_to_probes",
            csv_path=self.csv_path, k=4,
        )
        return materialize_inputs(src, root=self.inputs_root, run_id=run_id)

    def test_run_combo_command_writes_outputs(self) -> None:
        self._materialize()
        result = self.runner.invoke(app, [
            "run-combo",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "cli-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        combo_dir = (
            self.outputs_root / "cli-test" / "generic_csv" / "anchors_to_probes" / "fold_0"
            / "speed_of_internet__planar_circle__geometric_centroid"
        )
        self.assertTrue((combo_dir / "run.json").exists())
        self.assertTrue((combo_dir / "targets.parquet").exists())

    def test_run_combo_fails_without_materialized_inputs(self) -> None:
        result = self.runner.invoke(app, [
            "run-combo",
            "--source", "generic_csv", "--slice", "fold_2",
            "--setup", "anchors_to_probes",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "cli-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("materialize-inputs", result.output)

    def test_materialize_forwards_source_kwargs(self) -> None:
        """`--source-kwargs` JSON is parsed and forwarded as **kwargs to the
        source constructor — exercised here with a generic_csv override that
        points at a temp CSV."""
        alt_csv = Path(self.tmp.name) / "alt.csv"
        alt_csv.write_text(_SYNTH_CSV)
        # Path the source through --source-kwargs rather than positional args.
        result = self.runner.invoke(app, [
            "materialize-inputs",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--run-id", "kw-test",
            "--inputs-root", str(self.inputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(alt_csv), "k": 4}),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        manifest = (
            self.inputs_root / "generic_csv" / "kw-test" / "anchors_to_probes" / "fold_0"
            / "manifest.json"
        )
        self.assertTrue(manifest.exists())

    def test_materialize_rejects_invalid_source_kwargs_json(self) -> None:
        result = self.runner.invoke(app, [
            "materialize-inputs",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--run-id", "kw-test",
            "--inputs-root", str(self.inputs_root),
            "--source-kwargs", "not-json",
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_materialize_accepts_eval_kept_traffic_fraction(self) -> None:
        csv = textwrap.dedent("""
            vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,target_city,rtt_ms,weight
            1.1.1.1,33.0,-84.0,t1,40.0,-100.0,atlanta,10.0,10
            2.2.2.2,47.0,-122.0,t1,40.0,-100.0,atlanta,11.0,1
            1.1.1.1,33.0,-84.0,t2,41.0,-101.0,boston,12.0,9
            2.2.2.2,47.0,-122.0,t2,41.0,-101.0,boston,13.0,1
        """).strip() + "\n"
        weighted = Path(self.tmp.name) / "weighted_eval_frac.csv"
        weighted.write_text(csv)
        result = self.runner.invoke(app, [
            "materialize-inputs",
            "--source", "traffic_weighted_csv", "--slice", "all",
            "--setup", "anchors_to_probes",
            "--run-id", "frac-test",
            "--inputs-root", str(self.inputs_root),
            "--source-kwargs", json.dumps({"mesh_csv_path": str(weighted)}),
            "--eval-kept-traffic-fraction", "0.95",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        manifest = (
            self.inputs_root / "traffic_weighted_csv" / "frac-test"
            / "anchors_to_probes" / "all" / "manifest.json"
        )
        self.assertTrue(manifest.exists())

    def test_materialize_rejects_weight_flag_on_an_unsupported_source(self) -> None:
        """generic_csv no longer takes the weighted kwargs. The CLI must say so
        and name the source that does, rather than raising a bare TypeError out
        of the constructor."""
        result = self.runner.invoke(app, [
            "materialize-inputs",
            "--source", "generic_csv", "--slice", "all",
            "--setup", "anchors_to_probes",
            "--run-id", "reject-test",
            "--inputs-root", str(self.inputs_root),
            "--source-kwargs", json.dumps({"csv_path": "unused.csv"}),
            "--eval-kept-traffic-fraction", "0.95",
        ])
        self.assertEqual(result.exit_code, 2, msg=result.output)
        self.assertIn("traffic_weighted_csv", result.output)

    def test_materialize_rejects_eval_threshold_and_fraction_together(self) -> None:
        result = self.runner.invoke(app, [
            "materialize-inputs",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--run-id", "kw-test",
            "--inputs-root", str(self.inputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
            "--eval-pair-weight-min", "1.0",
            "--eval-kept-traffic-fraction", "0.95",
        ])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("Pass only one", result.output)

    def test_summarize_aggregates_combos(self) -> None:
        self._materialize(run_id="sum-test")
        # Run two combos under one run id.
        for ltd_name in ("speed_of_internet", "low_envelope"):
            r = self.runner.invoke(app, [
                "run-combo",
                "--source", "generic_csv", "--slice", "fold_0",
                "--setup", "anchors_to_probes",
                "--ltd", ltd_name, "--mtl", "planar_circle", "--ctr", "geometric_centroid",
                "--run-id", "sum-test",
                "--inputs-root", str(self.inputs_root),
                "--outputs-root", str(self.outputs_root),
                "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
            ])
            self.assertEqual(r.exit_code, 0, msg=r.output)

        result = self.runner.invoke(app, [
            "summarize",
            "--run-id", "sum-test",
            "--outputs-root", str(self.outputs_root),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        summary_path = self.outputs_root / "sum-test" / "summary.parquet"
        self.assertTrue(summary_path.exists())
        table = pq.read_table(summary_path)
        self.assertEqual(table.num_rows, 2)
        combos = set(table.column("ltd").to_pylist())
        self.assertEqual(combos, {"speed_of_internet", "low_envelope"})

    def _write_tiny_airports(self) -> Path:
        """A hermetic 3-airport reference parquet so the test doesn't depend on
        the (uncommitted, regenerated) full OurAirports artifact."""
        import pandas as pd
        path = Path(self.tmp.name) / "airports.parquet"
        pd.DataFrame({
            "iata_code": ["ATL", "SEA", "LHR"],
            "latitude_deg": [33.6407, 47.4502, 51.4700],
            "longitude_deg": [-84.4277, -122.3088, -0.4543],
            "municipality": ["Atlanta", "Seattle", "London"],
        }).to_parquet(path, index=False)
        return path

    def test_airport_eval_annotates_targets_and_writes_summary(self) -> None:
        from scripts.benchmark.v2.airport_eval import AIRPORT_COLUMNS

        airports = self._write_tiny_airports()
        self._materialize(run_id="ap-test")
        r = self.runner.invoke(app, [
            "run-combo",
            "--source", "generic_csv", "--slice", "fold_0",
            "--setup", "anchors_to_probes",
            "--ltd", "speed_of_internet", "--mtl", "planar_circle", "--ctr", "geometric_centroid",
            "--run-id", "ap-test",
            "--inputs-root", str(self.inputs_root),
            "--outputs-root", str(self.outputs_root),
            "--source-kwargs", json.dumps({"csv_path": str(self.csv_path), "k": 4}),
        ])
        self.assertEqual(r.exit_code, 0, msg=r.output)

        result = self.runner.invoke(app, [
            "airport-eval",
            "--run-id", "ap-test",
            "--outputs-root", str(self.outputs_root),
            "--airports", str(airports),
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)

        combo_dir = (
            self.outputs_root / "ap-test" / "generic_csv" / "anchors_to_probes" / "fold_0"
            / "speed_of_internet__planar_circle__geometric_centroid"
        )
        cols = set(pq.read_table(combo_dir / "targets.parquet").column_names)
        for col in AIRPORT_COLUMNS:
            self.assertIn(col, cols)

        summary_path = self.outputs_root / "ap-test" / "airport_summary.parquet"
        self.assertTrue(summary_path.exists())
        summ = pq.read_table(summary_path)
        self.assertEqual(summ.num_rows, 1)
        self.assertIn("airport_match_rate", summ.column_names)


if __name__ == "__main__":
    unittest.main()


# Two VPs measure six targets as a full mesh; two more targets are seen by a
# single VP. `min_obs=2` therefore discriminates -- which is what separates
# "the source's target set" from "the CSV's unique target_ids".
_CFG_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,vp_asn,vp_country,target_id,target_lat,target_lon,target_asn,target_country,rtt_ms
    1.1.1.1,33.0,-84.0,20473,US,1001,33.5,-84.5,7922,US,5.0
    2.2.2.2,47.0,-122.0,40,US,1001,33.5,-84.5,7922,US,45.0
    1.1.1.1,33.0,-84.0,20473,US,1002,32.5,-83.5,7922,US,6.0
    2.2.2.2,47.0,-122.0,40,US,1002,32.5,-83.5,7922,US,46.0
    1.1.1.1,33.0,-84.0,20473,US,1003,33.5,-83.5,7922,US,7.0
    2.2.2.2,47.0,-122.0,40,US,1003,33.5,-83.5,7922,US,47.0
    1.1.1.1,33.0,-84.0,20473,US,1004,32.5,-84.5,7922,US,5.5
    2.2.2.2,47.0,-122.0,40,US,1004,32.5,-84.5,7922,US,44.0
    1.1.1.1,33.0,-84.0,20473,US,1005,46.5,-122.5,7922,US,40.0
    2.2.2.2,47.0,-122.0,40,US,1005,46.5,-122.5,7922,US,8.0
    1.1.1.1,33.0,-84.0,20473,US,1006,47.5,-122.5,7922,US,41.0
    2.2.2.2,47.0,-122.0,40,US,1006,47.5,-122.5,7922,US,9.0
    1.1.1.1,33.0,-84.0,20473,US,1007,47.5,-121.5,7922,US,42.0
    2.2.2.2,47.0,-122.0,40,US,1008,46.0,-121.0,7922,US,7.5
""").strip() + "\n"

_CFG_SLICES = ["fold_0", "fold_1", "fold_2", "fold_3"]


class TestResolveBenchConfig(unittest.TestCase):
    """`_resolve_bench_config` must agree with `Snakefile`'s `bcfg`.

    The Snakefile is the format's definition and cannot be imported, so the
    rule it implements -- `benchmark:` wins key by key, top level is the
    fallback -- is pinned here instead. A drift would make the target space
    disagree with the run built from the same file.
    """

    def _write(self, body: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "cfg.yaml"
        p.write_text(textwrap.dedent(body))
        return p

    def test_benchmark_block_wins_key_by_key(self) -> None:
        from scripts.benchmark.v2.cli import _resolve_bench_config

        cfg = _resolve_bench_config(self._write("""
            run_id: top-level-id
            source: top_level_source
            benchmark:
              source: block_source
              slices: [fold_0]
        """))
        self.assertEqual(cfg["source"], "block_source")
        # Not shadowed by the block, so the top level still supplies it.
        self.assertEqual(cfg["run_id"], "top-level-id")
        self.assertEqual(cfg["slices"], ["fold_0"])

    def test_flat_shape_resolves(self) -> None:
        from scripts.benchmark.v2.cli import _resolve_bench_config

        cfg = _resolve_bench_config(self._write("""
            run_id: flat
            source: generic_csv
            slices: [fold_0]
        """))
        self.assertEqual(cfg["source"], "generic_csv")
        self.assertEqual(cfg["slices"], ["fold_0"])

    def test_non_mapping_benchmark_block_raises(self) -> None:
        from scripts.benchmark.v2.cli import _resolve_bench_config

        with self.assertRaises(TypeError):
            _resolve_bench_config(self._write("""
                run_id: x
                benchmark: [not, a, mapping]
            """))


class TestEdgeCsvFromKwargs(unittest.TestCase):
    """Which CSV holds a run's edge set, and whether it is exact."""

    def test_plain_mesh_is_exact(self) -> None:
        from scripts.benchmark.v2.cli import _edge_csv_from_kwargs

        path, superset = _edge_csv_from_kwargs({"csv_path": "a.csv"})
        self.assertEqual(path, Path("a.csv"))
        self.assertFalse(superset)

    def test_precomputed_weighted_uses_the_weighted_file(self) -> None:
        from scripts.benchmark.v2.cli import _edge_csv_from_kwargs

        path, superset = _edge_csv_from_kwargs(
            {"mesh_csv_path": "mesh.csv", "weighted_csv_path": "w.csv"}
        )
        self.assertEqual(path, Path("w.csv"))
        self.assertFalse(superset)

    def test_on_the_fly_weighted_flags_the_mesh_as_a_superset(self) -> None:
        """No file holds an on-the-fly arm's pruned flows, so the mesh is all
        there is -- and must be labelled as more edges than the run evaluates."""
        from scripts.benchmark.v2.cli import _edge_csv_from_kwargs

        for flag in ("eval_kept_traffic_fraction", "eval_pair_weight_min"):
            with self.subTest(flag=flag):
                path, superset = _edge_csv_from_kwargs(
                    {"mesh_csv_path": "mesh.csv", flag: 0.95}
                )
                self.assertEqual(path, Path("mesh.csv"))
                self.assertTrue(superset)

    def test_no_csv_kwarg_yields_none(self) -> None:
        from scripts.benchmark.v2.cli import _edge_csv_from_kwargs

        self.assertEqual(_edge_csv_from_kwargs({"k": 5}), (None, False))


class TestMaterializeTargetSpaceFromConfig(unittest.TestCase):
    """`--configfile` builds the target space with no benchmark run on disk."""

    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.csv_path = self.root / "canonical.csv"
        self.csv_path.write_text(_CFG_CSV)
        self.outputs_root = self.root / "outputs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _config(self, *, min_obs: int | None = None, run_id: str = "cfg-test") -> Path:
        kw = {"csv_path": str(self.csv_path), "k": 4}
        if min_obs is not None:
            kw["min_obs"] = min_obs
        body = {
            "run_id": run_id,
            "benchmark": {
                "source": "generic_csv",
                "setup": "anchors_to_probes",
                "slices": _CFG_SLICES,
                "source_kwargs": kw,
            },
        }
        p = self.root / f"{run_id}.yaml"
        p.write_text(json.dumps(body))  # JSON is valid YAML
        return p

    def _run(self, *args) -> "object":
        return self.runner.invoke(app, [
            "materialize-target-space",
            "--outputs-root", str(self.outputs_root),
            *args,
        ])

    def _setup_dir(self, run_id: str = "cfg-test") -> Path:
        return self.outputs_root / run_id / "generic_csv" / "anchors_to_probes"

    def _expected_nodes(self, **kwargs) -> tuple[set, set]:
        """The union the *source* yields — the benchmark's own definition."""
        from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource

        targets, vps = set(), set()
        for sl in _CFG_SLICES:
            src = GenericCSVSource(
                slice=sl, setup="anchors_to_probes",
                csv_path=self.csv_path, k=4, **kwargs,
            )
            targets |= {t.target_id for t in src.iter_eval_targets()}
            vps |= {v.vp_id for v in src.iter_vp_configs()}
        return targets, vps

    def test_builds_node_sets_and_clusters_with_no_run_outputs(self) -> None:
        import pandas as pd

        result = self._run("--configfile", str(self._config()))
        self.assertEqual(result.exit_code, 0, result.output)

        d = self._setup_dir()
        self.assertTrue((d / "targets.csv").exists())
        self.assertTrue((d / "vps.csv").exists())
        self.assertTrue((d / "clusters" / "clusters.csv").exists())
        self.assertTrue((d / "clusters" / "assignments.csv").exists())
        # Nothing a benchmark run would have written.
        self.assertEqual(list(d.glob("fold_*")), [])

        want_t, want_v = self._expected_nodes()
        self.assertEqual(set(pd.read_csv(d / "targets.csv")["target_id"].astype(str)),
                         {str(t) for t in want_t})
        self.assertEqual(set(pd.read_csv(d / "vps.csv")["vp_id"].astype(str)),
                         {str(v) for v in want_v})

    def test_vps_csv_schema_matches_the_run_id_path(self) -> None:
        """Both modes must write one file, not two schemas of the same data."""
        import pandas as pd

        self.assertEqual(self._run("--configfile", str(self._config())).exit_code, 0)
        self.assertEqual(
            list(pd.read_csv(self._setup_dir() / "vps.csv").columns),
            ["vp_id", "vp_lat", "vp_lon", "vp_asn", "vp_country",
             "continent", "region", "city"],
        )

    def test_min_obs_is_applied_via_the_source(self) -> None:
        """Proves the target set comes from the source, not from the CSV.

        Targets 1007/1008 have a single observation each, so `min_obs=2` drops
        them — a `drop_duplicates` over the CSV's `target_id` column could not
        know that, and would publish a target space the run never evaluates.
        """
        import pandas as pd

        cfg = self._config(min_obs=2, run_id="min-obs")
        self.assertEqual(self._run("--configfile", str(cfg)).exit_code, 0)

        got = set(pd.read_csv(self._setup_dir("min-obs") / "targets.csv")["target_id"].astype(str))
        want, _ = self._expected_nodes(min_obs=2)
        self.assertEqual(got, {str(t) for t in want})
        self.assertNotIn("1007", got)
        self.assertNotIn("1008", got)

    def test_target_space_json_records_the_canonical_csv(self) -> None:
        self.assertEqual(self._run("--configfile", str(self._config())).exit_code, 0)

        space = json.loads((self._setup_dir() / "target_space.json").read_text())
        self.assertEqual(space["derived_from"], "config")
        self.assertEqual(space["slices"], _CFG_SLICES)
        self.assertFalse(space["csv_is_mesh_superset"])
        self.assertTrue(space["csv"].endswith("canonical.csv"))
        self.assertEqual(space["n_vps"], 2)

    def test_requires_exactly_one_of_run_id_and_configfile(self) -> None:
        cfg = self._config()
        self.assertEqual(self._run().exit_code, 2)
        self.assertEqual(
            self._run("--configfile", str(cfg), "--run-id", "cfg-test").exit_code, 2
        )

    def test_weight_key_on_a_source_that_rejects_it_exits_2(self) -> None:
        """The traffic keys sit at the config's top level, so a mesh config
        carrying one would otherwise reach `generic_csv` as a bare TypeError."""
        cfg = self.root / "bad.yaml"
        cfg.write_text(json.dumps({
            "run_id": "bad",
            "eval_kept_traffic_fraction": 0.95,
            "benchmark": {
                "source": "generic_csv",
                "setup": "anchors_to_probes",
                "slices": ["fold_0"],
                "source_kwargs": {"csv_path": str(self.csv_path), "k": 4},
            },
        }))
        result = self._run("--configfile", str(cfg))
        self.assertEqual(result.exit_code, 2)
        self.assertIn("traffic_weighted_csv", result.output)

    def test_empty_slices_exits_2(self) -> None:
        cfg = self.root / "noslices.yaml"
        cfg.write_text(json.dumps({
            "run_id": "noslices",
            "benchmark": {"source": "generic_csv", "slices": [],
                          "source_kwargs": {"csv_path": str(self.csv_path)}},
        }))
        self.assertEqual(self._run("--configfile", str(cfg)).exit_code, 2)

    def test_with_eval_source_writes_where_build_proximity_looks(self) -> None:
        """`eval-source`'s own default --out-dir is the CSV's directory, which
        is not where `analysis.v3` reads it from."""
        self.assertEqual(
            self._run("--configfile", str(self._config()), "--with-eval-source").exit_code,
            0,
        )
        eval_dir = self.outputs_root / "cfg-test" / "eval_source"
        self.assertTrue(sorted(eval_dir.glob("*_eval_per_target.csv")))
        self.assertTrue(sorted(eval_dir.glob("*_eval_stats.json")))

    def test_run_id_mode_still_errors_without_run_outputs(self) -> None:
        (self.outputs_root / "cfg-test").mkdir(parents=True)
        result = self._run("--run-id", "cfg-test")
        self.assertEqual(result.exit_code, 1)
        self.assertIn("--configfile", result.output)
