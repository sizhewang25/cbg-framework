"""Naming a pooled artifact: the declared label, and the hashed directory.

`cross.py` had no tests. The bug it grew was a naming bug that corrupted a
number -- `short_dataset` read the dataset by splitting the run id on its first
hyphen, so every `pro-*` run read `pro`, and the octant-finetuning pool then
namespaced its sites on that constant and merged three datasets into one. So
these tests are mostly about the seam: what is display (may repeat, may be
absent) and what is identity (may not).
"""

from __future__ import annotations

import json

import pytest
import yaml

from scripts.analysis.v5.modules import cross, labels


def _run_tree(tmp_path, run_id, *, label=None, config_name=None, manifest=True):
    """A run directory v5 can resolve, optionally with a config declaring a label.

    Mirrors the real layout: `<root>/<run>/<source>/<setup>/fold_*` is what
    `discover_runs` looks for, and `target_space.json` sits beside the folds.
    """
    root = tmp_path / "outputs"
    setup = root / run_id / "generic_csv" / "anchors_to_probes"
    (setup / "fold_0").mkdir(parents=True, exist_ok=True)

    if manifest:
        body = {"run_id": run_id, "source": "generic_csv"}
        if config_name is not None:
            cfg = tmp_path / "configs" / config_name
            cfg.parent.mkdir(parents=True, exist_ok=True)
            if label is not None:
                cfg.write_text(yaml.safe_dump(
                    {"run_id": run_id, "analysis": {"common": {"dataset_label": label}}}
                ))
            body["config"] = str(cfg)
        (setup / "target_space.json").write_text(json.dumps(body))
    return root


@pytest.fixture(autouse=True)
def _clear_label_cache():
    """`dataset_label` is memoized on (run_id, root); tmp_path makes the root
    unique per test, but the cache still grows across them."""
    labels.dataset_label.cache_clear()
    yield
    labels.dataset_label.cache_clear()


class TestTheDeclaredLabelWins:
    def test_a_declared_label_is_used(self, tmp_path):
        root = _run_tree(tmp_path, "pro-as01-mesh", label="as01",
                         config_name="pro-as01-mesh.yaml")
        assert cross.short_dataset("pro-as01-mesh", outputs_root=root) == "as01"

    def test_a_prefix_no_longer_collapses_three_runs_into_one(self, tmp_path):
        """The bug, stated as a test. All three used to read `pro`."""
        root = None
        for n in ("01", "02", "03"):
            root = _run_tree(tmp_path, f"pro-as{n}-mesh", label=f"as{n}",
                             config_name=f"pro-as{n}-mesh.yaml")
        runs = [f"pro-as{n}-mesh" for n in ("01", "02", "03")]
        assert [cross.short_dataset(r, outputs_root=root) for r in runs] == [
            "as01", "as02", "as03"]

    def test_two_differently_named_runs_can_share_a_dataset(self, tmp_path):
        """The point of declaring it: the run id no longer has to carry it."""
        root = _run_tree(tmp_path, "pro-as01-mesh", label="as01",
                         config_name="pro-as01-mesh.yaml")
        _run_tree(tmp_path, "as01-260728-260802-mesh", label="as01",
                  config_name="as01-260728-260802-mesh.yaml")
        assert cross.short_dataset("pro-as01-mesh", outputs_root=root) == "as01"
        assert cross.short_dataset(
            "as01-260728-260802-mesh", outputs_root=root) == "as01"


class TestTheFallbackIsTheRunId:
    """Never raises, and never guesses. A run id is always a correct label."""

    def test_an_unknown_run(self, tmp_path):
        root = _run_tree(tmp_path, "other", label="x", config_name="other.yaml")
        assert cross.short_dataset("nope", outputs_root=root) == "nope"

    def test_a_run_with_no_manifest(self, tmp_path):
        root = _run_tree(tmp_path, "as7018_us_test01", manifest=False)
        assert cross.short_dataset(
            "as7018_us_test01", outputs_root=root) == "as7018_us_test01"

    def test_a_manifest_with_no_config_key(self, tmp_path):
        root = _run_tree(tmp_path, "r1", config_name=None)
        assert cross.short_dataset("r1", outputs_root=root) == "r1"

    def test_a_config_that_has_been_deleted(self, tmp_path):
        """`as01-materialization-test` is this case on the real tree."""
        root = _run_tree(tmp_path, "r1", label="as01", config_name="gone.yaml")
        (tmp_path / "configs" / "gone.yaml").unlink()
        assert cross.short_dataset("r1", outputs_root=root) == "r1"

    def test_a_config_declaring_no_label(self, tmp_path):
        root = _run_tree(tmp_path, "r1", label=None, config_name="r1.yaml")
        (tmp_path / "configs" / "r1.yaml").write_text(
            yaml.safe_dump({"run_id": "r1", "analysis": {"common": {"grid": "h3"}}}))
        assert cross.short_dataset("r1", outputs_root=root) == "r1"

    def test_a_config_that_is_not_valid_yaml(self, tmp_path):
        root = _run_tree(tmp_path, "r1", label="as01", config_name="r1.yaml")
        (tmp_path / "configs" / "r1.yaml").write_text("{ this: is: not: yaml")
        assert cross.short_dataset("r1", outputs_root=root) == "r1"

    def test_a_non_scalar_label_is_refused(self, tmp_path):
        """It would be carried into a CSV column and a caption."""
        root = _run_tree(tmp_path, "r1", label=None, config_name="r1.yaml")
        (tmp_path / "configs" / "r1.yaml").write_text(yaml.safe_dump(
            {"analysis": {"common": {"dataset_label": ["as01", "as02"]}}}))
        assert cross.short_dataset("r1", outputs_root=root) == "r1"


class TestTheSlugStaysReadable:
    """It is printed on figures and written into the LaTeX the paper includes,
    so it is the one name that must not become a hash."""

    def test_joins_the_declared_labels(self, tmp_path):
        root = None
        for n in ("01", "02", "03"):
            root = _run_tree(tmp_path, f"pro-as{n}-mesh", label=f"as{n}",
                             config_name=f"pro-as{n}-mesh.yaml")
        runs = [f"pro-as{n}-mesh" for n in ("01", "02", "03")]
        assert cross.dataset_slug(runs, outputs_root=root) == "as01+as02+as03"

    def test_does_not_depend_on_run_order(self, tmp_path):
        root = None
        for n in ("01", "02"):
            root = _run_tree(tmp_path, f"pro-as{n}-mesh", label=f"as{n}",
                             config_name=f"pro-as{n}-mesh.yaml")
        a = cross.dataset_slug(["pro-as01-mesh", "pro-as02-mesh"], outputs_root=root)
        b = cross.dataset_slug(["pro-as02-mesh", "pro-as01-mesh"], outputs_root=root)
        assert a == b == "as01+as02"


class TestTheDirectoryIsContentAddressed:
    def test_name_is_the_count_and_a_digest(self, tmp_path):
        name = cross.cross_name(["a", "b", "c"])
        assert name.startswith("3-runs-")
        assert len(name.split("-")[-1]) == cross.SLUG_HASH_CHARS

    def test_name_does_not_depend_on_order(self):
        assert cross.cross_name(["b", "a"]) == cross.cross_name(["a", "b"])

    def test_different_run_sets_get_different_names(self):
        """What `<datasets>@<arm>` could not guarantee: two arms of one dataset
        set, and a two-run pool against a three-run one."""
        mesh = ["as01-260728-260802-mesh", "as02-260728-260802-mesh"]
        weighted = ["as01-260728-260802-weighted", "as02-260728-260802-weighted"]
        assert cross.cross_name(mesh) != cross.cross_name(weighted)
        assert cross.cross_name(mesh) != cross.cross_name(mesh + ["as03-x"])

    def test_prefixed_runs_no_longer_share_a_directory(self):
        """These three collapsed to a single `pro` directory."""
        pro = ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]
        assert cross.cross_name(pro) == f"3-runs-{cross.runs_hash(pro)}"
        assert cross.cross_name(pro) != cross.cross_name(pro[:2])

    def test_the_directory_says_what_it_holds(self, tmp_path):
        """A digest reads as nothing, so the run ids ride along beside it."""
        runs = ["pro-as02-mesh", "pro-as01-mesh"]
        d = cross.cross_dir(runs, analysis_root=tmp_path, kind="classify")
        assert d.name == cross.cross_name(runs)
        body = json.loads((d / cross.RUNS_JSON).read_text())
        assert body["run_ids"] == ["pro-as01-mesh", "pro-as02-mesh"]
        assert body["kind"] == "classify"

    def test_runs_json_is_restored_if_lost(self, tmp_path):
        runs = ["a", "b"]
        d = cross.cross_dir(runs, analysis_root=tmp_path)
        (d / cross.RUNS_JSON).unlink()
        assert (cross.cross_dir(runs, analysis_root=tmp_path)
                / cross.RUNS_JSON).exists()


class TestGuardDistinctLabels:
    def test_distinct_labels_pass_through(self):
        got = {"r1": "as01", "r2": "as02"}
        assert cross.guard_distinct_labels(got) == got

    def test_a_shared_label_is_refused(self):
        with pytest.raises(ValueError, match="declared by"):
            cross.guard_distinct_labels({"pro-as01-mesh": "as01", "r2": "as01"})

    def test_the_message_names_both_runs(self):
        with pytest.raises(ValueError) as exc:
            cross.guard_distinct_labels({"r2": "as01", "r1": "as01"})
        assert "'r1', 'r2'" in str(exc.value).replace('"', "'")

    def test_the_run_id_fallback_never_trips_it(self, tmp_path):
        """Two runs with no declared label fall back to run ids, which are
        distinct by construction -- so the fallback is safe to pool."""
        root = _run_tree(tmp_path, "r1", manifest=False)
        _run_tree(tmp_path, "r2", manifest=False)
        got = cross.labels_for(["r1", "r2"], outputs_root=root)
        assert cross.guard_distinct_labels(got) == {"r1": "r1", "r2": "r2"}
