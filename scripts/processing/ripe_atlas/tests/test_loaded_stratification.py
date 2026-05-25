"""Tests for LoadedStratification — consumes a precomputed stratification JSON.

Covers JSON loading, active-vs-stratification intersection, empty-fold
guards, and constructor validation. Integration with RipeAtlasSource lives
in test_sources.py.
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from scripts.processing.ripe_atlas.stratification import (
    AnchorInfo,
    LoadedStratification,
)


def _write_stratification(
    tmpdir: Path,
    fold_assignments: dict[str, int],
    *,
    policy_class: str = "DistGeoStratification",
    k: int = 5,
    seed: int = 42,
    asn_bucket_top_n: int = 20,
    spatial_clusters: int | None = None,
    filename: str = "s.json",
) -> Path:
    """Write a minimal stratification JSON matching what `stratify.py` emits."""
    policy: dict = {
        "class": policy_class,
        "k": k,
        "seed": seed,
        "asn_bucket_top_n": asn_bucket_top_n,
        "kind": "dist_geo_kfold" if policy_class == "DistGeoStratification" else "sechidis_kfold",
    }
    if policy_class == "SechidisStratification":
        policy["spatial_clusters"] = spatial_clusters
        policy["labels"] = ["country", "asn_bucket"]
    fold_sizes = [
        sum(1 for f in fold_assignments.values() if f == i) for i in range(k)
    ]
    payload = {
        "policy": policy,
        "corpus": {"source": "test", "n_anchors_yielded": len(fold_assignments)},
        "generated_at": "2026-05-24T00:00:00+00:00",
        "fold_sizes": fold_sizes,
        "fold_assignments": fold_assignments,
    }
    path = tmpdir / filename
    path.write_text(json.dumps(payload))
    return path


def _anchor(ip: str) -> AnchorInfo:
    return AnchorInfo(ip=ip, lat=0.0, lon=0.0, country=None, asn=None)


class TestLoadedStratificationLoad(unittest.TestCase):
    def test_compute_returns_loaded_assignments_for_matching_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            assignments = {f"10.0.0.{i}": i % 5 for i in range(50)}
            path = _write_stratification(tmp, assignments, k=5)
            policy = LoadedStratification(path=path, fold_index=0)
            anchors = [_anchor(ip) for ip in assignments]
            result = policy.compute_fold_assignments(anchors)
            self.assertEqual(result, assignments)

    def test_k_property_reads_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            assignments = {f"10.0.0.{i}": i % 7 for i in range(14)}
            path = _write_stratification(tmp, assignments, k=7)
            policy = LoadedStratification(path=path, fold_index=0)
            self.assertEqual(policy.k, 7)


class TestLoadedStratificationIntersection(unittest.TestCase):
    def test_drops_active_anchors_not_in_stratification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Stratification only knows about 40 anchors; active has 50.
            assignments = {f"10.0.0.{i}": i % 5 for i in range(40)}
            path = _write_stratification(tmp, assignments, k=5)
            policy = LoadedStratification(path=path, fold_index=0)
            anchors = [_anchor(f"10.0.0.{i}") for i in range(50)]
            with self.assertLogs(
                "scripts.processing.ripe_atlas.stratification", level="WARNING"
            ) as logs:
                result = policy.compute_fold_assignments(anchors)
            self.assertEqual(len(result), 40)
            self.assertTrue(
                any("10 active anchor(s)" in m for m in logs.output),
                f"expected drop warning in logs: {logs.output}",
            )

    def test_ignores_stratification_anchors_not_in_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Stratification has 50 anchors; active only has 30.
            assignments = {f"10.0.0.{i}": i % 5 for i in range(50)}
            path = _write_stratification(tmp, assignments, k=5)
            policy = LoadedStratification(path=path, fold_index=0)
            anchors = [_anchor(f"10.0.0.{i}") for i in range(30)]
            with self.assertLogs(
                "scripts.processing.ripe_atlas.stratification", level="WARNING"
            ) as logs:
                result = policy.compute_fold_assignments(anchors)
            self.assertEqual(len(result), 30)
            self.assertTrue(
                any("20 stratification anchor(s)" in m for m in logs.output),
                f"expected ignore warning in logs: {logs.output}",
            )


class TestLoadedStratificationEmptyFolds(unittest.TestCase):
    def test_raises_when_eval_fold_empty_after_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 10 anchors all in fold 0; pick fold_index=1 → empty eval.
            assignments = {f"10.0.0.{i}": 0 for i in range(10)}
            # Need at least one assignment in fold 1 to write a valid
            # stratification for fold_index=1 to be constructible; trick:
            # write 1 anchor in fold 1 so __post_init__ passes, then drop
            # it from active.
            assignments["10.0.0.99"] = 1
            path = _write_stratification(tmp, assignments, k=2)
            policy = LoadedStratification(path=path, fold_index=1)
            # Active set excludes the lone fold-1 anchor.
            anchors = [_anchor(f"10.0.0.{i}") for i in range(10)]
            with self.assertRaises(ValueError) as ctx:
                policy.compute_fold_assignments(anchors)
            self.assertIn("eval set is empty", str(ctx.exception))

    def test_raises_when_fit_fold_empty_after_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 10 anchors all in fold 1, plus 1 in fold 0 (so file is valid
            # for K=2); active drops the fold-0 anchor → fit is empty for
            # fold_index=1.
            assignments = {f"10.0.0.{i}": 1 for i in range(10)}
            assignments["10.0.0.99"] = 0
            path = _write_stratification(tmp, assignments, k=2)
            policy = LoadedStratification(path=path, fold_index=1)
            anchors = [_anchor(f"10.0.0.{i}") for i in range(10)]
            with self.assertRaises(ValueError) as ctx:
                policy.compute_fold_assignments(anchors)
            self.assertIn("fit set is empty", str(ctx.exception))


class TestLoadedStratificationValidation(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            LoadedStratification(path=Path("/does/not/exist.json"))

    def test_fold_index_out_of_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = _write_stratification(tmp, {"a": 0, "b": 1}, k=2)
            with self.assertRaises(ValueError):
                LoadedStratification(path=path, fold_index=2)
            with self.assertRaises(ValueError):
                LoadedStratification(path=path, fold_index=-1)

    def test_string_path_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = _write_stratification(
                tmp, {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}, k=5,
            )
            # Construct with string path.
            policy = LoadedStratification(path=str(path), fold_index=0)
            self.assertIsInstance(policy.path, Path)

    def test_missing_required_keys_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = tmp / "bad.json"
            path.write_text(json.dumps({"policy": {"k": 5}}))  # no fold_assignments
            with self.assertRaises(ValueError):
                LoadedStratification(path=path, fold_index=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
