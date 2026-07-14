"""GenericPresplitSource tests — two-file (train, test) source, no splitting.

Synthetic in-memory fixtures only; both CSV and Parquet paths are exercised.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark.v2.sources import SOURCES
from scripts.benchmark.v2.sources.generic_presplit import GenericPresplitSource

# Shared VPs (identical coords across files), disjoint targets: train gets
# 1001..1004, test gets 2001..2002.
_TRAIN_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,vp_asn,vp_country,target_id,target_lat,target_lon,target_asn,target_country,target_city,rtt_ms,weight
    1.1.1.1,33.0,-84.0,20473,US,1001,40.0,-100.0,7922,US,Denver,10.0,5.0
    1.1.1.1,33.0,-84.0,20473,US,1002,41.0,-101.0,7922,US,Denver,12.0,3.0
    1.1.1.1,33.0,-84.0,20473,US,1003,42.0,-102.0,3356,US,Boise,13.0,2.0
    1.1.1.1,33.0,-84.0,20473,US,1004,43.0,-103.0,3356,US,Boise,14.0,1.0
    2.2.2.2,47.0,-122.0,40,US,1001,40.0,-100.0,7922,US,Denver,11.0,4.0
    2.2.2.2,47.0,-122.0,40,US,1002,41.0,-101.0,7922,US,Denver,11.5,2.0
    2.2.2.2,47.0,-122.0,40,US,1003,42.0,-102.0,3356,US,Boise,12.5,1.0
""").strip() + "\n"

_TEST_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,vp_asn,vp_country,target_id,target_lat,target_lon,target_asn,target_country,target_city,rtt_ms,weight
    1.1.1.1,33.0,-84.0,20473,US,2001,44.0,-104.0,7018,US,Fargo,15.0,8.0
    1.1.1.1,33.0,-84.0,20473,US,2002,45.0,-105.0,7018,US,Fargo,16.0,0.5
    2.2.2.2,47.0,-122.0,40,US,2001,44.0,-104.0,7018,US,Fargo,14.5,6.0
    2.2.2.2,47.0,-122.0,40,US,2002,45.0,-105.0,7018,US,Fargo,15.5,0.25
    3.3.3.3,25.0,-80.0,64512,US,2001,44.0,-104.0,7018,US,Fargo,20.0,1.0
    3.3.3.3,25.0,-80.0,64512,US,2003,46.0,-106.0,7018,US,Fargo,21.0,0.1
""").strip() + "\n"


class _PresplitBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train_path = self.root / "train.csv"
        self.test_path = self.root / "test.csv"
        self.train_path.write_text(_TRAIN_CSV)
        self.test_path.write_text(_TEST_CSV)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make(self, slice: str = "all", **kwargs) -> GenericPresplitSource:
        kwargs.setdefault("train_path", self.train_path)
        kwargs.setdefault("test_path", self.test_path)
        return GenericPresplitSource(slice=slice, **kwargs)


class TestGenericPresplitSource_Basics(_PresplitBase):
    def test_registered_in_sources(self) -> None:
        self.assertIs(SOURCES["generic_presplit"], GenericPresplitSource)

    def test_slice_round_trip_and_setup_is_fixed(self) -> None:
        # The legacy setup axis is ignored: whatever the CLI passes, the
        # source reports the fixed "vp_to_target" path label.
        src = self._make(slice="all", setup="probes_to_anchors")
        self.assertEqual(src.slice_id(), "all")
        self.assertEqual(src.setup_id(), "vp_to_target")
        self.assertEqual(self._make().setup_id(), "vp_to_target")
        self.assertEqual(src.name, "generic_presplit")

    def test_fit_samples_come_from_train_only(self) -> None:
        src = self._make()
        fit_target_lats = {fs.probe_coord.lat for fs in src.iter_fit_samples()}
        # Train targets have lat 40..43; test targets (44, 45) must not appear.
        self.assertEqual(fit_target_lats, {40.0, 41.0, 42.0, 43.0})
        self.assertEqual(len(list(src.iter_fit_samples())), 7)

    def test_eval_targets_come_from_test_only(self) -> None:
        src = self._make()
        eval_ids = {t.target_id for t in src.iter_eval_targets()}
        self.assertEqual(eval_ids, {"2001", "2002", "2003"})

    def test_eval_obs_grouping_and_weights(self) -> None:
        src = self._make()
        by_id = {t.target_id: t for t in src.iter_eval_targets()}
        self.assertEqual(len(by_id["2001"].obs), 3)
        self.assertEqual(len(by_id["2002"].obs), 2)
        self.assertEqual(sorted(by_id["2001"].obs_weights), [1.0, 6.0, 8.0])

    def test_vp_configs_are_union_of_both_files(self) -> None:
        src = self._make()
        vp_ids = {vp.vp_id for vp in src.iter_vp_configs()}
        # 3.3.3.3 only exists in the test file — still cataloged.
        self.assertEqual(vp_ids, {"1.1.1.1", "2.2.2.2", "3.3.3.3"})

    def test_tg_configs_are_union_of_both_files(self) -> None:
        src = self._make()
        tg_ids = {t.tg_id for t in src.iter_tg_configs()}
        self.assertEqual(
            tg_ids, {"1001", "1002", "1003", "1004", "2001", "2002", "2003"}
        )

    def test_iterators_are_pure(self) -> None:
        src = self._make()
        a = [t.target_id for t in src.iter_eval_targets()]
        b = [t.target_id for t in src.iter_eval_targets()]
        self.assertEqual(a, b)


class TestGenericPresplitSource_Validation(_PresplitBase):
    def test_target_overlap_is_hard_error(self) -> None:
        # Append a train-side target to the test file.
        leaky = _TEST_CSV + (
            "1.1.1.1,33.0,-84.0,20473,US,1001,40.0,-100.0,7922,US,Denver,9.0,1.0\n"
        )
        self.test_path.write_text(leaky)
        src = self._make()
        with self.assertRaisesRegex(ValueError, "BOTH"):
            list(src.iter_eval_targets())

    def test_vp_coord_conflict_is_hard_error(self) -> None:
        conflicted = _TEST_CSV.replace(
            "2.2.2.2,47.0,-122.0", "2.2.2.2,48.0,-122.0"
        )
        self.test_path.write_text(conflicted)
        src = self._make()
        with self.assertRaisesRegex(ValueError, "conflicting coordinates"):
            list(src.iter_eval_targets())

    def test_vp_coord_conflict_tolerates_float_precision_noise(self) -> None:
        # Sub-tolerance drift (float32/text round-trip noise) must NOT trip
        # the guardrail — this is the same VP, not a genuine conflict.
        near = _TEST_CSV.replace(
            "2.2.2.2,47.0,-122.0", "2.2.2.2,47.0000001,-122.0000001"
        )
        self.test_path.write_text(near)
        src = self._make()
        # Should not raise.
        self.assertEqual(
            {t.target_id for t in src.iter_eval_targets()}, {"2001", "2002", "2003"}
        )

    def test_missing_paths_raise(self) -> None:
        with self.assertRaises(ValueError):
            GenericPresplitSource(slice="all", train_path=self.train_path)
        with self.assertRaises(ValueError):
            GenericPresplitSource(slice="all", test_path=self.test_path)

    def test_unknown_slice_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make(slice="fold_0")
        with self.assertRaises(ValueError):
            self._make(slice="wsplit20")
        with self.assertRaises(ValueError):
            self._make(slice="garbage")

    def test_any_setup_value_is_ignored(self) -> None:
        # No ALLOWED_SETUPS validation — the axis doesn't exist for this
        # source, so arbitrary values are accepted and discarded.
        self.assertEqual(self._make(setup="bogus").setup_id(), "vp_to_target")

    def test_unsupported_extension_raises(self) -> None:
        bad = self.root / "train.tsv"
        bad.write_text(_TRAIN_CSV)
        src = self._make(train_path=bad)
        with self.assertRaisesRegex(ValueError, "unsupported extension"):
            list(src.iter_fit_samples())

    def test_missing_required_column_raises(self) -> None:
        df = pd.read_csv(self.test_path).drop(columns=["rtt_ms"])
        df.to_csv(self.test_path, index=False)
        src = self._make()
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            list(src.iter_eval_targets())


class TestGenericPresplitSource_Parquet(_PresplitBase):
    def test_mixed_parquet_train_csv_test(self) -> None:
        pq_train = self.root / "train.parquet"
        pd.read_csv(self.train_path).to_parquet(pq_train, index=False)
        src = self._make(train_path=pq_train)
        self.assertEqual(len(list(src.iter_fit_samples())), 7)
        self.assertEqual(
            {t.target_id for t in src.iter_eval_targets()},
            {"2001", "2002", "2003"},
        )

    def test_both_parquet(self) -> None:
        pq_train = self.root / "train.parquet"
        pq_test = self.root / "test.parquet"
        pd.read_csv(self.train_path).to_parquet(pq_train, index=False)
        pd.read_csv(self.test_path).to_parquet(pq_test, index=False)
        src = self._make(train_path=pq_train, test_path=pq_test)
        by_id = {t.target_id: t for t in src.iter_eval_targets()}
        self.assertEqual(set(by_id), {"2001", "2002", "2003"})
        self.assertEqual(len(by_id["2001"].obs), 3)


class TestGenericPresplitSource_Filters(_PresplitBase):
    def test_head_slice_limits_test_side_only(self) -> None:
        src = self._make(slice="head1")
        self.assertEqual(
            {t.target_id for t in src.iter_eval_targets()}, {"2001"}
        )
        # Train side untouched.
        self.assertEqual(len(list(src.iter_fit_samples())), 7)

    def test_min_obs_applies_per_file(self) -> None:
        # min_obs=2: train target 1004 (1 obs) dropped from the fit side,
        # test target 2003 (1 obs) dropped from the eval side — independently.
        src = self._make(min_obs=2)
        fit_lats = {fs.probe_coord.lat for fs in src.iter_fit_samples()}
        self.assertNotIn(43.0, fit_lats)  # 1004 dropped from train
        self.assertEqual(
            {t.target_id for t in src.iter_eval_targets()}, {"2001", "2002"}
        )

    def test_min_obs_wiping_a_side_raises(self) -> None:
        # Every train target has <= 2 obs, so min_obs=3 empties the fit
        # corpus — that must be loud, not a silent no-training run.
        src = self._make(min_obs=3)
        with self.assertRaisesRegex(ValueError, "zero train targets"):
            list(src.iter_eval_targets())

    def test_eval_pair_weight_min_masks_targets_and_obs(self) -> None:
        # thr=1.0: 2002's obs (0.5, 0.25) all fail → 2002 dropped entirely;
        # 2001 keeps all 3 obs (8.0, 6.0, 1.0).
        src = self._make(eval_pair_weight_min=1.0)
        targets = list(src.iter_eval_targets())
        self.assertEqual({t.target_id for t in targets}, {"2001"})
        self.assertEqual(len(targets[0].obs), 3)
        # thr=5.0: 2001 keeps only the two heavy obs.
        src5 = self._make(eval_pair_weight_min=5.0)
        targets5 = list(src5.iter_eval_targets())
        self.assertEqual({t.target_id for t in targets5}, {"2001"})
        self.assertEqual(sorted(targets5[0].obs_weights), [6.0, 8.0])
        # Train side untouched either way.
        self.assertEqual(len(list(src5.iter_fit_samples())), 7)

    def test_eval_pair_weight_min_emptying_eval_raises(self) -> None:
        src = self._make(eval_pair_weight_min=100.0)
        with self.assertRaisesRegex(ValueError, "zero eval targets"):
            list(src.iter_eval_targets())

    def test_eval_kept_traffic_fraction_derives_threshold(self) -> None:
        # Test-side (vp_id, target_city) pairs dedupe to per-pair max weight:
        # (1.1.1.1, Fargo)=8.0, (2.2.2.2, Fargo)=6.0, (3.3.3.3, Fargo)=1.0.
        # Total 15; frac=0.5 → cumsum hits 7.5 at the first pair → thr=8.0.
        src = self._make(eval_kept_traffic_fraction=0.5)
        targets = list(src.iter_eval_targets())
        self.assertEqual(src._eval_pair_weight_min, 8.0)
        self.assertEqual({t.target_id for t in targets}, {"2001"})
        self.assertEqual(targets[0].obs_weights, [8.0])

    def test_mutually_exclusive_weight_kwargs_raise(self) -> None:
        with self.assertRaises(ValueError):
            self._make(eval_pair_weight_min=1.0, eval_kept_traffic_fraction=0.5)
        with self.assertRaises(ValueError):
            self._make(eval_kept_traffic_fraction=1.5)
        with self.assertRaises(ValueError):
            self._make(eval_pair_weight_min=-1.0)


if __name__ == "__main__":
    unittest.main()
