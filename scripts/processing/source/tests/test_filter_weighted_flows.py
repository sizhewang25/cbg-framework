"""Invariants of the keyless flow-level traffic filter.

Two things are load-bearing here and easy to regress:

  * the descending-cumsum boundary (`side="left"` plus the `min(idx, len-1)`
    clamp), and
  * agreement with `GenericCSVSource._derive_eval_weight_min_from_fraction` —
    if the two drift, the dataset-characterisation figures describe a different
    subset than the benchmark actually scores.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.processing.source.filter_weighted_flows import (
    _default_output,
    _default_summary,
    derive_flow_weight_threshold,
    filter_flows,
    resolve_column,
)


def _mesh(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(vp_id, target_id, weight) -> a canonical-schema frame."""
    vps = {v: i for i, v in enumerate(dict.fromkeys(r[0] for r in rows))}
    tgs = {t: i for i, t in enumerate(dict.fromkeys(r[1] for r in rows))}
    return pd.DataFrame({
        "vp_id": [r[0] for r in rows],
        "vp_lat": [30.0 + vps[r[0]] for r in rows],
        "vp_lon": [-80.0 - vps[r[0]] for r in rows],
        "target_id": [r[1] for r in rows],
        "target_lat": [40.0 + tgs[r[1]] for r in rows],
        "target_lon": [-100.0 - tgs[r[1]] for r in rows],
        "rtt_ms": [10.0 + i for i in range(len(rows))],
        "weight": [r[2] for r in rows],
    })


def _threshold(weights: list[float], frac: float) -> float:
    return derive_flow_weight_threshold(np.array(weights, dtype=float), frac)[0]


class TestThresholdKernel(unittest.TestCase):
    def test_simple_prefix(self) -> None:
        # total 20; 0.75*20 = 15; cum [10, 15, 18, 20] clears at index 1 -> 5.
        self.assertEqual(_threshold([10, 5, 3, 2], 0.75), 5.0)

    def test_exact_cumsum_hit_takes_no_extra_element(self) -> None:
        # 0.5*20 = 10 == cum[0]; side="left" must not advance to index 1.
        self.assertEqual(_threshold([10, 5, 3, 2], 0.5), 10.0)

    def test_fraction_of_one_is_identity(self) -> None:
        """Guards the min(idx, len-1) clamp: cum sums the sorted array while
        total sums the original, so float error can push searchsorted past
        the end."""
        weights = [0.1, 0.2, 0.30000000000000004, 0.7, 1.9]
        self.assertEqual(_threshold(weights, 1.0), min(weights))

    def test_weights_not_summing_to_one_renormalize(self) -> None:
        # total 0.9; 0.7*0.9 = 0.63; cum [0.4, 0.7, 0.9] clears at index 1.
        # Against a total of 1.0 the answer would be 0.4 instead.
        self.assertAlmostEqual(_threshold([0.4, 0.3, 0.2], 0.7), 0.3)

    def test_single_flow(self) -> None:
        self.assertEqual(_threshold([5.0], 0.95), 5.0)

    def test_invalid_fraction_raises(self) -> None:
        for frac in (0.0, 1.5, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                _threshold([1.0, 2.0], frac)

    def test_all_zero_weights_raise(self) -> None:
        with self.assertRaises(ValueError):
            _threshold([0.0, 0.0, 0.0], 0.95)

    def test_all_equal_weights_raise(self) -> None:
        """Constant weights make `>=` retain 100% at every fraction, so the
        filter would be a silent no-op wearing a .traffic-weighted.csv name."""
        with self.assertRaises(ValueError):
            _threshold([5.0, 5.0, 5.0, 5.0], 0.5)


class TestFilterFlows(unittest.TestCase):
    def test_keeps_heavy_flows_and_reports_achieved(self) -> None:
        df = _mesh([("v1", "t1", 10), ("v2", "t1", 5), ("v1", "t2", 3), ("v2", "t2", 2)])
        kept, summary = filter_flows(df, kept_traffic_fraction=0.75)
        self.assertEqual(summary["eval_pair_weight_min"], 5.0)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(summary["kept_traffic_fraction_achieved"], 0.75)

    def test_no_dedup_flows_from_one_vp_count_separately(self) -> None:
        """Keyless: two flows from one VP are two flows, never collapsed."""
        df = _mesh([("v1", "t1", 6.0), ("v1", "t2", 4.0)])
        _, summary = filter_flows(df, kept_traffic_fraction=0.95)
        self.assertEqual(summary["flows_in"], 2)
        self.assertEqual(summary["eval_pair_weight_min"], 4.0)

    def test_target_fully_eliminated(self) -> None:
        df = _mesh([("v1", "t1", 100), ("v2", "t1", 90), ("v1", "t2", 0.1)])
        kept, summary = filter_flows(df, kept_traffic_fraction=0.9)
        self.assertNotIn("t2", set(kept["target_id"]))
        self.assertEqual(summary["targets_eliminated"], 1)
        self.assertEqual(summary["targets_surviving_ids"], ["t1"])

    def test_vp_fully_eliminated(self) -> None:
        df = _mesh([("v1", "t1", 100), ("v2", "t1", 90), ("v3", "t1", 0.1)])
        kept, summary = filter_flows(df, kept_traffic_fraction=0.9)
        self.assertNotIn("v3", set(kept["vp_id"]))
        self.assertEqual(summary["vps_eliminated_ids"], ["v3"])

    def test_surviving_target_can_drop_below_three_vps(self) -> None:
        """min_vp_obs=3 is NOT re-imposed — but it must be visible, because
        a 1-VP target is still scored downstream."""
        df = _mesh([
            ("v1", "t1", 100), ("v2", "t1", 0.1), ("v3", "t1", 0.1),
            ("v1", "t2", 90), ("v2", "t2", 80), ("v3", "t2", 70),
        ])
        _, summary = filter_flows(df, kept_traffic_fraction=0.9)
        self.assertEqual(summary["targets_out_by_vp_count"]["1"], 1)
        self.assertEqual(summary["targets_out_with_ge_3_vps"], 1)

    def test_preserves_columns_and_row_order(self) -> None:
        df = _mesh([("v1", "t1", 10), ("v2", "t1", 1), ("v1", "t2", 9), ("v2", "t2", 8)])
        kept, _ = filter_flows(df, kept_traffic_fraction=0.95)
        self.assertEqual(list(kept.columns), list(df.columns))
        self.assertEqual(list(kept["rtt_ms"]), sorted(kept["rtt_ms"]))

    def test_summary_is_json_native(self) -> None:
        df = _mesh([("v1", "t1", 10), ("v2", "t1", 1)])
        _, summary = filter_flows(df, kept_traffic_fraction=0.95)
        for key, value in summary.items():
            self.assertNotIsInstance(value, np.generic, msg=key)
        json.dumps(summary)  # must not raise on np.int64 / np.float64


class TestWeightHygiene(unittest.TestCase):
    def _df(self) -> pd.DataFrame:
        return _mesh([("v1", "t1", 10), ("v2", "t1", 1)])

    def test_missing_weight_column_raises(self) -> None:
        df = self._df().drop(columns=["weight"])
        with self.assertRaises(ValueError) as ctx:
            filter_flows(df, kept_traffic_fraction=0.95)
        message = str(ctx.exception)
        self.assertIn("weight", message)
        self.assertIn("vp_id", message)  # dumps the columns actually present

    def test_nan_weight_raises(self) -> None:
        df = self._df()
        df.loc[0, "weight"] = np.nan
        with self.assertRaises(ValueError):
            filter_flows(df, kept_traffic_fraction=0.95)

    def test_non_numeric_weight_raises(self) -> None:
        df = self._df().astype({"weight": object})
        df.loc[0, "weight"] = "heavy"
        with self.assertRaises(ValueError):
            filter_flows(df, kept_traffic_fraction=0.95)

    def test_negative_weight_raises(self) -> None:
        df = self._df()
        df.loc[0, "weight"] = -1.0
        with self.assertRaises(ValueError):
            filter_flows(df, kept_traffic_fraction=0.95)

    def test_conflicting_duplicate_flow_raises(self) -> None:
        """A value mask would keep the heavy row and drop the light one,
        splitting one physical flow across the filter boundary."""
        df = _mesh([("v1", "t1", 10), ("v1", "t1", 1), ("v2", "t2", 5)])
        with self.assertRaises(ValueError):
            filter_flows(df, kept_traffic_fraction=0.95)

    def test_identical_duplicate_flow_counted_once(self) -> None:
        df = _mesh([("v1", "t1", 4.0), ("v1", "t1", 4.0), ("v2", "t2", 6.0)])
        _, summary = filter_flows(df, kept_traffic_fraction=0.95)
        self.assertEqual(summary["flows_in"], 3)          # rows
        self.assertEqual(summary["weight_total"], 10.0)   # deduped flows


class TestColumnResolution(unittest.TestCase):
    def test_exact_match_wins(self) -> None:
        df = pd.DataFrame({"weight": [1.0]})
        self.assertEqual(resolve_column(df, "weight"), "weight")

    def test_unique_case_insensitive_match(self) -> None:
        df = pd.DataFrame({"WEIGHT": [1.0]})
        self.assertEqual(resolve_column(df, "weight"), "WEIGHT")

    def test_exact_match_beats_a_case_variant(self) -> None:
        """An exact hit is unambiguous even when a case variant exists."""
        df = pd.DataFrame({"weight": [1.0], "WEIGHT": [2.0]})
        self.assertEqual(resolve_column(df, "weight"), "weight")

    def test_ambiguous_case_insensitive_match_raises(self) -> None:
        """Two case variants and no exact hit — refuse to guess."""
        df = pd.DataFrame({"WEIGHT": [1.0], "Weight": [2.0]})
        with self.assertRaises(ValueError):
            resolve_column(df, "weight")

    def test_missing_column_lists_what_is_present(self) -> None:
        df = pd.DataFrame({"vp_id": ["v1"], "rtt_ms": [1.0]})
        with self.assertRaises(ValueError) as ctx:
            resolve_column(df, "weight")
        self.assertIn("rtt_ms", str(ctx.exception))

    def test_uppercase_header_is_not_rewritten(self) -> None:
        df = _mesh([("v1", "t1", 10), ("v2", "t1", 1)]).rename(
            columns={"vp_id": "VP_ID", "target_id": "TARGET_ID", "weight": "WEIGHT"}
        )
        kept, _ = filter_flows(df, kept_traffic_fraction=0.95)
        self.assertEqual(list(kept.columns), list(df.columns))


class TestDefaultPaths(unittest.TestCase):
    def test_multi_dot_stem_keeps_every_stage_tag(self) -> None:
        out = _default_output(Path("d/as02.mainland.sanitized.csv"))
        self.assertEqual(out.name, "as02.mainland.sanitized.traffic-weighted.csv")

    def test_summary_does_not_collide_with_the_soi_summary(self) -> None:
        out = _default_output(Path("d/as02.mainland.sanitized.csv"))
        summary = _default_summary(out)
        self.assertEqual(
            summary.name, "as02.mainland.sanitized.traffic-weighted.summary.json"
        )
        self.assertNotEqual(summary.name, "as02.mainland.sanitized.summary.json")


class TestAgreesWithGenericCSVSource(unittest.TestCase):
    """The script and the benchmark must derive the same threshold."""

    def test_thresholds_match(self) -> None:
        from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource

        df = _mesh([
            ("v1", "t1", 10.0), ("v2", "t1", 6.0), ("v3", "t1", 1.0),
            ("v1", "t2", 0.5), ("v2", "t2", 0.25), ("v3", "t3", 0.1),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.csv"
            df.to_csv(path, index=False)
            for frac in (0.5, 0.75, 0.95, 1.0):
                _, summary = filter_flows(df, kept_traffic_fraction=frac)
                src = GenericCSVSource(
                    slice="all", setup="anchors_to_probes", csv_path=path,
                    eval_kept_traffic_fraction=frac,
                )
                list(src.iter_eval_targets())
                self.assertEqual(
                    summary["eval_pair_weight_min"],
                    src._eval_pair_weight_min,
                    msg=f"disagreement at frac={frac}",
                )

    def test_surviving_targets_match_the_benchmarks_eval_roster(self) -> None:
        from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource

        df = _mesh([
            ("v1", "t1", 10.0), ("v2", "t1", 6.0),
            ("v1", "t2", 0.5), ("v2", "t2", 0.25),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.csv"
            df.to_csv(path, index=False)
            _, summary = filter_flows(df, kept_traffic_fraction=0.9)
            src = GenericCSVSource(
                slice="all", setup="anchors_to_probes", csv_path=path,
                eval_kept_traffic_fraction=0.9,
            )
            roster = {t.target_id for t in src.iter_eval_targets()}
        self.assertEqual(roster, set(summary["targets_surviving_ids"]))


if __name__ == "__main__":
    unittest.main()
