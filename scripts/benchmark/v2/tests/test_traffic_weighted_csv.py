"""TrafficWeightedCSVSource — the eval-side traffic mask, both modes.

The invariant the whole design rests on is in `TestModesAgree`: precomputed
(read the subset off a file) and on-the-fly (derive a threshold from the mesh)
must produce the *same* eval set at the same fraction, because both run the
identical keyless whole-mesh derivation.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark.v2.sources import SOURCES
from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource
from scripts.benchmark.v2.sources.traffic_weighted_csv import TrafficWeightedCSVSource


def _mesh_csv(rows: list[tuple[str, str, float, float]]) -> str:
    """(vp_id, target_id, rtt_ms, weight) -> a canonical-schema CSV body."""
    vps = {v: i for i, v in enumerate(dict.fromkeys(r[0] for r in rows))}
    tgs = {t: i for i, t in enumerate(dict.fromkeys(r[1] for r in rows))}
    head = ("vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms,weight")
    body = [
        f"{v},{30.0 + vps[v]},{-80.0 - vps[v]},{t},{40.0 + tgs[t]},"
        f"{-100.0 - tgs[t]},{rtt},{w}"
        for v, t, rtt, w in rows
    ]
    return "\n".join([head, *body]) + "\n"


_EVAL_MASK_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms,weight
    1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0,100
    2.2.2.2,47.0,-122.0,t1,40.0,-100.0,11.0,3
    1.1.1.1,33.0,-84.0,t2,41.0,-101.0,12.0,5
    1.1.1.1,33.0,-84.0,t3,42.0,-102.0,13.0,50
    2.2.2.2,47.0,-122.0,t3,42.0,-102.0,14.0,60
    1.1.1.1,33.0,-84.0,t4,43.0,-103.0,15.0,8
    2.2.2.2,47.0,-122.0,t4,43.0,-103.0,16.0,12
""").strip() + "\n"

_EVAL_MASK_SURVIVORS = {"t1", "t3", "t4"}


class TestTrafficWeightedCSV_ExplicitThreshold(unittest.TestCase):
    """Materialize-time traffic-weighted eval mask (`eval_pair_weight_min`):
    an eval target survives iff >= 1 obs clears the threshold, surviving
    targets keep only the clearing obs, and the fit side is untouched."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "eval_mask.csv"
        self.csv_path.write_text(_EVAL_MASK_CSV)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make(
        self,
        slice: str = "fold_0",
        eval_pair_weight_min: float | None = 10.0,
        **kwargs,
    ) -> TrafficWeightedCSVSource:
        return TrafficWeightedCSVSource(
            slice=slice, setup="anchors_to_probes",
            mesh_csv_path=self.csv_path, k=2,
            eval_pair_weight_min=eval_pair_weight_min,
            **kwargs,
        )

    def _unmasked(self, slice: str = "fold_0") -> GenericCSVSource:
        """The control arm. `TrafficWeightedCSVSource` cannot be built without a
        subset definition, so "same source, no mask" is `GenericCSVSource`."""
        return GenericCSVSource(
            slice=slice, setup="anchors_to_probes", csv_path=self.csv_path, k=2,
        )

    def test_fold_eval_set_is_unfiltered_eval_intersect_survivors(self) -> None:
        """Fold membership is DistGeo's business; the mask must only
        intersect the fold's eval set with the weight-clearing targets."""
        unfiltered = {t.target_id for t in self._unmasked().iter_eval_targets()}
        filtered = {t.target_id for t in self._make().iter_eval_targets()}
        self.assertEqual(filtered, unfiltered & _EVAL_MASK_SURVIVORS)

    def test_fit_side_untouched(self) -> None:
        """Full-mesh training: fit target set and fit sample stream are
        byte-identical with and without the mask. Dropped eval targets do
        NOT migrate into fit."""
        plain, masked = self._unmasked(), self._make()
        fit_plain = list(plain.iter_fit_samples())
        fit_masked = list(masked.iter_fit_samples())
        self.assertEqual(fit_plain, fit_masked)
        self.assertEqual(plain._fit_targets, masked._fit_targets)

    def test_surviving_targets_keep_only_clearing_obs(self) -> None:
        unfiltered_n_obs = {
            t.target_id: len(t.obs)
            for t in self._unmasked().iter_eval_targets()
        }
        expected_n_obs = {"t1": 1, "t3": 2, "t4": 1}  # at thr=10 (t2 dropped)
        for t in self._make().iter_eval_targets():
            assert t.obs_weights is not None
            self.assertEqual(len(t.obs), len(t.obs_weights))
            self.assertTrue(all(w >= 10.0 for w in t.obs_weights))
            self.assertEqual(len(t.obs), expected_n_obs[t.target_id])
            self.assertLessEqual(len(t.obs), unfiltered_n_obs[t.target_id])

    def test_mixed_target_loses_the_light_vp(self) -> None:
        """slice='all' makes every target an eval target, so the per-obs
        restriction is checkable deterministically: t1 keeps only its heavy
        flow's VP, t4 only its 12-weight VP, t2 vanishes."""
        by_id = {t.target_id: t for t in self._make(slice="all").iter_eval_targets()}
        self.assertEqual(set(by_id), _EVAL_MASK_SURVIVORS)
        self.assertEqual([str(vp) for vp, _, _ in by_id["t1"].obs], ["1.1.1.1"])
        self.assertEqual([str(vp) for vp, _, _ in by_id["t4"].obs], ["2.2.2.2"])
        self.assertEqual(len(by_id["t3"].obs), 2)

    def test_all_slice_fit_stream_stays_full(self) -> None:
        """On split-less slices the mask sets _eval_targets but leaves
        _fit_targets None — the fit stream still yields every CSV row."""
        src = self._make(slice="all")
        self.assertEqual(len(list(src.iter_fit_samples())), 7)

    def test_threshold_dropping_every_target_raises(self) -> None:
        src = self._make(eval_pair_weight_min=1000.0)
        with self.assertRaises(ValueError):
            list(src.iter_eval_targets())

    def test_negative_threshold_raises_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            self._make(eval_pair_weight_min=-1.0)

    def test_absent_weight_column_rejects_weighted_eval(self) -> None:
        """No weight column + a weighted-eval request is a hard error.

        The uniform 1.0 fill would otherwise make any threshold <= 1.0 retain
        100% of flows — a "weighted" run byte-identical to the unweighted one.
        """
        csv = textwrap.dedent("""
            vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms
            1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0
            1.1.1.1,33.0,-84.0,t2,41.0,-101.0,11.0
        """).strip() + "\n"
        path = Path(self.tmp.name) / "no_weight.csv"
        path.write_text(csv)
        # Previously the synthesized 1.0 fill made thr <= 1.0 a silent no-op:
        # a "traffic-weighted" run byte-identical to the unweighted one. Now it
        # refuses, because that artifact would be published as the weighted arm.
        for thr in (1.0, 2.0):
            src = TrafficWeightedCSVSource(
                slice="all", setup="anchors_to_probes",
                mesh_csv_path=path, eval_pair_weight_min=thr,
            )
            with self.assertRaises(ValueError) as ctx:
                list(src.iter_eval_targets())
            self.assertIn("weight", str(ctx.exception))

        # Without a weighted-eval request the 1.0 fill stays fine.
        unweighted = GenericCSVSource(
            slice="all", setup="anchors_to_probes", csv_path=path,
        )
        self.assertEqual(
            {t.target_id for t in unweighted.iter_eval_targets()}, {"t1", "t2"}
        )

    def test_min_obs_runs_before_the_mask(self) -> None:
        """min_obs=2 first drops single-obs targets (t2), then the mask
        restricts what's left — eval ends up exactly the multi-obs
        survivors, each with only clearing obs."""
        src = self._make(slice="all", min_obs=2)
        by_id = {t.target_id: t for t in src.iter_eval_targets()}
        self.assertEqual(set(by_id), {"t1", "t3", "t4"})
        for t in by_id.values():
            assert t.obs_weights is not None
            self.assertTrue(all(w >= 10.0 for w in t.obs_weights))


class TestTrafficWeightedCSV_KeptTrafficFraction(unittest.TestCase):
    """Keyless, whole-mesh derivation of eval_pair_weight_min from a fraction."""

    # Flow weights 10/1/9/1 (total 21). A city column is present precisely to
    # prove it is IGNORED — the derivation no longer keys on it.
    _CSV = textwrap.dedent("""
        vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,target_city,rtt_ms,weight
        1.1.1.1,33.0,-84.0,t1,40.0,-100.0,atlanta,10.0,10
        2.2.2.2,47.0,-122.0,t1,40.0,-100.0,atlanta,11.0,1
        1.1.1.1,33.0,-84.0,t2,41.0,-101.0,boston,12.0,9
        2.2.2.2,47.0,-122.0,t2,41.0,-101.0,boston,13.0,1
    """).strip() + "\n"

    # Same flows, no city column at all — must work identically.
    _CSV_NO_CITY = textwrap.dedent("""
        vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms,weight
        1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0,10
        2.2.2.2,47.0,-122.0,t1,40.0,-100.0,11.0,1
        1.1.1.1,33.0,-84.0,t2,41.0,-101.0,12.0,9
        2.2.2.2,47.0,-122.0,t2,41.0,-101.0,13.0,1
    """).strip() + "\n"

    # Fractional weights summing to 0.9, not 1.0 — the mesh is itself a sample,
    # so the cut must renormalize against the mesh total.
    _CSV_FRACTIONAL = textwrap.dedent("""
        vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms,weight
        1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0,0.4
        2.2.2.2,47.0,-122.0,t2,41.0,-101.0,11.0,0.3
        3.3.3.3,51.0,-114.0,t3,42.0,-102.0,12.0,0.2
    """).strip() + "\n"

    _CSV_ZERO = textwrap.dedent("""
        vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms,weight
        1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0,0
        2.2.2.2,47.0,-122.0,t2,41.0,-101.0,11.0,0
    """).strip() + "\n"

    _CSV_NO_WEIGHT = textwrap.dedent("""
        vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms
        1.1.1.1,33.0,-84.0,t1,40.0,-100.0,10.0
        2.2.2.2,47.0,-122.0,t2,41.0,-101.0,11.0
    """).strip() + "\n"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "eval_kept_frac.csv"
        self.csv_path.write_text(self._CSV)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, body: str) -> Path:
        path = Path(self.tmp.name) / name
        path.write_text(body)
        return path

    def _src(self, path: Path, **kw) -> GenericCSVSource:
        return TrafficWeightedCSVSource(
            slice=kw.pop("slice", "all"), setup="anchors_to_probes",
            mesh_csv_path=path, **kw,
        )

    def test_fraction_derives_threshold_and_filters_eval_only(self) -> None:
        src = self._src(self.csv_path, eval_kept_traffic_fraction=0.95)
        eval_targets = {t.target_id: t for t in src.iter_eval_targets()}
        # Flow weights [10, 9, 1, 1] desc, total 21; 0.95*21 = 19.95, cum
        # [10, 19, 20, 21] first reaches it at index 2 -> threshold 1.
        self.assertEqual(src._eval_pair_weight_min, 1.0)
        self.assertEqual(set(eval_targets), {"t1", "t2"})
        self.assertEqual(len(eval_targets["t1"].obs), 2)
        fit = list(src.iter_fit_samples())
        self.assertEqual(len(fit), 4)  # fit stays full-mesh under slice=all

    def test_derivation_ignores_target_city(self) -> None:
        """A city column must not change the answer — the key is the flow."""
        with_city = self._src(self.csv_path, eval_kept_traffic_fraction=0.8)
        list(with_city.iter_eval_targets())
        without = self._src(
            self._write("no_city.csv", self._CSV_NO_CITY),
            eval_kept_traffic_fraction=0.8,
        )
        list(without.iter_eval_targets())
        self.assertEqual(
            with_city._eval_pair_weight_min, without._eval_pair_weight_min
        )

    def test_no_target_city_column_is_fine(self) -> None:
        src = self._src(
            self._write("no_city2.csv", self._CSV_NO_CITY),
            eval_kept_traffic_fraction=0.95,
        )
        self.assertEqual(len(list(src.iter_eval_targets())), 2)
        self.assertEqual(src._eval_pair_weight_min, 1.0)

    def test_weights_not_summing_to_one_renormalize(self) -> None:
        """Mesh weights are shares of a larger universe; total here is 0.9."""
        src = self._src(
            self._write("frac.csv", self._CSV_FRACTIONAL),
            eval_kept_traffic_fraction=0.7,
        )
        list(src.iter_eval_targets())
        # total 0.9; 0.7*0.9 = 0.63; cum [0.4, 0.7, 0.9] reaches it at index 1
        # -> threshold 0.3. Against a total of 1.0 the answer would be 0.4.
        self.assertAlmostEqual(src._eval_pair_weight_min, 0.3)

    def test_threshold_is_fold_independent(self) -> None:
        """Whole-mesh normalization is what makes every fold agree."""
        thresholds = set()
        for fold in range(2):
            src = self._src(
                self.csv_path, slice=f"fold_{fold}", k=2,
                eval_kept_traffic_fraction=0.95,
            )
            list(src.iter_eval_targets())
            thresholds.add(src._eval_pair_weight_min)
        self.assertEqual(len(thresholds), 1)

    def test_fraction_of_one_keeps_every_flow(self) -> None:
        src = self._src(self.csv_path, eval_kept_traffic_fraction=1.0)
        obs = sum(len(t.obs) for t in src.iter_eval_targets())
        self.assertEqual(src._eval_pair_weight_min, 1.0)
        self.assertEqual(obs, 4)

    def test_zero_total_weight_raises(self) -> None:
        src = self._src(
            self._write("zero.csv", self._CSV_ZERO),
            eval_kept_traffic_fraction=0.95,
        )
        with self.assertRaises(ValueError):
            list(src.iter_eval_targets())

    def test_missing_weight_column_raises(self) -> None:
        """The 1.0 fill would make the mask a silent 100%-retention no-op."""
        src = self._src(
            self._write("no_weight.csv", self._CSV_NO_WEIGHT),
            eval_kept_traffic_fraction=0.95,
        )
        with self.assertRaises(ValueError) as ctx:
            list(src.iter_eval_targets())
        self.assertIn("weight", str(ctx.exception))

    def test_missing_weight_column_raises_for_explicit_threshold_too(self) -> None:
        src = self._src(
            self._write("no_weight2.csv", self._CSV_NO_WEIGHT),
            eval_pair_weight_min=0.5,
        )
        with self.assertRaises(ValueError):
            list(src.iter_eval_targets())

    def test_missing_weight_column_is_fine_when_unweighted(self) -> None:
        src = GenericCSVSource(
            slice="all", setup="anchors_to_probes",
            csv_path=self._write("no_weight3.csv", self._CSV_NO_WEIGHT),
        )
        self.assertEqual(len(list(src.iter_eval_targets())), 2)

    def test_fraction_and_explicit_threshold_together_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._src(
                self.csv_path,
                eval_pair_weight_min=1.0, eval_kept_traffic_fraction=0.95,
            )

    def test_invalid_fraction_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._src(self.csv_path, eval_kept_traffic_fraction=0.0)




# Six flows over three targets. Weights desc [10, 8, 5, 3, 2, 1], total 29.
_TWO_MODE_ROWS = [
    ("v1", "t1", 10.0, 10.0), ("v2", "t1", 11.0, 3.0),
    ("v1", "t2", 12.0, 8.0),  ("v2", "t2", 13.0, 2.0),
    ("v1", "t3", 14.0, 5.0),  ("v2", "t3", 15.0, 1.0),
]


class _TwoModeBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mesh = self.root / "mesh.csv"
        self.mesh.write_text(_mesh_csv(_TWO_MODE_ROWS))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_weighted(self, name: str, keys: list[tuple[str, str]]) -> Path:
        """A weighted CSV holding exactly `keys`, carved out of the mesh."""
        df = pd.read_csv(self.mesh)
        mask = df.apply(lambda r: (r["vp_id"], r["target_id"]) in set(keys), axis=1)
        path = self.root / name
        df[mask].to_csv(path, index=False)
        return path

    def _derive_weighted(self, frac: float, name: str = "weighted.csv") -> Path:
        """Generate the weighted CSV the way the pipeline does, so the two
        modes are compared on a genuinely shared derivation."""
        from scripts.processing.source.filter_weighted_flows import filter_flows

        kept, _ = filter_flows(pd.read_csv(self.mesh), kept_traffic_fraction=frac)
        path = self.root / name
        kept.to_csv(path, index=False)
        return path

    def _src(self, **kwargs) -> TrafficWeightedCSVSource:
        kwargs.setdefault("slice", "all")
        return TrafficWeightedCSVSource(
            setup="anchors_to_probes", mesh_csv_path=self.mesh, **kwargs
        )


class TestModesAgree(_TwoModeBase):
    """The invariant the design rests on."""

    def _roster(self, src) -> dict:
        return {
            t.target_id: sorted((str(v), round(float(w), 12))
                                for (v, _, _), w in zip(t.obs, t.obs_weights))
            for t in src.iter_eval_targets()
        }

    def test_precomputed_equals_on_the_fly_at_the_same_fraction(self) -> None:
        for frac in (0.5, 0.75, 0.95, 1.0):
            weighted = self._derive_weighted(frac, f"w{frac}.csv")
            otf = self._src(eval_kept_traffic_fraction=frac)
            pre = self._src(weighted_csv_path=weighted)
            self.assertEqual(self._roster(otf), self._roster(pre),
                             msg=f"modes disagree at frac={frac}")

    def test_agreement_holds_per_fold(self) -> None:
        weighted = self._derive_weighted(0.95)
        for fold in range(2):
            otf = self._src(slice=f"fold_{fold}", k=2, eval_kept_traffic_fraction=0.95)
            pre = self._src(slice=f"fold_{fold}", k=2, weighted_csv_path=weighted)
            self.assertEqual(self._roster(otf), self._roster(pre))

    def test_fit_matches_the_plain_mesh_in_both_modes(self) -> None:
        weighted = self._derive_weighted(0.95)
        plain = list(GenericCSVSource(
            slice="all", setup="anchors_to_probes", csv_path=self.mesh,
        ).iter_fit_samples())
        for src in (self._src(eval_kept_traffic_fraction=0.95),
                    self._src(weighted_csv_path=weighted)):
            self.assertEqual(list(src.iter_fit_samples()), plain)


class TestPrecomputedMode(_TwoModeBase):
    def test_ignores_mesh_weights_entirely(self) -> None:
        """Membership, not thresholds: perturbing the mesh's weights without
        regenerating the weighted CSV must change nothing."""
        weighted = self._write_weighted("w.csv", [("v1", "t1"), ("v1", "t2")])
        before = {t.target_id: len(t.obs)
                  for t in self._src(weighted_csv_path=weighted).iter_eval_targets()}

        scrambled = self.root / "scrambled.csv"
        df = pd.read_csv(self.mesh)
        df["weight"] = df["weight"].to_numpy()[::-1]
        df.to_csv(scrambled, index=False)
        after = {
            t.target_id: len(t.obs)
            for t in TrafficWeightedCSVSource(
                slice="all", setup="anchors_to_probes",
                mesh_csv_path=scrambled, weighted_csv_path=weighted,
            ).iter_eval_targets()
        }
        self.assertEqual(before, after)

    def test_works_on_a_mesh_with_no_weight_column(self) -> None:
        """Precomputed mode consults no weights, so the mesh needs none."""
        weighted = self._write_weighted("w.csv", [("v1", "t1"), ("v1", "t2")])
        bare = self.root / "bare_mesh.csv"
        pd.read_csv(self.mesh).drop(columns=["weight"]).to_csv(bare, index=False)
        src = TrafficWeightedCSVSource(
            slice="all", setup="anchors_to_probes",
            mesh_csv_path=bare, weighted_csv_path=weighted,
        )
        self.assertEqual({t.target_id for t in src.iter_eval_targets()}, {"t1", "t2"})

    def test_head_slice_does_not_trip_the_subset_guard(self) -> None:
        """`head<k>` drops rows from the mesh frame. Those flows are still in
        the mesh FILE, so the file-pair guard must not fire -- they simply fall
        outside the slice."""
        weighted = self._write_weighted("w.csv", [("v1", "t1"), ("v1", "t3")])
        src = self._src(slice="head1", weighted_csv_path=weighted)
        self.assertEqual({t.target_id for t in src.iter_eval_targets()}, {"t1"})

    def test_flow_absent_from_the_mesh_raises(self) -> None:
        """The stale/mismatched file-pair guard."""
        rogue = self.root / "rogue.csv"
        df = pd.read_csv(self.mesh)
        df.loc[0, "vp_id"] = "v-does-not-exist"
        df.to_csv(rogue, index=False)
        with self.assertRaises(ValueError) as ctx:
            list(self._src(weighted_csv_path=rogue).iter_eval_targets())
        self.assertIn("absent", str(ctx.exception))

    def test_duplicate_flow_in_the_weighted_csv_raises(self) -> None:
        dup = self.root / "dup.csv"
        df = pd.read_csv(self.mesh)
        pd.concat([df, df.head(1)]).to_csv(dup, index=False)
        with self.assertRaises(ValueError) as ctx:
            list(self._src(weighted_csv_path=dup).iter_eval_targets())
        self.assertIn("edge-unique", str(ctx.exception))

    def test_empty_weighted_csv_raises(self) -> None:
        empty = self.root / "empty.csv"
        pd.read_csv(self.mesh).head(0).to_csv(empty, index=False)
        with self.assertRaises(ValueError):
            list(self._src(weighted_csv_path=empty).iter_eval_targets())

    def test_missing_flow_key_columns_raise(self) -> None:
        bad = self.root / "bad.csv"
        pd.read_csv(self.mesh).drop(columns=["target_id"]).to_csv(bad, index=False)
        with self.assertRaises(ValueError):
            list(self._src(weighted_csv_path=bad).iter_eval_targets())


class TestModeSelection(_TwoModeBase):
    def test_no_subset_definition_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._src()
        self.assertIn("exactly once", str(ctx.exception))

    def test_two_subset_definitions_raise(self) -> None:
        weighted = self._write_weighted("w.csv", [("v1", "t1")])
        for kwargs in (
            {"weighted_csv_path": weighted, "eval_kept_traffic_fraction": 0.9},
            {"weighted_csv_path": weighted, "eval_pair_weight_min": 5.0},
            {"eval_kept_traffic_fraction": 0.9, "eval_pair_weight_min": 5.0},
        ):
            with self.assertRaises(ValueError):
                self._src(**kwargs)

    def test_mesh_csv_path_is_required(self) -> None:
        with self.assertRaises(ValueError):
            TrafficWeightedCSVSource(slice="all", eval_kept_traffic_fraction=0.9)

    def test_on_the_fly_needs_a_real_weight_column(self) -> None:
        bare = self.root / "bare.csv"
        pd.read_csv(self.mesh).drop(columns=["weight"]).to_csv(bare, index=False)
        src = TrafficWeightedCSVSource(
            slice="all", setup="anchors_to_probes",
            mesh_csv_path=bare, eval_kept_traffic_fraction=0.95,
        )
        with self.assertRaises(ValueError) as ctx:
            list(src.iter_eval_targets())
        self.assertIn("weight", str(ctx.exception))

    def test_min_obs_runs_before_the_mask(self) -> None:
        """A target pruned by min_obs must not reappear via the weighted set."""
        weighted = self._write_weighted("w.csv", [("v1", "t1"), ("v1", "t2")])
        sparse = self.root / "sparse.csv"
        df = pd.read_csv(self.mesh)
        df[~((df.target_id == "t2") & (df.vp_id == "v2"))].to_csv(sparse, index=False)
        src = TrafficWeightedCSVSource(
            slice="all", setup="anchors_to_probes", mesh_csv_path=sparse,
            weighted_csv_path=weighted, min_obs=2,
        )
        self.assertEqual({t.target_id for t in src.iter_eval_targets()}, {"t1"})


class TestGenericCSVRejectsWeightKwargs(unittest.TestCase):
    """Guards the split itself: the logic must not drift back."""

    def test_registry_has_both_sources(self) -> None:
        self.assertIn("generic_csv", SOURCES)
        self.assertIn("traffic_weighted_csv", SOURCES)

    def test_generic_csv_signature_has_no_weight_kwargs(self) -> None:
        import inspect

        params = inspect.signature(GenericCSVSource.__init__).parameters
        self.assertNotIn("eval_pair_weight_min", params)
        self.assertNotIn("eval_kept_traffic_fraction", params)

    def test_generic_csv_raises_on_weight_kwargs(self) -> None:
        with self.assertRaises(TypeError):
            GenericCSVSource(slice="all", csv_path="x.csv", eval_pair_weight_min=1.0)
