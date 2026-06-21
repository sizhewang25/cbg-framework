"""DataSource adapter tests.

GenericCSVSource is exercised against a synthetic in-memory canonical-schema
CSV (no dependency on any real CSV being present on disk). RipeAtlasSource is
exercised only for its JSON coord-loading half — ClickHouse is mocked out.
"""

from __future__ import annotations

import json as _json
import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.benchmark.v2.sources import SOURCES
from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.benchmark.v2.sources.ripe_atlas_asn_corpora import (
    RipeAtlasASNCorporaSource,
)
from scripts.processing.ripe_atlas.continents import continent_of


def _write_stratification(
    path: Path,
    anchor_ips: list[str],
    k: int = 2,
    policy_class: str = "DistGeoStratification",
    seed: int = 42,
) -> Path:
    """Write a minimal stratification JSON (the shape `stratify.py` emits).

    Round-robins `anchor_ips` into K folds in input order, so the caller
    controls which anchors land where. Used by P2A test fixtures, where
    `RipeAtlasSource(slice="fold_N")` requires a stratification file."""
    sorted_ips = list(anchor_ips)
    assignments = {ip: i % k for i, ip in enumerate(sorted_ips)}
    fold_sizes = [sum(1 for f in assignments.values() if f == i) for i in range(k)]
    payload = {
        "policy": {
            "class": policy_class,
            "kind": "dist_geo_kfold" if policy_class == "DistGeoStratification" else "sechidis_kfold",
            "k": k, "seed": seed, "asn_bucket_top_n": 20,
        },
        "corpus": {"source": "test", "n_anchors_yielded": len(assignments)},
        "generated_at": "2026-05-24T00:00:00+00:00",
        "fold_sizes": fold_sizes,
        "fold_assignments": assignments,
    }
    path.write_text(_json.dumps(payload))
    return path


# Canonical-schema synth CSV: vp_* = anchor side (acting as VP),
# target_* = probe side (the entity being geolocated). 2 unique VPs ×
# 6 unique targets = 12 rows, mirroring the original Vultr layout.
_SYNTH_CSV = textwrap.dedent("""
    vp_id,vp_lat,vp_lon,vp_asn,vp_country,target_id,target_lat,target_lon,target_asn,target_country,rtt_ms
    1.1.1.1,33.0,-84.0,20473,US,1001,40.0,-100.0,7922,US,10.0
    1.1.1.1,33.0,-84.0,20473,US,1002,41.0,-101.0,7922,US,12.0
    1.1.1.1,33.0,-84.0,20473,US,1003,42.0,-102.0,3356,US,13.0
    1.1.1.1,33.0,-84.0,20473,US,1004,43.0,-103.0,3356,US,14.0
    1.1.1.1,33.0,-84.0,20473,US,1005,44.0,-104.0,7018,US,15.0
    1.1.1.1,33.0,-84.0,20473,US,1006,45.0,-105.0,7018,US,16.0
    2.2.2.2,47.0,-122.0,40,US,1001,40.0,-100.0,7922,US,11.0
    2.2.2.2,47.0,-122.0,40,US,1002,41.0,-101.0,7922,US,11.5
    2.2.2.2,47.0,-122.0,40,US,1003,42.0,-102.0,3356,US,12.5
    2.2.2.2,47.0,-122.0,40,US,1004,43.0,-103.0,3356,US,13.5
    2.2.2.2,47.0,-122.0,40,US,1005,44.0,-104.0,7018,US,14.5
    2.2.2.2,47.0,-122.0,40,US,1006,45.0,-105.0,7018,US,15.5
""").strip() + "\n"


class TestGenericCSVSource_Stratified(unittest.TestCase):
    """Stratified `fold_N` slice path — the leakage-free contract under
    GenericCSVSource. Uses the canonical-schema synthetic CSV."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "canonical_smoke.csv"
        self.csv_path.write_text(_SYNTH_CSV)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make(self, slice: str = "fold_0", k: int = 2) -> GenericCSVSource:
        return GenericCSVSource(
            slice=slice, setup="anchors_to_probes",
            csv_path=self.csv_path, k=k,
        )

    def test_slice_id_and_setup_id_round_trip(self) -> None:
        src = self._make(slice="fold_1", k=2)
        self.assertEqual(src.slice_id(), "fold_1")
        self.assertEqual(src.setup_id(), "anchors_to_probes")
        self.assertEqual(src.name, "generic_csv")

    def test_vp_configs_yield_unique_vps(self) -> None:
        src = self._make()
        vps = list(src.iter_vp_configs())
        # 2 unique vp_ids in the synthetic CSV.
        self.assertEqual(len(vps), 2)
        self.assertEqual({vp.vp_id for vp in vps}, {"1.1.1.1", "2.2.2.2"})

    def test_tg_configs_yield_all_unique_targets(self) -> None:
        """The static catalog (tg_configs.parquet) covers eval ∪ fit — i.e.
        every unique target_id in the CSV, regardless of fold."""
        src = self._make()
        tg_ids = {t.tg_id for t in src.iter_tg_configs()}
        self.assertEqual(tg_ids, {"1001", "1002", "1003", "1004", "1005", "1006"})

    def test_fold_eval_fit_disjoint(self) -> None:
        """The leakage-free contract: no target_id appears in both the eval
        target set and the fit sample set."""
        src = self._make()
        eval_ids = {t.target_id for t in src.iter_eval_targets()}
        assert src._eval_targets is not None and src._fit_targets is not None
        self.assertEqual(src._eval_targets, eval_ids)
        self.assertTrue(src._eval_targets.isdisjoint(src._fit_targets))
        self.assertGreater(len(src._eval_targets), 0)
        self.assertGreater(len(src._fit_targets), 0)

    def test_fold_partition_covers_all_targets(self) -> None:
        """fold_N + complement (cached on the instance) covers every target."""
        src = self._make()
        list(src.iter_tg_configs())  # force loading
        assert src._eval_targets is not None and src._fit_targets is not None
        all_targets = src._eval_targets | src._fit_targets
        self.assertEqual(all_targets, {"1001", "1002", "1003", "1004", "1005", "1006"})

    def test_fold_partition_deterministic(self) -> None:
        """Same (k, seed, asn_bucket_top_n) → identical eval set across runs."""
        src_a = self._make()
        src_b = self._make()
        list(src_a.iter_eval_targets())
        list(src_b.iter_eval_targets())
        self.assertEqual(src_a._eval_targets, src_b._eval_targets)

    def test_fit_sample_routes_canonical_columns(self) -> None:
        """`vp_*` columns always populate vp_coord; `target_*` columns always
        populate probe_coord (the v2 FitSample 'probe_coord' field is the
        ground-truth target coord, despite the historical name)."""
        src = self._make()
        sample = next(iter(src.iter_fit_samples()))
        # vp_* side: lat ∈ {33.0, 47.0}.
        self.assertIn(sample.vp_coord.lat, (33.0, 47.0))
        # target_* side: lat ∈ [40, 45] for the 6 unique targets.
        self.assertGreaterEqual(sample.probe_coord.lat, 40.0)
        self.assertLessEqual(sample.probe_coord.lat, 45.0)

    def test_eval_obs_count_matches_vp_count_per_target(self) -> None:
        """Each eval target carries one obs per VP that measured it."""
        src = self._make()
        targets = list(src.iter_eval_targets())
        for t in targets:
            self.assertEqual(len(t.obs), 2)

    def test_unknown_slice_raises(self) -> None:
        with self.assertRaises(ValueError):
            GenericCSVSource(slice="all_us", csv_path=self.csv_path, k=2)
        with self.assertRaises(ValueError):
            GenericCSVSource(slice="garbage", csv_path=self.csv_path, k=2)

    def test_fold_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            GenericCSVSource(slice="fold_5", csv_path=self.csv_path, k=2)

    def test_unknown_setup_raises(self) -> None:
        with self.assertRaises(ValueError):
            GenericCSVSource(
                slice="fold_0", setup="bogus",
                csv_path=self.csv_path, k=2,
            )

    def test_missing_csv_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            GenericCSVSource(slice="fold_0", csv_path=None)

    def test_all_slice_skips_stratification(self) -> None:
        """`slice='all'` yields every row from both iterators and leaves the
        partition caches as None (no stratification)."""
        src = GenericCSVSource(slice="all", csv_path=self.csv_path)
        fit_targets = {fs.probe_coord.lat for fs in src.iter_fit_samples()}
        eval_targets = {t.target_id for t in src.iter_eval_targets()}
        self.assertIsNone(src._eval_targets)
        self.assertIsNone(src._fit_targets)
        # 6 unique targets contribute fit samples (every row goes to both
        # iterators under no-stratification mode).
        self.assertEqual(len(eval_targets), 6)
        self.assertEqual(len(fit_targets), 6)


class TestRipeAtlasSourceCoordLoad(unittest.TestCase):
    def test_load_coords_from_synthetic_json(self) -> None:
        # _load_coords runs before _apply_holdout, so the partition file
        # doesn't need to exist for this construction — we just need a
        # path argument to satisfy the P2A signature.
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {
                    "address_v4": "1.1.1.1",
                    "asn_v4": 7922, "country_code": "US",
                    "geometry": {"coordinates": [-84.0, 33.0]},   # lon, lat
                    "is_anchor": True,
                },
                {
                    "address_v4": "2.2.2.2",
                    "asn_v4": 3356, "country_code": "US",
                    "geometry": {"coordinates": [-100.0, 40.0]},
                    "is_anchor": False,
                },
            ]))
            src = RipeAtlasSource(
                slice="fold_0",
                probes_and_anchors_file=probes_json,
                stratification_path=Path(tmp) / "stratification.json",
            )
            src._load_coords()
            assert src._coords_by_ip is not None
            self.assertEqual(len(src._coords_by_ip), 2)
            self.assertAlmostEqual(src._coords_by_ip["1.1.1.1"].lat, 33.0)
            self.assertAlmostEqual(src._coords_by_ip["1.1.1.1"].lon, -84.0)
            self.assertEqual(src._anchor_ips, {"1.1.1.1"})

    def test_anchors_to_probes_setup_transposes_groups(self) -> None:
        """In anchors_to_probes mode, EvalTargets are probes, not anchors.
        Fold semantics don't apply to A2P, so slice is just a label."""
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"address_v4": "1.1.1.1", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
                {"address_v4": "vp-a", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-100.0, 40.0]}, "is_anchor": False},
                {"address_v4": "vp-b", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-101.0, 41.0]}, "is_anchor": False},
            ]))
            src = RipeAtlasSource(
                slice="all", setup="anchors_to_probes",
                probes_and_anchors_file=probes_json,
                rtt_query=lambda *a, **kw: {"1.1.1.1": {"vp-a": [10.0], "vp-b": [12.0]}},
                sanitize=False,
            )
            targets = list(src.iter_eval_targets())
        # 1 anchor × 2 probes → 2 EvalTargets (one per probe), each with the
        # single anchor as a VP.
        self.assertEqual(len(targets), 2)
        target_ids = {t.target_id for t in targets}
        self.assertEqual(target_ids, {"vp-a", "vp-b"})
        for t in targets:
            self.assertEqual(len(t.obs), 1)
            self.assertEqual(str(t.obs[0][0]), "1.1.1.1")

    def test_iter_eval_targets_groups_by_anchor(self) -> None:
        """Inject a fake rtt_query so the test doesn't require ClickHouse.
        Two anchors split round-robin into K=2 folds; slice='fold_0' picks
        one as the eval target."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            probes_json = tmp_path / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"address_v4": "1.1.1.1", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
                {"address_v4": "1.1.1.2", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-85.0, 34.0]}, "is_anchor": True},
                {"address_v4": "vp-a", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-100.0, 40.0]}, "is_anchor": False},
                {"address_v4": "vp-b", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-101.0, 41.0]}, "is_anchor": False},
            ]))
            stratification_path = _write_stratification(
                tmp_path / "stratification.json", ["1.1.1.1", "1.1.1.2"], k=2,
            )
            src = RipeAtlasSource(
                slice="fold_0",
                probes_and_anchors_file=probes_json,
                stratification_path=stratification_path,
                rtt_query=lambda *a, **kw: {
                    "1.1.1.1": {"vp-a": [10.0], "vp-b": [12.0]},
                    "1.1.1.2": {"vp-a": [11.0], "vp-b": [13.0]},
                },
                sanitize=False,
            )
            targets = list(src.iter_eval_targets())
        # fold_0 contains the lexicographically-first of the two anchors.
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].target_id, "1.1.1.1")
        self.assertEqual(len(targets[0].obs), 2)
        self.assertAlmostEqual(targets[0].true_coord.lat, 33.0)


class TestRipeAtlasAnchorCityEnrichment(unittest.TestCase):
    """`anchor_city.json` → VpConfig.city wiring (ANCHORS_TO_PROBES surfaces
    cities on VPs since anchors *are* the VPs in that setup)."""

    def test_city_populated_from_nominatim_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"id": 6025, "address_v4": "1.1.1.1", "asn_v4": 8839,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.7485, 48.5795]}, "is_anchor": True},
                {"id": 7777, "address_v4": "9.9.9.9", "asn_v4": 1,
                 "country_code": "US",
                 "geometry": {"coordinates": [-100.0, 40.0]}, "is_anchor": True},
                {"id": 1234, "address_v4": "vp-x", "asn_v4": 1,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.0, 48.0]}, "is_anchor": False},
            ]))
            city_json = Path(tmp) / "anchor_city.json"
            city_json.write_text(_json.dumps({
                # Full Nominatim shape — proves _extract_city walks the
                # features[0].properties.address.city path correctly.
                "6025": {"features": [{"properties": {"address": {"city": "Strasbourg"}}}]},
                # Fallback chain: no `city`, falls through to `village`.
                "7777": {"features": [{"properties": {"address": {"village": "Smalltown"}}}]},
                # An anchor id with no matching probe entry — should be harmless.
                "9999": {"features": [{"properties": {"address": {"city": "Ghost"}}}]},
            }))
            src = RipeAtlasSource(
                slice="all", setup="anchors_to_probes",
                probes_and_anchors_file=probes_json,
                anchor_city_file=city_json,
                rtt_query=lambda *a, **kw: {
                    "1.1.1.1": {"vp-x": [10.0]},
                    "9.9.9.9": {"vp-x": [12.0]},
                },
                sanitize=False,
            )
            vps = {v.vp_id: v for v in src.iter_vp_configs()}

        self.assertEqual(vps["1.1.1.1"].city, "Strasbourg")
        self.assertEqual(vps["9.9.9.9"].city, "Smalltown")  # fallback chain
        # vp-x is a probe (no anchor city entry) → city stays None.
        # In anchors_to_probes vp-x isn't a VP, but in probes_to_anchors it
        # would be — either way its city should be None.

    def test_missing_city_file_is_silently_skipped(self) -> None:
        """No anchor_city.json → cities are all None, no crash."""
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"id": 1, "address_v4": "1.1.1.1", "asn_v4": 1,
                 "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
            ]))
            src = RipeAtlasSource(
                slice="all", setup="anchors_to_probes",
                probes_and_anchors_file=probes_json,
                anchor_city_file=Path(tmp) / "does_not_exist.json",
                rtt_query=lambda *a, **kw: {"1.1.1.1": {"1.1.1.1": [1.0]}},
                sanitize=False,
            )
            vps = list(src.iter_vp_configs())
        self.assertEqual(len(vps), 1)
        self.assertIsNone(vps[0].city)


class TestTgConfigsEmission(unittest.TestCase):
    """tg_configs.parquet rows: GenericCSV side. Targets are always the
    target_* side of the canonical schema."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "canonical_smoke.csv"
        self.csv_path.write_text(_SYNTH_CSV)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_targets_emitted_with_canonical_schema(self) -> None:
        src = GenericCSVSource(
            slice="fold_0", setup="anchors_to_probes",
            csv_path=self.csv_path, k=2,
        )
        tgs = list(src.iter_tg_configs())
        # 6 unique target_ids; city is always None (no city column in canonical).
        self.assertEqual(len(tgs), 6)
        for t in tgs:
            self.assertIsNone(t.city)
        # Spot-check one: target_id 1001 → (40, -100), target_asn 7922
        t1001 = next(t for t in tgs if t.tg_id == "1001")
        self.assertAlmostEqual(t1001.lat, 40.0)
        self.assertEqual(t1001.asn, 7922)


class TestTgConfigsParquetWriter(unittest.TestCase):
    """materialize_inputs() writes tg_configs.parquet with the declared schema
    and the per-setup row count (one row per unique target)."""

    def test_writes_parquet_with_expected_schema_and_rows(self) -> None:
        import pyarrow.parquet as pq
        from scripts.benchmark.v2.inputs import materialize_inputs
        from scripts.benchmark.v2 import schema as bench_schema

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv = tmp_path / "synth.csv"
            csv.write_text(_SYNTH_CSV)
            src = GenericCSVSource(
                slice="fold_0", setup="anchors_to_probes",
                csv_path=csv, k=2,
            )
            out_dir = materialize_inputs(src, root=tmp_path / "inputs", run_id="test-run")
            tg_path = out_dir / "tg_configs.parquet"
            self.assertTrue(tg_path.exists())

            table = pq.read_table(tg_path)
            self.assertEqual(table.schema, bench_schema.TG_CONFIGS_SCHEMA)
            # 6 unique targets → 6 tg_config rows (catalog covers eval ∪ fit).
            self.assertEqual(table.num_rows, 6)
            row_by_id = {r["tg_id"]: r for r in table.to_pylist()}
            self.assertIn("1001", row_by_id)
            # Canonical schema has no city column.
            self.assertIsNone(row_by_id["1001"]["city"])

            # Manifest reflects the new artifact's row count.
            manifest = (out_dir / "manifest.json").read_text()
            self.assertIn('"n_tg_configs": 6', manifest)


class TestRipeAtlasTgConfigs(unittest.TestCase):
    """RIPE Atlas side: anchors in PROBES_TO_ANCHORS, probes in ANCHORS_TO_PROBES,
    plus city propagation from anchor_city.json."""

    def test_probes_to_anchors_targets_are_anchors_with_city(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            probes_json = tmp_path / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"id": 6025, "address_v4": "anchor-a", "asn_v4": 8839,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.7485, 48.5795]}, "is_anchor": True},
                {"id": 6026, "address_v4": "anchor-b", "asn_v4": 8839,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.0, 48.0]}, "is_anchor": True},
                {"id": 999, "address_v4": "vp-x", "asn_v4": 1,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.0, 48.0]}, "is_anchor": False},
            ]))
            city_json = tmp_path / "anchor_city.json"
            city_json.write_text(_json.dumps({
                "6025": {"features": [{"properties": {"address": {"city": "Strasbourg"}}}]},
            }))
            stratification_path = _write_stratification(
                tmp_path / "stratification.json", ["anchor-a", "anchor-b"], k=2,
            )
            src = RipeAtlasSource(
                slice="fold_0",
                probes_and_anchors_file=probes_json,
                anchor_city_file=city_json,
                stratification_path=stratification_path,
                rtt_query=lambda *a, **kw: {
                    "anchor-a": {"vp-x": [10.0]},
                    "anchor-b": {"vp-x": [11.0]},
                },
                sanitize=False,
            )
            tgs = list(src.iter_tg_configs())
        # tg_configs surfaces every active anchor regardless of fold (the fold
        # split only filters fit_samples / eval_targets).
        self.assertEqual({t.tg_id for t in tgs}, {"anchor-a", "anchor-b"})
        anchor_a = next(t for t in tgs if t.tg_id == "anchor-a")
        self.assertEqual(anchor_a.city, "Strasbourg")

    def test_anchors_to_probes_targets_are_probes_without_city(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"id": 6025, "address_v4": "anchor-a", "asn_v4": 8839,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.7485, 48.5795]}, "is_anchor": True},
                {"id": 100, "address_v4": "vp-a", "asn_v4": 7922,
                 "country_code": "US",
                 "geometry": {"coordinates": [-100.0, 40.0]}, "is_anchor": False},
                {"id": 101, "address_v4": "vp-b", "asn_v4": 7922,
                 "country_code": "US",
                 "geometry": {"coordinates": [-101.0, 41.0]}, "is_anchor": False},
            ]))
            city_json = Path(tmp) / "anchor_city.json"
            city_json.write_text(_json.dumps({
                "6025": {"features": [{"properties": {"address": {"city": "Strasbourg"}}}]},
            }))
            src = RipeAtlasSource(
                slice="all", setup="anchors_to_probes",
                probes_and_anchors_file=probes_json,
                anchor_city_file=city_json,
                rtt_query=lambda *a, **kw: {"anchor-a": {"vp-a": [10.0], "vp-b": [12.0]}},
                sanitize=False,
            )
            tgs = {t.tg_id: t for t in src.iter_tg_configs()}
        # Anchor is the sole VP, probes are targets — neither probe has a city.
        self.assertEqual(set(tgs), {"vp-a", "vp-b"})
        self.assertIsNone(tgs["vp-a"].city)
        self.assertIsNone(tgs["vp-b"].city)
        self.assertEqual(tgs["vp-a"].asn, 7922)


class TestSourceRegistry(unittest.TestCase):
    def test_known_sources_registered(self) -> None:
        self.assertIn("generic_csv", SOURCES)
        self.assertIn("ripe_atlas", SOURCES)
        self.assertIn("ripe_atlas_asn_corpora", SOURCES)


class TestRipeAtlasSourceStratification(unittest.TestCase):
    """End-to-end checks that the fold slice partitions iter_fit_samples and
    iter_eval_targets disjointly.

    The source consumes a precomputed stratification JSON; this fixture
    writes one over a 20-anchor / 20-probe synthetic corpus on the fly.
    Algorithm-level invariants live in
    test_sechidis_stratification.py / test_dist_geo_stratification.py."""

    @staticmethod
    def _write_stratification_for_anchors(
        path: Path, anchor_entries: list[dict], k: int,
    ) -> None:
        """Compute fold assignments over `anchor_entries` using
        DistGeoStratification and write the stratification JSON to `path`
        (matching `stratify.py`'s output schema)."""
        from scripts.processing.ripe_atlas.stratification import (
            AnchorInfo, DistGeoStratification, normalize_asn,
        )
        anchor_infos = [
            AnchorInfo(
                ip=e["address_v4"],
                lat=e["geometry"]["coordinates"][1],
                lon=e["geometry"]["coordinates"][0],
                country=e["country_code"],
                asn=normalize_asn(e["asn_v4"]),
            )
            for e in anchor_entries
        ]
        algo = DistGeoStratification(k=k, fold_index=0, seed=42)
        assignments = algo.compute_fold_assignments(anchor_infos)
        path.write_text(_json.dumps({
            "policy": {
                "class": "DistGeoStratification", "kind": "dist_geo_kfold",
                "k": k, "seed": 42, "asn_bucket_top_n": 20,
            },
            "corpus": {"source": "test", "n_anchors_yielded": len(assignments)},
            "generated_at": "2026-05-24T00:00:00+00:00",
            "fold_sizes": [
                sum(1 for f in assignments.values() if f == i) for i in range(k)
            ],
            "fold_assignments": assignments,
        }))

    def _build_source(
        self, *, slice: str, k: int = 4,
        stratification_override: "Path | None" = None,
    ):
        """Build a RipeAtlasSource backed by 20 synthetic anchors arranged in
        4 tight geographic clusters × 5 anchors each. Each anchor has 4
        probes pinging it. ASN cycles through 100/200/300/400; country
        through US/DE/JP/BR.

        `slice` must be `fold_N`. By default the fixture precomputes a
        DistGeoStratification over the synthetic corpus; pass
        `stratification_override` to use an externally-prepared file."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_root = Path(tmp.name)
        probes_json = tmp_root / "probes_and_anchors.json"
        cluster_centers = [(40, -100), (50, 10), (35, 135), (-15, -50)]
        countries = ["US", "DE", "JP", "BR"]
        entries = []
        # 20 anchors: 4 clusters × 5 anchors.
        for c_idx, (clat, clon) in enumerate(cluster_centers):
            for i in range(5):
                entries.append({
                    "id": c_idx * 100 + i,
                    "address_v4": f"10.{c_idx}.0.{i}",
                    "asn_v4": 100 + (c_idx * 5 + i) * 100 % 400,
                    "country_code": countries[c_idx],
                    "geometry": {"coordinates": [clon + i * 0.1, clat + i * 0.1]},
                    "is_anchor": True,
                })
        # 20 probes: spread separately. Their ASN/country don't matter.
        for i in range(20):
            entries.append({
                "id": 9000 + i,
                "address_v4": f"192.168.{i // 16}.{i % 16}",
                "asn_v4": 7922,
                "country_code": "US",
                "geometry": {"coordinates": [-100.0 + i, 40.0 + i * 0.1]},
                "is_anchor": False,
            })
        probes_json.write_text(_json.dumps(entries))

        anchor_entries = [e for e in entries if e["is_anchor"]]
        anchor_ips = [e["address_v4"] for e in anchor_entries]
        probe_ips = [e["address_v4"] for e in entries if not e["is_anchor"]]
        # Each anchor has 4 probes pinging it (deterministic by anchor index).
        def rtt_query(*_a, **_kw):
            out = {}
            for i, a_ip in enumerate(anchor_ips):
                out[a_ip] = {
                    probe_ips[(i + j) % len(probe_ips)]: [10.0 + j]
                    for j in range(4)
                }
            return out

        if stratification_override is not None:
            stratification_path = stratification_override
        else:
            stratification_path = tmp_root / "stratification.json"
            self._write_stratification_for_anchors(stratification_path, anchor_entries, k)

        src = RipeAtlasSource(
            slice=slice,
            probes_and_anchors_file=probes_json,
            rtt_query=rtt_query,
            sanitize=False,
            stratification_path=stratification_path,
        )
        return src, set(anchor_ips)

    def test_fold_slice_yields_disjoint_fit_and_eval_targets(self) -> None:
        src, _ = self._build_source(slice="fold_0", k=4)
        list(src.iter_eval_targets())  # forces load
        self.assertIsNotNone(src._train_anchors)
        self.assertIsNotNone(src._test_anchors)
        assert src._train_anchors is not None and src._test_anchors is not None
        self.assertEqual(src._train_anchors & src._test_anchors, set())

        # iter_eval_targets emits only test anchors.
        eval_ids = {t.target_id for t in src.iter_eval_targets()}
        self.assertEqual(eval_ids, src._test_anchors)

        # iter_fit_samples never references a held-out anchor as the target
        # (target = probe_coord in v2 FitSample; back-derive via coord lookup).
        test_coords = {
            (src._coords_by_ip[ip].lat, src._coords_by_ip[ip].lon)
            for ip in src._test_anchors
        }
        for fs in src.iter_fit_samples():
            self.assertNotIn(
                (fs.probe_coord.lat, fs.probe_coord.lon), test_coords,
                "fit sample's target (probe_coord) is a held-out anchor — leakage!",
            )

    def test_fold_union_covers_all_anchors(self) -> None:
        """Sweeping fold_index across [0, k): union of eval sets equals corpus."""
        all_evals: set[str] = set()
        all_anchors_collected: set[str] = set()
        for fold in range(4):
            src, all_anchors_collected = self._build_source(slice=f"fold_{fold}", k=4)
            evals = {t.target_id for t in src.iter_eval_targets()}
            self.assertGreater(len(evals), 0, f"fold {fold} produced empty eval set")
            self.assertTrue(
                evals.isdisjoint(all_evals),
                f"fold {fold} overlaps an earlier fold's eval set",
            )
            all_evals |= evals
        self.assertEqual(all_evals, all_anchors_collected)

    def test_slice_id_is_the_slice(self) -> None:
        """slice_id() returns the fold slice verbatim — no suffix encoding."""
        src, _ = self._build_source(slice="fold_2", k=4)
        self.assertEqual(src.slice_id(), "fold_2")

    def test_vp_configs_unchanged_by_fold(self) -> None:
        """vp_configs is metadata — fold filtering only affects fit/eval row
        emission, not the VP roster. Two different folds must yield the same
        VP roster."""
        src_a, _ = self._build_source(slice="fold_0", k=4)
        src_b, _ = self._build_source(slice="fold_1", k=4)
        self.assertEqual(
            len(list(src_a.iter_vp_configs())),
            len(list(src_b.iter_vp_configs())),
        )

    def test_drops_active_anchors_missing_from_stratification(self) -> None:
        """If the stratification was computed over a smaller corpus than what
        the source ends up with, active anchors absent from the stratification
        must be dropped (logged) from both fit and eval."""
        # Discover the active anchor set first by building a source with a
        # complete stratification.
        src_probe, all_anchors = self._build_source(slice="fold_0", k=4)
        list(src_probe.iter_eval_targets())

        # Pick a half-corpus subset, distribute across folds so neither fold
        # ends up empty.
        sorted_anchors = sorted(all_anchors)
        subset = sorted_anchors[: len(sorted_anchors) // 2 + 1]
        assignments = {ip: i % 4 for i, ip in enumerate(subset)}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stratification.json"
            path.write_text(_json.dumps({
                "policy": {
                    "class": "DistGeoStratification",
                    "kind": "dist_geo_kfold",
                    "k": 4, "seed": 42, "asn_bucket_top_n": 20,
                },
                "corpus": {"source": "test", "n_anchors_yielded": len(assignments)},
                "generated_at": "2026-05-24T00:00:00+00:00",
                "fold_sizes": [
                    sum(1 for f in assignments.values() if f == i) for i in range(4)
                ],
                "fold_assignments": assignments,
            }))

            with self.assertLogs(
                "scripts.processing.ripe_atlas.stratification", level="WARNING",
            ) as logs:
                src, _ = self._build_source(
                    slice="fold_0", stratification_override=path,
                )
                list(src.iter_eval_targets())
            self.assertTrue(
                any("active anchor(s) missing from stratification" in m for m in logs.output),
                f"expected drop warning in logs: {logs.output}",
            )
            # Eval set is a strict subset of the stratification's fold-0 anchors.
            assert src._test_anchors is not None
            expected_test = {ip for ip, f in assignments.items() if f == 0}
            self.assertEqual(set(src._test_anchors), expected_test)
            # Active anchors absent from the stratification appear in neither set.
            dropped = set(all_anchors) - set(assignments)
            for ip in dropped:
                self.assertNotIn(ip, src._train_anchors)
                self.assertNotIn(ip, src._test_anchors)

    def test_invalid_slice_for_p2a_rejected(self) -> None:
        """For P2A, slice must match `fold_N`; anything else is a hard error."""
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"address_v4": "1.1.1.1", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
            ]))
            with self.assertRaises(ValueError):
                RipeAtlasSource(
                    slice="all_anchors",
                    probes_and_anchors_file=probes_json,
                    stratification_path=Path(tmp) / "stratification.json",
                )

    def test_missing_stratification_path_for_p2a_rejected(self) -> None:
        """For P2A, stratification_path is mandatory."""
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"address_v4": "1.1.1.1", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
            ]))
            with self.assertRaises(ValueError):
                RipeAtlasSource(
                    slice="fold_0",
                    probes_and_anchors_file=probes_json,
                )

    def test_anchors_to_probes_ignores_stratification_path(self) -> None:
        """For setup=anchors_to_probes the eval targets are probes (noisy GT,
        secondary setup); stratification_path is dropped at construction with
        a logged warning, and slice_id() reflects the slice as-is."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            probes_json = tmp_root / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"address_v4": "1.1.1.1", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
                {"address_v4": "vp-a", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-100.0, 40.0]}, "is_anchor": False},
            ]))
            stratification_path = tmp_root / "stratification.json"
            stratification_path.write_text(_json.dumps({
                "policy": {
                    "class": "DistGeoStratification", "kind": "dist_geo_kfold",
                    "k": 2, "seed": 42, "asn_bucket_top_n": 20,
                },
                "corpus": {"source": "test", "n_anchors_yielded": 1},
                "generated_at": "2026-05-24T00:00:00+00:00",
                "fold_sizes": [1, 0],
                "fold_assignments": {"1.1.1.1": 0},
            }))
            with self.assertLogs(
                "scripts.benchmark.v2.sources.ripe_atlas", level="WARNING",
            ) as captured:
                src = RipeAtlasSource(
                    slice="all", setup="anchors_to_probes",
                    probes_and_anchors_file=probes_json,
                    rtt_query=lambda *a, **kw: {"1.1.1.1": {"vp-a": [10.0]}},
                    sanitize=False,
                    stratification_path=stratification_path,
                )
            self.assertTrue(any("anchors_to_probes" in m for m in captured.output))
            self.assertEqual(src.slice_id(), "all")


class TestRipeAtlasASNCorporaGeoFilter(unittest.TestCase):
    """target_continent / target_country restrict the WHOLE anchor set
    (eval + fit + tg_configs catalog), parallel to how probe_asn scopes VPs."""

    # 6 anchors across 4 countries / 3 continents, round-robined into 2 folds.
    _ANCHORS = [
        # (ip, country_code, lon, lat)
        ("10.0.0.1", "US", -100.0, 40.0),   # North America
        ("10.0.0.2", "CA", -106.0, 52.0),   # North America
        ("10.0.0.3", "FR", 2.0, 48.0),      # Europe
        ("10.0.0.4", "DE", 13.0, 52.0),     # Europe
        ("10.0.0.5", "BR", -47.0, -15.0),   # South America
        ("10.0.0.6", "US", -122.0, 37.0),   # North America
    ]
    _PROBES = [
        ("192.168.0.1", -101.0, 40.5),
        ("192.168.0.2", 3.0, 48.5),
    ]

    def _build(self, *, slice: str = "fold_0", k: int = 2, **geo_kwargs):
        """Write k anchor_fold files + a probes_of_as_7018.json into a temp dir
        and return a source with an injected rtt_query (no ClickHouse)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        probe_dir = root / "probes"
        anchor_dir = root / "anchors"
        probe_dir.mkdir()
        anchor_dir.mkdir()

        # K folds: round-robin the anchors by index so every fold is non-empty.
        folds: list[list[dict]] = [[] for _ in range(k)]
        for i, (ip, cc, lon, lat) in enumerate(self._ANCHORS):
            folds[i % k].append({
                "address_v4": ip, "asn_v4": 12345, "country_code": cc,
                "continent": continent_of(cc),
                "geometry": {"coordinates": [lon, lat]}, "is_anchor": True,
            })
        for i, entries in enumerate(folds):
            (anchor_dir / f"anchor_fold_{i}.json").write_text(_json.dumps(entries))

        (probe_dir / "probes_of_as_7018.json").write_text(_json.dumps([
            {"address_v4": ip, "asn_v4": 7018, "country_code": "US",
             "geometry": {"coordinates": [lon, lat]}}
            for ip, lon, lat in self._PROBES
        ]))

        # Every anchor is pinged by both probes.
        anchor_ips = [a[0] for a in self._ANCHORS]
        probe_ips = [p[0] for p in self._PROBES]

        def rtt_query(*_a, **_kw):
            return {a_ip: {p_ip: [10.0 + j] for j, p_ip in enumerate(probe_ips)}
                    for a_ip in anchor_ips}

        return RipeAtlasASNCorporaSource(
            slice=slice,
            probe_data_dir=probe_dir,
            probe_asn=7018,
            anchor_data_dir=anchor_dir,
            rtt_query=rtt_query,
            **geo_kwargs,
        )

    def _all_anchor_ids(self, src) -> set[str]:
        """eval ∪ fit ∪ catalog anchor ids (should all agree under a filter)."""
        eval_ids = {t.target_id for t in src.iter_eval_targets()}
        catalog_ids = {t.tg_id for t in src.iter_tg_configs()}
        # fit-sample targets are anchors (FitSample.probe_coord); back them out
        # via the catalog coords rather than ids (ids aren't on the sample).
        return eval_ids | catalog_ids

    def test_country_filter_keeps_only_that_country(self) -> None:
        src = self._build(target_country="US")
        ids = self._all_anchor_ids(src)
        self.assertEqual(ids, {"10.0.0.1", "10.0.0.6"})  # the two US anchors

    def test_country_filter_is_case_insensitive(self) -> None:
        src = self._build(target_country="us")
        self.assertEqual(self._all_anchor_ids(src), {"10.0.0.1", "10.0.0.6"})

    def test_continent_filter_keeps_only_that_continent(self) -> None:
        src = self._build(target_continent="Europe")
        self.assertEqual(self._all_anchor_ids(src), {"10.0.0.3", "10.0.0.4"})

    def test_continent_filter_is_case_insensitive(self) -> None:
        src = self._build(target_continent="europe")
        self.assertEqual(self._all_anchor_ids(src), {"10.0.0.3", "10.0.0.4"})

    def test_filter_applies_to_fit_corpus_too(self) -> None:
        """Across all folds, fit-sample target coords must lie in-region — the
        calibration corpus is filtered, not just the eval set."""
        eu_coords = {(48.0, 2.0), (52.0, 13.0)}  # (lat, lon) of FR, DE anchors
        seen: set[tuple[float, float]] = set()
        for fold in range(2):
            src = self._build(slice=f"fold_{fold}", target_continent="Europe")
            for fs in src.iter_fit_samples():
                coord = (fs.probe_coord.lat, fs.probe_coord.lon)
                self.assertIn(coord, eu_coords, "fit sample target is out-of-region")
                seen.add(coord)
        # Both EU anchors should serve as a fit target in some fold.
        self.assertEqual(seen, eu_coords)

    def test_both_params_set_raises_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            self._build(target_continent="Europe", target_country="US")

    def test_no_match_raises_on_load(self) -> None:
        src = self._build(target_country="ZZ")  # no anchor is in ZZ
        with self.assertRaises(ValueError):
            list(src.iter_eval_targets())  # forces _load_anchors

    def test_no_filter_retains_all_anchors(self) -> None:
        src = self._build()
        self.assertEqual(self._all_anchor_ids(src), {a[0] for a in self._ANCHORS})


if __name__ == "__main__":
    unittest.main()
