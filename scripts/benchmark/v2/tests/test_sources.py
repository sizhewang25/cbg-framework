"""DataSource adapter tests.

VultrCSVSource is exercised against a synthetic in-memory CSV (no dependency
on the real Vultr CSV being present on disk). RipeAtlasSource is exercised
only for its JSON coord-loading half — ClickHouse is mocked out.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.benchmark.v2.sources import SOURCES
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.benchmark.v2.sources.vultr_csv import VultrCSVSource


_SYNTH_CSV = textwrap.dedent("""
    src_ip,dst_ip,prb_id,min_rtt,mean_rtt,sent,rcvd,msm_id,date,probe_asn,probe_country,probe_latitude,probe_longitude,anchor_asn,anchor_country,anchor_latitude,anchor_longitude,anchor_city
    10.0.0.1,1.1.1.1,1001,10.0,10.0,3,3,1,2023-05-01,7922,US,40.0,-100.0,20473,US,33.0,-84.0,Atlanta
    10.0.0.2,1.1.1.1,1002,12.0,12.0,3,3,2,2023-05-01,7922,US,41.0,-101.0,20473,US,33.0,-84.0,Atlanta
    10.0.0.3,2.2.2.2,1003,11.0,11.0,3,3,3,2023-05-01,3356,US,42.0,-102.0,40,US,47.0,-122.0,Seattle
    10.0.0.4,2.2.2.2,1004,11.5,11.5,3,3,4,2023-05-01,3356,US,43.0,-103.0,40,US,47.0,-122.0,Seattle
    10.0.0.5,2.2.2.2,1005,12.5,12.5,3,3,5,2023-05-01,3356,US,44.0,-104.0,40,US,47.0,-122.0,Seattle
""").strip() + "\n"


class TestVultrCSVSource(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "vultr_smoke.csv"
        self.csv_path.write_text(_SYNTH_CSV)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_all_us_emits_every_row(self) -> None:
        src = VultrCSVSource(slice="all_us", csv_path=self.csv_path)
        self.assertEqual(len(list(src.iter_vp_configs())), 5)
        self.assertEqual(len(list(src.iter_fit_samples())), 5)
        self.assertEqual(len(list(src.iter_eval_targets())), 2)  # 2 distinct dst_ip

    def test_top1_picks_asn_with_most_probes(self) -> None:
        # ASN 3356 has 3 unique probes vs 7922's 2 → top1 must pick 3356.
        src = VultrCSVSource(slice="top1", csv_path=self.csv_path)
        targets = list(src.iter_eval_targets())
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].target_id, "2.2.2.2")
        self.assertEqual(len(targets[0].obs), 3)

    def test_slice_id_round_trip(self) -> None:
        src = VultrCSVSource(slice="top1", csv_path=self.csv_path)
        self.assertEqual(src.slice_id(), "top1")
        self.assertEqual(src.name, "vultr_csv")

    def test_unknown_slice_raises(self) -> None:
        src = VultrCSVSource(slice="garbage", csv_path=self.csv_path)
        with self.assertRaises(ValueError):
            list(src.iter_fit_samples())

    def test_fit_sample_maps_anchor_coords_into_probe_coord_field(self) -> None:
        """The v2 FitSample contract puts the ground-truth target coord in
        `probe_coord`. For Vultr, that's the anchor coord (the entity we
        ultimately want to geolocate)."""
        src = VultrCSVSource(slice="all_us", csv_path=self.csv_path)
        sample = next(iter(src.iter_fit_samples()))
        # First row's anchor coords are (33.0, -84.0)
        self.assertAlmostEqual(sample.probe_coord.lat, 33.0)
        self.assertAlmostEqual(sample.probe_coord.lon, -84.0)
        # First row's probe coords are (40.0, -100.0) → vp_coord
        self.assertAlmostEqual(sample.vp_coord.lat, 40.0)
        self.assertAlmostEqual(sample.vp_coord.lon, -100.0)

    def test_anchors_to_probes_setup_swaps_roles(self) -> None:
        """anchors_to_probes: anchors become VPs, probes become targets."""
        src = VultrCSVSource(
            slice="all_us", setup="anchors_to_probes", csv_path=self.csv_path,
        )
        self.assertEqual(src.setup_id(), "anchors_to_probes")

        # 5 unique probes in the synthetic CSV → 5 EvalTargets.
        targets = list(src.iter_eval_targets())
        self.assertEqual(len(targets), 5)
        # 2 unique anchors → 2 VPs.
        vps = list(src.iter_vp_configs())
        self.assertEqual(len(vps), 2)
        # FitSample.vp_coord must now be the anchor side.
        sample = next(iter(src.iter_fit_samples()))
        self.assertAlmostEqual(sample.vp_coord.lat, 33.0)  # first row's anchor
        self.assertAlmostEqual(sample.probe_coord.lat, 40.0)  # first row's probe

    def test_unknown_setup_raises(self) -> None:
        with self.assertRaises(ValueError):
            VultrCSVSource(slice="all_us", setup="bogus", csv_path=self.csv_path)


class TestRipeAtlasSourceCoordLoad(unittest.TestCase):
    def test_load_coords_from_synthetic_json(self) -> None:
        import json as _json
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
            src = RipeAtlasSource(slice="all_anchors", probes_and_anchors_file=probes_json)
            src._load_coords()
            assert src._coords_by_ip is not None
            self.assertEqual(len(src._coords_by_ip), 2)
            self.assertAlmostEqual(src._coords_by_ip["1.1.1.1"].lat, 33.0)
            self.assertAlmostEqual(src._coords_by_ip["1.1.1.1"].lon, -84.0)
            self.assertEqual(src._anchor_ips, {"1.1.1.1"})

    def test_anchors_to_probes_setup_transposes_groups(self) -> None:
        """In anchors_to_probes mode, EvalTargets are probes, not anchors."""
        import json as _json
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
                slice="all_anchors", setup="anchors_to_probes",
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
        """Inject a fake rtt_query so the test doesn't require ClickHouse."""
        import json as _json
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
                slice="all_anchors",
                probes_and_anchors_file=probes_json,
                rtt_query=lambda *a, **kw: {"1.1.1.1": {"vp-a": [10.0], "vp-b": [12.0]}},
                sanitize=False,
            )
            targets = list(src.iter_eval_targets())
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].target_id, "1.1.1.1")
        self.assertEqual(len(targets[0].obs), 2)
        self.assertAlmostEqual(targets[0].true_coord.lat, 33.0)


class TestRipeAtlasAnchorCityEnrichment(unittest.TestCase):
    """`anchor_city.json` → VpConfig.city wiring (ANCHORS_TO_PROBES surfaces
    cities on VPs since anchors *are* the VPs in that setup)."""

    def test_city_populated_from_nominatim_response(self) -> None:
        import json as _json
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
                slice="all_anchors", setup="anchors_to_probes",
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
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"id": 1, "address_v4": "1.1.1.1", "asn_v4": 1,
                 "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
            ]))
            src = RipeAtlasSource(
                slice="all_anchors", setup="anchors_to_probes",
                probes_and_anchors_file=probes_json,
                anchor_city_file=Path(tmp) / "does_not_exist.json",
                rtt_query=lambda *a, **kw: {"1.1.1.1": {"1.1.1.1": [1.0]}},
                sanitize=False,
            )
            vps = list(src.iter_vp_configs())
        self.assertEqual(len(vps), 1)
        self.assertIsNone(vps[0].city)


class TestTgConfigsEmission(unittest.TestCase):
    """tg_configs.parquet rows: VultrCSV side. In PROBES_TO_ANCHORS, targets
    are the dst_ip anchors; in ANCHORS_TO_PROBES, targets are the probes."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "vultr_smoke.csv"
        self.csv_path.write_text(_SYNTH_CSV)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_probes_to_anchors_targets_are_anchors_with_city(self) -> None:
        src = VultrCSVSource(slice="all_us", csv_path=self.csv_path)
        tgs = sorted(src.iter_tg_configs(), key=lambda t: t.tg_id)
        # 2 unique anchors in the fixture (1.1.1.1 Atlanta, 2.2.2.2 Seattle)
        self.assertEqual([t.tg_id for t in tgs], ["1.1.1.1", "2.2.2.2"])
        self.assertEqual(tgs[0].city, "Atlanta")
        self.assertEqual(tgs[1].city, "Seattle")
        self.assertAlmostEqual(tgs[0].lat, 33.0)
        self.assertAlmostEqual(tgs[1].lat, 47.0)
        self.assertEqual(tgs[0].asn, 20473)
        self.assertEqual(tgs[1].country, "US")

    def test_anchors_to_probes_targets_are_probes_without_city(self) -> None:
        src = VultrCSVSource(
            slice="all_us", setup="anchors_to_probes", csv_path=self.csv_path,
        )
        tgs = list(src.iter_tg_configs())
        # 5 unique probes; none have city data in the Vultr CSV schema.
        self.assertEqual(len(tgs), 5)
        for t in tgs:
            self.assertIsNone(t.city)
        # Spot-check one: prb_id 1001 → (40, -100), ASN 7922
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
            src = VultrCSVSource(slice="all_us", csv_path=csv)
            out_dir = materialize_inputs(src, root=tmp_path / "inputs")
            tg_path = out_dir / "tg_configs.parquet"
            self.assertTrue(tg_path.exists())

            table = pq.read_table(tg_path)
            self.assertEqual(table.schema, bench_schema.TG_CONFIGS_SCHEMA)
            self.assertEqual(table.num_rows, 2)
            row_by_id = {r["tg_id"]: r for r in table.to_pylist()}
            self.assertEqual(row_by_id["1.1.1.1"]["city"], "Atlanta")
            self.assertEqual(row_by_id["2.2.2.2"]["city"], "Seattle")

            # Manifest reflects the new artifact's row count.
            manifest = (out_dir / "manifest.json").read_text()
            self.assertIn('"n_tg_configs": 2', manifest)


class TestRipeAtlasTgConfigs(unittest.TestCase):
    """RIPE Atlas side: anchors in PROBES_TO_ANCHORS, probes in ANCHORS_TO_PROBES,
    plus city propagation from anchor_city.json."""

    def test_probes_to_anchors_targets_are_anchors_with_city(self) -> None:
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            probes_json = Path(tmp) / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"id": 6025, "address_v4": "anchor-a", "asn_v4": 8839,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.7485, 48.5795]}, "is_anchor": True},
                {"id": 999, "address_v4": "vp-x", "asn_v4": 1,
                 "country_code": "FR",
                 "geometry": {"coordinates": [7.0, 48.0]}, "is_anchor": False},
            ]))
            city_json = Path(tmp) / "anchor_city.json"
            city_json.write_text(_json.dumps({
                "6025": {"features": [{"properties": {"address": {"city": "Strasbourg"}}}]},
            }))
            src = RipeAtlasSource(
                slice="all_anchors",
                probes_and_anchors_file=probes_json,
                anchor_city_file=city_json,
                rtt_query=lambda *a, **kw: {"anchor-a": {"vp-x": [10.0]}},
                sanitize=False,
            )
            tgs = list(src.iter_tg_configs())
        self.assertEqual([t.tg_id for t in tgs], ["anchor-a"])
        self.assertEqual(tgs[0].city, "Strasbourg")

    def test_anchors_to_probes_targets_are_probes_without_city(self) -> None:
        import json as _json
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
                slice="all_anchors", setup="anchors_to_probes",
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
        self.assertIn("vultr_csv", SOURCES)
        self.assertIn("ripe_atlas", SOURCES)


class TestRipeAtlasSourceHoldout(unittest.TestCase):
    """End-to-end checks that PartitionPolicy partitions iter_fit_samples and
    iter_eval_targets disjointly and that slice_id() carries the fold suffix.

    The source consumes a precomputed partition JSON; this fixture writes one
    over the 20-anchor / 20-probe synthetic corpus on the fly. Algorithm-level
    invariants live in test_holdout.py / test_dist_geo_holdout.py."""

    @staticmethod
    def _write_partition_json(
        path: Path, anchor_entries: list[dict], k: int, policy_class: str,
    ) -> None:
        """Compute fold assignments over `anchor_entries` using `policy_class`
        and write the partition JSON to `path` (matching `partition.py`'s
        output schema)."""
        import json as _json
        from scripts.processing.ripe_atlas.holdout import (
            AnchorInfo, DistGeoKFoldPolicy, HoldoutPolicy,
        )
        anchor_infos = [
            AnchorInfo(
                ip=e["address_v4"],
                lat=e["geometry"]["coordinates"][1],
                lon=e["geometry"]["coordinates"][0],
                country=e["country_code"],
                asn=e["asn_v4"],
            )
            for e in anchor_entries
        ]
        if policy_class == "DistGeoKFoldPolicy":
            algo = DistGeoKFoldPolicy(k=k, fold_index=0, seed=42)
            policy_payload: dict = {
                "class": "DistGeoKFoldPolicy", "kind": "dist_geo_kfold",
                "k": k, "seed": 42, "asn_bucket_top_n": 20,
            }
        elif policy_class == "HoldoutPolicy":
            algo = HoldoutPolicy(k=k, fold_index=0, seed=42, spatial_clusters=k)
            policy_payload = {
                "class": "HoldoutPolicy", "kind": "sechidis_kfold",
                "k": k, "seed": 42, "asn_bucket_top_n": 20,
                "spatial_clusters": k, "labels": ["country", "asn_bucket"],
            }
        else:
            raise ValueError(f"unsupported policy_class for test fixture: {policy_class!r}")
        assignments = algo.compute_fold_assignments(anchor_infos)
        path.write_text(_json.dumps({
            "policy": policy_payload,
            "corpus": {"source": "test", "n_anchors_yielded": len(assignments)},
            "generated_at": "2026-05-24T00:00:00+00:00",
            "fold_sizes": [
                sum(1 for f in assignments.values() if f == i) for i in range(k)
            ],
            "fold_assignments": assignments,
        }))

    def _build_source(
        self, *, fold_index: int | None = None, k: int = 4,
        policy_class: str = "DistGeoKFoldPolicy",
        partition_override: "Path | None" = None,
    ):
        """Build a RipeAtlasSource backed by 20 synthetic anchors arranged in
        4 tight geographic clusters × 5 anchors each. Each anchor has 4 probes
        pinging it. ASN cycles through 100/200/300/400; country through
        US/DE/JP/BR.

        When `fold_index` is set, also precomputes a partition over the
        synthetic corpus (using `policy_class`) and wraps it in a
        PartitionPolicy passed via `holdout=`. `partition_override` skips the
        compute step and uses an externally-prepared partition file instead.
        """
        import json as _json
        from scripts.processing.ripe_atlas.holdout import PartitionPolicy

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

        if partition_override is not None:
            holdout = PartitionPolicy(
                path=partition_override,
                fold_index=fold_index if fold_index is not None else 0,
            )
        elif fold_index is None:
            holdout = None
        else:
            partition_path = tmp_root / "partition.json"
            self._write_partition_json(partition_path, anchor_entries, k, policy_class)
            holdout = PartitionPolicy(path=partition_path, fold_index=fold_index)

        src = RipeAtlasSource(
            slice="all_anchors",
            probes_and_anchors_file=probes_json,
            rtt_query=rtt_query,
            sanitize=False,
            holdout=holdout,
        )
        return src, set(anchor_ips)

    def test_holdout_none_preserves_full_eval_set(self) -> None:
        """holdout=None: behavior must be byte-identical to today (no filter)."""
        src, all_anchors = self._build_source(fold_index=None)
        targets = {t.target_id for t in src.iter_eval_targets()}
        self.assertEqual(targets, all_anchors)
        self.assertEqual(src.slice_id(), "all_anchors")

    def test_holdout_fold_yields_disjoint_fit_and_eval_targets(self) -> None:
        src, _ = self._build_source(fold_index=0, k=4)
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

    def test_holdout_fold_union_covers_all_anchors(self) -> None:
        """Sweeping fold_index across [0, k): union of eval sets equals corpus."""
        all_evals: set[str] = set()
        all_anchors_collected: set[str] = set()
        for fold in range(4):
            src, all_anchors_collected = self._build_source(fold_index=fold, k=4)
            evals = {t.target_id for t in src.iter_eval_targets()}
            self.assertGreater(len(evals), 0, f"fold {fold} produced empty eval set")
            self.assertTrue(
                evals.isdisjoint(all_evals),
                f"fold {fold} overlaps an earlier fold's eval set",
            )
            all_evals |= evals
        self.assertEqual(all_evals, all_anchors_collected)

    def test_slice_id_reconstructs_underlying_policy_suffix(self) -> None:
        """PartitionPolicy reconstructs the suffix from the partition JSON's
        policy.class field, so slice_id reflects whichever algorithm produced
        the file (DistGeo vs Sechidis)."""
        cases = [
            ("DistGeoKFoldPolicy", "all_anchors__distgeo_fold2of4_seed42"),
            ("HoldoutPolicy", "all_anchors__fold2of4_seed42"),
        ]
        for policy_class, expected in cases:
            with self.subTest(policy_class=policy_class):
                src, _ = self._build_source(
                    fold_index=2, k=4, policy_class=policy_class,
                )
                self.assertEqual(src.slice_id(), expected)

    def test_vp_configs_unchanged_by_holdout(self) -> None:
        """vp_configs is metadata — fold filtering only affects fit/eval row
        emission, not the VP roster."""
        src_none, _ = self._build_source(fold_index=None)
        n_vps_none = len(list(src_none.iter_vp_configs()))
        src_fold, _ = self._build_source(fold_index=0, k=4)
        n_vps_fold = len(list(src_fold.iter_vp_configs()))
        self.assertEqual(n_vps_none, n_vps_fold)

    def test_partition_policy_drops_active_anchors_missing_from_partition(self) -> None:
        """If the partition was computed over a smaller corpus than what the
        source ends up with, active anchors not in the partition must be
        dropped (logged) from both fit and eval."""
        import json as _json

        # Build the source first to discover its active anchor set.
        src_probe, all_anchors = self._build_source(fold_index=None)
        list(src_probe.iter_eval_targets())

        # Pick a half-corpus subset for the partition. Distribute across folds
        # so neither fold ends up empty.
        sorted_anchors = sorted(all_anchors)
        subset = sorted_anchors[: len(sorted_anchors) // 2 + 1]
        assignments = {ip: i % 4 for i, ip in enumerate(subset)}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partition.json"
            path.write_text(_json.dumps({
                "policy": {
                    "class": "DistGeoKFoldPolicy",
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
                "scripts.processing.ripe_atlas.holdout", level="WARNING",
            ) as logs:
                src, _ = self._build_source(fold_index=0, partition_override=path)
                list(src.iter_eval_targets())
            self.assertTrue(
                any("active anchor(s) missing from partition" in m for m in logs.output),
                f"expected drop warning in logs: {logs.output}",
            )
            # Eval set is a strict subset of the partition's fold-0 anchors.
            assert src._test_anchors is not None
            expected_test = {ip for ip, f in assignments.items() if f == 0}
            self.assertEqual(set(src._test_anchors), expected_test)
            # Active anchors absent from partition appear in neither train nor test.
            dropped = set(all_anchors) - set(assignments)
            for ip in dropped:
                self.assertNotIn(ip, src._train_anchors)
                self.assertNotIn(ip, src._test_anchors)

    def test_anchors_to_probes_setup_strips_holdout_with_warning(self) -> None:
        """For setup=anchors_to_probes the eval targets are probes (noisy GT,
        secondary setup); a PartitionPolicy is silently dropped at construction
        with a logged warning, and slice_id() must not carry the suffix."""
        from scripts.processing.ripe_atlas.holdout import PartitionPolicy
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            probes_json = tmp_root / "probes_and_anchors.json"
            probes_json.write_text(_json.dumps([
                {"address_v4": "1.1.1.1", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-84.0, 33.0]}, "is_anchor": True},
                {"address_v4": "2.2.2.2", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-122.0, 47.0]}, "is_anchor": True},
                {"address_v4": "vp-a", "asn_v4": 7922, "country_code": "US",
                 "geometry": {"coordinates": [-100.0, 40.0]}, "is_anchor": False},
            ]))
            partition_path = tmp_root / "partition.json"
            partition_path.write_text(_json.dumps({
                "policy": {
                    "class": "DistGeoKFoldPolicy", "kind": "dist_geo_kfold",
                    "k": 2, "seed": 42, "asn_bucket_top_n": 20,
                },
                "corpus": {"source": "test", "n_anchors_yielded": 2},
                "generated_at": "2026-05-24T00:00:00+00:00",
                "fold_sizes": [1, 1],
                "fold_assignments": {"1.1.1.1": 0, "2.2.2.2": 1},
            }))
            with self.assertLogs(
                "scripts.benchmark.v2.sources.ripe_atlas", level="WARNING",
            ) as captured:
                src = RipeAtlasSource(
                    slice="all_anchors", setup="anchors_to_probes",
                    probes_and_anchors_file=probes_json,
                    rtt_query=lambda *a, **kw: {"1.1.1.1": {"vp-a": [10.0]}},
                    sanitize=False,
                    holdout=PartitionPolicy(path=partition_path, fold_index=0),
                )
            self.assertTrue(any("anchors_to_probes" in m for m in captured.output))
            self.assertEqual(src.slice_id(), "all_anchors")


if __name__ == "__main__":
    unittest.main()
