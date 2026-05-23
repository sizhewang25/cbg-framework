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


class TestSourceRegistry(unittest.TestCase):
    def test_known_sources_registered(self) -> None:
        self.assertIn("vultr_csv", SOURCES)
        self.assertIn("ripe_atlas", SOURCES)


if __name__ == "__main__":
    unittest.main()
