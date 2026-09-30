"""`d_pni` against the S-P gap, and the k-means clusters the RTT figure reads.

The load-bearing classes are `TestTheGeometryIsThePlottedOne` and
`TestClustersAreReproducible`. k-means is Euclidean, so clustering raw km while
drawing symlog would draw boundaries a reader cannot see; and k-means labels
are arbitrary, so without renumbering, "cluster 2" would name a different
group on every rerun and the RTT boxes would change colour under the reader.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v5.modules import figure_pni_gap as F  # noqa: E402
from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules.paths import MissingArtifactError  # noqa: E402


def _pnis(tmp_path, rows, columns=("pni_id", "pni_lat", "pni_lon")):
    path = tmp_path / "pnis.csv"
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    return path


def _pop(sites_and_gaps, *, replicas=1, d_pni=None):
    """A `population`-shaped frame: `[(lat, lon, gap), ...]`, `replicas` TGs each."""
    rows = []
    for i, (lat, lon, gap) in enumerate(sites_and_gaps):
        for r in range(replicas):
            rows.append({"tg_id": f"tg-{i}-{r}", "tg_lat": lat, "tg_lon": lon,
                         P.GAP: gap, P.D_PNI: d_pni[i] if d_pni else float(i)})
    pop = pd.DataFrame(rows)
    pop["site_key"] = "r|" + pop.tg_lat.astype(str) + "," + pop.tg_lon.astype(str)
    return pop


class TestLoadPnis:
    def test_reads_a_valid_list_case_insensitively(self, tmp_path):
        path = _pnis(tmp_path, [("a", 40.0, -100.0)], columns=("PNI_ID", "Pni_Lat", "pni_lon"))
        assert P.load_pnis(path).pni_id.tolist() == ["a"]

    def test_missing_column_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="missing PNI columns"):
            P.load_pnis(_pnis(tmp_path, [("a", 40.0)], columns=("pni_id", "pni_lat")))

    def test_a_swapped_coordinate_is_refused(self, tmp_path):
        """lon in the lat column still parses, and moves every nearest PNI."""
        with pytest.raises(ValueError, match="Swapped"):
            P.load_pnis(_pnis(tmp_path, [("a", -100.0, 40.0)]))

    def test_a_blank_coordinate_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no usable coordinate"):
            P.load_pnis(_pnis(tmp_path, [("a", None, -100.0)]))

    def test_duplicate_ids_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate"):
            P.load_pnis(_pnis(tmp_path, [("a", 40.0, -100.0), ("a", 41.0, -100.0)]))

    def test_an_empty_list_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no PNI"):
            P.load_pnis(_pnis(tmp_path, []))

    def test_an_absent_file_is_a_missing_artifact(self, tmp_path):
        with pytest.raises(MissingArtifactError):
            P.load_pnis(tmp_path / "nope.csv")


class TestNearestPni:
    def test_picks_the_nearer_of_two(self):
        pnis = pd.DataFrame({"pni_id": ["a", "b"], "pni_lat": [0.0, 0.0], "pni_lon": [0.0, 10.0]})
        d = P.nearest_pni_km([0.0, 0.0], [1.0, 9.0], pnis)
        assert d[0] == pytest.approx(d[1])            # 1 degree from each's nearest
        assert d[0] == pytest.approx(111.19, abs=0.1)

    def test_a_tg_on_a_pni_is_zero(self):
        pnis = pd.DataFrame({"pni_id": ["a"], "pni_lat": [40.0], "pni_lon": [-100.0]})
        assert P.nearest_pni_km([40.0], [-100.0], pnis)[0] == pytest.approx(0.0, abs=1e-9)


class TestTgCoordinates:
    def test_a_tg_with_two_coordinates_is_refused(self, tmp_path):
        path = tmp_path / "e.csv"
        pd.DataFrame(
            [{"vp_id": "v1", "vp_lat": 1, "vp_lon": 1, "target_id": "t", "target_lat": 1, "target_lon": 1, "rtt_ms": 1},
             {"vp_id": "v2", "vp_lat": 1, "vp_lon": 1, "target_id": "t", "target_lat": 2, "target_lon": 1, "rtt_ms": 1}]
        ).to_csv(path, index=False)
        with pytest.raises(ValueError, match="more than one coordinate"):
            P.tg_coordinates(path)


class TestPoints:
    def test_replicas_collapse_to_one_point(self):
        pop = _pop([(1.0, 1.0, 5.0), (2.0, 2.0, 9.0)], replicas=4)
        pts = P.points(pop)
        assert len(pts) == 2 and pts.n_tgs.tolist() == [4, 4]

    def test_a_split_site_is_two_points(self):
        """Replicas that picked different S-P VPs are two observations."""
        pop = _pop([(1.0, 1.0, 5.0)], replicas=4)
        pop.loc[pop.index[:1], P.GAP] = 3000.0
        pts = P.points(pop)
        assert sorted(pts.n_tgs) == [1, 3]

    def test_every_tg_gets_its_point(self):
        pop = _pop([(1.0, 1.0, 5.0), (2.0, 2.0, 9.0)], replicas=3)
        pts = P.points(pop)
        assert pop[P.POINT_COL].notna().all()
        assert pop.groupby(P.POINT_COL).size().tolist() == pts.n_tgs.tolist()

    def test_point_ids_do_not_depend_on_row_order(self):
        pop = _pop([(1.0, 1.0, 5.0), (2.0, 2.0, 9.0), (3.0, 3.0, 1.0)], replicas=2)
        a = P.points(pop.copy())
        b = P.points(pop.sample(frac=1.0, random_state=3).copy())
        pd.testing.assert_frame_equal(a, b)


class TestTheGeometryIsThePlottedOne:
    """k-means must see the distances the reader sees."""

    def test_linear_block_and_one_decade(self):
        lin = P.LINSCALE / (1 - 1 / 10)
        u = P.symlog_units([0.0, 50.0, 100.0, 1000.0, 4000.0])
        assert u[0] == 0.0
        assert u[1] == pytest.approx(0.5 * lin)
        assert u[2] == pytest.approx(lin)
        assert u[3] - u[2] == pytest.approx(1.0)
        assert u[4] - u[3] == pytest.approx(np.log10(4.0))

    def test_matches_the_drawn_axis(self):
        """Screen distance on the figure's axis is proportional to `symlog_units`."""
        fig, ax = plt.subplots()
        ax.set_xscale("symlog", linthresh=P.LINTHRESH_KM, linscale=P.LINSCALE)
        ax.set_xlim(0, P.AXIS_MAX_KM)
        km = np.array([0.0, 25.0, 100.0, 450.0, 4000.0])
        px = ax.transData.transform(np.column_stack([km, np.zeros_like(km)]))[:, 0]
        plt.close(fig)
        u = P.symlog_units(km)
        np.testing.assert_allclose((px - px[0]) / (px[-1] - px[0]), u / u[-1], rtol=1e-9)

    def test_raw_km_would_cluster_differently(self):
        """Pins why the transform exists: on raw km the 0-100 block is one blob."""
        # Raw km spends two clusters on the far pair (~950 km apart) and lumps
        # 0-100 into one; on the drawn axes the far pair is the tight one.
        pts = pd.DataFrame({P.D_PNI: [1.0, 2.0, 60.0, 70.0, 900.0, 1200.0],
                            P.GAP: [0.0, 1.0, 60.0, 70.0, 3000.0, 3900.0], "n_tgs": 1,
                            P.POINT_COL: range(6)})
        from sklearn.cluster import KMeans

        out, _ = P.cluster(pts, k=3)
        assert out[P.CLUSTER_COL].iloc[0] != out[P.CLUSTER_COL].iloc[2]      # symlog: 0-100 splits
        raw = KMeans(3, n_init=P.N_INIT, random_state=P.SEED).fit_predict(pts[[P.D_PNI, P.GAP]].to_numpy())
        assert raw[0] == raw[2]                                              # raw km: one blob


class TestChoosingK:
    def test_silhouette_argmax_with_ties_to_the_smaller_k(self):
        assert P.choose_k({2: 0.5, 3: 0.8, 4: 0.8}) == 3

    def test_fewer_than_three_distinct_points_are_refused(self):
        pts = pd.DataFrame({P.D_PNI: [1.0, 1.0, 9.0], P.GAP: [2.0, 2.0, 9.0], "n_tgs": 1, P.POINT_COL: range(3)})
        with pytest.raises(ValueError, match="at least 3"):
            P.cluster(pts)

    @pytest.mark.parametrize("k", [1, 5])
    def test_an_out_of_range_k_is_refused(self, k):
        pts = pd.DataFrame({P.D_PNI: [1.0, 2.0, 50.0, 900.0], P.GAP: [0.0, 1.0, 50.0, 3000.0],
                            "n_tgs": 1, P.POINT_COL: range(4)})
        with pytest.raises(ValueError, match="out of range"):
            P.cluster(pts, k=k)

    def test_candidates_are_clipped_to_the_points(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert P.candidate_ks(X) == [2, 3]


class TestClustersAreReproducible:
    def test_designed_groups_are_recovered_and_numbered_by_gap(self, pni_inputs):
        run, edge_csv, pni_csv, tg_group, _ = pni_inputs
        tgs, _, meta = P.compute(run, pni_csv, source_csv=edge_csv)
        assert meta["clustering"]["k"] == 3
        by_group = tgs.assign(group=tgs.tg_id.map(tg_group)).groupby("group")[P.CLUSTER_COL].unique()
        assert {g: list(v) for g, v in by_group.items()} == {"near": [1], "far": [2], "trombone": [3]}

    def test_labels_survive_a_reshuffle_and_other_seeds(self, pni_inputs):
        run, edge_csv, pni_csv, _, _ = pni_inputs
        tgs, pts, meta = P.compute(run, pni_csv, source_csv=edge_csv)
        shuffled, _ = P.cluster(pts.drop(columns=P.CLUSTER_COL).sample(frac=1.0, random_state=7))
        again = shuffled.set_index(P.POINT_COL)[P.CLUSTER_COL].sort_index()
        assert again.tolist() == pts.set_index(P.POINT_COL)[P.CLUSTER_COL].sort_index().tolist()
        assert meta["clustering"]["stability"]["min_ari"] == pytest.approx(1.0)

    def test_every_tg_has_a_cluster(self, pni_inputs):
        run, edge_csv, pni_csv, tg_group, _ = pni_inputs
        tgs, _, _ = P.compute(run, pni_csv, source_csv=edge_csv)
        assert set(tgs.tg_id) == set(tg_group) and tgs[P.CLUSTER_COL].notna().all()


class TestWrittenArtifacts:
    def test_build_writes_all_four(self, pni_inputs):
        run, edge_csv, pni_csv, tg_group, root = pni_inputs
        [png] = F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        out = png.parent
        assert out == root / run.run_id / P.KIND / "test-pni"
        for name in (P.CLUSTERS_CSV, P.POINTS_CSV, P.MANIFEST_NAME, F.PNG_NAME):
            assert (out / name).stat().st_size > 0
        tgs, meta = P.read_clusters(out, run_id=run.run_id)
        assert len(tgs) == len(tg_group) == meta["n_tgs"]
        assert meta["source_csv_sha256"] == P.sha256_file(edge_csv)

    def test_an_unpaintable_k_writes_nothing(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        with pytest.raises(ValueError, match="no colour"):
            F.build_for_run(run, pni_csv, k=len(F.CLUSTER_HUES) + 1, analysis_root=root, source_csv=edge_csv)
        assert not (root / run.run_id).exists()

    def test_read_clusters_refuses_another_runs_file(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        [png] = F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        with pytest.raises(ValueError, match="written for run"):
            P.read_clusters(png.parent, run_id="someone-else")


class TestPrivacy:
    FORBIDDEN = ("lat", "lon", "city", "country", "site", "region", "asn", "pni_id")

    def test_no_output_column_carries_a_location(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        [png] = F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        for name in (P.CLUSTERS_CSV, P.POINTS_CSV):
            for col in pd.read_csv(png.parent / name).columns:
                assert not any(f in col.lower() for f in self.FORBIDDEN), (name, col)

    def test_manifest_holds_no_coordinate_or_pni_name(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        [png] = F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        text = (png.parent / P.MANIFEST_NAME).read_text()
        body = json.loads(text)
        assert not re.search(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}", text)
        assert "pni-a" not in text and "pni-b" not in text

        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k
                    yield from keys(v)
            elif isinstance(node, list):
                for v in node:
                    yield from keys(v)

        assert not {k.lower() for k in keys(body)} & {"lat", "lon", "site", "site_id", "site_key", "city"}
