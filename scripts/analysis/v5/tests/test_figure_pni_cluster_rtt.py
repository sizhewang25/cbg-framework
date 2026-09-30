"""The RTT boxes read `plot-pni-gap`'s clusters, and refuse stale ones.

The load-bearing class is `TestStaleClustersAreRefused`. The boxes are only
worth anything if they describe the partition the scatter drew; every way the
two can drift apart (another run, an edited edge CSV, an edited clusters file)
must raise rather than draw.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_pni_cluster_rtt as R
from scripts.analysis.v5.modules import figure_pni_gap as F
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules.paths import MissingArtifactError


@pytest.fixture
def clustered(pni_inputs):
    run, edge_csv, pni_csv, tg_group, root = pni_inputs
    [png] = F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
    return run, edge_csv, pni_csv, tg_group, root, png.parent


class TestConsumesTheWrittenClusters:
    def test_writes_png_csv_manifest_beside_the_clusters(self, clustered):
        run, edge_csv, pni_csv, _, root, out = clustered
        [png] = R.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        assert png.parent == out
        for name in (R.PNG_NAME, R.CSV_NAME, R.MANIFEST_NAME):
            assert (out / name).stat().st_size > 0

    def test_one_row_per_scope_and_cluster(self, clustered):
        run, edge_csv, pni_csv, _, root, out = clustered
        pairs, tgs, meta, _ = R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        stats = R.stats_table(pairs, tgs, meta)
        k = meta["clustering"]["k"]
        assert sorted(zip(stats.scope, stats.cluster)) == sorted(
            (s, c) for s, _ in R.SCOPES for c in range(1, k + 1)
        )

    def test_the_floor_panel_is_the_recorded_sp_rtt(self, clustered):
        """Each TG's smallest RTT is its S-P VP's: the fixture gives it 0.5 ms."""
        run, edge_csv, pni_csv, _, root, _ = clustered
        pairs, tgs, meta, _ = R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        floor = R.stats_table(pairs, tgs, meta).query("scope == @R.TG_MIN")
        assert np.allclose(floor[["min_ms", "p50_ms", "max_ms"]].to_numpy(), 0.5)

    def test_pair_counts_cover_every_edge(self, clustered):
        run, edge_csv, pni_csv, tg_group, root, _ = clustered
        pairs, tgs, meta, _ = R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        stats = R.stats_table(pairs, tgs, meta)
        n_edges = len(pd.read_csv(edge_csv))
        assert stats.query("scope == @R.PAIRS").n.sum() == n_edges
        assert stats.query("scope == @R.TG_MIN").n.sum() == len(tg_group)

    def test_whiskers_are_p5_and_p95(self, clustered):
        run, edge_csv, pni_csv, _, root, _ = clustered
        pairs, tgs, meta, _ = R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        stats = R.stats_table(pairs, tgs, meta).query("scope == @R.PAIRS").set_index("cluster")
        for c, block in pairs.groupby("cluster"):
            assert stats.loc[c, "p5_ms"] == pytest.approx(np.percentile(block.rtt_ms, 5))
            assert stats.loc[c, "p95_ms"] == pytest.approx(np.percentile(block.rtt_ms, 95))
        box = R.bxp_stats({k: 1.0 * i for i, k in enumerate(("p5", "p25", "p50", "p75", "p95"))})
        assert (box["whislo"], box["whishi"], box["fliers"]) == (0.0, 4.0, [])


class TestStaleClustersAreRefused:
    def test_no_clusters_yet_is_a_missing_artifact(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        with pytest.raises(MissingArtifactError, match="plot-pni-gap"):
            R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)

    def test_another_runs_manifest_is_refused(self, clustered):
        run, edge_csv, pni_csv, _, root, out = clustered
        manifest = out / P.MANIFEST_NAME
        body = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({**body, "run_id": "other"}))
        with pytest.raises(ValueError, match="written for run"):
            R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)

    def test_an_edited_edge_csv_is_refused(self, clustered):
        run, edge_csv, pni_csv, _, root, _ = clustered
        df = pd.read_csv(edge_csv)
        df.loc[0, "rtt_ms"] += 7.0
        df.to_csv(edge_csv, index=False)
        with pytest.raises(ValueError, match="not the CSV the clusters were computed from"):
            R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)

    def test_an_edited_pni_list_is_refused(self, clustered):
        """Same file name, so same directory, but not the list that was clustered."""
        run, edge_csv, pni_csv, _, root, _ = clustered
        with open(pni_csv, "a") as f:
            f.write("pni-c,30.0,-90.0\n")
        with pytest.raises(ValueError, match="has changed since the clusters"):
            R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)

    def test_a_dropped_tg_is_refused(self, clustered):
        run, edge_csv, pni_csv, _, root, out = clustered
        csv = out / P.CLUSTERS_CSV
        pd.read_csv(csv).iloc[1:].to_csv(csv, index=False)
        with pytest.raises(ValueError, match="manifest says"):
            R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)

    def test_an_edited_floor_is_refused(self, clustered):
        """Same TGs, same edge CSV, but the clusters file disagrees about an RTT."""
        run, edge_csv, pni_csv, _, root, out = clustered
        csv = out / P.CLUSTERS_CSV
        df = pd.read_csv(csv)
        df.loc[0, P.SP_RTT] += 1.0
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="smallest RTT differs"):
            R.load(run, pni_csv, analysis_root=root, source_csv=edge_csv)


class TestPrivacy:
    def test_no_output_column_carries_a_location(self, clustered):
        run, edge_csv, pni_csv, _, root, out = clustered
        R.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        forbidden = ("lat", "lon", "city", "country", "region", "asn", "pni_id")
        for col in pd.read_csv(out / R.CSV_NAME).columns:
            assert not any(f in col.lower() for f in forbidden), col
