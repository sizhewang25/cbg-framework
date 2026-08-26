"""CLI test for `cluster-score`, which lives in scripts/analysis/cli.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from scripts.analysis.cli import app


class TestClusterScoreCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.inputs_root = root / "inputs"
        self.outputs_root = root / "outputs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cluster_score_topn_adds_columns_and_keeps_aliases(self) -> None:
        import pandas as pd

        root = Path(self.tmp.name)
        run_id = "score-test"
        source = "mocksrc"
        setup = "probes_to_anchors"
        fold = "fold_0"
        combo = "speed_of_internet__planar_circle__geometric_centroid"

        run_combo_dir = self.outputs_root / run_id / source / setup / fold / combo
        run_combo_dir.mkdir(parents=True, exist_ok=True)

        # targets.parquet consumed by cluster-score via load_targets().
        pd.DataFrame({
            "target_id": ["t1", "t2", "t3"],
            "status": ["SUCCESS", "SUCCESS", "FALLBACK"],
            "target_lat": [0.0, 0.0, 1.0],
            "target_lon": [0.0, 1.0, 0.0],
            "pred_lat": [0.0, 1.0, 4.0],
            "pred_lon": [0.0, 0.0, 4.0],
            "error_km": [10.0, 20.0, float("nan")],
        }).to_parquet(run_combo_dir / "targets.parquet", index=False)

        clusters_dir = root / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "cluster_id": [0, 1, 2, 3],
            "centroid_lat": [0.0, 0.0, 1.0, 4.0],
            "centroid_lon": [0.0, 1.0, 0.0, 4.0],
            "n_members": [1, 1, 1, 1],
        }).to_csv(clusters_dir / "clusters.csv", index=False)
        pd.DataFrame({
            "target_id": ["t1", "t2", "t3"],
            "cluster_id": [0, 1, 2],
            "dist_to_centroid_km": [0.0, 0.0, 0.0],
        }).to_csv(clusters_dir / "assignments.csv", index=False)

        out_dir = root / "cluster_scored"
        result = self.runner.invoke(app, [
            "cluster-score",
            "--run-id", run_id,
            "--source", source,
            "--clusters-dir", str(clusters_dir),
            "--out-dir", str(out_dir),
            "--outputs-root", str(self.outputs_root),
            "--inputs-root", str(self.inputs_root),
            "--top-n", "3",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)

        scored = pd.read_csv(out_dir / f"{combo}_scored.csv")
        for col in ["match", "match_top1", "match_top2", "match_top3"]:
            self.assertIn(col, scored.columns)
        self.assertTrue((scored["match"].astype(bool) == scored["match_top1"].astype(bool)).all())

        m1 = scored["match_top1"].astype(bool).to_numpy(dtype=bool)
        m2 = scored["match_top2"].astype(bool).to_numpy(dtype=bool)
        m3 = scored["match_top3"].astype(bool).to_numpy(dtype=bool)
        self.assertTrue((m1 <= m2).all())
        self.assertTrue((m2 <= m3).all())


if __name__ == "__main__":
    unittest.main()
