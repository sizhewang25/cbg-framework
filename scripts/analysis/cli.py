"""Typer CLI — post-benchmark analysis commands.

Using an analysis config (recommended):

    python -m scripts.analysis.cli cluster-score \
        --top-n 3

    python -m scripts.analysis.cli eval-bench-results \
        --lenient

Using explicit flags:

    python -m scripts.analysis.cli cluster-score \
        --source generic_csv \
        --top-n 3

    python -m scripts.analysis.cli eval-bench-results \
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
import typer
import yaml

from scripts.benchmark.v2.inputs import DEFAULT_INPUTS_ROOT
from scripts.benchmark.v2.cluster_topn import (
    build_truth_neighbor_index,
    compute_topk_match_from_truth_neighbors,
    resolve_effective_top_n,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CBG analysis CLI — post-benchmark scoring and evaluation.",
)

DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "v2" / "outputs"


def _load_analysis_config(config_path: Path) -> dict[str, Any]:
    """Parse an analysis YAML config and resolve relative paths against cwd."""
    cfg: dict[str, Any] = yaml.safe_load(config_path.read_text())
    cwd = Path.cwd()
    for key in ("v2_outputs_root", "v2_inputs_root", "analysis_root"):
        if key in cfg and not Path(str(cfg[key])).is_absolute():
            cfg[key] = str(cwd / cfg[key])
    return cfg


# ---- cluster-score (score predictions against answer space) ------------------

@app.command("cluster-score")
def cmd_cluster_score(
    config: Optional[Path] = typer.Option(
        None, help="Analysis config YAML (scripts/analysis/config/clusters/*.yaml). "
                   "When given, --run-id/--source/--clusters-dir/--out-dir are derived automatically."),
    run_id: Optional[str] = typer.Option(None, help="Run id to score (overrides config)."),
    source: Optional[str] = typer.Option(None, help="Source name (overrides config)."),
    clusters_dir: Optional[Path] = typer.Option(
        None, help="materialize-target-space output dir containing clusters.csv and assignments.csv (overrides config)."),
    out_dir: Optional[Path] = typer.Option(None, help="Where to write per-combo *_scored.csv and baseline.csv (overrides config)."),
    outputs_root: Optional[Path] = typer.Option(None, help="Root containing <run_id>/. (overrides config)."),
    inputs_root: Optional[Path] = typer.Option(None, help="Root for eval_observations (baseline, overrides config)."),
    inputs_dir: Optional[Path] = typer.Option(
        None, help="Explicit inputs dir for baseline (overrides auto-derive from run layout)."),
    top_n: int = typer.Option(
        1,
        help=(
            "Top-N target-side centroid answer space size. "
            "For each target, S_N(y) = {true centroid} + (N-1) nearest centroids to it. "
            "A prediction is correct at top-N when its nearest centroid is in S_N(y)."
        ),
    ),
) -> None:
    """Score every combo's CBG predictions against the precomputed cluster answer space.

    Pass --config for automatic path derivation, or --run-id + explicit dirs.
    Reads clusters.csv + assignments.csv from `--clusters-dir` (written by
    `materialize-target-space`). For each combo: joins truth-side cluster from assignments
    (no BallTree re-query), runs a single BallTree query for prediction coords,
    writes ``<out_dir>/<combo_id>_scored.csv`` with columns
    ``target_id, status, success, match, error_to_centroid_km, truth_centroid_km, error_km``.
    Also computes the shortest-ping-VP baseline from eval_observations and
    writes ``<out_dir>/baseline.csv`` with columns
    ``target_id, vp_matches_centroid, vp_to_centroid_km``.
    The plot scripts consume these CSVs directly, skipping redundant BallTree work.
    """
    # Resolve config-derived defaults ----------------------------------------
    if config is not None:
        cfg = _load_analysis_config(config)
        run_id = run_id or cfg.get("run_id")
        source = source or cfg.get("source")
        _setup = cfg.get("setup", "anchors_to_probes")
        _outputs_root = Path(cfg["v2_outputs_root"]) if "v2_outputs_root" in cfg else DEFAULT_OUTPUTS_ROOT
        _inputs_root = Path(cfg["v2_inputs_root"]) if "v2_inputs_root" in cfg else DEFAULT_INPUTS_ROOT
        outputs_root = outputs_root or _outputs_root
        inputs_root = inputs_root or _inputs_root
        if clusters_dir is None:
            clusters_dir = outputs_root / run_id / source / _setup / "clusters"
        if out_dir is None:
            out_dir = outputs_root / run_id / source / _setup / "cluster_scored"

    # Validate required params ------------------------------------------------
    missing = [name for name, val in [("--run-id", run_id), ("--source", source),
                                       ("--clusters-dir", clusters_dir), ("--out-dir", out_dir)] if val is None]
    if missing:
        typer.echo(f"Missing required options: {', '.join(missing)} (or pass --config)", err=True)
        raise typer.Exit(code=2)

    outputs_root = outputs_root or DEFAULT_OUTPUTS_ROOT
    inputs_root = inputs_root or DEFAULT_INPUTS_ROOT

    import numpy as np
    import pandas as pd

    from sklearn.neighbors import BallTree

    from scripts.analysis._v2_io import (
        discover_combos,
        group_combos_by_id,
        load_targets,
    )
    from scripts.libs.cbg.rtt_model import haversine_distance

    # Load answer space --------------------------------------------------------
    cdir = Path(clusters_dir)
    clusters_csv = pd.read_csv(cdir / "clusters.csv")
    assignments = pd.read_csv(cdir / "assignments.csv")

    c_lat = clusters_csv["centroid_lat"].to_numpy(dtype=float)
    c_lon = clusters_csv["centroid_lon"].to_numpy(dtype=float)
    _tree = BallTree(np.radians(np.column_stack([c_lat, c_lon])), metric="haversine")
    eff_top_n = resolve_effective_top_n(top_n, len(c_lat))
    truth_neighbors = build_truth_neighbor_index(c_lat, c_lon, eff_top_n)

    def _query(lats, lons):
        lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
        idx = np.full(len(lats), -1, dtype=int)
        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.any():
            _, i = _tree.query(
                np.radians(np.column_stack([lats[valid], lons[valid]])), k=1
            )
            idx[valid] = i[:, 0]
        return idx

    def _dist(lats, lons, idx):
        idx = np.asarray(idx)
        ok = idx >= 0
        out = np.full(len(idx), float("nan"), dtype=float)
        if ok.any():
            out[ok] = haversine_distance(
                np.asarray(lats, dtype=float)[ok],
                np.asarray(lons, dtype=float)[ok],
                c_lat[idx[ok]],
                c_lon[idx[ok]],
            )
        return out

    truth_cluster = assignments.set_index("target_id")["cluster_id"].to_dict()
    truth_dist_km = assignments.set_index("target_id")["dist_to_centroid_km"].to_dict()
    allowed_ids = set(assignments["target_id"])

    # Discover and score combos ------------------------------------------------
    run_dir = outputs_root / run_id
    combo_dirs = discover_combos(run_dir, source, None)
    grouped = group_combos_by_id(combo_dirs)

    out_dir.mkdir(parents=True, exist_ok=True)

    for combo_id, dirs in sorted(grouped.items()):
        frames = [load_targets(d).to_pandas() for d in dirs]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            df = df[df["target_id"].isin(allowed_ids)].reset_index(drop=True)
        n = len(df)
        if n == 0:
            typer.echo(f"  [{combo_id}] no in-scope targets; skipping")
            continue

        success = df["status"].isin(("SUCCESS",)).to_numpy()
        t_cl = np.array(
            [truth_cluster.get(tid, -1) for tid in df["target_id"]], dtype=int
        )
        t_km = np.array(
            [truth_dist_km.get(tid, float("nan")) for tid in df["target_id"]],
            dtype=float,
        )

        match = np.zeros(n, dtype=bool)
        err = np.full(n, float("nan"), dtype=float)
        topk_cols = {
            f"match_top{k}": np.zeros(n, dtype=bool)
            for k in range(1, eff_top_n + 1)
        }
        if success.any():
            sub = df[success]
            p_idx = _query(sub["pred_lat"], sub["pred_lon"])
            ts = t_cl[success]
            match[success] = (ts == p_idx) & (p_idx >= 0) & (ts >= 0)
            err[success] = _dist(sub["pred_lat"], sub["pred_lon"], ts)
            topk_sub = compute_topk_match_from_truth_neighbors(
                p_idx, ts, truth_neighbors, eff_top_n, column_prefix="match_top"
            )
            for k in range(1, eff_top_n + 1):
                topk_cols[f"match_top{k}"][success] = topk_sub[f"match_top{k}"]

        scored = pd.DataFrame({
            "target_id": df["target_id"].to_numpy(),
            "status": df["status"].to_numpy(),
            "success": success,
            "match": match,
            "error_to_centroid_km": err,
            "truth_centroid_km": t_km,
            "error_km": df["error_km"].to_numpy(dtype=float),
        })
        for k in range(1, eff_top_n + 1):
            scored[f"match_top{k}"] = topk_cols[f"match_top{k}"]
        scored.to_csv(out_dir / f"{combo_id}_scored.csv", index=False)
        acc = float(match.sum()) / n
        typer.echo(f"  [{combo_id}] n={n}  accuracy={acc:.1%}")

    # Shortest-ping-VP baseline ------------------------------------------------
    if inputs_dir is not None:
        idir: Optional[Path] = inputs_dir
    elif combo_dirs:
        first = combo_dirs[0]
        setup = first.parents[1].name
        cand = inputs_root / source / run_id / setup
        idir = cand if cand.exists() else None
    else:
        idir = None

    if idir is not None:
        direct = idir / "eval_observations.parquet"
        obs_paths = (
            [direct] if direct.exists()
            else sorted(idir.glob("*/eval_observations.parquet"))
        )
        if obs_paths:
            obs = pd.concat(
                [pq.read_table(p).to_pandas() for p in obs_paths], ignore_index=True
            )
            if not obs.empty:
                idx_min = obs.groupby("target_id")["latency_ms"].idxmin()
                cols = ["target_id", "target_lat", "target_lon", "vp_lat", "vp_lon"]
                rows = obs.loc[idx_min, cols].reset_index(drop=True)
                rows = rows[rows["target_id"].isin(allowed_ids)].reset_index(drop=True)
                if len(rows):
                    t_cl_b = np.array(
                        [truth_cluster.get(tid, -1) for tid in rows["target_id"]],
                        dtype=int,
                    )
                    v_idx = _query(rows["vp_lat"], rows["vp_lon"])
                    vp_to_centroid = _dist(rows["vp_lat"], rows["vp_lon"], t_cl_b)
                    baseline_topk = compute_topk_match_from_truth_neighbors(
                        v_idx, t_cl_b, truth_neighbors, eff_top_n,
                        column_prefix="vp_matches_centroid_top",
                    )
                    baseline = pd.DataFrame({
                        "target_id": rows["target_id"].to_numpy(),
                        "vp_matches_centroid": (
                            (v_idx == t_cl_b) & (v_idx >= 0) & (t_cl_b >= 0)
                        ),
                        "vp_to_centroid_km": vp_to_centroid,
                    })
                    for k in range(1, eff_top_n + 1):
                        baseline[f"vp_matches_centroid_top{k}"] = baseline_topk[
                            f"vp_matches_centroid_top{k}"
                        ]
                    baseline.to_csv(out_dir / "baseline.csv", index=False)
                    typer.echo(f"  baseline: {len(rows)} targets → {out_dir / 'baseline.csv'}")
        else:
            typer.echo(f"  no eval_observations under {idir}; skipping baseline", err=True)
    else:
        typer.echo(
            "  no inputs dir resolved; skipping baseline (pass --inputs-dir to enable)",
            err=True,
        )

    typer.echo(f"cluster-score done: {len(grouped)} combos → {out_dir}")


# ---- eval-bench-results (per-target metrics) ---------------------------------

@app.command("eval-bench-results")
def cmd_eval_bench_results(
    config: Optional[Path] = typer.Option(
        None, help="Analysis config YAML (scripts/analysis/config/clusters/*.yaml). "
                   "When given, --run-id/--outputs-root/--inputs-root are derived automatically."),
    run_id: Optional[str] = typer.Option(None, help="Run id whose targets.parquet files to score (overrides config)."),
    outputs_root: Optional[Path] = typer.Option(None, help="Root containing <run_id>/. (overrides config)."),
    inputs_root: Optional[Path] = typer.Option(
        None,
        help=(
            "Root containing materialized inputs — needed to reconstruct each "
            "target's full LTDResult set (overrides config)."
        ),
    ),
    source: Optional[str] = typer.Option(None, help="Filter combo discovery to this source name (overrides config)."),
    combos: Optional[str] = typer.Option(
        None, help="Comma-separated combo_ids to restrict to (default: every combo found on disk)."
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="Output directory (default: <run_dir>/bench_eval/)."
    ),
    skip_loo: bool = typer.Option(
        True,
        "--skip-loo",
        help=(
            "Skip leave-one-participant-out brittleness (the expensive metric "
            "group: one extra MTL+CTR rerun per participant per eligible "
            "target — costly with a Monte Carlo medoid CTR). Every other "
            "metric group still runs; loo_* columns come back as their "
            "not-computed defaults (loo_computed=False)."
        ),
    ),
) -> None:
    """Score a completed run's CBG *outputs* per (combo, target).

    Pass --config for automatic path derivation, or --run-id + explicit roots.
    The output-side twin of `eval-source`: participant-VP stats, MTL region
    area/truth-inclusion (with outer-vs-inner exclusion direction),
    includer/excluder VP stats, the closest-VP/shortest-ping-VP bridge,
    answer-space cell gap, and leave-one-participant-out brittleness (does
    dropping any single deciding VP flip MTL/CTR to failure or blow up the
    error — skip with `--skip-loo`). See
    `scripts/benchmark/v2/eval_bench_results.py`'s module docstring for the
    full metric list and the reconstruction approach. Writes
    `<out_dir>/<combo_id>_bench_per_target.csv` per combo and
    `<out_dir>/summary.parquet` (one row per combo).
    """
    from scripts.benchmark.v2.eval_bench_results import eval_bench_results

    # Resolve config-derived defaults ----------------------------------------
    if config is not None:
        cfg = _load_analysis_config(config)
        run_id = run_id or cfg.get("run_id")
        source = source or cfg.get("source")
        if outputs_root is None and "v2_outputs_root" in cfg:
            outputs_root = Path(cfg["v2_outputs_root"])
        if inputs_root is None and "v2_inputs_root" in cfg:
            inputs_root = Path(cfg["v2_inputs_root"])

    if run_id is None:
        typer.echo("Missing required option: --run-id (or pass --config)", err=True)
        raise typer.Exit(code=2)

    outputs_root = outputs_root or DEFAULT_OUTPUTS_ROOT
    inputs_root = inputs_root or DEFAULT_INPUTS_ROOT

    run_dir = outputs_root / run_id
    if not run_dir.exists():
        typer.echo(f"No such run dir: {run_dir}", err=True)
        raise typer.Exit(code=2)

    combos_list = [c.strip() for c in combos.split(",") if c.strip()] if combos else None
    stats = eval_bench_results(
        run_dir, inputs_root, out_dir=out_dir, source=source, combos=combos_list,
        compute_loo=not skip_loo,
    )
    typer.echo(f"Wrote {len(stats['per_target_csvs'])} per-target CSVs + {stats['summary_parquet']}")
    for combo_id in sorted(stats["per_target_csvs"]):
        s = stats[combo_id]
        loo_desc = (
            f"loo_any_flip={s['loo_any_flip_share']} (n_eligible={s['loo_n_eligible']})"
            if s["loo_computed"] else "loo=skipped"
        )
        typer.echo(
            f"  [{combo_id}] n={s['n_targets']} success={s['n_success']} "
            f"truth_in_region={s['truth_in_region_share']} {loo_desc} "
            f"recompute_matches={s['recompute_matches_share']}"
        )


if __name__ == "__main__":
    app()
