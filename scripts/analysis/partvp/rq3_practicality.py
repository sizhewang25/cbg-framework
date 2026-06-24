"""RQ3 — practicality (runtime / memory) per CBG variant.

Reads per-target stage timings + memory from a benchmark run's targets.parquet
(pooled across folds) and one-time fit cost + run RSS from run.json, and joins
per-combo classification accuracy from the matching partvp feature table. Emits:

  rq3_runtime_<run>.csv   per-combo: per-stage p50/p95 ms, total inference ms,
                          throughput (targets/s), peak alloc per stage (KB),
                          fit_ms, run-attributable peak RSS (MB), accuracy.
  pareto_<run>.png        classification accuracy vs throughput, coloured by CTR
                          family — the accuracy/cost trade-off + Pareto frontier.
  phases_<run>.png        stacked per-stage median latency per combo.

Memory note: `*_alloc_peak_bytes` (tracemalloc, Python/NumPy) is the reliable
channel; `*_rss_peak_bytes` is coarse (5 ms sampler) and often 0 for fast
stages, so we report alloc for per-stage memory and run-level RSS delta only for
the whole-process footprint.

CLI:
    python -m scripts.analysis.partvp.rq3_practicality \\
        --run-dir scripts/benchmark/v2/outputs/global_as16509_final \\
        --features scripts/analysis/outputs/partvp/data/global_as16509_final.parquet \\
        --out-dir scripts/analysis/outputs/partvp/analysis_rq3
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CTR family inferred from combo recipe (for colouring the Pareto plot).
def _ctr_family(combo: str, ctr_ms_p50: float) -> str:
    if combo.endswith("_geo"):
        return "geometric_centroid"
    if ctr_ms_p50 is not None and ctr_ms_p50 > 50:
        return "monte_carlo_medoid"
    return "boundary_vertex_mean"


def collect(run_dir: Path, features: Path) -> pd.DataFrame:
    setup = run_dir / "ripe_atlas_asn_corpora" / "probes_to_anchors"
    acc = (pd.read_parquet(features).groupby("combo_id")["match"].mean()
           if features and features.exists() else pd.Series(dtype=float))
    rows = []
    for cdir in sorted(glob.glob(str(setup / "fold_0" / "*"))):
        combo = Path(cdir).name
        tpaths = glob.glob(str(setup / "fold_*" / combo / "targets.parquet"))
        rpaths = glob.glob(str(setup / "fold_*" / combo / "run.json"))
        if not tpaths:
            continue
        df = pd.concat([pd.read_parquet(f) for f in tpaths], ignore_index=True)
        runs = [json.load(open(f)) for f in rpaths]
        total = df[["ltd_ms", "mtl_ms", "ctr_ms"]].fillna(0).sum(axis=1)
        ctr_p50 = float(df["ctr_ms"].median())
        rows.append({
            "combo_id": combo,
            "ctr_family": _ctr_family(combo, ctr_p50),
            "ltd_ms_p50": round(float(df["ltd_ms"].median()), 3),
            "mtl_ms_p50": round(float(df["mtl_ms"].median()), 3),
            "ctr_ms_p50": round(ctr_p50, 3),
            "total_ms_p50": round(float(total.median()), 2),
            "total_ms_p95": round(float(total.quantile(0.95)), 2),
            "throughput_per_s": round(1000.0 / float(total.median()), 1),
            # Single-core wall-clock to geolocate 1M targets (inference only; fit
            # is one-time and negligible at this scale). Embarrassingly parallel,
            # so divide by #cores in deployment.
            "hours_per_1M_1core": round(1e6 * float(total.median()) / 1000.0 / 3600.0, 2),
            "mtl_alloc_p95_kb": round(float(df["mtl_alloc_peak_bytes"].quantile(0.95)) / 1024, 1),
            "ctr_alloc_p95_kb": round(float(df["ctr_alloc_peak_bytes"].quantile(0.95)) / 1024, 1),
            "fit_ms_mean": round(float(np.mean([r["fit_ms"] for r in runs])), 1),
            "run_peak_rss_delta_mb": round(float(np.mean(
                [(r["run_peak_rss_bytes"] - r["run_baseline_rss_bytes"]) for r in runs])) / 1e6, 1),
            "same_centroid_acc": round(float(acc.get(combo, np.nan)), 4),
        })
    return pd.DataFrame(rows).sort_values("total_ms_p50")


_COLORS = {"boundary_vertex_mean": "#4E79A7", "geometric_centroid": "#59A14F",
           "monte_carlo_medoid": "#E15759"}


def plot_pareto(df: pd.DataFrame, run: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for fam, g in df.groupby("ctr_family"):
        ax.scatter(g["throughput_per_s"], g["same_centroid_acc"] * 100,
                   s=70, color=_COLORS.get(fam, "#888"), label=fam, alpha=0.85, zorder=3)
    for _, r in df.iterrows():
        if pd.notna(r["same_centroid_acc"]):
            ax.annotate(r["combo_id"], (r["throughput_per_s"], r["same_centroid_acc"] * 100),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("throughput (targets / s, log)  →  cheaper", fontsize=11)
    ax.set_ylabel("classification accuracy (same-centroid, %)", fontsize=11)
    ax.set_title(f"RQ3 accuracy vs cost — {run}", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="CTR phase", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_phases(df: pd.DataFrame, run: str, out: Path) -> None:
    d = df.sort_values("total_ms_p50")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(d) + 1)))
    left = np.zeros(len(d))
    for stage, c in [("ltd_ms_p50", "#9C755F"), ("mtl_ms_p50", "#4E79A7"), ("ctr_ms_p50", "#E15759")]:
        ax.barh(y, d[stage], left=left, color=c, label=stage.replace("_ms_p50", ""))
        left = left + d[stage].to_numpy()
    ax.set_yticks(y); ax.set_yticklabels(d["combo_id"], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("median per-target latency (ms, log)", fontsize=11)
    ax.set_title(f"RQ3 phase breakdown — {run}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="x", which="both", alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = args.run_dir.name
    df = collect(args.run_dir, args.features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / f"rq3_runtime_{run}.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    plot_pareto(df, run, args.out_dir / f"pareto_{run}.png")
    plot_phases(df, run, args.out_dir / f"phases_{run}.png")
    logger.info("wrote rq3_runtime_%s.csv + figures to %s", run, args.out_dir)


if __name__ == "__main__":
    main()
