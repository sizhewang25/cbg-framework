"""Inward vs outward natural experiment (EU runs).

A single-region EU fleet (AS3209 DE-central / AS3215 FR-western) geolocating all
Europe anchors sweeps angular geometry while holding distance roughly within
continental range. Each target is labelled by the *whole-fleet* angular coverage
as seen from it (combo-independent):

  outward  avail_max_gap_deg >= 180°  → target outside the VP convex hull (one-sided)
  inward   avail_max_gap_deg <  180°  → target inside the hull (surrounded)

The key control: we compare inward vs outward **within bins of closest-VP
distance**, so any inward advantage is attributable to angular geometry, not
proximity. Outputs a per-run table + a grouped-bar figure of Tier-1 rate by
(distance bin × inward/outward), and the distance-stratified angular effect.

CLI:
    python -m scripts.analysis.partvp.analyze_inward_outward \\
        --features scripts/analysis/outputs/partvp/data/europe_as3209_eu.parquet \\
                   scripts/analysis/outputs/partvp/data/europe_as3215_eu.parquet \\
        --out-dir scripts/analysis/outputs/partvp/analysis_eu
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DIST_BINS = [0, 50, 200, 500, 1000, 1e9]
DIST_LABELS = ["≤50", "50-200", "200-500", "500-1000", ">1000"]


def label_io(df: pd.DataFrame) -> pd.Series:
    return np.where(df["avail_max_gap_deg"] >= 180.0, "outward", "inward")


def run_tables(df: pd.DataFrame, combo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = df[df["combo_id"] == combo].copy()
    v["io"] = label_io(v)
    v["dbin"] = pd.cut(v["avail_min_vp_km"], DIST_BINS, labels=DIST_LABELS, right=False)
    v["is_t1"] = (v["tier"] == "tier1_high").astype(float)
    v["is_geoloc"] = v["tier"].isin(["tier1_high", "tier2_med"]).astype(float)
    # overall
    overall = v.groupby("io").agg(n=("is_t1", "size"), t1=("is_t1", "mean"),
                                  geoloc=("is_geoloc", "mean"),
                                  med_min_vp_km=("avail_min_vp_km", "median"),
                                  med_circ_var=("avail_circ_var", "median")).reset_index()
    # distance-controlled: Tier-1 rate by (dbin × io)
    strat = v.groupby(["dbin", "io"], observed=True).agg(
        n=("is_t1", "size"), t1=("is_t1", "mean")).reset_index()
    overall.insert(0, "combo", combo)
    strat.insert(0, "combo", combo)
    return overall, strat


def plot_strat(strat: pd.DataFrame, run: str, out: Path) -> None:
    combos = sorted(strat["combo"].unique())
    fig, axes = plt.subplots(1, len(combos), figsize=(5 * len(combos), 4.2), squeeze=False)
    for ax, combo in zip(axes[0], combos):
        s = strat[strat["combo"] == combo]
        piv = s.pivot(index="dbin", columns="io", values="t1").reindex(DIST_LABELS)
        npiv = s.pivot(index="dbin", columns="io", values="n").reindex(DIST_LABELS)
        x = np.arange(len(DIST_LABELS)); w = 0.38
        for k, io in enumerate(["inward", "outward"]):
            if io in piv:
                bars = ax.bar(x + (k - 0.5) * w, piv[io].values, w, label=io,
                              color=("#2ca02c" if io == "inward" else "#d62728"), alpha=0.85)
                for xi, (val, nn) in enumerate(zip(piv[io].values, npiv[io].values if io in npiv else [np.nan]*len(x))):
                    if np.isfinite(val):
                        ax.text(xi + (k - 0.5) * w, val + 0.01, f"{val:.0%}\n({int(nn)})",
                                ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(DIST_LABELS, fontsize=8)
        ax.set_xlabel("closest-VP distance (km)"); ax.set_ylabel("Tier-1 rate")
        ax.set_title(combo, fontsize=11); ax.set_ylim(0, 1); ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{run}: Tier-1 rate by distance bin × inward/outward "
                 "(angular effect, distance-controlled)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--combos", nargs="+", default=["octant_cbg", "vanilla_cbg", "million_scale_cbg"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = sorted({p for g in args.features for p in glob.glob(g)})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_overall, all_strat = [], []
    for p in paths:
        df = pd.read_parquet(p)
        run = df["run_id"].iloc[0]
        logger.info("\n########## %s (n_targets=%d) ##########", run, df["target_id"].nunique())
        ov_run, st_run = [], []
        for combo in args.combos:
            if combo not in df["combo_id"].unique():
                continue
            ov, st = run_tables(df, combo)
            ov["run_id"] = run; st["run_id"] = run
            ov_run.append(ov); st_run.append(st)
        ov_run = pd.concat(ov_run); st_run = pd.concat(st_run)
        logger.info("overall inward vs outward:\n%s", ov_run.to_string(index=False))
        plot_strat(st_run, run, args.out_dir / f"strat_{run}.png")
        all_overall.append(ov_run); all_strat.append(st_run)
    pd.concat(all_overall).to_csv(args.out_dir / "io_overall.csv", index=False)
    pd.concat(all_strat).to_csv(args.out_dir / "io_stratified.csv", index=False)
    logger.info("wrote io_overall.csv / io_stratified.csv to %s", args.out_dir)


if __name__ == "__main__":
    main()
