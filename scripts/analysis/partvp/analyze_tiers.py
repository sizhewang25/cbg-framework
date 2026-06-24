"""Analyze how per-target features drive the confidence tier.

Consumes one or more feature tables from `extract_features.py` and answers two
operator-facing questions per run, per CBG family:

  Q1 "geolocatable?"  tier3_low  vs  {tier1_high ∪ tier2_med}
  Q2 "precise?"       tier1_high vs  tier2_med   (among matched targets only)

Outputs (under --out-dir):
  tier_composition.csv        run × combo × tier counts / fractions
  per_tier_feature_stats.csv  run × combo × tier × feature  (p25/p50/p75/mean)
  driver_separation.csv       run × combo × question × feature  (AUC, direction, MWU p)
  tree_rules.txt              depth-3 decision-tree rules + train acc per question
  box_<run>_<combo>.png       per-tier box plots of the key features

A feature's AUC is the area under the ROC of using that single feature to answer
the question (0.5 = no signal; >0.5 means "higher feature → positive class",
<0.5 the reverse). `direction` records the sign so the report reads cleanly.

CLI:
    python -m scripts.analysis.partvp.analyze_tiers \\
        --features scripts/analysis/outputs/partvp/data/*.parquet \\
        --out-dir scripts/analysis/outputs/partvp/analysis
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
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

logger = logging.getLogger(__name__)

TEXTBOOK = ["vanilla_cbg", "million_scale_cbg", "octant_cbg", "spotter_cbg"]

# Features carried into separation / tree analysis (numeric, per-target).
FEATURES = [
    "avail_min_vp_km", "avail_min_rtt_ms", "n_obs",
    "n_part", "part_min_dist_km", "part_mean_dist_km", "part_med_dist_km",
    "part_min_rtt_ms", "part_mean_rtt_ms", "part_med_rtt_ms",
    "part_max_gap_deg", "part_circ_var", "part_mean_infl", "part_min_infl",
    "nearest_other_centroid_km", "truth_centroid_km",
]
# Compact set for box plots / the headline narrative.
KEY = ["avail_min_vp_km", "part_min_dist_km", "part_mean_dist_km",
       "part_min_rtt_ms", "part_mean_infl", "part_max_gap_deg",
       "part_circ_var", "nearest_other_centroid_km"]


def _auc_dir(y: np.ndarray, x: np.ndarray) -> tuple[float, str, float]:
    """Single-feature AUC for label y (1=positive). Returns (auc, direction, mwu_p).

    NaNs dropped pairwise. direction='higher→pos' if auc>=0.5 else 'lower→pos'.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(np.unique(y)) < 2 or len(x) < 8:
        return float("nan"), "n/a", float("nan")
    try:
        auc = roc_auc_score(y, x)
    except ValueError:
        return float("nan"), "n/a", float("nan")
    pos, neg = x[y == 1], x[y == 0]
    try:
        _, p = mannwhitneyu(pos, neg, alternative="two-sided")
    except ValueError:
        p = float("nan")
    return float(auc), ("higher→pos" if auc >= 0.5 else "lower→pos"), float(p)


def _tree_rules(df: pd.DataFrame, label: np.ndarray, feats: list[str], *, depth=3) -> tuple[str, float]:
    X = df[feats].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(label)
    X, y = X[ok], label[ok].astype(int)
    if len(np.unique(y)) < 2 or len(X) < 20:
        return "(insufficient / single-class data)", float("nan")
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(10, len(X) // 20),
                                 class_weight="balanced", random_state=0)
    clf.fit(X, y)
    acc = clf.score(X, y)
    return export_text(clf, feature_names=feats, max_depth=depth), float(acc)


def composition(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["run_id", "combo_id", "tier"]).size().rename("n").reset_index()
    tot = df.groupby(["run_id", "combo_id"]).size().rename("total").reset_index()
    out = g.merge(tot, on=["run_id", "combo_id"])
    out["frac"] = out["n"] / out["total"]
    return out


def per_tier_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, combo, tier), sub in df.groupby(["run_id", "combo_id", "tier"]):
        for f in FEATURES:
            v = sub[f].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            rows.append({"run_id": run, "combo_id": combo, "tier": tier, "feature": f,
                         "n": len(v), "p25": np.percentile(v, 25), "p50": np.percentile(v, 50),
                         "p75": np.percentile(v, 75), "mean": v.mean()})
    return pd.DataFrame(rows)


def separation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, combo), sub in df.groupby(["run_id", "combo_id"]):
        # Q1 geolocatable: tier3 negative, tier1/2 positive (all targets)
        y1 = sub["tier"].isin(["tier1_high", "tier2_med"]).to_numpy().astype(float)
        # Q2 precise: among matched, tier1 positive, tier2 negative
        m = sub["tier"].isin(["tier1_high", "tier2_med"])
        sub_m = sub[m]
        y2 = (sub_m["tier"] == "tier1_high").to_numpy().astype(float)
        for f in FEATURES:
            auc1, d1, p1 = _auc_dir(y1, sub[f].to_numpy(dtype=float))
            rows.append({"run_id": run, "combo_id": combo, "question": "Q1_geolocatable",
                         "feature": f, "auc": auc1, "direction": d1, "mwu_p": p1,
                         "n_pos": int(y1.sum()), "n_neg": int((y1 == 0).sum())})
            auc2, d2, p2 = _auc_dir(y2, sub_m[f].to_numpy(dtype=float))
            rows.append({"run_id": run, "combo_id": combo, "question": "Q2_precise",
                         "feature": f, "auc": auc2, "direction": d2, "mwu_p": p2,
                         "n_pos": int(y2.sum()), "n_neg": int((y2 == 0).sum())})
    return pd.DataFrame(rows)


def boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["tier1_high", "tier2_med", "tier3_low"]
    for (run, combo), sub in df.groupby(["run_id", "combo_id"]):
        if combo not in TEXTBOOK:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, f in zip(axes.ravel(), KEY):
            data = [sub.loc[sub["tier"] == t, f].dropna().to_numpy() for t in order]
            if all(len(d) == 0 for d in data):
                ax.set_visible(False)
                continue
            ax.boxplot(data, labels=["T1", "T2", "T3"], showfliers=False)
            ax.set_title(f, fontsize=10)
            if "km" in f or "rtt" in f or "infl" in f:
                ax.set_yscale("symlog")
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"{run} — {combo}: features by confidence tier "
                     f"(T1={int((sub.tier=='tier1_high').sum())}, "
                     f"T2={int((sub.tier=='tier2_med').sum())}, "
                     f"T3={int((sub.tier=='tier3_low').sum())})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = out_dir / f"box_{run}_{combo}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True, help="feature parquet glob(s)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--trees", action="store_true", help="also fit decision trees")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = sorted({p for g in args.features for p in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no feature files matched {args.features}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    logger.info("loaded %d rows from %d files; runs=%s", len(df), len(paths),
                sorted(df["run_id"].unique()))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    composition(df).to_csv(args.out_dir / "tier_composition.csv", index=False)
    per_tier_stats(df).to_csv(args.out_dir / "per_tier_feature_stats.csv", index=False)
    sep = separation(df)
    sep.to_csv(args.out_dir / "driver_separation.csv", index=False)
    boxplots(df, args.out_dir)

    # Headline AUC ranking (textbook combos), printed for quick reading.
    tb = sep[sep["combo_id"].isin(TEXTBOOK) & sep["auc"].notna()]
    for q in ["Q1_geolocatable", "Q2_precise"]:
        rank = (tb[tb["question"] == q].assign(strength=lambda d: (d["auc"] - 0.5).abs())
                .groupby("feature")["strength"].mean().sort_values(ascending=False))
        logger.info("\n=== %s — mean |AUC-0.5| across runs×textbook combos ===\n%s", q, rank.to_string())

    if args.trees:
        lines = []
        for (run, combo), sub in df.groupby(["run_id", "combo_id"]):
            if combo not in TEXTBOOK:
                continue
            y1 = sub["tier"].isin(["tier1_high", "tier2_med"]).to_numpy().astype(float)
            r1, a1 = _tree_rules(sub, y1, FEATURES)
            m = sub["tier"].isin(["tier1_high", "tier2_med"])
            sub_m = sub[m]
            y2 = (sub_m["tier"] == "tier1_high").to_numpy().astype(float)
            r2, a2 = _tree_rules(sub_m, y2, FEATURES)
            lines += [f"\n##### {run} — {combo}",
                      f"Q1 geolocatable (train acc {a1:.2f}):\n{r1}",
                      f"Q2 precise (train acc {a2:.2f}):\n{r2}"]
        (args.out_dir / "tree_rules.txt").write_text("\n".join(lines))
        logger.info("wrote tree_rules.txt")


if __name__ == "__main__":
    main()
