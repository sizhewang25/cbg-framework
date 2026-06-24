"""Assess how much CBG failure is explained by fleet geometry alone.

This is a deliberately narrow follow-up to ``characterize_failures.py``. It
keeps complementary fleet-geometry metrics:

  * ``fleet_abs_km``: distance from the closest available VP to the target's
    truth centroid.
  * ``target_distinguishable_vp_distance_km``: loose per-target bound
    ``cell_gap_km / 2``. If ``fleet_abs_km`` is below this bound, the closest
    VP is guaranteed to be closer to the truth centroid than to the nearest
    competing centroid.
  * ``target_distinguishable_vp_margin_km``: bound minus ``fleet_abs_km``.
    Positive means the fleet has a target-distinguishing VP; non-positive means
    it does not under this loose centroid rule.

The first metric is absolute proximity. The margin is the answer-space-relative
VP-proximity certificate, still expressed in km.

Outputs under ``--out-dir``:

  fleet_geometry_per_target.parquet   one row per config x variant x target
  fleet_geometry_by_config.csv        geometry distribution over unique targets
  fleet_geometry_auc.csv              AUC of each feature predicting failure
  fleet_geometry_rule_quality.csv     simple threshold-rule quality
  fleet_geometry_combined_bins.csv    fail rate in fixed-km/margin combinations
  fleet_geometry_model_auc.csv        in-sample logistic AUC for metric sets
  fleet_geometry_bins.png             2-bin margin figure (primary; threshold-free)
  fleet_geometry_combined_bins.png    4-bin abs_km+margin figure (reference only)
  FLEET_GEOMETRY_ONLY.md              short report
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis._v2_io import discover_combos, group_combos_by_id, load_targets
from scripts.analysis.plot_cluster_cdf import build_answer_space
from scripts.analysis.partvp.characterize_failures import CONFIGS, TEXTBOOK, _haversine_vec
from scripts.analysis.partvp.extract_features import _nearest_other_centroid_km

logger = logging.getLogger(__name__)

DEFAULT_ATTRIBUTION = Path("scripts/analysis/outputs/partvp/analysis_fail/per_target_failures.parquet")
DEFAULT_OUT_DIR = Path("scripts/analysis/outputs/partvp/analysis_fleet")
THRESHOLDS_KM = (5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000)


def _input_dir_for(run_dir: Path) -> Path:
    return Path("scripts/benchmark/v2/inputs/ripe_atlas_asn_corpora") / run_dir.name / "probes_to_anchors"


def _target_coords(run_dir: Path) -> pd.DataFrame:
    """Unique target coordinates from textbook combo outputs."""
    grouped = group_combos_by_id(discover_combos(run_dir, None, None))
    frames = []
    for combo_id in TEXTBOOK:
        for d in grouped.get(combo_id, []):
            frames.append(load_targets(d).to_pandas()[["target_id", "target_lat", "target_lon"]])
    if not frames:
        raise FileNotFoundError(f"no textbook targets under {run_dir}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("target_id")


def _fleet_geometry_for_config(config: str, run_dir: Path) -> pd.DataFrame:
    """Centroid-consistent closest-VP geometry for one config."""
    index, _, _ = build_answer_space(run_dir, None, None, 50.0, clusters_dir=None)
    target = _target_coords(run_dir)

    t_idx, _ = index.query(target["target_lat"], target["target_lon"])
    gap = _nearest_other_centroid_km(index.lat, index.lon)
    target = target.copy()
    target["truth_centroid_lat"] = index.lat[t_idx]
    target["truth_centroid_lon"] = index.lon[t_idx]
    target["cell_gap_km"] = gap[t_idx]

    inputs_dir = _input_dir_for(run_dir)
    paths = sorted(inputs_dir.glob("*/eval_observations.parquet"))
    direct = inputs_dir / "eval_observations.parquet"
    if not paths and direct.exists():
        paths = [direct]
    if not paths:
        raise FileNotFoundError(f"no eval_observations.parquet under {inputs_dir}")

    obs = pd.concat([pq.read_table(p).to_pandas() for p in paths], ignore_index=True)
    obs = obs.drop_duplicates(["target_id", "vp_id"])
    obs = obs.merge(
        target[["target_id", "truth_centroid_lat", "truth_centroid_lon", "cell_gap_km"]],
        on="target_id",
        how="inner",
    )
    obs["vp_to_centroid_km"] = _haversine_vec(
        obs["truth_centroid_lat"], obs["truth_centroid_lon"],
        obs["vp_lat"], obs["vp_lon"],
    )
    out = obs.groupby("target_id").agg(
        fleet_abs_km=("vp_to_centroid_km", "min"),
        cell_gap_km=("cell_gap_km", "first"),
        n_obs=("vp_id", "nunique"),
    ).reset_index()
    out["target_distinguishable_vp_distance_km"] = out["cell_gap_km"] / 2.0
    out["target_distinguishable_vp_margin_km"] = (
        out["target_distinguishable_vp_distance_km"] - out["fleet_abs_km"]
    )
    out["has_target_distinguishing_vp"] = out["target_distinguishable_vp_margin_km"] > 0
    out["config"] = config
    return out


def build_frame(attribution_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry = pd.concat(
        [_fleet_geometry_for_config(cfg, Path(run_dir)) for cfg, (_, run_dir) in CONFIGS.items()],
        ignore_index=True,
    )
    attr = pd.read_parquet(attribution_path)
    attr = attr[attr["combo_id"].isin(TEXTBOOK)].copy()
    if "outcome" not in attr:
        attr["outcome"] = np.where(attr["match"], "MATCH",
                            np.where(attr["status"].eq("SUCCESS"), "WRONG", "GIVE_UP"))
    attr["fail"] = attr["outcome"].ne("MATCH")
    keep = ["config", "run_id", "combo_id", "target_id", "status", "match", "outcome", "fail"]
    df = attr[keep].merge(geometry, on=["config", "target_id"], how="left")
    return df, geometry


def _auc_rank(y: pd.Series, x: pd.Series) -> float:
    """Mann-Whitney / rank-sum AUC; avoids requiring sklearn for this script."""
    s = pd.DataFrame({"y": y.astype(int), "x": x.astype(float)}).dropna()
    n_pos = int((s["y"] == 1).sum())
    n_neg = int((s["y"] == 0).sum())
    if len(s) < 10 or n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s["x"].rank(method="average")
    sum_pos = ranks[s["y"] == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rule_metrics(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    yv = y.to_numpy(dtype=bool)
    pv = pred.to_numpy(dtype=bool)
    tp = int((yv & pv).sum())
    tn = int((~yv & ~pv).sum())
    fp = int((~yv & pv).sum())
    fn = int((yv & ~pv).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and precision + recall else float("nan"))
    return {
        "n": len(yv),
        "bal_acc": (tpr + tnr) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(pv.mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def geometry_summary(geometry: pd.DataFrame) -> pd.DataFrame:
    return geometry.groupby("config").agg(
        n=("target_id", "size"),
        abs_p25=("fleet_abs_km", lambda s: s.quantile(0.25)),
        abs_med=("fleet_abs_km", "median"),
        abs_p75=("fleet_abs_km", lambda s: s.quantile(0.75)),
        gap_med=("cell_gap_km", "median"),
        distinguishable_distance_med=("target_distinguishable_vp_distance_km", "median"),
        distinguishable_margin_med=("target_distinguishable_vp_margin_km", "median"),
        pct_missing_target_distinguishing_vp=("has_target_distinguishing_vp", lambda s: (~s).mean()),
    ).reset_index()


def auc_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cfg, variant), d in df.groupby(["config", "combo_id"]):
        rows.append({
            "config": cfg,
            "variant": variant,
            "auc_abs": _auc_rank(d["fail"], d["fleet_abs_km"]),
            "auc_distinguishable_margin_bad": _auc_rank(
                d["fail"], -d["target_distinguishable_vp_margin_km"],
            ),
        })
    return pd.DataFrame(rows)


def rule_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", df)]
    scopes.extend((cfg, d) for cfg, d in df.groupby("config"))
    for scope, d in scopes:
        rows.append({
            "scope": scope,
            "rule": "target_distinguishable_vp_margin_km <= 0",
            "threshold": 0.0,
            **_rule_metrics(d["fail"], d["target_distinguishable_vp_margin_km"] <= 0),
        })
        for threshold in THRESHOLDS_KM:
            rows.append({
                "scope": scope,
                "rule": f"abs > {threshold:g} km",
                "threshold": float(threshold),
                **_rule_metrics(d["fail"], d["fleet_abs_km"] > threshold),
            })
    return pd.DataFrame(rows)


def best_abs_threshold(rules: pd.DataFrame, scope: str = "pooled") -> float:
    sub = rules[(rules["scope"] == scope) & rules["rule"].str.startswith("abs >")]
    return float(sub.sort_values(["bal_acc", "threshold"], ascending=[False, True]).iloc[0]["threshold"])


def combined_bins(df: pd.DataFrame, abs_threshold_km: float) -> pd.DataFrame:
    d = df.copy()
    d["abs_bad"] = d["fleet_abs_km"] > abs_threshold_km
    d["margin_bad"] = d["target_distinguishable_vp_margin_km"] <= 0
    d["geom_bin"] = np.select(
        [
            ~d["abs_bad"] & ~d["margin_bad"],
            d["abs_bad"] & ~d["margin_bad"],
            ~d["abs_bad"] & d["margin_bad"],
            d["abs_bad"] & d["margin_bad"],
        ],
        ["neither bad", "absolute only", "margin only", "both bad"],
        default="unknown",
    )
    rows = []
    for keys, sub in d.groupby(["scope", "geom_bin"], dropna=False):
        scope, geom_bin = keys
        rows.append({
            "scope": scope,
            "geom_bin": geom_bin,
            "n": len(sub),
            "share": len(sub) / len(d[d["scope"].eq(scope)]),
            "fail_rate": sub["fail"].mean(),
            "med_abs_km": sub["fleet_abs_km"].median(),
            "med_distinguishable_distance_km": sub["target_distinguishable_vp_distance_km"].median(),
            "med_distinguishable_margin_km": sub["target_distinguishable_vp_margin_km"].median(),
        })
    return pd.DataFrame(rows)


def model_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """In-sample logistic AUC for each metric alone and both together.

    This is descriptive separability, not a validated predictive model. The
    log transform keeps the heavy-tailed distance features well-scaled.
    """
    d = df.dropna(
        subset=["fleet_abs_km", "target_distinguishable_vp_margin_km", "fail"],
    ).copy()
    d["log_abs"] = np.log1p(d["fleet_abs_km"])
    margin_bad = -d["target_distinguishable_vp_margin_km"]
    d["signed_log_margin_bad"] = np.sign(margin_bad) * np.log1p(np.abs(margin_bad))

    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", d)]
    scopes.extend((cfg, sub) for cfg, sub in d.groupby("config"))
    specs = {
        "abs": ["log_abs"],
        "target_distinguishable_margin": ["signed_log_margin_bad"],
        "abs+target_distinguishable_margin": ["log_abs", "signed_log_margin_bad"],
    }
    for scope, sub in scopes:
        y = sub["fail"].astype(int).to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for model_name, cols in specs.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(sub[cols].to_numpy(), y)
            score = model.predict_proba(sub[cols].to_numpy())[:, 1]
            rows.append({
                "scope": scope,
                "model": model_name,
                "auc_in_sample": roc_auc_score(y, score),
            })
    return pd.DataFrame(rows)


def add_scopes(df: pd.DataFrame) -> pd.DataFrame:
    pooled = df.copy()
    pooled["scope"] = "pooled"
    per_cfg = df.copy()
    per_cfg["scope"] = per_cfg["config"]
    return pd.concat([pooled, per_cfg], ignore_index=True)


CONFIG_LABELS = {
    "global-global":   "Global\n(AS16509→Global)",
    "na-global":       "US→Global\n(AS7018→Global)",
    "europe-global":   "EU→Global\n(AS3209→Global)",
    "na-us":           "US→US\n(AS7018→US)",
    "na-na":           "US→NA\n(AS7018→NA)",
    "europe-europe":   "EU→EU\n(AS3209→EU)",
    "europe-country":  "EU→FR\n(AS3215→FR)",
}


GLOBAL_TARGET_CONFIGS = ["global-global", "na-global", "europe-global"]

VARIANT_LABELS = {
    "vanilla_cbg":       "Vanilla",
    "million_scale_cbg": "Million-scale",
    "octant_cbg":        "Octant",
    "spotter_cbg":       "Spotter",
}
VARIANT_COLORS = {
    "Vanilla":       "#4e79a7",
    "Million-scale": "#f28e2b",
    "Octant":        "#59a14f",
    "Spotter":       "#e15759",
}


def plot_margin_bins(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: failure rate for margin ≤ 0 targets, 4 variants × 3 global-target setups.

    Each group is one setup (global-global, na-global, europe-global); each bar
    within the group is one of the four textbook variants. Only targets with
    margin ≤ 0 are included, proving VP-limited geometry drives near-certain
    failure regardless of variant or VP fleet.
    """
    d = df[
        df["config"].isin(GLOBAL_TARGET_CONFIGS)
        & df["combo_id"].isin(VARIANT_LABELS)
        & (df["target_distinguishable_vp_margin_km"] <= 0)
    ].dropna(subset=["fail"]).copy()
    d["variant_label"] = d["combo_id"].map(VARIANT_LABELS)
    d["config_label"] = d["config"].map(CONFIG_LABELS).fillna(d["config"])

    configs = [CONFIG_LABELS[c] for c in GLOBAL_TARGET_CONFIGS]
    variants = list(VARIANT_LABELS.values())
    x = np.arange(len(configs))
    width = 0.18
    offsets = np.linspace(-(len(variants) - 1) / 2, (len(variants) - 1) / 2, len(variants)) * width

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, variant in zip(offsets, variants):
        rates, ns = [], []
        for cfg_label in configs:
            sub = d[(d["config_label"] == cfg_label) & (d["variant_label"] == variant)]
            rates.append(sub["fail"].mean() if len(sub) else float("nan"))
            ns.append(len(sub))
        ax.bar(x + offset, rates, width=width,
               label=variant, color=VARIANT_COLORS[variant])

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Failure rate")
    ax.set_title("Failure rate when no target-distinguishable VP exists (margin ≤ 0)")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.legend(title="Variant", loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bins(bins: pd.DataFrame, out_path: Path) -> None:
    """4-bin descriptive figure combining abs_km threshold and margin sign (kept for reference)."""
    order = ["neither bad", "absolute only", "margin only", "both bad"]
    sub = bins[bins["scope"] == "pooled"].set_index("geom_bin").reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(sub.index, sub["fail_rate"], color=["#59a14f", "#f28e2b", "#76b7b2", "#e15759"])
    for bar, n in zip(bars, sub["n"].fillna(0).astype(int)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("failure rate")
    ax.set_title("Failure rate by fleet-geometry bin (abs_km + margin)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    geometry: pd.DataFrame,
    auc: pd.DataFrame,
    rules: pd.DataFrame,
    bins: pd.DataFrame,
    model_auc: pd.DataFrame,
    abs_threshold_km: float,
    out_path: Path,
) -> None:
    pooled = rules[rules["scope"] == "pooled"].copy()
    best_abs = pooled[pooled["threshold"].eq(abs_threshold_km)].iloc[0]
    distinguishable = pooled[
        pooled["rule"].eq("target_distinguishable_vp_margin_km <= 0")
    ].iloc[0]
    show_bins = bins[bins["scope"].eq("pooled")].copy()
    model_piv = model_auc.pivot(index="scope", columns="model", values="auc_in_sample").reset_index()
    model_pooled = model_piv[model_piv["scope"].eq("pooled")].iloc[0]

    lines = []
    lines.append("# Fleet geometry only\n")
    lines.append("This analysis keeps centroid-consistent fleet metrics. The key VP-proximity "
                 "decision is whether `target_distinguishable_vp_margin_km = "
                 "d(truth centroid, nearest competing centroid) / 2 - fleet_abs_km` is "
                 "positive. If it is positive, the closest available VP is guaranteed to "
                 "favor the truth centroid over that nearest competitor.\n")
    lines.append("Fixed-km thresholds are kept only as descriptive comparisons. They are not "
                 "the primary VP-proximity definition because VP-target setups are not "
                 "exchangeable calibration folds.\n")
    lines.append("## Headline\n")
    lines.append(f"* Target-distinguishable VP absence alone: "
                 f"`target_distinguishable_vp_margin_km <= 0` has pooled balanced "
                 f"accuracy **{distinguishable.bal_acc:.2f}** "
                 f"(precision {distinguishable.precision:.2f}, "
                 f"recall {distinguishable.recall:.2f}).\n")
    lines.append(f"* Absolute distance alone is still useful as a continuous feature. For reference "
                 f"only, the descriptive fixed-km rule `fleet_abs_km > {abs_threshold_km:g} km` "
                 f"has pooled balanced accuracy **{best_abs.bal_acc:.2f}** "
                 f"(precision {best_abs.precision:.2f}, recall {best_abs.recall:.2f}).\n")
    lines.append("* Combined bins use the target-distinguishable VP margin and the descriptive "
                 "absolute-km cut. `margin only` means no VP is certified to favor the "
                 "truth centroid, even though the closest VP is below the descriptive "
                 "absolute-km cut.\n")
    lines.append(f"* A simple in-sample logistic separability check gives AUC "
                 f"**{model_pooled['abs']:.2f}** for absolute distance, "
                 f"**{model_pooled['target_distinguishable_margin']:.2f}** for the "
                 f"target-distinguishable VP margin, and "
                 f"**{model_pooled['abs+target_distinguishable_margin']:.2f}** for both.\n")
    lines.append("## Geometry by config\n")
    lines.append(geometry.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Feature AUC by config and variant\n")
    lines.append(auc.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Pooled rule quality\n")
    lines.append(pooled[["rule", "threshold", "n", "bal_acc", "precision", "recall", "f1", "pred_rate"]]
                 .round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Pooled fixed-km / margin bins\n")
    lines.append(show_bins[["geom_bin", "n", "share", "fail_rate", "med_abs_km",
                            "med_distinguishable_distance_km",
                            "med_distinguishable_margin_km"]]
                 .round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## In-sample logistic AUC\n")
    lines.append(model_piv.round(3).to_markdown(index=False))
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--abs-threshold-km", type=float, default=None,
                    help="Absolute-distance cut for combined bins. Default: best pooled balanced accuracy over grid.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df, geometry_target = build_frame(args.attribution)
    by_config = geometry_summary(geometry_target)
    auc = auc_table(df)
    rules = rule_quality(df)
    abs_threshold_km = args.abs_threshold_km
    if abs_threshold_km is None:
        abs_threshold_km = best_abs_threshold(rules)
    scoped = add_scopes(df)
    bins = combined_bins(scoped, abs_threshold_km)
    model_auc = model_auc_table(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "fleet_geometry_per_target.parquet", index=False)
    by_config.to_csv(args.out_dir / "fleet_geometry_by_config.csv", index=False)
    auc.to_csv(args.out_dir / "fleet_geometry_auc.csv", index=False)
    rules.to_csv(args.out_dir / "fleet_geometry_rule_quality.csv", index=False)
    bins.to_csv(args.out_dir / "fleet_geometry_combined_bins.csv", index=False)
    model_auc.to_csv(args.out_dir / "fleet_geometry_model_auc.csv", index=False)
    plot_margin_bins(df, args.out_dir / "fleet_geometry_bins.png")
    plot_bins(bins, args.out_dir / "fleet_geometry_combined_bins.png")
    write_report(
        by_config, auc, rules, bins, model_auc, abs_threshold_km,
        args.out_dir / "FLEET_GEOMETRY_ONLY.md",
    )
    logger.info("wrote fleet-geometry-only analysis to %s", args.out_dir)

    pd.set_option("display.width", 160, "display.max_columns", 30)
    print("\n=== GEOMETRY BY CONFIG ===")
    print(by_config.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== AUC BY CONFIG / VARIANT ===")
    print(auc.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== POOLED COMBINED BINS ===")
    print(bins[bins["scope"].eq("pooled")].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
