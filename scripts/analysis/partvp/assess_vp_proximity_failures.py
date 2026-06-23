"""Assess whether VP proximity explains failures across textbook CBG variants.

This script consumes the focused fleet-geometry frame produced by
``fleet_geometry_explainability.py`` and reports, for each setup and textbook
variant, how failures line up with the target-distinguishable VP margin:

    target_distinguishable_vp_margin_km =
        d(truth centroid, nearest competing centroid) / 2 - fleet_abs_km

Positive margin means at least one available VP is inside the loose
target-distinguishable bound. Non-positive margin means no available VP is
certified to favor the truth centroid under that bound.

Outputs under ``--out-dir``:

  vp_proximity_failure_assessment_by_setup_variant.csv
  vp_proximity_failure_assessment_by_setup.csv
  vp_proximity_failure_assessment_by_variant.csv
  vp_proximity_failure_residual_outcomes.csv
  VP_PROXIMITY_FAILURE_ASSESSMENT.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.partvp.characterize_failures import TEXTBOOK


DEFAULT_FLEET_FRAME = Path(
    "scripts/analysis/partvp/outputs/analysis_fleet/fleet_geometry_per_target.parquet"
)
DEFAULT_OUT_DIR = Path("scripts/analysis/partvp/outputs/analysis_fleet")
VARIANT_ORDER = {name: i for i, name in enumerate(TEXTBOOK)}


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else float("nan")


def _auc_rank(y: pd.Series, x: pd.Series) -> float:
    """Mann-Whitney / rank-sum AUC for a continuous score."""
    s = pd.DataFrame({"y": y.astype(int), "x": x.astype(float)}).dropna()
    n_pos = int((s["y"] == 1).sum())
    n_neg = int((s["y"] == 0).sum())
    if len(s) < 10 or n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s["x"].rank(method="average")
    sum_pos = ranks[s["y"] == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df = df[df["combo_id"].isin(TEXTBOOK)].copy()
    df["fail"] = df["outcome"].ne("MATCH")
    df["wrong"] = df["outcome"].eq("WRONG")
    df["give_up"] = df["outcome"].eq("GIVE_UP")
    df["missing_target_distinguishing_vp"] = (
        df["target_distinguishable_vp_margin_km"] <= 0
    )
    return df


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
        else:
            row = {"scope": "pooled"}

        fail = g["fail"]
        match = ~fail
        missing = g["missing_target_distinguishing_vp"]
        has_vp = ~missing

        tp = int((fail & missing).sum())
        fn = int((fail & has_vp).sum())
        fp = int((match & missing).sum())
        tn = int((match & has_vp).sum())
        n = len(g)
        n_fail = int(fail.sum())
        n_match = int(match.sum())
        n_missing = int(missing.sum())
        n_has_vp = int(has_vp.sum())
        recall = _safe_div(tp, n_fail)
        specificity = _safe_div(tn, n_match)

        row.update({
            "n": n,
            "n_fail": n_fail,
            "n_match": n_match,
            "fail_rate": fail.mean(),
            "wrong_rate": g["wrong"].mean(),
            "give_up_rate": g["give_up"].mean(),
            "missing_target_distinguishing_vp_rate": missing.mean(),
            "n_missing_target_distinguishing_vp": n_missing,
            "n_has_target_distinguishing_vp": n_has_vp,
            "n_fail_explained_by_missing_vp": tp,
            "n_residual_fail_with_target_distinguishing_vp": fn,
            "n_success_despite_missing_vp": fp,
            "n_success_with_target_distinguishing_vp": tn,
            "failure_share_explained_by_missing_vp": recall,
            "fail_rate_when_missing_vp": _safe_div(tp, n_missing),
            "fail_rate_when_has_target_distinguishing_vp": _safe_div(fn, n_has_vp),
            "success_share_with_missing_vp": _safe_div(fp, n_match),
            "balanced_accuracy_missing_vp_rule": (
                (recall + specificity) / 2
                if np.isfinite(recall) and np.isfinite(specificity)
                else float("nan")
            ),
            "auc_fleet_abs_km": _auc_rank(fail, g["fleet_abs_km"]),
            "auc_margin_bad": _auc_rank(fail, -g["target_distinguishable_vp_margin_km"]),
            "median_fleet_abs_km_fail": g.loc[fail, "fleet_abs_km"].median(),
            "median_fleet_abs_km_match": g.loc[match, "fleet_abs_km"].median(),
            "median_margin_km_fail": g.loc[fail, "target_distinguishable_vp_margin_km"].median(),
            "median_margin_km_match": g.loc[match, "target_distinguishable_vp_margin_km"].median(),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if "combo_id" in out:
        out["variant"] = out["combo_id"].str.replace("_cbg", "", regex=False)
    return _sort_summary(out)


def residual_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    states = {
        "missing_vp_fail": df["fail"] & df["missing_target_distinguishing_vp"],
        "residual_fail_with_target_distinguishing_vp": (
            df["fail"] & ~df["missing_target_distinguishing_vp"]
        ),
        "success_despite_missing_vp": (
            ~df["fail"] & df["missing_target_distinguishing_vp"]
        ),
    }
    for state, mask in states.items():
        for (config, combo_id), g in df[mask].groupby(["config", "combo_id"], dropna=False):
            counts = g["outcome"].value_counts()
            rows.append({
                "config": config,
                "combo_id": combo_id,
                "variant": combo_id.replace("_cbg", ""),
                "state": state,
                "n": len(g),
                "match": int(counts.get("MATCH", 0)),
                "wrong": int(counts.get("WRONG", 0)),
                "give_up": int(counts.get("GIVE_UP", 0)),
            })
    return _sort_summary(pd.DataFrame(rows))


def _sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    sort_cols = []
    if "config" in out:
        sort_cols.append("config")
    if "combo_id" in out:
        out["_variant_order"] = out["combo_id"].map(VARIANT_ORDER).fillna(999)
        sort_cols.append("_variant_order")
    if "state" in out:
        sort_cols.append("state")
    if sort_cols:
        out = out.sort_values(sort_cols).drop(columns=["_variant_order"], errors="ignore")
    return out.reset_index(drop=True)


def _pct(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{100 * v:.1f}%"


def _compact_table(df: pd.DataFrame, columns: list[str]) -> str:
    return df[columns].round(3).to_markdown(index=False)


def write_report(
    overall: pd.DataFrame,
    by_setup: pd.DataFrame,
    by_variant: pd.DataFrame,
    by_setup_variant: pd.DataFrame,
    residuals: pd.DataFrame,
    out_path: Path,
) -> None:
    o = overall.iloc[0]
    compact_cols = [
        "config",
        "variant",
        "n",
        "fail_rate",
        "missing_target_distinguishing_vp_rate",
        "failure_share_explained_by_missing_vp",
        "fail_rate_when_missing_vp",
        "fail_rate_when_has_target_distinguishing_vp",
        "auc_fleet_abs_km",
        "auc_margin_bad",
    ]
    setup_cols = [
        "config",
        "n",
        "fail_rate",
        "missing_target_distinguishing_vp_rate",
        "failure_share_explained_by_missing_vp",
        "fail_rate_when_missing_vp",
        "fail_rate_when_has_target_distinguishing_vp",
    ]
    variant_cols = [
        "variant",
        "n",
        "fail_rate",
        "missing_target_distinguishing_vp_rate",
        "failure_share_explained_by_missing_vp",
        "fail_rate_when_missing_vp",
        "fail_rate_when_has_target_distinguishing_vp",
    ]
    residual_cols = ["config", "variant", "state", "n", "match", "wrong", "give_up"]

    lines = []
    lines.append("# VP Proximity Failure Assessment\n")
    lines.append(
        "A failure is counted as explained by VP proximity when "
        "`target_distinguishable_vp_margin_km <= 0`, meaning no available VP is "
        "inside `d(truth centroid, nearest competing centroid) / 2`.\n"
    )
    lines.append("## Pooled result\n")
    lines.append(
        f"* Targets evaluated: **{int(o.n)}** across all textbook variants and setups.\n"
    )
    lines.append(
        f"* Overall failure rate: **{_pct(o.fail_rate)}**; missing target-distinguishing "
        f"VP rate: **{_pct(o.missing_target_distinguishing_vp_rate)}**.\n"
    )
    lines.append(
        f"* Missing target-distinguishing VP covers **{_pct(o.failure_share_explained_by_missing_vp)}** "
        f"of failures. Among targets missing such a VP, **{_pct(o.fail_rate_when_missing_vp)}** "
        "fail.\n"
    )
    lines.append(
        f"* Residual failure rate when a target-distinguishing VP exists: "
        f"**{_pct(o.fail_rate_when_has_target_distinguishing_vp)}**.\n"
    )
    lines.append(
        "This is a strong geometry signal, but not a complete failure model: residual "
        "failures remain substantial for some variants, especially Spotter.\n"
    )
    spotter = by_variant[by_variant["variant"].eq("spotter")].iloc[0]
    million = by_variant[by_variant["variant"].eq("million_scale")].iloc[0]
    europe_country = by_setup[by_setup["config"].eq("europe-country")].iloc[0]
    global_million = by_setup_variant[
        by_setup_variant["config"].eq("global-global")
        & by_setup_variant["variant"].eq("million_scale")
    ].iloc[0]
    lines.append("## Main reads\n")
    lines.append(
        f"* Million-scale is the cleanest fit to this geometry story: missing "
        f"target-distinguishing VP covers **{_pct(million.failure_share_explained_by_missing_vp)}** "
        f"of its failures, and its residual failure rate with such a VP present is "
        f"**{_pct(million.fail_rate_when_has_target_distinguishing_vp)}**.\n"
    )
    lines.append(
        f"* The strongest setup x variant case is `global-global / million_scale`: "
        f"**{_pct(global_million.failure_share_explained_by_missing_vp)}** of failures "
        f"are covered, and residual failure with a target-distinguishing VP is only "
        f"**{_pct(global_million.fail_rate_when_has_target_distinguishing_vp)}**.\n"
    )
    lines.append(
        f"* Spotter is not primarily explained by this VP-proximity condition. Although "
        f"missing VP covers **{_pct(spotter.failure_share_explained_by_missing_vp)}** "
        f"of Spotter failures, the residual failure rate when a target-distinguishing "
        f"VP exists is still **{_pct(spotter.fail_rate_when_has_target_distinguishing_vp)}**.\n"
    )
    lines.append(
        f"* `europe-country` has very good fleet proximity by this rule: only "
        f"**{_pct(europe_country.missing_target_distinguishing_vp_rate)}** of rows miss "
        f"a target-distinguishing VP. Its failures are therefore mostly residual, not "
        f"fleet-proximity failures.\n"
    )
    lines.append("## By setup\n")
    lines.append(_compact_table(by_setup, setup_cols))
    lines.append("")
    lines.append("## By variant\n")
    lines.append(_compact_table(by_variant, variant_cols))
    lines.append("")
    lines.append("## Setup x variant\n")
    lines.append(_compact_table(by_setup_variant, compact_cols))
    lines.append("")
    lines.append("## Residual outcome mix\n")
    lines.append(
        "The residual state is the important one for follow-up: these are failures even "
        "though a target-distinguishing VP exists under the loose margin rule.\n"
    )
    lines.append(_compact_table(residuals, residual_cols))
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fleet-frame", type=Path, default=DEFAULT_FLEET_FRAME)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = load_frame(args.fleet_frame)
    overall = summarize(df, [])
    by_setup = summarize(df, ["config"])
    by_variant = summarize(df, ["combo_id"])
    by_setup_variant = summarize(df, ["config", "combo_id"])
    residuals = residual_outcomes(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_setup_variant.to_csv(
        args.out_dir / "vp_proximity_failure_assessment_by_setup_variant.csv",
        index=False,
    )
    by_setup.to_csv(args.out_dir / "vp_proximity_failure_assessment_by_setup.csv", index=False)
    by_variant.to_csv(args.out_dir / "vp_proximity_failure_assessment_by_variant.csv", index=False)
    residuals.to_csv(args.out_dir / "vp_proximity_failure_residual_outcomes.csv", index=False)
    write_report(
        overall,
        by_setup,
        by_variant,
        by_setup_variant,
        residuals,
        args.out_dir / "VP_PROXIMITY_FAILURE_ASSESSMENT.md",
    )

    pd.set_option("display.width", 180, "display.max_columns", 30)
    print("\n=== POOLED ===")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== SETUP X VARIANT ===")
    show_cols = [
        "config",
        "variant",
        "n",
        "fail_rate",
        "missing_target_distinguishing_vp_rate",
        "failure_share_explained_by_missing_vp",
        "fail_rate_when_missing_vp",
        "fail_rate_when_has_target_distinguishing_vp",
        "auc_fleet_abs_km",
        "auc_margin_bad",
    ]
    print(by_setup_variant[show_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote VP proximity failure assessment to {args.out_dir}")


if __name__ == "__main__":
    main()
