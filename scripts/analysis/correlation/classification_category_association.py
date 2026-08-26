from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


DEFAULT_ANALYSIS_ROOT = Path("scripts/analysis/outputs")
DEFAULT_BENCH_OUTPUTS_ROOT = Path("scripts/benchmark/v2/outputs")
DEFAULT_OUT_DIR = Path("scripts/analysis/correlation/outputs")

EVAL_COLUMNS = [
    "target_id",
    "n_avail_vps",
    "closest_vp_km",
    "shortest_ping_vp_km",
    "min_rtt_ms",
    "min_inflation",
    "rtt_weighted_dist_km",
    "cell_gap_km",
    "target_distinguishable_vp_dist_km",
    "closest_vp_in_same_cluster",
    "shortest_ping_vp_in_same_cluster",
    "closest_vp_to_centroid_km",
    "shortest_ping_vp_to_centroid_km",
    "n_discriminative_vps",
    "has_vp_proximity",
    "shortest_ping_vp_is_discriminative",
    "best_discriminative_rtt_rank",
    "proximity_label",
    "anycast_suspect",
]

ASSOCIATION_METRICS = [
    "n_avail_vps",
    "closest_vp_km",
    "shortest_ping_vp_km",
    "min_rtt_ms",
    "min_inflation",
    "rtt_weighted_dist_km",
    "cell_gap_km",
    "target_distinguishable_vp_dist_km",
    "closest_vp_in_same_cluster",
    "shortest_ping_vp_in_same_cluster",
    "closest_vp_to_centroid_km",
    "shortest_ping_vp_to_centroid_km",
    "n_discriminative_vps",
    "has_vp_proximity",
    "shortest_ping_vp_is_discriminative",
    "best_discriminative_rtt_rank",
    "anycast_suspect",
]

SUMMARY_SPECS = [
    ("has_vp_proximity", "share"),
    ("shortest_ping_vp_is_discriminative", "share"),
    ("closest_vp_in_same_cluster", "share"),
    ("shortest_ping_vp_in_same_cluster", "share"),
    ("anycast_suspect", "share"),
    ("n_discriminative_vps", "mean"),
    ("n_discriminative_vps", "median"),
    ("target_distinguishable_vp_dist_km", "median"),
    ("cell_gap_km", "median"),
    ("closest_vp_to_centroid_km", "median"),
    ("shortest_ping_vp_to_centroid_km", "median"),
]


def _as_numeric_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("float64")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
    )
    return mapped.astype("float64")


def _load_config(config_path: Path) -> dict:
    data = yaml.safe_load(config_path.read_text()) or {}
    if "run_id" not in data:
        raise ValueError(f"{config_path}: missing run_id")
    return data


def _find_eval_csv(run_dir: Path) -> Path:
    hits = sorted((run_dir / "eval_source").glob("*_eval_per_target.csv"))
    if not hits:
        raise FileNotFoundError(f"No *_eval_per_target.csv under {run_dir / 'eval_source'}")
    if len(hits) > 1:
        raise ValueError(f"Expected one *_eval_per_target.csv under {run_dir / 'eval_source'}, found {len(hits)}")
    return hits[0]


def _load_run_frame(config_path: Path, top_n: int) -> tuple[pd.DataFrame, dict[str, object]]:
    cfg = _load_config(config_path)
    run_id = cfg["run_id"]
    analysis_root = Path(cfg.get("analysis_root", DEFAULT_ANALYSIS_ROOT))
    outputs_root = Path(cfg.get("v2_outputs_root", DEFAULT_BENCH_OUTPUTS_ROOT))

    cls_path = analysis_root / run_id / "cluster" / f"{run_id}_classification_analysis_top{top_n}.csv"
    if not cls_path.exists():
        raise FileNotFoundError(f"Missing classification analysis CSV: {cls_path}")

    run_dir = outputs_root / run_id
    eval_path = _find_eval_csv(run_dir)

    cls_df = pd.read_csv(cls_path)
    eval_df = pd.read_csv(eval_path)
    cls_df["target_id"] = cls_df["target_id"].astype(str)
    eval_df["target_id"] = eval_df["target_id"].astype(str)

    keep_cols = [col for col in EVAL_COLUMNS if col in eval_df.columns]
    merged = cls_df.merge(eval_df[keep_cols], on="target_id", how="inner")
    if merged.empty:
        raise ValueError(f"{run_id}: merge produced zero rows")

    bool_cols = [
        "has_vp_proximity",
        "shortest_ping_vp_is_discriminative",
        "closest_vp_in_same_cluster",
        "shortest_ping_vp_in_same_cluster",
        "anycast_suspect",
    ]
    for col in bool_cols:
        if col in merged.columns:
            merged[col] = _as_numeric_bool(merged[col])

    numeric_cols = [
        "n_avail_vps",
        "closest_vp_km",
        "shortest_ping_vp_km",
        "min_rtt_ms",
        "min_inflation",
        "rtt_weighted_dist_km",
        "cell_gap_km",
        "target_distinguishable_vp_dist_km",
        "closest_vp_to_centroid_km",
        "shortest_ping_vp_to_centroid_km",
        "n_discriminative_vps",
        "best_discriminative_rtt_rank",
        "n_correct",
        "n_methods",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["run_id"] = run_id
    merged["config"] = str(config_path)
    merged["top_n"] = int(top_n)
    merged["is_all_true"] = (merged["category"] == "all_true").astype(float)
    merged["is_all_false"] = (merged["category"] == "all_false").astype(float)

    # Per-method binary outcome: was "shortest_ping" in the correct_methods list?
    if "correct_methods" in merged.columns:
        merged["is_shortest_ping_correct"] = (
            merged["correct_methods"]
            .fillna("")
            .astype(str)
            .str.contains(r"(?:^|,)\s*shortest_ping\s*(?:,|$)", regex=True)
            .astype(float)
        )

    meta = {
        "run_id": run_id,
        "config": str(config_path),
        "classification_csv": str(cls_path),
        "eval_csv": str(eval_path),
        "top_n": int(top_n),
        "n_targets": int(len(merged)),
        "n_all_true": int((merged["category"] == "all_true").sum()),
        "n_all_false": int((merged["category"] == "all_false").sum()),
        "n_partial": int((merged["category"] == "partial").sum()),
    }
    return merged, meta


def _summary_value(series: pd.Series, stat: str) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    if stat == "share":
        return float(clean.mean())
    if stat == "mean":
        return float(clean.mean())
    if stat == "median":
        return float(clean.median())
    raise ValueError(f"Unknown stat: {stat}")


def _build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (run_id, category), sub_df in df.groupby(["run_id", "category"], sort=True):
        row: dict[str, object] = {
            "run_id": run_id,
            "category": category,
            "n_targets": int(len(sub_df)),
        }
        for metric, stat in SUMMARY_SPECS:
            if metric not in sub_df.columns:
                row[f"{metric}_{stat}"] = float("nan")
                continue
            row[f"{metric}_{stat}"] = _summary_value(sub_df[metric], stat)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["run_id", "category"]).reset_index(drop=True)


def _build_proximity_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "proximity_label" not in df.columns:
        return pd.DataFrame(columns=["run_id", "category", "proximity_label", "n_targets", "share"])

    rows: list[dict[str, object]] = []
    for (run_id, category), sub_df in df.groupby(["run_id", "category"], sort=True):
        counts = sub_df["proximity_label"].value_counts(dropna=False)
        total = len(sub_df)
        for label, n_targets in counts.items():
            rows.append(
                {
                    "run_id": run_id,
                    "category": category,
                    "proximity_label": label,
                    "n_targets": int(n_targets),
                    "share": float(n_targets / total) if total else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["run_id", "category", "proximity_label"]).reset_index(drop=True)


def _pointbiserial(feature: pd.Series, outcome: pd.Series) -> tuple[float, float, int]:
    sample = pd.DataFrame({"x": feature, "y": outcome}).dropna()
    if len(sample) < 3 or sample["x"].nunique() < 2 or sample["y"].nunique() < 2:
        return float("nan"), float("nan"), int(len(sample))
    effect, p_value = stats.pointbiserialr(sample["y"].to_numpy(), sample["x"].to_numpy())
    return float(effect), float(p_value), int(len(sample))


def _build_association_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, run_df in df.groupby("run_id", sort=True):
        for outcome_col in ("is_all_true", "is_all_false", "is_shortest_ping_correct"):
            for metric in ASSOCIATION_METRICS:
                if metric not in run_df.columns:
                    continue
                effect, p_value, n = _pointbiserial(run_df[metric], run_df[outcome_col])
                positive = run_df[run_df[outcome_col] == 1.0][metric]
                negative = run_df[run_df[outcome_col] == 0.0][metric]
                rows.append(
                    {
                        "run_id": run_id,
                        "outcome": outcome_col,
                        "metric": metric,
                        "effect": effect,
                        "p_value": p_value,
                        "n": n,
                        "mean_when_true": float(positive.dropna().mean()) if positive.notna().any() else float("nan"),
                        "mean_when_false": float(negative.dropna().mean()) if negative.notna().any() else float("nan"),
                        "median_when_true": float(positive.dropna().median()) if positive.notna().any() else float("nan"),
                        "median_when_false": float(negative.dropna().median()) if negative.notna().any() else float("nan"),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_effect"] = out["effect"].abs()
    return out.sort_values(["run_id", "outcome", "abs_effect"], ascending=[True, True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Condition-as-classifier: treat (has_vp_proximity & shortest_ping_vp_is_discriminative)
# as a binary predictor and evaluate it as a classifier of is_all_true / is_all_false.
# High precision => sufficient condition; high recall => necessary condition.
# MCC (= phi coefficient for 2×2) captures both in a single [-1, 1] score.
# ---------------------------------------------------------------------------

CONDITION_OUTCOMES = ["is_all_true", "is_all_false", "is_shortest_ping_correct"]


def _safe_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return float("nan")
    return float((tp * tn - fp * fn) / denom ** 0.5)


def _build_condition_classifier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether (has_vp_proximity=1 AND shortest_ping_vp_is_discriminative=1)
    is a sufficient (precision) and necessary (recall) condition for each binary outcome.

    Columns returned per (run_id, outcome) row:
        tp, fp, tn, fn,
        precision  (sufficiency: P(outcome=1 | condition=1)),
        recall     (necessity:   P(condition=1 | outcome=1)),
        f1, mcc,
        chi2, chi2_p_value   (significance of the association),
        n_valid
    """
    required = {"has_vp_proximity", "shortest_ping_vp_is_discriminative"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for run_id, run_df in df.groupby("run_id", sort=True):
        condition = (run_df["has_vp_proximity"] == 1) & (
            run_df["shortest_ping_vp_is_discriminative"] == 1
        )
        for outcome_col in CONDITION_OUTCOMES:
            if outcome_col not in run_df.columns:
                continue
            outcome = run_df[outcome_col]
            valid = condition.notna() & outcome.notna()
            c = condition[valid].astype(bool)
            o = (outcome[valid] == 1.0)
            n = int(valid.sum())
            if n < 3:
                continue

            tp = int((c & o).sum())
            fp = int((c & ~o).sum())
            fn = int((~c & o).sum())
            tn = int((~c & ~o).sum())

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
            f1 = (
                float(2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else float("nan")
            )
            mcc = _safe_mcc(tp, tn, fp, fn)

            # Chi-squared test of independence on the 2×2 table.
            # Skip if any marginal is zero (degenerate table; chi2 undefined).
            contingency = np.array([[tp, fp], [fn, tn]], dtype=float)
            row_sums = contingency.sum(axis=1)
            col_sums = contingency.sum(axis=0)
            if (row_sums == 0).any() or (col_sums == 0).any():
                chi2_val, chi2_p = float("nan"), float("nan")
            else:
                chi2_val, chi2_p, *_ = stats.chi2_contingency(contingency, correction=False)

            rows.append(
                {
                    "run_id": run_id,
                    "outcome": outcome_col,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "chi2": float(chi2_val),
                    "chi2_p_value": float(chi2_p),
                    "n_valid": n,
                }
            )
    return pd.DataFrame(rows).sort_values(["run_id", "outcome"]).reset_index(drop=True)


def _write_outputs(
    out_dir: Path,
    merged: pd.DataFrame,
    meta: pd.DataFrame,
    category_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
    association_summary: pd.DataFrame,
    condition_classifier: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "merged_per_target.parquet", index=False)
    meta.to_csv(out_dir / "run_summary.csv", index=False)
    category_summary.to_csv(out_dir / "category_proximity_summary.csv", index=False)
    label_summary.to_csv(out_dir / "category_proximity_label_summary.csv", index=False)
    association_summary.to_csv(out_dir / "binary_association_summary.csv", index=False)
    condition_classifier.to_csv(out_dir / "condition_classifier_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Associate pre-benchmark per-target eval metrics with post-classification target categories."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        required=True,
        help="Benchmark config YAML files whose run_id outputs should be analyzed.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Which classification_analysis_topN CSV to load. Default: 1.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: scripts/analysis/correlation/outputs/classification_category_association_top{N}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (DEFAULT_OUT_DIR / f"classification_category_association_top{args.top_n}")

    frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []
    for config_path in args.configs:
        frame, meta = _load_run_frame(config_path, top_n=args.top_n)
        frames.append(frame)
        meta_rows.append(meta)

    merged = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(meta_rows).sort_values("run_id").reset_index(drop=True)
    category_summary = _build_category_summary(merged)
    label_summary = _build_proximity_label_summary(merged)
    association_summary = _build_association_summary(merged)
    condition_classifier = _build_condition_classifier_summary(merged)
    _write_outputs(out_dir, merged, meta, category_summary, label_summary, association_summary, condition_classifier)

    print(f"Wrote outputs to {out_dir}")
    print("run counts:")
    print(meta[["run_id", "n_targets", "n_all_true", "n_all_false", "n_partial"]].to_string(index=False))
    print("top proximity effects:")
    if association_summary.empty:
        print("  no associations computed")
    else:
        top = association_summary.groupby(["run_id", "outcome"], as_index=False).head(5)
        print(top[["run_id", "outcome", "metric", "effect", "mean_when_true", "mean_when_false", "n"]].to_string(index=False))
    print("\ncondition (has_prox & has_sping) as classifier:")
    if condition_classifier.empty:
        print("  required columns not available")
    else:
        print(condition_classifier[["run_id", "outcome", "precision", "recall", "f1", "mcc", "chi2_p_value", "tp", "fp", "fn", "tn"]].to_string(index=False))


if __name__ == "__main__":
    main()