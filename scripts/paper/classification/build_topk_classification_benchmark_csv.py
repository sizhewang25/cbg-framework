"""Build a combined top-k classification benchmark CSV across multiple configs.

Reads one or more analysis cluster config YAML files and aggregates per-model
classification accuracy for top-1 and top-3 from each run's `cluster_scored`
outputs.

Output columns:
    run_id, model, top1_cls_acc, top3_cls_acc

Notes:
- `model` comes from each config's `classification_combos` labels.
- Accuracies are fractions in [0, 1] (not percentages), matching benchmark-table
  friendly numeric format.
- Per-model score files are expected at:
  <v2_outputs_root>/<run_id>/<source>/<setup>/cluster_scored/<combo>_scored.csv
- Shortest-ping baseline is read from:
    <v2_outputs_root>/<run_id>/<source>/<setup>/cluster_scored/baseline.csv

Example:
    /home/sw6456/geomodel/cbg-framework/.venv/bin/python \
      -m scripts.paper.classification.build_topk_classification_benchmark_csv \
      --configs \
        scripts/analysis/config/clusters/config-final.yaml \
        scripts/analysis/config/clusters/config-final.yaml \
        scripts/analysis/config/clusters/config-final.yaml \
      --out-csv scripts/paper/classification/topk_classification_benchmark.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import yaml


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _scored_dir_from_config(cfg: dict) -> Path:
    v2_root = Path(cfg.get("v2_outputs_root", "scripts/benchmark/v2/outputs"))
    run_id = cfg["run_id"]
    source = cfg["source"]
    setup = cfg.get("setup", "probes_to_anchors")
    return v2_root / run_id / source / setup / "cluster_scored"


def _inputs_dir_from_config(cfg: dict) -> Path:
    v2_root = Path(cfg.get("v2_inputs_root", "scripts/benchmark/v2/inputs"))
    run_id = cfg["run_id"]
    source = cfg["source"]
    setup = cfg.get("setup", "probes_to_anchors")
    return v2_root / source / run_id / setup


def _resolve_eval_observation_paths(inputs_dir: Path) -> list[Path]:
    direct = inputs_dir / "eval_observations.parquet"
    if direct.exists():
        return [direct]
    return sorted(inputs_dir.glob("*/eval_observations.parquet"))


def _load_target_weights(inputs_dir: Path) -> pd.DataFrame:
    paths = _resolve_eval_observation_paths(inputs_dir)
    if not paths:
        raise FileNotFoundError(
            f"no eval_observations.parquet at {inputs_dir} or {inputs_dir}/*/"
        )

    def _read_obs(path: Path) -> pd.DataFrame:
        # Prefer city-level location keys when available; otherwise use lat/lon.
        try:
            return pd.read_parquet(
                path,
                columns=["target_id", "target_city", "vp_city", "weight"],
            )
        except Exception:
            return pd.read_parquet(
                path,
                columns=[
                    "target_id",
                    "target_lat",
                    "target_lon",
                    "vp_lat",
                    "vp_lon",
                    "weight",
                ],
            )

    frames = [_read_obs(p) for p in paths]
    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    req = {"target_id", "weight"}
    missing = req - set(obs.columns)
    if missing:
        raise ValueError(
            f"eval observations under {inputs_dir} missing columns: {sorted(missing)}"
        )

    if obs.empty:
        return pd.DataFrame(columns=["target_id", "target_weight"])

    obs = obs.copy()
    obs["target_id"] = obs["target_id"].astype(str)

    use_city_keys = {"target_city", "vp_city"}.issubset(set(obs.columns))
    if use_city_keys:
        tgt_loc = obs["target_city"].astype(str).str.strip()
        vp_loc = obs["vp_city"].astype(str).str.strip()
        blank_tgt = ~obs["target_city"].notna() | (tgt_loc == "")
        blank_vp = ~obs["vp_city"].notna() | (vp_loc == "")
        bad = blank_tgt | blank_vp
        if bad.any():
            warnings.warn(
                f"{inputs_dir}: dropping {int(bad.sum())} rows with blank city location keys",
                stacklevel=2,
            )
            obs = obs.loc[~bad].copy()
            tgt_loc = obs["target_city"].astype(str).str.strip()
            vp_loc = obs["vp_city"].astype(str).str.strip()

        obs["target_loc_key"] = tgt_loc
        obs["vp_loc_key"] = vp_loc
    else:
        need = {"target_lat", "target_lon", "vp_lat", "vp_lon"}
        missing_loc = need - set(obs.columns)
        if missing_loc:
            raise ValueError(
                f"eval observations under {inputs_dir} missing location columns: {sorted(missing_loc)}"
            )

        bad = (
            obs[["target_lat", "target_lon", "vp_lat", "vp_lon"]]
            .isna()
            .any(axis=1)
        )
        if bad.any():
            warnings.warn(
                f"{inputs_dir}: dropping {int(bad.sum())} rows with missing lat/lon location keys",
                stacklevel=2,
            )
            obs = obs.loc[~bad].copy()

        obs["target_loc_key"] = (
            obs["target_lat"].map(lambda x: f"{float(x):.8f}")
            + ","
            + obs["target_lon"].map(lambda x: f"{float(x):.8f}")
        )
        obs["vp_loc_key"] = (
            obs["vp_lat"].map(lambda x: f"{float(x):.8f}")
            + ","
            + obs["vp_lon"].map(lambda x: f"{float(x):.8f}")
        )

    if obs.empty:
        return pd.DataFrame(columns=["target_id", "target_weight"])

    # Dedup VP->target-location traffic first, mirroring eval pair semantics.
    per_pair = (
        obs.groupby(["vp_loc_key", "target_loc_key"], as_index=False)
        .agg(weight=("weight", "max"))
    )
    city_weight = (
        per_pair.groupby("target_loc_key", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "target_weight"})
    )

    id_city = obs[["target_id", "target_loc_key"]].drop_duplicates()
    dup_city_map = id_city.groupby("target_id")["target_loc_key"].nunique()
    n_inconsistent = int((dup_city_map > 1).sum())
    if n_inconsistent:
        warnings.warn(
            "target_id maps to multiple target location keys; keeping first mapping "
            f"for {n_inconsistent} target_id(s)",
            stacklevel=2,
        )
        id_city = id_city.drop_duplicates(subset=["target_id"], keep="first")

    tgt = id_city.merge(city_weight, on="target_loc_key", how="left")
    if tgt["target_weight"].isna().any():
        missing_w = int(tgt["target_weight"].isna().sum())
        warnings.warn(
            f"{inputs_dir}: {missing_w} target_id(s) missing location weight after "
            "dedup aggregation; dropping them",
            stacklevel=2,
        )
        tgt = tgt.dropna(subset=["target_weight"])

    return tgt[["target_id", "target_weight"]]


def _load_target_clusters(assignments_csv: Path) -> pd.DataFrame:
    if not assignments_csv.exists():
        raise FileNotFoundError(f"missing assignments file: {assignments_csv}")

    cols = ["target_id", "cluster_id"]
    df = pd.read_csv(assignments_csv, usecols=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)

    return df.drop_duplicates(subset=["target_id"], keep="first")


def _weighted_cluster_accuracy(
    match_df: pd.DataFrame,
    match_col: str,
    target_clusters: pd.DataFrame,
    target_weights: pd.DataFrame,
    context: str,
) -> float:
    required = {"target_id", match_col}
    missing = required - set(match_df.columns)
    if missing:
        raise ValueError(f"{context} missing columns: {sorted(missing)}")

    work = match_df[["target_id", match_col]].copy()
    start_rows = len(work)

    joined = work.merge(target_clusters, on="target_id", how="left")
    miss_cluster = int(joined["cluster_id"].isna().sum())
    if miss_cluster:
        warnings.warn(
            f"{context}: dropped {miss_cluster}/{start_rows} rows with missing cluster_id",
            stacklevel=2,
        )
    joined = joined.dropna(subset=["cluster_id"])

    joined = joined.merge(target_weights, on="target_id", how="left")
    miss_weight = int(joined["target_weight"].isna().sum())
    if miss_weight:
        warnings.warn(
            f"{context}: dropped {miss_weight}/{start_rows} rows with missing target weight",
            stacklevel=2,
        )
    joined = joined.dropna(subset=["target_weight"])

    if joined.empty:
        raise ValueError(f"{context}: no rows left after joins with clusters/weights")

    joined[match_col] = joined[match_col].astype(bool)
    per_cluster_acc = (
        joined.groupby("cluster_id", as_index=False)[match_col]
        .mean()
        .rename(columns={match_col: "cluster_acc"})
    )
    per_cluster_w = (
        joined[["target_id", "cluster_id", "target_weight"]]
        .drop_duplicates(subset=["target_id", "cluster_id"])
        .groupby("cluster_id", as_index=False)["target_weight"]
        .sum()
        .rename(columns={"target_weight": "cluster_weight"})
    )

    agg = per_cluster_acc.merge(per_cluster_w, on="cluster_id", how="inner")
    if agg.empty:
        raise ValueError(f"{context}: no cluster overlap between accuracy and weights")

    total_w = float(agg["cluster_weight"].sum())
    if total_w <= 0:
        warnings.warn(
            f"{context}: total cluster weight is zero; returning NaN",
            stacklevel=2,
        )
        return float("nan")

    return float((agg["cluster_acc"] * agg["cluster_weight"]).sum() / total_w)


def _read_acc(csv_path: Path, col: str) -> float:
    df = pd.read_csv(csv_path, usecols=[col])
    if len(df) == 0:
        return float("nan")
    return float(df[col].mean())


def _read_baseline_acc(baseline_csv: Path) -> tuple[float, float]:
    df = pd.read_csv(baseline_csv)

    top1_col = "vp_matches_centroid_top1"
    top3_col = "vp_matches_centroid_top3"

    # Backward compatibility: older files may only have vp_matches_centroid.
    if top1_col not in df.columns:
        if "vp_matches_centroid" in df.columns:
            top1_col = "vp_matches_centroid"
        else:
            raise ValueError(
                f"{baseline_csv} missing baseline top-1 column "
                "(vp_matches_centroid_top1 or vp_matches_centroid)"
            )

    if top3_col not in df.columns:
        # If top-3 is absent, fall back to top-1 (equivalent N=1 scoring output).
        top3_col = top1_col

    if len(df) == 0:
        return float("nan"), float("nan")

    return float(df[top1_col].mean()), float(df[top3_col].mean())


def _read_weighted_model_acc(
    csv_path: Path,
    target_clusters: pd.DataFrame,
    target_weights: pd.DataFrame,
) -> tuple[float, float]:
    df = pd.read_csv(csv_path)
    top1 = _weighted_cluster_accuracy(
        match_df=df,
        match_col="match_top1",
        target_clusters=target_clusters,
        target_weights=target_weights,
        context=str(csv_path),
    )
    top3 = _weighted_cluster_accuracy(
        match_df=df,
        match_col="match_top3",
        target_clusters=target_clusters,
        target_weights=target_weights,
        context=str(csv_path),
    )
    return top1, top3


def _read_weighted_baseline_acc(
    baseline_csv: Path,
    target_clusters: pd.DataFrame,
    target_weights: pd.DataFrame,
) -> tuple[float, float]:
    df = pd.read_csv(baseline_csv)

    top1_col = "vp_matches_centroid_top1"
    top3_col = "vp_matches_centroid_top3"

    if top1_col not in df.columns:
        if "vp_matches_centroid" in df.columns:
            top1_col = "vp_matches_centroid"
        else:
            raise ValueError(
                f"{baseline_csv} missing baseline top-1 column "
                "(vp_matches_centroid_top1 or vp_matches_centroid)"
            )
    if top3_col not in df.columns:
        top3_col = top1_col

    top1 = _weighted_cluster_accuracy(
        match_df=df,
        match_col=top1_col,
        target_clusters=target_clusters,
        target_weights=target_weights,
        context=str(baseline_csv),
    )
    top3 = _weighted_cluster_accuracy(
        match_df=df,
        match_col=top3_col,
        target_clusters=target_clusters,
        target_weights=target_weights,
        context=str(baseline_csv),
    )
    return top1, top3


def build_rows(config_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []

    for cfg_path in config_paths:
        cfg = _load_yaml(cfg_path)
        run_id = str(cfg["run_id"])
        combos = cfg.get("classification_combos") or []
        if not combos:
            raise ValueError(
                f"{cfg_path} must define non-empty classification_combos"
            )

        scored_dir = _scored_dir_from_config(cfg)
        if not scored_dir.exists():
            raise FileNotFoundError(
                f"scored dir not found for {cfg_path}: {scored_dir}"
            )

        weighted_mode = bool(cfg.get("weight_cls_accuracy", False))
        target_clusters: pd.DataFrame | None = None
        target_weights: pd.DataFrame | None = None
        if weighted_mode:
            assignments_csv = scored_dir.parent / "clusters" / "assignments.csv"
            inputs_dir = _inputs_dir_from_config(cfg)
            target_clusters = _load_target_clusters(assignments_csv)
            target_weights = _load_target_weights(inputs_dir)

        for entry in combos:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    f"invalid classification_combos entry in {cfg_path}: {entry!r}"
                )

            combo_name = str(entry["name"])
            model = str(entry.get("label", combo_name))
            csv_path = scored_dir / f"{combo_name}_scored.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"missing scored file: {csv_path}")

            if weighted_mode:
                assert target_clusters is not None
                assert target_weights is not None
                top1, top3 = _read_weighted_model_acc(
                    csv_path=csv_path,
                    target_clusters=target_clusters,
                    target_weights=target_weights,
                )
            else:
                top1 = _read_acc(csv_path, "match_top1")
                top3 = _read_acc(csv_path, "match_top3")

            rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "top1_cls_acc": top1,
                    "top3_cls_acc": top3,
                }
            )

        baseline_csv = scored_dir / "baseline.csv"
        if not baseline_csv.exists():
            raise FileNotFoundError(f"missing baseline file: {baseline_csv}")
        if weighted_mode:
            assert target_clusters is not None
            assert target_weights is not None
            b_top1, b_top3 = _read_weighted_baseline_acc(
                baseline_csv=baseline_csv,
                target_clusters=target_clusters,
                target_weights=target_weights,
            )
        else:
            b_top1, b_top3 = _read_baseline_acc(baseline_csv)
        rows.append(
            {
                "run_id": run_id,
                "model": "Shortest Ping",
                "top1_cls_acc": b_top1,
                "top3_cls_acc": b_top3,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Stable benchmark ordering: by run, then config order preserved by input rows.
    return out[["run_id", "model", "top1_cls_acc", "top3_cls_acc"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        required=True,
        help="List of analysis cluster config YAML files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output combined benchmark CSV path.",
    )
    args = parser.parse_args()

    table = build_rows(args.configs)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv} ({len(table)} rows)")


if __name__ == "__main__":
    main()
