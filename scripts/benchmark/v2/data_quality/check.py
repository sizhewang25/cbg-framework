from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.analysis.plot_stratification import load_partition, plot_stratification
from scripts.benchmark.v2.sources.cluster_ground_truth import cluster_ground_truth
from scripts.libs.cbg.rtt_model import haversine_distance
from scripts.visualization.cluster.plot_ground_truth_clusters import plot_clusters
from scripts.visualization.cluster.plot_targets_vps import plot_vp_target_flows

# 2/3 c where c=300 km/ms -> 200 km/ms
SPEED_OF_INTERNET_KM_MS = 200.0

# Quality gate thresholds
MIN_VPS_PER_TARGET = 3
LONG_PAIR_KM = 10_000.0
STRAT_RATIO_WARN = 1.2
SMALL_RTT_MS = 5.0
LONG_DIST_FOR_SMALL_RTT_KM = 500.0
LONG_RTT_MS = 100.0
SHORT_DIST_FOR_LONG_RTT_KM = 100.0


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"config must be a YAML mapping: {path}")
    return cfg


def _require_cfg(cfg: dict[str, Any], key: str) -> Any:
    val = cfg.get(key)
    if val is None:
        raise SystemExit(f"missing required config key: {key}")
    return val


def _normalize_raw_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    required = [
        "vp_id",
        "vp_lat",
        "vp_lon",
        "target_id",
        "target_lat",
        "target_lon",
        "rtt_ms",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"raw CSV missing required columns: {missing}")

    for c in ("vp_lat", "vp_lon", "target_lat", "target_lon", "rtt_ms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["vp_id"] = df["vp_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)
    return df


def _pair_min_rtt_and_distance(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby(["vp_id", "target_id"], dropna=False)
        .agg(
            vp_lat=("vp_lat", "median"),
            vp_lon=("vp_lon", "median"),
            target_lat=("target_lat", "median"),
            target_lon=("target_lon", "median"),
            rtt_min_ms=("rtt_ms", "min"),
            n_obs=("rtt_ms", "size"),
        )
        .reset_index()
    )

    valid = grp[["vp_lat", "vp_lon", "target_lat", "target_lon", "rtt_min_ms"]].notna().all(axis=1)
    grp = grp[valid].copy()

    grp["distance_km"] = grp.apply(
        lambda r: haversine_distance(
            float(r["vp_lat"]),
            float(r["vp_lon"]),
            float(r["target_lat"]),
            float(r["target_lon"]),
        ),
        axis=1,
    )
    grp["implied_km"] = grp["rtt_min_ms"] * 0.5 * SPEED_OF_INTERNET_KM_MS
    grp["violation"] = grp["implied_km"] + 1e-9 < grp["distance_km"]
    grp["violation_ratio"] = np.where(
        grp["implied_km"] > 0,
        grp["distance_km"] / grp["implied_km"],
        np.nan,
    )
    return grp


def _plot_pair_count_distribution(pair_counts: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    vals = pair_counts.to_numpy(dtype=float)
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No pairs", ha="center", va="center")
    else:
        p5 = float(np.percentile(vals, 5))
        minority = vals < p5
        bins = np.arange(1, int(vals.max()) + 2) - 0.5
        ax.hist(vals[~minority], bins=bins, color="#4c78a8", alpha=0.8, label="normal")
        if minority.any():
            ax.hist(vals[minority], bins=bins, color="#e45756", alpha=0.9, label=f"minority (<p5={p5:.1f})")
        ax.set_xlabel("observations per VP-target pair")
        ax.set_ylabel("pair count")
        ax.set_title("VP-target pair observation count distribution")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_pair_distance_distribution(pair_stats: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    d = pair_stats["distance_km"].to_numpy(dtype=float)

    if len(d) == 0:
        ax1.text(0.5, 0.5, "No pair distances", ha="center", va="center")
        ax2.text(0.5, 0.5, "No pair distances", ha="center", va="center")
    else:
        p95 = float(np.percentile(d, 95))
        long_mask = d > p95

        ax1.hist(d[~long_mask], bins=40, color="#72b7b2", alpha=0.8, label="<= p95")
        if long_mask.any():
            ax1.hist(d[long_mask], bins=40, color="#f58518", alpha=0.9, label=f"> p95 ({p95:.1f} km)")
        ax1.axvline(p95, color="#f58518", linestyle="--", linewidth=1.2)
        ax1.set_xlabel("VP-target haversine distance (km)")
        ax1.set_ylabel("pair count")
        ax1.set_title("Distance histogram")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend()

        d_sorted = np.sort(d)
        cdf = np.arange(1, len(d_sorted) + 1) / len(d_sorted)
        ax2.plot(d_sorted, cdf, color="#54a24b", linewidth=2.0)
        ax2.axvline(p95, color="#f58518", linestyle="--", linewidth=1.2)
        ax2.set_xlabel("VP-target haversine distance (km)")
        ax2.set_ylabel("CDF")
        ax2.set_title("Distance CDF")
        ax2.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _format_id_list(ids: list[str], max_items: int = 20) -> str:
    if not ids:
        return "none"
    shown = ids[:max_items]
    suffix = "" if len(ids) <= max_items else f" ... (+{len(ids) - max_items} more)"
    return ", ".join(shown) + suffix


def _write_summary_md(
    out_path: Path,
    *,
    run_id: str,
    source: str,
    setup: str,
    n_rows: int,
    n_vps: int,
    n_targets: int,
    missing_vp_coord_ids: list[str],
    missing_target_coord_ids: list[str],
    vp_city_missing: tuple[int, int] | None,
    target_city_missing: tuple[int, int] | None,
    low_coverage: pd.DataFrame,
    long_pairs: pd.DataFrame,
    speed_violations: pd.DataFrame,
    small_rtt_long_dist: pd.DataFrame,
    long_rtt_short_dist: pd.DataFrame,
    fold_counts: pd.Series,
) -> None:
    lines: list[str] = []
    lines.append(f"# Data Quality Summary: {run_id}")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Source: {source}")
    lines.append(f"- Setup: {setup}")
    lines.append(f"- Raw observations: {n_rows:,}")
    lines.append(f"- Unique VPs: {n_vps:,}")
    lines.append(f"- Unique targets: {n_targets:,}")
    lines.append("")
    lines.append("## Quality Gates")
    lines.append("")

    has_missing_coords = bool(missing_vp_coord_ids or missing_target_coord_ids)
    if has_missing_coords:
        lines.append("### ⚠️ Missing Coordinates")
        lines.append(f"- VPs with missing lat/lon: {len(missing_vp_coord_ids):,}")
        lines.append(f"- Targets with missing lat/lon: {len(missing_target_coord_ids):,}")
        lines.append(f"- VP IDs: {_format_id_list(missing_vp_coord_ids)}")
        lines.append(f"- Target IDs: {_format_id_list(missing_target_coord_ids)}")
    else:
        lines.append("### ✅ Missing Coordinates")
        lines.append("- No VP/target coordinates are missing.")
    lines.append("")

    city_warn = False
    city_text: list[str] = []
    if vp_city_missing is None:
        city_warn = True
        city_text.append("- `vp_city` column missing from raw CSV.")
    else:
        miss, total = vp_city_missing
        if miss > 0:
            city_warn = True
        city_text.append(f"- VP city missing: {miss:,} / {total:,} ({(100.0 * miss / max(total, 1)):.1f}%)")

    if target_city_missing is None:
        city_warn = True
        city_text.append("- `target_city` column missing from raw CSV.")
    else:
        miss, total = target_city_missing
        if miss > 0:
            city_warn = True
        city_text.append(f"- Target city missing: {miss:,} / {total:,} ({(100.0 * miss / max(total, 1)):.1f}%)")

    lines.append("### ⚠️ Missing City Metadata" if city_warn else "### ✅ Missing City Metadata")
    lines.extend(city_text)
    lines.append("")

    if low_coverage.empty:
        lines.append("### ✅ Low VP Coverage Per Target")
        lines.append(f"- All targets have at least {MIN_VPS_PER_TARGET} unique VPs.")
    else:
        lines.append("### ⚠️ Low VP Coverage Per Target")
        lines.append(f"- {len(low_coverage):,} targets have < {MIN_VPS_PER_TARGET} unique VPs.")
        lines.append("")
        lines.append("| target_id | n_vps |")
        lines.append("|---|---:|")
        for r in low_coverage.head(20).itertuples(index=False):
            lines.append(f"| {r.target_id} | {int(r.n_vps)} |")
    lines.append("")

    if long_pairs.empty:
        lines.append("### ✅ Long-Distance Pairs")
        lines.append(f"- No VP-target pairs exceed {LONG_PAIR_KM:,.0f} km.")
    else:
        lines.append("### ⚠️ Long-Distance Pairs")
        lines.append(f"- {len(long_pairs):,} pairs exceed {LONG_PAIR_KM:,.0f} km.")
        lines.append("")
        lines.append("| vp_id | target_id | distance_km |")
        lines.append("|---|---|---:|")
        for r in long_pairs.sort_values("distance_km", ascending=False).head(10).itertuples(index=False):
            lines.append(f"| {r.vp_id} | {r.target_id} | {r.distance_km:.1f} |")
    lines.append("")

    if speed_violations.empty:
        lines.append("### ✅ Speed-of-Internet Violations")
        lines.append("- No physically impossible pairs under 2/3c bound.")
    else:
        lines.append("### ⚠️ Speed-of-Internet Violations")
        lines.append(
            "- Pairs where implied distance from min RTT is smaller than labeled geodesic distance "
            "(suggesting bad VP/target coordinates or measurement issues)."
        )
        lines.append(f"- Violating pairs: {len(speed_violations):,}")
        lines.append("")
        lines.append("| vp_id | target_id | rtt_min_ms | implied_km | distance_km | ratio |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in speed_violations.sort_values("violation_ratio", ascending=False).head(10).itertuples(index=False):
            lines.append(
                f"| {r.vp_id} | {r.target_id} | {r.rtt_min_ms:.3f} | {r.implied_km:.1f} | "
                f"{r.distance_km:.1f} | {r.violation_ratio:.3f} |"
            )
    lines.append("")

    anomaly_warn = (not small_rtt_long_dist.empty) or (not long_rtt_short_dist.empty)
    lines.append("### ⚠️ RTT-Distance Anomalies" if anomaly_warn else "### ✅ RTT-Distance Anomalies")
    lines.append(
        f"- Small RTT + long distance: RTT < {SMALL_RTT_MS:.1f} ms and distance > {LONG_DIST_FOR_SMALL_RTT_KM:.0f} km"
    )
    lines.append(f"  - Count: {len(small_rtt_long_dist):,}")
    lines.append(
        f"- Long RTT + short distance: RTT > {LONG_RTT_MS:.0f} ms and distance < {SHORT_DIST_FOR_LONG_RTT_KM:.0f} km"
    )
    lines.append(f"  - Count: {len(long_rtt_short_dist):,}")
    if not small_rtt_long_dist.empty:
        lines.append("- Top small-RTT/long-distance examples:")
        lines.append("| vp_id | target_id | rtt_min_ms | distance_km |")
        lines.append("|---|---|---:|---:|")
        for r in small_rtt_long_dist.sort_values("distance_km", ascending=False).head(5).itertuples(index=False):
            lines.append(f"| {r.vp_id} | {r.target_id} | {r.rtt_min_ms:.3f} | {r.distance_km:.1f} |")
    if not long_rtt_short_dist.empty:
        lines.append("- Top long-RTT/short-distance examples:")
        lines.append("| vp_id | target_id | rtt_min_ms | distance_km |")
        lines.append("|---|---|---:|---:|")
        for r in long_rtt_short_dist.sort_values("rtt_min_ms", ascending=False).head(5).itertuples(index=False):
            lines.append(f"| {r.vp_id} | {r.target_id} | {r.rtt_min_ms:.3f} | {r.distance_km:.1f} |")
    lines.append("")

    if fold_counts.empty:
        lines.append("### ⚠️ Stratification Balance")
        lines.append("- Could not read materialized fold partition.")
    else:
        mn = int(fold_counts.min())
        mx = int(fold_counts.max())
        ratio = float(mx / max(mn, 1))
        warn = ratio > STRAT_RATIO_WARN
        lines.append("### ⚠️ Stratification Balance" if warn else "### ✅ Stratification Balance")
        lines.append(f"- Fold size max/min ratio: {ratio:.3f} (threshold {STRAT_RATIO_WARN:.2f})")
        lines.append("- Per-fold target counts:")
        for fold, n in fold_counts.sort_index().items():
            lines.append(f"  - fold {int(fold)}: {int(n):,}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _write_clusters_outputs(targets: pd.DataFrame, clusters_dir: Path, radius_km: float) -> tuple[pd.DataFrame, Any]:
    res = cluster_ground_truth(
        targets["target_lat"].to_numpy(dtype=float),
        targets["target_lon"].to_numpy(dtype=float),
        radius_km=radius_km,
    )

    clusters_dir.mkdir(parents=True, exist_ok=True)

    clusters = pd.DataFrame(
        {
            "cluster_id": np.arange(res.n_clusters),
            "centroid_lat": res.centroid_lat,
            "centroid_lon": res.centroid_lon,
            "n_members": res.member_counts,
            "radius_km": np.round(res.radius_km, 3),
            "diameter_km": np.round(res.diameter_km, 3),
            "is_singleton": res.member_counts == 1,
        }
    )
    assignments = pd.DataFrame(
        {
            "target_id": targets["target_id"].to_numpy(),
            "target_lat": targets["target_lat"].to_numpy(),
            "target_lon": targets["target_lon"].to_numpy(),
            "cluster_id": res.labels,
            "dist_to_centroid_km": np.round(res.dist_km, 3),
        }
    )
    meta = {
        "radius_km": float(radius_km),
        "n_targets": int(len(targets)),
        "n_clusters": int(res.n_clusters),
        "n_singletons": int((res.member_counts == 1).sum()),
    }

    clusters.to_csv(clusters_dir / "clusters.csv", index=False)
    assignments.to_csv(clusters_dir / "assignments.csv", index=False)
    (clusters_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return assignments, res


def run_checks(args: argparse.Namespace) -> dict[str, Path]:
    cfg = _load_config(args.config)
    source = str(_require_cfg(cfg, "source"))
    run_id = str(_require_cfg(cfg, "run_id"))
    setup = str(cfg.get("setup", "probes_to_anchors"))
    source_kwargs = cfg.get("source_kwargs") or {}
    if not isinstance(source_kwargs, dict):
        raise SystemExit("config source_kwargs must be a mapping")

    csv_path_raw = source_kwargs.get("csv_path")
    if csv_path_raw is None:
        raise SystemExit("config source_kwargs.csv_path is required")

    csv_path = Path(str(csv_path_raw))
    if not csv_path.is_absolute():
        # Most benchmark configs store csv_path relative to repo root.
        cwd_candidate = (Path.cwd() / csv_path).resolve()
        cfg_candidate = (args.config.parent / csv_path).resolve()
        if cwd_candidate.exists():
            csv_path = cwd_candidate
        else:
            csv_path = cfg_candidate
    if not csv_path.exists():
        raise SystemExit(f"raw CSV not found: {csv_path}")

    inputs_dir = args.inputs_root / source / run_id / setup
    if not inputs_dir.exists():
        raise SystemExit(f"materialized inputs dir not found for stratification: {inputs_dir}")

    out_dir = args.out_dir or (
        Path(__file__).resolve().parent / "outputs" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _normalize_raw_csv(pd.read_csv(csv_path))
    n_rows = len(df)
    n_vps = int(df["vp_id"].nunique())
    n_targets = int(df["target_id"].nunique())

    # Missing coordinates
    missing_vp_coord_ids = sorted(df[df[["vp_lat", "vp_lon"]].isna().any(axis=1)]["vp_id"].unique().tolist())
    missing_target_coord_ids = sorted(
        df[df[["target_lat", "target_lon"]].isna().any(axis=1)]["target_id"].unique().tolist()
    )

    # Missing city
    vp_city_missing = None
    if "vp_city" in df.columns:
        by_vp = df[["vp_id", "vp_city"]].drop_duplicates("vp_id")
        miss = int(by_vp["vp_city"].isna().sum() + (by_vp["vp_city"].astype(str).str.strip() == "").sum())
        vp_city_missing = (miss, len(by_vp))

    target_city_missing = None
    if "target_city" in df.columns:
        by_tg = df[["target_id", "target_city"]].drop_duplicates("target_id")
        miss = int(by_tg["target_city"].isna().sum() + (by_tg["target_city"].astype(str).str.strip() == "").sum())
        target_city_missing = (miss, len(by_tg))

    # Pair counts / coverage
    pair_counts = df.groupby(["vp_id", "target_id"]).size().rename("n_obs")
    target_vp_counts = (
        df.groupby("target_id")["vp_id"].nunique().rename("n_vps").reset_index()
    )
    low_coverage = target_vp_counts[target_vp_counts["n_vps"] < MIN_VPS_PER_TARGET].sort_values("n_vps")

    # Pair min RTT + distances + violations
    pair_stats = _pair_min_rtt_and_distance(df)
    long_pairs = pair_stats[pair_stats["distance_km"] > LONG_PAIR_KM].copy()
    speed_violations = pair_stats[pair_stats["violation"]].copy()

    small_rtt_long_dist = pair_stats[
        (pair_stats["rtt_min_ms"] < SMALL_RTT_MS)
        & (pair_stats["distance_km"] > LONG_DIST_FOR_SMALL_RTT_KM)
    ].copy()
    long_rtt_short_dist = pair_stats[
        (pair_stats["rtt_min_ms"] > LONG_RTT_MS)
        & (pair_stats["distance_km"] < SHORT_DIST_FOR_LONG_RTT_KM)
    ].copy()

    # Check 2 plot
    pair_count_plot = out_dir / "pair_count_distribution.png"
    _plot_pair_count_distribution(pair_counts, pair_count_plot)

    # Check 3 plot
    pair_distance_plot = out_dir / "pair_distance_distribution.png"
    _plot_pair_distance_distribution(pair_stats, pair_distance_plot)

    # Check 4 flow map (raw CSV)
    flow_pairs = pair_stats[["vp_lat", "vp_lon", "target_lat", "target_lon"]].drop_duplicates()
    flow_plot = out_dir / "vp_target_flows.png"
    extent = tuple(args.extent) if args.extent is not None else None
    plot_vp_target_flows(flow_pairs, flow_plot, extent=extent)

    # Check 5 stratification (materialized folds)
    strat_plot = out_dir / "stratification.png"
    plot_stratification(inputs_dir, strat_plot)
    partition = load_partition(inputs_dir)
    fold_counts = partition.groupby("fold").size().sort_index()

    # Check 6 target cluster map (raw CSV targets)
    targets = (
        df[["target_id", "target_lat", "target_lon"]]
        .dropna(subset=["target_lat", "target_lon"])
        .drop_duplicates(subset=["target_id"])
        .copy()
    )
    clusters_dir = out_dir / "clusters"
    assignments, res = _write_clusters_outputs(targets, clusters_dir, radius_km=float(args.radius_km))
    cluster_plot = out_dir / "ground_truth_clusters.png"
    plot_clusters(assignments, res, cluster_plot, extent=extent)

    # Summary markdown
    summary_md = out_dir / "summary.md"
    _write_summary_md(
        summary_md,
        run_id=run_id,
        source=source,
        setup=setup,
        n_rows=n_rows,
        n_vps=n_vps,
        n_targets=n_targets,
        missing_vp_coord_ids=missing_vp_coord_ids,
        missing_target_coord_ids=missing_target_coord_ids,
        vp_city_missing=vp_city_missing,
        target_city_missing=target_city_missing,
        low_coverage=low_coverage,
        long_pairs=long_pairs,
        speed_violations=speed_violations,
        small_rtt_long_dist=small_rtt_long_dist,
        long_rtt_short_dist=long_rtt_short_dist,
        fold_counts=fold_counts,
    )

    print(f"Raw CSV: {csv_path}")
    print(f"Materialized inputs: {inputs_dir}")
    print(f"Observations: {n_rows:,}, VPs: {n_vps:,}, targets: {n_targets:,}")
    print(f"Speed-of-internet violations: {len(speed_violations):,}")
    print(f"Wrote {summary_md}")

    return {
        "summary_md": summary_md,
        "pair_count_distribution": pair_count_plot,
        "pair_distance_distribution": pair_distance_plot,
        "vp_target_flows": flow_plot,
        "stratification": strat_plot,
        "ground_truth_clusters": cluster_plot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run data quality checks for a benchmark config (raw CSV + materialized stratification)."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark v2 YAML config.")
    parser.add_argument(
        "--inputs-root",
        type=Path,
        default=Path("scripts/benchmark/v2/inputs"),
        help="Root containing materialized inputs for stratification check.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to scripts/benchmark/v2/data_quality/outputs/<run_id>/",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=50.0,
        help="Radius cap for target clustering visualization.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=None,
        metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"),
        help="Optional map extent for flow and cluster maps.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"config not found: {args.config}")

    run_checks(args)


if __name__ == "__main__":
    main()
