"""Classification accuracy bar chart for a configured combo subset.

Reads ``classification_combos`` from an analysis cluster config YAML (see
``scripts/analysis/config/clusters/``). Each entry is a ``{name, label}``
dict — ``name`` matches the combo_id in the scored CSVs, ``label`` is the
y-axis display string.

Differences from ``plot_cluster_match_bars``:
- No within-R metric or display.
- x-axis in percent (not fraction).
- No (n=…) annotation beside bars — just the accuracy value.
- x-axis label: "Classification Accuracy (%)".
- Shortest-ping same-centroid baseline line is kept; within-R baseline removed.

The scored dir and clusters dir are derived from the config's path fields
(``v2_outputs_root / run_id / source / setup / cluster_scored``) unless
overridden via ``--scored-dir`` / ``--clusters-dir``.

CLI:
    python -m scripts.analysis.plot_classification_match_bars \\
        --config scripts/analysis/config/clusters/europe_as3209_final.yaml
"""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from scripts.analysis._v2_io import (
    add_geo_filter_args,
    analysis_out_dir,
    route_geo_path,
    set_geo_filter_from_args,
)
from scripts.analysis._cluster_data import _read_meta

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _resolve_paths(cfg: dict, outputs_root_override=None):
    v2_root = Path(outputs_root_override or cfg.get("v2_outputs_root", "scripts/benchmark/v2/outputs"))
    run_id = cfg["run_id"]
    source = cfg["source"]
    setup = cfg.get("setup", "probes_to_anchors")
    setup_dir = v2_root / run_id / source / setup
    return v2_root / run_id, setup_dir / "cluster_scored", setup_dir / "clusters"


def _load_allowed_targets(features_csv: Path | None) -> set[str] | None:
    if features_csv is None:
        return None

    df = pd.read_csv(features_csv, usecols=["target_id"])
    return set(df["target_id"].astype(str))


def plot_bars(
    rates: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    overlay_col: str | None = None,
    overlay_label: str | None = None,
) -> plt.Figure:
    df = rates.sort_values("accuracy_pct", ascending=True)  # ascending → best on top
    y = list(range(len(df)))

    bar_colors = df["color"].tolist() if "color" in df.columns else ["#4E79A7"] * len(df)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(df) + 1.5)))
    ax.barh(
        y,
        df["accuracy_pct"],
        color=bar_colors,
        alpha=0.85,
        zorder=2,
        label="classification accuracy",
    )

    if overlay_col is not None:
        ax.barh(
            y,
            df[overlay_col],
            facecolor="none",
            edgecolor="#E15759",
            hatch="///",
            linewidth=1.2,
            zorder=3,
            label=overlay_label or overlay_col,
        )

    for yi, val in enumerate(df["accuracy_pct"]):
        if pd.notna(val):
            ax.text(val + 0.3, yi, f"{val:.1f}%", va="center", fontsize=8)

    if overlay_col is not None:
        for yi, val in enumerate(df[overlay_col]):
            if pd.notna(val):
                ax.text(val + 0.3, yi - 0.22, f"{val:.1f}%", va="center", fontsize=7, color="#E15759")

    xs = [float(df["accuracy_pct"].max()) if len(df) else 0.0]
    if overlay_col is not None and len(df):
        xs.append(float(df[overlay_col].max()))
    ax.set_xlim(0, min(100.0, max(10.0, max(xs) * 1.25)))
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=8)
    ax.set_xlabel("Classification Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out_path)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="Analysis cluster config YAML (scripts/analysis/config/clusters/).")
    parser.add_argument("--outputs-root", type=Path, default=None,
                        help="Override v2_outputs_root from config.")
    parser.add_argument("--scored-dir", type=Path, default=None,
                        help="Override the scored dir derived from config.")
    parser.add_argument("--clusters-dir", type=Path, default=None,
                        help="Override the clusters dir derived from config (for meta only).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: scripts/analysis/outputs/<run_id>/cluster).")
    parser.add_argument("--features-csv", type=Path, default=None,
                        help="Optional per-target features CSV used to filter targets before plotting.")
    parser.add_argument(
        "--match-top-k",
        type=int,
        default=1,
        help=(
            "Use top-k match columns from scored CSVs. "
            "k=1 uses legacy `match`/`vp_matches_centroid`; "
            "k>1 uses `match_top{k}`/`vp_matches_centroid_top{k}`."
        ),
    )
    parser.add_argument(
        "--topn-combo",
        type=str,
        default=None,
        help=(
            "Optional top-k overlay pair formatted like '[1,3]'. "
            "First value is the base filled bar, second value is striped overlay."
        ),
    )
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    cfg = _load_config(args.config)
    run_dir, scored_dir_default, clusters_dir_default = _resolve_paths(cfg, args.outputs_root)
    scored_dir = Path(args.scored_dir) if args.scored_dir else scored_dir_default
    clusters_dir = Path(args.clusters_dir) if args.clusters_dir else clusters_dir_default
    allowed_targets = _load_allowed_targets(args.features_csv)

    def _parse_topn_combo(raw: str) -> tuple[int, int]:
        try:
            v = ast.literal_eval(raw)
        except Exception as e:
            raise ValueError(f"invalid --topn-combo value {raw!r}: {e}") from e
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError("--topn-combo must be a list/tuple of exactly two ints, e.g. '[1,3]'")
        a, b = int(v[0]), int(v[1])
        if a < 1 or b < 1:
            raise ValueError("--topn-combo values must be >= 1")
        if a == b:
            raise ValueError("--topn-combo values must be different")
        return a, b

    if args.match_top_k < 1:
        raise ValueError("--match-top-k must be >= 1")

    combo_pair = _parse_topn_combo(args.topn_combo) if args.topn_combo else None

    if args.match_top_k == 1:
        match_col = "match"
        baseline_col = "vp_matches_centroid"
    else:
        match_col = f"match_top{args.match_top_k}"
        baseline_col = f"vp_matches_centroid_top{args.match_top_k}"

    overlay_match_col = None
    overlay_baseline_col = None
    overlay_k = None
    if combo_pair is not None:
        base_k, overlay_k = combo_pair
        if base_k == 1:
            match_col = "match"
            baseline_col = "vp_matches_centroid"
        else:
            match_col = f"match_top{base_k}"
            baseline_col = f"vp_matches_centroid_top{base_k}"

        if overlay_k == 1:
            overlay_match_col = "match"
            overlay_baseline_col = "vp_matches_centroid"
        else:
            overlay_match_col = f"match_top{overlay_k}"
            overlay_baseline_col = f"vp_matches_centroid_top{overlay_k}"

    classification_combos = cfg.get("classification_combos") or []
    if not classification_combos:
        raise ValueError("config must define classification_combos as a non-empty list of {name, label}")
    combo_map = {entry["name"]: entry["label"] for entry in classification_combos}

    radius_km = float(cfg.get("radius_km", 50))
    n_centroids, n_targets = _read_meta(clusters_dir) if clusters_dir.exists() else (0, 0)
    logger.info("answer space: %d targets → %d centroids (R=%.0f km)", n_targets, n_centroids, radius_km)

    rows = []
    for csv_path in sorted(scored_dir.glob("*_scored.csv")):
        combo_id = csv_path.stem[: -len("_scored")]
        if combo_id not in combo_map:
            continue
        df = pd.read_csv(csv_path)
        if allowed_targets is not None:
            if "target_id" not in df.columns:
                raise ValueError(
                    f"{csv_path} is missing target_id; rerun cluster-score with the updated schema first"
                )
            df = df[df["target_id"].astype(str).isin(allowed_targets)].reset_index(drop=True)
        if match_col not in df.columns:
            raise ValueError(
                f"{csv_path} is missing {match_col}; rerun cluster-score with --top-n >= {args.match_top_k}"
            )
        if overlay_match_col is not None and overlay_match_col not in df.columns:
            need_k = overlay_k if overlay_k is not None else "<k>"
            raise ValueError(
                f"{csv_path} is missing {overlay_match_col}; rerun cluster-score with --top-n >= {need_k}"
            )
        n = len(df)
        row = {
            "combo_id": combo_id,
            "label": combo_map[combo_id],
            "n": n,
            "n_scored": int(df["success"].sum()) if n else 0,
            "accuracy_pct": float(df[match_col].mean()) * 100 if n else float("nan"),
            "color": "#4E79A7",
        }
        if overlay_match_col is not None:
            row["accuracy_pct_overlay"] = float(df[overlay_match_col].mean()) * 100 if n else float("nan")
        rows.append(row)
    rates = pd.DataFrame(rows)

    if rates.empty:
        logger.warning("no matching scored CSVs found in %s for the configured combos", scored_dir)
        return

    base_acc = float("nan")
    base_acc_overlay = float("nan")
    n_base = 0
    bpath = scored_dir / "baseline.csv"
    if bpath.exists():
        bdf = pd.read_csv(bpath)
        if allowed_targets is not None:
            bdf = bdf[bdf["target_id"].astype(str).isin(allowed_targets)].reset_index(drop=True)
        if baseline_col not in bdf.columns:
            raise ValueError(
                f"{bpath} is missing {baseline_col}; rerun cluster-score with --top-n >= {args.match_top_k}"
            )
        if overlay_baseline_col is not None and overlay_baseline_col not in bdf.columns:
            need_k = overlay_k if overlay_k is not None else "<k>"
            raise ValueError(
                f"{bpath} is missing {overlay_baseline_col}; rerun cluster-score with --top-n >= {need_k}"
            )
        n_base = len(bdf)
        if n_base:
            base_acc = float(bdf[baseline_col].mean()) * 100
            if overlay_baseline_col is not None:
                base_acc_overlay = float(bdf[overlay_baseline_col].mean()) * 100
        if overlay_baseline_col is None:
            logger.info("baseline: top-%d=%.1f%% (n=%d)", args.match_top_k, base_acc, n_base)
        else:
            logger.info(
                "baseline: top-%d=%.1f%%, top-%d=%.1f%% (n=%d)",
                combo_pair[0], base_acc, combo_pair[1], base_acc_overlay, n_base
            )
    else:
        logger.warning("no baseline.csv in %s; skipping shortest-ping baseline", scored_dir)

    out_dir = route_geo_path(args.out_dir) if args.out_dir else analysis_out_dir(run_dir, "cluster")
    if combo_pair is None:
        suffix = "" if args.match_top_k == 1 else f"_top{args.match_top_k}"
    else:
        suffix = f"_top{combo_pair[0]}_{combo_pair[1]}"
    png_path = out_dir / f"{run_dir.name}_classification_accuracy{suffix}.png"

    if combo_pair is None:
        title = (
            f"Classification accuracy (top-{args.match_top_k}) — "
            f"{run_dir.name} ({n_centroids} centroids, R={radius_km:.0f} km)"
        )
        base_label = f"shortest-ping VP top-{args.match_top_k}"
        overlay_label = None
    else:
        title = (
            f"Classification accuracy (top-{combo_pair[0]} vs top-{combo_pair[1]}) — "
            f"{run_dir.name} ({n_centroids} centroids, R={radius_km:.0f} km)"
        )
        base_label = f"shortest-ping VP top-{combo_pair[0]}"
        overlay_label = f"classification accuracy top-{combo_pair[1]}"

    # Add baseline as a bar row in rates (must be after base_label is defined)
    if n_base and pd.notna(base_acc):
        baseline_row_data: dict = {
            "combo_id": "shortest_ping_baseline",
            "label": "Shortest Ping",
            "n": n_base,
            "n_scored": n_base,
            "accuracy_pct": base_acc,
            "color": "#888888",
        }
        if combo_pair is not None and pd.notna(base_acc_overlay):
            baseline_row_data["accuracy_pct_overlay"] = base_acc_overlay
        rates = pd.concat([rates, pd.DataFrame([baseline_row_data])], ignore_index=True)

    fig = plot_bars(
        rates, png_path,
        title=title,
        overlay_col="accuracy_pct_overlay" if combo_pair is not None else None,
        overlay_label=overlay_label,
    )
    plt.close(fig)

    csv = rates.sort_values("accuracy_pct", ascending=False).copy()
    csv["n_failed"] = csv["n"] - csv["n_scored"]
    csv["accuracy"] = csv["accuracy_pct"] / 100
    csv = csv[["combo_id", "label", "n", "n_scored", "n_failed", "accuracy"]]
    csv.to_csv(png_path.with_suffix(".csv"), index=False)
    logger.info("Saved: %s", png_path.with_suffix(".csv"))
    logger.info("Plotted %d combos to %s", len(rates), out_dir)


if __name__ == "__main__":
    main()
