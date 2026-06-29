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


def plot_bars(
    rates: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    baseline_acc: float = float("nan"),
    n_base: int = 0,
) -> plt.Figure:
    df = rates.sort_values("accuracy_pct", ascending=True)  # ascending → best on top
    y = list(range(len(df)))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(df) + 1.5)))
    ax.barh(y, df["accuracy_pct"], color="#4E79A7", alpha=0.85, zorder=2)

    if n_base and pd.notna(baseline_acc):
        ax.axvline(baseline_acc, color="#666666", linestyle=":", linewidth=1.8, zorder=1,
                   label=f"shortest-ping VP ({baseline_acc:.1f}%)")

    for yi, val in enumerate(df["accuracy_pct"]):
        if pd.notna(val):
            ax.text(val + 0.3, yi, f"{val:.1f}%", va="center", fontsize=8)

    xs = [float(df["accuracy_pct"].max()) if len(df) else 0.0]
    if pd.notna(baseline_acc):
        xs.append(baseline_acc)
    ax.set_xlim(0, min(100.0, max(10.0, max(xs) * 1.25)))
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=8)
    ax.set_xlabel("Classification Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

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
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)

    cfg = _load_config(args.config)
    run_dir, scored_dir_default, clusters_dir_default = _resolve_paths(cfg, args.outputs_root)
    scored_dir = Path(args.scored_dir) if args.scored_dir else scored_dir_default
    clusters_dir = Path(args.clusters_dir) if args.clusters_dir else clusters_dir_default

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
        n = len(df)
        rows.append({
            "combo_id": combo_id,
            "label": combo_map[combo_id],
            "n": n,
            "n_scored": int(df["success"].sum()) if n else 0,
            "accuracy_pct": float(df["match"].mean()) * 100 if n else float("nan"),
        })
    rates = pd.DataFrame(rows)

    if rates.empty:
        logger.warning("no matching scored CSVs found in %s for the configured combos", scored_dir)
        return

    base_acc = float("nan")
    n_base = 0
    bpath = scored_dir / "baseline.csv"
    if bpath.exists():
        bdf = pd.read_csv(bpath)
        n_base = len(bdf)
        if n_base:
            base_acc = float(bdf["vp_matches_centroid"].mean()) * 100
        logger.info("baseline: same-centroid=%.1f%% (n=%d)", base_acc, n_base)
    else:
        logger.warning("no baseline.csv in %s; skipping shortest-ping baseline", scored_dir)

    out_dir = route_geo_path(args.out_dir) if args.out_dir else analysis_out_dir(run_dir, "cluster")
    png_path = out_dir / f"{run_dir.name}_classification_accuracy.png"

    fig = plot_bars(
        rates, png_path,
        title=f"Classification accuracy — {run_dir.name} ({n_centroids} centroids, R={radius_km:.0f} km)",
        baseline_acc=base_acc,
        n_base=n_base,
    )
    plt.close(fig)

    csv = rates.sort_values("accuracy_pct", ascending=False).copy()
    csv["n_failed"] = csv["n"] - csv["n_scored"]
    csv["accuracy"] = csv["accuracy_pct"] / 100
    csv = csv[["combo_id", "label", "n", "n_scored", "n_failed", "accuracy"]]
    baseline_row = pd.DataFrame([{
        "combo_id": "shortest_ping_baseline",
        "label": "Shortest ping VP",
        "n": n_base, "n_scored": n_base, "n_failed": 0,
        "accuracy": base_acc / 100 if pd.notna(base_acc) else float("nan"),
    }])
    pd.concat([csv, baseline_row], ignore_index=True).to_csv(
        png_path.with_suffix(".csv"), index=False
    )
    logger.info("Saved: %s", png_path.with_suffix(".csv"))
    logger.info("Plotted %d combos to %s", len(rates), out_dir)


if __name__ == "__main__":
    main()
