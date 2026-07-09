"""Inspect a canonical-schema benchmark CSV (the GenericCSVSource input).

Given one canonical CSV (one row per (vp, target, RTT) flow — see
scripts/benchmark/v2/sources/README.md for the schema), this reports how the
flows are distributed over the two endpoint populations:

  - unique VP and target (TG) counts;
  - a two-panel CDF: left = per-VP occurrence count (number of flows that
    include each VP), right = per-TG observation count (flows per target);
  - a JSON stats file with the same distributions summarised as percentiles.

CLI:
    python -m scripts.visualization.benchmark.inspect_source \\
        --csv path/to/canonical.csv [--out-dir path/to/outputs]

Outputs (default: alongside the CSV, named after its stem):
    <out-dir>/<stem>_occurrence_cdf.png
    <out-dir>/<stem>_stats.json
"""

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

_REQUIRED = ("vp_id", "target_id")

_PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _count_stats(counts: pd.Series) -> dict[str, Any]:
    """Percentile summary of a per-endpoint flow-count distribution."""
    q = np.percentile(counts, _PCTS)
    return {
        "n": int(len(counts)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": round(float(counts.mean()), 3),
        "percentiles": {f"p{p}": float(v) for p, v in zip(_PCTS, q)},
    }


def _plot_cdf(ax: plt.Axes, counts: pd.Series, title: str, xlabel: str) -> None:
    x = np.sort(counts.to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post")
    # Log-x for heavy-tailed counts; linear when the whole range sits
    # inside one decade (e.g. a complete mesh), where log ticks are noise.
    if x[-1] / max(x[0], 1) >= 10:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    med = np.median(x)
    ax.axvline(med, color="crimson", linestyle="--", linewidth=0.8)
    ax.annotate(f"median = {med:g}", xy=(med, 0.5), xytext=(5, 0),
                textcoords="offset points", color="crimson", fontsize=8)


def inspect(csv_path: Path, out_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is not a canonical CSV — missing columns: {missing}"
        )
    df["vp_id"] = df["vp_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)

    vp_counts = df.groupby("vp_id").size()
    tg_counts = df.groupby("target_id").size()

    stats = {
        "csv": str(csv_path),
        "n_flows": int(len(df)),
        "n_unique_vps": int(vp_counts.size),
        "n_unique_targets": int(tg_counts.size),
        "vp_occurrences": _count_stats(vp_counts),
        "target_observations": _count_stats(tg_counts),
        "has_pair_weight": "pair_weight" in df.columns,
    }

    fig, (ax_vp, ax_tg) = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_cdf(ax_vp, vp_counts,
              f"VPs (n={vp_counts.size})", "flows per VP")
    _plot_cdf(ax_tg, tg_counts,
              f"Targets (n={tg_counts.size})", "flows per target")
    fig.suptitle(f"{csv_path.name} — {len(df)} flows", fontsize=11)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{csv_path.stem}_occurrence_cdf.png"
    json_path = out_dir / f"{csv_path.stem}_stats.json"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    json_path.write_text(json.dumps(stats, indent=2) + "\n")

    stats["figure"] = str(fig_path)
    stats["stats_json"] = str(json_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="canonical-schema CSV to inspect")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: the CSV's directory)")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
    stats = inspect(args.csv, out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
