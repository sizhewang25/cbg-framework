"""Inspect a canonical-schema benchmark CSV (the GenericCSVSource input).

Given one canonical CSV (one row per (vp, target, RTT) flow — see
scripts/benchmark/v2/sources/README.md for the schema), this reports how the
flows are distributed over the two endpoint populations:

  - unique VP and target (TG) counts;
  - a two-panel CDF: left = per-VP occurrence count (number of flows that
    include each VP), right = per-TG observation count (flows per target);
  - a JSON stats file with the same distributions summarised as percentiles;
  - when the CSV has vp/target lat-lon columns, an interactive Leaflet flow
    map (self-contained HTML): click a VP to isolate its flows, click a TG
    to isolate the flows reaching it, click a flow line for pair details
    (obs count, distance, RTT, pair weight); double-click resets the focus.

CLI:
    python -m scripts.visualization.benchmark.inspect_source \\
        --csv path/to/canonical.csv [--out-dir path/to/outputs]

Outputs (default: alongside the CSV, named after its stem):
    <out-dir>/<stem>_occurrence_cdf.png
    <out-dir>/<stem>_stats.json
    <out-dir>/<stem>_flow_map.html
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

_MAP_COORD_COLS = ("vp_lat", "vp_lon", "target_lat", "target_lon")

# One polyline per unique (vp, target) pair; past this the HTML gets too
# heavy for a browser to be a useful inspection tool.
_MAX_MAP_PAIRS = 50_000

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


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


def _opt_float(value: Any, digits: int) -> float | None:
    return None if pd.isna(value) else round(float(value), digits)


def _build_flow_map(df: pd.DataFrame, csv_path: Path, out_path: Path) -> Path | None:
    """Write the interactive VP↔TG flow map; None when the CSV can't support one."""
    d = df.copy()
    for col in _MAP_COORD_COLS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=list(_MAP_COORD_COLS))
    if d.empty:
        print("no rows with complete coordinates — skipping the flow map")
        return None

    agg_spec: dict[str, tuple[str, str]] = {
        **{col: (col, "first") for col in _MAP_COORD_COLS},
        "n_obs": ("vp_id", "size"),
    }
    if "rtt_ms" in d.columns:
        d["rtt_ms"] = pd.to_numeric(d["rtt_ms"], errors="coerce")
        agg_spec["rtt_min"] = ("rtt_ms", "min")
        agg_spec["rtt_med"] = ("rtt_ms", "median")
    if "weight" in d.columns:
        d["weight"] = pd.to_numeric(d["weight"], errors="coerce")
        agg_spec["weight"] = ("weight", "sum")
    pairs = d.groupby(["vp_id", "target_id"], as_index=False).agg(**agg_spec)
    if len(pairs) > _MAX_MAP_PAIRS:
        print(f"{len(pairs)} unique (vp, target) pairs exceeds the "
              f"{_MAX_MAP_PAIRS} flow-map cap — skipping the flow map")
        return None

    lat1, lon1, lat2, lon2 = (
        np.radians(pairs[c].to_numpy()) for c in _MAP_COORD_COLS
    )
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    pairs["gc_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(h))

    vps = pairs.groupby("vp_id").agg(
        lat=("vp_lat", "first"), lon=("vp_lon", "first"),
        n_targets=("target_id", "nunique"), n_obs=("n_obs", "sum"),
    ).reset_index()
    tgs = pairs.groupby("target_id").agg(
        lat=("target_lat", "first"), lon=("target_lon", "first"),
        n_vps=("vp_id", "nunique"), n_obs=("n_obs", "sum"),
    ).reset_index()

    has_rtt = "rtt_min" in pairs.columns
    has_weight = "weight" in pairs.columns
    flows: list[dict[str, Any]] = []
    for row in pairs.itertuples(index=False):
        flow: dict[str, Any] = {
            "vp": row.vp_id,
            "tg": row.target_id,
            "coords": [[round(float(row.vp_lat), 5), round(float(row.vp_lon), 5)],
                       [round(float(row.target_lat), 5), round(float(row.target_lon), 5)]],
            "n_obs": int(row.n_obs),
            "gc_km": round(float(row.gc_km), 1),
        }
        if has_rtt:
            flow["rtt_min"] = _opt_float(row.rtt_min, 3)
            flow["rtt_med"] = _opt_float(row.rtt_med, 3)
        if has_weight:
            flow["weight"] = _opt_float(row.weight, 6)
        flows.append(flow)

    payload = {
        "vps": [{"id": r.vp_id, "lat": round(float(r.lat), 5),
                 "lon": round(float(r.lon), 5), "n_targets": int(r.n_targets),
                 "n_obs": int(r.n_obs)} for r in vps.itertuples(index=False)],
        "tgs": [{"id": r.target_id, "lat": round(float(r.lat), 5),
                 "lon": round(float(r.lon), 5), "n_vps": int(r.n_vps),
                 "n_obs": int(r.n_obs)} for r in tgs.itertuples(index=False)],
        "flows": flows,
    }

    html = (_TEMPLATE_DIR / "inspect_source_map.html").read_text()
    js = (_TEMPLATE_DIR / "inspect_source_map.js").read_text()
    out_path.write_text(
        html.replace("__TITLE__", csv_path.name)
            .replace("__SCRIPT__", js)
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    )
    return out_path


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
        "has_weight": "weight" in df.columns,
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

    if all(c in df.columns for c in _MAP_COORD_COLS):
        map_path = _build_flow_map(
            df, csv_path, out_dir / f"{csv_path.stem}_flow_map.html"
        )
        if map_path is not None:
            stats["flow_map"] = str(map_path)
    else:
        print("no vp/target lat-lon columns — skipping the flow map")
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
