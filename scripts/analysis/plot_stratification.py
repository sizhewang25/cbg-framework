"""Stratification diagnostic plot from materialized benchmark fold inputs.

Reads ``eval_observations.parquet`` from each ``fold_N`` sub-directory of a
setup tree to reconstruct the target-to-fold assignment, then joins
``tg_configs.parquet`` (the full target catalog, identical across folds) for
geographic metadata.  Works for any DataSource — no dependency on
``stratification.json`` or ``anchor_fold_*.json``.

Four panels:
  [1] World map — targets colored by fold
  [2] Top-10 countries × fold bar chart
  [3] Top-N ASN buckets × fold bar chart
  [4] Intra-fold pairwise great-circle distance histogram

CLI:
    python -m scripts.analysis.plot_stratification \\
        --inputs-dir scripts/benchmark/v2/inputs/ripe_atlas_asn_corpora/europe_as3209_final_de/probes_to_anchors \\
        --out scripts/analysis/outputs/europe_as3209_final_de/cluster/europe_as3209_final_de_stratification.png
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_FOLD_DIR_RE = re.compile(r"^fold_(\d+)$")
_MAX_SAMPLE = 80  # per-fold sample for pairwise distance (O(n²) guard)
_TOP_COUNTRIES = 10
_TOP_ASNS = 15


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6367.0 * 2 * math.asin(math.sqrt(max(h, 0.0)))


def _intra_fold_dists(fold_df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    sample = fold_df.sample(n=min(_MAX_SAMPLE, len(fold_df)), random_state=seed)
    pts = sample[["lat", "lon"]].to_numpy()
    n = len(pts)
    return np.array([
        _haversine_km(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
        for i in range(n)
        for j in range(i + 1, n)
    ])


def load_partition(inputs_dir: Path) -> pd.DataFrame:
    """Reconstruct fold assignments from materialized parquet files.

    Returns a DataFrame with columns:
      target_id, fold (int), lat, lon, country, asn
    """
    fold_dirs = sorted(
        (d for d in inputs_dir.iterdir() if d.is_dir() and _FOLD_DIR_RE.match(d.name)),
        key=lambda d: int(_FOLD_DIR_RE.match(d.name).group(1)),  # type: ignore[union-attr]
    )
    if not fold_dirs:
        raise FileNotFoundError(f"no fold_* directories found in {inputs_dir}")

    fold_by_target: dict[str, int] = {}
    for fold_dir in fold_dirs:
        fold_idx = int(_FOLD_DIR_RE.match(fold_dir.name).group(1))  # type: ignore[union-attr]
        ev = pd.read_parquet(fold_dir / "eval_observations.parquet", columns=["target_id"])
        for tid in ev["target_id"].unique():
            fold_by_target[str(tid)] = fold_idx

    tg = pd.read_parquet(fold_dirs[0] / "tg_configs.parquet")
    tg["tg_id"] = tg["tg_id"].astype(str)

    assigned = tg["tg_id"].map(fold_by_target)
    tg = tg[assigned.notna()].copy()
    tg["fold"] = tg["tg_id"].map(fold_by_target).astype(int)

    for c in ("country", "asn"):
        if c not in tg.columns:
            tg[c] = None

    return (
        tg.rename(columns={"tg_id": "target_id"})
        [["target_id", "fold", "lat", "lon", "country", "asn"]]
        .reset_index(drop=True)
    )


def plot_stratification(inputs_dir: Path, out: Path) -> Path:
    """Render four-panel stratification diagnostic and write *out*."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    df = load_partition(inputs_dir)
    K = int(df["fold"].max()) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    fold_sizes = df.groupby("fold").size().to_dict()

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.5, 1], hspace=0.38, wspace=0.38)

    # ---- [1] world map ----------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor="#f4f1ea")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#e8eef5")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555")
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888")
    ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    for f in range(K):
        sub = df[df["fold"] == f]
        ax_map.scatter(
            sub["lon"], sub["lat"],
            s=14, c=[colors[f]], alpha=0.85, edgecolors="none",
            label=f"fold {f}  (n={fold_sizes.get(f, 0)})",
            transform=ccrs.PlateCarree(), zorder=3,
        )
    ax_map.set_title("Targets colored by fold assignment", fontsize=12)
    ax_map.legend(loc="lower left", fontsize=9, framealpha=0.85, ncol=K)

    # ---- [2] country × fold bars ------------------------------------------------
    ax_country = fig.add_subplot(gs[1, 0])
    valid_countries = df["country"].dropna()
    valid_countries = valid_countries[valid_countries.astype(str).str.strip() != ""]
    top_countries = valid_countries.value_counts().head(_TOP_COUNTRIES).index.tolist()
    if top_countries:
        ctab = (
            df[df["country"].isin(top_countries)]
            .pivot_table(index="country", columns="fold", aggfunc="size", fill_value=0)
            .reindex(top_countries)
        )
        for f in range(K):
            if f not in ctab.columns:
                ctab[f] = 0
        ctab[list(range(K))].plot(
            kind="barh", stacked=False, ax=ax_country,
            color=[colors[f] for f in range(K)], legend=False,
        )
    ax_country.set_xlabel("target count", fontsize=9)
    ax_country.set_ylabel("country", fontsize=9)
    ax_country.set_title(f"Top {_TOP_COUNTRIES} countries × fold", fontsize=10)
    ax_country.invert_yaxis()
    ax_country.tick_params(labelsize=8)

    # ---- [3] ASN-bucket × fold bars --------------------------------------------
    ax_asn = fig.add_subplot(gs[1, 1])
    asn_counts = df["asn"].dropna()
    top_asn_vals = asn_counts.value_counts().head(_TOP_ASNS).index.tolist()
    top_asn_set = set(top_asn_vals)

    def _bucket(asn: object) -> str:
        if asn is None or (isinstance(asn, float) and math.isnan(asn)):
            return "asn_none"
        return f"AS{int(asn)}" if asn in top_asn_set else "other_AS"

    df_asn = df.copy()
    df_asn["asn_bucket"] = df_asn["asn"].map(_bucket)
    atab = df_asn.pivot_table(index="asn_bucket", columns="fold", aggfunc="size", fill_value=0)
    atab["_total"] = atab[list(range(K))].sum(axis=1)
    atab = atab.sort_values("_total", ascending=False).drop(columns="_total").head(_TOP_ASNS + 2)
    for f in range(K):
        if f not in atab.columns:
            atab[f] = 0
    atab[list(range(K))].plot(
        kind="barh", stacked=False, ax=ax_asn,
        color=[colors[f] for f in range(K)], legend=False,
    )
    ax_asn.set_xlabel("target count", fontsize=9)
    ax_asn.set_ylabel("ASN bucket", fontsize=9)
    ax_asn.set_title(f"Top-{_TOP_ASNS} ASNs + other × fold", fontsize=10)
    ax_asn.invert_yaxis()
    ax_asn.tick_params(labelsize=7)

    # ---- [4] intra-fold pairwise distance histogram ----------------------------
    ax_dist = fig.add_subplot(gs[1, 2])
    for f in range(K):
        sub_df = df[df["fold"] == f]
        dists = _intra_fold_dists(sub_df)
        if len(dists) == 0:
            continue
        ax_dist.hist(
            dists, bins=30, alpha=0.35, color=colors[f],
            label=f"fold {f}  (med={np.median(dists):.0f} km)",
        )
    ax_dist.set_xlabel("pairwise distance (km)", fontsize=9)
    ax_dist.set_ylabel("count", fontsize=9)
    ax_dist.set_title(f"Intra-fold spread (sample ≤{_MAX_SAMPLE})", fontsize=10)
    ax_dist.legend(fontsize=8, loc="upper right")
    ax_dist.tick_params(labelsize=8)

    run_id = inputs_dir.parent.name
    sizes_str = "  |  ".join(f"fold {f}: {fold_sizes.get(f, 0)}" for f in range(K))
    fig.suptitle(f"{run_id}  —  stratification  ({sizes_str})", fontsize=11, y=1.01)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs-dir", type=Path, required=True,
        help=(
            "Parent of fold_N sub-directories "
            "(e.g. scripts/benchmark/v2/inputs/<source>/<run_id>/<setup>)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()
    if not args.inputs_dir.is_dir():
        raise SystemExit(f"inputs-dir not found: {args.inputs_dir}")
    out = plot_stratification(args.inputs_dir, args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
