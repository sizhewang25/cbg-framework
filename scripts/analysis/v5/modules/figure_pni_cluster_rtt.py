"""RTT per `pni_gap` cluster, as boxes: does each cluster share a latency regime?

`plot-pni-gap` groups TGs by where they sit on the `d_pni` x gap scatter. This
figure reads that grouping **off disk** (`pni_gap.read_clusters`) and asks what
the RTTs look like inside each group. It does not recompute the clusters, so
the boxes cannot describe a different partition from the scatter.

## Two panels, one y axis

* **Every (TG, VP) pair**, each at its minimum RTT (`edges.load_min_rtt`, the
  RTT every LTD was built from). This is the whole latency profile a
  cluster's TGs present to the fleet.
* **Each TG's smallest RTT**: the RTT of its S-P VP, the delay no VP in the
  fleet avoids. On as01 this is the quantity that separates the mechanism
  groups (0.6-4.0 / 13.0-14.5 / 59.1-63.8 ms), so it gets its own panel
  instead of being lost at the bottom of the first.

The y axis is shared and linear from 0 ms: both panels are RTTs, and a
reader compares a cluster's floor in the right panel with its spread in the
left.

## Whiskers are p5 and p95

As in `figure_cost_box`: boxes are drawn from precomputed percentiles
(`Axes.bxp`), whiskers p5/p95, hinges p25/p75, no fliers. The min and max are
in the CSV. A Tukey whisker moves with the IQR, so its end is not a
percentile anyone can quote.

## What is checked before drawing

The clusters were computed from one canonical CSV; the RTTs here are read
from whatever CSV the run resolves to now. So:

* the manifest's `run_id` must be this run;
* the CSV's sha256 must equal the one the clusters were computed from;
* the TG sets must be identical, and each TG's smallest RTT here must equal
  the `sp_rtt_ms` the clusters CSV recorded.

Any mismatch raises and says to re-run `plot-pni-gap`.

## Replicas

~20 TGs per site share their VP geometry, so a cluster of 20 TGs at one site
is one site's RTTs repeated, not 20 samples. Each tick label carries the
cluster's site count next to its TG count for that reason.

Command: `plot-pni-cluster-rtt`. Writes beside the clusters, in
`<run>/pni-gap/<pni-stem>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import edges  # noqa: E402
from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules.cost import cost_stats  # noqa: E402
from scripts.analysis.v5.modules.figure_cost_box import _box as bxp_stats  # noqa: E402
from scripts.analysis.v5.modules.figure_pni_gap import cluster_label, cluster_style  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK, INK_2  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

PNG_NAME = "pni_cluster_rtt.png"
CSV_NAME = "pni_cluster_rtt.csv"
MANIFEST_NAME = "pni_cluster_rtt.manifest.json"

PAIRS = "pairs"
TG_MIN = "tg_min"
#: Panel order and titles.
SCOPES = (
    (PAIRS, "every (TG, VP) pair"),
    (TG_MIN, "each TG's smallest RTT (S-P VP)"),
)

#: Both sides are the same float off the same CSV; this absorbs rounding only.
RTT_MATCH_TOL_MS = 1e-6


def load(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """`(pairs, tgs, cluster_meta, out_dir)`: RTT edges labelled by cluster, checked.

    `pairs` is one row per `(tg_id, vp_id)` with `cluster`; `tgs` is the
    clusters CSV as written.
    """
    out_dir = P.output_dir(run.run_id, pni_csv, analysis_root=analysis_root)
    tgs, meta = P.read_clusters(out_dir, run_id=run.run_id)

    csv = edges.resolve_source_csv(run, source_csv)
    sha = P.sha256_file(Path(csv))
    if sha != meta.get("source_csv_sha256"):
        raise ValueError(
            f"{csv} is not the CSV the clusters were computed from "
            f"(sha256 {sha[:12]} vs {str(meta.get('source_csv_sha256'))[:12]}). "
            f"Re-run `plot-pni-gap --run-id {run.run_id}`."
        )
    rtt = edges.load_min_rtt(run, source_csv=csv)

    have, want = set(rtt.tg_id), set(tgs.tg_id)
    if have != want:
        raise ValueError(
            f"the edge CSV holds {len(have - want)} TGs the clusters do not, and "
            f"lacks {len(want - have)} they do. Re-run `plot-pni-gap --run-id {run.run_id}`."
        )
    floor = rtt.groupby("tg_id").rtt_ms.min()
    recorded = tgs.set_index("tg_id")[P.SP_RTT].reindex(floor.index)
    off = (floor - recorded).abs() > RTT_MATCH_TOL_MS
    if off.any():
        raise ValueError(
            f"{int(off.sum())} TGs' smallest RTT differs from the clusters CSV's "
            f"{P.SP_RTT}, e.g. {floor.index[off][0]!r}. Re-run `plot-pni-gap`."
        )
    pairs = rtt.merge(tgs[["tg_id", P.CLUSTER_COL]], on="tg_id", how="left", validate="m:1")
    return pairs, tgs, meta, out_dir


def stats_table(pairs: pd.DataFrame, tgs: pd.DataFrame, cluster_meta: dict) -> pd.DataFrame:
    """One row per `(scope, cluster)`: percentiles, extrema, counts."""
    n_sites = {c["cluster"]: c["n_sites"] for c in cluster_meta["clusters"]}
    floors = pairs.groupby(["tg_id", P.CLUSTER_COL], as_index=False).rtt_ms.min()
    rows = []
    for scope, frame in ((PAIRS, pairs), (TG_MIN, floors)):
        for c, block in frame.groupby(P.CLUSTER_COL):
            rows.append(
                {
                    "scope": scope,
                    P.CLUSTER_COL: int(c),
                    "n_tgs": int(block.tg_id.nunique()),
                    "n_sites": int(n_sites[int(c)]),
                    **{f"{k}_ms" if k != "n" else "n": v for k, v in cost_stats(block.rtt_ms.to_numpy()).items()},
                }
            )
    return pd.DataFrame(rows)


def plot(stats: pd.DataFrame, *, out_png: Path) -> Path:
    clusters = sorted(stats[P.CLUSTER_COL].unique())
    fig, axes = plt.subplots(1, len(SCOPES), figsize=(6.4, 2.9), sharey=True)
    for ax, (scope, title) in zip(axes, SCOPES):
        block = stats[stats.scope == scope].set_index(P.CLUSTER_COL).reindex(clusters)
        boxes = [
            bxp_stats({k: r[f"{k}_ms"] for k in ("p5", "p25", "p50", "p75", "p95")})
            for _, r in block.iterrows()
        ]
        art = ax.bxp(boxes, positions=range(1, len(clusters) + 1), widths=0.55, patch_artist=True,
                     showfliers=False, medianprops={"color": INK, "lw": 1.2},
                     whiskerprops={"color": INK_2, "lw": 0.8}, capprops={"color": INK_2, "lw": 0.8})
        for patch, c in zip(art["boxes"], clusters):
            hue, _ = cluster_style(c)
            patch.set(facecolor=hue, alpha=0.6, edgecolor=hue, linewidth=0.8)
        for i, (_, r) in enumerate(block.iterrows(), start=1):
            ax.text(i + 0.33, r["p50_ms"], f"{r['p50_ms']:.1f}", fontsize=6.5, va="center", color=INK_2)
        ax.set_xticks(
            range(1, len(clusters) + 1),
            [f"{cluster_label(c)}\n{int(r.n_sites)} sites, {int(r.n_tgs)} TGs\nn={int(r.n):,}"
             for c, (_, r) in zip(clusters, block.iterrows())],
            fontsize=6.5,
        )
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5, length=3)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("RTT (ms)", fontsize=8)
    axes[0].set_ylim(bottom=0)
    fig.tight_layout(pad=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _manifest(cluster_meta: dict, stats: pd.DataFrame) -> str:
    return json.dumps(
        {
            "figure": PNG_NAME,
            "csv": CSV_NAME,
            "run_id": cluster_meta["run_id"],
            "clusters_from": P.MANIFEST_NAME,
            "k": cluster_meta["clustering"]["k"],
            "source_csv_sha256": cluster_meta["source_csv_sha256"],
            "pni_csv_sha256": cluster_meta["pni_csv_sha256"],
            "scopes": {
                PAIRS: "every (TG, VP) pair at its minimum RTT",
                TG_MIN: "each TG's smallest RTT over the fleet, i.e. its S-P VP's RTT",
            },
            "boxes": "whiskers p5/p95, hinges p25/p75, line = median; no fliers. min/max in the CSV.",
            "replicas_note": (
                "~20 TGs share a site's VP geometry; n_sites, not n_tgs, is the "
                "number of independent observations per cluster."
            ),
            "n_rows": int(len(stats)),
        },
        indent=2,
    )


def build_for_run(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> list[Path]:
    """Stats CSV, manifest and the two-panel PNG. Returns the PNG in a list."""
    pairs, tgs, meta, out_dir = load(run, pni_csv, analysis_root=analysis_root, source_csv=source_csv)
    stats = stats_table(pairs, tgs, meta)
    stats.to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(_manifest(meta, stats))
    return [plot(stats, out_png=out_dir / PNG_NAME)]
