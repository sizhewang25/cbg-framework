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
* the PNI list's sha256 must be unchanged (the directory is keyed on its file
  stem, so an edited list would otherwise read the old list's clusters);
* the edge CSV's sha256 must equal the one the clusters were computed from;
* the TG sets must be identical, and each TG's smallest RTT here must equal
  the `sp_rtt_ms` the clusters CSV recorded.

Any mismatch raises and says to re-run `plot-pni-gap`.

## Replicas

~20 TGs per site share their VP geometry, so a cluster of 20 TGs at one site
is one site's RTTs repeated, not 20 samples. Each tick label carries the
cluster's site count next to its TG count for that reason.

Pooled (`--layout pooled`), the clusters come from `_cross/pni-gap/...` and
every check above runs per run, each against its own PNI list and edge CSV.

Command: `plot-pni-cluster-rtt`. Writes beside the clusters it reads.
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


def _checked_run(run: RunPaths, pni_csv: Path, record: dict, source_csv: Path | None) -> pd.DataFrame:
    """One run's min-RTT edges, after checking its inputs are the clustered ones."""
    rerun = "Re-run `plot-pni-gap` over the same --run-id set."
    if P.sha256_file(Path(pni_csv)) != record.get("pni_csv_sha256"):
        raise ValueError(
            f"{run.run_id}: {pni_csv} has changed since the clusters were computed "
            f"(the output directory is not keyed on its content). {rerun}"
        )
    csv = edges.resolve_source_csv(run, source_csv)
    sha = P.sha256_file(Path(csv))
    if sha != record.get("source_csv_sha256"):
        raise ValueError(
            f"{run.run_id}: {csv} is not the CSV the clusters were computed from "
            f"(sha256 {sha[:12]} vs {str(record.get('source_csv_sha256'))[:12]}). {rerun}"
        )
    rtt = edges.load_min_rtt(run, source_csv=csv)
    rtt.insert(0, "run_id", run.run_id)
    return rtt


def load_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layout: str = P.PER_RUN,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """`(pairs, tgs, cluster_meta, out_dir)`: RTT edges labelled by cluster, checked.

    `pairs` is one row per `(run_id, tg_id, vp_id)` with `cluster`; `tgs` is
    the clusters CSV as written. Every run is checked on its own: PNI list,
    edge CSV, TG set, and each TG's smallest RTT.
    """
    run_ids = [r.run_id for r in runs]
    if layout == P.POOLED:
        out_dir = P.pooled_output_dir(run_ids, analysis_root=analysis_root)
    elif len(runs) == 1:
        out_dir = P.output_dir(runs[0].run_id, pni_csvs[runs[0].run_id], analysis_root=analysis_root)
    else:
        raise ValueError(f"layout {layout!r} takes one run; got {len(runs)}")
    tgs, meta = P.read_clusters(out_dir, run_ids=run_ids)
    records = {r["run_id"]: r for r in meta.get("runs", [])}

    rtt = pd.concat(
        [_checked_run(run, pni_csvs[run.run_id], records.get(run.run_id, {}),
                      (source_csvs or {}).get(run.run_id)) for run in runs],
        ignore_index=True,
    )
    key = ["run_id", "tg_id"]
    have = set(map(tuple, rtt[key].drop_duplicates().to_numpy()))
    want = set(map(tuple, tgs[key].to_numpy()))
    if have != want:
        raise ValueError(
            f"the edge CSVs hold {len(have - want)} TGs the clusters do not, and "
            f"lack {len(want - have)} they do. Re-run `plot-pni-gap`."
        )
    floor = rtt.groupby(key).rtt_ms.min()
    recorded = tgs.set_index(key)[P.SP_RTT].reindex(floor.index)
    off = (floor - recorded).abs() > RTT_MATCH_TOL_MS
    if off.any():
        raise ValueError(
            f"{int(off.sum())} TGs' smallest RTT differs from the clusters CSV's "
            f"{P.SP_RTT}, e.g. {floor.index[off][0]!r}. Re-run `plot-pni-gap`."
        )
    pairs = rtt.merge(tgs[key + [P.CLUSTER_COL]], on=key, how="left", validate="m:1")
    return pairs, tgs, meta, out_dir


def load(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """`load_runs` for one run, per-run layout."""
    return load_runs(
        [run], {run.run_id: pni_csv}, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
    )


def stats_table(pairs: pd.DataFrame, tgs: pd.DataFrame, cluster_meta: dict) -> pd.DataFrame:
    """One row per `(scope, cluster)`: percentiles, extrema, counts."""
    n_sites = {c["cluster"]: c["n_sites"] for c in cluster_meta["clusters"]}
    key = ["run_id", "tg_id"]
    floors = pairs.groupby(key + [P.CLUSTER_COL], as_index=False).rtt_ms.min()
    rows = []
    for scope, frame in ((PAIRS, pairs), (TG_MIN, floors)):
        for c, block in frame.groupby(P.CLUSTER_COL):
            rows.append(
                {
                    "scope": scope,
                    P.CLUSTER_COL: int(c),
                    "n_tgs": int(block.groupby(key).ngroups),
                    "n_sites": int(n_sites[int(c)]),
                    **{f"{k}_ms" if k != "n" else "n": v for k, v in cost_stats(block.rtt_ms.to_numpy()).items()},
                }
            )
    return pd.DataFrame(rows)


def plot(stats: pd.DataFrame, *, out_png: Path) -> Path:
    clusters = sorted(stats[P.CLUSTER_COL].unique())
    # Width grows with k so each tick's three-line label keeps its own slot.
    width = max(6.4, 1.1 * len(clusters) * len(SCOPES) + 0.8)
    fig, axes = plt.subplots(1, len(SCOPES), figsize=(width, 2.9), sharey=True)
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
            [f"{cluster_label(c)}\n{int(r.n_sites)} sites\n{int(r.n_tgs)} TGs\nn={int(r.n):,}"
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
            "layout": cluster_meta.get("layout"),
            "run_ids": cluster_meta["run_ids"],
            "clusters_from": P.MANIFEST_NAME,
            "k": cluster_meta["clustering"]["k"],
            "inputs": [
                {key: r[key] for key in ("run_id", "source_csv_sha256", "pni_csv_sha256")}
                for r in cluster_meta["runs"]
            ],
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


def _write(pairs: pd.DataFrame, tgs: pd.DataFrame, meta: dict, out_dir: Path) -> Path:
    stats = stats_table(pairs, tgs, meta)
    stats.to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(_manifest(meta, stats))
    return plot(stats, out_png=out_dir / PNG_NAME)


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> list[Path]:
    """Boxes for every requested layout, each beside the clusters it reads."""
    bad = [lay for lay in layouts if lay not in P.LAYOUTS]
    if bad:
        raise ValueError(f"unknown layout {bad}; expected {list(P.LAYOUTS)}")
    pngs = []
    if P.PER_RUN in layouts:
        for run in runs:
            one = {run.run_id: source_csvs[run.run_id]} if source_csvs and run.run_id in source_csvs else None
            pairs, tgs, meta, out_dir = load_runs([run], pni_csvs, analysis_root=analysis_root, source_csvs=one)
            pngs.append(_write(pairs, tgs, meta, out_dir))
    if P.POOLED in layouts:
        pairs, tgs, meta, out_dir = load_runs(runs, pni_csvs, layout=P.POOLED,
                                              analysis_root=analysis_root, source_csvs=source_csvs)
        pngs.append(_write(pairs, tgs, meta, out_dir))
    return pngs


def build_for_run(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> list[Path]:
    """Stats CSV, manifest and the two-panel PNG for one run, per-run layout."""
    return build_for_runs(
        [run], {run.run_id: pni_csv}, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
    )
