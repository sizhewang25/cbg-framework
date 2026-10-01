"""Each TG's smallest RTT, per `pni_gap` cluster, as boxes: one latency regime per cluster?

`plot-pni-gap` groups TGs by where they sit on the `d_pni` x gap scatter. This
figure reads that grouping **off disk** (`pni_gap.read_clusters`) and asks what
the RTTs look like inside each group. It does not recompute the clusters, so
the boxes cannot describe a different partition from the scatter.

## One quantity: each TG's smallest RTT

The box per cluster is over its TGs' **smallest RTT across the fleet**: the
RTT of the S-P VP, read off `edges.load_min_rtt` (each pair at its minimum,
the RTT every LTD was built from). It is the delay no VP avoids, and on as01
it is the quantity that separates the mechanism groups (0.6-4.0 / 13.0-14.5 /
59.1-63.8 ms). The distribution over every (TG, VP) pair was drawn beside it
once; it mostly restated the fleet's geography and was dropped. The y axis is
linear from 0 ms.

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
from matplotlib.ticker import MultipleLocator  # noqa: E402
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

#: The y-axis label: what each box is over. Short, because the figure is small;
#: the caption spells it out ("each TG's smallest RTT over the fleet").
QUANTITY = "min RTT (ms)"

#: A small figure. Width is fixed rather than grown with k; at k <= 6 a
#: 0.5 in slot per box still fits a two-line tick label.
FIGSIZE = (3.0, 2.0)

#: y tick spacing, in ms.
Y_STEP_MS = 10.0

#: How far above the x axis, in points, a median label's centre must sit.
MEDIAN_LABEL_CLEARANCE_PT = 4.0

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
    """One row per cluster: percentiles and extrema of its TGs' smallest RTT, and counts."""
    n_sites = {c["cluster"]: c["n_sites"] for c in cluster_meta["clusters"]}
    floors = pairs.groupby(["run_id", "tg_id", P.CLUSTER_COL], as_index=False).rtt_ms.min()
    rows = []
    for c, block in floors.groupby(P.CLUSTER_COL):
        stats = cost_stats(block.rtt_ms.to_numpy())
        rows.append(
            {
                P.CLUSTER_COL: int(c),
                "n_tgs": int(stats.pop("n")),
                "n_sites": int(n_sites[int(c)]),
                **{f"{k}_ms": v for k, v in stats.items()},
            }
        )
    return pd.DataFrame(rows)


def plot(stats: pd.DataFrame, *, out_png: Path) -> Path:
    stats = stats.sort_values(P.CLUSTER_COL).reset_index(drop=True)
    clusters = stats[P.CLUSTER_COL].tolist()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    boxes = [bxp_stats({k: r[f"{k}_ms"] for k in ("p5", "p25", "p50", "p75", "p95")})
             for _, r in stats.iterrows()]
    art = ax.bxp(boxes, positions=range(1, len(clusters) + 1), widths=0.5, patch_artist=True,
                 showfliers=False, medianprops={"color": INK, "lw": 0.9},
                 whiskerprops={"color": INK_2, "lw": 0.6}, capprops={"color": INK_2, "lw": 0.6})
    for patch, c in zip(art["boxes"], clusters):
        hue, _ = cluster_style(c)
        patch.set(facecolor=hue, alpha=0.6, edgecolor=hue, linewidth=0.6)
    # Sites, not TGs, on the tick: replicas are one observation repeated. TG
    # counts are in the CSV.
    ax.set_xticks(
        range(1, len(clusters) + 1),
        [f"{cluster_label(r[P.CLUSTER_COL])}\n{int(r.n_sites)} sites" for _, r in stats.iterrows()],
        fontsize=5.5,
    )
    ax.set_ylabel(QUANTITY, fontsize=6)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(Y_STEP_MS))
    ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1.5)
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    # Median labels, after layout so the axes' height is known. A label centred
    # on a ~1 ms median sits on the axis line, so its centre is kept at least
    # `MEDIAN_LABEL_CLEARANCE_PT` above it, whatever the figure height.
    height_pt = ax.bbox.height * 72.0 / fig.dpi
    floor = ax.get_ylim()[1] * MEDIAN_LABEL_CLEARANCE_PT / height_pt
    for i, (_, r) in enumerate(stats.iterrows(), start=1):
        ax.text(i + 0.3, max(r["p50_ms"], floor), f"{r['p50_ms']:.1f}", fontsize=5.5,
                va="center", color=INK_2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
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
            "quantity": "each TG's smallest RTT over the fleet, i.e. its S-P VP's RTT",
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
