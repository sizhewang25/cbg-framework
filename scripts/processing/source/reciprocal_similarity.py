"""How similar are the public and private bipartite graphs, before and after matching?

Companion report to `reciprocal_grid_filter`. It answers "are these two datasets actually
comparable now?" by measuring the same quantities at all **three stages** of the pipeline
and printing the contrast:

  1. **raw**        -- the two inputs as they arrive
  2. **reciprocal** -- after the presence filter (same cells)
  3. **balanced**   -- after per-cell count matching (same cells AND same counts)

Reporting one stage alone would be misleading in both directions: a high similarity at the
end says nothing without the baseline it started from, and the two standard metrics each
saturate once *their* stage has done its work.

Three tiers, because they answer different questions
----------------------------------------------------
**Tier 1 -- exact by construction.** Node counts and per-cell occupancy. After balancing
these must agree exactly; a difference is a bug, not a finding, so they are asserted in the
report rather than interpreted.

**Tier 2 -- the cell-level bipartite graph.** Each dataset collapses to a weighted
`VP-cell x TG-cell` biadjacency matrix `A`. Both are built on the **union** of the two
datasets' occupied cells, never the intersection: before the presence filter the footprints
genuinely differ, and a shared index with zeros is what makes that mismatch visible instead
of hiding it in the choice of index. Node correspondence is therefore known, so the standard
aligned-graph measures apply directly:

  * `jaccard_support` / `hamming_norm` -- **footprint overlap**, i.e. graph edit distance
    under known node correspondence. These do all their work in the reciprocal stage and
    then saturate.
  * `frobenius_rel` = `||A_pub - A_priv||_F / ||A_priv||_F` -- the weighted generalization,
    measuring **size and weight agreement**. Nearly blind to the presence filter; collapses
    under balancing. The only one that still discriminates when an edge set is genuinely
    pruned, as in the traffic-weighted arm.
  * `cosine` -- the one number that moves monotonically across the whole pipeline, so it is
    the best single summary.

> **A caveat that has to be stated with the numbers.** For two *complete* meshes with
> per-cell counts equalized, `A == outer(n_vp_per_cell, n_tg_per_cell)` exactly. Balancing
> equalizes both count vectors by construction, so at that point cell-level similarity is
> **forced rather than measured** -- it confirms the construction worked, and nothing more.
> The one real degree of freedom left at cell level is deviation from completeness, which is
> `connectance` per dataset. Read that, not the similarity, as the substantive number for a
> mesh-vs-mesh pair.

**Tier 3 -- residual within-cell geometry**, the part balancing cannot fix and therefore the
only place a real difference can still hide. Each distribution is reported as p50/p90 on
both sides plus a divergence: Wasserstein `w1` in native units (so it reads as "9.7 km
apart") with the KS statistic beside it. The statistic, never a p-value -- on a few dozen
deliberately selected nodes a p-value would be theatre.

CLI::

    .venv/bin/python -m scripts.processing.source.reciprocal_similarity \
        --public  datasets/ripe_as7018/as7018-us-test01.csv \
        --private datasets/final/as02-20260728-20260802.mainland.sanitized.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v3.modules.answer_space import elementwise_km
from scripts.analysis.v3.modules.bipartite import angular_features, bearings_deg
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    GRID_NAMES,
    Grid,
    get_grid,
)
from scripts.processing.source.reciprocal_grid_filter import (
    _DATASETS,
    _SIDES,
    nearest_counterpart_km,
    read_canonical_csv,
    reciprocal_filter,
)

logger = logging.getLogger(__name__)

_SUFFIX = ".similarity.json"

#: Coarsening rungs for the multi-scale concentration check. Balancing pins the working
#: resolution only, so agreement above and below it is a genuine finding either way.
_LADDER = (5, 4, 3, 2)


def _default_output(public_path: Path) -> Path:
    return public_path.with_name(public_path.stem + _SUFFIX)


def cell_biadjacency(
    frames: dict[str, pd.DataFrame], *, grid: Grid, resolution: int
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Weighted `VP-cell x TG-cell` matrices for both datasets, on the **union** index.

    Union, not intersection: at the raw stage the two footprints differ, and that mismatch
    is the measurement. Intersecting would silently discard it and report a flattering
    number.
    """
    counts, cells = {}, {"vp": set(), "target": set()}
    for name, df in frames.items():
        keys = {}
        for side in _SIDES:
            _, lat_col, lon_col = (
                f"{side}_id", f"{side}_lat", f"{side}_lon",
            )
            part = grid.partition(
                df[lat_col].to_numpy(dtype=float),
                df[lon_col].to_numpy(dtype=float),
                resolution,
            )
            keys[side] = [str(c) for c in part.cell_id]
            cells[side] |= set(keys[side])
        counts[name] = (
            pd.DataFrame({"vc": keys["vp"], "tc": keys["target"]})
            .groupby(["vc", "tc"])
            .size()
        )

    vc, tc = sorted(cells["vp"]), sorted(cells["target"])
    idx = pd.MultiIndex.from_product([vc, tc])
    mats = {
        name: counts[name].reindex(idx).fillna(0).to_numpy().reshape(len(vc), len(tc))
        for name in frames
    }
    return mats, vc, tc


def graph_similarity(p: np.ndarray, q: np.ndarray) -> dict:
    """Aligned-graph similarity of two biadjacency matrices of identical shape."""
    sp, sq = p > 0, q > 0
    union = np.logical_or(sp, sq).sum()
    nq = np.linalg.norm(q)
    return {
        "measured_cell_pairs_public": int(sp.sum()),
        "measured_cell_pairs_private": int(sq.sum()),
        "jaccard_support": (
            round(float(np.logical_and(sp, sq).sum() / union), 4) if union else None
        ),
        "hamming_norm": round(float(np.logical_xor(sp, sq).sum() / sp.size), 4),
        "cosine": (
            round(float((p * q).sum() / (np.linalg.norm(p) * nq)), 4)
            if nq and np.linalg.norm(p)
            else None
        ),
        "frobenius_rel": round(float(np.linalg.norm(p - q) / nq), 4) if nq else None,
    }


def divergence(a: np.ndarray, b: np.ndarray, *, unit: str) -> dict:
    """p50/p90 of both distributions plus Wasserstein and KS between them.

    `w1` is in `unit`, which is the point: "9.7 km apart" is a sentence a reader can check
    against the cell size, where a normalized score is not. No p-value -- see the module
    docstring.
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {
        "unit": unit,
        "public_p50": round(float(np.percentile(a, 50)), 3) if a.size else None,
        "public_p90": round(float(np.percentile(a, 90)), 3) if a.size else None,
        "private_p50": round(float(np.percentile(b, 50)), 3) if b.size else None,
        "private_p90": round(float(np.percentile(b, 90)), 3) if b.size else None,
    }
    if a.size and b.size:
        out["w1"] = round(float(wasserstein_distance(a, b)), 4)
        out["ks"] = round(float(ks_2samp(a, b).statistic), 4)
    else:
        out["w1"] = out["ks"] = None
    return out


def angular_per_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(max_gap_deg, circular_variance)` per target, over the VPs that measured it.

    §7.3's arrangement term: identical VP counts at identical distances can still constrain
    a target very differently depending on whether they surround it. Balancing does not
    control this, which is exactly why it is measured.
    """
    gaps, cvar = [], []
    for _, grp in df.groupby("target_id", sort=True):
        b = bearings_deg(
            float(grp["target_lat"].iloc[0]),
            float(grp["target_lon"].iloc[0]),
            grp["vp_lat"].to_numpy(dtype=float),
            grp["vp_lon"].to_numpy(dtype=float),
        )
        g, c = angular_features(b)
        gaps.append(g)
        cvar.append(c)
    return np.asarray(gaps, dtype=float), np.asarray(cvar, dtype=float)


def stage_report(
    frames: dict[str, pd.DataFrame], *, grid: Grid, resolution: int
) -> dict:
    """The three tiers for one stage of the pipeline."""
    nodes, ladders, edge_km, near_km, ang = {}, {}, {}, {}, {}
    for name, df in frames.items():
        nodes[name] = {
            "vps": int(df["vp_id"].nunique()),
            "targets": int(df["target_id"].nunique()),
            "edges": int(len(df)),
        }
        nodes[name]["connectance"] = (
            round(len(df) / (nodes[name]["vps"] * nodes[name]["targets"]), 4)
            if nodes[name]["vps"] and nodes[name]["targets"]
            else None
        )
        ladders[name] = {}
        for side in _SIDES:
            uniq = df.drop_duplicates(f"{side}_id")
            ladders[name][side] = {
                str(r): c
                for r, c in grid.occupied_cell_hierarchy(
                    uniq[f"{side}_lat"].to_numpy(dtype=float),
                    uniq[f"{side}_lon"].to_numpy(dtype=float),
                    _LADDER,
                ).items()
            }
        edge_km[name] = elementwise_km(
            df["vp_lat"].to_numpy(dtype=float), df["vp_lon"].to_numpy(dtype=float),
            df["target_lat"].to_numpy(dtype=float), df["target_lon"].to_numpy(dtype=float),
        )
        near_km[name] = {
            side: nearest_counterpart_km(df, side).to_numpy(dtype=float)
            for side in _SIDES
        }
        ang[name] = angular_per_target(df)

    mats, vc, tc = cell_biadjacency(frames, grid=grid, resolution=resolution)
    sim = graph_similarity(mats["public"], mats["private"])
    sim["union_vp_cells"] = len(vc)
    sim["union_tg_cells"] = len(tc)
    sim["weights_are_forced_outer_product"] = {
        name: bool(np.array_equal(m, _outer_of_node_counts(frames[name], grid, resolution)))
        for name, m in mats.items()
    }

    return {
        "nodes": nodes,
        "cell_graph": sim,
        "occupied_cell_ladder": ladders,
        "residual_geometry": {
            "nearest_measured_vp_km": divergence(
                near_km["public"]["target"], near_km["private"]["target"], unit="km"
            ),
            "nearest_measured_target_km": divergence(
                near_km["public"]["vp"], near_km["private"]["vp"], unit="km"
            ),
            "edge_length_km": divergence(
                edge_km["public"], edge_km["private"], unit="km"
            ),
            "max_angular_gap_deg": divergence(
                ang["public"][0], ang["private"][0], unit="deg"
            ),
            "circular_variance": divergence(
                ang["public"][1], ang["private"][1], unit="dimensionless"
            ),
        },
    }


def _outer_of_node_counts(df: pd.DataFrame, grid: Grid, resolution: int) -> np.ndarray:
    """`outer(n_vp_per_cell, n_tg_per_cell)` on the dataset's own cells.

    Equals the cell-level weight matrix exactly when the dataset is a complete mesh, which
    is the tautology the Tier-2 caveat rests on.
    """
    per = {}
    for side in _SIDES:
        uniq = df.drop_duplicates(f"{side}_id")
        part = grid.partition(
            uniq[f"{side}_lat"].to_numpy(dtype=float),
            uniq[f"{side}_lon"].to_numpy(dtype=float),
            resolution,
        )
        per[side] = pd.Series(part.counts, index=[str(c) for c in part.cells]).sort_index()
    return np.outer(per["vp"].to_numpy(), per["target"].to_numpy())


def similarity_report(
    public: pd.DataFrame, private: pd.DataFrame, *, grid: Grid, resolution: int
) -> dict:
    """Measure all three stages and return one report."""
    resolution = grid.validate_resolution(resolution)
    stages = {"raw": {"public": public, "private": private}}

    pub_r, priv_r, _ = reciprocal_filter(
        public, private, grid=grid, resolution=resolution, balance=False
    )
    stages["reciprocal"] = {"public": pub_r, "private": priv_r}

    pub_b, priv_b, _ = reciprocal_filter(
        public, private, grid=grid, resolution=resolution, balance=True
    )
    stages["balanced"] = {"public": pub_b, "private": priv_b}

    return {
        "grid": grid.describe(resolution),
        "stages": {
            name: stage_report(frames, grid=grid, resolution=resolution)
            for name, frames in stages.items()
        },
    }


_STAGES = ("raw", "reciprocal", "balanced")


def log_report(report: dict) -> None:
    """The contrast table -- the thing this script exists to print."""
    logger.info(
        "cell-level bipartite similarity on %s %s=%d",
        report["grid"]["scheme"], "res", report["grid"]["resolution"],
    )
    logger.info(
        "  %-11s %-17s %-11s %8s %8s %8s %8s",
        "stage", "nodes pub/priv", "cells", "jaccard", "hamming", "cosine", "frob_rel",
    )
    for st in _STAGES:
        s = report["stages"][st]
        n, c = s["nodes"], s["cell_graph"]
        logger.info(
            "  %-11s %-17s %-11s %8s %8s %8s %8s",
            st,
            f"{n['public']['vps']}/{n['public']['targets']}"
            f" {n['private']['vps']}/{n['private']['targets']}",
            f"{c['union_vp_cells']}x{c['union_tg_cells']}",
            c["jaccard_support"], c["hamming_norm"], c["cosine"], c["frobenius_rel"],
        )
    b = report["stages"]["balanced"]
    logger.info(
        "  connectance (balanced): public %s | private %s",
        b["nodes"]["public"]["connectance"], b["nodes"]["private"]["connectance"],
    )
    forced = b["cell_graph"]["weights_are_forced_outer_product"]
    logger.info(
        "  cell weights forced by node counts: public %s | private %s"
        "  <- where True, similarity confirms the construction rather than measuring it",
        forced["public"], forced["private"],
    )
    logger.info("  residual within-cell geometry (balanced stage):")
    for key, d in b["residual_geometry"].items():
        logger.info(
            "      %-26s p50 %8s vs %8s   w1=%-8s ks=%s",
            key, d["public_p50"], d["private_p50"], d["w1"], d["ks"],
        )
    logger.info("  occupied-cell ladder, res %s:", "/".join(map(str, _LADDER)))
    for side in _SIDES:
        lad = b["occupied_cell_ladder"]
        logger.info(
            "      %-8s public %s   private %s",
            side + "s",
            [lad["public"][side][str(r)] for r in _LADDER],
            [lad["private"][side][str(r)] for r in _LADDER],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--public", type=Path, required=True,
                        help="Public canonical CSV (unfiltered; stages are derived here).")
    parser.add_argument("--private", type=Path, required=True,
                        help="Private canonical CSV (unfiltered).")
    parser.add_argument("--grid", default=DEFAULT_GRID, choices=list(GRID_NAMES),
                        help=GRID_HELP)
    parser.add_argument("--resolution", type=int, default=None,
                        help="Grid resolution. Defaults to the grid's own default (h3 res 4).")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Report JSON. Defaults to <public-stem>{_SUFFIX}.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, path in (("public", args.public), ("private", args.private)):
        if not path.exists():
            raise SystemExit(f"{name} input not found: {path}")

    grid = get_grid(args.grid)
    resolution = grid.DEFAULT_RESOLUTION if args.resolution is None else args.resolution
    try:
        resolution = grid.validate_resolution(resolution)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report = similarity_report(
        read_canonical_csv(args.public),
        read_canonical_csv(args.private),
        grid=grid,
        resolution=resolution,
    )
    report = {
        "inputs": {"public": str(args.public), "private": str(args.private)},
        **report,
    }

    out = args.output or _default_output(args.public)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    log_report(report)
    logger.info("  wrote %s", out)


if __name__ == "__main__":
    main()
