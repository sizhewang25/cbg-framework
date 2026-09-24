"""Two labels per prediction: `ring` (grid) and `cell_label` (cell).

## `ring` -- distance, bounded

Ported unchanged from v4. `ring == 0` is the TG's own grid, 1 and 2 the first
and second neighbour rings, `-1` beyond. Local by construction, and `ring <= k`
is monotone non-increasing as grids grow (HEALPix nests exactly), which
`monotonicity_violations` asserts.

## `cell_label` -- direction, bounded by the landmass

* `outland` -- the prediction is outside the landmass (US mainland buffered by
  `grid_km`);
* `true` -- inside it, and its nearest seed is the TG's seed: the prediction
  is in the TG's cell, its serving region;
* `wrong` -- inside it, in another seed's cell;
* `none` -- no prediction to label.

The cell label is **not** a correctness verdict on its own, and it is not the
nearest-seed rule v4 retired: that rule had no outland, so it credited a
prediction 2,360 km away in the Canadian Arctic to Seattle. Here the landmass
bounds it, and the ring bounds how far a `true` can be from the TG. Read the
two together: within each ring tier, how many landed in the right serving
region, a wrong one, or off the landmass.

The cell axis is **not** monotone across rungs -- the seeds change with the
rung -- so it is reported but not asserted.

## Denominator

Every evaluated TG. FALLBACK and ERROR rows are wrong, not excluded: a method
that declines to answer has not earned a smaller denominator than one that
answers badly. A FALLBACK row still gets both labels (its prediction is the
shortest-ping VP's coordinate), but only `solved_mask` rows enter the counts.
An answered row with no coordinate counts as failed, not as beyond: it has no
cell label to put in the cross-tab.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.answer_space import (
    BENCHMARK_TG_COLUMNS,
    GLOSSARY,
    AnswerSpace,
    load_answer_space,
)
from scripts.analysis.v5.modules.geodesy import elementwise_km, pairwise_km
from scripts.analysis.v5.modules.landmass import load_landmass
from scripts.analysis.v5.modules.paths import CLASSIFY_KIND, MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask

ACCURACY_CSV = "accuracy.csv"
TGS_PARQUET = "{method}_tgs.parquet"
MANIFEST_JSON = "manifest.json"
BY_GRID_CSV = "accuracy_by_grid.csv"

#: The labelled outcomes of the cell axis, and the label of a row with no
#: prediction.
CELL_LABELS: tuple[str, ...] = ("true", "wrong", "outland")
NO_PREDICTION = "none"

#: The ring tiers, exclusive, finest first.
RING_TIERS: tuple[str, ...] = ("ring0", "ring1", "ring2", "beyond")

#: Exclusive outcome counts partitioning `n_tgs`: the four tiers, then failed.
OUTCOME_COUNTS: tuple[str, ...] = ("n_ring0", "n_ring1", "n_ring2", "n_beyond", "n_failed")

#: Per-TG columns every scoring path needs, in v5 names.
TG_FRAME_COLUMNS = ("tg_id", "tg_lat", "tg_lon", "pred_lat", "pred_lon", "status")
_BENCHMARK_FRAME_COLUMNS = (*BENCHMARK_TG_COLUMNS, "pred_lat", "pred_lon", "status")


def load_method_frame(run: RunPaths, method: str) -> pd.DataFrame:
    """One row per evaluated TG for `method`, across every fold, in v5 names."""
    frames = []
    for fold in run.fold_ids:
        path = run.combo_dir(method, fold) / "targets.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path, columns=list(_BENCHMARK_FRAME_COLUMNS)).to_pandas()
        t["fold"] = int(fold.split("_")[1])
        frames.append(t)
    if not frames:
        raise MissingArtifactError(f"no targets.parquet for method {method!r} under {run.setup_dir}")
    return pd.concat(frames, ignore_index=True).rename(columns=BENCHMARK_TG_COLUMNS)


_SPING_COLUMNS = ("target_id", "shortest_ping_vp_lat", "shortest_ping_vp_lon")


def load_shortest_ping_frame(run: RunPaths, roster: pd.DataFrame) -> pd.DataFrame:
    """The Shortest-Ping control, shaped like any method frame.

    Read from `eval_source` (the benchmark's own resolution of the lowest-RTT
    VP) and scored over the evaluated `roster`, so it shares every CBG arm's
    denominator: eval-source TGs the run never evaluated are dropped, and
    evaluated TGs without a row keep one with no prediction.
    """
    path = run.eval_file("eval_per_target.csv")
    raw = pd.read_csv(path)
    missing = [c for c in _SPING_COLUMNS if c not in raw.columns]
    if missing:
        raise MissingArtifactError(f"{path} is missing {missing}")
    sping = raw[list(_SPING_COLUMNS)].drop_duplicates("target_id").set_index("target_id")
    out = roster[["tg_id", "tg_lat", "tg_lon", "fold"]].copy()
    hit = sping.reindex(out["tg_id"])
    out["pred_lat"] = hit["shortest_ping_vp_lat"].to_numpy()
    out["pred_lon"] = hit["shortest_ping_vp_lon"].to_numpy()
    out["status"] = "BASELINE"
    out.attrs["n_evaluated_without_baseline"] = int(out["pred_lat"].isna().sum())
    return out


def score_method(
    frame: pd.DataFrame, space: AnswerSpace, *, max_ring: int = G.MAX_RING
) -> pd.DataFrame:
    """Per-TG grid and cell labels, and both distances.

    Every TG in `frame` must be in the answer space. One it does not know is
    refused rather than dropped: the two are built from the same folds, so a
    mismatch means any number over the intersection has the wrong denominator.
    """
    nside = space.nside
    placed = space.tgs.set_index("tg_id")
    unknown = set(frame["tg_id"]) - set(placed.index)
    if unknown:
        raise ValueError(
            f"{len(unknown)} TG(s) are not in the nside={nside} answer space, "
            f"e.g. {sorted(unknown)[:3]}; rebuild it from the same folds"
        )

    out = frame.copy()
    rows = placed.reindex(out["tg_id"])
    out["site_id"] = rows["site_id"].to_numpy()
    out["tg_grid_id"] = rows["tg_grid_id"].to_numpy()
    out["tg_seed_id"] = rows["tg_seed_id"].to_numpy()

    has_pred = (out["pred_lat"].notna() & out["pred_lon"].notna()).to_numpy()
    idx = np.flatnonzero(has_pred)
    plat = out["pred_lat"].to_numpy(dtype=float)[idx]
    plon = out["pred_lon"].to_numpy(dtype=float)[idx]

    # -- grid axis ----------------------------------------------------------
    pred_grid = np.full(len(out), -1, dtype=np.int64)
    ring = np.full(len(out), -1, dtype=np.int64)
    if idx.size:
        pred_grid[idx] = G.ang2pix(plat, plon, nside)
        ring[idx] = G.ring_distance(
            out["tg_grid_id"].to_numpy()[idx], pred_grid[idx], nside, max_ring=max_ring
        )
    out["pred_grid_id"] = pred_grid
    out["ring"] = ring

    dist_tg = np.full(len(out), np.nan)
    if idx.size:
        dist_tg[idx] = elementwise_km(
            out["tg_lat"].to_numpy()[idx], out["tg_lon"].to_numpy()[idx], plat, plon
        )
    out["pred_dist_to_tg_km"] = np.round(dist_tg, 3)

    # -- cell axis ----------------------------------------------------------
    seeds = space.seeds
    in_landmass = np.zeros(len(out), dtype=bool)
    pred_seed = np.full(len(out), -1, dtype=np.int64)
    dist_seed = np.full(len(out), np.nan)
    if idx.size:
        in_landmass[idx] = load_landmass(space.grid_km).contains(plat, plon)
        inside = idx[in_landmass[idx]]
        if inside.size:
            d = pairwise_km(
                out["pred_lat"].to_numpy(dtype=float)[inside],
                out["pred_lon"].to_numpy(dtype=float)[inside],
                seeds["seed_lat"], seeds["seed_lon"],
            )
            pred_seed[inside] = seeds["seed_id"].to_numpy()[d.argmin(axis=1)]
        tg_seed_xy = (
            seeds.set_index("seed_id")
            .loc[out["tg_seed_id"].to_numpy()[idx], ["seed_lat", "seed_lon"]]
            .to_numpy()
        )
        dist_seed[idx] = elementwise_km(plat, plon, tg_seed_xy[:, 0], tg_seed_xy[:, 1])
    out["pred_in_landmass"] = in_landmass
    out["pred_seed_id"] = pred_seed
    out["pred_dist_to_seed_km"] = np.round(dist_seed, 3)

    label = np.full(len(out), NO_PREDICTION, dtype=object)
    label[has_pred & ~in_landmass] = "outland"
    label[in_landmass & (pred_seed == out["tg_seed_id"].to_numpy())] = "true"
    label[in_landmass & (pred_seed != out["tg_seed_id"].to_numpy())] = "wrong"
    out["cell_label"] = label
    return out


def _tier(ring: np.ndarray) -> np.ndarray:
    """Exclusive ring tier per row, as a `RING_TIERS` name."""
    placed = (ring >= 0) & (ring <= G.MAX_RING)
    return np.array(RING_TIERS, dtype=object)[np.where(placed, ring, len(RING_TIERS) - 1)]


def summarize(scored: dict[str, pd.DataFrame], nside: int) -> pd.DataFrame:
    """One row per method: the grid axis, the cell axis, and their cross-tab."""
    rows = []
    for method, df in scored.items():
        n = len(df)
        answered = solved_mask(df).to_numpy() & df["pred_lat"].notna().to_numpy()
        ring = df["ring"].to_numpy()
        label = df["cell_label"].to_numpy()
        tier = _tier(ring)
        status = df["status"].astype(str)
        row = {
            "method": method,
            "nside": nside,
            "grid_km": round(G.grid_km(nside), 1),
            "n_tgs": n,
            "n_solved": int(answered.sum()),
            "n_fallback": int((status == "FALLBACK").sum()),
            "n_error": int((status == "ERROR").sum()),
        }
        # Grid axis. Cumulative: a reader comparing two methods wants "within
        # one ring", not "in exactly the first ring".
        for k in range(G.MAX_RING + 1):
            row[f"accuracy_ring{k}"] = round(
                float((answered & (ring >= 0) & (ring <= k)).mean()), 4
            )
        for name in RING_TIERS:
            row[f"n_{name}"] = int((answered & (tier == name)).sum())
        row["n_failed"] = int((~answered).sum())

        # Cell axis, and the cross-tab that is the point of v5.
        row["accuracy_cell_true"] = round(float((answered & (label == "true")).mean()), 4)
        for lab in CELL_LABELS:
            row[f"n_cell_{lab}"] = int((answered & (label == lab)).sum())
        for name in RING_TIERS:
            for lab in CELL_LABELS:
                row[f"n_{name}_cell_{lab}"] = int(
                    (answered & (tier == name) & (label == lab)).sum()
                )

        for col in ("pred_dist_to_tg_km", "pred_dist_to_seed_km"):
            v = df.loc[answered, col].dropna()
            row[f"{col}_p50"] = round(float(v.quantile(0.50)), 3) if len(v) else None
            row[f"{col}_p90"] = round(float(v.quantile(0.90)), 3) if len(v) else None
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("accuracy_ring0", ascending=False)
    guard_partition(out)
    guard_cross_tab(out)
    return out


def guard_partition(summary: pd.DataFrame) -> None:
    """The five outcome counts must sum to `n_tgs`, per method."""
    total = summary[list(OUTCOME_COUNTS)].sum(axis=1)
    bad = summary.loc[total != summary["n_tgs"]]
    if len(bad):
        raise ValueError(
            f"outcome counts do not partition n_tgs for {bad['method'].tolist()}: "
            f"{total[bad.index].tolist()} vs {bad['n_tgs'].tolist()}"
        )


def guard_cross_tab(summary: pd.DataFrame) -> None:
    """Each tier's three cell labels sum to the tier; the cell totals to `n_solved`.

    Asserted because the failure is invisible in the artifact: a stack that
    sums to 98% looks like a stack.
    """
    problems = []
    for name in RING_TIERS:
        parts = summary[[f"n_{name}_cell_{lab}" for lab in CELL_LABELS]].sum(axis=1)
        for i in summary.index[parts != summary[f"n_{name}"]]:
            problems.append(f"{summary.at[i, 'method']} {name}: {parts[i]} vs {summary.at[i, f'n_{name}']}")
    cells = summary[[f"n_cell_{lab}" for lab in CELL_LABELS]].sum(axis=1)
    for i in summary.index[cells != summary["n_solved"]]:
        problems.append(f"{summary.at[i, 'method']} cell total: {cells[i]} vs {summary.at[i, 'n_solved']}")
    if problems:
        raise ValueError("cell labels do not partition the ring tiers: " + "; ".join(problems))


def score_rung(
    run: RunPaths,
    nside: int,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Score every method at one rung, write the rung's artifacts, return the summary."""
    space_dir = run.answer_space_dir(nside, root=analysis_root)
    if not (space_dir / "meta.json").exists():
        raise MissingArtifactError(
            f"no answer space at {space_dir}; run `build-answer-space` first"
        )
    space = load_answer_space(space_dir)
    combos = run.combo_ids
    wanted = list(methods) if methods else [*combos, SHORTEST_PING]
    scored: dict[str, pd.DataFrame] = {}
    for method in wanted:
        if method == SHORTEST_PING:
            if not combos:
                continue
            frame = load_shortest_ping_frame(run, load_method_frame(run, combos[0]))
        else:
            frame = load_method_frame(run, method)
        scored[method] = score_method(frame, space)

    summary = summarize(scored, nside)
    out_dir = run.classify_dir(nside, root=analysis_root)
    for method, df in scored.items():
        df.to_parquet(out_dir / TGS_PARQUET.format(method=method), index=False)
    summary.to_csv(out_dir / ACCURACY_CSV, index=False)
    (out_dir / MANIFEST_JSON).write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "source": run.source,
                "setup": run.setup,
                "grid": G.describe(nside),
                "landmass": space.meta["landmass"],
                "seed_rule": space.meta["seed_rule"],
                "n_sites": space.meta["n_sites"],
                "n_seeds": space.meta["n_seeds"],
                "methods": sorted(scored),
                "max_ring": G.MAX_RING,
                "ring": "grid steps from the TG's grid; ringK cumulative; -1 beyond",
                "cell_label": (
                    "outland = outside the landmass; true / wrong = nearest seed is / "
                    "is not the TG's seed; none = no prediction"
                ),
                "cross_tab": "n_<tier>_cell_<label>, exclusive, answered rows only",
                "fallback_policy": "FALLBACK and ERROR rows are wrong, not excluded",
                "glossary": GLOSSARY,
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def score_for_run(
    run: RunPaths,
    *,
    nsides: tuple[int, ...] = G.NSIDE_LADDER,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Score every rung and write `accuracy_by_grid.csv`. Returns the long frame."""
    parts = [
        score_rung(run, nside, methods=methods, analysis_root=analysis_root)
        for nside in sorted({G.validate_nside(n) for n in nsides}, reverse=True)
    ]
    long = pd.concat(parts, ignore_index=True)
    if len(parts) > 1:
        long.to_csv(run.analysis_dir(CLASSIFY_KIND, root=analysis_root) / BY_GRID_CSV, index=False)
    return long


def monotonicity_violations(long: pd.DataFrame) -> pd.DataFrame:
    """Rows where a coarser rung scored a lower `accuracy_ring0`. Must be empty.

    The grid axis only: exact nesting guarantees it. The cell axis has no such
    guarantee, because the seeds are regrouped at every rung.
    """
    bad = []
    for method, g in long.groupby("method"):
        g = g.sort_values("nside", ascending=False)
        acc = g["accuracy_ring0"].to_numpy()
        ns = g["nside"].to_numpy()
        for i in range(1, len(acc)):
            if acc[i] < acc[i - 1] - 1e-9:
                bad.append(
                    {
                        "method": method,
                        "finer_nside": int(ns[i - 1]),
                        "coarser_nside": int(ns[i]),
                        "finer_accuracy": float(acc[i - 1]),
                        "coarser_accuracy": float(acc[i]),
                    }
                )
    return pd.DataFrame(bad)
