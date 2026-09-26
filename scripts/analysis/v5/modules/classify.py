"""Two labels per prediction: `ring` (grid) and `cell_label` (cell).

## `ring` -- distance, bounded

Ported unchanged from v4. `ring == 0` is the TG's own grid, 1 and 2 the first
and second neighbour rings, `-1` beyond. Local by construction, and `ring <= k`
would be monotone non-increasing as grids grew, because HEALPix nests exactly
-- v5 runs one resolution, so nothing exercises that any more.

`pred_dist_to_tg_grid` is the same measure uncapped, and the two agree exactly
wherever `ring` answers at all. The cap is what the tiers and the figures are
built on and it stays; the uncapped column exists because `-1` is over half of
the rows on the current runs, spanning 3 to 68 grids out, and one flat "further
out" bucket hides all of that. Read `ring` for the verdict, this for the shape.

## `cell_label` -- direction, unbounded

* `correct` -- the prediction's nearest seed is the TG's seed: it is in the
  TG's cell, its serving region;
* `wrong` -- it is in another seed's cell;
* `unanswered` -- there is no prediction to label.

The three labels partition every evaluated TG, so the cell axis and the grid
axis share one denominator.

**This is deliberately the unbounded nearest-seed rule** -- the one v4 retired
for crediting a prediction 2,360 km away in the Canadian Arctic to Seattle.
v4's fix was a landmass polygon and a fourth label, `outland`. v5 removes both
and keeps the failure visible, because the failure is the finding: a
nearest-seed verdict alone will call a prediction correct at any distance,
and `ring` is what bounds it. Read the two labels together -- within each ring
tier, how many landed in the right serving region and how many in a
neighbour's.

## Denominator

Every evaluated TG, on both axes. FALLBACK and ERROR rows are wrong, not
excluded: a method that declines to answer has not earned a smaller
denominator than one that answers badly. A FALLBACK row still gets both labels
(its prediction is the shortest-ping VP's coordinate), but only `solved_mask`
rows enter the `correct`/`wrong` counts -- the rest are `unanswered`, which is
a label rather than an exclusion, so `n_cell_unanswered == n_failed` by
construction and `guard_cross_tab` asserts it.
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
from scripts.analysis.v5.modules.paths import CLASSIFY_KIND, MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask

ACCURACY_CSV = "accuracy.csv"
TGS_PARQUET = "{method}_tgs.parquet"
MANIFEST_JSON = "manifest.json"
BY_GRID_CSV = "accuracy_by_grid.csv"

#: The cell axis, exhaustive: every evaluated TG carries exactly one. The
#: ungraded one comes last -- it has no ring, so no tier breakdown.
CELL_LABELS: tuple[str, ...] = ("correct", "wrong", "unanswered")

#: The cell labels that carry a ring tier, i.e. everything but `unanswered`.
GRADED_CELL_LABELS: tuple[str, ...] = ("correct", "wrong")
UNANSWERED = "unanswered"

#: The uncapped grid axis: `ring` without the cap, so `-1` here means only
#: "no prediction". Written by `score_method`; parquets from before it exists
#: are refused by `summarize` rather than silently dropping the column.
GRID_OFFSET = "pred_dist_to_tg_grid"

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

    # The same measure as `ring` without the cap, so the spread `ring` pools
    # into `-1` stays readable. Note the two distance-to-TG columns mark "no
    # answer" differently: `-1` here because the column is integral, NaN for
    # the kilometre one below.
    offset = np.full(len(out), -1, dtype=np.int64)
    if idx.size:
        offset[idx] = G.grid_offset(out["tg_grid_id"].to_numpy()[idx], pred_grid[idx], nside)
    out["pred_dist_to_tg_grid"] = offset

    dist_tg = np.full(len(out), np.nan)
    if idx.size:
        dist_tg[idx] = elementwise_km(
            out["tg_lat"].to_numpy()[idx], out["tg_lon"].to_numpy()[idx], plat, plon
        )
    out["pred_dist_to_tg_km"] = np.round(dist_tg, 3)

    # -- cell axis ----------------------------------------------------------
    seeds = space.seeds
    pred_seed = np.full(len(out), -1, dtype=np.int64)
    dist_seed = np.full(len(out), np.nan)
    if idx.size:
        # Every prediction gets a nearest seed. No containment test, no gate:
        # the rule is unbounded and this is the whole of it.
        d = pairwise_km(plat, plon, seeds["seed_lat"], seeds["seed_lon"])
        pred_seed[idx] = seeds["seed_id"].to_numpy()[d.argmin(axis=1)]
        tg_seed_xy = (
            seeds.set_index("seed_id")
            .loc[out["tg_seed_id"].to_numpy()[idx], ["seed_lat", "seed_lon"]]
            .to_numpy()
        )
        dist_seed[idx] = elementwise_km(plat, plon, tg_seed_xy[:, 0], tg_seed_xy[:, 1])
    out["pred_seed_id"] = pred_seed
    out["pred_dist_to_seed_km"] = np.round(dist_seed, 3)

    # `has_pred` is load-bearing on both lines. Without it a row with no
    # prediction keeps `pred_seed == -1`, never matches `tg_seed_id`, and is
    # silently labelled `wrong` instead of `unanswered`.
    label = np.full(len(out), UNANSWERED, dtype=object)
    label[has_pred & (pred_seed == out["tg_seed_id"].to_numpy())] = "correct"
    label[has_pred & (pred_seed != out["tg_seed_id"].to_numpy())] = "wrong"
    out["cell_label"] = label
    return out


def _tier(ring: np.ndarray) -> np.ndarray:
    """Exclusive ring tier per row, as a `RING_TIERS` name."""
    placed = (ring >= 0) & (ring <= G.MAX_RING)
    return np.array(RING_TIERS, dtype=object)[np.where(placed, ring, len(RING_TIERS) - 1)]


def summarize(scored: dict[str, pd.DataFrame], nside: int) -> pd.DataFrame:
    """One row per method: the grid axis, the cell axis, and their cross-tab."""
    stale = [m for m, df in scored.items() if GRID_OFFSET not in df.columns]
    if stale:
        raise MissingArtifactError(
            f"{stale} have no {GRID_OFFSET!r} column; these are per-TG frames from "
            f"before it existed -- re-run `classify` to rewrite them"
        )
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

        # Cell axis, and the cross-tab that is the point of v5. `unanswered`
        # is the complement of `answered`, so the three counts close on n_tgs
        # and only the two graded labels get a tier breakdown.
        row["accuracy_cell_correct"] = round(
            float((answered & (label == "correct")).mean()), 4
        )
        for lab in GRADED_CELL_LABELS:
            row[f"n_cell_{lab}"] = int((answered & (label == lab)).sum())
        row[f"n_cell_{UNANSWERED}"] = int((~answered).sum())
        for name in RING_TIERS:
            for lab in GRADED_CELL_LABELS:
                row[f"n_{name}_cell_{lab}"] = int(
                    (answered & (tier == name) & (label == lab)).sum()
                )

        for col in ("pred_dist_to_tg_km", "pred_dist_to_seed_km"):
            v = df.loc[answered, col].dropna()
            row[f"{col}_p50"] = round(float(v.quantile(0.50)), 3) if len(v) else None
            row[f"{col}_p90"] = round(float(v.quantile(0.90)), 3) if len(v) else None

        # The grid axis as a distribution rather than four tiers: this is what
        # `n_beyond` pools. Its own block, not a third column in the loop
        # above -- `dropna` does not catch the `-1`, and a count wants a max
        # and coarser rounding than a kilometre does. `answered` already
        # implies a prediction, so the filter is belt and braces.
        g = df.loc[answered, GRID_OFFSET]
        g = g[g >= 0]
        row[f"{GRID_OFFSET}_p50"] = round(float(g.quantile(0.50)), 1) if len(g) else None
        row[f"{GRID_OFFSET}_p90"] = round(float(g.quantile(0.90)), 1) if len(g) else None
        row[f"{GRID_OFFSET}_max"] = int(g.max()) if len(g) else None
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
    """The cell axis closes, per tier and overall, and agrees with the grid axis.

    Three assertions. Each tier's two graded labels sum to the tier; all three
    cell labels sum to `n_tgs`; and `n_cell_unanswered` equals the grid axis's
    `n_failed`, which is the same rows counted along the other axis and so is
    a free cross-check on both.

    Asserted because the failure is invisible in the artifact: a stack that
    sums to 98% looks like a stack.
    """
    problems = []
    for name in RING_TIERS:
        parts = summary[[f"n_{name}_cell_{lab}" for lab in GRADED_CELL_LABELS]].sum(axis=1)
        for i in summary.index[parts != summary[f"n_{name}"]]:
            problems.append(
                f"{summary.at[i, 'method']} {name}: {parts[i]} vs {summary.at[i, f'n_{name}']}"
            )
    cells = summary[[f"n_cell_{lab}" for lab in CELL_LABELS]].sum(axis=1)
    for i in summary.index[cells != summary["n_tgs"]]:
        problems.append(
            f"{summary.at[i, 'method']} cell total: {cells[i]} vs {summary.at[i, 'n_tgs']}"
        )
    unanswered = summary[f"n_cell_{UNANSWERED}"]
    for i in summary.index[unanswered != summary["n_failed"]]:
        problems.append(
            f"{summary.at[i, 'method']} unanswered: {unanswered[i]} vs "
            f"n_failed {summary.at[i, 'n_failed']}"
        )
    if problems:
        raise ValueError("the cell axis does not close: " + "; ".join(problems))


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
                "projection": space.meta["projection"],
                "seed_rule": space.meta["seed_rule"],
                "n_sites": space.meta["n_sites"],
                "n_seeds": space.meta["n_seeds"],
                "methods": sorted(scored),
                "max_ring": G.MAX_RING,
                "ring": "grid steps from the TG's grid; ringK cumulative; -1 beyond",
                GRID_OFFSET: (
                    "the same grid steps uncapped, so the spread ring pools into "
                    "-1 stays readable; -1 only when there is no prediction"
                ),
                "cell_label": (
                    "correct / wrong = the prediction's nearest seed is / is not "
                    "the TG's seed, unbounded -- no containment test of any kind; "
                    "unanswered = no prediction. The three partition n_tgs."
                ),
                "cross_tab": "n_<tier>_cell_<label>, exclusive; graded labels only",
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
