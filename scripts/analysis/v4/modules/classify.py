"""Laddered, ring-graded classification accuracy — v4's reason for existing.

## What was wrong

v3 scored a prediction by **nearest-seed assignment**: whichever class seed was
closest won, and the prediction was correct if that was the truth's class. A
Voronoi partition over K seeds labels *every point on Earth*, so the rule can
never answer "this is in no class". Verified consequence, target `tg-e1a1545`
on as01: truth Seattle (47.45, -122.31), Spotter predicted (63.74, -97.47) in
the Canadian Arctic — **2,360 km away, scored CORRECT**, because the nearest of
18 US seeds was Seattle at 2,380 km and the runner-up Omaha at 2,492 km. A
113 km margin between two absurd options decided it.

Systematic, not anecdotal. Among v3's *correct* classifications on as01, **345
of 354** Spotter predictions sat more than 45 km from the seed they were
credited to (210/210 on as02, 263/268 on as03) — while the answer space's own
metadata said no real target is more than 25.3 km from its seed.

## What replaces it

**Containment, graded by ring, over a ladder of cell sizes.**

* `ring == 0` — the prediction is in the truth's own cell.
* `ring == 1` — in one of its 8 neighbours.
* `ring == 2` — in the second ring.
* `ring == -1` — further out. Deliberately not a number: once the prediction
  has left the neighbourhood the metric asks about, *how far* is `error_km`'s
  question, and a large ring count would invite reading it as a distance.

This is **local by construction**, which is precisely the property nearest-seed
lacked: no arrangement of far-away cells can make a distant cell adjacent. The
Arctic prediction is `-1` at every rung of the ladder.

Resolution supplies the tolerance dial. nside 128 asks "the right 51 km cell?";
nside 16 asks "the right 407 km region?". Because HEALPix is aperture-4 and
exactly nested, `accuracy_ring0` is **monotone non-increasing** as cells grow —
a guarantee, not an observation. The same question on H3 produced 705
monotonicity violations on this repo's data, which is why v4 is HEALPix-only.

## The retired number is kept for one release

`accuracy_nearest_seed_retired` reproduces v3's rule alongside the new columns.
Not for use — its name says so — but the gap between it and `accuracy_ring0`
is the measurement that justifies the change, and it can only be taken while
both are computed on identical inputs.

## Fallback and error rows

A FALLBACK or ERROR row is **wrong, not excluded**. The denominator is every
target the run evaluated, because a method that declines to answer has not
earned a smaller denominator than one that answers badly. `error_km`
percentiles exclude them, since a row with no prediction has no error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.answer_space import (
    AnswerSpace,
    elementwise_km,
    load_answer_space,
    pairwise_km,
)
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths

#: The Shortest-Ping control. Not a combo — it has no LTD/MTL/CTR and no
#: `targets.parquet`; its prediction is the nearest-by-RTT VP's own coordinate.
SHORTEST_PING = "shortest_ping"

ACCURACY_CSV = "accuracy.csv"
CELLS_PARQUET = "{method}_cells.parquet"
MANIFEST_JSON = "manifest.json"

#: Merged across rungs, one level above the rung directories.
BY_RESOLUTION_CSV = "accuracy_by_resolution.healpix.csv"

#: The flat per-target columns every scoring path needs. Public because
#: `map_mtl` asks for these *plus* the nested constraint columns, and a second
#: literal copy of the list is the thing that drifts.
TARGET_COLUMNS = (
    "target_id", "target_lat", "target_lon", "pred_lat", "pred_lon", "status",
)


def solved_mask(df: pd.DataFrame) -> pd.Series:
    """Which rows the method actually answered.

    `SUCCESS` for a CBG arm. The Shortest-Ping control writes `BASELINE` on
    every row and has no fallback path to take, so an all-`BASELINE` frame is
    wholly solved — otherwise its accuracy would be divided by zero answers.
    """
    status = df["status"].astype(str)
    if (status == "BASELINE").all():
        return pd.Series(True, index=df.index)
    return status == "SUCCESS"


def load_method_frame(
    run: RunPaths, method: str, *, columns: tuple[str, ...] = TARGET_COLUMNS
) -> pd.DataFrame:
    """One row per evaluated target for `method`, across every fold.

    `columns` exists for the case viewer, which needs the nested
    `ltd_predictions` / `mtl_participants` structs alongside the flat columns.
    Widening here rather than adding a second fold walker: this is the only
    function in v4 that knows how `fold_*/<combo>/targets.parquet` is laid out
    and how the `fold` column is stamped, and a second one would be free to
    disagree with it.

    Columns `score_method` does not write ride through it untouched (it copies
    the frame and only assigns named columns), so a caller that widens here
    gets them back on the scored frame.
    """
    frames = []
    for fold in run.fold_ids:
        path = run.combo_dir(method, fold) / "targets.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path, columns=list(columns)).to_pandas()
        t["fold"] = int(fold.split("_")[1])
        frames.append(t)
    if not frames:
        raise MissingArtifactError(
            f"no targets.parquet for method {method!r} under {run.setup_dir}"
        )
    return pd.concat(frames, ignore_index=True)


#: Columns `load_shortest_ping_frame` needs. A run missing any of them cannot
#: be scored against the baseline at all, so the check belongs here rather than
#: at each call site.
_SPING_COLUMNS = ("target_id", "shortest_ping_vp_lat", "shortest_ping_vp_lon")


def load_shortest_ping_frame(run: RunPaths, roster: pd.DataFrame) -> pd.DataFrame:
    """The Shortest-Ping control, shaped like any other method's frame.

    Its estimate is the lowest-RTT VP's own coordinate, which `eval_source`
    already resolved. Read from there rather than re-minimising over the
    canonical CSV's RTTs: a second minimisation would break ties differently on
    any target whose two lowest RTTs are equal, leaving v4 disagreeing with the
    benchmark instead of agreeing with it.

    Fold-independent by construction — K-fold splits targets, not VPs, so every
    VP is available in every fold and a target's shortest-ping VP does not
    depend on which fold held it out. `fold` therefore comes from the roster.

    **Scored over the evaluated roster, not over the eval source**, and the
    restriction cuts both ways:

    * eval-source targets the run never evaluated are dropped;
    * evaluated targets the eval source has no row for keep a row with **no
      prediction**, so the baseline scores a miss there rather than shrinking
      its denominator.

    Without that, the baseline's population would be whatever the eval CSV
    describes, and every figure comparing it with a CBG arm would divide by two
    different numbers.
    """
    path = run.eval_file("eval_per_target.csv")
    raw = pd.read_csv(path)
    missing = [c for c in _SPING_COLUMNS if c not in raw.columns]
    if missing:
        raise MissingArtifactError(f"{path} is missing {missing}")

    sping = raw[list(_SPING_COLUMNS)].drop_duplicates("target_id").set_index("target_id")
    out = roster[["target_id", "target_lat", "target_lon", "fold"]].copy()
    hit = sping.reindex(out["target_id"])
    out["pred_lat"] = hit["shortest_ping_vp_lat"].to_numpy()
    out["pred_lon"] = hit["shortest_ping_vp_lon"].to_numpy()
    # BASELINE, not SUCCESS: the control has no LTD/MTL/CTR pipeline and so no
    # fallback path, and `solved_mask` reads an all-BASELINE frame as wholly
    # solved. A row with no eval source still counts in the denominator and
    # scores a miss, because its prediction is NaN.
    out["status"] = "BASELINE"
    out.attrs["n_baseline_only_dropped"] = int(
        len(set(sping.index) - set(out["target_id"]))
    )
    out.attrs["n_evaluated_without_baseline"] = int(out["pred_lat"].isna().sum())
    return out


def score_method(
    frame: pd.DataFrame, space: AnswerSpace, *, max_ring: int = H.MAX_RING
) -> pd.DataFrame:
    """Per-target cells, ring distance, and the retired nearest-seed verdict.

    Every target in `frame` must be in the answer space. A target the space does
    not know is refused rather than dropped: the space is built from the same
    fold parquets, so a mismatch means the two are out of step and any number
    computed over the intersection would be reported against the wrong
    denominator.
    """
    nside = space.nside
    assign = space.assignments.set_index("target_id")
    unknown = set(frame["target_id"]) - set(assign.index)
    if unknown:
        raise ValueError(
            f"{len(unknown)} target(s) are not in the nside={nside} answer space, "
            f"e.g. {sorted(unknown)[:3]}; rebuild it from the same folds"
        )

    out = frame.copy()
    rows = assign.reindex(out["target_id"])
    out["tg_cell"] = rows["cell_id"].to_numpy()
    out["tg_seed_id"] = rows["seed_id"].to_numpy()

    has_pred = out["pred_lat"].notna() & out["pred_lon"].notna()
    out["pred_cell"] = np.int64(-1)
    if has_pred.any():
        out.loc[has_pred, "pred_cell"] = H.ang2pix(
            out.loc[has_pred, "pred_lat"], out.loc[has_pred, "pred_lon"], nside
        )

    # Ring distance, only where there is a prediction to place.
    ring = np.full(len(out), -1, dtype=np.int64)
    if has_pred.any():
        idx = np.flatnonzero(has_pred.to_numpy())
        ring[idx] = H.ring_distance(
            out["tg_cell"].to_numpy()[idx],
            out["pred_cell"].to_numpy()[idx],
            nside,
            max_ring=max_ring,
        )
    out["ring"] = ring

    # Error to the true target coordinate — independent of the grid, and the
    # metric to read once `ring` says the prediction left the neighbourhood.
    err = np.full(len(out), np.nan)
    if has_pred.any():
        m = has_pred.to_numpy()
        err[m] = elementwise_km(
            out.loc[has_pred, "target_lat"], out.loc[has_pred, "target_lon"],
            out.loc[has_pred, "pred_lat"], out.loc[has_pred, "pred_lon"],
        )
    out["error_km"] = np.round(err, 3)

    # The retired rule, recomputed on identical inputs so the gap is measurable.
    seeds = space.seeds
    nearest = np.full(len(out), -1, dtype=np.int64)
    if has_pred.any() and len(seeds):
        d = pairwise_km(
            out.loc[has_pred, "pred_lat"], out.loc[has_pred, "pred_lon"],
            seeds["seed_lat"], seeds["seed_lon"],
        )
        nearest[has_pred.to_numpy()] = seeds["seed_id"].to_numpy()[d.argmin(axis=1)]
    out["nearest_seed_id_retired"] = nearest
    return out


def summarize(scored: dict[str, pd.DataFrame], nside: int) -> pd.DataFrame:
    """One row per method: ring accuracies, the retired number, error spread."""
    rows = []
    for method, df in scored.items():
        solved = solved_mask(df)
        n = len(df)
        ring = df["ring"].to_numpy()
        placed = ring >= 0
        err = df.loc[solved, "error_km"].dropna()
        row = {
            "method": method,
            "nside": nside,
            "cell_km": round(H.nominal_cell_km(nside), 1),
            "n_targets": n,
            "n_solved": int(solved.sum()),
            "n_fallback": int((df["status"].astype(str) == "FALLBACK").sum()),
            "n_error": int((df["status"].astype(str) == "ERROR").sum()),
            "fallback_rate": round(
                float((df["status"].astype(str) == "FALLBACK").mean()), 4
            ),
        }
        # Cumulative: ring1 includes ring0. A reader comparing two methods wants
        # "within one cell", not "in exactly the first ring".
        for k in range(H.MAX_RING + 1):
            hit = placed & (ring <= k) & solved.to_numpy()
            row[f"accuracy_ring{k}"] = round(float(hit.mean()), 4)
        row["n_unplaced"] = int((~placed).sum())

        # EXCLUSIVE counts, so the outcome is a partition of `n_targets` that a
        # figure can stack directly. Derived from the rounded cumulative shares
        # instead, a bar would not close: four values rounded to 4dp and
        # differenced can miss the total by a target or two, and the reader
        # would see a stack that does not reach 100%.
        sv = solved.to_numpy()
        for k in range(H.MAX_RING + 1):
            row[f"n_ring{k}"] = int((sv & placed & (ring == k)).sum())
        # Answered, but further out than the metric grades. Distinct from
        # `n_failed`: this method produced a coordinate and it was nowhere near.
        row["n_beyond"] = int((sv & ~placed).sum())
        # Never produced a coordinate at all -- fallback or crash.
        row["n_failed"] = int((~sv).sum())
        row["accuracy_nearest_seed_retired"] = round(
            float(
                (
                    (df["nearest_seed_id_retired"] == df["tg_seed_id"])
                    & solved
                ).mean()
            ),
            4,
        )
        row["error_km_p50"] = round(float(err.quantile(0.50)), 3) if len(err) else None
        row["error_km_p90"] = round(float(err.quantile(0.90)), 3) if len(err) else None
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("accuracy_ring0", ascending=False)
    guard_partition(out)
    return out


#: The exclusive outcome segments, bottom of a stack to top: how much landed in
#: the cell, how much one ring out, two rings out, further than that, and how
#: much never answered.
OUTCOME_COUNTS: tuple[str, ...] = (
    "n_ring0", "n_ring1", "n_ring2", "n_beyond", "n_failed",
)


def guard_partition(summary: pd.DataFrame) -> None:
    """The five outcome counts must sum to `n_targets`, per method.

    Asserted rather than trusted because the failure is invisible in the
    artifact: a stack that sums to 0.98 looks like a stack, and the missing 2%
    would be read as a rendering quirk rather than as a scoring bug.
    """
    total = summary[list(OUTCOME_COUNTS)].sum(axis=1)
    bad = summary.loc[total != summary["n_targets"]]
    if len(bad):
        raise ValueError(
            "outcome counts do not partition n_targets for "
            f"{bad['method'].tolist()}: "
            f"{total[bad.index].tolist()} vs {bad['n_targets'].tolist()}"
        )


def score_rung(
    run: RunPaths,
    nside: int,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Score every method at one rung, write the rung's artifacts, return the
    summary."""
    space = load_answer_space(run.answer_space_dir(nside, root=analysis_root))
    combos = run.combo_ids
    # The control is appended rather than discovered: it has no
    # `targets.parquet`, so `combo_ids` cannot see it, and leaving it out was
    # the gap this fixes -- every figure lost its baseline column.
    wanted = list(methods) if methods else [*combos, SHORTEST_PING]
    scored: dict[str, pd.DataFrame] = {}
    for method in wanted:
        if method == SHORTEST_PING:
            if not combos:
                continue
            roster = load_method_frame(run, combos[0])
            frame = load_shortest_ping_frame(run, roster)
        else:
            frame = load_method_frame(run, method)
        scored[method] = score_method(frame, space)

    summary = summarize(scored, nside)
    out_dir = run.cls_accuracy_dir(nside, root=analysis_root)
    for method, df in scored.items():
        df.to_parquet(out_dir / CELLS_PARQUET.format(method=method), index=False)
    summary.to_csv(out_dir / ACCURACY_CSV, index=False)
    (out_dir / MANIFEST_JSON).write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "source": run.source,
                "setup": run.setup,
                "grid": H.describe(nside),
                "n_seeds": int(space.n_seeds),
                "methods": sorted(scored),
                "max_ring": H.MAX_RING,
                "correctness": (
                    "ring0 = prediction in the target's own cell; ringK cumulative "
                    "out to K neighbour rings; unplaced beyond that"
                ),
                "retired": (
                    "accuracy_nearest_seed_retired reproduces v3's nearest-seed rule "
                    "for comparison only — it credits a prediction to whichever class "
                    "seed is closest, with no bound, so it scores a 2,360 km miss as "
                    "correct. Do not publish it."
                ),
                "fallback_policy": "FALLBACK and ERROR rows are wrong, not excluded",
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def score_for_run(
    run: RunPaths,
    *,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """Score every rung and write the merged curve. Returns the long frame."""
    parts = [
        score_rung(run, nside, methods=methods, analysis_root=analysis_root)
        for nside in sorted({H.validate_nside(n) for n in nsides}, reverse=True)
    ]
    long = pd.concat(parts, ignore_index=True)
    if len(parts) > 1:
        long.to_csv(
            run.analysis_dir("target-cls-accuracy", root=analysis_root)
            / BY_RESOLUTION_CSV,
            index=False,
        )
    return long


def monotonicity_violations(long: pd.DataFrame) -> pd.DataFrame:
    """Rows where a coarser rung scored *worse* than a finer one.

    Must be empty. Exact nesting means a prediction sharing a cell at nside N
    shares one at every coarser rung, so `accuracy_ring0` cannot fall as cells
    grow. This is the check that H3 fails 705 times, and the reason the grid is
    not a parameter.
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
