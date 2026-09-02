"""Score predictions against an answer space: distance to *every* seed (§8.1).

The answer space is an **explicit input** (`--answer-space`), never implied, so
the same predictions can be re-scored under a different quantization without
re-running anything upstream.

For each (method, target) this emits the distance from the estimated coordinate
to all K seeds. That is deliberately more than a label: because the full
distance vector is kept, the rank of the true seed falls out of it, and top-N
classification accuracy for *any* N is then `(truth_seed_rank < N).mean()` with
no recomputation. Rank 0 is the ordinary top-1 answer.

**Policy separation.** The per-target artifact is neutral: it carries a distance
row for every target that has a predicted coordinate, `FALLBACK` rows included,
since a coordinate does exist there. The paper §7.2 rule that fallbacks count as
failures is applied in the *summary*, which reports accuracy both with and
without fallback rows. Keeping the raw table policy-free is what lets the
fallback question be re-asked later.

Shortest-Ping is scored here as a method, not a footnote — same answer space,
same scorer, same folds (§5). Its coordinate is the shortest-ping VP's own
location, read from `eval_source`.

Command: `classify`. Writes to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.answer_space import (
    AnswerSpace,
    load_answer_space,
    pairwise_km,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    RunPaths,
    discover_runs,
    resolve_run,
)

#: Method id for the baseline. Not a combo — synthesized from eval_source.
SHORTEST_PING = "shortest_ping"

#: Ns reported in the summary; the per-target artifact supports any N.
DEFAULT_TOPN = (1, 2, 3, 5)

_DIST_PREFIX = "dist_km__seed_"
TOPN_CSV = "topn_accuracy.csv"
MANIFEST_JSON = "manifest.json"


def _seed_distance_frame(
    space: AnswerSpace,
    *,
    method: str,
    target_id: pd.Series,
    fold: pd.Series,
    status: pd.Series,
    pred_lat: pd.Series,
    pred_lon: pd.Series,
    truth_seed_id: pd.Series,
) -> pd.DataFrame:
    """Distances from each prediction to all K seeds, plus the derived ranks.

    Rows whose prediction is missing (no coordinate at all) keep NaN distances
    and a null `pred_seed_id`; they are still emitted so the target set stays
    the complete denominator.
    """
    seeds = space.seeds
    seed_ids = seeds["seed_id"].to_numpy()
    d = pairwise_km(
        pred_lat.to_numpy(dtype=float),
        pred_lon.to_numpy(dtype=float),
        seeds["centroid_lat"].to_numpy(),
        seeds["centroid_lon"].to_numpy(),
    )

    has_pred = np.isfinite(pred_lat.to_numpy(dtype=float)) & np.isfinite(
        pred_lon.to_numpy(dtype=float)
    )
    d = np.where(has_pred[:, None], d, np.nan)

    truth = truth_seed_id.to_numpy()
    # Column position of each target's true seed (seed_id is 0..K-1 by
    # construction, but map explicitly so a filtered answer space still works).
    pos_of_seed = {int(s): i for i, s in enumerate(seed_ids)}
    truth_pos = np.array([pos_of_seed.get(int(s), -1) for s in truth])
    if (truth_pos < 0).any():
        bad = np.unique(truth[truth_pos < 0])[:5]
        raise ValueError(
            f"truth seed id(s) {bad.tolist()} are absent from the answer space; "
            f"predictions and answer space disagree on the seed set"
        )

    rows = np.arange(len(d))
    err_to_truth = d[rows, truth_pos]

    pred_pos = np.full(len(d), -1, dtype=int)
    rank = np.full(len(d), -1, dtype=int)
    if has_pred.any():
        sub = d[has_pred]
        pred_pos[has_pred] = sub.argmin(axis=1)
        # Rank of the true seed when seeds are ordered by distance ascending:
        # how many seeds are strictly closer than the true one. Ties resolve in
        # the true seed's favour, which matches "the estimate is consistent with
        # this class".
        rank[has_pred] = (
            sub < err_to_truth[has_pred][:, None] - 1e-9
        ).sum(axis=1)

    out = pd.DataFrame(
        {
            "method": method,
            "target_id": target_id.to_numpy(),
            "fold": fold.to_numpy(),
            "status": status.to_numpy(),
            "pred_lat": pred_lat.to_numpy(dtype=float),
            "pred_lon": pred_lon.to_numpy(dtype=float),
            "truth_seed_id": truth,
            "pred_seed_id": np.where(pred_pos >= 0, seed_ids[pred_pos], -1),
            "truth_seed_rank": rank,
            "error_to_truth_seed_km": np.round(err_to_truth, 3),
            "error_to_pred_seed_km": np.round(
                np.where(pred_pos >= 0, d[rows, np.clip(pred_pos, 0, None)], np.nan), 3
            ),
        }
    )
    dist_cols = pd.DataFrame(
        np.round(d, 3), columns=[f"{_DIST_PREFIX}{int(s)}" for s in seed_ids]
    )
    return pd.concat([out, dist_cols], axis=1)


def score_combo(run: RunPaths, space: AnswerSpace, combo_id: str) -> pd.DataFrame:
    """Seed distances for one CBG combo, pooled over folds."""
    df = io.load_folds(run, combo_id)
    truth = space.assignments.set_index("target_id")["seed_id"]
    unknown = ~df["target_id"].isin(truth.index)
    if unknown.any():
        missing = df.loc[unknown, "target_id"].unique()[:5].tolist()
        raise ValueError(
            f"combo {combo_id!r}: {int(unknown.sum())} target(s) absent from the answer "
            f"space (e.g. {missing}). The answer space must be built from the same "
            f"run's targets.csv."
        )
    return _seed_distance_frame(
        space,
        method=combo_id,
        target_id=df["target_id"],
        fold=df["fold"],
        status=df["status"],
        pred_lat=df["pred_lat"],
        pred_lon=df["pred_lon"],
        truth_seed_id=df["target_id"].map(truth),
    )


def score_shortest_ping(run: RunPaths, space: AnswerSpace) -> pd.DataFrame:
    """Seed distances for the Shortest-Ping baseline.

    The estimate is the lowest-RTT VP's own coordinate. `eval_source` already
    resolves that VP per target, and the choice is fold-independent because
    every VP is available in every fold — only targets are folded.

    `fold` is taken from any CBG combo's fold assignment so the baseline sits on
    the same fold partition as the variants; it is left as -1 if no combo exists.
    """
    ev = io.load_eval_per_target(run)
    need = {"target_id", "shortest_ping_vp_lat", "shortest_ping_vp_lon"}
    missing = need - set(ev.columns)
    if missing:
        raise ValueError(
            f"{run.eval_file('eval_per_target.csv')} lacks {sorted(missing)}; "
            f"cannot score the Shortest-Ping baseline"
        )

    truth = space.assignments.set_index("target_id")["seed_id"]
    ev = ev[ev["target_id"].isin(truth.index)].reset_index(drop=True)

    fold = pd.Series(-1, index=ev.index, dtype=int)
    combos = run.combo_ids
    if combos:
        ref = io.load_folds(run, combos[0], columns=["target_id"])
        # `load_folds` adds `fold`; map it onto the baseline's target order.
        fold_by_target = ref.set_index("target_id")["fold"]
        fold = ev["target_id"].map(fold_by_target).fillna(-1).astype(int)

    return _seed_distance_frame(
        space,
        method=SHORTEST_PING,
        target_id=ev["target_id"],
        fold=fold,
        status=pd.Series("BASELINE", index=ev.index),
        pred_lat=ev["shortest_ping_vp_lat"],
        pred_lon=ev["shortest_ping_vp_lon"],
        truth_seed_id=ev["target_id"].map(truth),
    )


def topn_summary(
    frames: dict[str, pd.DataFrame], *, ns: tuple[int, ...] = DEFAULT_TOPN
) -> pd.DataFrame:
    """Top-N accuracy per method, with and without fallbacks counted.

    `accuracy_topN` is over the full target set with non-SUCCESS rows counted as
    wrong — the paper §7.2 rule. `accuracy_topN_success_only` restricts the
    denominator to rows the pipeline actually solved, which is the number a
    variant's own geometry earned; the gap between the two is the fallback cost.
    """
    rows: list[dict] = []
    for method, df in frames.items():
        n_total = len(df)
        is_baseline = (df["status"] == "BASELINE").all() if n_total else False
        solved = (
            np.ones(n_total, dtype=bool)
            if is_baseline
            else df["status"].isin(io.CBG_SUCCESS_STATUSES).to_numpy()
        )
        rank = df["truth_seed_rank"].to_numpy()
        err = df["error_to_truth_seed_km"].to_numpy(dtype=float)
        row = {
            "method": method,
            "n_targets": n_total,
            "n_solved": int(solved.sum()),
            "n_fallback": int((df["status"] == "FALLBACK").sum()),
            "n_error": int((df["status"] == "ERROR").sum()),
            "fallback_rate": round(float((df["status"] == "FALLBACK").mean()), 4)
            if n_total
            else np.nan,
        }
        for n in ns:
            hit = (rank >= 0) & (rank < n)
            row[f"accuracy_top{n}"] = (
                round(float((hit & solved).mean()), 4) if n_total else np.nan
            )
            row[f"accuracy_top{n}_success_only"] = (
                round(float(hit[solved].mean()), 4) if solved.any() else np.nan
            )
        e_solved = err[solved]
        e_solved = e_solved[np.isfinite(e_solved)]
        for label, p in (("p50", 50), ("p90", 90)):
            row[f"error_km_{label}_success_only"] = (
                round(float(np.percentile(e_solved, p)), 3) if e_solved.size else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def score_run(
    run: RunPaths,
    space: AnswerSpace,
    *,
    combo_ids: list[str] | None = None,
    include_baseline: bool = True,
) -> dict[str, pd.DataFrame]:
    """Seed-distance frames for the baseline plus every requested combo."""
    frames: dict[str, pd.DataFrame] = {}
    if include_baseline:
        frames[SHORTEST_PING] = score_shortest_ping(run, space)
    for combo_id in combo_ids if combo_ids is not None else run.combo_ids:
        frames[combo_id] = score_combo(run, space, combo_id)
    return frames


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("classify")
    def classify_cmd(
        run_id: str = typer.Option(
            None, help="Run to score. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Score every run under --outputs-root."
        ),
        answer_space: Path = typer.Option(
            None,
            help="Answer-space dir (from build-answer-space). Defaults to this "
                 "run's target-answer-space/ under --analysis-root.",
        ),
        combo: list[str] = typer.Option(
            None, "--combo", help="Restrict to these combo ids (repeatable)."
        ),
        no_baseline: bool = typer.Option(
            False, "--no-baseline", help="Skip the Shortest-Ping baseline."
        ),
        topn: str = typer.Option(
            ",".join(str(n) for n in DEFAULT_TOPN),
            help="Comma-separated Ns for the summary. The per-target artifact "
                 "supports any N regardless of this.",
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Score predictions against an answer space; emit distance-to-all-seeds.

        Writes <method>_seed_distances.parquet + topn_accuracy.csv to
        <analysis_root>/<run_id>/target-cls-accuracy/.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and answer_space is not None:
            raise typer.BadParameter("--answer-space cannot be combined with --all-runs")
        ns = tuple(int(x) for x in topn.split(",") if x.strip())

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        for run in runs:
            space_dir = answer_space or run.answer_space_dir(root=analysis_root)
            space = load_answer_space(space_dir)
            frames = score_run(
                run,
                space,
                combo_ids=list(combo) if combo else None,
                include_baseline=not no_baseline,
            )
            out_dir = run.cls_accuracy_dir(root=analysis_root)
            for method, df in frames.items():
                df.to_parquet(out_dir / f"{method}_seed_distances.parquet", index=False)
            summary = topn_summary(frames, ns=ns)
            summary.to_csv(out_dir / TOPN_CSV, index=False)
            (out_dir / MANIFEST_JSON).write_text(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "answer_space": str(space_dir),
                        "n_seeds": space.n_seeds,
                        "nside": space.meta.get("grid", {}).get("nside"),
                        "methods": sorted(frames),
                        "topn_reported": list(ns),
                        "fallback_policy": (
                            "per-target parquet is neutral (FALLBACK rows carry "
                            "distances); topn_accuracy.csv counts fallbacks as "
                            "failures in accuracy_topN and excludes them in "
                            "accuracy_topN_success_only"
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            best = summary.sort_values("accuracy_top1", ascending=False).iloc[0]
            typer.echo(
                f"{run.run_id}: K={space.n_seeds} · {len(frames)} methods · "
                f"best top1={best['accuracy_top1']:.3f} ({best['method']}) -> {out_dir}"
            )
