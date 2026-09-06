"""Score predictions against an answer space: distance to *every* seed (§8.1).

The answer space is an **explicit input** (`--answer-space`), never implied, so
the same predictions can be re-scored under a different quantization without
re-running anything upstream.

For each (method, target) this emits the distance from the estimated coordinate
to all K seeds. That is deliberately more than a label: because the full
distance vector is kept, the rank of the true seed falls out of it, and top-N
classification accuracy for *any* N is then `(tg_seed_rank < N).mean()` with
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

from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    get_grid,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.answer_space import (
    AnswerSpace,
    elementwise_km,
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
DEFAULT_TOPN = (1, 3)

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
    tg_seed_id: pd.Series,
) -> pd.DataFrame:
    """Distances from each prediction to all K seeds, plus the derived ranks.

    Rows whose prediction is missing (no coordinate at all) keep NaN distances
    and a null `pred_seed_id`; they are still emitted so the target set stays
    the complete denominator.

    Emits two error columns, which answer different questions and must not be
    used interchangeably:

    * `error_to_target_km` — to the raw ground-truth coordinate. **This is the
      error distance.** It owes nothing to the grid or the seeds.
    * `error_to_tg_seed_km` — to the true seed, i.e. to the centre of the
      cell the target falls in. A diagnostic of the *classification* geometry,
      not an error metric: it has a floor equal to that target's
      `cell_offset_km`, so a perfect prediction reports ~17 km on `h3-4` rather
      than 0 (§7.4).
    """
    seeds = space.seeds
    seed_ids = seeds["seed_id"].to_numpy()
    d = pairwise_km(
        pred_lat.to_numpy(dtype=float),
        pred_lon.to_numpy(dtype=float),
        seeds["seed_lat"].to_numpy(),
        seeds["seed_lon"].to_numpy(),
    )

    has_pred = np.isfinite(pred_lat.to_numpy(dtype=float)) & np.isfinite(
        pred_lon.to_numpy(dtype=float)
    )
    d = np.where(has_pred[:, None], d, np.nan)

    # Error distance is measured to the **raw target**, never to its seed. The
    # seed and the grid exist to form the tessellation, which serves
    # classification; error distance has a perfectly good ground truth of its
    # own, and routing it through the seed would import the quantization offset
    # into a measurement that does not need one. The two metrics look at the
    # same prediction from different angles and do not share an answer space.
    #
    # Computed here from `space.assignments` rather than taken from the
    # upstream `error_km` for two reasons: `score_shortest_ping` synthesizes
    # predictions from a VP coordinate and has no such column, and `error_km`
    # carries the FALLBACK trap documented in SCHEMA.md §3.
    tgt = space.assignments.set_index("target_id")
    unknown = ~target_id.isin(tgt.index)
    if unknown.any():
        missing = target_id[unknown].unique()[:5].tolist()
        raise ValueError(
            f"{int(unknown.sum())} target_id(s) are absent from the answer space "
            f"(e.g. {missing}); cannot measure error to their true coordinate"
        )
    t_lat = target_id.map(tgt["target_lat"]).to_numpy(dtype=float)
    t_lon = target_id.map(tgt["target_lon"]).to_numpy(dtype=float)
    err_to_target = np.where(
        has_pred,
        elementwise_km(
            pred_lat.to_numpy(dtype=float), pred_lon.to_numpy(dtype=float), t_lat, t_lon
        ),
        np.nan,
    )

    tg_seed = tg_seed_id.to_numpy()
    # Column position of each target's true seed (seed_id is 0..K-1 by
    # construction, but map explicitly so a filtered answer space still works).
    pos_of_seed = {int(s): i for i, s in enumerate(seed_ids)}
    tg_pos = np.array([pos_of_seed.get(int(s), -1) for s in tg_seed])
    if (tg_pos < 0).any():
        bad = np.unique(tg_seed[tg_pos < 0])[:5]
        raise ValueError(
            f"tg seed id(s) {bad.tolist()} are absent from the answer space; "
            f"predictions and answer space disagree on the seed set"
        )

    rows = np.arange(len(d))
    err_to_tg_seed = d[rows, tg_pos]

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
            sub < err_to_tg_seed[has_pred][:, None] - 1e-9
        ).sum(axis=1)

    out = pd.DataFrame(
        {
            "method": method,
            "target_id": target_id.to_numpy(),
            "fold": fold.to_numpy(),
            "status": status.to_numpy(),
            "pred_lat": pred_lat.to_numpy(dtype=float),
            "pred_lon": pred_lon.to_numpy(dtype=float),
            "tg_seed_id": tg_seed,
            "pred_seed_id": np.where(pred_pos >= 0, seed_ids[pred_pos], -1),
            "tg_seed_rank": rank,
            "error_to_target_km": np.round(err_to_target, 3),
            "error_to_tg_seed_km": np.round(err_to_tg_seed, 3),
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
    tg_seed = space.assignments.set_index("target_id")["seed_id"]
    unknown = ~df["target_id"].isin(tg_seed.index)
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
        tg_seed_id=df["target_id"].map(tg_seed),
    )


def score_shortest_ping(run: RunPaths, space: AnswerSpace) -> pd.DataFrame:
    """Seed distances for the Shortest-Ping baseline.

    The estimate is the lowest-RTT VP's own coordinate, resolved by
    `io.load_sping_vp` — the single reader `proximity` shares, so this score and
    that module's `has_proximate_sping_vp` cannot drift apart. The choice is
    fold-independent because every VP is available in every fold; only targets
    are folded.

    `fold` is taken from any CBG combo's fold assignment so the baseline sits on
    the same fold partition as the variants; it is left as -1 if no combo exists.
    """
    ev = io.load_sping_vp(run)

    tg_seed = space.assignments.set_index("target_id")["seed_id"]
    ev = ev[ev["target_id"].isin(tg_seed.index)].reset_index(drop=True)

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
        pred_lat=ev["sping_vp_lat"],
        pred_lon=ev["sping_vp_lon"],
        tg_seed_id=ev["target_id"].map(tg_seed),
    )


def topn_summary(
    frames: dict[str, pd.DataFrame], *, ns: tuple[int, ...] = DEFAULT_TOPN
) -> pd.DataFrame:
    """Top-N accuracy and error distance per method.

    `accuracy_topN` is over the full target set with non-SUCCESS rows counted as
    wrong — the paper §7.2 rule. `fallback_rate` sits beside it, so the cost of
    those failures is readable without a second accuracy column.

    `error_km_p50` / `error_km_p90` are the distance to the **raw target**, not
    to its seed. Accuracy and error look at the same prediction from different
    angles and do not share an answer space: the seed exists to define the
    classes, while error distance has its own ground truth and would only
    inherit the grid's quantization by going through the seed.

    They are computed over **solved rows only**, despite the unqualified name. A
    FALLBACK row's coordinate is the Shortest-Ping VP's, so its error is the
    baseline's error rather than a CBG one; pooling it would corrupt the error
    distribution the same way crediting it would corrupt accuracy.
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
        rank = df["tg_seed_rank"].to_numpy()
        err = df["error_to_target_km"].to_numpy(dtype=float)
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
        e_solved = err[solved]
        e_solved = e_solved[np.isfinite(e_solved)]
        for label, p in (("p50", 50), ("p90", 90)):
            row[f"error_km_{label}"] = (
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
                 "run's target-answer-space/<grid>-<resolution>/ under "
                 "--analysis-root.",
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
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
        <analysis_root>/<run_id>/target-cls-accuracy/<grid>-<resolution>/, where
        both are read from the answer space itself rather than from --grid /
        --resolution, so an explicit --answer-space still lands in the directory
        matching the grid it was actually built on.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        if all_runs and answer_space is not None:
            raise typer.BadParameter("--answer-space cannot be combined with --all-runs")
        ns = tuple(int(x) for x in topn.split(",") if x.strip())
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)

        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]
        jobs = (
            [(r, None) for r in runs]
            if answer_space is not None
            else [(r, res) for r in runs for res in resolutions]
        )
        for run, want_res in jobs:
            space_dir = answer_space or run.answer_space_dir(
                root=analysis_root, grid=g.name, resolution=want_res
            )
            space = load_answer_space(space_dir)
            space_grid = str(space.seeds["grid_scheme"].iloc[0])
            space_res = int(space.seeds["grid_resolution"].iloc[0])
            frames = score_run(
                run,
                space,
                combo_ids=list(combo) if combo else None,
                include_baseline=not no_baseline,
            )
            out_dir = run.cls_accuracy_dir(
                root=analysis_root, grid=space_grid, resolution=space_res
            )
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
                        "grid": space_grid,
                        "resolution": space_res,
                        "methods": sorted(frames),
                        "topn_reported": list(ns),
                        "fallback_policy": (
                            "per-target parquet is neutral (FALLBACK rows carry "
                            "distances); topn_accuracy.csv counts fallbacks as "
                            "failures in accuracy_topN, and excludes them from "
                            "error_km_p50/p90"
                        ),
                        "error_metric": (
                            "error_km_p50/p90 measure distance to the raw target "
                            "(error_to_target_km), not to its seed: accuracy is "
                            "defined by the seeds, error distance is not"
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            best = summary.sort_values("accuracy_top1", ascending=False).iloc[0]
            typer.echo(
                f"{run.run_id}: {space_grid} "
                f"{get_grid(space_grid).resolution_arg}={space_res} · "
                f"K={space.n_seeds} · "
                f"{len(frames)} methods · best top1="
                f"{best['accuracy_top1']:.3f} ({best['method']}) -> {out_dir}"
            )
