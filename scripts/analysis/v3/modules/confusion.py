"""Where wrong answers land: local answer-space density, and how far off (§8.1).

`breakdown-accuracy` asks which targets a method gets right. This asks what its
mistakes look like — the question an operator actually has, because "wrong by
one cell" and "wrong by a continent" are the same zero in an accuracy column and
completely different operationally.

## Two questions, two tables

**Does crowding cause the mistakes?** `confusion_by_density.csv` bins targets by
their true seed's `nearest_seed_km` — how far the nearest competing class sits —
and reports accuracy and error percentiles per bin. A method that only fails
where classes are 30 km apart is losing to the *quantization*, not to geolocation
error, and the fix is a coarser grid rather than a better variant. A method that
fails uniformly across the density range is genuinely mislocating.

**How far off are the mistakes?** `confusion_pairs.csv` takes every wrong
prediction and asks where the predicted seed sits in the true seed's own
neighbour ranking, read off `seed_mesh_km.csv`. Rank 1 is the true seed's nearest
neighbour — a near miss that a top-3 answer would have caught. A rank in the
teens is a different region entirely.

## Three error columns that are not interchangeable

Each names a different pair of endpoints, and the module reports all three
because the gaps between them are the diagnostic:

* `error_to_target_km` — prediction to the raw ground truth. **The** error
  distance; owes nothing to the grid.
* `error_to_tg_seed_km` — prediction to the true seed. Floored by the target's
  own `cell_offset_km`, so a perfect answer reads ~17 km at `h3-4`.
* `error_to_pred_seed_km` — prediction to the seed it *chose*. Small means the
  estimate sits confidently inside the wrong cell; comparable to
  `error_to_tg_seed_km` means it landed near the boundary and the class flip was
  marginal.

That last comparison is what `boundary_margin_km` reports:
`error_to_tg_seed_km - error_to_pred_seed_km`, per wrong row. Near zero, the
answer was one tie-break away from correct.

## Bins are quantiles over targets, not over seeds

There are 18-27 seeds and 400-odd targets, and the targets are what the rates are
taken over. Quantile bins over targets keep every bin populated; fixed-width
bins on `nearest_seed_km` would put most of a run in one bucket, since seed
spacing at `h3-4` is dominated by the grid pitch. Boundaries are recorded in the
manifest so a bin can be read back to kilometres.

Command: `confusion-density`. Writes to
`outputs/analysis/v3/<run_id>/target-cls-accuracy/<grid>-<resolution>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules.bipartite import quantile_bins
from scripts.analysis.v3.modules import io
from scripts.analysis.v3.modules.answer_space import (
    load_answer_space,
    pairwise_km,
    seed_crossing_matrix,
)
from scripts.analysis.v3.modules.classify import DEFAULT_TOPN
from scripts.analysis.v3.modules.diagram.common.labels import label_for
from scripts.analysis.v3.modules.diagram.common.membership import available_methods
from scripts.analysis.v3.modules.grid import (
    DEFAULT_GRID,
    GRID_HELP,
    RESOLUTION_HELP,
    SWEEP_HELP,
    resolve_cli_grid,
)
from scripts.analysis.v3.modules.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUTS_ROOT,
    MissingArtifactError,
    RunPaths,
    discover_runs,
    resolve_run,
)

BY_DENSITY_CSV = "confusion_by_density.csv"
PAIRS_CSV = "confusion_pairs.csv"
MANIFEST_JSON = "confusion_manifest.json"

#: Tertiles: dense / medium / sparse. Enough to show a monotone trend without
#: splitting 400 targets into bins too thin to read a rate off.
DEFAULT_N_BINS = 3

_ERROR_COLUMNS = ("error_to_target_km", "error_to_tg_seed_km", "error_to_pred_seed_km")


def density_bins(
    nearest_seed_km: pd.Series, *, n_bins: int = DEFAULT_N_BINS
) -> tuple[pd.Series, list[float]]:
    """Quantile-bin targets by their true seed's distance to the nearest other seed.

    The binning itself is `bipartite.quantile_bins`, which this delegates to now
    that three modules stratify on a distance quantile. Kept as a named wrapper
    because the column it produces is `density_bin` and the docstring above is
    what §8.1's confusion table means by density; the generic helper documents
    the tie-collapse and two-edge guarantees the callers rely on.
    """
    idx, edges = quantile_bins(nearest_seed_km, n_bins=n_bins)
    return idx.rename("density_bin"), edges


def _percentiles(values: np.ndarray, prefix: str) -> dict:
    v = values[np.isfinite(values)]
    return {
        f"{prefix}_p50": round(float(np.percentile(v, 50)), 3) if v.size else np.nan,
        f"{prefix}_p90": round(float(np.percentile(v, 90)), 3) if v.size else np.nan,
    }


def load_scored(cls_dir: Path, method: str) -> pd.DataFrame:
    """One method's per-target scoring rows, distance-to-all-seeds columns dropped.

    The wide `dist_km__seed_*` block is what makes the parquet large and none of
    it is needed here — the true seed's rank and the three error columns already
    carry the answer.
    """
    path = Path(cls_dir) / f"{method}_seed_distances.parquet"
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify` first")
    return pd.read_parquet(
        path,
        columns=[
            "target_id",
            "status",
            "tg_seed_id",
            "pred_seed_id",
            "tg_seed_rank",
            *_ERROR_COLUMNS,
        ],
    )


def confusion_by_density(
    scored: dict[str, pd.DataFrame],
    seeds: pd.DataFrame,
    *,
    ns: tuple[int, ...] = DEFAULT_TOPN,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[pd.DataFrame, list[float]]:
    """Accuracy and error percentiles per (method, density bin, N).

    Bins are assigned from the **true** seed, not the predicted one: the question
    is whether a crowded neighbourhood caused the mistake, and the predicted seed
    is downstream of the mistake rather than upstream of it.
    """
    by_seed = seeds.set_index("seed_id")["nearest_seed_km"]
    edges: list[float] = []
    rows: list[dict] = []
    for method, df in scored.items():
        near = df["tg_seed_id"].map(by_seed)
        bins, edges = density_bins(near, n_bins=n_bins)
        solved = (df["status"] == "BASELINE") | df["status"].isin(
            io.CBG_SUCCESS_STATUSES
        )
        for b in sorted(bins.unique()):
            m = (bins == b).to_numpy()
            sub, sub_solved = df.loc[m], solved.to_numpy()[m]
            row = {
                "method": method,
                "method_label": label_for(method),
                "density_bin": int(b),
                "nearest_seed_km_lo": round(float(edges[int(b)]), 3),
                "nearest_seed_km_hi": round(float(edges[int(b) + 1]), 3),
                "n_targets": int(m.sum()),
                "nearest_seed_km_p50": round(float(np.nanmedian(near.to_numpy()[m])), 3)
                if m.any()
                else np.nan,
            }
            rank = sub["tg_seed_rank"].to_numpy()
            for n in ns:
                hit = (rank >= 0) & (rank < n) & sub_solved
                row[f"accuracy_top{n}"] = (
                    round(float(hit.mean()), 4) if hit.size else np.nan
                )
            for col in _ERROR_COLUMNS:
                row.update(
                    _percentiles(
                        sub.loc[sub_solved, col].to_numpy(dtype=float), col[: -len("_km")]
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows), edges


def confusion_pairs(
    scored: dict[str, pd.DataFrame],
    seeds: pd.DataFrame,
    seed_mesh_km: pd.DataFrame,
    *,
    top_n: int = 1,
    crossings: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per wrong prediction: which class it chose, and how far off that was.

    "Wrong" is `tg_seed_rank >= top_n` **or** an unsolved status, so a FALLBACK
    row appears here as the failure §7.2 counts it as — with its predicted seed
    still recorded, because a coordinate does exist and where it landed is the
    information this table is for.

    Two columns say how far off the class was, and they are not the same
    question. `seeds_crossed` counts the **class boundaries** between the true
    seed and the predicted one (`seed_crossing_matrix`); `pred_seed_neighbour_rank`
    is the predicted seed's position in the true seed's **distance** ordering,
    read off `seed_mesh_km.csv` rather than recomputed so it cannot disagree with
    the answer space's own geometry.

    `seeds_crossed == 1` is the one to read as "a neighbouring cell", with the
    caveat above that it means *on the direct path*. Distance rank 1 implies it
    but is much narrower: measured at `h3-4`, 27-55% of wrong top-1 rows are at
    rank 1 while 59-75% are one boundary away. Reporting rank alone understates
    near-misses by up to a factor of three, because a seed can be the
    fifth-nearest and still share a boundary.
    """
    if seed_mesh_km.empty:
        raise MissingArtifactError(
            "the answer space has no seed_mesh_km.csv; rebuild with "
            "`build-answer-space` so neighbour ranks can be read off it"
        )
    mesh = seed_mesh_km.copy()
    mesh.columns = [int(c) for c in mesh.columns]
    # Rank each row's other seeds by distance; the diagonal (self, 0 km) is rank
    # 0, so a true neighbour starts at 1 and the numbers read as "nth nearest".
    order = mesh.rank(axis=1, method="min").astype(int) - 1
    near_by_seed = seeds.set_index("seed_id")["nearest_seed_km"]
    if crossings is None:
        crossings = seed_crossing_matrix(seeds)
    pos = {int(s): i for i, s in enumerate(seeds["seed_id"].to_numpy())}

    frames = []
    for method, df in scored.items():
        solved = (df["status"] == "BASELINE") | df["status"].isin(
            io.CBG_SUCCESS_STATUSES
        )
        wrong = ~((df["tg_seed_rank"] >= 0) & (df["tg_seed_rank"] < top_n) & solved)
        sub = df.loc[wrong.to_numpy()].copy()
        if sub.empty:
            continue
        has_pred = sub["pred_seed_id"] >= 0
        pairs = list(zip(sub["tg_seed_id"], sub["pred_seed_id"]))
        sub["method"] = method
        sub["method_label"] = label_for(method)
        sub["top_n"] = top_n
        sub["tg_seed_nearest_seed_km"] = sub["tg_seed_id"].map(near_by_seed).round(3)
        sub["tg_to_pred_seed_km"] = [
            round(float(mesh.at[t, p]), 3) if p >= 0 else np.nan for t, p in pairs
        ]
        sub["pred_seed_neighbour_rank"] = [
            int(order.at[t, p]) if p >= 0 else -1 for t, p in pairs
        ]
        sub["seeds_crossed"] = [
            int(crossings[pos[int(t)], pos[int(p)]]) if p >= 0 else -1
            for t, p in pairs
        ]
        # Near zero means the estimate sat almost equidistant from both seeds:
        # the class flip was a tie-break, not a mislocation.
        sub["boundary_margin_km"] = (
            sub["error_to_tg_seed_km"] - sub["error_to_pred_seed_km"]
        ).round(3)
        sub.loc[~has_pred, "boundary_margin_km"] = np.nan
        frames.append(sub)
    if not frames:
        return pd.DataFrame(
            columns=[
                "method", "method_label", "top_n", "target_id", "status",
                "tg_seed_id", "pred_seed_id", "tg_seed_rank",
                "tg_seed_nearest_seed_km", "tg_to_pred_seed_km", "seeds_crossed",
                "pred_seed_neighbour_rank", "boundary_margin_km", *_ERROR_COLUMNS,
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[
        [
            "method", "method_label", "top_n", "target_id", "status",
            "tg_seed_id", "pred_seed_id", "tg_seed_rank",
            "tg_seed_nearest_seed_km", "tg_to_pred_seed_km", "seeds_crossed",
            "pred_seed_neighbour_rank", "boundary_margin_km", *_ERROR_COLUMNS,
        ]
    ]


def build_for_run(
    run: RunPaths,
    *,
    analysis_root: Path | None = None,
    grid: str = DEFAULT_GRID,
    resolution: int | None = None,
    methods: list[str] | None = None,
    ns: tuple[int, ...] = DEFAULT_TOPN,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    cls_dir = run.cls_accuracy_dir(root=analysis_root, grid=grid, resolution=resolution)
    space = load_answer_space(
        run.answer_space_dir(root=analysis_root, grid=grid, resolution=resolution)
    )
    chosen = methods or available_methods(cls_dir)
    scored = {m: load_scored(cls_dir, m) for m in chosen}

    by_density, edges = confusion_by_density(
        scored, space.seeds, ns=ns, n_bins=n_bins
    )
    # Computed once and passed down: it is a property of the answer space, not
    # of a method or an N, and the K x K walk would otherwise be repeated per N.
    crossings = seed_crossing_matrix(space.seeds)
    pairs = pd.concat(
        [
            confusion_pairs(
                scored, space.seeds, space.seed_mesh_km, top_n=n, crossings=crossings
            )
            for n in ns
        ],
        ignore_index=True,
    )
    off_diag = crossings[~np.eye(len(space.seeds), dtype=bool)] if space.n_seeds > 1 else np.array([])
    manifest = {
        "run_id": run.run_id,
        "grid": {
            "scheme": str(space.seeds["grid_scheme"].iloc[0]),
            "resolution": int(space.seeds["grid_resolution"].iloc[0]),
        },
        "n_seeds": space.n_seeds,
        "methods": chosen,
        "topn_reported": list(ns),
        "density_bin_edges_km": [round(float(e), 3) for e in edges],
        "class_adjacency": {
            "mean_degree": round(float((crossings == 1).sum(axis=1).mean()), 3)
            if space.n_seeds
            else None,
            "max_seeds_crossed": int(off_diag.max()) if off_diag.size else 0,
            "agrees_with_seeds_csv": "class_adjacency_degree",
            "basis": (
                "seeds_crossed counts the Voronoi class boundaries on the direct "
                "geodesic between two seeds. seeds_crossed == 1 is this layer's "
                "class adjacency and matches seeds.csv's class_adjacency_degree, "
                "which answer_space computes from the same walk. Stricter than "
                "sharing a Voronoi edge: pairs whose cells touch only far from "
                "both seeds are excluded, and those are the far pairs (median "
                "separation 1410 km vs 728 km on as01), which no small "
                "coordinate error could confuse."
            ),
        },
        "density_bin_basis": (
            "quantiles of the TRUE seed's nearest_seed_km, taken over targets. "
            "Fixed-width bins would put most of a run in one bucket, since seed "
            "spacing is dominated by the grid pitch."
        ),
        "distance_vs_adjacency": (
            "pred_seed_neighbour_rank orders seeds by DISTANCE; seeds_crossed "
            "counts BOUNDARIES. rank 1 implies seeds_crossed 1, not the "
            "reverse: a seed can be fifth-nearest and still be one boundary "
            "away. Read seeds_crossed == 1 as 'a neighbouring class'."
        ),
        "error_columns": {
            "error_to_target_km": "prediction to the raw ground truth -- THE error distance",
            "error_to_tg_seed_km": "prediction to the true seed; floored by cell_offset_km",
            "error_to_pred_seed_km": "prediction to the seed it chose",
            "boundary_margin_km": (
                "error_to_tg_seed_km - error_to_pred_seed_km on wrong rows; near "
                "zero means the class flip was a tie-break, not a mislocation"
            ),
        },
        "wrong_row_policy": (
            "a row is wrong when tg_seed_rank >= top_n OR the status is not "
            "solved, so FALLBACK rows appear here as the failures §7.2 counts "
            "them as, with their predicted seed still recorded"
        ),
    }
    return by_density, pairs, manifest, cls_dir


# ---- CLI --------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command("confusion-density")
    def confusion_cmd(
        run_id: str = typer.Option(
            None, help="Run to analyse. Omit with --all-runs to do every run."
        ),
        all_runs: bool = typer.Option(
            False, "--all-runs", help="Analyse every run under --outputs-root."
        ),
        grid: str = typer.Option(DEFAULT_GRID, "--grid", help=GRID_HELP),
        resolution: list[int] = typer.Option(
            [], "--resolution", "-r", help=RESOLUTION_HELP
        ),
        sweep: bool = typer.Option(False, "--sweep", help=SWEEP_HELP),
        method: list[str] = typer.Option(
            None, "--method", help="Restrict to these method ids (repeatable)."
        ),
        topn: str = typer.Option(
            ",".join(str(n) for n in DEFAULT_TOPN),
            help="Comma-separated Ns; each gets its own set of wrong rows.",
        ),
        n_bins: int = typer.Option(
            DEFAULT_N_BINS, "--n-bins", help="Density quantile bins over targets."
        ),
        outputs_root: Path = typer.Option(
            DEFAULT_OUTPUTS_ROOT, help="Root holding <run_id>/ benchmark outputs."
        ),
        analysis_root: Path = typer.Option(
            DEFAULT_ANALYSIS_ROOT, help="Root for v3 analysis outputs."
        ),
    ) -> None:
        """Misclassification against answer-space density, and how far off it was.

        Writes confusion_by_density.csv + confusion_pairs.csv into
        target-cls-accuracy/<grid>-<resolution>/. Needs `classify` to have run
        on the same quantization.
        """
        if all_runs == (run_id is not None):
            raise typer.BadParameter("pass exactly one of --run-id or --all-runs")
        ns = tuple(int(x) for x in topn.split(",") if x.strip())
        g, resolutions = resolve_cli_grid(grid, resolution, sweep=sweep)
        runs = discover_runs(outputs_root) if all_runs else [resolve_run(run_id, outputs_root)]

        for run in runs:
            for res in resolutions:
                by_density, pairs, manifest, out_dir = build_for_run(
                    run,
                    analysis_root=analysis_root,
                    grid=g.name,
                    resolution=res,
                    methods=list(method) if method else None,
                    ns=ns,
                    n_bins=n_bins,
                )
                by_density.to_csv(out_dir / BY_DENSITY_CSV, index=False)
                pairs.to_csv(out_dir / PAIRS_CSV, index=False)
                (out_dir / MANIFEST_JSON).write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                first = pairs[pairs.top_n == ns[0]]
                adjacent = (
                    float((first["seeds_crossed"] == 1).mean())
                    if len(first)
                    else float("nan")
                )
                typer.echo(
                    f"{run.run_id}: {len(manifest['methods'])} methods · "
                    f"{len(by_density)} density rows · {len(first):,} wrong at "
                    f"top{ns[0]} ({adjacent:.1%} one class boundary away) "
                    f"-> {out_dir}"
                )
