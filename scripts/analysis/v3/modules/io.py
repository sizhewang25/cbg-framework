"""IO library — read v2 benchmark outputs, merging K-fold shards into one frame.

The v2 runner writes one `targets.parquet` per (fold, combo). K-fold test sets
are disjoint by construction, so the union across folds is exactly one row per
eval target. `load_folds` performs that union and labels every row with its
`fold` index, which is the single entry point the rest of v3 uses.

Two schema traps this module exists to contain (see ../SCHEMA.md):

1. `error_km` is populated on `FALLBACK` rows, where it measures the *fallback*
   (shortest-ping VP) coordinate, not a CBG estimate. Pooling it across statuses
   silently floors CBG error at the baseline's — the comparison paper §7.2
   forbids. Every frame returned here therefore carries `status` beside it, and
   this module offers no status-blind error helper.
2. `(ltd, mtl, ctr)` does not identify a combo — `octant_cbg_hull` and
   `octant_cbg_spl` share a triple and differ only in `ltd_kwargs.fit_spline`.
   `load_run_configs` returns the kwargs so callers can tell them apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.analysis.v3.modules.paths import MissingArtifactError, RunPaths

#: Statuses whose `pred_lat`/`pred_lon` came from the CBG pipeline itself.
#: `FALLBACK` rows carry a coordinate too (the shortest-ping VP), but it is not
#: a CBG answer — paper §7.2 counts it as a failure.
CBG_SUCCESS_STATUSES = frozenset({"SUCCESS"})

#: Columns every caller needs; nested list-of-struct columns are excluded by
#: default because they are large and only the phase-forensics work reads them.
_NESTED_COLUMNS = ("ltd_predictions", "mtl_participants")


def fold_index(fold_id: str) -> int:
    """`"fold_3"` -> `3`."""
    return int(fold_id.split("_")[1])


def load_folds(
    run: RunPaths,
    combo_id: str,
    *,
    columns: list[str] | None = None,
    include_nested: bool = False,
) -> pd.DataFrame:
    """Every fold's `targets.parquet` for one combo, merged and fold-labelled.

    Returns one row per eval target with a `fold` column (int) and a `combo_id`
    column. Column order is `combo_id, fold, <parquet columns>`.

    `include_nested` pulls in `ltd_predictions` / `mtl_participants`; they are
    dropped by default since they dominate the file size and only the
    phase-attribution analyses need them. `columns` (if given) is passed to the
    parquet reader and takes precedence.

    Raises if the fold test sets overlap, which would mean the K-fold protocol
    leaked and any pooled accuracy figure would double-count targets.
    """
    folds = run.fold_ids
    if not folds:
        raise MissingArtifactError(f"no fold_* dirs under {run.setup_dir}")

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for fold_id in folds:
        path = run.combo_dir(combo_id, fold_id) / "targets.parquet"
        if not path.exists():
            missing.append(fold_id)
            continue
        read_cols = columns
        if read_cols is None and not include_nested:
            all_cols = pq.read_schema(path).names
            read_cols = [c for c in all_cols if c not in _NESTED_COLUMNS]
        df = pq.read_table(path, columns=read_cols).to_pandas()
        df.insert(0, "fold", fold_index(fold_id))
        frames.append(df)

    if not frames:
        raise MissingArtifactError(
            f"combo {combo_id!r} has no targets.parquet in any fold under {run.setup_dir}"
        )
    if missing:
        raise MissingArtifactError(
            f"combo {combo_id!r} is missing targets.parquet for folds {missing}; "
            f"pooling the rest would silently drop those targets"
        )

    merged = pd.concat(frames, ignore_index=True)
    merged.insert(0, "combo_id", combo_id)

    dup = merged["target_id"].duplicated()
    if dup.any():
        offenders = merged.loc[dup, "target_id"].unique()[:5].tolist()
        raise ValueError(
            f"combo {combo_id!r}: {int(dup.sum())} target_id(s) appear in more than one "
            f"fold (e.g. {offenders}). K-fold test sets must be disjoint; pooling "
            f"them would double-count targets."
        )
    return merged


def load_all_folds(
    run: RunPaths,
    combo_ids: list[str] | None = None,
    *,
    columns: list[str] | None = None,
    include_nested: bool = False,
) -> pd.DataFrame:
    """`load_folds` over several combos, stacked long on `combo_id`."""
    combos = combo_ids if combo_ids is not None else run.combo_ids
    frames = [
        load_folds(run, c, columns=columns, include_nested=include_nested) for c in combos
    ]
    if not frames:
        return pd.DataFrame(columns=["combo_id", "fold", "target_id"])
    return pd.concat(frames, ignore_index=True)


def load_run_configs(run: RunPaths, combo_ids: list[str] | None = None) -> pd.DataFrame:
    """One row per (combo, fold) from `run.json`: phase choice, kwargs, cost.

    Phase kwargs are kept as JSON strings so the frame stays flat and hashable —
    they are what distinguishes combos that share an `(ltd, mtl, ctr)` triple.
    """
    combos = combo_ids if combo_ids is not None else run.combo_ids
    rows: list[dict] = []
    for combo_id in combos:
        for fold_id in run.fold_ids:
            path = run.combo_dir(combo_id, fold_id) / "run.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            counts = d.get("status_counts") or {}
            rows.append(
                {
                    "combo_id": combo_id,
                    "fold": fold_index(fold_id),
                    "ltd": d.get("ltd"),
                    "mtl": d.get("mtl"),
                    "ctr": d.get("ctr"),
                    "ltd_kwargs": json.dumps(d.get("ltd_kwargs") or {}, sort_keys=True),
                    "mtl_kwargs": json.dumps(d.get("mtl_kwargs") or {}, sort_keys=True),
                    "ctr_kwargs": json.dumps(d.get("ctr_kwargs") or {}, sort_keys=True),
                    "enable_fallback": d.get("enable_fallback"),
                    "pair_weight_min": d.get("pair_weight_min"),
                    "n_fit_samples": d.get("n_fit_samples"),
                    "n_targets": d.get("n_targets"),
                    "n_success": counts.get("SUCCESS"),
                    "n_fallback": counts.get("FALLBACK"),
                    "n_error": counts.get("ERROR"),
                    "fit_ms": d.get("fit_ms"),
                    "fit_alloc_peak_bytes": d.get("fit_alloc_peak_bytes"),
                    "fit_rss_peak_bytes": d.get("fit_rss_peak_bytes"),
                    # Both RSS numbers are absolute `getrusage` peaks, so only
                    # their difference is attributable to the run — the baseline
                    # is ~174 MB of interpreter and imports.
                    "run_baseline_rss_bytes": d.get("run_baseline_rss_bytes"),
                    "run_peak_rss_bytes": d.get("run_peak_rss_bytes"),
                }
            )
    return pd.DataFrame(rows)


def load_summary(run: RunPaths) -> pd.DataFrame:
    """`summary.parquet` — per (setup, slice, combo) accuracy and cost rollup.

    Note its `error_km_*` percentiles pool FALLBACK rows; recompute from
    `load_folds` when a fallback-excluded figure is needed.
    """
    if not run.summary_path.exists():
        raise MissingArtifactError(f"{run.summary_path} missing; run `cli summarize`")
    df = pq.read_table(run.summary_path).to_pandas()
    if "slice" in df.columns:
        df["fold"] = df["slice"].map(
            lambda s: fold_index(s) if isinstance(s, str) and s.startswith("fold_") else pd.NA
        )
    return df


def load_targets(run: RunPaths) -> pd.DataFrame:
    """`targets.csv` — the run's full ground-truth target set (fold-independent).

    This is the population the answer space is built from: K-fold splits
    targets, so the union over folds is this file.
    """
    if not run.targets_csv.exists():
        raise MissingArtifactError(f"{run.targets_csv} missing")
    return pd.read_csv(run.targets_csv)


def load_vps(run: RunPaths) -> pd.DataFrame:
    """`vps.csv` — the run's vantage points (shared across every fold)."""
    if not run.vps_csv.exists():
        raise MissingArtifactError(f"{run.vps_csv} missing")
    return pd.read_csv(run.vps_csv)


def load_eval_per_target(run: RunPaths) -> pd.DataFrame:
    """`eval_source/<basename>_eval_per_target.csv` — method-free target geometry.

    Carries the shortest-ping VP coordinate per target
    (`shortest_ping_vp_lat`/`_lon`) used to score the baseline, plus the §8.2
    proximity decomposition (`proximity_label`, `n_discriminative_vps`, ...).
    Fold-independent: every VP is available to every fold, so a target's
    shortest-ping VP does not depend on which fold held it out.
    """
    return pd.read_csv(run.eval_file("eval_per_target.csv"))


def load_eval_stats(run: RunPaths) -> dict:
    return json.loads(run.eval_file("eval_stats.json").read_text())


def load_dataset_stats(run: RunPaths) -> dict:
    """`eval_dataset/<basename>_dataset_stats.json` (never written to eval_source)."""
    return json.loads(
        run.eval_file("dataset_stats.json", prefer_source=False).read_text()
    )
