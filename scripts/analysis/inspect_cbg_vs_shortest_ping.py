"""Find eval targets where CBG predicts worse than the shortest-ping baseline.

Baseline = pick the lowest-RTT VP for the target, predict its coord; error =
haversine to the target's true coord. This is what `model.geolocate` returns
on `status=FALLBACK` — so we filter the CBG side to `status=SUCCESS` to
isolate cases where CBG *succeeded* but still lost to the trivial baseline.

Outputs (one CLI invocation, two artifacts):
  - PNG: error-diff CDF, one curve per CBG combo (delta_km = error_CBG −
    error_baseline). Built by reusing `plot_error_diff_cdf.plot_error_diff_cdf`
    with a synthetic `shortest_ping` pseudo-combo.
  - CSV: per-combo percentile examples at p5, p25, p50, p75, p95 of the
    delta distribution. Forensic columns include fold (for trace-back to
    `anchor_fold_<N>.json`), n_obs, n_ltd_success, mtl_intersection_kind.

Join key is `<fold>/<target_id>` matching the convention at
`plot_error_diff_cdf.py:117-121`; `<fold>` comes from the parent dir name of
each `eval_observations.parquet` so K-fold merged mode preserves provenance.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.analysis._v2_io import (
    active_geo_filter,
    add_geo_filter_args,
    discover_combos,
    load_targets,
    route_geo_path,
    set_geo_filter_from_args,
)
from scripts.analysis.plot_error_cdf import (
    _load_ip_to_continent,
    _normalize_continent,
)
from scripts.analysis.plot_error_diff_cdf import plot_error_diff_cdf
from scripts.libs.cbg.rtt_model import haversine_distance

logger = logging.getLogger(__name__)

_BASELINE_NAME = "shortest_ping"
_PERCENTILES = (5, 25, 50, 75, 95)


def _load_nearest_ping_full(inputs_dir: Path) -> dict[str, dict]:
    """Per-target shortest-ping baseline keyed by `<fold>/<target_id>`.

    Mirrors the recipe in `plot_error_cdf._load_nearest_ping_baseline_by_target`
    but (a) keys by fold/target_id to align with the CBG side and (b) carries
    the nearest-VP id/coord in the row payload so the percentile CSV can
    surface it.
    """
    direct = inputs_dir / "eval_observations.parquet"
    if direct.exists():
        per_fold = [(inputs_dir.name, direct)]
    else:
        paths = sorted(inputs_dir.glob("*/eval_observations.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No eval_observations.parquet at {inputs_dir} or under "
                f"{inputs_dir}/*/. Pass --inputs-dir pointing at a fold "
                "input dir or at its parent (for merged-fold mode)."
            )
        per_fold = [(p.parent.name, p) for p in paths]

    frames = []
    for fold, p in per_fold:
        df = pq.read_table(p).to_pandas()
        df["fold"] = fold
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return {}

    idx = df.groupby(["fold", "target_id"])["latency_ms"].idxmin()
    nearest = df.loc[idx].reset_index(drop=True)
    err = haversine_distance(
        nearest["target_lat"].to_numpy(dtype=float),
        nearest["target_lon"].to_numpy(dtype=float),
        nearest["vp_lat"].to_numpy(dtype=float),
        nearest["vp_lon"].to_numpy(dtype=float),
    )

    out: dict[str, dict] = {}
    for i, row in nearest.iterrows():
        key = f"{row['fold']}/{row['target_id']}"
        out[key] = {
            "fold": str(row["fold"]),
            "target_id": str(row["target_id"]),
            "target_lat": float(row["target_lat"]),
            "target_lon": float(row["target_lon"]),
            "nearest_vp_id": str(row["vp_id"]),
            "nearest_vp_lat": float(row["vp_lat"]),
            "nearest_vp_lon": float(row["vp_lon"]),
            "nearest_latency_ms": float(row["latency_ms"]),
            "error_km": float(err[i]),
        }
    return out


def _load_cbg_targets_success_only(
    run_dir: Path,
    source: Optional[str],
    slice_: Optional[str],
    combos: Optional[list[str]] = None,
) -> dict[str, dict[str, dict]]:
    """Per-combo SUCCESS-only target rows keyed by `<fold>/<target_id>`.

    Returns `{combo_id: {key: row_dict}}`. FALLBACK / ERROR rows are dropped —
    FALLBACK is literally the shortest-ping prediction (delta ≡ 0) and ERROR
    has no coord to compare.
    """
    combo_dirs = discover_combos(run_dir, source, slice_, combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")

    out: dict[str, dict[str, dict]] = {}
    for combo_dir in combo_dirs:
        tbl = load_targets(combo_dir)
        cols = {
            c: tbl.column(c).to_pylist()
            for c in (
                "target_id", "target_lat", "target_lon", "n_obs",
                "status", "error_km", "n_ltd_success",
                "mtl_intersection_kind",
            )
        }
        fold = combo_dir.parent.name
        bucket = out.setdefault(combo_dir.name, {})
        for i in range(tbl.num_rows):
            if cols["status"][i] != "SUCCESS":
                continue
            err = cols["error_km"][i]
            if err is None or (isinstance(err, float) and np.isnan(err)):
                continue
            tid = cols["target_id"][i]
            bucket[f"{fold}/{tid}"] = {
                "fold": fold,
                "target_id": tid,
                "target_lat": float(cols["target_lat"][i]),
                "target_lon": float(cols["target_lon"][i]),
                "n_obs": int(cols["n_obs"][i]),
                "status": cols["status"][i],
                "error_km": float(err),
                "n_ltd_success": int(cols["n_ltd_success"][i]),
                "mtl_intersection_kind": cols["mtl_intersection_kind"][i],
            }
    return out


def _join_deltas(
    cbg: dict[str, dict[str, dict]],
    baseline: dict[str, dict],
) -> dict[str, list[dict]]:
    """Inner-join per combo on `<fold>/<target_id>`; compute delta_km."""
    out: dict[str, list[dict]] = {}
    for combo_id, rows in cbg.items():
        joined = []
        for key, cbg_row in rows.items():
            base_row = baseline.get(key)
            if base_row is None:
                continue
            joined.append({
                "combo_id": combo_id,
                "fold": cbg_row["fold"],
                "target_id": cbg_row["target_id"],
                "target_lat": cbg_row["target_lat"],
                "target_lon": cbg_row["target_lon"],
                "nearest_vp_id": base_row["nearest_vp_id"],
                "error_cbg_km": cbg_row["error_km"],
                "error_baseline_km": base_row["error_km"],
                "delta_km": cbg_row["error_km"] - base_row["error_km"],
                "n_obs": cbg_row["n_obs"],
                "n_ltd_success": cbg_row["n_ltd_success"],
                "mtl_intersection_kind": cbg_row["mtl_intersection_kind"],
                "status": cbg_row["status"],
            })
        out[combo_id] = joined
    return out


def pick_percentile_examples(
    joined: dict[str, list[dict]],
    percentiles: tuple[int, ...] = _PERCENTILES,
) -> list[dict]:
    """For each combo, pick one row per percentile of `delta_km`.

    Uses `round((n-1)·q/100)` over the delta-sorted records — the same row a
    quantile call with `method='nearest'` would land on, but explicit so the
    selected row is the one written.
    """
    rows = []
    for combo_id in sorted(joined):
        records = joined[combo_id]
        if not records:
            logger.warning("%s: 0 joined targets — skipping", combo_id)
            continue
        sorted_records = sorted(records, key=lambda r: r["delta_km"])
        n = len(sorted_records)
        for q in percentiles:
            i = max(0, min(n - 1, round((n - 1) * q / 100.0)))
            r = dict(sorted_records[i])
            r["percentile"] = q
            rows.append(r)
    return rows


def _partition_by_continent(
    joined: dict[str, list[dict]],
    ip_to_continent: dict[str, str],
    target_continent_canon: str,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], set[str]]:
    """Split each combo's joined records by whether the target sits in
    `target_continent_canon`. Anchors with unknown continent are dropped
    from both groups; their target_ids are returned for logging.
    """
    in_grp: dict[str, list[dict]] = {}
    rest_grp: dict[str, list[dict]] = {}
    unknown_ids: set[str] = set()
    for combo_id, records in joined.items():
        in_grp[combo_id] = []
        rest_grp[combo_id] = []
        for r in records:
            cont = ip_to_continent.get(r["target_id"], "Unknown")
            if cont == "Unknown":
                unknown_ids.add(r["target_id"])
                continue
            if cont == target_continent_canon:
                in_grp[combo_id].append(r)
            else:
                rest_grp[combo_id].append(r)
    return in_grp, rest_grp, unknown_ids


def _suffix_path(p: Path, suffix: str) -> Path:
    """`/dir/foo.png` + `north_america` -> `/dir/foo_north_america.png`."""
    return p.with_name(f"{p.stem}_{suffix}{p.suffix}")


def write_percentile_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "combo_id", "percentile", "fold", "target_id",
        "target_lat", "target_lon", "nearest_vp_id",
        "error_cbg_km", "error_baseline_km", "delta_km",
        "n_obs", "n_ltd_success", "mtl_intersection_kind",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    logger.info("Saved percentile CSV: %s (%d rows)", path, len(rows))


def _emit(
    joined: dict[str, list[dict]],
    cbg_combo_ids: list[str],
    out_png: Path,
    out_csv: Path,
    title: str,
) -> None:
    """Diff CDF PNG + percentile CSV from one (possibly continent-filtered)
    joined dict. Synthesizes a `shortest_ping` pseudo-combo over the same
    record set so the inner-join in `plot_error_diff_cdf` lines up."""
    target_errors_by_combo: dict[str, dict[str, float]] = {}
    baseline_errors: dict[str, float] = {}
    for combo_id, records in joined.items():
        target_errors_by_combo[combo_id] = {
            f"{r['fold']}/{r['target_id']}": r["error_cbg_km"] for r in records
        }
        for r in records:
            baseline_errors[f"{r['fold']}/{r['target_id']}"] = r["error_baseline_km"]
    target_errors_by_combo[_BASELINE_NAME] = baseline_errors

    pairs = [(cid, _BASELINE_NAME) for cid in cbg_combo_ids]
    fig = plot_error_diff_cdf(target_errors_by_combo, pairs, out_png, title=title)
    plt.close(fig)

    write_percentile_csv(pick_percentile_examples(joined), out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CBG combos against the shortest-ping baseline. "
            "SUCCESS-only on the CBG side; FALLBACK is the baseline itself."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to outputs/<run_id>/")
    parser.add_argument("--source", default=None)
    parser.add_argument("--slice", dest="slice_", default=None,
                        help="Single slice; omit to concat across folds")
    parser.add_argument("--inputs-dir", type=Path, required=True,
                        help=("inputs/<source>/<run_id>/<setup>/<slice>/ for "
                              "single-slice mode, or the <setup>/ parent for "
                              "K-fold merged mode (globs fold_*/eval_observations.parquet)"))
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--split-by-main-continent",
        default=None,
        help=("When set, partition joined records by whether each target sits in "
              "this continent (e.g. 'north_america') and emit TWO PNG/CSV pairs "
              "derived from --out-png/--out-csv: <stem>_<slug>{.png,.csv} and "
              "<stem>_rest{.png,.csv}. Suppresses the unsplit outputs."),
    )
    parser.add_argument(
        "--filtered-anchors",
        type=Path,
        default=Path("datasets/ripe_atlas/filtered_anchors.json"),
        help=("Anchor metadata for target→continent lookup via address_v4 + "
              "continent_of(country_code). Used only with "
              "--split-by-main-continent."),
    )
    parser.add_argument(
        "--combos", nargs="*", default=None,
        help="Restrict to these combo_ids (default: every combo found on disk).",
    )
    add_geo_filter_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_geo_filter_from_args(args)
    if active_geo_filter() is not None and args.split_by_main_continent is not None:
        raise SystemExit(
            "--geo-level/--geo-value and --split-by-main-continent both slice "
            "by geography; use one or the other."
        )
    args.out_png = route_geo_path(args.out_png)
    args.out_csv = route_geo_path(args.out_csv)

    cbg = _load_cbg_targets_success_only(
        args.run_dir, args.source, args.slice_, combos=args.combos,
    )
    baseline = _load_nearest_ping_full(args.inputs_dir)

    cbg_combo_ids = sorted(cbg)
    if not cbg_combo_ids:
        raise SystemExit(f"No combos discovered under {args.run_dir}")
    logger.info(
        "Loaded %d combos × baseline | combos=%s",
        len(cbg_combo_ids), cbg_combo_ids,
    )

    joined = _join_deltas(cbg, baseline)
    base_title = args.title or f"CBG vs shortest_ping ({args.run_dir.name}, SUCCESS-only)"

    if args.split_by_main_continent is None:
        _emit(joined, cbg_combo_ids, args.out_png, args.out_csv, base_title)
        return

    canon = _normalize_continent(args.split_by_main_continent)
    ip_to_continent = _load_ip_to_continent(args.filtered_anchors)
    in_grp, rest_grp, unknown_ids = _partition_by_continent(
        joined, ip_to_continent, canon,
    )
    if unknown_ids:
        logger.warning(
            "%d anchor IPs missing from %s or with unknown country_code; "
            "dropped from both subsets",
            len(unknown_ids), args.filtered_anchors,
        )

    slug = canon.lower().replace(" ", "_")
    _emit(
        in_grp, cbg_combo_ids,
        _suffix_path(args.out_png, slug),
        _suffix_path(args.out_csv, slug),
        f"{base_title} — in {canon}",
    )
    _emit(
        rest_grp, cbg_combo_ids,
        _suffix_path(args.out_png, "rest"),
        _suffix_path(args.out_csv, "rest"),
        f"{base_title} — rest of world",
    )


if __name__ == "__main__":
    main()
