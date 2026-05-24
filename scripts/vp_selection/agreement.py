"""Stage d.2 — agreement + detection sweep.

Consumes outputs from earlier stages:
  - `selections/{strategy}_{seed}.parquet`  (Stage d.1 — one per strategy×seed)
  - `claims.parquet`                        (Stage b — target → (claim, is_real))
  - `border_distances.parquet`              (Stage c — (vp, country) → km)
  - `speed_calibration.json`                (Stage a — calibrated S)

Computes two metrics:
  1. **Agreement vs full pool** (Cho 2024): % of (target, claim) pairs where
     `sub_verdict == full_pool_verdict`. Curve per (strategy, k).
  2. **Detection vs ground truth**: TPR (% of fakes correctly rejected by
     subset) and FPR (% of reals wrongly rejected). Curve per (strategy, k).

Subcommands:
  - `--sweep`: run the full evaluation, write `agreement_rows.parquet` +
    `agreement_summary.json`
  - `--plot`: render `agreement_curve.png` (3 panels) from the summary
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

from scripts.benchmark.v2.sources.base import DataSource
from scripts.benchmark.v2.sources.ripe_atlas import RipeAtlasSource
from scripts.vp_selection.iclab_verifier import iclab_verify

logger = logging.getLogger(__name__)


# ---- pure compute helpers -------------------------------------------------


def sub_verdict_at_k(first_violator_k: Optional[int], k: int) -> str:
    """Derive the subset verdict from a sequence-strategy first-violator scan."""
    if first_violator_k is None:
        return "accept"
    return "reject" if k >= first_violator_k else "accept"


def find_first_violator(
    selection: list[str],
    tgt_rtts: Mapping[str, float],
    claim: str,
    distances: Mapping[tuple[str, str], float],
    speed_limit_km_per_ms: float,
) -> Optional[int]:
    """Position (1-indexed) of the first landmark in `selection` whose
    implied propagation speed exceeds `speed_limit_km_per_ms` for this
    (target, claim). None if no landmark in the entire sequence violates."""
    for k, lm_id in enumerate(selection, start=1):
        rtt = tgt_rtts.get(lm_id)
        if rtt is None or rtt <= 0:
            continue
        d = distances.get((lm_id, claim))
        if d is None:
            continue
        owtt = rtt / 2.0
        if d / owtt > speed_limit_km_per_ms:
            return k
    return None


def compute_sequence_rows(
    selection_sequences: Mapping[tuple[str, int], list[str]],
    full_landmark_rtts_by_target: Mapping[str, Mapping[str, float]],
    targets_and_claims: list[tuple[str, str, bool]],
    distances: Mapping[tuple[str, str], float],
    speed_limit_km_per_ms: float,
    full_verdicts: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """One row per (strategy, seed, target, claim) with `first_violator_k`."""
    rows: list[dict[str, Any]] = []
    for (strategy, seed), selection in selection_sequences.items():
        sel_len = len(selection)
        for target_id, claim, is_real in targets_and_claims:
            tgt_rtts = full_landmark_rtts_by_target.get(target_id, {})
            fvk = find_first_violator(
                selection, tgt_rtts, claim, distances, speed_limit_km_per_ms,
            )
            rows.append({
                "strategy": strategy,
                "seed": seed,
                "target_id": target_id,
                "claimed_country": claim,
                "is_real": is_real,
                "full_verdict": full_verdicts[(target_id, claim)],
                "first_violator_k": fvk,
                "selection_length": sel_len,
                "k": None,
                "sub_verdict": None,
            })
    return rows


def compute_sampling_rows(
    selection_subsets: Mapping[tuple[str, int, int], list[str]],
    full_landmark_rtts_by_target: Mapping[str, Mapping[str, float]],
    targets_and_claims: list[tuple[str, str, bool]],
    distances: Mapping[tuple[str, str], float],
    speed_limit_km_per_ms: float,
    full_verdicts: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """One row per (strategy, seed, k, target, claim) with sub_verdict."""
    rows: list[dict[str, Any]] = []
    for (strategy, seed, k), subset_list in selection_subsets.items():
        subset = set(subset_list)
        for target_id, claim, is_real in targets_and_claims:
            tgt_rtts = full_landmark_rtts_by_target.get(target_id, {})
            sub_rtts = {lm: rtt for lm, rtt in tgt_rtts.items() if lm in subset}
            sub_v = iclab_verify(
                sub_rtts, claim, distances, speed_limit_km_per_ms,
            )
            rows.append({
                "strategy": strategy,
                "seed": seed,
                "target_id": target_id,
                "claimed_country": claim,
                "is_real": is_real,
                "full_verdict": full_verdicts[(target_id, claim)],
                "first_violator_k": None,
                "selection_length": None,
                "k": k,
                "sub_verdict": sub_v,
            })
    return rows


def agreement_curve(
    rows: list[dict[str, Any]],
    k_grid: list[int],
) -> dict[tuple[str, int, bool], dict[str, Any]]:
    """Compute {(strategy, k, is_real): {n, agree, rate}} across all rows.

    Sequence rows (k=None, has first_violator_k): derive sub_verdict at every
    K in k_grid, contribute one count per K.
    Sampling rows (k=<int>, has sub_verdict): contribute one count at that K.
    """
    bucket: dict[tuple[str, int, bool], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "agree": 0}
    )
    for r in rows:
        is_real = r["is_real"]
        strategy = r["strategy"]
        full = r["full_verdict"]
        if r["k"] is None:
            # sequence row → derive at every K
            for k in k_grid:
                sub = sub_verdict_at_k(r["first_violator_k"], k)
                key = (strategy, k, is_real)
                bucket[key]["n"] += 1
                if sub == full:
                    bucket[key]["agree"] += 1
        else:
            key = (strategy, r["k"], is_real)
            bucket[key]["n"] += 1
            if r["sub_verdict"] == full:
                bucket[key]["agree"] += 1
    out: dict = {}
    for key, b in bucket.items():
        rate = b["agree"] / b["n"] if b["n"] else None
        out[key] = {"n": b["n"], "agree": b["agree"], "rate": rate}
    return out


def detection_curve(
    rows: list[dict[str, Any]],
    k_grid: list[int],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Compute {(strategy, k): {tpr, fpr, n_fake, n_real}} using is_real
    labels. sub_verdict==reject on a fake = TP; sub_verdict==reject on a
    real = FP."""
    fake_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"n_fake": 0, "tp": 0, "n_real": 0, "fp": 0}
    )
    for r in rows:
        strategy = r["strategy"]
        is_real = r["is_real"]
        if r["k"] is None:
            # sequence row → derive at every K
            for k in k_grid:
                sub = sub_verdict_at_k(r["first_violator_k"], k)
                key = (strategy, k)
                if is_real:
                    fake_counts[key]["n_real"] += 1
                    if sub == "reject":
                        fake_counts[key]["fp"] += 1
                else:
                    fake_counts[key]["n_fake"] += 1
                    if sub == "reject":
                        fake_counts[key]["tp"] += 1
        else:
            key = (strategy, r["k"])
            if is_real:
                fake_counts[key]["n_real"] += 1
                if r["sub_verdict"] == "reject":
                    fake_counts[key]["fp"] += 1
            else:
                fake_counts[key]["n_fake"] += 1
                if r["sub_verdict"] == "reject":
                    fake_counts[key]["tp"] += 1
    out: dict = {}
    for key, c in fake_counts.items():
        tpr = c["tp"] / c["n_fake"] if c["n_fake"] else None
        fpr = c["fp"] / c["n_real"] if c["n_real"] else None
        out[key] = {
            "tpr": tpr, "fpr": fpr,
            "n_fake": c["n_fake"], "n_real": c["n_real"],
            "tp": c["tp"], "fp": c["fp"],
        }
    return out


# ---- I/O helpers ----------------------------------------------------------


def load_speed_limit(path: Path) -> float:
    with open(path) as f:
        data = json.load(f)
    return float(data["summary"]["S_one_way_km_per_ms"])


def load_claims(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except ImportError:
        import csv
        with open(path.with_suffix(".csv")) as f:
            return [{**r, "is_real": r["is_real"] in ("True", "true")}
                    for r in csv.DictReader(f)]


def load_border_distances(path: Path) -> dict[tuple[str, str], float]:
    try:
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    except ImportError:
        import csv
        with open(path.with_suffix(".csv")) as f:
            rows = [{"vp_id": r["vp_id"], "country": r["country"],
                     "distance_km": float(r["distance_km"])}
                    for r in csv.DictReader(f)]
    return {(r["vp_id"], r["country"]): float(r["distance_km"]) for r in rows}


def load_selection_parquet(path: Path) -> dict[str, Any]:
    """Return {"shape": "sequence"|"sampling", "rows": [...]} from one
    selection parquet. Detects shape via columns."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        cols = set(table.column_names)
        rows = table.to_pylist()
    except ImportError:
        import csv
        with open(path.with_suffix(".csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            cols = set(rows[0].keys()) if rows else set()
    shape = "sequence" if "position" in cols else "sampling"
    return {"shape": shape, "rows": rows}


def write_rows_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            table = pa.table({})
        pq.write_table(table, path)
    except ImportError:
        import csv
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            if not rows:
                return
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# ---- sweep orchestration --------------------------------------------------


def run_sweep(
    selections_paths: list[Path],
    claims_path: Path,
    borders_path: Path,
    speed_path: Path,
    output_rows: Path,
    output_summary: Path,
    k_min: int = 100,
    k_step: int = 100,
) -> dict[str, Any]:
    speed_limit = load_speed_limit(speed_path)
    logger.info("speed limit: %.3f km/ms", speed_limit)

    border_distances = load_border_distances(borders_path)
    logger.info("loaded %d border distance entries", len(border_distances))

    claim_rows = load_claims(claims_path)
    targets_and_claims: list[tuple[str, str, bool]] = [
        (r["target_id"], r["claimed_country"], bool(r["is_real"]))
        for r in claim_rows
    ]
    logger.info("loaded %d (target, claim) pairs", len(targets_and_claims))

    # Pull RTTs from RipeAtlasSource (same load path as calibrate_speed.py)
    source = RipeAtlasSource(
        slice="all_anchors",
        setup=DataSource.PROBES_TO_ANCHORS,
        sanitize=True,
    )
    rtts_by_target: dict[str, dict[str, float]] = defaultdict(dict)
    for et in source.iter_eval_targets():
        for vp_id, _coord, rtt in et.obs:
            rtts_by_target[et.target_id][str(vp_id)] = float(rtt)
    logger.info("loaded RTTs for %d targets", len(rtts_by_target))

    # Full-pool verdicts — used by both sequence and sampling row builders
    full_verdicts: dict[tuple[str, str], str] = {}
    for target_id, claim, _is_real in targets_and_claims:
        full_verdicts[(target_id, claim)] = iclab_verify(
            rtts_by_target.get(target_id, {}),
            claim,
            border_distances,
            speed_limit,
        )
    n_full_reject = sum(1 for v in full_verdicts.values() if v == "reject")
    logger.info("full-pool verdicts: %d reject / %d total (%.1f%%)",
                n_full_reject, len(full_verdicts),
                100 * n_full_reject / max(1, len(full_verdicts)))

    # Parse the selection parquets — group by shape
    selection_sequences: dict[tuple[str, int], list[str]] = {}
    selection_subsets: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    pool_size = 0
    for path in selections_paths:
        bundle = load_selection_parquet(path)
        if bundle["shape"] == "sequence":
            # Group rows by (strategy, seed); sort by position
            by_key: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
            for r in bundle["rows"]:
                by_key[(r["strategy"], int(r["seed"]))].append(
                    (int(r["position"]), r["vp_id"])
                )
            for key, items in by_key.items():
                items.sort()
                selection_sequences[key] = [v for _p, v in items]
                pool_size = max(pool_size, len(items))
        else:  # sampling
            for r in bundle["rows"]:
                k = int(r["k"])
                selection_subsets[(r["strategy"], int(r["seed"]), k)].append(r["vp_id"])
                pool_size = max(pool_size, k)
    logger.info("parsed %d sequence selections, %d sampling subsets",
                len(selection_sequences), len(selection_subsets))

    k_grid = []
    if pool_size > 0:
        k = pool_size
        while k >= max(1, k_min):
            k_grid.append(k)
            k -= k_step
        if not k_grid:
            k_grid = [pool_size]
    logger.info("k_grid: %d values (%d..%d)",
                len(k_grid),
                k_grid[-1] if k_grid else 0,
                k_grid[0] if k_grid else 0)

    rows: list[dict[str, Any]] = []
    if selection_sequences:
        rows.extend(compute_sequence_rows(
            selection_sequences,
            rtts_by_target,
            targets_and_claims,
            border_distances,
            speed_limit,
            full_verdicts,
        ))
    if selection_subsets:
        rows.extend(compute_sampling_rows(
            selection_subsets,
            rtts_by_target,
            targets_and_claims,
            border_distances,
            speed_limit,
            full_verdicts,
        ))

    write_rows_parquet(rows, output_rows)

    agree = agreement_curve(rows, k_grid)
    detect = detection_curve(rows, k_grid)

    full_tp = sum(
        1 for r in claim_rows
        if not r["is_real"]
        and full_verdicts[(r["target_id"], r["claimed_country"])] == "reject"
    )
    full_fp = sum(
        1 for r in claim_rows
        if r["is_real"]
        and full_verdicts[(r["target_id"], r["claimed_country"])] == "reject"
    )
    n_fake = sum(1 for r in claim_rows if not r["is_real"])
    n_real = sum(1 for r in claim_rows if r["is_real"])

    summary = {
        "speed_limit_km_per_ms": speed_limit,
        "pool_size": pool_size,
        "k_grid": k_grid,
        "full_pool": {
            "n_targets": len(claim_rows),
            "n_fake": n_fake,
            "n_real": n_real,
            "tp": full_tp,
            "fp": full_fp,
            "tpr": full_tp / max(1, n_fake),
            "fpr": full_fp / max(1, n_real),
            "reject_rate": n_full_reject / max(1, len(claim_rows)),
        },
        "agreement_curve": [
            {"strategy": s, "k": k, "is_real": r, **v}
            for (s, k, r), v in sorted(agree.items())
        ],
        "detection_curve": [
            {"strategy": s, "k": k, **v}
            for (s, k), v in sorted(detect.items())
        ],
    }
    with open(output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s + %s", output_rows, output_summary)
    return summary


# ---- plot ----------------------------------------------------------------


def render_plot(summary_path: Path, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    strategies = sorted({r["strategy"] for r in summary["agreement_curve"]})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Agreement (alt-claim — the meaningful signal)
    ax = axes[0]
    for strategy in strategies:
        ks, rates = [], []
        for r in summary["agreement_curve"]:
            if r["strategy"] != strategy or r["is_real"]:
                continue
            if r["rate"] is None:
                continue
            ks.append(r["k"])
            rates.append(r["rate"])
        if ks:
            ks, rates = zip(*sorted(zip(ks, rates)))
            ax.plot(ks, rates, label=strategy, marker=".", markersize=3)
    ax.set_xlabel("K (subset size)")
    ax.set_ylabel("Agreement rate")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Agreement vs full pool (alt claims)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Detection TPR
    ax = axes[1]
    for strategy in strategies:
        ks, vals = [], []
        for r in summary["detection_curve"]:
            if r["strategy"] != strategy:
                continue
            if r["tpr"] is None:
                continue
            ks.append(r["k"])
            vals.append(r["tpr"])
        if ks:
            ks, vals = zip(*sorted(zip(ks, vals)))
            ax.plot(ks, vals, label=strategy, marker=".", markersize=3)
    full = summary.get("full_pool", {})
    full_tpr = full.get("tpr")
    if full_tpr is not None:
        ax.axhline(full_tpr, linestyle="--", color="black", alpha=0.6,
                   label=f"full pool TPR={full_tpr:.2f}")
    ax.set_xlabel("K")
    ax.set_ylabel("TPR (fakes correctly rejected)")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Detection TPR")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 3: Detection FPR
    ax = axes[2]
    for strategy in strategies:
        ks, vals = [], []
        for r in summary["detection_curve"]:
            if r["strategy"] != strategy:
                continue
            if r["fpr"] is None:
                continue
            ks.append(r["k"])
            vals.append(r["fpr"])
        if ks:
            ks, vals = zip(*sorted(zip(ks, vals)))
            ax.plot(ks, vals, label=strategy, marker=".", markersize=3)
    full_fpr = full.get("fpr") if full else None
    if full_fpr is not None:
        ax.axhline(full_fpr, linestyle="--", color="black", alpha=0.6,
                   label=f"full pool FPR={full_fpr:.3f}")
    ax.set_xlabel("K")
    ax.set_ylabel("FPR (reals wrongly rejected)")
    ax.set_xscale("log")
    ax.set_title("Detection FPR")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("ICLab agreement + detection vs subset size K")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", output_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="Run agreement + detection sweep.")
    sweep.add_argument("--selections", nargs="+", type=Path, required=True)
    sweep.add_argument("--claims", type=Path, required=True)
    sweep.add_argument("--borders", type=Path, required=True)
    sweep.add_argument("--speed", type=Path, required=True)
    sweep.add_argument("--rows", type=Path, required=True)
    sweep.add_argument("--summary", type=Path, required=True)
    sweep.add_argument("--k-min", type=int, default=100)
    sweep.add_argument("--k-step", type=int, default=100)

    plot = sub.add_parser("plot", help="Render the 3-panel agreement/detection figure.")
    plot.add_argument("--summary", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "sweep":
        run_sweep(
            selections_paths=args.selections,
            claims_path=args.claims,
            borders_path=args.borders,
            speed_path=args.speed,
            output_rows=args.rows,
            output_summary=args.summary,
            k_min=args.k_min,
            k_step=args.k_step,
        )
    elif args.command == "plot":
        render_plot(args.summary, args.output)


if __name__ == "__main__":
    main()
