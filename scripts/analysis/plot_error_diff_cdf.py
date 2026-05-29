"""Error-diff CDF: per-target error_A − error_B for selected combo pairs.

Inner-joins two combos' `targets.parquet` on `target_id` (the v2 analogue of
the legacy `probe_ip` join key) and plots the CDF of pairwise deltas.
Negative deltas = A is better.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis._v2_io import discover_combos, load_targets
from scripts.analysis.plot_error_cdf import (
    _load_nearest_ping_baseline_by_fold_target,
    _short_label,
)

_BASELINE_NAME = "shortest_ping"

logger = logging.getLogger(__name__)

_DIFF_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#000000", "#F0E442",
]


def compute_error_diff(
    errors_a: dict[str, float],
    errors_b: dict[str, float],
) -> np.ndarray:
    """Per-target error difference, restricted to targets present in both."""
    common = sorted(set(errors_a) & set(errors_b))
    if not common:
        return np.array([])
    return np.array([errors_a[t] - errors_b[t] for t in common])


def plot_error_diff_cdf(
    target_errors_by_combo: dict[str, dict[str, float]],
    pairs: list[tuple[str, str]],
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> plt.Figure:
    """CDF of error_A − error_B for each (A, B) pair."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (id_a, id_b) in enumerate(pairs):
        deltas = compute_error_diff(
            target_errors_by_combo[id_a],
            target_errors_by_combo[id_b],
        )
        if len(deltas) == 0:
            logger.warning("Pair (%s, %s): no shared targets, skipping", id_a, id_b)
            continue
        sorted_d = np.sort(deltas)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        pct_a_better = float(np.mean(deltas < 0) * 100)
        median_delta = float(np.median(deltas))

        a_short, b_short = _short_label(id_a), _short_label(id_b)
        ax.plot(
            sorted_d, cdf,
            color=_DIFF_COLORS[i % len(_DIFF_COLORS)],
            linewidth=2,
            label=(
                f"{a_short} − {b_short}\n"
                f"  {a_short} better: {pct_a_better:.0f}%, "
                f"med Δ={median_delta:+.0f} km, N={len(deltas)}"
            ),
        )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Error difference (km): A − B", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(
        title or "Error-Diff CDF — Pairwise Combo Comparison",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.98, "← A better", transform=ax.transAxes,
            fontsize=10, color="green", va="top", ha="left", alpha=0.7)
    ax.text(0.98, 0.98, "B better →", transform=ax.transAxes,
            fontsize=10, color="red", va="top", ha="right", alpha=0.7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    return fig


def _load_from_run(
    run_dir: Path,
    source: Optional[str],
    slice_: Optional[str],
    combos: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Walk `run_dir`, return {combo_id: {target_key: error_km}} for rows
    with a non-null error_km.

    With `slice_=None` on a K-fold layout, every combo_id collects entries
    from every fold; target keys are prefixed with the fold dir name to
    keep folds disjoint in the dict even if target_id is reused.
    """
    combo_dirs = discover_combos(run_dir, source, slice_, combos)
    if not combo_dirs:
        raise FileNotFoundError(f"No combos found under {run_dir}")

    out: dict[str, dict[str, float]] = {}
    for combo_dir in combo_dirs:
        tbl = load_targets(combo_dir)
        target_ids = tbl.column("target_id").to_pylist()
        errors = tbl.column("error_km").to_numpy(zero_copy_only=False)
        fold = combo_dir.parent.name
        bucket = out.setdefault(combo_dir.name, {})
        for tid, err in zip(target_ids, errors):
            if np.isnan(err):
                continue
            bucket[f"{fold}/{tid}"] = float(err)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot pairwise error-diff CDF from a v2 benchmark run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to outputs/<run_id>/.",
    )
    parser.add_argument("--source", default=None)
    parser.add_argument("--slice", dest="slice_", default=None)
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("A", "B"),
        action="append",
        required=True,
        help="Combo IDs to compare. Repeat for multiple pairs.",
    )
    parser.add_argument(
        "--combos", nargs="*", default=None,
        help="Restrict to these combo_ids (default: every combo found on disk). "
             "All combos named in --pair must be in this set if it's non-empty.",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=None,
        help=(
            "Path to inputs/<source>/<setup>/<slice>/ (single-fold) or its "
            "parent (merged-folds). Required when any --pair references the "
            f"synthetic '{_BASELINE_NAME}' pseudo-combo: nearest-VP error is "
            "computed on the fly per <fold>/<target_id> and injected as a "
            "pseudo-combo so the existing join works unchanged."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target_errors = _load_from_run(
        args.run_dir, args.source, args.slice_, combos=args.combos,
    )

    # Synthesize the `shortest_ping` pseudo-combo on the fly when referenced
    # by any pair — keys land on the same `<fold>/<target_id>` shape as the
    # CBG side, so the existing inner-join in `compute_error_diff` Just Works.
    if any(_BASELINE_NAME in pair for pair in args.pair):
        if args.inputs_dir is None:
            raise SystemExit(
                f"--pair references '{_BASELINE_NAME}' but --inputs-dir was not "
                "given. Pass --inputs-dir pointing at the eval_observations.parquet "
                "directory (single-fold) or its parent (merged-folds)."
            )
        target_errors[_BASELINE_NAME] = _load_nearest_ping_baseline_by_fold_target(
            args.inputs_dir,
        )
        logger.info(
            "%s baseline: %d targets loaded",
            _BASELINE_NAME, len(target_errors[_BASELINE_NAME]),
        )

    missing = {
        cid
        for pair in args.pair
        for cid in pair
        if cid not in target_errors
    }
    if missing:
        raise SystemExit(
            f"Combo(s) not found under {args.run_dir}: {sorted(missing)}. "
            f"Available: {sorted(target_errors)}"
        )
    fig = plot_error_diff_cdf(
        target_errors,
        [(a, b) for a, b in args.pair],
        args.out,
        title=args.title,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
