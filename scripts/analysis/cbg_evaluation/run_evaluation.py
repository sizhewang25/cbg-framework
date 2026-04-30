"""CLI entry point: run all configured pipeline combinations and generate plots.

Usage:
    python scripts/analysis/cbg_evaluation/run_evaluation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.cbg_evaluation.combinations import (
    COMBINATIONS,
    DIFF_PAIRS,
    SPECS_BY_ID,
)
from scripts.analysis.cbg_evaluation.evaluate import (
    evaluate_all,
    get_errors,
    load_and_prepare,
    print_statistics,
)
from scripts.analysis.cbg_evaluation.plot_error_cdf import plot_error_cdf
from scripts.analysis.cbg_evaluation.plot_error_diff_cdf import (
    compute_error_diff,
    plot_error_diff_cdf,
)
from scripts.analysis.cbg_evaluation.plot_rtt_error_scatter import (
    plot_rtt_error_scatter,
)
from scripts.analysis.cbg_evaluation.reporting import (
    count_fitted_anchors,
    count_result_outcomes,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LOG_DIR = Path(__file__).resolve().parent / "logs"


def _setup_logging(output_dir: Path) -> None:
    """Configure root logger with file + console handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler — full log
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    logger.info("Logging to %s", log_path)


def save_json_summary(
    all_results,
    output_path,
    anchor_coords,
    lp_models,
    octant_models,
):
    """Save per-combination statistics and diff-pair summaries as JSON."""
    summary = {
        "dataset": "vultr_pings_us_only.csv",
        "asn": 7922,
        "n_combinations": len(COMBINATIONS),
        "combinations": {},
        "diff_pairs": {},
    }

    for spec in COMBINATIONS:
        errors = get_errors(all_results[spec.combo_id])
        results = all_results[spec.combo_id]
        counts = count_result_outcomes(results)
        n_fallback = counts.fallback_count
        fallback_reasons = {}
        for r in results:
            if r.fallback_used:
                reason = r.fallback_reason or "unknown"
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

        entry = {
            "label": spec.label,
            "config": {
                "distance": spec.distance,
                "filtering": spec.filtering,
                "multilateration": spec.multilateration,
                "centroid": spec.centroid,
                "multilateration_kwargs": spec.multilateration_kwargs or {},
            },
            "n_fitted_anchors": count_fitted_anchors(
                spec,
                anchor_coords,
                lp_models=lp_models,
                octant_models=octant_models,
            ),
            "n_probes": counts.total_probes,
            "estimated_count": counts.estimated_count,
            "estimated_rate_pct": round(
                counts.estimated_count / max(counts.total_probes, 1) * 100,
                1,
            ),
            "intersection_count": counts.intersection_count,
            "intersection_rate_pct": round(
                counts.intersection_count / max(counts.total_probes, 1) * 100,
                1,
            ),
            "multilateration_success_count": counts.multilateration_success_count,
            "fallback_count": n_fallback,
            "fallback_rate_pct": round(
                n_fallback / max(counts.total_probes, 1) * 100,
                1,
            ),
            "no_estimate_count": counts.no_estimate_count,
            "fallback_reasons": fallback_reasons,
        }
        if len(errors) > 0:
            entry.update({
                "median_error_km": round(float(np.median(errors)), 1),
                "mean_error_km": round(float(np.mean(errors)), 1),
                "p25_km": round(float(np.percentile(errors, 25)), 1),
                "p75_km": round(float(np.percentile(errors, 75)), 1),
                "p90_km": round(float(np.percentile(errors, 90)), 1),
                "within_100km_pct": round(float(np.mean(errors <= 100) * 100), 1),
                "within_500km_pct": round(float(np.mean(errors <= 500) * 100), 1),
                "within_1000km_pct": round(float(np.mean(errors <= 1000) * 100), 1),
            })
        summary["combinations"][spec.combo_id] = entry

    for id_a, id_b in DIFF_PAIRS:
        deltas = compute_error_diff(all_results[id_a], all_results[id_b])
        if len(deltas) > 0:
            summary["diff_pairs"][f"{id_a}_vs_{id_b}"] = {
                "n_common_probes": len(deltas),
                "median_delta_km": round(float(np.median(deltas)), 1),
                "a_better_pct": round(float(np.mean(deltas < 0) * 100), 1),
                "b_better_pct": round(float(np.mean(deltas > 0) * 100), 1),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(LOG_DIR)
    total_start = time.perf_counter()

    # 1. Load data, fit models
    data = load_and_prepare()

    # 2. Evaluate all configured combinations
    logger.info("=" * 60)
    logger.info("EVALUATING ALL COMBINATIONS")
    logger.info("=" * 60)
    all_results = evaluate_all(
        COMBINATIONS,
        data["lp_models"],
        data["octant_models"],
        data["octant_delta"],
        data["anchor_coords"],
        data["probe_targets"],
    )

    # 3. Statistics table
    print_statistics(all_results, COMBINATIONS)

    # 4. Error CDF
    logger.info("=" * 60)
    logger.info("GENERATING ERROR CDF")
    logger.info("=" * 60)
    fig = plot_error_cdf(all_results, COMBINATIONS, OUTPUT_DIR / "error_cdf_all.png")
    plt.close(fig)

    # 5. Error-Diff CDF
    logger.info("=" * 60)
    logger.info("GENERATING ERROR-DIFF CDF")
    logger.info("=" * 60)
    fig = plot_error_diff_cdf(
        all_results, SPECS_BY_ID, DIFF_PAIRS,
        OUTPUT_DIR / "error_diff_cdf.png",
    )
    plt.close(fig)

    # 6. RTT-Error Scatter
    logger.info("=" * 60)
    logger.info("GENERATING RTT-ERROR SCATTER")
    logger.info("=" * 60)
    fig = plot_rtt_error_scatter(
        all_results, COMBINATIONS, OUTPUT_DIR / "rtt_error_scatter.png",
    )
    plt.close(fig)

    # 7. Percentile Maps
    logger.info("=" * 60)
    logger.info("GENERATING PERCENTILE MAPS")
    logger.info("=" * 60)
    try:
        from scripts.analysis.cbg_evaluation.plot_percentile_maps import (
            plot_percentile_maps,
        )
        plot_percentile_maps(
            all_results,
            SPECS_BY_ID,
            data["lp_models"],
            data["octant_models"],
            data["octant_delta"],
            data["anchor_coords"],
            data["probe_targets"],
            OUTPUT_DIR / "maps",
        )
    except ImportError as e:
        logger.warning("Skipping percentile maps (missing dependency): %s", e)
    except Exception as e:
        logger.error("Percentile maps failed: %s", e)

    # 8. JSON summary
    save_json_summary(
        all_results,
        OUTPUT_DIR / "evaluation_summary.json",
        data["anchor_coords"],
        data["lp_models"],
        data["octant_models"],
    )

    elapsed = time.perf_counter() - total_start
    logger.info("Total runtime: %.1fs", elapsed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
