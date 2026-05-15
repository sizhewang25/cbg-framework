"""Persist per-combination and diff-pair summary statistics as JSON."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.libs.core.combinations import COMBINATIONS, DIFF_PAIRS  # noqa: E402
from scripts.libs.core.evaluate import get_errors  # noqa: E402
from scripts.libs.core.reporting import (  # noqa: E402
    count_fitted_anchors,
    count_result_outcomes,
)
from scripts.libs.plotting.plot_error_diff_cdf import compute_error_diff  # noqa: E402


def save_json_summary(
    all_results,
    output_path,
    artifacts_by_combo,
    benchmark_raw_path=None,
    benchmark_summary_path=None,
    combinations=None,
    diff_pairs=None,
    dataset="vultr_pings_us_only.csv",
    asn: Optional[int] = 7922,
    dataset_metadata: Optional[Mapping[str, Any]] = None,
    benchmark_scope="per_setting_end_to_end",
):
    """Save per-combination statistics and diff-pair summaries as JSON."""
    specs = list(COMBINATIONS if combinations is None else combinations)
    pairs = list(DIFF_PAIRS if diff_pairs is None else diff_pairs)
    summary = {
        "dataset": dataset,
        "asn": asn,
        "dataset_metadata": dict(dataset_metadata or {}),
        "n_combinations": len(specs),
        "benchmark_scope": benchmark_scope,
        "setting_benchmark_ms": {
            combo_id: {
                k: round(float(v), 3)
                for k, v in artifact.benchmark_ms.items()
            }
            for combo_id, artifact in artifacts_by_combo.items()
        },
        "benchmark_raw_csv": (
            _display_path(benchmark_raw_path)
            if benchmark_raw_path is not None
            else None
        ),
        "benchmark_summary_json": (
            _display_path(benchmark_summary_path)
            if benchmark_summary_path is not None
            else None
        ),
        "combinations": {},
        "diff_pairs": {},
    }

    for spec in specs:
        errors = get_errors(all_results[spec.combo_id])
        results = all_results[spec.combo_id]
        artifact = artifacts_by_combo[spec.combo_id]
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
                artifact.anchor_coords,
                lp_models=artifact.lp_models,
                octant_models=artifact.octant_models,
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

    for id_a, id_b in pairs:
        if id_a not in all_results or id_b not in all_results:
            continue
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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
