"""Benchmark summary plots from evaluation JSON and benchmark CSV/JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

from scripts.analysis.cbg_evaluation.combinations import COMBINATIONS, SPECS_BY_ID


PHASE_STACK_ORDER: List[Tuple[str, str]] = [
    ("load_data", "load data"),
    ("prepare_data", "prepare data"),
    ("data_fingerprint", "fingerprint"),
    ("model_cache_lookup", "cache lookup"),
    ("fit_lp_model", "fit LP"),
    ("fit_octant_model", "fit Octant"),
    ("pipeline_build", "build pipeline"),
    ("distance_estimation", "distance"),
    ("filtering", "filtering"),
    ("multilateration", "multilateration"),
    ("centroid", "centroid"),
    ("pipeline_overhead", "pipeline overhead"),
]

PHASE_COLORS: Dict[str, str] = {
    "load_data": "#6C757D",
    "prepare_data": "#ADB5BD",
    "data_fingerprint": "#CED4DA",
    "model_cache_lookup": "#495057",
    "fit_lp_model": "#8D6E63",
    "fit_octant_model": "#A1887F",
    "pipeline_build": "#B0BEC5",
    "distance_estimation": "#4E79A7",
    "filtering": "#F28E2B",
    "multilateration": "#59A14F",
    "centroid": "#E15759",
    "pipeline_overhead": "#B07AA1",
}


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file as a dictionary."""
    with open(path) as f:
        return json.load(f)


def plot_benchmark_summary(
    evaluation_summary_path: Path,
    benchmark_summary_path: Path,
    benchmark_raw_path: Path,
    output_dir: Path,
) -> List[Path]:
    """Generate all benchmark summary charts from benchmark output files."""
    evaluation_summary = load_json(evaluation_summary_path)
    benchmark_summary = load_json(benchmark_summary_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "benchmark_end_to_end_latency.png",
        output_dir / "benchmark_intersection_rate.png",
        output_dir / "benchmark_phase_latency_memory.png",
        output_dir / "benchmark_phase_memory.png",
        output_dir / "benchmark_per_ip_e2e_latency_boxplot.png",
    ]

    fig = plot_end_to_end_latency(evaluation_summary, outputs[0])
    plt.close(fig)
    fig = plot_intersection_rate(evaluation_summary, outputs[1])
    plt.close(fig)
    fig = plot_phase_latency_memory(benchmark_summary, outputs[2])
    plt.close(fig)
    fig = plot_phase_memory(benchmark_summary, outputs[3])
    plt.close(fig)
    fig = plot_per_ip_e2e_latency_boxplot(benchmark_raw_path, outputs[4])
    plt.close(fig)

    return outputs


def plot_end_to_end_latency(
    evaluation_summary: Dict[str, Any],
    output_path: Path,
) -> plt.Figure:
    """Plot per-setting end-to-end latency in seconds."""
    combo_ids = ordered_combo_ids(evaluation_summary)
    values = extract_end_to_end_seconds(evaluation_summary, combo_ids)
    combo_ids, values = rank_by_value_desc(combo_ids, values)
    colors = combo_colors(combo_ids)

    fig, ax = plt.subplots(figsize=_figure_size(combo_ids, height=5.0))
    x = np.arange(len(combo_ids))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(combo_ids)
    ax.set_ylabel("End-to-end latency (s)")
    ax.set_title("End-to-End Evaluation Latency by Pipeline ID", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _annotate_bars(ax, bars, values, suffix="s")

    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_intersection_rate(
    evaluation_summary: Dict[str, Any],
    output_path: Path,
) -> plt.Figure:
    """Plot intersection_count / total probes per setting."""
    combo_ids = ordered_combo_ids(evaluation_summary)
    rates = extract_intersection_rates(evaluation_summary, combo_ids)
    combo_ids, rates = rank_by_value_desc(combo_ids, rates)
    colors = combo_colors(combo_ids)

    fig, ax = plt.subplots(figsize=_figure_size(combo_ids, height=5.0))
    x = np.arange(len(combo_ids))
    bars = ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(combo_ids)
    ax.set_ylabel("Intersection rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Intersection Rate by Pipeline ID", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _annotate_bars(ax, bars, rates, suffix="%")

    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_phase_latency_memory(
    benchmark_summary: Dict[str, Any],
    output_path: Path,
) -> plt.Figure:
    """Plot stacked phase latency; filename kept for backward compatibility."""
    combo_ids = ordered_combo_ids(benchmark_summary)
    phase_seconds = extract_phase_latency_seconds(benchmark_summary, combo_ids)
    stack_totals = [
        sum(phase_seconds[phase][idx] for phase, _ in PHASE_STACK_ORDER)
        for idx in range(len(combo_ids))
    ]
    ranked_indices = rank_indices_desc(stack_totals)
    combo_ids = reorder(combo_ids, ranked_indices)
    phase_seconds = {
        phase: reorder(values, ranked_indices)
        for phase, values in phase_seconds.items()
    }

    fig, ax = plt.subplots(figsize=_figure_size(combo_ids, height=6.5))
    x = np.arange(len(combo_ids))
    bottoms = np.zeros(len(combo_ids), dtype=float)

    for phase, label in PHASE_STACK_ORDER:
        values = np.array(phase_seconds[phase], dtype=float)
        if not np.any(values):
            continue
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(combo_ids)
    ax.set_ylabel("Stacked phase latency (s)")
    ax.set_title("Phase Latency by Pipeline ID", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    bars_handles, bars_labels = ax.get_legend_handles_labels()
    ax.legend(
        bars_handles,
        bars_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.12),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, output_path)
    return fig


def plot_phase_memory(
    benchmark_summary: Dict[str, Any],
    output_path: Path,
) -> plt.Figure:
    """Plot phase-local Python memory bars with process max RSS line."""
    combo_ids = ordered_combo_ids(benchmark_summary)
    phase_memory_mb = extract_phase_memory_mb(benchmark_summary, combo_ids)
    rss_after_mb = extract_max_rss_mb(benchmark_summary, combo_ids)
    stack_totals = [
        sum(phase_memory_mb[phase][idx] for phase, _ in PHASE_STACK_ORDER)
        for idx in range(len(combo_ids))
    ]
    ranked_indices = rank_indices_desc(stack_totals)
    combo_ids = reorder(combo_ids, ranked_indices)
    phase_memory_mb = {
        phase: reorder(values, ranked_indices)
        for phase, values in phase_memory_mb.items()
    }
    rss_after_mb = reorder(rss_after_mb, ranked_indices)

    fig, ax = plt.subplots(figsize=_figure_size(combo_ids, height=7.0))
    x = np.arange(len(combo_ids))
    bottoms = np.zeros(len(combo_ids), dtype=float)

    for phase, label in PHASE_STACK_ORDER:
        values = np.array(phase_memory_mb[phase], dtype=float)
        if not np.any(values):
            continue
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(combo_ids)
    ax.set_ylabel("Stacked phase-local Python peak delta (MB)")
    ax.set_title("Phase-Local Memory and Max RSS by Pipeline ID", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    ax_rss = ax.twinx()
    ax_rss.plot(
        x,
        rss_after_mb,
        color="#D1495B",
        marker="s",
        linewidth=2.5,
        label="max RSS after MB",
    )
    ax_rss.set_ylabel("Process max RSS after phase (MB)")

    bars_handles, bars_labels = ax.get_legend_handles_labels()
    rss_handles, rss_labels = ax_rss.get_legend_handles_labels()
    ax.legend(
        bars_handles,
        bars_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    ax_rss.legend(
        rss_handles,
        rss_labels,
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    fig.text(
        0.01,
        0.01,
        "Memory bars are per-phase tracemalloc peak deltas; "
        "stacks are attribution aids, not concurrent RSS totals.",
        fontsize=8,
        color="#495057",
    )

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, output_path)
    return fig


def plot_per_ip_e2e_latency_boxplot(
    benchmark_raw_path: Path,
    output_path: Path,
) -> plt.Figure:
    """Plot per-probe end-to-end latency with 5th/95th percentile whiskers."""
    latency_by_combo = load_per_ip_e2e_latency_ms(
        benchmark_raw_path,
        include_fallback=False,
    )
    combo_ids = ordered_combo_ids_from_keys(latency_by_combo.keys())
    combo_ids = [combo_id for combo_id in combo_ids if latency_by_combo[combo_id]]
    samples = [latency_by_combo[combo_id] for combo_id in combo_ids]
    combo_ids, samples = rank_samples_by_median_desc(combo_ids, samples)

    fig, ax = plt.subplots(figsize=_figure_size(combo_ids, height=6.0))
    if not samples:
        ax.text(
            0.5,
            0.5,
            "No per-IP total_geolocate rows found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        _save(fig, output_path)
        return fig

    box = ax.boxplot(
        samples,
        whis=(5, 95),
        patch_artist=True,
        showfliers=False,
        labels=combo_ids,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], combo_colors(combo_ids)):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Per-IP end-to-end latency (ms)")
    ax.set_ylim(0, 500)
    ax.set_title(
        "Per-IP End-to-End Latency by Pipeline ID, No Fallback (5/95 Whiskers)",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    counts = [len(values) for values in samples]
    y_offset = 10.0
    for idx, count in enumerate(counts, start=1):
        ax.text(
            idx,
            500 - y_offset,
            f"n={count}",
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, output_path)
    return fig


def ordered_combo_ids(summary: Dict[str, Any]) -> List[str]:
    """Return combo IDs in registry order, with JSON-only IDs appended."""
    combinations = summary.get("combinations", {})
    return ordered_combo_ids_from_keys(combinations.keys())


def ordered_combo_ids_from_keys(combo_ids: Iterable[str]) -> List[str]:
    """Return combo IDs in registry order, preserving unknown IDs after that."""
    combo_ids = list(dict.fromkeys(combo_ids))
    registry_order = [spec.combo_id for spec in COMBINATIONS if spec.combo_id in combo_ids]
    extras = [combo_id for combo_id in combo_ids if combo_id not in registry_order]
    return registry_order + extras


def extract_end_to_end_seconds(
    evaluation_summary: Dict[str, Any],
    combo_ids: Iterable[str],
) -> List[float]:
    """Extract setting_total_ms as seconds for each combo."""
    setting_benchmark = evaluation_summary.get("setting_benchmark_ms", {})
    return [
        float(setting_benchmark.get(combo_id, {}).get("setting_total_ms", 0.0))
        / 1000.0
        for combo_id in combo_ids
    ]


def extract_intersection_rates(
    evaluation_summary: Dict[str, Any],
    combo_ids: Iterable[str],
) -> List[float]:
    """Extract intersection_count / n_probes as percentages."""
    rates = []
    for combo_id in combo_ids:
        entry = evaluation_summary.get("combinations", {}).get(combo_id, {})
        total = int(entry.get("n_probes", 0))
        intersections = int(entry.get("intersection_count", 0))
        rates.append(intersections / total * 100.0 if total else 0.0)
    return rates


def extract_phase_latency_seconds(
    benchmark_summary: Dict[str, Any],
    combo_ids: Iterable[str],
) -> Dict[str, List[float]]:
    """Extract stackable phase total latencies as seconds."""
    combinations = benchmark_summary.get("combinations", {})
    phase_seconds: Dict[str, List[float]] = {
        phase: [] for phase, _ in PHASE_STACK_ORDER
    }
    for combo_id in combo_ids:
        phases = combinations.get(combo_id, {}).get("phases", {})
        for phase, _ in PHASE_STACK_ORDER:
            total_ms = float(phases.get(phase, {}).get("total_ms", 0.0))
            phase_seconds[phase].append(total_ms / 1000.0)
    return phase_seconds


def extract_memory_mb(
    benchmark_summary: Dict[str, Any],
    combo_ids: Iterable[str],
) -> Tuple[List[float], List[float]]:
    """Extract one max tracemalloc and max RSS-after value per combo."""
    combinations = benchmark_summary.get("combinations", {})
    tracemalloc_values = []
    rss_after_values = []
    for combo_id in combo_ids:
        phases = combinations.get(combo_id, {}).get("phases", {})
        tracemalloc_values.append(
            _max_metric(phases.values(), "max_tracemalloc_peak_mb")
        )
        rss_after_values.append(_max_metric(phases.values(), "max_rss_after_mb"))
    return tracemalloc_values, rss_after_values


def extract_phase_memory_mb(
    benchmark_summary: Dict[str, Any],
    combo_ids: Iterable[str],
) -> Dict[str, List[float]]:
    """Extract stackable phase-local tracemalloc peak deltas as MB."""
    combinations = benchmark_summary.get("combinations", {})
    phase_memory: Dict[str, List[float]] = {
        phase: [] for phase, _ in PHASE_STACK_ORDER
    }
    for combo_id in combo_ids:
        phases = combinations.get(combo_id, {}).get("phases", {})
        for phase, _ in PHASE_STACK_ORDER:
            phase_entry = phases.get(phase, {})
            value = _metric_or_fallback(
                phase_entry,
                "max_tracemalloc_phase_peak_delta_mb",
                "max_tracemalloc_peak_mb",
            )
            phase_memory[phase].append(value)
    return phase_memory


def extract_max_rss_mb(
    benchmark_summary: Dict[str, Any],
    combo_ids: Iterable[str],
) -> List[float]:
    """Extract max RSS-after value per combo."""
    combinations = benchmark_summary.get("combinations", {})
    values = []
    for combo_id in combo_ids:
        phases = combinations.get(combo_id, {}).get("phases", {})
        values.append(_max_metric(phases.values(), "max_rss_after_mb"))
    return values


def load_per_ip_e2e_latency_ms(
    benchmark_raw_path: Path,
    include_fallback: bool = False,
) -> Dict[str, List[float]]:
    """Load per-probe total geolocation latencies in ms from raw benchmark CSV."""
    latency_by_combo: Dict[str, List[float]] = {}
    with open(benchmark_raw_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") != "total_geolocate":
                continue
            if not row.get("probe_ip"):
                continue
            if not include_fallback and _csv_bool(row.get("fallback_used")):
                continue
            combo_id = row["combo_id"]
            elapsed_ms = float(row["elapsed_ms"])
            latency_by_combo.setdefault(combo_id, []).append(elapsed_ms)
    return latency_by_combo


def rank_by_value_desc(
    combo_ids: List[str],
    values: List[float],
) -> Tuple[List[str], List[float]]:
    """Rank combo IDs and scalar values by value descending."""
    indices = rank_indices_desc(values)
    return reorder(combo_ids, indices), reorder(values, indices)


def rank_samples_by_median_desc(
    combo_ids: List[str],
    samples: List[List[float]],
) -> Tuple[List[str], List[List[float]]]:
    """Rank combo IDs and sample lists by sample median descending."""
    medians = [
        float(np.median(values)) if values else float("-inf")
        for values in samples
    ]
    indices = rank_indices_desc(medians)
    return reorder(combo_ids, indices), reorder(samples, indices)


def rank_indices_desc(values: List[float]) -> List[int]:
    """Return indices sorted by value descending and original order as tie-breaker."""
    return sorted(range(len(values)), key=lambda idx: (-float(values[idx]), idx))


def reorder(values: List[Any], indices: List[int]) -> List[Any]:
    """Return values reordered by indices."""
    return [values[idx] for idx in indices]


def combo_colors(combo_ids: Iterable[str]) -> List[str]:
    """Return registered plot colors for combo IDs, with a default fallback."""
    return [SPECS_BY_ID.get(combo_id).color if combo_id in SPECS_BY_ID else "#4E79A7"
            for combo_id in combo_ids]


def _max_metric(phases: Iterable[Dict[str, Any]], metric: str) -> float:
    values = [
        float(phase[metric])
        for phase in phases
        if phase.get(metric) is not None
    ]
    return max(values) if values else 0.0


def _metric_or_fallback(
    phase: Dict[str, Any],
    metric: str,
    fallback_metric: str,
) -> float:
    value = phase.get(metric)
    if value is None:
        value = phase.get(fallback_metric)
    return float(value) if value is not None else 0.0


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _figure_size(combo_ids: List[str], height: float) -> Tuple[float, float]:
    return (max(10.0, len(combo_ids) * 0.85), height)


def _annotate_bars(ax, bars, values: List[float], suffix: str) -> None:
    if not values:
        return
    y_offset = max(values) * 0.015 if max(values) > 0 else 1.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + y_offset,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark charts from CBG evaluation summary JSON files.",
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "evaluation_summary.json",
    )
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "benchmark_phase_summary.json",
    )
    parser.add_argument(
        "--benchmark-raw",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "benchmark_phase_raw.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for output_path in plot_benchmark_summary(
        args.evaluation_summary,
        args.benchmark_summary,
        args.benchmark_raw,
        args.output_dir,
    ):
        logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
