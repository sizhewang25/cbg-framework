"""Runner for scaled benchmark evaluations."""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.benchmark.dataset import (  # noqa: E402
    CSVDataLoader,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_ROOT,
)
from scripts.analysis.cbg_evaluation.benchmarking import BenchmarkRecorder  # noqa: E402
from scripts.analysis.cbg_evaluation.combinations import (  # noqa: E402
    DIFF_PAIRS,
    SPECS_BY_ID,
    PipelineSpec,
)
from scripts.analysis.cbg_evaluation.evaluate import (  # noqa: E402
    evaluate_all,
    print_statistics,
)
from scripts.analysis.cbg_evaluation.plot_benchmark_summary import (  # noqa: E402
    plot_benchmark_summary,
)
from scripts.analysis.cbg_evaluation.plot_error_cdf import plot_error_cdf  # noqa: E402
from scripts.analysis.cbg_evaluation.plot_error_diff_cdf import (  # noqa: E402
    plot_error_diff_cdf,
)
from scripts.analysis.cbg_evaluation.plot_rtt_error_scatter import (  # noqa: E402
    plot_rtt_error_scatter,
)
from scripts.analysis.cbg_evaluation.run_evaluation import (  # noqa: E402
    save_json_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_COMBO_IDS = ("S1", "S2", "L1", "L2", "B1", "B2", "B3", "B4")


def parse_combo_ids(combo_ids: str | Sequence[str]) -> List[str]:
    """Parse comma-separated or sequence combo ids."""
    if isinstance(combo_ids, str):
        parsed = [part.strip() for part in combo_ids.split(",") if part.strip()]
    else:
        parsed = [str(part).strip() for part in combo_ids if str(part).strip()]
    if not parsed:
        raise ValueError("At least one combo id is required")
    unknown = [combo_id for combo_id in parsed if combo_id not in SPECS_BY_ID]
    if unknown:
        raise ValueError(f"Unknown combo ids: {', '.join(unknown)}")
    return parsed


def select_combinations(combo_ids: str | Sequence[str]) -> List[PipelineSpec]:
    return [SPECS_BY_ID[combo_id] for combo_id in parse_combo_ids(combo_ids)]


def run_benchmark_evaluation(
    dataset_id: str,
    input_csv: Path = DEFAULT_INPUT_CSV,
    combo_ids: str | Sequence[str] = DEFAULT_COMBO_IDS,
    output_dir: Path | None = None,
    preselected: bool = False,
    generate_maps: bool = False,
) -> Path:
    """Run selected CBG combinations for one scaled dataset."""
    output_dir = output_dir or DEFAULT_OUTPUT_ROOT / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = select_combinations(combo_ids)
    active_pairs = _active_diff_pairs(specs)
    data_loader = CSVDataLoader(
        input_csv=input_csv,
        dataset_id=dataset_id,
        preselected=preselected,
    )

    start = time.perf_counter()
    benchmark_recorder = BenchmarkRecorder()
    evaluation_run = evaluate_all(
        specs,
        benchmark_recorder=benchmark_recorder,
        data_loader=data_loader,
    )
    all_results = evaluation_run.all_results

    benchmark_raw_path = output_dir / "benchmark_phase_raw.csv"
    benchmark_summary_path = output_dir / "benchmark_phase_summary.json"
    benchmark_recorder.write_raw_csv(benchmark_raw_path)
    benchmark_recorder.write_summary_json(benchmark_summary_path)

    print_statistics(all_results, specs)

    fig = plot_error_cdf(all_results, specs, output_dir / "error_cdf_all.png")
    plt.close(fig)

    if active_pairs:
        fig = plot_error_diff_cdf(
            all_results,
            {spec.combo_id: spec for spec in specs},
            active_pairs,
            output_dir / "error_diff_cdf.png",
        )
        plt.close(fig)

    fig = plot_rtt_error_scatter(
        all_results,
        specs,
        output_dir / "rtt_error_scatter.png",
    )
    plt.close(fig)

    if generate_maps:
        from scripts.analysis.cbg_evaluation.plot_percentile_maps import (
            plot_percentile_maps,
        )

        plot_percentile_maps(
            all_results,
            {spec.combo_id: spec for spec in specs},
            evaluation_run.artifacts_by_combo,
            output_dir / "maps",
        )

    summary_path = output_dir / "evaluation_summary.json"
    save_json_summary(
        all_results,
        summary_path,
        evaluation_run.artifacts_by_combo,
        benchmark_raw_path=benchmark_raw_path,
        benchmark_summary_path=benchmark_summary_path,
        combinations=specs,
        diff_pairs=active_pairs,
        dataset=input_csv.name,
        asn=None,
        dataset_metadata=data_loader.manifest(),
    )

    plot_benchmark_summary(
        summary_path,
        benchmark_summary_path,
        benchmark_raw_path,
        output_dir,
    )

    elapsed = time.perf_counter() - start
    logger.info("Finished %s in %.1fs", dataset_id, elapsed)
    return summary_path


def collect_summaries(results_root: Path) -> List[Dict[str, Any]]:
    """Collect per-dataset evaluation summaries into compact rows."""
    rows = []
    for summary_path in sorted(
        results_root.glob("*/evaluation_summary.json"),
        key=lambda path: _dataset_sort_key(path.parent.name),
    ):
        with open(summary_path) as f:
            summary = json.load(f)
        metadata = summary.get("dataset_metadata", {})
        combinations = summary.get("combinations", {})
        rows.append(
            {
                "dataset_id": metadata.get("dataset_id", summary_path.parent.name),
                "n_rows": metadata.get("n_rows"),
                "n_probes": metadata.get("n_probes"),
                "n_anchors": metadata.get("n_anchors"),
                "n_combinations": summary.get("n_combinations"),
                "combo_ids": ",".join(combinations.keys()),
                "summary_path": _display_path(summary_path),
            }
        )
    return rows


def write_summary_index(results_root: Path, output_json: Path, output_csv: Path) -> None:
    """Write JSON and CSV indexes over all per-dataset summaries."""
    rows = collect_summaries(results_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2) + "\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "n_rows",
        "n_probes",
        "n_anchors",
        "n_combinations",
        "combo_ids",
        "summary_path",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _active_diff_pairs(specs: Sequence[PipelineSpec]) -> List[tuple[str, str]]:
    active_ids = {spec.combo_id for spec in specs}
    return [
        (id_a, id_b)
        for id_a, id_b in DIFF_PAIRS
        if id_a in active_ids and id_b in active_ids
    ]


def _dataset_sort_key(dataset_id: str) -> tuple[int, int | str]:
    if dataset_id.startswith("top"):
        try:
            return (0, int(dataset_id.removeprefix("top")))
        except ValueError:
            return (2, dataset_id)
    if dataset_id == "all_us":
        return (1, 0)
    return (2, dataset_id)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
